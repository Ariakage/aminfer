//! aminfer command-line entry point.

const std = @import("std");
const Io = std.Io;
const aminfer = @import("aminfer");

const usage =
    \\usage: aminfer <command> [options]
    \\
    \\commands:
    \\  probe             print the kernel, Metal and Neural Engine facts of this machine
    \\  verify-amf <path> validate an .amf v0 file's alignment, CRCs and wired budget fit
    \\  bench             run full M1, M2, M2.5, M3, M4 and M4.5 benchmark suites
    \\  bench-m1          run M1 C9 kernel bandwidth calibration and Spike S2 hot-path probe
    \\  bench-m2          run M2 Spike S1, G3 MoE, G4 GQA and end-to-end .amf decode benchmarks
    \\  bench-m25         run M2.5 native FP16/BF16 GEMV/GEMM calibration and BF16 .amf decode
    \\  bench-m3          run M3 Spikes S4/S5, D5 gate, L1 SAM latency and synthetic spec decode
    \\  bench-m4          run M4 Spikes S3/S6, A2 predictor, A1/G6 hybrid prefill, cold NVMe & D4
    \\  bench-m45         run M4.5 Spike S7 SME2 microkernels, C13 GPU+SME2 bus gate & L2 router cal
    \\  bench-a2          evaluate real-checkpoint A2 ANE cross-layer expert predictor recall/lead-time
    \\                    --amf <path> --config <path> --tokenizer <path> [--prompt <text>]
    \\                    [--capture-steps N] [--lookahead N] [--ane-predictor <path>] [--no-verify-crc]
    \\  bench-decode      benchmark end-to-end greedy decode on a specific .amf + config.json
    \\                    --amf <path> --config <path> [--scale-layers N] [--prefill-tokens N]
    \\                    [--decode-steps N] [--cold-experts-per-layer N] [--heat-profile <path>]
    \\                    [--cold-slots N] [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--no-verify-crc]
    \\  decode            run end-to-end native BPE encode -> greedy/stochastic .amf decode -> BPE decode
    \\                    --amf <path> --config <path> --tokenizer <path>
    \\                    --prompt <text> [--max-tokens N] [--temperature T] [--seed N] [--ignore-eos]
    \\                    [--sme-router auto|sme2|gpu]
    \\                    [--cold-experts-per-layer N] [--heat-profile <path>] [--cold-slots N]
    \\                    [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--no-verify-crc]
    \\  spec-decode       run speculative decoding (L1 SAM + EAGLE-3/EAGLE-2/MTP/Small + D7 pruning)
    \\                    --amf <path> --config <path> --tokenizer <path> --prompt <text>
    \\                    [--draft-amf <path> --draft-config <path>]
    \\                    [--draft-mode l1|eagle3|eagle2|mtp|small] [--placement auto|gpu|ane]
    \\                    [--sme-router auto|sme2|gpu]
    \\                    [--max-tree-nodes N] [--max-draft-depth N] [--branch-width N]
    \\                    [--top-k N] [--no-prune] [--temperature T] [--seed N]
    \\                    [--cold-experts-per-layer N] [--heat-profile <path>] [--cold-slots N]
    \\                    [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--max-tokens N] [--ignore-eos] [--no-verify-crc]
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buffer: [8192]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    if (args.len == 2 and std.mem.eql(u8, args[1], "probe")) {
        try probe(out);
    } else if (args.len == 3 and std.mem.eql(u8, args[1], "verify-amf")) {
        try verifyAmf(arena, args[2], out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench")) {
        try runBenchAll(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m1")) {
        try runBenchM1(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m2")) {
        try runBenchM2(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m25")) {
        try runBenchM25(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m3")) {
        try runBenchM3(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m4")) {
        try runBenchM4(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m45")) {
        try runBenchM45(arena, out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-a2")) {
        try runBenchA2(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-decode")) {
        try runBenchDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "decode")) {
        try runDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "spec-decode")) {
        try runSpecDecode(arena, args[2..], out);
    } else {
        try out.writeAll(usage);
    }
    try out.flush();
}

fn parseIoBackend(str: []const u8) error{InvalidIoBackend}!aminfer.aio.IoBackend {
    if (std.mem.eql(u8, str, "auto")) return .auto;
    if (std.mem.eql(u8, str, "aio") or std.mem.eql(u8, str, "kqueue_aio")) return .kqueue_aio;
    if (std.mem.eql(u8, str, "pread") or std.mem.eql(u8, str, "ecore_pread")) return .ecore_pread;
    return error.InvalidIoBackend;
}

fn buildBudgetOverride(wired_cap_mib: ?u64) !?aminfer.budget.Budget {
    const cap_mib = wired_cap_mib orelse return null;
    if (cap_mib == 0) return error.InvalidBudgetCap;
    const facts = try aminfer.sysinfo.Facts.read();
    const device = try aminfer.metal.Device.open();
    defer device.close();
    const info = try device.info();
    return aminfer.budget.Budget.fromHost(facts, info, .{
        .configured_cap = cap_mib << 20,
    });
}

fn writeColdExpertStats(prefix: []const u8, stats: aminfer.qwen3.ColdExpertStats, out: *Io.Writer) !void {
    const ct = stats.cache_telemetry;
    try out.print(
        \\{s}.cold_experts              {d}
        \\{s}.cold_slots                {d}
        \\{s}.cold_io_backend           {s}
        \\{s}.cold_ane_predictor        {s} (uses_ne={s}, compile_ms={d:.1})
        \\{s}.cold_hot_hits             {d}
        \\{s}.cold_cache_hits           {d}
        \\{s}.cold_prefetch_hits        {d}
        \\{s}.cold_sync_misses          {d}
        \\{s}.cold_prefetch_hit_rate    {d:.3}
        \\{s}.cold_hit_rate             {d:.3}
        \\{s}.cold_bytes_read           {d}
        \\{s}.cold_evictions            {d}
        \\{s}.cold_unused_prefetched    {d}
        \\{s}.cold_prefetch_wait_us     {d:.1}
        \\{s}.cold_sync_miss_wait_us    {d:.1}
        \\
    , .{
        prefix,                                           stats.total_cold_experts,
        prefix,                                           stats.slot_count,
        prefix,                                           stats.io_backend.label(),
        prefix,                                           if (stats.ane_predictor_active) "on" else "off",
        if (stats.ane_predictor_uses_ne) "yes" else "no", stats.ane_predictor_compile_ms,
        prefix,                                           ct.hot_hits,
        prefix,                                           ct.cache_hits,
        prefix,                                           ct.prefetch_hits,
        prefix,                                           ct.sync_misses,
        prefix,                                           ct.prefetchHitRate(),
        prefix,                                           ct.coldHitRate(),
        prefix,                                           ct.bytes_read,
        prefix,                                           ct.evictions,
        prefix,                                           ct.unused_prefetched_experts,
        prefix,                                           ct.prefetchWaitUs(),
        prefix,                                           ct.syncMissWaitUs(),
    });
}

fn readFileAllocPosix(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) ![]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);

    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.FileStatFailed;
    if (st.size < 0) return error.InvalidFileSize;
    const size: usize = @intCast(st.size);
    if (size > max_bytes) return error.FileTooLarge;

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);

    var done: usize = 0;
    while (done < size) {
        const rc = std.c.read(fd, buf.ptr + done, size - done);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.FileReadFailed;
        }
        if (rc == 0) return error.UnexpectedEof;
        done += @intCast(rc);
    }
    return buf;
}

const DecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    tokenizer_path: [:0]const u8,
    prompt: []const u8,
    max_tokens: u32 = 16,
    temperature: f32 = 0.0,
    seed: ?u64 = null,
    sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto,
    ignore_eos: bool = false,
    verify_crcs: bool = true,
    cold_experts_per_layer: u32 = 0,
    heat_profile_path: ?[:0]const u8 = null,
    cold_slots: u32 = 64,
    wired_cap_mib: ?u64 = null,
    io_backend: aminfer.aio.IoBackend = .auto,
    prefill_mode: aminfer.qwen3.PrefillMode = .auto,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
};

fn parseDecodeArgs(args: []const [:0]const u8) !DecodeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var tokenizer_path: ?[:0]const u8 = null;
    var prompt: []const u8 = "";
    var max_tokens: u32 = 16;
    var temperature: f32 = 0.0;
    var seed: ?u64 = null;
    var sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto;
    var ignore_eos = false;
    var verify_crcs = true;
    var cold_experts_per_layer: u32 = 0;
    var heat_profile_path: ?[:0]const u8 = null;
    var cold_slots: u32 = 64;
    var wired_cap_mib: ?u64 = null;
    var io_backend: aminfer.aio.IoBackend = .auto;
    var prefill_mode: aminfer.qwen3.PrefillMode = .auto;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--tokenizer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            tokenizer_path = args[i];
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prompt = args[i];
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            temperature = try std.fmt.parseFloat(f32, args[i]);
            if (!std.math.isFinite(temperature) or temperature < 0.0) return error.InvalidTemperature;
        } else if (std.mem.eql(u8, flag, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--sme-router")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            sme_router_mode = try aminfer.qwen3.SmeRouterMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--ignore-eos")) {
            ignore_eos = true;
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            verify_crcs = false;
        } else if (std.mem.eql(u8, flag, "--cold-experts-per-layer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_experts_per_layer = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--heat-profile")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            heat_profile_path = args[i];
        } else if (std.mem.eql(u8, flag, "--cold-slots")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_slots = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--wired-cap-mib")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            wired_cap_mib = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--io-backend")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            io_backend = try parseIoBackend(args[i]);
        } else if (std.mem.eql(u8, flag, "--prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prefill_mode = try aminfer.qwen3.PrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path orelse return error.MissingRequiredFlag,
        .tokenizer_path = tokenizer_path orelse return error.MissingRequiredFlag,
        .prompt = prompt,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .seed = seed,
        .sme_router_mode = sme_router_mode,
        .ignore_eos = ignore_eos,
        .verify_crcs = verify_crcs,
        .cold_experts_per_layer = cold_experts_per_layer,
        .heat_profile_path = heat_profile_path,
        .cold_slots = cold_slots,
        .wired_cap_mib = wired_cap_mib,
        .io_backend = io_backend,
        .prefill_mode = prefill_mode,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
    };
}

const BenchDecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    scale_layers: ?u32 = null,
    prefill_tokens: u32 = 511,
    decode_steps: u32 = 16,
    verify_crcs: bool = true,
    cold_experts_per_layer: u32 = 0,
    heat_profile_path: ?[:0]const u8 = null,
    cold_slots: u32 = 64,
    wired_cap_mib: ?u64 = null,
    io_backend: aminfer.aio.IoBackend = .auto,
    prefill_mode: aminfer.qwen3.PrefillMode = .auto,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
};

fn parseBenchDecodeArgs(args: []const [:0]const u8) !BenchDecodeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var scale_layers: ?u32 = null;
    var prefill_tokens: u32 = 511;
    var decode_steps: u32 = 16;
    var verify_crcs = true;
    var cold_experts_per_layer: u32 = 0;
    var heat_profile_path: ?[:0]const u8 = null;
    var cold_slots: u32 = 64;
    var wired_cap_mib: ?u64 = null;
    var io_backend: aminfer.aio.IoBackend = .auto;
    var prefill_mode: aminfer.qwen3.PrefillMode = .auto;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--scale-layers")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            scale_layers = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--prefill-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prefill_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--decode-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            decode_steps = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            verify_crcs = false;
        } else if (std.mem.eql(u8, flag, "--cold-experts-per-layer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_experts_per_layer = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--heat-profile")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            heat_profile_path = args[i];
        } else if (std.mem.eql(u8, flag, "--cold-slots")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_slots = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--wired-cap-mib")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            wired_cap_mib = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--io-backend")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            io_backend = try parseIoBackend(args[i]);
        } else if (std.mem.eql(u8, flag, "--prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prefill_mode = try aminfer.qwen3.PrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path orelse return error.MissingRequiredFlag,
        .scale_layers = scale_layers,
        .prefill_tokens = prefill_tokens,
        .decode_steps = decode_steps,
        .verify_crcs = verify_crcs,
        .cold_experts_per_layer = cold_experts_per_layer,
        .heat_profile_path = heat_profile_path,
        .cold_slots = cold_slots,
        .wired_cap_mib = wired_cap_mib,
        .io_backend = io_backend,
        .prefill_mode = prefill_mode,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
    };
}

const BenchA2CliArgs = struct {
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    tokenizer_path: [:0]const u8,
    prompt: []const u8 = "Explain how B-trees keep search, insertion, and deletion logarithmic in height and provide a Python implementation:",
    capture_steps: u32 = 16,
    lookahead: u32 = 3,
    ane_predictor_path: ?[:0]const u8 = null,
    verify_crcs: bool = true,
};

fn parseBenchA2Args(args: []const [:0]const u8) !BenchA2CliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var tokenizer_path: ?[:0]const u8 = null;
    var prompt: []const u8 = "Explain how B-trees keep search, insertion, and deletion logarithmic in height and provide a Python implementation:";
    var capture_steps: u32 = 16;
    var lookahead: u32 = 3;
    var ane_predictor_path: ?[:0]const u8 = null;
    var verify_crcs = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--tokenizer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            tokenizer_path = args[i];
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prompt = args[i];
        } else if (std.mem.eql(u8, flag, "--capture-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            capture_steps = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--lookahead")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            lookahead = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            verify_crcs = false;
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path orelse return error.MissingRequiredFlag,
        .tokenizer_path = tokenizer_path orelse return error.MissingRequiredFlag,
        .prompt = prompt,
        .capture_steps = capture_steps,
        .lookahead = lookahead,
        .ane_predictor_path = ane_predictor_path,
        .verify_crcs = verify_crcs,
    };
}

fn runBenchA2(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseBenchA2Args(raw_args);
    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const tok_bytes = try readFileAllocPosix(allocator, cli.tokenizer_path, 64 * 1024 * 1024);
    defer allocator.free(tok_bytes);
    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    var prompt_ids = try tok.encodeAlloc(allocator, cli.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        allocator.free(prompt_ids);
        prompt_ids = try allocator.alloc(u32, 1);
        prompt_ids[0] = cfg.bos_token_id orelse 0;
    }

    const report = try aminfer.bench.runRealA2Eval(allocator, cli.amf_path, cfg, prompt_ids, .{
        .capture_steps = cli.capture_steps,
        .max_lookahead = cli.lookahead,
        .ane_predictor_path = cli.ane_predictor_path,
        .verify_crcs = cli.verify_crcs,
    });
    try report.write(out);
}

fn runDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseDecodeArgs(raw_args);

    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

    const tok_bytes = try readFileAllocPosix(allocator, cli.tokenizer_path, 64 * 1024 * 1024);
    defer allocator.free(tok_bytes);
    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    var prompt_ids = try tok.encodeAlloc(allocator, cli.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        allocator.free(prompt_ids);
        prompt_ids = try allocator.alloc(u32, 1);
        prompt_ids[0] = cfg.bos_token_id orelse 0;
    }

    const needed_seq: u32 = @intCast(prompt_ids.len + cli.max_tokens + 16);
    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(@as(u32, 512), needed_seq));
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);

    var model = try aminfer.qwen3.Qwen3Model.loadFromFile(allocator, cli.amf_path, cfg, .{
        .max_seq_len = max_seq_len,
        .verify_crcs = cli.verify_crcs,
        .budget_override = budget_override,
        .cold_experts_per_layer = cli.cold_experts_per_layer,
        .heat_profile_json = heat_bytes,
        .cold_slot_count = cli.cold_slots,
        .io_backend = cli.io_backend,
        .enable_ane_predictor = cli.enable_ane_predictor,
        .ane_predictor_path = cli.ane_predictor_path,
        .prefill_mode = cli.prefill_mode,
        .enable_hybrid_prefill = (cli.prefill_mode == .hybrid),
        .sme_router_mode = cli.sme_router_mode,
    });
    defer model.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;

    // Two-tier per-request sampling RNG (D11): hardware Secure Enclave TRNG via
    // `getentropy(2)` when `T > 0` and `--seed` is omitted; zero hot-path mallocs.
    var req_rng: ?aminfer.rng.RequestRng = if (cli.temperature > 0.0)
        try aminfer.rng.RequestRng.init(cli.seed)
    else
        null;
    const p_res: []f32 = if (cli.temperature > 0.0)
        try allocator.alloc(f32, cfg.vocab_size)
    else
        &.{};
    defer if (p_res.len > 0) allocator.free(p_res);

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    const prefill_t0 = try aminfer.vmstat.nowNs();

    var next_tok: u32 = try model.prefillPrompt(prompt_ids, true);
    if (cli.temperature > 0.0) {
        next_tok = try model.sampleFromLastLogits(cli.temperature, req_rng.?.random(), p_res);
    }
    var pos: u32 = @intCast(prompt_ids.len);
    const prefill_t1 = try aminfer.vmstat.nowNs();

    var decode_ns: u64 = 0;
    var timed_decode_steps: u64 = 0;
    var decode_heap_delta: aminfer.vmstat.HeapDelta = std.mem.zeroes(aminfer.vmstat.HeapDelta);
    if (cli.max_tokens > 0) {
        generated[0] = next_tok;
        gen_count = 1;
        const hit_eos = (!cli.ignore_eos and cfg.eos_token_id != null and next_tok == cfg.eos_token_id.?);
        if (!hit_eos and gen_count < cli.max_tokens) {
            model.resetColdTelemetry();
            const heap_before = try aminfer.vmstat.Snapshot.capture();
            const dec_t0 = try aminfer.vmstat.nowNs();
            while (gen_count < cli.max_tokens) {
                next_tok = try model.stepDecode(next_tok, pos);
                if (cli.temperature > 0.0) {
                    next_tok = try model.sampleFromLastLogits(cli.temperature, req_rng.?.random(), p_res);
                }
                pos += 1;
                generated[gen_count] = next_tok;
                gen_count += 1;
                timed_decode_steps += 1;
                if (!cli.ignore_eos and cfg.eos_token_id != null and next_tok == cfg.eos_token_id.?) break;
            }
            const dec_t1 = try aminfer.vmstat.nowNs();
            const heap_after = try aminfer.vmstat.Snapshot.capture();
            decode_ns = dec_t1 - dec_t0;
            decode_heap_delta = aminfer.vmstat.heapDelta(heap_before, heap_after);
        }
    }

    const vm_after = try aminfer.vmstat.Snapshot.capture();
    try aminfer.vmstat.checkNoSwap(vm_before, vm_after);

    const decoded_text = try tok.decodeAlloc(allocator, generated[0..gen_count]);
    defer allocator.free(decoded_text);

    const prefill_ms = @as(f64, @floatFromInt(prefill_t1 - prefill_t0)) / 1_000_000.0;
    const decode_ms_per_tok: f64 = if (timed_decode_steps > 0)
        (@as(f64, @floatFromInt(decode_ns)) / @as(f64, @floatFromInt(timed_decode_steps))) / 1_000_000.0
    else
        prefill_ms;
    const decode_tok_per_s: f64 = if (decode_ms_per_tok > 0.0) 1000.0 / decode_ms_per_tok else 0.0;
    const bytes_per_tok = model.bytesPerDecodeToken(pos);
    const effective_gbs: f64 = if (decode_ms_per_tok > 0.0)
        @as(f64, @floatFromInt(bytes_per_tok)) / (decode_ms_per_tok * 1_000_000.0)
    else
        0.0;

    const dec_seed_used: u64 = if (req_rng) |rr| rr.seed_used else (cli.seed orelse 0);
    const dec_seed_source: []const u8 = if (req_rng) |rr| rr.seedSourceName() else "greedy";

    try out.print(
        \\decode.arch                      {s} ({s})
        \\decode.prefill_mode              {s}
        \\decode.temperature               {d:.2}
        \\decode.seed_used                 {d}
        \\decode.seed_source               {s}
        \\decode.layers                    {d}
        \\decode.shards                    {d}
        \\decode.wired_mib                 {d}
        \\decode.prompt_tokens             {d}
        \\decode.generated_tokens          {d}
        \\decode.prefill_ms                {d:.3}
        \\decode.decode_ms_per_tok         {d:.3}
        \\decode.decode_tok_per_s          {d:.2}
        \\decode.bytes_per_tok             {d}
        \\decode.effective_gbs             {d:.2}
        \\decode.malloc_blocks_delta       {d}
        \\decode.malloc_bytes_delta        {d}
        \\decode.swapouts_delta            {d}
        \\
    , .{
        cfg.arch.label(),
        cfg.precisionLabel(),
        cli.prefill_mode.label(),
        cli.temperature,
        dec_seed_used,
        dec_seed_source,
        cfg.num_hidden_layers,
        model.shardCount(),
        model.totalWiredBytes() >> 20,
        prompt_ids.len,
        gen_count,
        prefill_ms,
        decode_ms_per_tok,
        decode_tok_per_s,
        bytes_per_tok,
        effective_gbs,
        decode_heap_delta.blocks_in_use,
        decode_heap_delta.size_in_use,
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    });
    if (model.hasSme2Weights()) {
        try out.writeAll("decode.sme2_weights              yes\n");
    }
    if (model.coldStats()) |cs| {
        try writeColdExpertStats("decode", cs, out);
    }
    try out.writeAll("decode.token_ids                 [");
    for (generated[0..gen_count], 0..) |tid, idx| {
        if (idx > 0) try out.writeAll(", ");
        try out.print("{d}", .{tid});
    }
    try out.print("]\ndecode.text                      {s}\n", .{decoded_text});
}

fn runBenchDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseBenchDecodeArgs(raw_args);
    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const max_seq_len: u32 = @max(@as(u32, 576), cli.prefill_tokens + 64);
    const res = try aminfer.bench.runDecodeBenchFromFile(allocator, cli.amf_path, cfg, .{
        .label = cfg.arch.label(),
        .scale_layers = cli.scale_layers orelse cfg.num_hidden_layers,
        .prefill_tokens = cli.prefill_tokens,
        .decode_steps = cli.decode_steps,
        .max_seq_len = max_seq_len,
        .verify_crcs = cli.verify_crcs,
        .budget_override = budget_override,
        .cold_experts_per_layer = cli.cold_experts_per_layer,
        .heat_profile_json = heat_bytes,
        .cold_slot_count = cli.cold_slots,
        .io_backend = cli.io_backend,
        .enable_ane_predictor = cli.enable_ane_predictor,
        .ane_predictor_path = cli.ane_predictor_path,
        .prefill_mode = cli.prefill_mode,
        .enable_hybrid_prefill = (cli.prefill_mode == .hybrid),
    });
    try res.write(out);
}

fn runBenchM1(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.run(allocator, .{});
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM2(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM2(allocator);
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM25(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM25(allocator);
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM3(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM3(allocator);
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM4(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM4(allocator);
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM45(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    const report = try aminfer.bench.runM45(allocator);
    try report.write(out);
}

fn runBenchAll(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    try runBenchM1(allocator, out);
    try out.flush();
    try runBenchM2(allocator, out);
    try out.flush();
    try runBenchM25(allocator, out);
    try out.flush();
    try runBenchM3(allocator, out);
    try out.flush();
    try runBenchM4(allocator, out);
    try out.flush();
    try runBenchM45(allocator, out);
}

const SpecDecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    tokenizer_path: [:0]const u8,
    prompt: []const u8 = "",
    draft_amf_path: ?[:0]const u8 = null,
    draft_config_path: ?[:0]const u8 = null,
    draft_mode: ?aminfer.draft.DraftMode = null,
    placement: aminfer.draft.Placement = .auto,
    sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto,
    max_tree_nodes: u16 = 16,
    max_draft_depth: u16 = 4,
    branch_width: u16 = 2,
    top_k: u32 = 3,
    enable_pruning: bool = true,
    temperature: f32 = 0.0,
    seed: ?u64 = null,
    max_tokens: u32 = 64,
    ignore_eos: bool = false,
    verify_crcs: bool = true,
    cold_experts_per_layer: u32 = 0,
    heat_profile_path: ?[:0]const u8 = null,
    cold_slots: u32 = 64,
    wired_cap_mib: ?u64 = null,
    io_backend: aminfer.aio.IoBackend = .auto,
    prefill_mode: aminfer.qwen3.PrefillMode = .auto,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
};

fn parseSpecDecodeArgs(args: []const [:0]const u8) !SpecDecodeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var tokenizer_path: ?[:0]const u8 = null;
    var prompt: []const u8 = "";
    var draft_amf_path: ?[:0]const u8 = null;
    var draft_config_path: ?[:0]const u8 = null;
    var draft_mode: ?aminfer.draft.DraftMode = null;
    var placement: aminfer.draft.Placement = .auto;
    var sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto;
    var max_tree_nodes: u16 = 16;
    var max_draft_depth: u16 = 4;
    var branch_width: u16 = 2;
    var top_k: u32 = 3;
    var enable_pruning = true;
    var temperature: f32 = 0.0;
    var seed: ?u64 = null;
    var max_tokens: u32 = 64;
    var ignore_eos = false;
    var verify_crcs = true;
    var cold_experts_per_layer: u32 = 0;
    var heat_profile_path: ?[:0]const u8 = null;
    var cold_slots: u32 = 64;
    var wired_cap_mib: ?u64 = null;
    var io_backend: aminfer.aio.IoBackend = .auto;
    var prefill_mode: aminfer.qwen3.PrefillMode = .auto;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--tokenizer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            tokenizer_path = args[i];
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prompt = args[i];
        } else if (std.mem.eql(u8, flag, "--draft-amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            draft_amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--draft-config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            draft_config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--draft-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            draft_mode = try aminfer.draft.DraftMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--placement")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            placement = try aminfer.draft.Placement.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--sme-router")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            sme_router_mode = try aminfer.qwen3.SmeRouterMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--max-tree-nodes")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_tree_nodes = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--max-draft-depth")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_draft_depth = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--branch-width")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            branch_width = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            top_k = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--no-prune")) {
            enable_pruning = false;
        } else if (std.mem.eql(u8, flag, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            temperature = try std.fmt.parseFloat(f32, args[i]);
            if (!std.math.isFinite(temperature) or temperature < 0.0) return error.InvalidTemperature;
        } else if (std.mem.eql(u8, flag, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--ignore-eos")) {
            ignore_eos = true;
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            verify_crcs = false;
        } else if (std.mem.eql(u8, flag, "--cold-experts-per-layer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_experts_per_layer = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--heat-profile")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            heat_profile_path = args[i];
        } else if (std.mem.eql(u8, flag, "--cold-slots")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            cold_slots = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--wired-cap-mib")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            wired_cap_mib = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--io-backend")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            io_backend = try parseIoBackend(args[i]);
        } else if (std.mem.eql(u8, flag, "--prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prefill_mode = try aminfer.qwen3.PrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path orelse return error.MissingRequiredFlag,
        .tokenizer_path = tokenizer_path orelse return error.MissingRequiredFlag,
        .prompt = prompt,
        .draft_amf_path = draft_amf_path,
        .draft_config_path = draft_config_path,
        .draft_mode = draft_mode,
        .placement = placement,
        .sme_router_mode = sme_router_mode,
        .max_tree_nodes = max_tree_nodes,
        .max_draft_depth = max_draft_depth,
        .branch_width = branch_width,
        .top_k = top_k,
        .enable_pruning = enable_pruning,
        .temperature = temperature,
        .seed = seed,
        .max_tokens = max_tokens,
        .ignore_eos = ignore_eos,
        .verify_crcs = verify_crcs,
        .cold_experts_per_layer = cold_experts_per_layer,
        .heat_profile_path = heat_profile_path,
        .cold_slots = cold_slots,
        .wired_cap_mib = wired_cap_mib,
        .io_backend = io_backend,
        .prefill_mode = prefill_mode,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
    };
}

fn runSpecDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseSpecDecodeArgs(raw_args);

    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

    var draft_cfg_opt: ?aminfer.draft.DraftConfig = null;
    var small_draft_cfg_opt: ?aminfer.qwen3.ModelConfig = null;
    var mode: aminfer.draft.DraftMode = cli.draft_mode orelse .l1_only;

    if (cli.draft_config_path) |dc_path| {
        const dcfg_bytes = try readFileAllocPosix(allocator, dc_path, 4 * 1024 * 1024);
        defer allocator.free(dcfg_bytes);
        if (cli.draft_mode == .small_draft) {
            small_draft_cfg_opt = try aminfer.qwen3.ModelConfig.fromJson(allocator, dcfg_bytes);
            mode = .small_draft;
        } else if (cli.draft_mode == null) {
            if (aminfer.draft.DraftConfig.fromJson(allocator, dcfg_bytes)) |dc| {
                draft_cfg_opt = dc;
                mode = dc.format;
            } else |err| switch (err) {
                error.UnsupportedFormat => {
                    small_draft_cfg_opt = try aminfer.qwen3.ModelConfig.fromJson(allocator, dcfg_bytes);
                    mode = .small_draft;
                },
                else => return err,
            }
        } else {
            draft_cfg_opt = try aminfer.draft.DraftConfig.fromJson(allocator, dcfg_bytes);
            draft_cfg_opt.?.format = mode;
        }
    }

    if (mode == .small_draft) {
        if (cli.draft_amf_path == null or small_draft_cfg_opt == null) return error.MissingRequiredFlag;
    } else if (mode != .l1_only) {
        if (cli.draft_amf_path == null or draft_cfg_opt == null) return error.MissingRequiredFlag;
        draft_cfg_opt.?.format = mode;
    }

    const tok_bytes = try readFileAllocPosix(allocator, cli.tokenizer_path, 64 * 1024 * 1024);
    defer allocator.free(tok_bytes);
    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    var prompt_ids = try tok.encodeAlloc(allocator, cli.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        allocator.free(prompt_ids);
        prompt_ids = try allocator.alloc(u32, 1);
        prompt_ids[0] = cfg.bos_token_id orelse 0;
    }

    const needed_seq: u32 = @intCast(prompt_ids.len + cli.max_tokens + 64);
    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(@as(u32, 512), needed_seq));
    const target_tree_cap: u32 = if (cli.enable_pruning)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, cli.max_tree_nodes) * 2, 24))
    else
        @as(u32, cli.max_tree_nodes);

    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const aux_ids: ?[3]u32 = if (draft_cfg_opt) |dc| dc.aux_layer_ids else null;
    const needs_taps = (mode != .l1_only and mode != .small_draft);
    var model = try aminfer.qwen3.Qwen3Model.loadFromFile(allocator, cli.amf_path, cfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = target_tree_cap,
        .enable_feature_taps = needs_taps,
        .aux_layer_ids = aux_ids,
        .verify_crcs = cli.verify_crcs,
        .budget_override = budget_override,
        .cold_experts_per_layer = cli.cold_experts_per_layer,
        .heat_profile_json = heat_bytes,
        .cold_slot_count = cli.cold_slots,
        .io_backend = cli.io_backend,
        .enable_ane_predictor = cli.enable_ane_predictor,
        .ane_predictor_path = cli.ane_predictor_path,
        .prefill_mode = cli.prefill_mode,
        .enable_hybrid_prefill = (cli.prefill_mode == .hybrid),
        .sme_router_mode = cli.sme_router_mode,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinit(allocator, &model);

    var small_draft_storage: aminfer.draft.SmallDraftModel = undefined;
    var small_draft_ptr: ?*aminfer.draft.SmallDraftModel = null;
    defer if (small_draft_ptr) |sd| sd.deinit(allocator);

    if (mode == .small_draft) {
        small_draft_storage = try aminfer.draft.SmallDraftModel.loadFromFile(
            allocator,
            &model,
            cli.draft_amf_path.?,
            small_draft_cfg_opt.?,
            .{
                .placement = cli.placement,
                .max_seq_len = max_seq_len,
                .max_tree_slots = target_tree_cap,
                .verify_crcs = cli.verify_crcs,
            },
        );
        small_draft_ptr = &small_draft_storage;
    } else if (mode != .l1_only) {
        draft_head_storage = try aminfer.draft.DraftHead.loadFromFile(
            allocator,
            &model,
            cli.draft_amf_path.?,
            draft_cfg_opt.?,
            .{
                .placement = cli.placement,
                .max_seq_len = max_seq_len,
                .max_tree_slots = target_tree_cap,
                .verify_crcs = cli.verify_crcs,
            },
        );
        draft_head_ptr = &draft_head_storage;
    }

    const spec_cfg: aminfer.engine.SpecConfig = .{
        .mode = mode,
        .placement = cli.placement,
        .max_tree_nodes = cli.max_tree_nodes,
        .max_draft_depth = cli.max_draft_depth,
        .branch_width = cli.branch_width,
        .top_k_per_node = cli.top_k,
        .temperature = cli.temperature,
        .enable_pruning = cli.enable_pruning,
    };
    var eng = if (small_draft_ptr) |sd|
        try aminfer.engine.SpecEngine.initWithSmallDraft(allocator, &model, sd, spec_cfg)
    else
        try aminfer.engine.SpecEngine.init(allocator, &model, draft_head_ptr, spec_cfg);
    defer eng.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;
    var round_buf: [aminfer.engine.MAX_TREE_NODES]u32 = undefined;

    // Two-tier per-request sampling RNG (D11): hardware Secure Enclave TRNG via
    // `getentropy(2)` when `T > 0` and `--seed` is omitted; deterministic replay when provided.
    var req_rng: ?aminfer.rng.RequestRng = if (cli.temperature > 0.0)
        try aminfer.rng.RequestRng.init(cli.seed)
    else
        null;
    const rng_opt: ?std.Random = if (req_rng != null) req_rng.?.random() else null;

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    const prefill_t0 = try aminfer.vmstat.nowNs();
    const first_tok = try eng.prefill(prompt_ids, rng_opt);
    const prefill_t1 = try aminfer.vmstat.nowNs();

    var decode_ns: u64 = 0;
    var timed_decode_tokens: u64 = 0;
    var decode_heap_delta: aminfer.vmstat.HeapDelta = std.mem.zeroes(aminfer.vmstat.HeapDelta);
    var decode_vm_before = try aminfer.vmstat.Snapshot.capture();
    var decode_vm_after = decode_vm_before;

    if (cli.max_tokens > 0) {
        generated[0] = first_tok;
        gen_count = 1;
        const first_is_eos = (!cli.ignore_eos and cfg.eos_token_id != null and first_tok == cfg.eos_token_id.?);
        if (!first_is_eos and gen_count < cli.max_tokens) {
            model.resetColdTelemetry();
            decode_vm_before = try aminfer.vmstat.Snapshot.capture();
            const dec_t0 = try aminfer.vmstat.nowNs();

            while (gen_count < cli.max_tokens) {
                const rem = cli.max_tokens - gen_count;
                const cap = @min(rem, round_buf.len);
                const rs = try eng.stepRound(rng_opt, round_buf[0..cap]);
                var hit_eos = false;
                for (round_buf[0..rs.emitted_count]) |tid| {
                    generated[gen_count] = tid;
                    gen_count += 1;
                    timed_decode_tokens += 1;
                    if (!cli.ignore_eos and cfg.eos_token_id != null and tid == cfg.eos_token_id.?) {
                        hit_eos = true;
                        break;
                    }
                    if (gen_count >= cli.max_tokens) break;
                }
                if (hit_eos) break;
            }

            const dec_t1 = try aminfer.vmstat.nowNs();
            decode_vm_after = try aminfer.vmstat.Snapshot.capture();
            decode_ns = dec_t1 - dec_t0;
            decode_heap_delta = aminfer.vmstat.heapDelta(decode_vm_before, decode_vm_after);
        }
    }

    const vm_after = try aminfer.vmstat.Snapshot.capture();
    try aminfer.vmstat.checkNoSwap(decode_vm_before, vm_after);

    const decoded_text = try tok.decodeAlloc(allocator, generated[0..gen_count]);
    defer allocator.free(decoded_text);

    const prefill_ms = @as(f64, @floatFromInt(prefill_t1 - prefill_t0)) / 1_000_000.0;
    const decode_ms_per_tok: f64 = if (timed_decode_tokens > 0)
        (@as(f64, @floatFromInt(decode_ns)) / @as(f64, @floatFromInt(timed_decode_tokens))) / 1_000_000.0
    else
        prefill_ms;
    const decode_tok_per_s: f64 = if (decode_ms_per_tok > 0.0) 1000.0 / decode_ms_per_tok else 0.0;
    const rounds_f: f64 = if (eng.stats.rounds > 0) @floatFromInt(eng.stats.rounds) else 1.0;
    const bytes_per_acc: f64 = eng.stats.bytesPerAcceptedToken();
    const effective_gbs: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(eng.stats.total_bytes_moved)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;

    try out.print(
        \\spec.arch                        {s} ({s})
        \\spec.mode                        {s}
        \\spec.placement                   {s} (requested={s})
        \\
    , .{
        cfg.arch.label(),
        cfg.precisionLabel(),
        mode.label(),
        eng.effectivePlacement().label(),
        cli.placement.label(),
    });
    if (small_draft_ptr) |sd| {
        if (sd.placement_gate_ane_us > 0.0 or sd.placement_gate_gpu_us > 0.0) {
            try out.print(
                \\spec.placement_gate_ane_us       {d:.1}
                \\spec.placement_gate_gpu_us       {d:.1}
                \\
            , .{ sd.placement_gate_ane_us, sd.placement_gate_gpu_us });
        }
    } else if (draft_head_ptr) |dh| {
        if (dh.placement_gate_ane_us > 0.0 or dh.placement_gate_gpu_us > 0.0) {
            try out.print(
                \\spec.placement_gate_ane_us       {d:.1}
                \\spec.placement_gate_gpu_us       {d:.1}
                \\
            , .{ dh.placement_gate_ane_us, dh.placement_gate_gpu_us });
        }
    }

    if (cfg.arch == .qwen3_moe) {
        const rcal = eng.routerPredCalibration();
        try out.print(
            \\spec.router_pred_backend         {s} (requested={s})
            \\spec.router_pred_cal             sme2_us={d:.1} gpu_us={d:.1} bank_kib={d:.1}
            \\
        , .{
            rcal.backend.label(),
            cli.sme_router_mode.label(),
            rcal.sme2_us,
            rcal.gpu_us,
            @as(f64, @floatFromInt(rcal.bank_bytes)) / 1024.0,
        });
    }

    const spec_seed_used: u64 = if (req_rng) |rr| rr.seed_used else (cli.seed orelse 0);
    const spec_seed_source: []const u8 = if (req_rng) |rr| rr.seedSourceName() else "greedy";

    try out.print(
        \\spec.pruning                     {s} (cap={d}, depth={d}, width={d})
        \\spec.temperature                 {d:.2}
        \\spec.seed_used                   {d}
        \\spec.seed_source                 {s}
        \\spec.layers                      {d}
        \\spec.shards                      {d}
        \\spec.wired_mib                   {d}
        \\spec.prompt_tokens               {d}
        \\spec.generated_tokens            {d}
        \\spec.rounds                      {d}
        \\spec.mean_accept_len             {d:.3}
        \\spec.accept_hist                 [
    , .{
        if (cli.enable_pruning) "enabled" else "disabled",
        cli.max_tree_nodes,
        cli.max_draft_depth,
        cli.branch_width,
        cli.temperature,
        spec_seed_used,
        spec_seed_source,
        cfg.num_hidden_layers,
        model.shardCount(),
        eng.totalWiredBytes() >> 20,
        prompt_ids.len,
        gen_count,
        eng.stats.rounds,
        eng.stats.meanAcceptanceLength(),
    });

    var first_hist = true;
    for (1..eng.stats.accept_hist.len) |k| {
        const cnt = eng.stats.accept_hist[k];
        if (cnt > 0) {
            if (!first_hist) try out.writeAll(", ");
            try out.print("{d}:{d}", .{ k, cnt });
            first_hist = false;
        }
    }
    try out.print(
        \\]
        \\spec.drafted_nodes_per_round     {d:.2}
        \\spec.verified_nodes_per_round    {d:.2}
        \\spec.l1_seed_rounds              {d}
        \\spec.unique_experts_per_layer    {d:.2}
        \\spec.prefill_ms                  {d:.3}
        \\spec.decode_ms_per_tok           {d:.3}
        \\spec.decode_tok_per_s            {d:.2}
        \\spec.draft_ms_per_round          {d:.3}
        \\spec.prune_ms_per_round          {d:.3}
        \\spec.verify_ms_per_round         {d:.3}
        \\spec.commit_sync_ms_per_round    {d:.3}
        \\spec.bytes_per_accepted_tok      {d:.0}
        \\spec.effective_gbs               {d:.2}
        \\spec.malloc_blocks_delta         {d}
        \\spec.malloc_bytes_delta          {d}
        \\spec.prefill_swapouts_delta      {d}
        \\spec.swapouts_delta              {d}
        \\
    , .{
        eng.stats.meanDraftedNodes(),
        eng.stats.meanVerifiedNodes(),
        eng.stats.l1_seed_rounds,
        eng.stats.meanUniqueExpertsPerLayer(),
        prefill_ms,
        decode_ms_per_tok,
        decode_tok_per_s,
        (@as(f64, @floatFromInt(eng.stats.draft_ns)) / rounds_f) / 1_000_000.0,
        (@as(f64, @floatFromInt(eng.stats.prune_ns)) / rounds_f) / 1_000_000.0,
        (@as(f64, @floatFromInt(eng.stats.verify_ns)) / rounds_f) / 1_000_000.0,
        (@as(f64, @floatFromInt(eng.stats.commit_sync_ns)) / rounds_f) / 1_000_000.0,
        bytes_per_acc,
        effective_gbs,
        decode_heap_delta.blocks_in_use,
        decode_heap_delta.size_in_use,
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, decode_vm_before),
        aminfer.vmstat.Snapshot.swapoutDelta(decode_vm_before, vm_after),
    });
    if (model.coldStats()) |cs| {
        try writeColdExpertStats("spec", cs, out);
    }
    try out.writeAll("spec.token_ids                   [");
    for (generated[0..gen_count], 0..) |tid, idx| {
        if (idx > 0) try out.writeAll(", ");
        try out.print("{d}", .{tid});
    }
    try out.print("]\nspec.text                        {s}\n", .{decoded_text});
}

fn verifyAmf(allocator: std.mem.Allocator, path: [:0]const u8, out: *Io.Writer) !void {
    var file = try aminfer.amf.AmfFile.open(allocator, path);
    defer file.close(allocator);
    try file.verifyAllBlobs(allocator);

    var total_wired_bytes: u64 = 0;
    var cold_bytes: u64 = 0;
    for (file.parsed.entries) |*entry| {
        const kind = try entry.tensorKind();
        if (kind == .cold_expert) {
            cold_bytes += entry.blob_bytes;
        } else {
            total_wired_bytes += entry.blob_bytes;
        }
    }

    const facts = try aminfer.sysinfo.Facts.read();
    const device = try aminfer.metal.Device.open();
    defer device.close();
    const info = try device.info();
    const wired_budget = aminfer.budget.Budget.fromHost(facts, info, .{});
    try wired_budget.checkPlan(total_wired_bytes);

    try out.print(
        \\amf.status       ok
        \\amf.model_id     {s}
        \\amf.tensors      {d}
        \\amf.file_bytes   {d}
        \\amf.wired_bytes  {d}
        \\amf.cold_bytes   {d}
        \\
    , .{
        file.parsed.modelId(),
        file.parsed.entries.len,
        file.parsed.file_size,
        total_wired_bytes,
        cold_bytes,
    });
}

fn probe(out: *Io.Writer) !void {
    const facts = try aminfer.sysinfo.Facts.read();
    try out.print(
        \\sys.page_size                    {d}
        \\sys.memory_mib                   {d}
        \\cpu.performance_cores            {d}
        \\cpu.efficiency_cores             {d}
        \\cpu.pcore_clusters               {d}
        \\cpu.pcore_l2_kib                 {d}
        \\cpu.sme2                         {}
        \\cpu.sme_bf16f32                  {}
        \\cpu.sme_f16f32                   {}
        \\cpu.sme_svl_bytes                {d}
        \\wire.user_limit_bytes            {d}
        \\wire.global_limit_bytes          {d}
        \\wire.gpu_limit_mib               {d}
        \\
    , .{
        facts.page_size,
        facts.memory_bytes >> 20,
        facts.performance_cores,
        facts.efficiency_cores,
        facts.pcore_clusters,
        facts.pcoreL2Kib(),
        facts.has_sme2,
        facts.has_sme_b16f32,
        facts.has_sme_f16f32,
        facts.sme_max_svl_bytes,
        facts.user_wire_limit,
        facts.global_user_wire_limit,
        facts.gpu_wired_limit_mib,
    });

    const device = try aminfer.metal.Device.open();
    defer device.close();
    const info = try device.info();
    const wired_budget = aminfer.budget.Budget.fromHost(facts, info, .{});
    try out.print(
        \\gpu.name                         {s}
        \\gpu.apple_family                 {d}
        \\gpu.metal4                       {}
        \\gpu.unified_memory               {}
        \\gpu.recommended_working_set_mib  {d}
        \\gpu.max_buffer_mib               {d}
        \\ane.cores                        {d}
        \\budget.wired_limit_mib           {d}
        \\budget.binding_limit             {s}
        \\
    , .{
        info.name(),
        info.appleGpuFamily(),
        info.supportsMetal4(),
        info.hasUnifiedMemory(),
        info.recommendedMaxWorkingSetSize() >> 20,
        info.maxBufferLength() >> 20,
        info.aneCoreCount(),
        wired_budget.limit_bytes >> 20,
        wired_budget.binding_limit.label(),
    });
}

test "parseDecodeArgs validates required flags and options" {
    const ok = try parseDecodeArgs(&.{
        "--amf",        "m.amf",
        "--config",     "c.json",
        "--tokenizer",  "t.json",
        "--prompt",     "hello world",
        "--max-tokens", "8",
        "--ignore-eos",
    });
    try std.testing.expectEqualStrings("m.amf", ok.amf_path);
    try std.testing.expectEqualStrings("c.json", ok.config_path);
    try std.testing.expectEqualStrings("t.json", ok.tokenizer_path);
    try std.testing.expectEqualStrings("hello world", ok.prompt);
    try std.testing.expectEqual(@as(u32, 8), ok.max_tokens);
    try std.testing.expect(ok.ignore_eos);
    try std.testing.expect(ok.verify_crcs);

    try std.testing.expectError(error.MissingRequiredFlag, parseDecodeArgs(&.{ "--amf", "m.amf" }));
    try std.testing.expectError(error.UnknownFlag, parseDecodeArgs(&.{"--bogus"}));
}

test "parseSpecDecodeArgs parses all speculative decoding flags and validates errors" {
    const ok = try parseSpecDecodeArgs(&.{
        "--amf",             "target.amf",
        "--config",          "config.json",
        "--tokenizer",       "tok.json",
        "--prompt",          "def fib(n):",
        "--draft-amf",       "eagle3.amf",
        "--draft-config",    "eagle3.json",
        "--draft-mode",      "eagle3",
        "--placement",       "ane",
        "--max-tree-nodes",  "24",
        "--max-draft-depth", "6",
        "--branch-width",    "4",
        "--top-k",           "5",
        "--no-prune",        "--temperature",
        "0.7",               "--seed",
        "123",               "--max-tokens",
        "64",                "--ignore-eos",
        "--no-verify-crc",
    });
    try std.testing.expectEqualStrings("target.amf", ok.amf_path);
    try std.testing.expectEqualStrings("eagle3.amf", ok.draft_amf_path.?);
    try std.testing.expectEqual(aminfer.draft.DraftMode.eagle3, ok.draft_mode.?);
    try std.testing.expectEqual(aminfer.draft.Placement.ane, ok.placement);
    try std.testing.expectEqual(@as(u16, 24), ok.max_tree_nodes);
    try std.testing.expectEqual(@as(u16, 6), ok.max_draft_depth);
    try std.testing.expectEqual(@as(u16, 4), ok.branch_width);
    try std.testing.expectEqual(@as(u32, 5), ok.top_k);
    try std.testing.expect(!ok.enable_pruning);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), ok.temperature, 1e-6);
    try std.testing.expectEqual(@as(?u64, 123), ok.seed);
    try std.testing.expectEqual(@as(u32, 64), ok.max_tokens);
    try std.testing.expect(ok.ignore_eos);
    try std.testing.expect(!ok.verify_crcs);

    // Default `--seed` is null (Secure Enclave TRNG D11) and `--sme-router` defaults to `auto`.
    const defaults = try parseSpecDecodeArgs(&.{
        "--amf",       "t.amf",
        "--config",    "c.json",
        "--tokenizer", "k.json",
    });
    try std.testing.expectEqual(@as(?u64, null), defaults.seed);
    try std.testing.expectEqual(aminfer.qwen3.SmeRouterMode.auto, defaults.sme_router_mode);

    const sme_cli = try parseSpecDecodeArgs(&.{
        "--amf",        "t.amf",
        "--config",     "c.json",
        "--tokenizer",  "k.json",
        "--sme-router", "sme2",
    });
    try std.testing.expectEqual(aminfer.qwen3.SmeRouterMode.sme2, sme_cli.sme_router_mode);

    const dec_trng = try parseDecodeArgs(&.{
        "--amf",         "m.amf",
        "--config",      "c.json",
        "--tokenizer",   "t.json",
        "--temperature", "0.7",
        "--sme-router",  "gpu",
    });
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), dec_trng.temperature, 1e-6);
    try std.testing.expectEqual(@as(?u64, null), dec_trng.seed);
    try std.testing.expectEqual(aminfer.qwen3.SmeRouterMode.gpu, dec_trng.sme_router_mode);

    const dec_seeded = try parseDecodeArgs(&.{
        "--amf",         "m.amf",
        "--config",      "c.json",
        "--tokenizer",   "t.json",
        "--temperature", "0.7",
        "--seed",        "9876543210",
    });
    try std.testing.expectEqual(@as(?u64, 9876543210), dec_seeded.seed);

    try std.testing.expectError(error.InvalidSmeRouterMode, parseSpecDecodeArgs(&.{
        "--amf",        "t.amf",
        "--config",     "c.json",
        "--tokenizer",  "k.json",
        "--sme-router", "npu",
    }));

    try std.testing.expectError(error.InvalidDraftMode, parseSpecDecodeArgs(&.{
        "--amf",        "t.amf",
        "--config",     "c.json",
        "--tokenizer",  "k.json",
        "--draft-mode", "unknown",
    }));
    try std.testing.expectError(error.InvalidPlacement, parseSpecDecodeArgs(&.{
        "--amf",       "t.amf",
        "--config",    "c.json",
        "--tokenizer", "k.json",
        "--placement", "tpu",
    }));
}

test "M4 CLI parsers handle cold-expert, prefill-mode, io-backend, bench-a2 and D4 small draft flags" {
    const dec = try parseDecodeArgs(&.{
        "--amf",                    "30b.amf",
        "--config",                 "config.json",
        "--tokenizer",              "tok.json",
        "--prompt",                 "test",
        "--cold-experts-per-layer", "64",
        "--heat-profile",           "heat.json",
        "--cold-slots",             "96",
        "--wired-cap-mib",          "8192",
        "--io-backend",             "aio",
        "--prefill-mode",           "hybrid",
        "--ane-predictor",          "pred.mlpackage",
        "--no-ane-predictor",
    });
    try std.testing.expectEqual(@as(u32, 64), dec.cold_experts_per_layer);
    try std.testing.expectEqualStrings("heat.json", dec.heat_profile_path.?);
    try std.testing.expectEqual(@as(u32, 96), dec.cold_slots);
    try std.testing.expectEqual(@as(?u64, 8192), dec.wired_cap_mib);
    try std.testing.expectEqual(aminfer.aio.IoBackend.kqueue_aio, dec.io_backend);
    try std.testing.expectEqual(aminfer.qwen3.PrefillMode.hybrid, dec.prefill_mode);
    try std.testing.expectEqualStrings("pred.mlpackage", dec.ane_predictor_path.?);
    try std.testing.expect(!dec.enable_ane_predictor);

    const bdec = try parseBenchDecodeArgs(&.{
        "--amf",                    "30b.amf",
        "--config",                 "config.json",
        "--cold-experts-per-layer", "32",
        "--io-backend",             "pread",
        "--prefill-mode",           "gpu",
        "--wired-cap-mib",          "8192",
    });
    try std.testing.expectEqual(@as(u32, 32), bdec.cold_experts_per_layer);
    try std.testing.expectEqual(aminfer.aio.IoBackend.ecore_pread, bdec.io_backend);
    try std.testing.expectEqual(aminfer.qwen3.PrefillMode.gpu, bdec.prefill_mode);
    try std.testing.expectEqual(@as(?u64, 8192), bdec.wired_cap_mib);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",                    "14b.amf",
        "--config",                 "14b.json",
        "--tokenizer",              "tok.json",
        "--draft-amf",              "0.6b.amf",
        "--draft-config",           "0.6b.json",
        "--draft-mode",             "small",
        "--cold-experts-per-layer", "16",
        "--io-backend",             "auto",
        "--prefill-mode",           "auto",
    });
    try std.testing.expectEqual(aminfer.draft.DraftMode.small_draft, spec.draft_mode.?);
    try std.testing.expectEqual(@as(u32, 16), spec.cold_experts_per_layer);

    const a2 = try parseBenchA2Args(&.{
        "--amf",           "30b.amf",
        "--config",        "config.json",
        "--tokenizer",     "tok.json",
        "--capture-steps", "12",
        "--lookahead",     "3",
        "--ane-predictor", "all48.mlpackage",
        "--no-verify-crc",
    });
    try std.testing.expectEqual(@as(u32, 12), a2.capture_steps);
    try std.testing.expectEqual(@as(u32, 3), a2.lookahead);
    try std.testing.expectEqualStrings("all48.mlpackage", a2.ane_predictor_path.?);
    try std.testing.expect(!a2.verify_crcs);

    try std.testing.expectError(error.InvalidIoBackend, parseIoBackend("uring"));
    try std.testing.expectError(error.InvalidPrefillMode, aminfer.qwen3.PrefillMode.parse("cpu"));
}
