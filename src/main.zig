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
    \\  bench             run full M1 (C9, S2), M2 (S1, G3, G4, .amf decode) and M2.5 (FP16/BF16) benchmarks
    \\  bench-m1          run M1 C9 kernel bandwidth calibration and Spike S2 hot-path probe
    \\  bench-m2          run M2 Spike S1, G3 MoE, G4 GQA and end-to-end .amf decode benchmarks
    \\  bench-m25         run M2.5 native FP16/BF16 GEMV/GEMM calibration and BF16 .amf decode
    \\  bench-decode      benchmark end-to-end greedy decode on a specific .amf + config.json
    \\                    --amf <path> --config <path> [--scale-layers N] [--prefill-tokens N]
    \\                    [--decode-steps N] [--no-verify-crc]
    \\  decode            run end-to-end native BPE encode -> greedy .amf decode -> BPE decode
    \\                    --amf <path> --config <path> --tokenizer <path>
    \\                    --prompt <text> [--max-tokens N] [--ignore-eos] [--no-verify-crc]
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-decode")) {
        try runBenchDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "decode")) {
        try runDecode(arena, args[2..], out);
    } else {
        try out.writeAll(usage);
    }
    try out.flush();
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
    ignore_eos: bool = false,
    verify_crcs: bool = true,
};

fn parseDecodeArgs(args: []const [:0]const u8) !DecodeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var tokenizer_path: ?[:0]const u8 = null;
    var prompt: []const u8 = "";
    var max_tokens: u32 = 16;
    var ignore_eos = false;
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
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--ignore-eos")) {
            ignore_eos = true;
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
        .max_tokens = max_tokens,
        .ignore_eos = ignore_eos,
        .verify_crcs = verify_crcs,
    };
}

const BenchDecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    scale_layers: ?u32 = null,
    prefill_tokens: u32 = 511,
    decode_steps: u32 = 16,
    verify_crcs: bool = true,
};

fn parseBenchDecodeArgs(args: []const [:0]const u8) !BenchDecodeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var scale_layers: ?u32 = null;
    var prefill_tokens: u32 = 511;
    var decode_steps: u32 = 16;
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
    };
}

fn runDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseDecodeArgs(raw_args);

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

    const needed_seq: u32 = @intCast(prompt_ids.len + cli.max_tokens + 16);
    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(@as(u32, 512), needed_seq));

    var model = try aminfer.qwen3.Qwen3Model.loadFromFile(allocator, cli.amf_path, cfg, .{
        .max_seq_len = max_seq_len,
        .verify_crcs = cli.verify_crcs,
    });
    defer model.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    const prefill_t0 = try aminfer.vmstat.nowNs();

    var pos: u32 = 0;
    var next_tok: u32 = 0;
    for (prompt_ids, 0..) |pid, idx| {
        if (idx + 1 < prompt_ids.len) {
            try model.prefillToken(pid, pos);
        } else {
            next_tok = try model.stepDecode(pid, pos);
        }
        pos += 1;
    }
    const prefill_t1 = try aminfer.vmstat.nowNs();

    var decode_ns: u64 = 0;
    var timed_decode_steps: u64 = 0;
    var decode_heap_delta: aminfer.vmstat.HeapDelta = std.mem.zeroes(aminfer.vmstat.HeapDelta);
    if (cli.max_tokens > 0) {
        generated[0] = next_tok;
        gen_count = 1;
        const hit_eos = (!cli.ignore_eos and cfg.eos_token_id != null and next_tok == cfg.eos_token_id.?);
        if (!hit_eos) {
            const heap_before = try aminfer.vmstat.Snapshot.capture();
            const dec_t0 = try aminfer.vmstat.nowNs();
            while (gen_count < cli.max_tokens) {
                next_tok = try model.stepDecode(next_tok, pos);
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

    try out.print(
        \\decode.arch                      {s} ({s})
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
        \\decode.token_ids                 [
    , .{
        cfg.arch.label(),
        cfg.precisionLabel(),
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

    const max_seq_len: u32 = @max(@as(u32, 576), cli.prefill_tokens + 64);
    const res = try aminfer.bench.runDecodeBenchFromFile(allocator, cli.amf_path, cfg, .{
        .label = cfg.arch.label(),
        .scale_layers = cli.scale_layers orelse cfg.num_hidden_layers,
        .prefill_tokens = cli.prefill_tokens,
        .decode_steps = cli.decode_steps,
        .max_seq_len = max_seq_len,
        .verify_crcs = cli.verify_crcs,
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

fn runBenchAll(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    try runBenchM1(allocator, out);
    try out.flush();
    try runBenchM2(allocator, out);
    try out.flush();
    try runBenchM25(allocator, out);
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
        \\wire.user_limit_bytes            {d}
        \\wire.global_limit_bytes          {d}
        \\wire.gpu_limit_mib               {d}
        \\
    , .{
        facts.page_size,
        facts.memory_bytes >> 20,
        facts.performance_cores,
        facts.efficiency_cores,
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
