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
    \\  bench             run full M1, M2, M2.5 and M3 benchmark suites
    \\  bench-m1          run M1 C9 kernel bandwidth calibration and Spike S2 hot-path probe
    \\  bench-m2          run M2 Spike S1, G3 MoE, G4 GQA and end-to-end .amf decode benchmarks
    \\  bench-m25         run M2.5 native FP16/BF16 GEMV/GEMM calibration and BF16 .amf decode
    \\  bench-m3          run M3 Spikes S4/S5, D5 gate, L1 SAM latency and synthetic spec decode
    \\  bench-decode      benchmark end-to-end greedy decode on a specific .amf + config.json
    \\                    --amf <path> --config <path> [--scale-layers N] [--prefill-tokens N]
    \\                    [--decode-steps N] [--no-verify-crc]
    \\  decode            run end-to-end native BPE encode -> greedy .amf decode -> BPE decode
    \\                    --amf <path> --config <path> --tokenizer <path>
    \\                    --prompt <text> [--max-tokens N] [--ignore-eos] [--no-verify-crc]
    \\  spec-decode       run speculative decoding (L1 SAM + EAGLE-3/EAGLE-2/MTP + D7 pruning)
    \\                    --amf <path> --config <path> --tokenizer <path> --prompt <text>
    \\                    [--draft-amf <path> --draft-config <path>]
    \\                    [--draft-mode l1|eagle3|eagle2|mtp] [--placement auto|gpu|ane]
    \\                    [--max-tree-nodes N] [--max-draft-depth N] [--branch-width N]
    \\                    [--top-k N] [--no-prune] [--temperature T] [--seed N]
    \\                    [--max-tokens N] [--ignore-eos] [--no-verify-crc]
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

fn runBenchM3(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM3(allocator);
    defer report.deinit(allocator);
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
    max_tree_nodes: u16 = 16,
    max_draft_depth: u16 = 4,
    branch_width: u16 = 2,
    top_k: u32 = 3,
    enable_pruning: bool = true,
    temperature: f32 = 0.0,
    seed: u64 = 42,
    max_tokens: u32 = 64,
    ignore_eos: bool = false,
    verify_crcs: bool = true,
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
    var max_tree_nodes: u16 = 16;
    var max_draft_depth: u16 = 4;
    var branch_width: u16 = 2;
    var top_k: u32 = 3;
    var enable_pruning = true;
    var temperature: f32 = 0.0;
    var seed: u64 = 42;
    var max_tokens: u32 = 64;
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
    };
}

fn runSpecDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseSpecDecodeArgs(raw_args);

    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    var draft_cfg_opt: ?aminfer.draft.DraftConfig = null;
    if (cli.draft_config_path) |dc_path| {
        const dcfg_bytes = try readFileAllocPosix(allocator, dc_path, 4 * 1024 * 1024);
        defer allocator.free(dcfg_bytes);
        draft_cfg_opt = try aminfer.draft.DraftConfig.fromJson(allocator, dcfg_bytes);
    }

    const mode: aminfer.draft.DraftMode = cli.draft_mode orelse if (draft_cfg_opt) |dc| dc.format else .l1_only;
    if (mode != .l1_only) {
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

    const aux_ids: ?[3]u32 = if (draft_cfg_opt) |dc| dc.aux_layer_ids else null;
    var model = try aminfer.qwen3.Qwen3Model.loadFromFile(allocator, cli.amf_path, cfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = target_tree_cap,
        .enable_feature_taps = (mode != .l1_only),
        .aux_layer_ids = aux_ids,
        .verify_crcs = cli.verify_crcs,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinit(allocator, &model);

    if (mode != .l1_only) {
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

    var eng = try aminfer.engine.SpecEngine.init(allocator, &model, draft_head_ptr, .{
        .mode = mode,
        .placement = cli.placement,
        .max_tree_nodes = cli.max_tree_nodes,
        .max_draft_depth = cli.max_draft_depth,
        .branch_width = cli.branch_width,
        .top_k_per_node = cli.top_k,
        .temperature = cli.temperature,
        .enable_pruning = cli.enable_pruning,
    });
    defer eng.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;
    var round_buf: [aminfer.engine.MAX_TREE_NODES]u32 = undefined;

    var prng = std.Random.DefaultPrng.init(cli.seed);
    const rng_opt: ?std.Random = if (cli.temperature > 0.0) prng.random() else null;

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
        \\spec.pruning                     {s} (cap={d}, depth={d}, width={d})
        \\spec.temperature                 {d:.2}
        \\spec.layers                      {d}
        \\spec.shards                      {d}
        \\spec.wired_mib                   {d}
        \\spec.prompt_tokens               {d}
        \\spec.generated_tokens            {d}
        \\spec.rounds                      {d}
        \\spec.mean_accept_len             {d:.3}
        \\spec.accept_hist                 [
    , .{
        cfg.arch.label(),
        cfg.precisionLabel(),
        mode.label(),
        eng.effectivePlacement().label(),
        cli.placement.label(),
        if (cli.enable_pruning) "enabled" else "disabled",
        cli.max_tree_nodes,
        cli.max_draft_depth,
        cli.branch_width,
        cli.temperature,
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
        \\spec.token_ids                   [
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
    try std.testing.expectEqual(@as(u64, 123), ok.seed);
    try std.testing.expectEqual(@as(u32, 64), ok.max_tokens);
    try std.testing.expect(ok.ignore_eos);
    try std.testing.expect(!ok.verify_crcs);

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
