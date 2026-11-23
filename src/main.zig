//! aminfer command-line entry point.

const std = @import("std");
const Io = std.Io;
const aminfer = @import("aminfer");

const usage =
    \\usage: aminfer <command> [options]
    \\
    \\commands:
    \\  probe             print the kernel, chip profile, Metal, NAX and Neural Engine facts of this machine
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\  ls [dir]          scan a directory of .amf models without mlocking weights (W18)
    \\  inspect <path>    print header, metadata, dtype breakdown, wired estimate and tier fit of an .amf file (W18)
    \\  monitor           start the rootless btop-style TrueColor TUI hardware & inference monitor (C19)
    \\                    [--theme <path.theme>] [--interval-ms N] [--once] [--json] [--server <host:port>]
    \\  chat              start the plain scrolling CLI chat REPL or stdin pipe mode (C20)
    \\                    --model <path.amf> [--config <path>] [--tokenizer <path>] [--system <text>]
    \\                    [--lora <name>=<path.amlora>] [--no-think] [--hide-think] [--no-color] [--no-stats]
    \\                    [--max-tokens N] [--temperature T] [--seed N] [--image <path>] [--video <path>]
    \\                    [--max-seq-len N] [--prefix-cache-chunks N] [--disk-prefix-cache <dir>]
    \\                    [--wired-cap-mib N] [--no-verify-crc]
    \\  verify-amf <path> validate an .amf v0 file's alignment, CRCs and wired budget fit
    \\  bench             run full M1, M2, M2.5, M3, M4, M4.5, M5, M6, M6.5, M6.8, M6.9, M7 and M7.5 benchmark suites
    \\  bench-m1          run M1 C9 kernel bandwidth calibration and Spike S2 hot-path probe
    \\  bench-m2          run M2 Spike S1, G3 MoE, G4 GQA and end-to-end .amf decode benchmarks
    \\  bench-m25         run M2.5 native FP16/BF16 GEMV/GEMM calibration and BF16 .amf decode
    \\  bench-m3          run M3 Spikes S4/S5, D5 gate, L1 SAM latency and synthetic spec decode
    \\  bench-m4          run M4 Spikes S3/S6, A2 predictor, A1/G6 hybrid prefill, cold NVMe & D4
    \\  bench-m45         run M4.5 Spike S7 SME2 microkernels, C13 GPU+SME2 bus gate & L2 router cal
    \\  bench-m5          run M5 G13 q2/hetero-MoE, C6 F_NOCACHE ngram, D8 PEARL & C7/C8 kqueue server
    \\  bench-m6          run M6 G7/G8/G9/G10, C10 PLE, scale-qwen4-512e+MTP, D12 & D13 cpu_sme2
    \\  bench-m65         run M6.5 G14 HW VideoToolbox vision, R5 32K q8 KV, R8 PrefixCache & C14 batch/ngram
    \\  bench-m68         run M6.8 G16 canary/Apple10, G15 wide prefill/tail-pruning A/B, C15 QoS/boundary & R9 live-RAM
    \\  bench-m69         run M6.9 C16 ChipProfile, G17 NAX metallib/fallback/demotion gates & Apple10+ Spike S8 sweep
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\  bench-m7          run M7 R7/G11 MLA+CSA2 FP4, G10 Mega-mHC, C10 Engram, W11/D10 deepseek41 CED+DSpark & X1-X3 cluster
    \\                    [--bus-bandwidth-gbs F]
    \\  bench-m75         run M7.5 W18 single-file AMF, W19 multi-LoRA, C22 grammar bitmask, C19 monitor, C21 embed/rerank, R10 DiskPrefixCache & C17 C-ABI SDK suite
    \\  peer-worker       start a zero-allocation kqueue expert-parallel peer worker (M7 X1-X3)
    \\                    --listen <ip:port> --amf <path> --config <path> --rank N --world-size N
    \\                    [--partition contiguous|round-robin] [--cluster-transport auto|tcp|rdma]
    \\                    [--strict-tb5-rdma] [--max-requests N] [--max-tokens N] [--wired-cap-mib N]
    \\                    [--no-verify-crc]
    \\  bench-a2          evaluate real-checkpoint A2 ANE cross-layer expert predictor recall/lead-time
    \\                    --amf <path> --config <path> --tokenizer <path> [--prompt <text>]
    \\                    [--capture-steps N] [--lookahead N] [--ane-predictor <path>] [--no-verify-crc]
    \\  bench-decode      benchmark end-to-end greedy decode on a specific .amf + config.json
    \\                    --amf <path> [--config <path>] [--scale-layers N] [--prefill-tokens N]
    \\                    [--decode-steps N] [--cold-experts-per-layer N] [--heat-profile <path>]
    \\                    [--cold-slots N] [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--gpu-prefill-mode wide|tree64]
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\                    [--strict-live-ram] [--ane-predictor <path>]
    \\                    [--kv-retire-q8-after-chunks N]
    \\                    [--ple-amf <path>] [--max-loops N] [--loop-adapter-amf <path>]
    \\                    [--loop-early-exit] [--peer <ip:port>] [--rank N] [--world-size N]
    \\                    [--partition contiguous|round-robin] [--cluster-transport auto|tcp|rdma]
    \\                    [--strict-tb5-rdma] [--shutdown-peers] [--no-ane-predictor] [--no-verify-crc]
    \\  decode            run end-to-end native BPE encode -> greedy/stochastic .amf decode -> BPE decode
    \\                    --amf <path> [--config <path>] [--tokenizer <path>]
    \\                    --prompt <text> [--max-tokens N] [--temperature T] [--seed N] [--ignore-eos]
    \\                    [--sme-router auto|sme2|gpu] [--kv-retire-q8-after-chunks N]
    \\                    [--image <path.png>] [--vt-test-video hevc|h264]
    \\                    [--ple-amf <path>] [--max-loops N] [--loop-adapter-amf <path>]
    \\                    [--loop-early-exit] [--peer <ip:port>] [--rank N] [--world-size N]
    \\                    [--partition contiguous|round-robin] [--cluster-transport auto|tcp|rdma]
    \\                    [--strict-tb5-rdma] [--shutdown-peers]
    \\                    [--cold-experts-per-layer N] [--heat-profile <path>] [--cold-slots N]
    \\                    [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--gpu-prefill-mode wide|tree64]
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\                    [--strict-live-ram] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--no-verify-crc]
    \\  spec-decode       run speculative decoding (L1 SAM + C6 N-gram + EAGLE/MTP/Small/Self-Recurrent/DSpark)
    \\                    --amf <path> [--config <path>] [--tokenizer <path>] --prompt <text>
    \\                    [--draft-amf <path> --draft-config <path>]
    \\                    [--draft-mode l1|eagle3|eagle2|mtp|small|self-recurrent|dspark]
    \\                    [--placement auto|gpu|ane|sme2] [--kv-retire-q8-after-chunks N]
    \\                    [--ple-amf <path>] [--max-loops N] [--loop-adapter-amf <path>]
    \\                    [--loop-early-exit] [--peer <ip:port>] [--rank N] [--world-size N]
    \\                    [--partition contiguous|round-robin] [--cluster-transport auto|tcp|rdma]
    \\                    [--strict-tb5-rdma] [--shutdown-peers]
    \\                    [--sme-router auto|sme2|gpu] [--pearl auto|on|off] [--pearl-min-target-ms F]
    \\                    [--ngram-nvme <path>] [--ngram-cache-pages N]
    \\                    [--max-tree-nodes N] [--max-draft-depth N] [--branch-width N]
    \\                    [--top-k N] [--no-prune] [--temperature T] [--seed N]
    \\                    [--cold-experts-per-layer N] [--heat-profile <path>] [--cold-slots N]
    \\                    [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--gpu-prefill-mode wide|tree64]
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\                    [--strict-live-ram] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--max-tokens N] [--ignore-eos] [--no-verify-crc]
    \\  serve             start the zero-allocation kqueue HTTP/SSE inference server (C7, C8, C14, C15, C18, C21, C22, W19, R10)
    \\                    --amf <path> [--config <path>] [--tokenizer <path>]
    \\                    [--host 127.0.0.1] [--port 8080] [--unix-socket <path>] [--api-key <secret>]
    \\                    [--model-id <id>] [--lora <name>=<path.amlora>] [--disk-prefix-cache <dir>]
    \\                    [--max-seq-len N] [--max-conns N] [--max-active-requests N]
    \\                    [--batch-mode auto|continuous|serial] [--prefix-cache-chunks N]
    \\                    [--decode-share F] [--adaptive-prefill-narrow on|off]
    \\                    [--message-boundary-cache on|off]
    \\                    [--kv-retire-q8-after-chunks N] [--ngram-pack <domain>=<path.ang>]
    \\                    [--draft-amf <path> --draft-config <path>]
    \\                    [--draft-mode l1|eagle3|eagle2|mtp|small|self-recurrent]
    \\                    [--placement auto|gpu|ane|sme2]
    \\                    [--ple-amf <path>] [--max-loops N] [--loop-adapter-amf <path>]
    \\                    [--loop-early-exit]
    \\                    [--sme-router auto|sme2|gpu] [--pearl auto|on|off] [--pearl-min-target-ms F]
    \\                    [--ngram-nvme <path>] [--ngram-cache-pages N]
    \\                    [--max-tree-nodes N] [--max-draft-depth N] [--branch-width N]
    \\                    [--top-k N] [--no-prune]
    \\                    [--cold-experts-per-layer N] [--heat-profile <path>] [--cold-slots N]
    \\                    [--wired-cap-mib N] [--io-backend auto|aio|pread]
    \\                    [--prefill-mode auto|gpu|hybrid] [--gpu-prefill-mode wide|tree64]
    \\                    [--prefill-matmul-backend auto|simdgroup|nax] [--strict-nax] [--bus-bandwidth-gbs F]
    \\                    [--strict-live-ram] [--ane-predictor <path>]
    \\                    [--no-ane-predictor] [--no-verify-crc]
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buffer: [8192]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    if (args.len >= 2 and std.mem.eql(u8, args[1], "probe")) {
        try probe(args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "ls")) {
        try runLs(arena, args[2..], out);
    } else if (args.len == 3 and std.mem.eql(u8, args[1], "inspect")) {
        try runInspect(arena, args[2], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "monitor")) {
        try out.flush();
        try runMonitor(arena, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "chat")) {
        try out.flush();
        try runChat(arena, args[2..]);
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
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m5")) {
        try runBenchM5(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m6")) {
        try runBenchM6(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m65")) {
        try runBenchM65(arena, out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m68")) {
        try runBenchM68(arena, out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-m69")) {
        try runBenchM69(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-m7")) {
        try runBenchM7(arena, args[2..], out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench-m75")) {
        try runBenchM75(arena, out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "peer-worker")) {
        try runPeerWorker(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-a2")) {
        try runBenchA2(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-decode")) {
        try runBenchDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "decode")) {
        try runDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "spec-decode")) {
        try runSpecDecode(arena, args[2..], out);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "serve")) {
        try runServe(arena, args[2..], out);
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

fn parseVtCodec(str: []const u8) error{InvalidVtCodec}!aminfer.metal.VtCodec {
    if (std.mem.eql(u8, str, "hevc") or std.mem.eql(u8, str, "h265")) return .hevc;
    if (std.mem.eql(u8, str, "h264") or std.mem.eql(u8, str, "avc")) return .h264;
    return error.InvalidVtCodec;
}

fn parseOnOff(str: []const u8) error{InvalidOnOff}!bool {
    if (std.mem.eql(u8, str, "on") or std.mem.eql(u8, str, "true") or std.mem.eql(u8, str, "1")) return true;
    if (std.mem.eql(u8, str, "off") or std.mem.eql(u8, str, "false") or std.mem.eql(u8, str, "0")) return false;
    return error.InvalidOnOff;
}

/// Auto-sizes optional `.ang` N-gram LRU cache pages (`P1-B`, ADR-0020) against
/// live `HOST_VM_INFO64` non-anonymous reclaimable RAM minus safety margin while
/// preserving the single-page operational minimum.
fn scaleOptionalNgramCachePages(
    requested_pages: usize,
    budget_override: ?aminfer.budget.Budget,
    already_wired_bytes: u64,
    strict_live_ram: bool,
) !usize {
    if (requested_pages <= 1) return requested_pages;
    const base_budget = budget_override orelse blk: {
        const facts = try aminfer.sysinfo.Facts.read();
        const dev = try aminfer.metal.Device.open();
        defer dev.close();
        const info = try dev.info();
        break :blk aminfer.budget.Budget.fromHost(facts, info, .{});
    };
    const rem_limit: u64 = if (base_budget.limit_bytes > already_wired_bytes)
        base_budget.limit_bytes - already_wired_bytes
    else
        0;
    const rem_budget: aminfer.budget.Budget = .{
        .limit_bytes = rem_limit,
        .binding_limit = base_budget.binding_limit,
        .raw_limit_bytes = rem_limit,
    };
    const snap = try aminfer.vmstat.Snapshot.capture();
    const min_bytes: u64 = aminfer.page.APPLE_PAGE_SIZE;
    const extra_pages: usize = requested_pages - 1;
    const extra_bytes: u64 = @as(u64, extra_pages) * aminfer.page.APPLE_PAGE_SIZE;
    const plan = try rem_budget.planWithLiveRam(
        min_bytes,
        extra_bytes,
        snap.liveZeroSwapReclaimableBytes(),
        .{ .strict_live_ram = strict_live_ram },
    );
    const granted_extra: usize = aminfer.budget.scaleOptionalCount(
        extra_pages,
        aminfer.page.APPLE_PAGE_SIZE,
        plan.granted_optional_cache_bytes,
        0,
    );
    return 1 + granted_extra;
}

const ServeBatchMode = enum {
    auto,
    continuous,
    serial,

    fn parse(str: []const u8) error{InvalidBatchMode}!ServeBatchMode {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "continuous")) return .continuous;
        if (std.mem.eql(u8, str, "serial")) return .serial;
        return error.InvalidBatchMode;
    }

    fn label(self: ServeBatchMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .continuous => "continuous",
            .serial => "serial",
        };
    }
};

const NgramPackArg = struct {
    domain: []const u8,
    path: [:0]const u8,
};

fn parseNgramPackArg(arg: [:0]const u8) error{InvalidNgramPackSpec}!NgramPackArg {
    const eq_idx = std.mem.indexOfScalar(u8, arg, '=') orelse return error.InvalidNgramPackSpec;
    if (eq_idx == 0 or eq_idx + 1 >= arg.len) return error.InvalidNgramPackSpec;
    const domain = arg[0..eq_idx];
    if (!aminfer.ngram_nvme.isValidDomainId(domain)) return error.InvalidNgramPackSpec;
    const path: [:0]const u8 = arg[eq_idx + 1 .. :0];
    return .{
        .domain = domain,
        .path = path,
    };
}

const LoraSpecArg = struct {
    name: []const u8,
    path: [:0]const u8,
};

fn parseLoraSpecArg(arg: [:0]const u8) error{InvalidLoraSpec}!LoraSpecArg {
    const eq_idx = std.mem.indexOfScalar(u8, arg, '=') orelse return error.InvalidLoraSpec;
    if (eq_idx == 0 or eq_idx + 1 >= arg.len) return error.InvalidLoraSpec;
    const name = arg[0..eq_idx];
    if (name.len > aminfer.lora.MAX_ADAPTER_NAME_LEN) return error.InvalidLoraSpec;
    for (name) |c| {
        const ok = (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '_' or c == '-' or c == '.';
        if (!ok) return error.InvalidLoraSpec;
    }
    const path: [:0]const u8 = arg[eq_idx + 1 .. :0];
    return .{
        .name = name,
        .path = path,
    };
}

const ResolvedMetadataPair = struct {
    config: aminfer.amf.ResolvedMetadata,
    tokenizer: aminfer.amf.ResolvedMetadata,
    amf_header_crc32: u32,
    amf_dir_crc32: u32,

    fn deinit(self: *ResolvedMetadataPair, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        self.config.deinit(allocator);
    }
};

/// Resolves `config.json` and `tokenizer.json` with `ADR-0022` `W18` precedence:
///   1. Explicit CLI flag (`--config` / `--tokenizer`)
///   2. Embedded cold metadata (`__meta.config_json` / `__meta.tokenizer_json`) inside `amf_path`
///   3. Sidecar `config.json` / `tokenizer.json` next to `amf_path`
fn resolveConfigAndTokenizerAlloc(
    allocator: std.mem.Allocator,
    amf_path: [:0]const u8,
    explicit_config_path: ?[:0]const u8,
    explicit_tokenizer_path: ?[:0]const u8,
) !ResolvedMetadataPair {
    var amf_file = try aminfer.amf.AmfFile.open(allocator, amf_path);
    defer amf_file.close(allocator);

    var cfg_meta = try aminfer.amf.resolveConfigJsonAlloc(
        allocator,
        &amf_file,
        amf_path,
        if (explicit_config_path) |p| @as([]const u8, p) else null,
    );
    errdefer cfg_meta.deinit(allocator);

    const tok_meta = try aminfer.amf.resolveTokenizerJsonAlloc(
        allocator,
        &amf_file,
        amf_path,
        if (explicit_tokenizer_path) |p| @as([]const u8, p) else null,
    );
    return .{
        .config = cfg_meta,
        .tokenizer = tok_meta,
        .amf_header_crc32 = amf_file.parsed.header.header_crc32,
        .amf_dir_crc32 = amf_file.parsed.header.directory_crc32,
    };
}

fn resolveConfigOnlyAlloc(
    allocator: std.mem.Allocator,
    amf_path: [:0]const u8,
    explicit_config_path: ?[:0]const u8,
) !aminfer.amf.ResolvedMetadata {
    var amf_file = try aminfer.amf.AmfFile.open(allocator, amf_path);
    defer amf_file.close(allocator);
    return aminfer.amf.resolveConfigJsonAlloc(
        allocator,
        &amf_file,
        amf_path,
        if (explicit_config_path) |p| @as([]const u8, p) else null,
    );
}

fn writeRetiredKvStats(
    prefix: []const u8,
    retired_chunks: u32,
    events: u64,
    active_bytes: u64,
    retired_bytes: u64,
    out: *Io.Writer,
) !void {
    try out.print(
        \\{s}.retired_kv_chunks         {d} (events={d})
        \\{s}.kv_wired_mib              active={d:.2} retired_q8={d:.2}
        \\
    , .{
        prefix,
        retired_chunks,
        events,
        prefix,
        @as(f64, @floatFromInt(active_bytes)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(retired_bytes)) / (1024.0 * 1024.0),
    });
}

/// Zero-allocation domain `.ang` lookup on the user-content tail of a tokenized
/// prompt (stripping trailing ChatML role/end tokens so real corpus 4-grams hit).
fn queryNgramPromptTail(
    reg: *aminfer.ngram_nvme.DomainNgramRegistry,
    tok: *const aminfer.tokenizer.Tokenizer,
    domain: []const u8,
    prompt_tokens: []const u32,
    err_counter: *u64,
) void {
    if (prompt_tokens.len < 2) return;
    const im_end = tok.addedTokenId("<|im_end|>");
    var end: usize = prompt_tokens.len;
    // Locate the final user `<|im_end|>` boundary when ChatML framing is present.
    if (im_end) |end_id| {
        var idx: usize = prompt_tokens.len;
        while (idx > 0) {
            idx -= 1;
            if (prompt_tokens[idx] == end_id) {
                end = idx;
                break;
            }
        }
    }
    while (end > 0 and tok.isSpecialToken(prompt_tokens[end - 1])) {
        end -= 1;
    }
    if (end < 2) end = prompt_tokens.len;
    const dom_opt: ?[]const u8 = if (domain.len > 0) domain else null;
    var tmp_cont: [aminfer.ngram_nvme.MAX_CONT_LEN]u32 = undefined;
    if (reg.queryDomain(dom_opt, prompt_tokens[0..end], &tmp_cont)) |_| {} else |_| {
        err_counter.* += 1;
    }
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

const M6GraphCliFlags = struct {
    ple_amf_paths: [16][:0]const u8 = @splat(""),
    ple_amf_count: usize = 0,
    max_loops: ?u32 = null,
    loop_adapter_amf_path: ?[:0]const u8 = null,
    loop_early_exit: bool = false,

    fn pleSlice(self: *const M6GraphCliFlags) []const [:0]const u8 {
        return self.ple_amf_paths[0..self.ple_amf_count];
    }

    fn appendPle(self: *M6GraphCliFlags, path: [:0]const u8) error{TooManyPleShards}!void {
        if (self.ple_amf_count >= self.ple_amf_paths.len) return error.TooManyPleShards;
        self.ple_amf_paths[self.ple_amf_count] = path;
        self.ple_amf_count += 1;
    }

    fn tryParseFlag(self: *M6GraphCliFlags, args: []const [:0]const u8, i: *usize) !bool {
        const flag = args[i.*];
        if (std.mem.eql(u8, flag, "--ple-amf")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            try self.appendPle(args[i.*]);
            return true;
        } else if (std.mem.eql(u8, flag, "--max-loops")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            const v = try std.fmt.parseInt(u32, args[i.*], 10);
            if (v == 0) return error.InvalidFlagValue;
            self.max_loops = v;
            return true;
        } else if (std.mem.eql(u8, flag, "--loop-adapter-amf")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            self.loop_adapter_amf_path = args[i.*];
            return true;
        } else if (std.mem.eql(u8, flag, "--loop-early-exit")) {
            self.loop_early_exit = true;
            return true;
        }
        return false;
    }
};

const M69CliFlags = struct {
    prefill_backend_mode: aminfer.dequant.PrefillBackendMode = .auto,
    strict_nax: bool = false,
    bus_bandwidth_gbs: ?f64 = null,

    fn tryParseFlag(self: *M69CliFlags, args: []const [:0]const u8, i: *usize) !bool {
        const flag = args[i.*];
        if (std.mem.eql(u8, flag, "--prefill-matmul-backend")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            self.prefill_backend_mode = aminfer.dequant.PrefillBackendMode.parse(args[i.*]) orelse
                return error.InvalidPrefillMatmulBackend;
            return true;
        } else if (std.mem.eql(u8, flag, "--strict-nax")) {
            self.strict_nax = true;
            return true;
        } else if (std.mem.eql(u8, flag, "--bus-bandwidth-gbs")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            const v = try std.fmt.parseFloat(f64, args[i.*]);
            if (!std.math.isFinite(v) or v <= 0.0) return error.InvalidBusBandwidthGbs;
            self.bus_bandwidth_gbs = v;
            return true;
        }
        return false;
    }
};

fn parseM69Args(args: []const [:0]const u8) !M69CliFlags {
    var m69: M69CliFlags = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (try m69.tryParseFlag(args, &i)) continue;
        return error.UnknownFlag;
    }
    return m69;
}

const ParsedHostPort = struct {
    host: [4]u8,
    port: u16,
};

fn parseHostPort(text: []const u8) !ParsedHostPort {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.InvalidPeerAddress;
    const host_s = text[0..colon];
    const port_s = text[colon + 1 ..];
    const host = aminfer.transport.parseIpv4(host_s) catch return error.InvalidPeerAddress;
    const port = std.fmt.parseInt(u16, port_s, 10) catch return error.InvalidPeerAddress;
    if (port == 0) return error.InvalidPeerAddress;
    return .{ .host = host, .port = port };
}

fn parsePartitionStrategy(text: []const u8) !aminfer.expert_parallel.PartitionStrategy {
    if (std.mem.eql(u8, text, "contiguous") or std.mem.eql(u8, text, "contiguous_by_rank")) {
        return .contiguous_by_rank;
    }
    if (std.mem.eql(u8, text, "round-robin") or std.mem.eql(u8, text, "round_robin")) {
        return .round_robin;
    }
    return error.InvalidPartitionStrategy;
}

fn parseClusterTransport(text: []const u8) !aminfer.transport.TransportKind {
    if (std.mem.eql(u8, text, "auto")) return .auto;
    if (std.mem.eql(u8, text, "tcp") or std.mem.eql(u8, text, "tb4_tcp")) return .tb4_tcp;
    if (std.mem.eql(u8, text, "rdma") or std.mem.eql(u8, text, "tb5_rdma")) return .tb5_rdma;
    return error.InvalidClusterTransport;
}

const M7ClusterCliFlags = struct {
    peer_specs: [aminfer.expert_parallel.MAX_WORLD_SIZE - 1][]const u8 = undefined,
    peer_count: usize = 0,
    rank: u16 = 0,
    world_size: u16 = 0,
    partition: aminfer.expert_parallel.PartitionStrategy = .contiguous_by_rank,
    transport_kind: aminfer.transport.TransportKind = .auto,
    strict_tb5_rdma: bool = false,
    shutdown_peers: bool = false,

    fn hasPeers(self: M7ClusterCliFlags) bool {
        return self.peer_count > 0;
    }

    fn effectiveTransport(self: M7ClusterCliFlags) aminfer.transport.TransportKind {
        return if (self.strict_tb5_rdma) .tb5_rdma else self.transport_kind;
    }

    fn effectiveWorldSize(self: M7ClusterCliFlags) u16 {
        if (self.world_size > 0) return self.world_size;
        return @intCast(self.peer_count + 1);
    }

    fn tryParseFlag(self: *M7ClusterCliFlags, args: []const [:0]const u8, i: *usize) !bool {
        const flag = args[i.*];
        if (std.mem.eql(u8, flag, "--peer")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            if (self.peer_count >= self.peer_specs.len) return error.TooManyPeers;
            // Validate peer syntax eagerly (`ip:port` or `rank=ip:port`).
            const spec = args[i.*];
            if (std.mem.indexOfScalar(u8, spec, '=')) |eq| {
                _ = std.fmt.parseInt(u16, spec[0..eq], 10) catch return error.InvalidPeerAddress;
                _ = try parseHostPort(spec[eq + 1 ..]);
            } else {
                _ = try parseHostPort(spec);
            }
            self.peer_specs[self.peer_count] = spec;
            self.peer_count += 1;
            return true;
        } else if (std.mem.eql(u8, flag, "--rank")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            self.rank = try std.fmt.parseInt(u16, args[i.*], 10);
            return true;
        } else if (std.mem.eql(u8, flag, "--world-size")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            const ws = try std.fmt.parseInt(u16, args[i.*], 10);
            if (ws == 0 or ws > aminfer.expert_parallel.MAX_WORLD_SIZE) return error.InvalidWorldSize;
            self.world_size = ws;
            return true;
        } else if (std.mem.eql(u8, flag, "--partition")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            self.partition = try parsePartitionStrategy(args[i.*]);
            return true;
        } else if (std.mem.eql(u8, flag, "--cluster-transport")) {
            i.* += 1;
            if (i.* >= args.len) return error.MissingFlagValue;
            self.transport_kind = try parseClusterTransport(args[i.*]);
            return true;
        } else if (std.mem.eql(u8, flag, "--strict-tb5-rdma")) {
            self.strict_tb5_rdma = true;
            return true;
        } else if (std.mem.eql(u8, flag, "--shutdown-peers")) {
            self.shutdown_peers = true;
            return true;
        }
        return false;
    }

    fn resolveEndpoints(self: *const M7ClusterCliFlags, out_buf: *[aminfer.expert_parallel.MAX_WORLD_SIZE - 1]aminfer.expert_parallel.PeerEndpoint) ![]const aminfer.expert_parallel.PeerEndpoint {
        const ws = self.effectiveWorldSize();
        if (self.rank >= ws) return error.InvalidRank;
        if (self.peer_count != ws - 1) return error.PeerCountMismatch;
        var next_rank: u16 = 0;
        for (0..self.peer_count) |idx| {
            const spec = self.peer_specs[idx];
            if (std.mem.indexOfScalar(u8, spec, '=')) |eq| {
                const r = std.fmt.parseInt(u16, spec[0..eq], 10) catch return error.InvalidPeerAddress;
                const hp = try parseHostPort(spec[eq + 1 ..]);
                out_buf[idx] = .{ .rank = r, .host = hp.host, .port = hp.port };
            } else {
                if (next_rank == self.rank) next_rank += 1;
                const hp = try parseHostPort(spec);
                out_buf[idx] = .{ .rank = next_rank, .host = hp.host, .port = hp.port };
                next_rank += 1;
            }
        }
        return out_buf[0..self.peer_count];
    }
};

const ClusterCoordinatorCtx = struct {
    store: aminfer.graph.LoadedCpuMoeStore,
    coordinator: aminfer.expert_parallel.ExpertParallelCoordinator,
    shutdown_peers: bool,

    fn initAlloc(
        allocator: std.mem.Allocator,
        amf_path: [:0]const u8,
        gcfg: aminfer.graph.GraphConfig,
        m7: M7ClusterCliFlags,
        max_tokens: u16,
        budget_override: ?aminfer.budget.Budget,
    ) !*ClusterCoordinatorCtx {
        var ep_buf: [aminfer.expert_parallel.MAX_WORLD_SIZE - 1]aminfer.expert_parallel.PeerEndpoint = undefined;
        const peers = try m7.resolveEndpoints(&ep_buf);
        const ws = m7.effectiveWorldSize();
        const plan = try aminfer.expert_parallel.ExpertPartitionPlan.init(
            ws,
            m7.rank,
            @intCast(gcfg.num_hidden_layers),
            @intCast(gcfg.num_experts),
            m7.partition,
        );
        const ctx = try allocator.create(ClusterCoordinatorCtx);
        errdefer allocator.destroy(ctx);
        ctx.store = try aminfer.graph.LoadedCpuMoeStore.loadFromFile(allocator, amf_path, gcfg, plan, budget_override);
        errdefer ctx.store.deinit(allocator);
        const top_k_cap: u32 = @max(gcfg.num_experts_per_tok, @max(gcfg.encoder_moe_top_k, gcfg.decoder_moe_top_k));
        ctx.coordinator = try aminfer.expert_parallel.ExpertParallelCoordinator.init(.{
            .plan = plan,
            .peers = peers,
            .hidden_size = gcfg.hidden_size,
            .max_tokens = max_tokens,
            .max_assignments = @as(u32, max_tokens) * @max(@as(u32, 1), top_k_cap),
            .local_backend = ctx.store.backend(),
            .transport_kind = m7.effectiveTransport(),
        });
        ctx.shutdown_peers = m7.shutdown_peers;
        return ctx;
    }

    fn deinit(self: *ClusterCoordinatorCtx, allocator: std.mem.Allocator) void {
        if (self.shutdown_peers) {
            self.coordinator.shutdownPeers() catch |err| {
                std.debug.print("warning: cluster peer shutdown failed: {s}\n", .{@errorName(err)});
            };
        }
        self.coordinator.deinit();
        self.store.deinit(allocator);
        allocator.destroy(self);
    }
};

fn transportKindLabel(k: aminfer.transport.TransportKind) []const u8 {
    return switch (k) {
        .auto => "auto",
        .tb4_tcp => "tb4_tcp",
        .tb5_rdma => "tb5_rdma",
    };
}

fn partitionLabel(p: aminfer.expert_parallel.PartitionStrategy) []const u8 {
    return switch (p) {
        .contiguous_by_rank => "contiguous",
        .round_robin => "round-robin",
    };
}

fn resolveDSparkPathAlloc(
    allocator: std.mem.Allocator,
    draft_amf_override: ?[:0]const u8,
    config_path: []const u8,
    gcfg: *const aminfer.graph.GraphConfig,
) ![:0]u8 {
    if (draft_amf_override) |p| {
        return std.fmt.allocPrintSentinel(allocator, "{s}", .{p}, 0);
    }
    const fname: []const u8 = if (gcfg.dspark_amf_file) |f| (if (f.len > 0) f else "dspark.amf") else "dspark.amf";
    if (std.fs.path.isAbsolute(fname)) {
        return std.fmt.allocPrintSentinel(allocator, "{s}", .{fname}, 0);
    }
    const dir = std.fs.path.dirname(config_path) orelse ".";
    return std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, fname }, 0);
}

fn shouldUseGraphModel(
    cfg_bytes: []const u8,
    m6: M6GraphCliFlags,
    m7: M7ClusterCliFlags,
    draft_mode: ?aminfer.draft.DraftMode,
) bool {
    if (aminfer.graph.GraphConfig.isGraphConfigJson(cfg_bytes)) return true;
    if (m6.ple_amf_count > 0) return true;
    if (m6.loop_adapter_amf_path != null) return true;
    if (m6.loop_early_exit) return true;
    if (m6.max_loops) |ml| {
        if (ml > 1) return true;
    }
    if (m7.hasPeers()) return true;
    if (draft_mode) |dm| {
        if (dm == .self_recurrent or dm == .dspark) return true;
    }
    return false;
}

fn resolvePlePathsAlloc(
    allocator: std.mem.Allocator,
    m6: *const M6GraphCliFlags,
    config_path: []const u8,
    gcfg: *const aminfer.graph.GraphConfig,
) ![][:0]u8 {
    if (m6.ple_amf_count > 0) {
        const out = try allocator.alloc([:0]u8, m6.ple_amf_count);
        for (m6.pleSlice(), 0..) |p, idx| {
            out[idx] = try std.fmt.allocPrintSentinel(allocator, "{s}", .{p}, 0);
        }
        return out;
    }
    if (gcfg.ple_shard_files.len == 0) {
        return try allocator.alloc([:0]u8, 0);
    }
    const dir = std.fs.path.dirname(config_path) orelse ".";
    const out = try allocator.alloc([:0]u8, gcfg.ple_shard_files.len);
    for (gcfg.ple_shard_files, 0..) |fname, idx| {
        if (std.fs.path.isAbsolute(fname)) {
            out[idx] = try std.fmt.allocPrintSentinel(allocator, "{s}", .{fname}, 0);
        } else {
            out[idx] = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, fname }, 0);
        }
    }
    return out;
}

fn freePlePaths(allocator: std.mem.Allocator, paths: [][:0]u8) void {
    for (paths) |p| allocator.free(p);
    allocator.free(paths);
}

fn writeLoopIsolationStats(prefix: []const u8, iso: aminfer.graph.LoopIsolationStats, out: *Io.Writer) !void {
    try out.print(
        \\{s}.loop_total_steps             {d}
        \\{s}.loop_early_exit_tokens       {d}
        \\{s}.loop_gdn_scratch_evals       {d}
        \\{s}.loop_gdn_final_commits       {d}
        \\{s}.loop_ple_r0_fills            {d}
        \\{s}.loop_ple_r_ge1_reuses        {d}
        \\{s}.loop_ple_r_ge1_nvme_reads    {d}
        \\{s}.loop_qsa_index_computes      {d}
        \\{s}.loop_qsa_index_reuses        {d}
        \\{s}.loop_moe_r_ge1_pinned_routes {d}
        \\{s}.loop_moe_r_ge1_cold_stalls   {d}
        \\{s}.loop_steplora_zero_skips     {d}
        \\{s}.loop_steplora_active_evals   {d}
        \\
    , .{
        prefix, iso.total_loop_steps,
        prefix, iso.early_exit_tokens,
        prefix, iso.gdn_scratch_evals,
        prefix, iso.gdn_final_commits,
        prefix, iso.ple_r0_fills,
        prefix, iso.ple_r_ge1_reuses,
        prefix, iso.ple_r_ge1_nvme_reads,
        prefix, iso.qsa_index_computes,
        prefix, iso.qsa_index_reuses,
        prefix, iso.moe_r_ge1_pinned_routes,
        prefix, iso.moe_r_ge1_cold_stalls,
        prefix, iso.steplora_zero_alpha_skips,
        prefix, iso.steplora_active_evals,
    });
}

const DecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
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
    gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide,
    strict_live_ram: bool = false,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
    kv_retire_q8_after_chunks: u32 = 0,
    image_path: ?[:0]const u8 = null,
    vt_test_video: ?aminfer.metal.VtCodec = null,
    m6: M6GraphCliFlags = .{},
    m69: M69CliFlags = .{},
    m7: M7ClusterCliFlags = .{},
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
    var gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide;
    var strict_live_ram = false;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;
    var kv_retire_q8_after_chunks: u32 = 0;
    var image_path: ?[:0]const u8 = null;
    var vt_test_video: ?aminfer.metal.VtCodec = null;
    var m6: M6GraphCliFlags = .{};
    var m69: M69CliFlags = .{};
    var m7: M7ClusterCliFlags = .{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (try m6.tryParseFlag(args, &i)) continue;
        if (try m69.tryParseFlag(args, &i)) continue;
        if (try m7.tryParseFlag(args, &i)) continue;
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
        } else if (std.mem.eql(u8, flag, "--gpu-prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            gpu_prefill_mode = try aminfer.qwen3.GpuPrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--strict-live-ram")) {
            strict_live_ram = true;
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else if (std.mem.eql(u8, flag, "--kv-retire-q8-after-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            kv_retire_q8_after_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--image")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            image_path = args[i];
        } else if (std.mem.eql(u8, flag, "--vt-test-video")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            vt_test_video = try parseVtCodec(args[i]);
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path,
        .tokenizer_path = tokenizer_path,
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
        .gpu_prefill_mode = gpu_prefill_mode,
        .strict_live_ram = strict_live_ram,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
        .kv_retire_q8_after_chunks = kv_retire_q8_after_chunks,
        .image_path = image_path,
        .vt_test_video = vt_test_video,
        .m6 = m6,
        .m69 = m69,
        .m7 = m7,
    };
}

const BenchDecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: ?[:0]const u8 = null,
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
    gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide,
    strict_live_ram: bool = false,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
    kv_retire_q8_after_chunks: u32 = 0,
    m6: M6GraphCliFlags = .{},
    m69: M69CliFlags = .{},
    m7: M7ClusterCliFlags = .{},
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
    var gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide;
    var strict_live_ram = false;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;
    var kv_retire_q8_after_chunks: u32 = 0;
    var m6: M6GraphCliFlags = .{};
    var m69: M69CliFlags = .{};
    var m7: M7ClusterCliFlags = .{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (try m6.tryParseFlag(args, &i)) continue;
        if (try m69.tryParseFlag(args, &i)) continue;
        if (try m7.tryParseFlag(args, &i)) continue;
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
        } else if (std.mem.eql(u8, flag, "--gpu-prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            gpu_prefill_mode = try aminfer.qwen3.GpuPrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--strict-live-ram")) {
            strict_live_ram = true;
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else if (std.mem.eql(u8, flag, "--kv-retire-q8-after-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            kv_retire_q8_after_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path,
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
        .gpu_prefill_mode = gpu_prefill_mode,
        .strict_live_ram = strict_live_ram,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
        .kv_retire_q8_after_chunks = kv_retire_q8_after_chunks,
        .m6 = m6,
        .m69 = m69,
        .m7 = m7,
    };
}

const BenchA2CliArgs = struct {
    amf_path: [:0]const u8,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
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
        .config_path = config_path,
        .tokenizer_path = tokenizer_path,
        .prompt = prompt,
        .capture_steps = capture_steps,
        .lookahead = lookahead,
        .ane_predictor_path = ane_predictor_path,
        .verify_crcs = verify_crcs,
    };
}

fn runBenchA2(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseBenchA2Args(raw_args);
    var meta = try resolveConfigAndTokenizerAlloc(allocator, cli.amf_path, cli.config_path, cli.tokenizer_path);
    defer meta.deinit(allocator);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, meta.config.bytes);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, meta.tokenizer.bytes);
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

fn runDecodeGraph(
    allocator: std.mem.Allocator,
    cli: DecodeCliArgs,
    cfg_bytes: []const u8,
    tok_bytes: []const u8,
    out: *Io.Writer,
) !void {
    var gcfg = try aminfer.graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    var prompt_ids = try tok.encodeAlloc(allocator, cli.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        allocator.free(prompt_ids);
        prompt_ids = try allocator.alloc(u32, 1);
        prompt_ids[0] = if (gcfg.bos_token_id < gcfg.vocab_size) gcfg.bos_token_id else 0;
    }

    const ple_paths = try resolvePlePathsAlloc(allocator, &cli.m6, cli.config_path orelse cli.amf_path, &gcfg);
    defer freePlePaths(allocator, ple_paths);

    const has_vision = (cli.image_path != null or cli.vt_test_video != null);
    const needed_seq: u32 = @intCast(prompt_ids.len + cli.max_tokens + 16 + (if (has_vision) @as(usize, 1024) else 0));
    const max_seq_len: u32 = @min(gcfg.max_position_embeddings, @max(@as(u32, 512), needed_seq));
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);

    if (!cli.m7.hasPeers() and cli.m7.effectiveTransport() == .tb5_rdma) {
        _ = try aminfer.transport.selectTransport(.tb5_rdma);
    }
    const cluster_ctx: ?*ClusterCoordinatorCtx = if (cli.m7.hasPeers())
        try ClusterCoordinatorCtx.initAlloc(
            allocator,
            cli.amf_path,
            gcfg,
            cli.m7,
            @intCast(@min(@as(usize, 1024), @max(prompt_ids.len, @as(usize, 64)))),
            budget_override,
        )
    else
        null;
    defer if (cluster_ctx) |cc| cc.deinit(allocator);

    var model = try aminfer.graph.GraphModel.loadFromFile(allocator, cli.amf_path, gcfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = if (cli.gpu_prefill_mode == .tree64) 64 else 16,
        .wired_budget_override = budget_override,
        .ple_paths = ple_paths,
        .loop_adapter_amf_path = cli.m6.loop_adapter_amf_path,
        .max_loops_override = cli.m6.max_loops,
        .enable_loop_early_exit = cli.m6.loop_early_exit,
        .cold_slot_count = cli.cold_slots,
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
        .cluster_coordinator = if (cluster_ctx) |cc| &cc.coordinator else null,
    });
    defer model.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;

    var req_rng: ?aminfer.rng.RequestRng = if (cli.temperature > 0.0)
        try aminfer.rng.RequestRng.init(cli.seed)
    else
        null;
    const p_res: []f32 = if (cli.temperature > 0.0)
        try allocator.alloc(f32, gcfg.vocab_size)
    else
        &.{};
    defer if (p_res.len > 0) allocator.free(p_res);

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    const prefill_t0 = try aminfer.vmstat.nowNs();
    var vision_res: ?aminfer.graph.GraphModel.MultimodalPrefillResult = null;
    var vision_source_label: []const u8 = "none";
    var next_tok: u32 = 0;
    var pos: u32 = @intCast(prompt_ids.len);
    if (cli.image_path) |img_path| {
        const frame = try aminfer.metal.VisionFrame.loadImage(model.device, img_path);
        defer frame.close();
        const frames = [_]aminfer.metal.VisionFrame{frame};
        const mm = try model.prefillMultimodal(allocator, prompt_ids, &.{}, &frames, .{
            .budget_override = budget_override,
        });
        vision_res = mm;
        vision_source_label = "image-iosurface";
        next_tok = mm.first_token;
        pos = mm.total_prompt_tokens;
    } else if (cli.vt_test_video) |codec| {
        const clip = try aminfer.metal.VtClip.encodeTestClip(codec, 128, 128, 2, 30);
        defer clip.close();
        const sess = try aminfer.metal.VtDecodeSession.init(model.device, clip, .nv12);
        defer sess.close();
        const f0 = try sess.decodeFrame(clip, 0);
        defer f0.close();
        const f1 = try sess.decodeFrame(clip, 1);
        defer f1.close();
        const frames = [_]aminfer.metal.VisionFrame{ f0, f1 };
        const mm = try model.prefillMultimodal(allocator, prompt_ids, &.{}, &frames, .{
            .budget_override = budget_override,
        });
        vision_res = mm;
        vision_source_label = if (codec == .hevc) "vt-hevc-nv12" else "vt-h264-nv12";
        next_tok = mm.first_token;
        pos = mm.total_prompt_tokens;
    } else {
        next_tok = try model.prefillChunk(prompt_ids, 0, true);
    }
    if (cli.temperature > 0.0) {
        next_tok = try model.sampleFromLastLogits(cli.temperature, req_rng.?.random(), p_res);
    }
    const prefill_t1 = try aminfer.vmstat.nowNs();

    var decode_ns: u64 = 0;
    var timed_decode_steps: u64 = 0;
    var decode_heap_delta: aminfer.vmstat.HeapDelta = std.mem.zeroes(aminfer.vmstat.HeapDelta);
    if (cli.max_tokens > 0) {
        generated[0] = next_tok;
        gen_count = 1;
        const hit_eos = (!cli.ignore_eos and next_tok == gcfg.eos_token_id);
        if (!hit_eos and gen_count < cli.max_tokens) {
            model.resetColdTelemetry();
            model.resetLoopIsolationStats();
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
                if (!cli.ignore_eos and next_tok == gcfg.eos_token_id) break;
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
        \\decode.arch                      graph ({s}, {s}, {s}, loop={s} R={d})
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
        gcfg.axes.attentionSummaryLabel(),
        gcfg.axes.ffn_kind.label(),
        gcfg.axes.residual_kind.label(),
        gcfg.axes.loop_schedule.mode.label(),
        model.effective_max_loops,
        cli.temperature,
        dec_seed_used,
        dec_seed_source,
        gcfg.num_hidden_layers,
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
    try writeLoopIsolationStats("decode", model.loopIsolationStats(), out);
    const m7_stats = model.m7Stats();
    if (gcfg.axes.backbone_kind == .causal_enc_dec or cluster_ctx != null) {
        try out.print(
            \\decode.m7.enc_prefill_dispatches {d}
            \\decode.m7.enc_decode_dispatches  {d}
            \\decode.m7.dec_decode_dispatches  {d}
            \\
        , .{
            m7_stats.encoder_layer_dispatches_prefill,
            m7_stats.encoder_layer_dispatches_decode,
            m7_stats.decoder_layer_dispatches_decode,
        });
    }
    if (cluster_ctx) |cc| {
        const avg_rpc_us: f64 = if (m7_stats.cluster_dispatches > 0)
            (@as(f64, @floatFromInt(m7_stats.cluster_remote_rpc_ns)) / @as(f64, @floatFromInt(m7_stats.cluster_dispatches))) / 1000.0
        else
            0.0;
        try out.print(
            \\decode.cluster.world_size        {d} (rank={d}, partition={s})
            \\decode.cluster.transport         {s} (rdma_reason={s})
            \\decode.cluster.dispatches        {d} (local={d}, remote={d})
            \\decode.cluster.rpc_us            {d:.2}
            \\decode.cluster.bytes_transferred {d}
            \\
        , .{
            cc.coordinator.config.plan.world_size,
            cc.coordinator.config.plan.local_rank,
            partitionLabel(cc.coordinator.config.plan.strategy),
            transportKindLabel(cc.coordinator.resolved_transport.kind),
            cc.coordinator.resolved_transport.probe.reason,
            m7_stats.cluster_dispatches,
            m7_stats.cluster_local_assignments,
            m7_stats.cluster_remote_assignments,
            avg_rpc_us,
            m7_stats.cluster_bytes_transferred,
        });
    }
    if (model.pleStats()) |ps| {
        try out.print(
            \\decode.ple_nvme_reads            {d}
            \\decode.ple_cache_hits            {d}
            \\
        , .{ ps.nvme_reads, ps.cache_hits });
    }
    if (model.coldExpertStats()) |cs| {
        try writeColdExpertStats("decode", cs, out);
    }
    if (model.kvRetireEnabled()) {
        try writeRetiredKvStats(
            "decode",
            model.retiredKvChunks(),
            model.kvRetireEvents(),
            model.activeKvWiredBytes(),
            model.retiredKvWiredBytes(),
            out,
        );
    }
    if (vision_res) |mm| {
        try out.print(
            \\decode.vision_source             {s}
            \\decode.vision_frames             {d} ({d}x{d})
            \\decode.vision_zero_copy          {s} (page16k={s})
            \\decode.vision_tokens             {d} (grid={d}x{d}x{d}, total_prompt={d})
            \\decode.vision_next_mrope_pos     {d}
            \\decode.vision_encode_us          {d:.2}
            \\
        , .{
            vision_source_label,
            mm.frame_count,
            mm.frame_width,
            mm.frame_height,
            if (mm.all_frames_zero_copy) "yes" else "no",
            if (mm.all_frames_page_aligned) "yes" else "no",
            mm.vision_tokens,
            mm.grid.grid_t,
            mm.grid.grid_h,
            mm.grid.grid_w,
            mm.total_prompt_tokens,
            mm.next_mrope_pos,
            mm.vision_encode_us,
        });
    }
    try out.writeAll("decode.token_ids                 [");
    for (generated[0..gen_count], 0..) |tid, idx| {
        if (idx > 0) try out.writeAll(", ");
        try out.print("{d}", .{tid});
    }
    try out.print("]\ndecode.text                      {s}\n", .{decoded_text});
}

fn runDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseDecodeArgs(raw_args);

    var meta = try resolveConfigAndTokenizerAlloc(allocator, cli.amf_path, cli.config_path, cli.tokenizer_path);
    defer meta.deinit(allocator);
    const cfg_bytes = meta.config.bytes;
    const tok_bytes = meta.tokenizer.bytes;

    if (shouldUseGraphModel(cfg_bytes, cli.m6, cli.m7, null)) {
        return runDecodeGraph(allocator, cli, cfg_bytes, tok_bytes, out);
    }
    if (cli.image_path != null or cli.vt_test_video != null) {
        return error.UnsupportedVisionTarget;
    }
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

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
        .max_tree_nodes = if (cli.gpu_prefill_mode == .tree64) 64 else 1,
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
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
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
        \\decode.prefill_mode              {s} (gpu={s})
        \\decode.aliased_scratch_saved_mib {d:.2}
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
        cli.gpu_prefill_mode.label(),
        @as(f64, @floatFromInt(model.aliased_scratch_saved_bytes)) / (1024.0 * 1024.0),
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
    if (model.isHeterogeneousMoe()) {
        try out.writeAll("decode.hetero_moe                yes\n");
    }
    if (model.coldStats()) |cs| {
        try writeColdExpertStats("decode", cs, out);
    }
    if (model.kvRetireEnabled()) {
        try writeRetiredKvStats(
            "decode",
            model.retiredKvChunks(),
            model.kvRetireEvents(),
            model.activeKvWiredBytes(),
            model.retiredKvWiredBytes(),
            out,
        );
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
    var cfg_meta = try resolveConfigOnlyAlloc(allocator, cli.amf_path, cli.config_path);
    defer cfg_meta.deinit(allocator);
    const cfg_bytes = cfg_meta.bytes;
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const max_seq_len: u32 = @max(@as(u32, 576), cli.prefill_tokens + 64);
    if (shouldUseGraphModel(cfg_bytes, cli.m6, cli.m7, null)) {
        var gcfg = try aminfer.graph.GraphConfig.fromJson(allocator, cfg_bytes);
        defer gcfg.deinit(allocator);
        const ple_paths = try resolvePlePathsAlloc(allocator, &cli.m6, cli.config_path orelse cli.amf_path, &gcfg);
        defer freePlePaths(allocator, ple_paths);
        if (!cli.m7.hasPeers() and cli.m7.effectiveTransport() == .tb5_rdma) {
            _ = try aminfer.transport.selectTransport(.tb5_rdma);
        }
        const cluster_ctx: ?*ClusterCoordinatorCtx = if (cli.m7.hasPeers())
            try ClusterCoordinatorCtx.initAlloc(
                allocator,
                cli.amf_path,
                gcfg,
                cli.m7,
                @intCast(@min(@as(u32, 1024), @max(cli.prefill_tokens, @as(u32, 64)))),
                budget_override,
            )
        else
            null;
        defer if (cluster_ctx) |cc| cc.deinit(allocator);
        const res = try aminfer.bench.runGraphDecodeBenchFromFile(allocator, cli.amf_path, gcfg, .{
            .prefill_tokens = cli.prefill_tokens,
            .decode_steps = cli.decode_steps,
            .max_seq_len = max_seq_len,
            .budget_override = budget_override,
            .ple_paths = ple_paths,
            .loop_adapter_amf_path = cli.m6.loop_adapter_amf_path,
            .max_loops_override = cli.m6.max_loops,
            .enable_loop_early_exit = cli.m6.loop_early_exit,
            .cold_slot_count = cli.cold_slots,
            .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
            .gpu_prefill_mode = cli.gpu_prefill_mode,
            .strict_live_ram = cli.strict_live_ram,
            .prefill_backend_mode = cli.m69.prefill_backend_mode,
            .strict_nax = cli.m69.strict_nax,
            .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
            .cluster_coordinator = if (cluster_ctx) |cc| &cc.coordinator else null,
        });
        try res.write(out);
        if (cluster_ctx) |cc| {
            try out.print(
                \\graph_bench.cluster.world_size   {d} (rank={d}, transport={s})
                \\graph_bench.cluster.dispatches   {d} (remote_assignments={d}, bytes={d})
                \\
            , .{
                cc.coordinator.config.plan.world_size,
                cc.coordinator.config.plan.local_rank,
                transportKindLabel(cc.coordinator.resolved_transport.kind),
                cc.coordinator.total_dispatches,
                cc.coordinator.total_remote_assignments,
                cc.coordinator.total_bytes_transferred,
            });
        }
        return;
    }
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

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
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
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

fn runBenchM5(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.runM5(allocator);
    defer report.deinit(allocator);
    try report.write(out);
}

fn runBenchM6(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    const report = try aminfer.bench.runM6(allocator);
    try report.write(out);
}

fn runBenchM65(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    const report = try aminfer.bench.runM65(allocator);
    try report.write(out);
}

fn runBenchM68(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    const report = try aminfer.bench.runM68(allocator);
    try report.write(out);
}

fn runBenchM69(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseM69Args(raw_args);
    const report = try aminfer.bench.runM69WithConfig(allocator, .{
        .prefill_backend_mode = cli.prefill_backend_mode,
        .strict_nax = cli.strict_nax,
        .bus_bandwidth_gbs_override = cli.bus_bandwidth_gbs,
    });
    try report.write(out);
}

fn runBenchM7(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseM69Args(raw_args);
    const report = try aminfer.bench.runM7WithConfig(allocator, .{
        .bus_bandwidth_gbs_override = cli.bus_bandwidth_gbs,
    });
    try report.write(out);
}

fn runBenchM75(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    const report = try aminfer.bench.runM75(allocator);
    try report.write(out);
}

const PeerWorkerCliArgs = struct {
    host: [4]u8,
    port: u16,
    amf_path: [:0]const u8,
    config_path: [:0]const u8,
    rank: u16,
    world_size: u16,
    partition: aminfer.expert_parallel.PartitionStrategy = .contiguous_by_rank,
    transport_kind: aminfer.transport.TransportKind = .auto,
    strict_tb5_rdma: bool = false,
    max_requests: u32 = 0,
    max_tokens: u16 = 64,
    wired_cap_mib: ?u64 = null,
    verify_crcs: bool = true,
};

fn parsePeerWorkerArgs(args: []const [:0]const u8) !PeerWorkerCliArgs {
    var listen: ?ParsedHostPort = null;
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var rank: ?u16 = null;
    var world_size: ?u16 = null;
    var partition: aminfer.expert_parallel.PartitionStrategy = .contiguous_by_rank;
    var transport_kind: aminfer.transport.TransportKind = .auto;
    var strict_tb5_rdma = false;
    var max_requests: u32 = 0;
    var max_tokens: u16 = 64;
    var wired_cap_mib: ?u64 = null;
    var verify_crcs = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--listen")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            listen = try parseHostPort(args[i]);
        } else if (std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            amf_path = args[i];
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--rank")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            rank = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--world-size")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            const ws = try std.fmt.parseInt(u16, args[i], 10);
            if (ws == 0 or ws > aminfer.expert_parallel.MAX_WORLD_SIZE) return error.InvalidWorldSize;
            world_size = ws;
        } else if (std.mem.eql(u8, flag, "--partition")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            partition = try parsePartitionStrategy(args[i]);
        } else if (std.mem.eql(u8, flag, "--cluster-transport")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            transport_kind = try parseClusterTransport(args[i]);
        } else if (std.mem.eql(u8, flag, "--strict-tb5-rdma")) {
            strict_tb5_rdma = true;
        } else if (std.mem.eql(u8, flag, "--max-requests")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_requests = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            const mt = try std.fmt.parseInt(u16, args[i], 10);
            if (mt == 0) return error.InvalidFlagValue;
            max_tokens = mt;
        } else if (std.mem.eql(u8, flag, "--wired-cap-mib")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            wired_cap_mib = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            verify_crcs = false;
        } else {
            return error.UnknownFlag;
        }
    }

    const hp = listen orelse return error.MissingRequiredFlag;
    const r = rank orelse return error.MissingRequiredFlag;
    const ws = world_size orelse return error.MissingRequiredFlag;
    if (r >= ws) return error.InvalidRank;

    return .{
        .host = hp.host,
        .port = hp.port,
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path orelse return error.MissingRequiredFlag,
        .rank = r,
        .world_size = ws,
        .partition = partition,
        .transport_kind = transport_kind,
        .strict_tb5_rdma = strict_tb5_rdma,
        .max_requests = max_requests,
        .max_tokens = max_tokens,
        .wired_cap_mib = wired_cap_mib,
        .verify_crcs = verify_crcs,
    };
}

fn runPeerWorker(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parsePeerWorkerArgs(raw_args);
    const cfg_bytes = try readFileAllocPosix(allocator, cli.config_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    var gcfg = try aminfer.graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const plan = try aminfer.expert_parallel.ExpertPartitionPlan.init(
        cli.world_size,
        cli.rank,
        @intCast(gcfg.num_hidden_layers),
        @intCast(gcfg.num_experts),
        cli.partition,
    );
    var store = try aminfer.graph.LoadedCpuMoeStore.loadFromFile(
        allocator,
        cli.amf_path,
        gcfg,
        plan,
        budget_override,
    );
    defer store.deinit(allocator);

    const eff_transport: aminfer.transport.TransportKind = if (cli.strict_tb5_rdma) .tb5_rdma else cli.transport_kind;
    const top_k_cap: u32 = @max(gcfg.num_experts_per_tok, @max(gcfg.encoder_moe_top_k, gcfg.decoder_moe_top_k));
    var server = try aminfer.expert_parallel.PeerWorkerServer.init(.{
        .host = cli.host,
        .port = cli.port,
        .transport_kind = eff_transport,
        .plan = plan,
        .hidden_size = gcfg.hidden_size,
        .max_tokens = cli.max_tokens,
        .max_assignments = @as(u32, cli.max_tokens) * @max(@as(u32, 1), top_k_cap),
        .max_requests = cli.max_requests,
        .backend = store.backend(),
    });
    defer server.deinit();

    const range = try plan.rankExpertRange(cli.rank);
    const vm_before = try aminfer.vmstat.Snapshot.capture();
    try out.print(
        \\peer_worker.listen               {d}.{d}.{d}.{d}:{d}
        \\peer_worker.rank                 {d}/{d} (partition={s})
        \\peer_worker.local_experts        {d}..{d} (of {d}, layers={d})
        \\peer_worker.transport            {s} (rdma_reason={s})
        \\peer_worker.max_requests         {d}
        \\peer_worker.status               ready
        \\
    , .{
        cli.host[0],
        cli.host[1],
        cli.host[2],
        cli.host[3],
        server.boundPort(),
        cli.rank,
        cli.world_size,
        partitionLabel(cli.partition),
        range.start,
        range.end,
        gcfg.num_experts,
        gcfg.num_hidden_layers,
        transportKindLabel(server.resolved_transport.kind),
        server.resolved_transport.probe.reason,
        cli.max_requests,
    });
    try out.flush();

    server.run();

    const vm_after = try aminfer.vmstat.Snapshot.capture();
    try aminfer.vmstat.checkNoSwap(vm_before, vm_after);
    const st = server.stats();
    try out.print(
        \\peer_worker.requests_served      {d}
        \\peer_worker.assignments_eval     {d}
        \\peer_worker.bytes_rx             {d}
        \\peer_worker.bytes_tx             {d}
        \\peer_worker.swapouts_delta       {d}
        \\
    , .{
        st.requests_served,
        st.assignments_evaluated,
        st.bytes_rx,
        st.bytes_tx,
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    });
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
    try out.flush();
    try runBenchM5(allocator, out);
    try out.flush();
    try runBenchM6(allocator, out);
    try out.flush();
    try runBenchM65(allocator, out);
    try out.flush();
    try runBenchM68(allocator, out);
    try out.flush();
    try runBenchM69(allocator, &.{}, out);
    try out.flush();
    try runBenchM7(allocator, &.{}, out);
    try out.flush();
    try runBenchM75(allocator, out);
}

const SpecDecodeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
    prompt: []const u8 = "",
    draft_amf_path: ?[:0]const u8 = null,
    draft_config_path: ?[:0]const u8 = null,
    draft_mode: ?aminfer.draft.DraftMode = null,
    placement: aminfer.draft.Placement = .auto,
    sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto,
    pearl_mode: aminfer.engine.PearlMode = .auto,
    pearl_min_target_ms: f64 = 40.0,
    ngram_nvme_path: ?[:0]const u8 = null,
    ngram_cache_pages: usize = 32,
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
    gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide,
    strict_live_ram: bool = false,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
    kv_retire_q8_after_chunks: u32 = 0,
    m6: M6GraphCliFlags = .{},
    m69: M69CliFlags = .{},
    m7: M7ClusterCliFlags = .{},
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
    var pearl_mode: aminfer.engine.PearlMode = .auto;
    var pearl_min_target_ms: f64 = 40.0;
    var ngram_nvme_path: ?[:0]const u8 = null;
    var ngram_cache_pages: usize = 32;
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
    var gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide;
    var strict_live_ram = false;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;
    var kv_retire_q8_after_chunks: u32 = 0;
    var m6: M6GraphCliFlags = .{};
    var m69: M69CliFlags = .{};
    var m7: M7ClusterCliFlags = .{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (try m6.tryParseFlag(args, &i)) continue;
        if (try m69.tryParseFlag(args, &i)) continue;
        if (try m7.tryParseFlag(args, &i)) continue;
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
        } else if (std.mem.eql(u8, flag, "--pearl")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            pearl_mode = try aminfer.engine.PearlMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--pearl-min-target-ms")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            pearl_min_target_ms = try std.fmt.parseFloat(f64, args[i]);
            if (!std.math.isFinite(pearl_min_target_ms) or pearl_min_target_ms < 0.0) return error.InvalidFlagValue;
        } else if (std.mem.eql(u8, flag, "--ngram-nvme")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ngram_nvme_path = args[i];
        } else if (std.mem.eql(u8, flag, "--ngram-cache-pages")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ngram_cache_pages = try std.fmt.parseInt(usize, args[i], 10);
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
        } else if (std.mem.eql(u8, flag, "--gpu-prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            gpu_prefill_mode = try aminfer.qwen3.GpuPrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--strict-live-ram")) {
            strict_live_ram = true;
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else if (std.mem.eql(u8, flag, "--kv-retire-q8-after-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            kv_retire_q8_after_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path,
        .tokenizer_path = tokenizer_path,
        .prompt = prompt,
        .draft_amf_path = draft_amf_path,
        .draft_config_path = draft_config_path,
        .draft_mode = draft_mode,
        .placement = placement,
        .sme_router_mode = sme_router_mode,
        .pearl_mode = pearl_mode,
        .pearl_min_target_ms = pearl_min_target_ms,
        .ngram_nvme_path = ngram_nvme_path,
        .ngram_cache_pages = ngram_cache_pages,
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
        .gpu_prefill_mode = gpu_prefill_mode,
        .strict_live_ram = strict_live_ram,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
        .kv_retire_q8_after_chunks = kv_retire_q8_after_chunks,
        .m6 = m6,
        .m69 = m69,
        .m7 = m7,
    };
}

fn runSpecDecodeGraph(
    allocator: std.mem.Allocator,
    cli: SpecDecodeCliArgs,
    cfg_bytes: []const u8,
    tok_bytes: []const u8,
    out: *Io.Writer,
) !void {
    var gcfg = try aminfer.graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    var prompt_ids = try tok.encodeAlloc(allocator, cli.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        allocator.free(prompt_ids);
        prompt_ids = try allocator.alloc(u32, 1);
        prompt_ids[0] = if (gcfg.bos_token_id < gcfg.vocab_size) gcfg.bos_token_id else 0;
    }

    var draft_cfg_opt: ?aminfer.draft.DraftConfig = null;
    var mode: aminfer.draft.DraftMode = cli.draft_mode orelse blk: {
        const has_dspark_file = if (gcfg.dspark_amf_file) |f| f.len > 0 else false;
        if (gcfg.axes.backbone_kind == .causal_enc_dec and (cli.draft_amf_path != null or has_dspark_file)) {
            break :blk .dspark;
        }
        if (cli.draft_amf_path == null and gcfg.axes.loop_schedule.isRecurrent()) break :blk .self_recurrent;
        break :blk .l1_only;
    };

    if (cli.draft_config_path) |dc_path| {
        if (mode != .dspark) {
            const dcfg_bytes = try readFileAllocPosix(allocator, dc_path, 4 * 1024 * 1024);
            defer allocator.free(dcfg_bytes);
            draft_cfg_opt = try aminfer.draft.DraftConfig.fromJson(allocator, dcfg_bytes);
            if (cli.draft_mode == null) {
                mode = draft_cfg_opt.?.format;
            } else {
                draft_cfg_opt.?.format = mode;
            }
        }
    }
    if (mode != .l1_only and mode != .self_recurrent and mode != .dspark) {
        if (cli.draft_amf_path == null or draft_cfg_opt == null) return error.MissingRequiredFlag;
        draft_cfg_opt.?.format = mode;
    }

    const ple_paths = try resolvePlePathsAlloc(allocator, &cli.m6, cli.config_path orelse cli.amf_path, &gcfg);
    defer freePlePaths(allocator, ple_paths);

    const needed_seq: u32 = @intCast(prompt_ids.len + cli.max_tokens + cli.max_tree_nodes + 32);
    const max_seq_len: u32 = @min(gcfg.max_position_embeddings, @max(@as(u32, 512), needed_seq));
    const target_tree_cap: u32 = if (cli.enable_pruning)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, cli.max_tree_nodes) * 2, 24))
    else
        @as(u32, cli.max_tree_nodes);
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const aux_ids: ?[3]u32 = if (draft_cfg_opt) |dc| dc.aux_layer_ids else null;
    const needs_taps = (mode != .l1_only and mode != .self_recurrent and mode != .dspark);

    if (!cli.m7.hasPeers() and cli.m7.effectiveTransport() == .tb5_rdma) {
        _ = try aminfer.transport.selectTransport(.tb5_rdma);
    }
    const cluster_ctx: ?*ClusterCoordinatorCtx = if (cli.m7.hasPeers())
        try ClusterCoordinatorCtx.initAlloc(
            allocator,
            cli.amf_path,
            gcfg,
            cli.m7,
            @intCast(@min(@as(usize, 1024), @max(prompt_ids.len, @as(usize, target_tree_cap)))),
            budget_override,
        )
    else
        null;
    defer if (cluster_ctx) |cc| cc.deinit(allocator);

    var model = try aminfer.graph.GraphModel.loadFromFile(allocator, cli.amf_path, gcfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = target_tree_cap,
        .enable_feature_taps = needs_taps,
        .aux_layer_ids = aux_ids,
        .wired_budget_override = budget_override,
        .ple_paths = ple_paths,
        .loop_adapter_amf_path = cli.m6.loop_adapter_amf_path,
        .max_loops_override = cli.m6.max_loops,
        .enable_loop_early_exit = cli.m6.loop_early_exit,
        .cold_slot_count = cli.cold_slots,
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
        .cluster_coordinator = if (cluster_ctx) |cc| &cc.coordinator else null,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinitForGraph(allocator, &model);
    if (needs_taps) {
        draft_head_storage = try aminfer.draft.DraftHead.loadFromFileForGraph(
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

    var dspark_loaded: ?*aminfer.engine.LoadedDSparkProposer = null;
    defer if (dspark_loaded) |dp| dp.deinit(allocator);
    if (mode == .dspark) {
        const dspark_path = try resolveDSparkPathAlloc(allocator, cli.draft_amf_path, cli.config_path orelse cli.amf_path, &gcfg);
        defer allocator.free(dspark_path);
        dspark_loaded = try aminfer.engine.LoadedDSparkProposer.loadFromFile(
            allocator,
            &model,
            dspark_path,
            null,
            budget_override,
        );
    }

    var ngram_tbl_storage: aminfer.ngram_nvme.NvmeNgramTable = undefined;
    var ngram_tbl_ptr: ?*aminfer.ngram_nvme.NvmeNgramTable = null;
    defer if (ngram_tbl_ptr) |nt| nt.deinit(allocator);
    if (cli.ngram_nvme_path) |ang_path| {
        const eff_ngram_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_tbl_storage = try aminfer.ngram_nvme.NvmeNgramTable.open(allocator, ang_path, .{
            .cache_pages = eff_ngram_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_tbl_ptr = &ngram_tbl_storage;
    }

    const eng_opts: aminfer.engine.SpecConfig = .{
        .mode = mode,
        .placement = cli.placement,
        .max_tree_nodes = cli.max_tree_nodes,
        .max_draft_depth = cli.max_draft_depth,
        .branch_width = cli.branch_width,
        .top_k_per_node = cli.top_k,
        .temperature = cli.temperature,
        .enable_pruning = cli.enable_pruning,
        .pearl_mode = cli.pearl_mode,
        .pearl_min_target_us = cli.pearl_min_target_ms * 1000.0,
        .ngram_table = ngram_tbl_ptr,
    };
    var eng = if (dspark_loaded) |dp|
        try aminfer.engine.SpecEngine.initWithDSpark(allocator, &model, &dp.proposer, eng_opts)
    else
        try aminfer.engine.SpecEngine.initWithGraph(allocator, &model, draft_head_ptr, eng_opts);
    defer eng.deinit(allocator);

    const generated = try allocator.alloc(u32, cli.max_tokens);
    defer allocator.free(generated);
    var gen_count: usize = 0;
    var round_buf: [aminfer.engine.MAX_TREE_NODES]u32 = undefined;

    var req_rng: ?aminfer.rng.RequestRng = if (cli.temperature > 0.0)
        try aminfer.rng.RequestRng.init(cli.seed)
    else
        null;
    const rng_opt: ?std.Random = if (req_rng != null) req_rng.?.random() else null;

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
        const first_is_eos = (!cli.ignore_eos and first_tok == gcfg.eos_token_id);
        if (!first_is_eos and gen_count < cli.max_tokens) {
            model.resetColdTelemetry();
            model.resetLoopIsolationStats();
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
                    if (!cli.ignore_eos and tid == gcfg.eos_token_id) {
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
    const spec_seed_used: u64 = if (req_rng) |rr| rr.seed_used else (cli.seed orelse 0);
    const spec_seed_source: []const u8 = if (req_rng) |rr| rr.seedSourceName() else "greedy";

    try out.print(
        \\spec.arch                        graph ({s}, {s}, {s}, loop={s} R={d})
        \\spec.mode                        {s}
        \\spec.placement                   {s} (requested={s})
        \\
    , .{
        gcfg.axes.attentionSummaryLabel(),
        gcfg.axes.ffn_kind.label(),
        gcfg.axes.residual_kind.label(),
        gcfg.axes.loop_schedule.mode.label(),
        model.effective_max_loops,
        mode.label(),
        eng.effectivePlacement().label(),
        cli.placement.label(),
    });
    if (draft_head_ptr) |dh| {
        if (dh.placement_gate_sme2_us > 0.0 or dh.placement_gate_gpu_us > 0.0) {
            try out.print(
                \\spec.placement_gate_sme2_us      {d:.1}
                \\spec.placement_gate_gpu_us       {d:.1}
                \\spec.sme2_proj_bytes             {d}
                \\
            , .{ dh.placement_gate_sme2_us, dh.placement_gate_gpu_us, dh.sme2_proj_bytes });
        }
    }
    try out.print(
        \\spec.temperature                 {d:.2}
        \\spec.seed_used                   {d}
        \\spec.seed_source                 {s}
        \\spec.layers                      {d}
        \\spec.wired_mib                   {d}
        \\spec.prompt_tokens               {d}
        \\spec.generated_tokens            {d}
        \\spec.rounds                      {d}
        \\spec.mean_accept_len             {d:.3}
        \\spec.self_recurrent_draft_steps  {d}
        \\spec.prefill_ms                  {d:.3}
        \\spec.decode_ms_per_tok           {d:.3}
        \\spec.decode_tok_per_s            {d:.2}
        \\spec.malloc_blocks_delta         {d}
        \\spec.malloc_bytes_delta          {d}
        \\spec.swapouts_delta              {d}
        \\
    , .{
        cli.temperature,
        spec_seed_used,
        spec_seed_source,
        gcfg.num_hidden_layers,
        eng.totalWiredBytes() >> 20,
        prompt_ids.len,
        gen_count,
        eng.stats.rounds,
        eng.stats.meanAcceptanceLength(),
        eng.stats.self_recurrent_draft_steps,
        prefill_ms,
        decode_ms_per_tok,
        decode_tok_per_s,
        decode_heap_delta.blocks_in_use,
        decode_heap_delta.size_in_use,
        aminfer.vmstat.Snapshot.swapoutDelta(decode_vm_before, vm_after),
    });
    try writeLoopIsolationStats("spec", model.loopIsolationStats(), out);
    if (gcfg.axes.backbone_kind == .causal_enc_dec or cluster_ctx != null) {
        const m7_stats = model.m7Stats();
        try out.print(
            \\spec.m7.enc_prefill_dispatches   {d}
            \\spec.m7.enc_decode_dispatches    {d}
            \\spec.m7.dec_decode_dispatches    {d}
            \\
        , .{
            m7_stats.encoder_layer_dispatches_prefill,
            m7_stats.encoder_layer_dispatches_decode,
            m7_stats.decoder_layer_dispatches_decode,
        });
    }
    if (model.pleStats()) |ps| {
        try out.print(
            \\spec.ple_nvme_reads              {d}
            \\spec.ple_cache_hits              {d}
            \\
        , .{ ps.nvme_reads, ps.cache_hits });
    }
    if (model.coldExpertStats()) |cs| {
        try writeColdExpertStats("spec", cs, out);
    }
    if (model.kvRetireEnabled()) {
        try writeRetiredKvStats(
            "spec",
            model.retiredKvChunks(),
            model.kvRetireEvents(),
            model.activeKvWiredBytes(),
            model.retiredKvWiredBytes(),
            out,
        );
    }
    try out.writeAll("spec.token_ids                   [");
    for (generated[0..gen_count], 0..) |tid, idx| {
        if (idx > 0) try out.writeAll(", ");
        try out.print("{d}", .{tid});
    }
    try out.print("]\nspec.text                        {s}\n", .{decoded_text});
}

fn runSpecDecode(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseSpecDecodeArgs(raw_args);

    var meta = try resolveConfigAndTokenizerAlloc(allocator, cli.amf_path, cli.config_path, cli.tokenizer_path);
    defer meta.deinit(allocator);
    const cfg_bytes = meta.config.bytes;
    const tok_bytes = meta.tokenizer.bytes;

    if (shouldUseGraphModel(cfg_bytes, cli.m6, cli.m7, cli.draft_mode)) {
        return runSpecDecodeGraph(allocator, cli, cfg_bytes, tok_bytes, out);
    }
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
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinit(allocator, &model);

    var small_draft_storage: aminfer.draft.SmallDraftModel = undefined;
    var small_draft_ptr: ?*aminfer.draft.SmallDraftModel = null;
    defer if (small_draft_ptr) |sd| sd.deinit(allocator);

    var ngram_tbl_storage: aminfer.ngram_nvme.NvmeNgramTable = undefined;
    var ngram_tbl_ptr: ?*aminfer.ngram_nvme.NvmeNgramTable = null;
    defer if (ngram_tbl_ptr) |nt| nt.deinit(allocator);
    if (cli.ngram_nvme_path) |ang_path| {
        const eff_ngram_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_tbl_storage = try aminfer.ngram_nvme.NvmeNgramTable.open(allocator, ang_path, .{
            .cache_pages = eff_ngram_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_tbl_ptr = &ngram_tbl_storage;
    }

    if (mode == .small_draft) {
        small_draft_storage = try aminfer.draft.SmallDraftModel.loadFromFile(
            allocator,
            &model,
            cli.draft_amf_path.?,
            small_draft_cfg_opt.?,
            .{
                .placement = cli.placement,
                .max_seq_len = max_seq_len,
                .max_tree_slots = @min(aminfer.attention.MAX_TREE_NODES, @max(@as(u32, 32), target_tree_cap)),
                .verify_crcs = cli.verify_crcs,
                .own_compute_queue = (cli.pearl_mode != .off),
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
                .max_tree_slots = @min(aminfer.attention.MAX_TREE_NODES, @max(@as(u32, 32), target_tree_cap)),
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
        .pearl_mode = cli.pearl_mode,
        .pearl_min_target_us = cli.pearl_min_target_ms * 1000.0,
        .ngram_table = ngram_tbl_ptr,
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
        if (dh.placement_gate_ane_us > 0.0 or dh.placement_gate_sme2_us > 0.0 or dh.placement_gate_gpu_us > 0.0) {
            if (dh.placement_gate_ane_us > 0.0) {
                try out.print("spec.placement_gate_ane_us       {d:.1}\n", .{dh.placement_gate_ane_us});
            }
            if (dh.placement_gate_sme2_us > 0.0) {
                try out.print(
                    \\spec.placement_gate_sme2_us      {d:.1}
                    \\spec.sme2_proj_bytes             {d}
                    \\
                , .{ dh.placement_gate_sme2_us, dh.sme2_proj_bytes });
            }
            if (dh.placement_gate_gpu_us > 0.0) {
                try out.print("spec.placement_gate_gpu_us       {d:.1}\n", .{dh.placement_gate_gpu_us});
            }
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
    if (model.isHeterogeneousMoe()) {
        try out.writeAll("spec.hetero_moe                  yes\n");
    }
    if (model.kvRetireEnabled()) {
        try writeRetiredKvStats(
            "spec",
            model.retiredKvChunks(),
            model.kvRetireEvents(),
            model.activeKvWiredBytes(),
            model.retiredKvWiredBytes(),
            out,
        );
    }
    if (ngram_tbl_ptr) |nt| {
        try out.print(
            \\spec.ngram_nvme                  on (buckets={d}, entries={d}, wired_kib={d})
            \\spec.ngram_queries               {d}
            \\spec.ngram_hits                  {d}
            \\spec.ngram_seed_hits             {d}
            \\spec.ngram_cache_hits            {d}
            \\spec.ngram_nvme_reads            {d}
            \\spec.ngram_mean_query_us         {d:.2}
            \\
        , .{
            nt.header.num_buckets,
            nt.header.total_entries,
            nt.totalWiredBytes() >> 10,
            eng.stats.ngram_queries,
            eng.stats.ngram_hits,
            eng.stats.ngram_seed_hits,
            eng.stats.ngram_cache_hits,
            eng.stats.ngram_nvme_reads,
            nt.stats.meanQueryUs(),
        });
    }
    if (mode == .small_draft or eng.stats.pearl_enabled or eng.stats.pearl_rounds > 0) {
        try out.print(
            \\spec.pearl_mode                  {s} (enabled={s}, own_queue={s})
            \\spec.pearl_rounds                {d}
            \\spec.pearl_hits                  {d}
            \\spec.pearl_partial_prefix_hits   {d}
            \\spec.pearl_rollbacks             {d}
            \\spec.pearl_saved_ms              {d:.3}
            \\spec.pearl_worker_ms             {d:.3}
            \\
        , .{
            cli.pearl_mode.label(),
            if (eng.stats.pearl_enabled) "yes" else "no",
            if (small_draft_ptr) |sd| (if (sd.hasOwnComputeQueue()) "yes" else "no") else "no",
            eng.stats.pearl_rounds,
            eng.stats.pearl_hits,
            eng.stats.pearl_partial_prefix_hits,
            eng.stats.pearl_rollbacks,
            @as(f64, @floatFromInt(eng.stats.pearl_saved_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(eng.stats.pearl_worker_ns)) / 1_000_000.0,
        });
    }
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

// ---------------------------------------------------------------------------
// C7 + C8 Zero-Allocation kqueue HTTP/SSE Inference Server (`aminfer serve`)
// ---------------------------------------------------------------------------

var g_active_serve_ptr: std.atomic.Value(usize) = .init(0);

fn handleServeShutdownSignal(_: std.c.SIG) callconv(.c) void {
    const ptr_val = g_active_serve_ptr.load(.acquire);
    if (ptr_val != 0) {
        const srv: *aminfer.server.Server = @ptrFromInt(ptr_val);
        srv.stop_requested.store(true, .seq_cst);
    }
}

const ServeCliArgs = struct {
    amf_path: [:0]const u8,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    api_key: ?[]const u8 = null,
    unix_socket_path: ?[]const u8 = null,
    lora_specs: [aminfer.lora.MAX_ADAPTERS]LoraSpecArg = undefined,
    lora_spec_count: usize = 0,
    disk_prefix_cache_dir: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
    max_seq_len: u32 = 4096,
    max_conns: usize = aminfer.server.DEFAULT_MAX_CONNS,
    max_active_requests: usize = aminfer.server.DEFAULT_MAX_ACTIVE_REQUESTS,
    max_active_requests_explicit: bool = false,
    batch_mode: ServeBatchMode = .auto,
    prefix_cache_chunks: u32 = 16,
    decode_share: f32 = aminfer.batch_scheduler.DEFAULT_DECODE_SHARE,
    adaptive_prefill_narrow: bool = true,
    message_boundary_cache: bool = true,
    kv_retire_q8_after_chunks: u32 = 0,
    ngram_packs: [aminfer.ngram_nvme.MAX_DOMAIN_PACKS]NgramPackArg = undefined,
    ngram_pack_count: usize = 0,
    draft_amf_path: ?[:0]const u8 = null,
    draft_config_path: ?[:0]const u8 = null,
    draft_mode: ?aminfer.draft.DraftMode = null,
    placement: aminfer.draft.Placement = .auto,
    sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto,
    pearl_mode: aminfer.engine.PearlMode = .auto,
    pearl_min_target_ms: f64 = 40.0,
    ngram_nvme_path: ?[:0]const u8 = null,
    ngram_cache_pages: usize = 32,
    max_tree_nodes: u16 = 16,
    max_draft_depth: u16 = 4,
    branch_width: u16 = 2,
    top_k: u32 = 3,
    enable_pruning: bool = true,
    verify_crcs: bool = true,
    cold_experts_per_layer: u32 = 0,
    heat_profile_path: ?[:0]const u8 = null,
    cold_slots: u32 = 64,
    wired_cap_mib: ?u64 = null,
    io_backend: aminfer.aio.IoBackend = .auto,
    prefill_mode: aminfer.qwen3.PrefillMode = .auto,
    gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide,
    strict_live_ram: bool = false,
    ane_predictor_path: ?[:0]const u8 = null,
    enable_ane_predictor: bool = true,
    m6: M6GraphCliFlags = .{},
    m69: M69CliFlags = .{},
};

fn parseServeArgs(args: []const [:0]const u8) !ServeCliArgs {
    var amf_path: ?[:0]const u8 = null;
    var config_path: ?[:0]const u8 = null;
    var tokenizer_path: ?[:0]const u8 = null;
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var api_key: ?[]const u8 = null;
    var unix_socket_path: ?[]const u8 = null;
    var lora_specs: [aminfer.lora.MAX_ADAPTERS]LoraSpecArg = undefined;
    var lora_spec_count: usize = 0;
    var disk_prefix_cache_dir: ?[]const u8 = null;
    var model_id: ?[]const u8 = null;
    var max_seq_len: u32 = 4096;
    var max_conns: usize = aminfer.server.DEFAULT_MAX_CONNS;
    var max_active_requests: usize = aminfer.server.DEFAULT_MAX_ACTIVE_REQUESTS;
    var max_active_requests_explicit = false;
    var batch_mode: ServeBatchMode = .auto;
    var prefix_cache_chunks: u32 = 16;
    var decode_share: f32 = aminfer.batch_scheduler.DEFAULT_DECODE_SHARE;
    var adaptive_prefill_narrow = true;
    var message_boundary_cache = true;
    var kv_retire_q8_after_chunks: u32 = 0;
    var ngram_packs: [aminfer.ngram_nvme.MAX_DOMAIN_PACKS]NgramPackArg = undefined;
    var ngram_pack_count: usize = 0;
    var draft_amf_path: ?[:0]const u8 = null;
    var draft_config_path: ?[:0]const u8 = null;
    var draft_mode: ?aminfer.draft.DraftMode = null;
    var placement: aminfer.draft.Placement = .auto;
    var sme_router_mode: aminfer.qwen3.SmeRouterMode = .auto;
    var pearl_mode: aminfer.engine.PearlMode = .auto;
    var pearl_min_target_ms: f64 = 40.0;
    var ngram_nvme_path: ?[:0]const u8 = null;
    var ngram_cache_pages: usize = 32;
    var max_tree_nodes: u16 = 16;
    var max_draft_depth: u16 = 4;
    var branch_width: u16 = 2;
    var top_k: u32 = 3;
    var enable_pruning = true;
    var verify_crcs = true;
    var cold_experts_per_layer: u32 = 0;
    var heat_profile_path: ?[:0]const u8 = null;
    var cold_slots: u32 = 64;
    var wired_cap_mib: ?u64 = null;
    var io_backend: aminfer.aio.IoBackend = .auto;
    var prefill_mode: aminfer.qwen3.PrefillMode = .auto;
    var gpu_prefill_mode: aminfer.qwen3.GpuPrefillMode = .wide;
    var strict_live_ram = false;
    var ane_predictor_path: ?[:0]const u8 = null;
    var enable_ane_predictor = true;
    var m6: M6GraphCliFlags = .{};
    var m69: M69CliFlags = .{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (try m6.tryParseFlag(args, &i)) continue;
        if (try m69.tryParseFlag(args, &i)) continue;
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
        } else if (std.mem.eql(u8, flag, "--host")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            host = args[i];
        } else if (std.mem.eql(u8, flag, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--model-id")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            model_id = args[i];
        } else if (std.mem.eql(u8, flag, "--max-seq-len")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_seq_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--max-conns")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_conns = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--max-active-requests")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            max_active_requests = try std.fmt.parseInt(usize, args[i], 10);
            max_active_requests_explicit = true;
        } else if (std.mem.eql(u8, flag, "--batch-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            batch_mode = try ServeBatchMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--prefix-cache-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            prefix_cache_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--decode-share")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            decode_share = try std.fmt.parseFloat(f32, args[i]);
            if (!std.math.isFinite(decode_share) or decode_share < 0.0 or decode_share > 1.0) {
                return error.InvalidDecodeShare;
            }
        } else if (std.mem.eql(u8, flag, "--adaptive-prefill-narrow")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            adaptive_prefill_narrow = try parseOnOff(args[i]);
        } else if (std.mem.eql(u8, flag, "--message-boundary-cache")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            message_boundary_cache = try parseOnOff(args[i]);
        } else if (std.mem.eql(u8, flag, "--kv-retire-q8-after-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            kv_retire_q8_after_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--ngram-pack")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (ngram_pack_count >= ngram_packs.len) return error.TooManyNgramPacks;
            ngram_packs[ngram_pack_count] = try parseNgramPackArg(args[i]);
            ngram_pack_count += 1;
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
        } else if (std.mem.eql(u8, flag, "--pearl")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            pearl_mode = try aminfer.engine.PearlMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--pearl-min-target-ms")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            pearl_min_target_ms = try std.fmt.parseFloat(f64, args[i]);
            if (!std.math.isFinite(pearl_min_target_ms) or pearl_min_target_ms < 0.0) return error.InvalidFlagValue;
        } else if (std.mem.eql(u8, flag, "--ngram-nvme")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ngram_nvme_path = args[i];
        } else if (std.mem.eql(u8, flag, "--ngram-cache-pages")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ngram_cache_pages = try std.fmt.parseInt(usize, args[i], 10);
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
        } else if (std.mem.eql(u8, flag, "--gpu-prefill-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            gpu_prefill_mode = try aminfer.qwen3.GpuPrefillMode.parse(args[i]);
        } else if (std.mem.eql(u8, flag, "--strict-live-ram")) {
            strict_live_ram = true;
        } else if (std.mem.eql(u8, flag, "--ane-predictor")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            ane_predictor_path = args[i];
        } else if (std.mem.eql(u8, flag, "--no-ane-predictor")) {
            enable_ane_predictor = false;
        } else if (std.mem.eql(u8, flag, "--api-key")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (args[i].len == 0 or args[i].len > aminfer.server.MAX_API_KEY_BYTES) return error.InvalidApiKey;
            api_key = args[i];
        } else if (std.mem.eql(u8, flag, "--unix-socket")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (args[i].len == 0 or args[i].len > aminfer.server.MAX_UNIX_SOCKET_PATH_BYTES) return error.InvalidUnixSocketPath;
            unix_socket_path = args[i];
        } else if (std.mem.eql(u8, flag, "--lora")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (lora_spec_count >= lora_specs.len) return error.TooManyLoraAdapters;
            lora_specs[lora_spec_count] = try parseLoraSpecArg(args[i]);
            lora_spec_count += 1;
        } else if (std.mem.eql(u8, flag, "--disk-prefix-cache")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (args[i].len == 0) return error.InvalidFlagValue;
            disk_prefix_cache_dir = args[i];
        } else {
            return error.UnknownFlag;
        }
    }

    return .{
        .amf_path = amf_path orelse return error.MissingRequiredFlag,
        .config_path = config_path,
        .tokenizer_path = tokenizer_path,
        .host = host,
        .port = port,
        .api_key = api_key,
        .unix_socket_path = unix_socket_path,
        .lora_specs = lora_specs,
        .lora_spec_count = lora_spec_count,
        .disk_prefix_cache_dir = disk_prefix_cache_dir,
        .model_id = model_id,
        .max_seq_len = max_seq_len,
        .max_conns = max_conns,
        .max_active_requests = max_active_requests,
        .max_active_requests_explicit = max_active_requests_explicit,
        .batch_mode = batch_mode,
        .prefix_cache_chunks = prefix_cache_chunks,
        .decode_share = decode_share,
        .adaptive_prefill_narrow = adaptive_prefill_narrow,
        .message_boundary_cache = message_boundary_cache,
        .kv_retire_q8_after_chunks = kv_retire_q8_after_chunks,
        .ngram_packs = ngram_packs,
        .ngram_pack_count = ngram_pack_count,
        .draft_amf_path = draft_amf_path,
        .draft_config_path = draft_config_path,
        .draft_mode = draft_mode,
        .placement = placement,
        .sme_router_mode = sme_router_mode,
        .pearl_mode = pearl_mode,
        .pearl_min_target_ms = pearl_min_target_ms,
        .ngram_nvme_path = ngram_nvme_path,
        .ngram_cache_pages = ngram_cache_pages,
        .max_tree_nodes = max_tree_nodes,
        .max_draft_depth = max_draft_depth,
        .branch_width = branch_width,
        .top_k = top_k,
        .enable_pruning = enable_pruning,
        .verify_crcs = verify_crcs,
        .cold_experts_per_layer = cold_experts_per_layer,
        .heat_profile_path = heat_profile_path,
        .cold_slots = cold_slots,
        .wired_cap_mib = wired_cap_mib,
        .io_backend = io_backend,
        .prefill_mode = prefill_mode,
        .gpu_prefill_mode = gpu_prefill_mode,
        .strict_live_ram = strict_live_ram,
        .ane_predictor_path = ane_predictor_path,
        .enable_ane_predictor = enable_ane_predictor,
        .m6 = m6,
        .m69 = m69,
    };
}

const ServeTarget = union(enum) {
    qwen3: *aminfer.qwen3.Qwen3Model,
    graph: *aminfer.graph.GraphModel,
};

/// Zero-allocation inference runner bridging `Qwen3Model` / `GraphModel` / `SpecEngine` +
/// `Tokenizer` into `server.Server`'s `EngineHandler` (`C7`, `C8`, `C18`, `C22`, `D11`, `W9`, `W19`).
const ServeRunner = struct {
    target: ServeTarget,
    spec_engine: ?*aminfer.engine.SpecEngine,
    ngram_registry: ?*aminfer.ngram_nvme.DomainNgramRegistry,
    tok: *const aminfer.tokenizer.Tokenizer,
    vocab_size: u32,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    max_seq_len: u32,
    stop_tokens: [4]u32,
    stop_token_count: usize,
    encode_scratch: aminfer.tokenizer.EncodeScratch,
    prompt_ids_buf: []u32,
    p_res: []f32,
    sampler_cand_ids: []u32,
    grammar_mask: []u32,
    vocab_index: ?aminfer.grammar.VocabIndex,
    penalty_tracker: *aminfer.sampling.DefaultPenaltyTracker,
    model_mutex: ?*std.c.pthread_mutex_t = null,
    server_ptr: ?*aminfer.server.Server = null,
    round_buf: [aminfer.engine.MAX_TREE_NODES]u32,
    steady_steps: u64 = 0,
    steady_heap_start: ?aminfer.vmstat.Snapshot = null,
    steady_heap_end: ?aminfer.vmstat.Snapshot = null,
    vmstat_errors: u64 = 0,
    ngram_query_errors: u64 = 0,

    const EmitResult = enum { ok, stop, cancelled };

    fn init(
        allocator: std.mem.Allocator,
        target: ServeTarget,
        spec_engine: ?*aminfer.engine.SpecEngine,
        ngram_registry: ?*aminfer.ngram_nvme.DomainNgramRegistry,
        tok: *const aminfer.tokenizer.Tokenizer,
        vocab_size: u32,
        bos_token_id: ?u32,
        eos_token_id: ?u32,
        max_seq_len: u32,
    ) !ServeRunner {
        var encode_scratch = try aminfer.tokenizer.EncodeScratch.init(allocator, @max(@as(usize, 8192), @as(usize, max_seq_len) * 2));
        errdefer encode_scratch.deinit(allocator);
        const prompt_ids_buf = try allocator.alloc(u32, max_seq_len);
        errdefer allocator.free(prompt_ids_buf);
        const p_res = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(p_res);
        const sampler_cand_ids = try allocator.alloc(u32, vocab_size);
        errdefer allocator.free(sampler_cand_ids);
        const mask_words = (@as(usize, vocab_size) + 31) / 32;
        const grammar_mask = try allocator.alloc(u32, @max(@as(usize, 1), mask_words));
        errdefer allocator.free(grammar_mask);
        const penalty_tracker = try allocator.create(aminfer.sampling.DefaultPenaltyTracker);
        errdefer allocator.destroy(penalty_tracker);
        penalty_tracker.* = aminfer.sampling.DefaultPenaltyTracker.init();

        var stop_tokens: [4]u32 = @splat(0);
        var stop_cnt: usize = 0;
        const candidates = [_]?u32{
            eos_token_id,
            tok.addedTokenId("<|im_end|>"),
            tok.addedTokenId("<|endoftext|>"),
        };
        for (candidates) |maybe_id| {
            const id = maybe_id orelse continue;
            var dup = false;
            for (stop_tokens[0..stop_cnt]) |existing| {
                if (existing == id) {
                    dup = true;
                    break;
                }
            }
            if (!dup and stop_cnt < stop_tokens.len) {
                stop_tokens[stop_cnt] = id;
                stop_cnt += 1;
            }
        }

        const vocab_index: ?aminfer.grammar.VocabIndex = if (aminfer.grammar.VocabIndex.initFromTokenizer(
            allocator,
            tok,
            stop_tokens[0..stop_cnt],
        )) |vi| vi else |_| null;

        return .{
            .target = target,
            .spec_engine = spec_engine,
            .ngram_registry = ngram_registry,
            .tok = tok,
            .vocab_size = vocab_size,
            .bos_token_id = bos_token_id,
            .eos_token_id = eos_token_id,
            .max_seq_len = max_seq_len,
            .stop_tokens = stop_tokens,
            .stop_token_count = stop_cnt,
            .encode_scratch = encode_scratch,
            .prompt_ids_buf = prompt_ids_buf,
            .p_res = p_res,
            .sampler_cand_ids = sampler_cand_ids,
            .grammar_mask = grammar_mask,
            .vocab_index = vocab_index,
            .penalty_tracker = penalty_tracker,
            .round_buf = @splat(0),
        };
    }

    fn recordSteadyStep(self: *ServeRunner) void {
        self.steady_steps += 1;
        if (self.steady_steps == 2) {
            if (aminfer.vmstat.Snapshot.capture()) |snap| {
                self.steady_heap_start = snap;
            } else |_| {
                self.vmstat_errors += 1;
            }
        } else if (self.steady_steps > 2) {
            if (aminfer.vmstat.Snapshot.capture()) |snap| {
                self.steady_heap_end = snap;
            } else |_| {
                self.vmstat_errors += 1;
            }
        }
    }

    fn steadyHeapDelta(self: *const ServeRunner) aminfer.vmstat.HeapDelta {
        if (self.steady_heap_start) |s| {
            if (self.steady_heap_end) |e| {
                return aminfer.vmstat.heapDelta(s, e);
            }
        }
        return std.mem.zeroes(aminfer.vmstat.HeapDelta);
    }

    fn deinit(self: *ServeRunner, allocator: std.mem.Allocator) void {
        if (self.vocab_index) |*vi| vi.deinit(allocator);
        allocator.destroy(self.penalty_tracker);
        allocator.free(self.grammar_mask);
        allocator.free(self.sampler_cand_ids);
        allocator.free(self.p_res);
        allocator.free(self.prompt_ids_buf);
        self.encode_scratch.deinit(allocator);
    }

    inline fn isStopToken(self: *const ServeRunner, tid: u32) bool {
        for (self.stop_tokens[0..self.stop_token_count]) |st| {
            if (tid == st) return true;
        }
        return false;
    }

    fn emitToken(
        self: *const ServeRunner,
        stream: *aminfer.server.TokenStream,
        tid: u32,
        ignore_eos: bool,
    ) EmitResult {
        if (!ignore_eos and self.isStopToken(tid)) return .stop;
        if (self.tok.isSpecialToken(tid)) return .ok;
        const piece = self.tok.tokenBytes(tid) catch return .ok;
        if (!stream.emitTokenBytes(piece)) return .cancelled;
        return .ok;
    }

    fn emitTokenWithStop(
        self: *const ServeRunner,
        stream: *aminfer.server.TokenStream,
        tid: u32,
        ignore_eos: bool,
        stop_cfg: *const aminfer.sampling.StopConfig,
        stop_hb: *aminfer.sampling.StopHoldback,
    ) EmitResult {
        if (!ignore_eos and self.isStopToken(tid)) return .stop;
        if (self.tok.isSpecialToken(tid)) return .ok;
        const piece = self.tok.tokenBytes(tid) catch return .ok;
        if (!stream.emitTokenBytes(piece)) return .cancelled;
        if (stop_cfg.count > 0) {
            const Noop = struct {
                fn onSafe(_: void, _: []const u8) error{}!void {}
            };
            const cont = stop_hb.feed(stop_cfg, piece, {}, Noop.onSafe) catch true;
            if (!cont) return .stop;
        }
        return .ok;
    }

    fn sampleConstrainedStep(
        self: *ServeRunner,
        job: *const aminfer.server.InferenceJob,
        gstate: *aminfer.grammar.GrammarState,
    ) ?u32 {
        const raw_logits = switch (self.target) {
            .qwen3 => |m| m.readLastLogitsF16(),
            .graph => |g| g.readLastLogitsF16(),
        };
        const n = @min(raw_logits.len, self.p_res.len);
        if (n == 0) return null;
        var params = job.sampling;
        if (job.grammar_program) |gp| {
            const vi = &(self.vocab_index orelse return null);
            const mask_len = (@min(n, vi.vocab_size) + 31) / 32;
            vi.fillAllowedTokenBitmask(gp, gstate, self.grammar_mask[0..mask_len]);
            params.allowed_token_mask = self.grammar_mask[0..mask_len];
        }
        const out = aminfer.sampling.sampleToken(
            f16,
            raw_logits[0..n],
            params,
            self.penalty_tracker.activeEntries(),
            job.rng,
            .{
                .logits_f32 = self.p_res[0..n],
                .cand_ids = self.sampler_cand_ids[0..n],
            },
        ) catch return null;
        if (job.grammar_program) |gp| {
            if (self.vocab_index) |*vi| {
                _ = vi.advanceToken(gp, gstate, self.tok, out.token_id);
            }
        }
        if (params.hasPenalties()) {
            self.penalty_tracker.recordToken(out.token_id);
        }
        return out.token_id;
    }

    fn handler(self: *ServeRunner) aminfer.server.EngineHandler {
        return .{
            .ctx = self,
            .generateFn = generateCallback,
        };
    }

    fn generateCallback(
        ctx: *anyopaque,
        job: *const aminfer.server.InferenceJob,
        stream: *aminfer.server.TokenStream,
    ) void {
        const self: *ServeRunner = @ptrCast(@alignCast(ctx));
        self.generate(job, stream);
    }

    fn generate(
        self: *ServeRunner,
        job: *const aminfer.server.InferenceJob,
        stream: *aminfer.server.TokenStream,
    ) void {
        if (self.model_mutex) |mtx| _ = std.c.pthread_mutex_lock(mtx);
        defer if (self.model_mutex) |mtx| {
            _ = std.c.pthread_mutex_unlock(mtx);
        };
        defer if (self.server_ptr) |s| {
            if (self.target == .qwen3) {
                s.setRetiredQ8Chunks(self.target.qwen3.retiredKvChunks());
            }
        };

        // W19 per-request LoRA selection (`active_adapters == 0` or `lora_adapter == null`
        // keeps `active_lora_slot = NO_ADAPTER` for the zero-overhead outside-layer-loop bypass).
        if (self.target == .qwen3) {
            const m = self.target.qwen3;
            if (job.lora_adapter) |adp| {
                const reg = m.lora_registry orelse {
                    stream.fail("requested LoRA adapter not loaded");
                    return;
                };
                const slot_id = reg.findSlotByName(adp) orelse {
                    stream.fail("requested LoRA adapter not found");
                    return;
                };
                m.setActiveLoraSlot(slot_id);
            } else {
                m.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);
            }
        }
        defer if (self.target == .qwen3) {
            self.target.qwen3.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);
        };

        var prompt_ids = self.tok.encodeWithScratch(
            &self.encode_scratch,
            job.prompt,
            self.prompt_ids_buf,
        ) catch {
            stream.fail("prompt exceeds max_seq_len or tokenizer scratch");
            return;
        };
        if (prompt_ids.len == 0) {
            self.prompt_ids_buf[0] = self.bos_token_id orelse 0;
            prompt_ids = self.prompt_ids_buf[0..1];
        }
        if (prompt_ids.len + 1 >= self.max_seq_len) {
            stream.fail("prompt exceeds max_seq_len");
            return;
        }
        const prompt_len_u32: u32 = @intCast(prompt_ids.len);
        const room: u32 = self.max_seq_len - prompt_len_u32 - 1;
        const max_gen: u32 = @min(job.max_tokens, room);
        if (max_gen == 0) {
            if (!stream.setPromptTokens(prompt_len_u32)) return;
            stream.finish(.length, prompt_len_u32);
            return;
        }

        const use_constrained = (job.grammar_program != null) or
            job.sampling.hasPenalties() or
            job.sampling.logit_bias.len > 0 or
            job.sampling.top_p < 1.0 or
            job.sampling.top_k > 0 or
            job.sampling.min_p > 0.0;

        if (!use_constrained and self.spec_engine != null) {
            const eng = self.spec_engine.?;
            eng.reset();
            eng.setActiveNgramDomain(job.ngram_domain);
            eng.config.temperature = job.temperature;
            const rng_opt: ?std.Random = if (job.temperature > 0.0) job.rng.random() else null;
            const first_tok = eng.prefill(prompt_ids, rng_opt) catch {
                stream.fail("speculative prefill failed");
                return;
            };
            if (!stream.setPromptTokens(prompt_len_u32)) return;

            var stop_hb: aminfer.sampling.StopHoldback = .{};
            switch (self.emitTokenWithStop(stream, first_tok, job.ignore_eos, job.stop_config, &stop_hb)) {
                .stop => {
                    stream.finish(.stop, prompt_len_u32);
                    return;
                },
                .cancelled => return,
                .ok => {},
            }
            var emitted: u32 = 1;
            while (emitted < max_gen and !stream.isCancelled()) {
                const rem: usize = max_gen - emitted;
                const cap = @min(rem, self.round_buf.len);
                const rs = eng.stepRound(rng_opt, self.round_buf[0..cap]) catch {
                    stream.fail("speculative decode round failed");
                    return;
                };
                self.recordSteadyStep();
                for (self.round_buf[0..rs.emitted_count]) |tid| {
                    switch (self.emitTokenWithStop(stream, tid, job.ignore_eos, job.stop_config, &stop_hb)) {
                        .stop => {
                            stream.finish(.stop, prompt_len_u32);
                            return;
                        },
                        .cancelled => return,
                        .ok => {},
                    }
                    emitted += 1;
                    if (emitted >= max_gen) break;
                }
            }
            if (!stream.isCancelled()) {
                stream.finish(.length, prompt_len_u32);
            }
            return;
        }

        if (self.ngram_registry) |reg| {
            queryNgramPromptTail(reg, self.tok, job.ngram_domain, prompt_ids, &self.ngram_query_errors);
        }

        // Strict zero-overhead unconstrained fast path when `use_constrained == false` and `stop_config.count == 0`.
        if (!use_constrained and job.stop_config.count == 0) {
            var next_tok: u32 = switch (self.target) {
                .qwen3 => |m| blk: {
                    m.seq_len = 0;
                    const t = m.prefillPrompt(prompt_ids, true) catch {
                        stream.fail("prefill failed");
                        return;
                    };
                    break :blk if (job.temperature > 0.0)
                        m.sampleFromLastLogits(job.temperature, job.rng.random(), self.p_res) catch {
                            stream.fail("sampling failed");
                            return;
                        }
                    else
                        t;
                },
                .graph => |g| blk: {
                    g.resetKvCache();
                    const t = g.prefillChunk(prompt_ids, 0, true) catch {
                        stream.fail("graph prefill failed");
                        return;
                    };
                    break :blk if (job.temperature > 0.0)
                        g.sampleFromLastLogits(job.temperature, job.rng.random(), self.p_res) catch {
                            stream.fail("sampling failed");
                            return;
                        }
                    else
                        t;
                },
            };
            if (!stream.setPromptTokens(prompt_len_u32)) return;

            switch (self.emitToken(stream, next_tok, job.ignore_eos)) {
                .stop => {
                    stream.finish(.stop, prompt_len_u32);
                    return;
                },
                .cancelled => return,
                .ok => {},
            }
            var pos: u32 = prompt_len_u32;
            var emitted: u32 = 1;
            while (emitted < max_gen and !stream.isCancelled()) {
                next_tok = switch (self.target) {
                    .qwen3 => |m| blk: {
                        const t = m.stepDecode(next_tok, pos) catch {
                            stream.fail("decode step failed");
                            return;
                        };
                        break :blk if (job.temperature > 0.0)
                            m.sampleFromLastLogits(job.temperature, job.rng.random(), self.p_res) catch {
                                stream.fail("sampling failed");
                                return;
                            }
                        else
                            t;
                    },
                    .graph => |g| blk: {
                        const t = g.stepDecode(next_tok, pos) catch {
                            stream.fail("graph decode step failed");
                            return;
                        };
                        break :blk if (job.temperature > 0.0)
                            g.sampleFromLastLogits(job.temperature, job.rng.random(), self.p_res) catch {
                                stream.fail("sampling failed");
                                return;
                            }
                        else
                            t;
                    },
                };
                self.recordSteadyStep();
                pos += 1;
                switch (self.emitToken(stream, next_tok, job.ignore_eos)) {
                    .stop => {
                        stream.finish(.stop, prompt_len_u32);
                        return;
                    },
                    .cancelled => return,
                    .ok => {},
                }
                emitted += 1;
            }
            if (!stream.isCancelled()) {
                stream.finish(.length, prompt_len_u32);
            }
            return;
        }

        // Constrained (`C22` grammar bitmask / `C18` sampling penalties / `StopHoldback`) decode path.
        var gstate: aminfer.grammar.GrammarState = if (job.grammar_program) |gp| gp.initialState() else undefined;
        self.penalty_tracker.reset();
        if (job.sampling.hasPenalties()) {
            self.penalty_tracker.recordSlice(prompt_ids);
        }
        var stop_hb: aminfer.sampling.StopHoldback = .{};

        switch (self.target) {
            .qwen3 => |m| {
                m.seq_len = 0;
                _ = m.prefillPrompt(prompt_ids, true) catch {
                    stream.fail("prefill failed");
                    return;
                };
            },
            .graph => |g| {
                g.resetKvCache();
                _ = g.prefillChunk(prompt_ids, 0, true) catch {
                    stream.fail("graph prefill failed");
                    return;
                };
            },
        }
        var next_tok = self.sampleConstrainedStep(job, &gstate) orelse {
            stream.fail("constrained sampling failed");
            return;
        };
        if (!stream.setPromptTokens(prompt_len_u32)) return;

        switch (self.emitTokenWithStop(stream, next_tok, job.ignore_eos, job.stop_config, &stop_hb)) {
            .stop => {
                stream.finish(.stop, prompt_len_u32);
                return;
            },
            .cancelled => return,
            .ok => {},
        }
        var pos: u32 = prompt_len_u32;
        var emitted: u32 = 1;
        while (emitted < max_gen and !stream.isCancelled()) {
            switch (self.target) {
                .qwen3 => |m| {
                    _ = m.stepDecode(next_tok, pos) catch {
                        stream.fail("decode step failed");
                        return;
                    };
                },
                .graph => |g| {
                    _ = g.stepDecode(next_tok, pos) catch {
                        stream.fail("graph decode step failed");
                        return;
                    };
                },
            }
            next_tok = self.sampleConstrainedStep(job, &gstate) orelse {
                stream.fail("constrained sampling failed");
                return;
            };
            self.recordSteadyStep();
            pos += 1;
            switch (self.emitTokenWithStop(stream, next_tok, job.ignore_eos, job.stop_config, &stop_hb)) {
                .stop => {
                    stream.finish(.stop, prompt_len_u32);
                    return;
                },
                .cancelled => return,
                .ok => {},
            }
            emitted += 1;
        }
        if (!stream.isCancelled()) {
            stream.finish(.length, prompt_len_u32);
        }
    }
};

/// Zero-allocation `C14` continuous-batching runner bridging `Qwen3BatchRuntime` or
/// `GraphBatchRuntime` + `Tokenizer` + `DomainNgramRegistry` into `Server.runBatchedWorkerLoop`.
const BatchServeRunner = struct {
    const PendingJobMeta = struct {
        lora_slot: u32 = aminfer.lora.NO_ADAPTER,
        grammar_program: ?*const aminfer.grammar.GrammarProgram = null,
        sampling: aminfer.sampling.SamplingParams = .{},
        rng: ?*aminfer.rng.RequestRng = null,
        constrained: bool = false,
    };

    const SlotConstraint = struct {
        active: bool = false,
        grammar_program: ?*const aminfer.grammar.GrammarProgram = null,
        gstate: aminfer.grammar.GrammarState = undefined,
        sampling: aminfer.sampling.SamplingParams = .{},
        rng: ?*aminfer.rng.RequestRng = null,
    };

    inner_backend: aminfer.batch_scheduler.BatchBackend,
    qwen3_batch_rt: ?*aminfer.qwen3.Qwen3BatchRuntime = null,
    model_mutex: ?*std.c.pthread_mutex_t = null,
    server_ptr: ?*aminfer.server.Server = null,
    ngram_registry: ?*aminfer.ngram_nvme.DomainNgramRegistry,
    tok: *const aminfer.tokenizer.Tokenizer,
    bos_token_id: ?u32,
    max_seq_len: u32,
    stop_tokens: [aminfer.batch_scheduler.MAX_STOP_TOKENS]u32,
    stop_token_count: u8,
    encode_scratch: aminfer.tokenizer.EncodeScratch,
    boundary_verify_buf: []u32,
    p_res: []f32,
    sampler_cand_ids: []u32,
    grammar_mask: []u32,
    vocab_index: ?aminfer.grammar.VocabIndex,
    pending_fifo: [aminfer.batch_scheduler.MAX_BATCH_SLOTS * 2]PendingJobMeta = @splat(.{}),
    pending_head: usize = 0,
    pending_tail: usize = 0,
    slot_constraints: [aminfer.batch_scheduler.MAX_BATCH_SLOTS]SlotConstraint = @splat(.{}),
    boundary_fallback_count: u64 = 0,
    steady_steps: u64 = 0,
    steady_heap_start: ?aminfer.vmstat.Snapshot = null,
    steady_heap_end: ?aminfer.vmstat.Snapshot = null,
    vmstat_errors: u64 = 0,
    ngram_query_errors: u64 = 0,

    fn init(
        allocator: std.mem.Allocator,
        inner_backend: aminfer.batch_scheduler.BatchBackend,
        ngram_registry: ?*aminfer.ngram_nvme.DomainNgramRegistry,
        tok: *const aminfer.tokenizer.Tokenizer,
        bos_token_id: ?u32,
        eos_token_id: ?u32,
        max_seq_len: u32,
    ) !BatchServeRunner {
        var encode_scratch = try aminfer.tokenizer.EncodeScratch.init(
            allocator,
            @max(@as(usize, 8192), @as(usize, max_seq_len) * 2),
        );
        errdefer encode_scratch.deinit(allocator);
        const boundary_verify_buf = try allocator.alloc(u32, @max(@as(usize, 1), @as(usize, max_seq_len)));
        errdefer allocator.free(boundary_verify_buf);
        const v_cap = @max(@as(usize, 1), tok.vocabSize());
        const p_res = try allocator.alloc(f32, v_cap);
        errdefer allocator.free(p_res);
        const sampler_cand_ids = try allocator.alloc(u32, v_cap);
        errdefer allocator.free(sampler_cand_ids);
        const mask_words = @max(@as(usize, 1), (v_cap + 31) / 32);
        const grammar_mask = try allocator.alloc(u32, mask_words);
        errdefer allocator.free(grammar_mask);

        var stop_tokens: [aminfer.batch_scheduler.MAX_STOP_TOKENS]u32 = @splat(0);
        var stop_cnt: u8 = 0;
        const candidates = [_]?u32{
            eos_token_id,
            tok.addedTokenId("<|im_end|>"),
            tok.addedTokenId("<|endoftext|>"),
        };
        for (candidates) |maybe_id| {
            const id = maybe_id orelse continue;
            var dup = false;
            for (stop_tokens[0..stop_cnt]) |existing| {
                if (existing == id) {
                    dup = true;
                    break;
                }
            }
            if (!dup and stop_cnt < stop_tokens.len) {
                stop_tokens[stop_cnt] = id;
                stop_cnt += 1;
            }
        }

        const vocab_index: ?aminfer.grammar.VocabIndex = if (aminfer.grammar.VocabIndex.initFromTokenizer(
            allocator,
            tok,
            stop_tokens[0..stop_cnt],
        )) |vi| vi else |_| null;

        return .{
            .inner_backend = inner_backend,
            .ngram_registry = ngram_registry,
            .tok = tok,
            .bos_token_id = bos_token_id,
            .max_seq_len = max_seq_len,
            .stop_tokens = stop_tokens,
            .stop_token_count = stop_cnt,
            .encode_scratch = encode_scratch,
            .boundary_verify_buf = boundary_verify_buf,
            .p_res = p_res,
            .sampler_cand_ids = sampler_cand_ids,
            .grammar_mask = grammar_mask,
            .vocab_index = vocab_index,
        };
    }

    fn deinit(self: *BatchServeRunner, allocator: std.mem.Allocator) void {
        if (self.vocab_index) |*vi| vi.deinit(allocator);
        allocator.free(self.grammar_mask);
        allocator.free(self.sampler_cand_ids);
        allocator.free(self.p_res);
        allocator.free(self.boundary_verify_buf);
        self.encode_scratch.deinit(allocator);
    }

    fn bindNextPendingToSlot(self: *BatchServeRunner, slot_id: u16) void {
        if (self.pending_head == self.pending_tail) return;
        const meta = self.pending_fifo[self.pending_head];
        self.pending_head = (self.pending_head + 1) % self.pending_fifo.len;
        if (self.qwen3_batch_rt) |brt| {
            brt.setSlotLora(slot_id, meta.lora_slot);
        }
        if (slot_id < self.slot_constraints.len) {
            self.slot_constraints[slot_id] = .{
                .active = meta.constrained,
                .grammar_program = meta.grammar_program,
                .gstate = if (meta.grammar_program) |gp| gp.initialState() else undefined,
                .sampling = meta.sampling,
                .rng = meta.rng,
            };
        }
    }

    fn sampleSlotConstrained(self: *BatchServeRunner, slot_id: u16, raw_logits: []const f16) ?u32 {
        if (slot_id >= self.slot_constraints.len) return null;
        var sc = &self.slot_constraints[slot_id];
        if (!sc.active) return null;
        const n = @min(raw_logits.len, self.p_res.len);
        if (n == 0) return null;
        var params = sc.sampling;
        if (sc.grammar_program) |gp| {
            const vi = &(self.vocab_index orelse return null);
            const mask_len = (@min(n, vi.vocab_size) + 31) / 32;
            vi.fillAllowedTokenBitmask(gp, &sc.gstate, self.grammar_mask[0..mask_len]);
            params.allowed_token_mask = self.grammar_mask[0..mask_len];
        }
        const rng_ptr = sc.rng orelse return null;
        const out = aminfer.sampling.sampleToken(
            f16,
            raw_logits[0..n],
            params,
            &.{},
            rng_ptr,
            .{
                .logits_f32 = self.p_res[0..n],
                .cand_ids = self.sampler_cand_ids[0..n],
            },
        ) catch return null;
        if (sc.grammar_program) |gp| {
            if (self.vocab_index) |*vi| {
                _ = vi.advanceToken(gp, &sc.gstate, self.tok, out.token_id);
            }
        }
        return out.token_id;
    }

    fn steadyHeapDelta(self: *const BatchServeRunner) aminfer.vmstat.HeapDelta {
        if (self.steady_heap_start) |s| {
            if (self.steady_heap_end) |e| {
                return aminfer.vmstat.heapDelta(s, e);
            }
        }
        return std.mem.zeroes(aminfer.vmstat.HeapDelta);
    }

    fn handler(self: *BatchServeRunner) aminfer.server.BatchEngineHandler {
        return .{
            .ctx = self,
            .tokenizeFn = tokenizeCallback,
            .tokenizeWithBoundaryFn = tokenizeWithBoundaryCallback,
            .detokenizeFn = detokenizeCallback,
            .backend = .{
                .ctx = self,
                .lookupPrefixFn = if (self.inner_backend.lookupPrefixFn != null) lookupPrefixCallback else null,
                .prefillSlotFn = prefillSlotCallback,
                .stepDecodeBatchFn = stepDecodeBatchCallback,
                .releaseSlotFn = releaseSlotCallback,
            },
            .eos_token_ids = self.stop_tokens,
            .eos_token_count = self.stop_token_count,
        };
    }

    fn tokenizeCallback(
        ctx: *anyopaque,
        job: *const aminfer.server.InferenceJob,
        out_tokens: []u32,
    ) aminfer.batch_scheduler.BackendError!usize {
        const res = try tokenizeWithBoundaryCallback(ctx, job, out_tokens);
        return res.prompt_len;
    }

    /// Zero-allocation multi-turn ChatML boundary-aware tokenizer (`C15`, ADR-0020):
    /// tokenizes the full prompt into `out_tokens`, then verifies that splitting at
    /// `job.last_user_msg_byte_offset` produces the exact same token sequence before
    /// returning `message_boundary_token`. Any cross-boundary BPE merge falls back
    /// gracefully to `message_boundary_token = null`.
    fn tokenizeWithBoundaryCallback(
        ctx: *anyopaque,
        job: *const aminfer.server.InferenceJob,
        out_tokens: []u32,
    ) aminfer.batch_scheduler.BackendError!aminfer.server.TokenizeResult {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        var lora_slot: u32 = aminfer.lora.NO_ADAPTER;
        if (job.lora_adapter) |adp| {
            const brt = self.qwen3_batch_rt orelse return error.TokenizeFailed;
            const reg = brt.lora_registry orelse return error.TokenizeFailed;
            lora_slot = reg.findSlotByName(adp) orelse return error.TokenizeFailed;
        }
        var prompt_ids = self.tok.encodeWithScratch(
            &self.encode_scratch,
            job.prompt,
            out_tokens,
        ) catch return error.TokenizeFailed;
        if (prompt_ids.len == 0) {
            if (out_tokens.len == 0) return error.TokenizeFailed;
            out_tokens[0] = self.bos_token_id orelse 0;
            prompt_ids = out_tokens[0..1];
        }
        if (prompt_ids.len + 1 >= self.max_seq_len) return error.TokenizeFailed;
        if (self.ngram_registry) |reg| {
            queryNgramPromptTail(reg, self.tok, job.ngram_domain, prompt_ids, &self.ngram_query_errors);
        }

        const constrained = (job.grammar_program != null) or
            job.sampling.logit_bias.len > 0 or
            job.sampling.top_p < 1.0 or
            job.sampling.top_k > 0 or
            job.sampling.min_p > 0.0;
        const next_tail = (self.pending_tail + 1) % self.pending_fifo.len;
        if (next_tail != self.pending_head) {
            self.pending_fifo[self.pending_tail] = .{
                .lora_slot = lora_slot,
                .grammar_program = job.grammar_program,
                .sampling = job.sampling,
                .rng = job.rng,
                .constrained = constrained,
            };
            self.pending_tail = next_tail;
        }

        var boundary_tok: ?u32 = null;
        if (job.last_user_msg_byte_offset) |b_off| {
            if (b_off > 0 and b_off < job.prompt.len and prompt_ids.len <= self.boundary_verify_buf.len) {
                if (self.tok.encodeWithScratch(&self.encode_scratch, job.prompt[0..b_off], self.boundary_verify_buf)) |prefix_ids| {
                    if (prefix_ids.len > 0 and prefix_ids.len < prompt_ids.len) {
                        if (self.tok.encodeWithScratch(
                            &self.encode_scratch,
                            job.prompt[b_off..],
                            self.boundary_verify_buf[prefix_ids.len..],
                        )) |suffix_ids| {
                            if (prefix_ids.len + suffix_ids.len == prompt_ids.len and
                                std.mem.eql(u32, self.boundary_verify_buf[0..prompt_ids.len], prompt_ids))
                            {
                                boundary_tok = @intCast(prefix_ids.len);
                            } else {
                                self.boundary_fallback_count += 1;
                            }
                        } else |_| {
                            self.boundary_fallback_count += 1;
                        }
                    }
                } else |_| {
                    self.boundary_fallback_count += 1;
                }
            }
        }
        return .{
            .prompt_len = prompt_ids.len,
            .message_boundary_token = boundary_tok,
        };
    }

    fn detokenizeCallback(
        ctx: *anyopaque,
        token_id: u32,
        _: []u8,
    ) []const u8 {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        if (self.tok.isSpecialToken(token_id)) return &.{};
        return self.tok.tokenBytes(token_id) catch &.{};
    }

    fn lookupPrefixCallback(
        ctx: *anyopaque,
        slot_id: u16,
        prompt_tokens: []const u32,
    ) aminfer.batch_scheduler.PrefixLookupResult {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        if (self.model_mutex) |mtx| _ = std.c.pthread_mutex_lock(mtx);
        defer if (self.model_mutex) |mtx| {
            _ = std.c.pthread_mutex_unlock(mtx);
        };
        self.bindNextPendingToSlot(slot_id);
        return self.inner_backend.lookupPrefix(slot_id, prompt_tokens);
    }

    fn prefillSlotCallback(
        ctx: *anyopaque,
        chunk: *const aminfer.batch_scheduler.PrefillChunkView,
    ) aminfer.batch_scheduler.BackendError!aminfer.batch_scheduler.PrefillResult {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        if (self.model_mutex) |mtx| _ = std.c.pthread_mutex_lock(mtx);
        defer if (self.model_mutex) |mtx| {
            _ = std.c.pthread_mutex_unlock(mtx);
        };
        if (self.inner_backend.lookupPrefixFn == null and chunk.start_pos == 0) {
            self.bindNextPendingToSlot(chunk.slot_id);
        }
        var res = try self.inner_backend.prefillSlot(chunk);
        if (chunk.is_final_chunk and chunk.slot_id < self.slot_constraints.len and self.slot_constraints[chunk.slot_id].active) {
            if (self.qwen3_batch_rt) |brt| {
                if (self.sampleSlotConstrained(chunk.slot_id, brt.model.readLastLogitsF16())) |ctok| {
                    res.first_token = ctok;
                }
            }
        }
        return res;
    }

    fn stepDecodeBatchCallback(
        ctx: *anyopaque,
        view: *const aminfer.batch_scheduler.BatchStepView,
        out_next_tokens: []u32,
    ) aminfer.batch_scheduler.BackendError!void {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        if (self.model_mutex) |mtx| _ = std.c.pthread_mutex_lock(mtx);
        defer if (self.model_mutex) |mtx| {
            _ = std.c.pthread_mutex_unlock(mtx);
        };
        try self.inner_backend.stepDecodeBatch(view, out_next_tokens);
        if (self.qwen3_batch_rt) |brt| {
            const v_size: usize = brt.model.config.vocab_size;
            for (0..view.batchSize()) |bi| {
                const sid = view.slot_ids[bi];
                if (sid < self.slot_constraints.len and self.slot_constraints[sid].active) {
                    const row = brt.model.logits_host[bi * v_size .. (bi + 1) * v_size];
                    if (self.sampleSlotConstrained(sid, row)) |ctok| {
                        out_next_tokens[bi] = ctok;
                    }
                }
            }
            if (self.server_ptr) |s| {
                s.setRetiredQ8Chunks(brt.stats().kv_retire_events);
            }
        }
        self.steady_steps += 1;
        if (self.steady_steps == 2) {
            if (aminfer.vmstat.Snapshot.capture()) |snap| {
                self.steady_heap_start = snap;
            } else |_| {
                self.vmstat_errors += 1;
            }
        } else if (self.steady_steps > 2) {
            if (aminfer.vmstat.Snapshot.capture()) |snap| {
                self.steady_heap_end = snap;
            } else |_| {
                self.vmstat_errors += 1;
            }
        }
    }

    fn releaseSlotCallback(
        ctx: *anyopaque,
        slot_id: u16,
        prefix_handle: u64,
    ) void {
        const self: *BatchServeRunner = @ptrCast(@alignCast(ctx));
        if (self.model_mutex) |mtx| _ = std.c.pthread_mutex_lock(mtx);
        defer if (self.model_mutex) |mtx| {
            _ = std.c.pthread_mutex_unlock(mtx);
        };
        if (self.qwen3_batch_rt) |brt| {
            brt.setSlotLora(slot_id, aminfer.lora.NO_ADAPTER);
        }
        if (slot_id < self.slot_constraints.len) {
            self.slot_constraints[slot_id] = .{};
        }
        self.inner_backend.releaseSlot(slot_id, prefix_handle);
    }
};

/// Preallocated zero-allocation `C21` `POST /v1/embeddings` and `POST /v1/rerank`
/// backend bound to `Qwen3Model` + `Tokenizer` (guarded by `model_mutex`).
const ServeEmbedRerankCtx = struct {
    model: *aminfer.qwen3.Qwen3Model,
    tok: *const aminfer.tokenizer.Tokenizer,
    model_mutex: *std.c.pthread_mutex_t,
    max_tokens_cap: usize,
    bos_token_id: u32,
    yes_token_id: u32,
    no_token_id: u32,
    encode_scratch: aminfer.tokenizer.EncodeScratch,
    token_ids_buf: []u32,
    rerank_prompt_buf: []u8,

    fn init(
        allocator: std.mem.Allocator,
        model: *aminfer.qwen3.Qwen3Model,
        tok: *const aminfer.tokenizer.Tokenizer,
        model_mutex: *std.c.pthread_mutex_t,
        max_tokens_cap: usize,
    ) !ServeEmbedRerankCtx {
        const cap = @max(@as(usize, 64), max_tokens_cap);
        var encode_scratch = try aminfer.tokenizer.EncodeScratch.init(allocator, @max(@as(usize, 8192), cap * 2));
        errdefer encode_scratch.deinit(allocator);
        const token_ids_buf = try allocator.alloc(u32, cap);
        errdefer allocator.free(token_ids_buf);
        const rerank_prompt_buf = try allocator.alloc(u8, 64 * 1024);
        errdefer allocator.free(rerank_prompt_buf);

        var yes_buf: [8]u32 = undefined;
        var no_buf: [8]u32 = undefined;
        const yes_id: u32 = tok.addedTokenId("yes") orelse tok.addedTokenId("Yes") orelse blk: {
            if (tok.encodeWithScratch(&encode_scratch, "yes", &yes_buf)) |s| {
                if (s.len > 0) break :blk s[0];
            } else |_| {}
            break :blk 1;
        };
        var no_id: u32 = tok.addedTokenId("no") orelse tok.addedTokenId("No") orelse blk: {
            if (tok.encodeWithScratch(&encode_scratch, "no", &no_buf)) |s| {
                if (s.len > 0) break :blk s[0];
            } else |_| {}
            break :blk 0;
        };
        if (no_id == yes_id) no_id = if (yes_id > 0) yes_id - 1 else 1;

        return .{
            .model = model,
            .tok = tok,
            .model_mutex = model_mutex,
            .max_tokens_cap = cap,
            .bos_token_id = model.config.bos_token_id orelse 0,
            .yes_token_id = yes_id,
            .no_token_id = no_id,
            .encode_scratch = encode_scratch,
            .token_ids_buf = token_ids_buf,
            .rerank_prompt_buf = rerank_prompt_buf,
        };
    }

    fn deinit(self: *ServeEmbedRerankCtx, allocator: std.mem.Allocator) void {
        allocator.free(self.rerank_prompt_buf);
        allocator.free(self.token_ids_buf);
        self.encode_scratch.deinit(allocator);
    }

    fn embeddingHandler(self: *ServeEmbedRerankCtx) aminfer.server.EmbeddingHandler {
        return .{
            .ctx = self,
            .embedFn = embedCallback,
        };
    }

    fn rerankHandler(self: *ServeEmbedRerankCtx) aminfer.server.RerankHandler {
        return .{
            .ctx = self,
            .rerankFn = rerankCallback,
        };
    }

    fn embedCallback(
        ctx: *anyopaque,
        req: *const aminfer.server.EmbedRequest,
        out_vectors: []f32,
    ) anyerror!aminfer.server.EmbedResult {
        const self: *ServeEmbedRerankCtx = @ptrCast(@alignCast(ctx));
        _ = std.c.pthread_mutex_lock(self.model_mutex);
        defer _ = std.c.pthread_mutex_unlock(self.model_mutex);

        if (req.model_tag.adapter) |adp| {
            const reg = self.model.lora_registry orelse return error.AdapterNotFound;
            const slot_id = reg.findSlotByName(adp) orelse return error.AdapterNotFound;
            self.model.setActiveLoraSlot(slot_id);
        } else {
            self.model.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);
        }
        defer self.model.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);

        const eff_dim: usize = try aminfer.embed_rerank.effectiveDimensions(
            self.model.config.hidden_size,
            req.dimensions,
        );
        if (req.input_count * eff_dim > out_vectors.len) return error.BufferTooSmall;

        var total_tokens: u32 = 0;
        for (req.inputs[0..req.input_count], 0..) |text, idx| {
            var ids = self.tok.encodeWithScratch(&self.encode_scratch, text, self.token_ids_buf) catch blk: {
                const sub = text[0..@min(text.len, self.token_ids_buf.len)];
                break :blk try self.tok.encodeWithScratch(&self.encode_scratch, sub, self.token_ids_buf);
            };
            if (ids.len == 0) {
                self.token_ids_buf[0] = self.bos_token_id;
                ids = self.token_ids_buf[0..1];
            }
            const dst = out_vectors[idx * eff_dim .. (idx + 1) * eff_dim];
            _ = try self.model.prefillEmbeddings(ids, .{
                .pooling = .last_token,
                .dimensions = req.dimensions,
                .normalize = true,
            }, dst);
            total_tokens +|= @intCast(ids.len);
        }
        return .{
            .dim = eff_dim,
            .prompt_tokens = total_tokens,
        };
    }

    fn rerankCallback(
        ctx: *anyopaque,
        req: *const aminfer.server.RerankRequest,
        out_items: []aminfer.server.RerankItem,
    ) anyerror!aminfer.server.RerankResult {
        const self: *ServeEmbedRerankCtx = @ptrCast(@alignCast(ctx));
        _ = std.c.pthread_mutex_lock(self.model_mutex);
        defer _ = std.c.pthread_mutex_unlock(self.model_mutex);

        if (req.model_tag.adapter) |adp| {
            const reg = self.model.lora_registry orelse return error.AdapterNotFound;
            const slot_id = reg.findSlotByName(adp) orelse return error.AdapterNotFound;
            self.model.setActiveLoraSlot(slot_id);
        } else {
            self.model.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);
        }
        defer self.model.setActiveLoraSlot(aminfer.lora.NO_ADAPTER);

        if (req.document_count > out_items.len or req.document_count > aminfer.server.MAX_RERANK_DOCS) {
            return error.BufferTooSmall;
        }
        var scores_buf: [aminfer.server.MAX_RERANK_DOCS]aminfer.embed_rerank.RerankScore = undefined;
        var total_tokens: u32 = 0;

        for (req.documents[0..req.document_count], 0..) |doc, idx| {
            const formatted = std.fmt.bufPrint(
                self.rerank_prompt_buf,
                "<|im_start|>user\nQuery: {s}\nDocument: {s}\nRelevant (yes/no):<|im_end|>\n<|im_start|>assistant\n",
                .{ req.query, doc },
            ) catch doc;
            var ids = self.tok.encodeWithScratch(&self.encode_scratch, formatted, self.token_ids_buf) catch blk: {
                const sub = formatted[0..@min(formatted.len, self.token_ids_buf.len)];
                break :blk try self.tok.encodeWithScratch(&self.encode_scratch, sub, self.token_ids_buf);
            };
            if (ids.len == 0) {
                self.token_ids_buf[0] = self.bos_token_id;
                ids = self.token_ids_buf[0..1];
            }
            scores_buf[idx] = try self.model.scoreRerankSequence(ids, @intCast(idx), .{
                .mode = .yes_no_sigmoid,
                .positive_token_id = @min(self.yes_token_id, self.model.config.vocab_size - 1),
                .negative_token_id = @min(self.no_token_id, self.model.config.vocab_size - 1),
                .temperature = 1.0,
            });
            total_tokens +|= @intCast(ids.len);
        }

        const filled = scores_buf[0..req.document_count];
        aminfer.embed_rerank.sortRerankScoresDesc(filled);
        const take: usize = if (req.top_n) |k| @min(@as(usize, k), filled.len) else filled.len;
        for (filled[0..take], 0..) |sc, i| {
            out_items[i] = .{
                .index = sc.index,
                .relevance_score = sc.score,
            };
        }
        return .{
            .count = take,
            .prompt_tokens = total_tokens,
        };
    }
};

fn runServeGraph(
    allocator: std.mem.Allocator,
    cli: ServeCliArgs,
    host_ipv4: [4]u8,
    cfg_bytes: []const u8,
    tok_bytes: []const u8,
    out: *Io.Writer,
) !void {
    var gcfg = try aminfer.graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    const use_spec = cli.draft_mode != null or cli.draft_amf_path != null or cli.ngram_nvme_path != null;
    const use_continuous: bool = switch (cli.batch_mode) {
        .serial => false,
        .continuous => blk: {
            if (use_spec) return error.InvalidFlagCombination;
            break :blk true;
        },
        .auto => (!use_spec and cli.max_active_requests > 1),
    };
    const max_seq_len: u32 = @min(gcfg.max_position_embeddings, @max(@as(u32, 512), cli.max_seq_len));
    var gcfg_kv_layers: u32 = 0;
    for (gcfg.axes.layer_attention_kinds) |ak| {
        if (ak != .gated_deltanet) gcfg_kv_layers += 1;
    }
    const raw_batch_slots: usize = if (use_continuous)
        @min(@max(cli.max_active_requests, 1), aminfer.batch_scheduler.MAX_BATCH_SLOTS)
    else
        1;
    const batch_slots: usize = if (use_continuous)
        autoClampServeBatchSlots(
            raw_batch_slots,
            cli.max_active_requests_explicit,
            max_seq_len,
            cli.kv_retire_q8_after_chunks,
            cli.prefix_cache_chunks,
            gcfg_kv_layers,
        )
    else
        1;
    var draft_cfg_opt: ?aminfer.draft.DraftConfig = null;
    var mode: aminfer.draft.DraftMode = cli.draft_mode orelse .l1_only;
    if (cli.draft_config_path) |dc_path| {
        const dcfg_bytes = try readFileAllocPosix(allocator, dc_path, 4 * 1024 * 1024);
        defer allocator.free(dcfg_bytes);
        draft_cfg_opt = try aminfer.draft.DraftConfig.fromJson(allocator, dcfg_bytes);
        if (cli.draft_mode == null) {
            mode = draft_cfg_opt.?.format;
        } else {
            draft_cfg_opt.?.format = mode;
        }
    }
    if (use_spec and mode != .l1_only and mode != .self_recurrent) {
        if (cli.draft_amf_path == null or draft_cfg_opt == null) return error.MissingRequiredFlag;
        draft_cfg_opt.?.format = mode;
    }

    const ple_paths = try resolvePlePathsAlloc(allocator, &cli.m6, cli.config_path orelse cli.amf_path, &gcfg);
    defer freePlePaths(allocator, ple_paths);

    const target_tree_cap: u32 = if (use_continuous)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, @intCast(batch_slots)), 32))
    else if (!use_spec)
        16
    else if (cli.enable_pruning)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, cli.max_tree_nodes) * 2, 24))
    else
        @as(u32, cli.max_tree_nodes);
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const aux_ids: ?[3]u32 = if (draft_cfg_opt) |dc| dc.aux_layer_ids else null;
    const needs_taps = use_spec and mode != .l1_only and mode != .self_recurrent;

    var model = try aminfer.graph.GraphModel.loadFromFile(allocator, cli.amf_path, gcfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = target_tree_cap,
        .enable_feature_taps = needs_taps,
        .aux_layer_ids = aux_ids,
        .wired_budget_override = budget_override,
        .ple_paths = ple_paths,
        .loop_adapter_amf_path = cli.m6.loop_adapter_amf_path,
        .max_loops_override = cli.m6.max_loops,
        .enable_loop_early_exit = cli.m6.loop_early_exit,
        .cold_slot_count = cli.cold_slots,
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
        .continuous_batch_only = use_continuous,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinitForGraph(allocator, &model);

    var ngram_tbl_storage: aminfer.ngram_nvme.NvmeNgramTable = undefined;
    var ngram_tbl_ptr: ?*aminfer.ngram_nvme.NvmeNgramTable = null;
    defer if (ngram_tbl_ptr) |nt| nt.deinit(allocator);
    if (cli.ngram_nvme_path) |ang_path| {
        const eff_ngram_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_tbl_storage = try aminfer.ngram_nvme.NvmeNgramTable.open(allocator, ang_path, .{
            .cache_pages = eff_ngram_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_tbl_ptr = &ngram_tbl_storage;
    }

    var ngram_reg_storage: aminfer.ngram_nvme.DomainNgramRegistry = undefined;
    var ngram_reg_ptr: ?*aminfer.ngram_nvme.DomainNgramRegistry = null;
    defer if (ngram_reg_ptr) |nr| nr.deinit(allocator);
    if (cli.ngram_pack_count > 0) {
        const eff_reg_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_reg_storage = try aminfer.ngram_nvme.DomainNgramRegistry.init(allocator, .{
            .cache_pages = eff_reg_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_reg_ptr = &ngram_reg_storage;
        for (cli.ngram_packs[0..cli.ngram_pack_count]) |pack| {
            _ = try ngram_reg_storage.registerPack(pack.domain, pack.path, cli.verify_crcs);
        }
    }

    var spec_eng_storage: aminfer.engine.SpecEngine = undefined;
    var spec_eng_ptr: ?*aminfer.engine.SpecEngine = null;
    defer if (spec_eng_ptr) |se| se.deinit(allocator);

    if (use_spec) {
        if (needs_taps) {
            draft_head_storage = try aminfer.draft.DraftHead.loadFromFileForGraph(
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
        spec_eng_storage = try aminfer.engine.SpecEngine.initWithGraph(allocator, &model, draft_head_ptr, .{
            .mode = mode,
            .placement = cli.placement,
            .max_tree_nodes = cli.max_tree_nodes,
            .max_draft_depth = cli.max_draft_depth,
            .branch_width = cli.branch_width,
            .top_k_per_node = cli.top_k,
            .temperature = 0.0,
            .enable_pruning = cli.enable_pruning,
            .pearl_mode = cli.pearl_mode,
            .pearl_min_target_us = cli.pearl_min_target_ms * 1000.0,
            .ngram_table = ngram_tbl_ptr,
            .ngram_registry = ngram_reg_ptr,
        });
        spec_eng_ptr = &spec_eng_storage;
    }

    var batch_rt_storage: aminfer.graph.GraphBatchRuntime = undefined;
    var batch_rt_ptr: ?*aminfer.graph.GraphBatchRuntime = null;
    defer if (batch_rt_ptr) |br| br.deinit(allocator);

    var batch_runner_storage: BatchServeRunner = undefined;
    var batch_runner_ptr: ?*BatchServeRunner = null;
    defer if (batch_runner_ptr) |br| br.deinit(allocator);

    var runner_storage: ServeRunner = undefined;
    var runner_ptr: ?*ServeRunner = null;
    defer if (runner_ptr) |r| r.deinit(allocator);

    const bos_opt: ?u32 = if (gcfg.bos_token_id < gcfg.vocab_size) gcfg.bos_token_id else null;
    const eos_opt: ?u32 = if (gcfg.eos_token_id < gcfg.vocab_size) gcfg.eos_token_id else null;
    if (use_continuous) {
        batch_rt_storage = try aminfer.graph.GraphBatchRuntime.init(allocator, &model, .{
            .max_slots = @intCast(batch_slots),
            .prefix_cache_chunks = cli.prefix_cache_chunks,
            .budget_override = budget_override,
            .gpu_prefill_mode = cli.gpu_prefill_mode,
            .strict_live_ram = cli.strict_live_ram,
        });
        batch_rt_ptr = &batch_rt_storage;
        batch_runner_storage = try BatchServeRunner.init(
            allocator,
            batch_rt_storage.batchBackend(),
            ngram_reg_ptr,
            &tok,
            bos_opt,
            eos_opt,
            max_seq_len,
        );
        batch_runner_ptr = &batch_runner_storage;
    } else {
        runner_storage = try ServeRunner.init(
            allocator,
            .{ .graph = &model },
            spec_eng_ptr,
            ngram_reg_ptr,
            &tok,
            gcfg.vocab_size,
            bos_opt,
            eos_opt,
            max_seq_len,
        );
        runner_ptr = &runner_storage;
    }

    const total_wired_bytes: u64 = (if (spec_eng_ptr) |se| se.totalWiredBytes() else model.totalWiredBytes()) +
        (if (batch_rt_ptr) |br| br.totalWiredBytes() else 0) +
        (if (ngram_tbl_ptr) |nt| nt.totalWiredBytes() else 0) +
        (if (ngram_reg_ptr) |nr| (if (spec_eng_ptr == null) nr.totalWiredBytes() else 0) else 0);
    const total_wired_mib: u64 = total_wired_bytes >> 20;
    const model_id_str = cli.model_id orelse std.fs.path.stem(cli.amf_path);

    const srv = try aminfer.server.Server.init(allocator, .{
        .host = host_ipv4,
        .port = cli.port,
        .model_id = model_id_str,
        .wired_mib = total_wired_mib,
        .max_conns = cli.max_conns,
        .max_active_requests = cli.max_active_requests,
        .max_batch_size = batch_slots,
        .max_prompt_tokens = max_seq_len,
        .message_boundary_cache = cli.message_boundary_cache,
        .decode_share = cli.decode_share,
        .adaptive_prefill_narrow = cli.adaptive_prefill_narrow,
        .ngram_registry = ngram_reg_ptr,
        .api_key = cli.api_key,
        .unix_socket_path = cli.unix_socket_path,
        .tokenizer = &tok,
    });
    defer srv.deinit(allocator);

    g_active_serve_ptr.store(@intFromPtr(srv), .release);
    defer g_active_serve_ptr.store(0, .release);

    const sa = std.c.Sigaction{
        .handler = .{ .handler = handleServeShutdownSignal },
        .mask = std.mem.zeroes(std.c.sigset_t),
        .flags = 0,
    };
    _ = std.c.sigaction(std.c.SIG.INT, &sa, null);
    _ = std.c.sigaction(std.c.SIG.TERM, &sa, null);

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    try srv.startNetworkThread();
    defer srv.stop();

    try out.print(
        \\serve.status                     listening
        \\serve.url                        http://{d}.{d}.{d}.{d}:{d}
        \\serve.model_id                   {s}
        \\serve.arch                       graph ({s}, {s}, loop={s} R={d})
        \\serve.batch_mode                 {s} (max_batch_size={d})
        \\serve.gpu_prefill_mode           {s}
        \\serve.decode_share               {d:.2} (adaptive_prefill_narrow={s}, message_boundary_cache={s})
        \\serve.prefix_cache_chunks        {d}
        \\serve.kv_retire_q8_after_chunks  {d}
        \\serve.ngram_packs                {d}
        \\serve.spec_mode                  {s}
        \\serve.wired_mib                  {d}
        \\serve.max_conns                  {d}
        \\serve.max_active_requests        {d}
        \\serve.max_seq_len                {d}
        \\
    , .{
        host_ipv4[0],
        host_ipv4[1],
        host_ipv4[2],
        host_ipv4[3],
        srv.boundPort(),
        srv.modelId(),
        gcfg.axes.attentionSummaryLabel(),
        gcfg.axes.ffn_kind.label(),
        gcfg.axes.loop_schedule.mode.label(),
        model.effective_max_loops,
        if (use_continuous) "continuous" else "serial",
        batch_slots,
        cli.gpu_prefill_mode.label(),
        cli.decode_share,
        if (cli.adaptive_prefill_narrow) "on" else "off",
        if (cli.message_boundary_cache) "on" else "off",
        if (use_continuous) cli.prefix_cache_chunks else 0,
        cli.kv_retire_q8_after_chunks,
        cli.ngram_pack_count,
        if (use_spec) mode.label() else "none",
        total_wired_mib,
        cli.max_conns,
        cli.max_active_requests,
        max_seq_len,
    });
    try out.flush();

    if (batch_runner_ptr) |br| {
        srv.runBatchedWorkerLoop(br.handler());
    } else {
        srv.runWorkerLoop(runner_ptr.?.handler());
    }
    srv.stop();

    const vm_after = try aminfer.vmstat.Snapshot.capture();
    const st = srv.stats();
    const steady_hd = if (batch_runner_ptr) |br| br.steadyHeapDelta() else runner_ptr.?.steadyHeapDelta();
    const batch_stats_opt: ?aminfer.graph.GraphBatchStats = if (batch_rt_ptr) |br| br.stats() else null;
    try out.print(
        \\serve.status                     stopped
        \\serve.total_requests             {d}
        \\serve.rejected_503               {d}
        \\serve.cancelled                  {d}
        \\serve.max_observed_batch_size    {d}
        \\serve.mean_batch_size            {d:.2}
        \\serve.total_batch_steps          {d}
        \\serve.prefix_hit_tokens          {d}
        \\serve.prefix_hit_requests        {d}
        \\serve.prefix_cow_copies          {d}
        \\serve.boundary_checkpoints       {d} (boundary_hits={d}, evict_first_evictions={d})
        \\serve.qos_steps                  decode={d} prefill={d} narrowed_chunks={d}
        \\serve.steady_malloc_blocks       {d}
        \\serve.steady_malloc_bytes        {d}
        \\serve.swapouts_delta             {d}
        \\
    , .{
        st.total_requests,
        st.rejected_503_count,
        st.cancelled_count,
        st.max_observed_batch_size,
        st.mean_batch_size,
        st.total_batch_steps,
        st.prefix_hit_tokens,
        st.prefix_hit_requests,
        if (batch_stats_opt) |bs| bs.prefix_cow_copies else 0,
        st.message_boundary_checkpoints,
        if (batch_stats_opt) |bs| bs.boundary_hits else 0,
        if (batch_stats_opt) |bs| bs.evict_first_evictions else 0,
        st.qos_decode_steps,
        st.qos_prefill_steps,
        st.narrowed_prefill_chunks,
        steady_hd.blocks_in_use,
        steady_hd.size_in_use,
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    });
    try aminfer.vmstat.checkNoSwap(vm_before, vm_after);
}

fn runServe(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseServeArgs(raw_args);
    const host_ipv4 = try aminfer.server.parseIpv4(cli.host);

    var meta = try resolveConfigAndTokenizerAlloc(allocator, cli.amf_path, cli.config_path, cli.tokenizer_path);
    defer meta.deinit(allocator);
    const cfg_bytes = meta.config.bytes;
    const tok_bytes = meta.tokenizer.bytes;

    if (shouldUseGraphModel(cfg_bytes, cli.m6, .{}, cli.draft_mode)) {
        return runServeGraph(allocator, cli, host_ipv4, cfg_bytes, tok_bytes, out);
    }
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, cfg_bytes);

    const heat_bytes: ?[]u8 = if (cli.heat_profile_path) |hp|
        try readFileAllocPosix(allocator, hp, 16 * 1024 * 1024)
    else
        null;
    defer if (heat_bytes) |hb| allocator.free(hb);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    const use_spec = cli.draft_mode != null or cli.draft_amf_path != null or cli.ngram_nvme_path != null;
    const use_continuous: bool = switch (cli.batch_mode) {
        .serial => false,
        .continuous => blk: {
            if (use_spec) return error.InvalidFlagCombination;
            break :blk true;
        },
        .auto => (!use_spec and cli.max_active_requests > 1),
    };
    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(@as(u32, 512), cli.max_seq_len));
    const raw_batch_slots: usize = if (use_continuous)
        @min(@max(cli.max_active_requests, 1), aminfer.batch_scheduler.MAX_BATCH_SLOTS)
    else
        1;
    const batch_slots: usize = if (use_continuous)
        autoClampServeBatchSlots(
            raw_batch_slots,
            cli.max_active_requests_explicit,
            max_seq_len,
            cli.kv_retire_q8_after_chunks,
            cli.prefix_cache_chunks,
            cfg.num_hidden_layers,
        )
    else
        1;
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

    if (use_spec) {
        if (mode == .small_draft) {
            if (cli.draft_amf_path == null or small_draft_cfg_opt == null) return error.MissingRequiredFlag;
        } else if (mode != .l1_only) {
            if (cli.draft_amf_path == null or draft_cfg_opt == null) return error.MissingRequiredFlag;
            draft_cfg_opt.?.format = mode;
        }
    }

    const target_tree_cap: u32 = if (use_continuous)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, @intCast(batch_slots)), 32))
    else if (!use_spec)
        1
    else if (cli.enable_pruning)
        @min(@as(u32, aminfer.engine.MAX_TREE_NODES), @max(@as(u32, cli.max_tree_nodes) * 2, 24))
    else
        @as(u32, cli.max_tree_nodes);

    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);
    const aux_ids: ?[3]u32 = if (draft_cfg_opt) |dc| dc.aux_layer_ids else null;
    const needs_taps = use_spec and mode != .l1_only and mode != .small_draft;

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
        .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
        .gpu_prefill_mode = cli.gpu_prefill_mode,
        .strict_live_ram = cli.strict_live_ram,
        .prefill_backend_mode = cli.m69.prefill_backend_mode,
        .strict_nax = cli.m69.strict_nax,
        .bus_bandwidth_gbs_override = cli.m69.bus_bandwidth_gbs,
        .continuous_batch_only = use_continuous,
    });
    defer model.deinit(allocator);

    var draft_head_storage: aminfer.draft.DraftHead = undefined;
    var draft_head_ptr: ?*aminfer.draft.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinit(allocator, &model);

    var small_draft_storage: aminfer.draft.SmallDraftModel = undefined;
    var small_draft_ptr: ?*aminfer.draft.SmallDraftModel = null;
    defer if (small_draft_ptr) |sd| sd.deinit(allocator);

    var ngram_tbl_storage: aminfer.ngram_nvme.NvmeNgramTable = undefined;
    var ngram_tbl_ptr: ?*aminfer.ngram_nvme.NvmeNgramTable = null;
    defer if (ngram_tbl_ptr) |nt| nt.deinit(allocator);
    if (cli.ngram_nvme_path) |ang_path| {
        const eff_ngram_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_tbl_storage = try aminfer.ngram_nvme.NvmeNgramTable.open(allocator, ang_path, .{
            .cache_pages = eff_ngram_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_tbl_ptr = &ngram_tbl_storage;
    }

    var ngram_reg_storage: aminfer.ngram_nvme.DomainNgramRegistry = undefined;
    var ngram_reg_ptr: ?*aminfer.ngram_nvme.DomainNgramRegistry = null;
    defer if (ngram_reg_ptr) |nr| nr.deinit(allocator);
    if (cli.ngram_pack_count > 0) {
        const eff_reg_pages = try scaleOptionalNgramCachePages(
            cli.ngram_cache_pages,
            budget_override,
            model.totalWiredBytes(),
            cli.strict_live_ram,
        );
        ngram_reg_storage = try aminfer.ngram_nvme.DomainNgramRegistry.init(allocator, .{
            .cache_pages = eff_reg_pages,
            .io_backend = cli.io_backend,
            .budget_override = budget_override,
        });
        ngram_reg_ptr = &ngram_reg_storage;
        for (cli.ngram_packs[0..cli.ngram_pack_count]) |pack| {
            _ = try ngram_reg_storage.registerPack(pack.domain, pack.path, cli.verify_crcs);
        }
    }

    var spec_eng_storage: aminfer.engine.SpecEngine = undefined;
    var spec_eng_ptr: ?*aminfer.engine.SpecEngine = null;
    defer if (spec_eng_ptr) |se| se.deinit(allocator);

    if (use_spec) {
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
                    .own_compute_queue = (cli.pearl_mode != .off),
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
            .temperature = 0.0,
            .enable_pruning = cli.enable_pruning,
            .pearl_mode = cli.pearl_mode,
            .pearl_min_target_us = cli.pearl_min_target_ms * 1000.0,
            .ngram_table = ngram_tbl_ptr,
            .ngram_registry = ngram_reg_ptr,
        };
        spec_eng_storage = if (small_draft_ptr) |sd|
            try aminfer.engine.SpecEngine.initWithSmallDraft(allocator, &model, sd, spec_cfg)
        else
            try aminfer.engine.SpecEngine.init(allocator, &model, draft_head_ptr, spec_cfg);
        spec_eng_ptr = &spec_eng_storage;
    }

    var lora_reg_storage: aminfer.lora.LoraRegistry = undefined;
    var lora_reg_ptr: ?*aminfer.lora.LoraRegistry = null;
    defer if (lora_reg_ptr) |lr| lr.deinit(allocator);
    if (cli.lora_spec_count > 0) {
        lora_reg_storage = try aminfer.lora.LoraRegistry.init(.{
            .pool_bytes = aminfer.lora.MAX_LORA_POOL_BYTES,
            .base_shape = model.baseLoraShape(),
            .sme_dispatcher = if (model.sme_rt) |srt| &srt.dispatcher else null,
            .wired_budget = budget_override,
        });
        lora_reg_ptr = &lora_reg_storage;
        for (cli.lora_specs[0..cli.lora_spec_count]) |ls| {
            _ = try lora_reg_storage.loadAdapterFromFile(allocator, ls.name, ls.path);
        }
        model.attachLoraRegistry(lora_reg_ptr);
    }

    var disk_prefix_storage: aminfer.prefix_cache.DiskPrefixCache = undefined;
    var disk_prefix_ptr: ?*aminfer.prefix_cache.DiskPrefixCache = null;
    defer if (disk_prefix_ptr) |dp| dp.deinit(allocator);
    if (cli.disk_prefix_cache_dir) |dpc_dir| {
        const kv_bytes_per_layer: usize = @as(usize, aminfer.prefix_cache.DEFAULT_BLOCK_TOKENS) *
            @as(usize, cfg.num_key_value_heads) * @as(usize, cfg.head_dim) * 2;
        const max_kv_art = @max(@as(usize, aminfer.page.APPLE_PAGE_SIZE), kv_bytes_per_layer * @as(usize, cfg.num_hidden_layers));
        disk_prefix_storage = try aminfer.prefix_cache.DiskPrefixCache.init(allocator, .{
            .cache_dir = dpc_dir,
            .amf_header_crc32 = meta.amf_header_crc32,
            .amf_dir_crc32 = meta.amf_dir_crc32,
            .max_tokens_per_artifact = aminfer.prefix_cache.DEFAULT_BLOCK_TOKENS,
            .max_kv_bytes_per_artifact = max_kv_art,
            .max_gdn_bytes_per_artifact = 0,
            .enable_async_worker = true,
        });
        disk_prefix_ptr = &disk_prefix_storage;
    }

    var model_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    defer _ = std.c.pthread_mutex_destroy(&model_mutex);

    const embed_tok_cap: usize = if (use_continuous)
        @min(@as(usize, max_seq_len), @as(usize, aminfer.attention.DEFAULT_CHUNK_TOKENS))
    else
        max_seq_len;
    var embed_ctx = try ServeEmbedRerankCtx.init(allocator, &model, &tok, &model_mutex, embed_tok_cap);
    defer embed_ctx.deinit(allocator);

    var batch_rt_storage: aminfer.qwen3.Qwen3BatchRuntime = undefined;
    var batch_rt_ptr: ?*aminfer.qwen3.Qwen3BatchRuntime = null;
    defer if (batch_rt_ptr) |br| br.deinit(allocator);

    var batch_runner_storage: BatchServeRunner = undefined;
    var batch_runner_ptr: ?*BatchServeRunner = null;
    defer if (batch_runner_ptr) |br| br.deinit(allocator);

    var runner_storage: ServeRunner = undefined;
    var runner_ptr: ?*ServeRunner = null;
    defer if (runner_ptr) |r| r.deinit(allocator);

    if (use_continuous) {
        batch_rt_storage = try aminfer.qwen3.Qwen3BatchRuntime.init(allocator, &model, .{
            .max_slots = @intCast(batch_slots),
            .prefix_cache_chunks = cli.prefix_cache_chunks,
            .kv_retire_q8_after_chunks = cli.kv_retire_q8_after_chunks,
            .budget_override = budget_override,
            .gpu_prefill_mode = cli.gpu_prefill_mode,
            .strict_live_ram = cli.strict_live_ram,
        });
        batch_rt_storage.attachLoraRegistry(lora_reg_ptr);
        batch_rt_ptr = &batch_rt_storage;
        batch_runner_storage = try BatchServeRunner.init(
            allocator,
            batch_rt_storage.batchBackend(),
            ngram_reg_ptr,
            &tok,
            cfg.bos_token_id,
            cfg.eos_token_id,
            max_seq_len,
        );
        batch_runner_storage.qwen3_batch_rt = &batch_rt_storage;
        batch_runner_storage.model_mutex = &model_mutex;
        batch_runner_ptr = &batch_runner_storage;
    } else {
        runner_storage = try ServeRunner.init(
            allocator,
            .{ .qwen3 = &model },
            spec_eng_ptr,
            ngram_reg_ptr,
            &tok,
            cfg.vocab_size,
            cfg.bos_token_id,
            cfg.eos_token_id,
            max_seq_len,
        );
        runner_storage.model_mutex = &model_mutex;
        runner_ptr = &runner_storage;
    }

    const total_wired_bytes: u64 = (if (spec_eng_ptr) |se| se.totalWiredBytes() else model.totalWiredBytes()) +
        (if (batch_rt_ptr) |br| br.totalWiredBytes() else 0) +
        (if (lora_reg_ptr) |lr| @as(u64, lr.stats().pool_capacity_bytes) else 0) +
        (if (ngram_tbl_ptr) |nt| nt.totalWiredBytes() else 0) +
        (if (ngram_reg_ptr) |nr| (if (spec_eng_ptr == null) nr.totalWiredBytes() else 0) else 0);
    const total_wired_mib: u64 = total_wired_bytes >> 20;
    const model_id_str = cli.model_id orelse std.fs.path.stem(cli.amf_path);

    const srv = try aminfer.server.Server.init(allocator, .{
        .host = host_ipv4,
        .port = cli.port,
        .model_id = model_id_str,
        .wired_mib = total_wired_mib,
        .max_conns = cli.max_conns,
        .max_active_requests = cli.max_active_requests,
        .max_batch_size = batch_slots,
        .max_prompt_tokens = max_seq_len,
        .message_boundary_cache = cli.message_boundary_cache,
        .decode_share = cli.decode_share,
        .adaptive_prefill_narrow = cli.adaptive_prefill_narrow,
        .ngram_registry = ngram_reg_ptr,
        .api_key = cli.api_key,
        .unix_socket_path = cli.unix_socket_path,
        .tokenizer = &tok,
        .embedding_handler = embed_ctx.embeddingHandler(),
        .rerank_handler = embed_ctx.rerankHandler(),
    });
    defer srv.deinit(allocator);
    if (batch_runner_ptr) |br| br.server_ptr = srv;
    if (runner_ptr) |r| r.server_ptr = srv;

    g_active_serve_ptr.store(@intFromPtr(srv), .release);
    defer g_active_serve_ptr.store(0, .release);

    const sa = std.c.Sigaction{
        .handler = .{ .handler = handleServeShutdownSignal },
        .mask = std.mem.zeroes(std.c.sigset_t),
        .flags = 0,
    };
    _ = std.c.sigaction(std.c.SIG.INT, &sa, null);
    _ = std.c.sigaction(std.c.SIG.TERM, &sa, null);

    const vm_before = try aminfer.vmstat.Snapshot.capture();
    try srv.startNetworkThread();
    defer srv.stop();

    try out.print(
        \\serve.status                     listening
        \\serve.url                        http://{d}.{d}.{d}.{d}:{d}
        \\serve.model_id                   {s}
        \\serve.arch                       {s} ({s})
        \\serve.batch_mode                 {s} (max_batch_size={d})
        \\serve.gpu_prefill_mode           {s}
        \\serve.decode_share               {d:.2} (adaptive_prefill_narrow={s}, message_boundary_cache={s})
        \\serve.prefix_cache_chunks        {d}
        \\serve.kv_retire_q8_after_chunks  {d}
        \\serve.ngram_packs                {d}
        \\serve.spec_mode                  {s}
        \\serve.pearl_mode                 {s} (enabled={s})
        \\serve.ngram_nvme                 {s}
        \\serve.wired_mib                  {d}
        \\serve.max_conns                  {d}
        \\serve.max_active_requests        {d}
        \\serve.max_seq_len                {d}
        \\
    , .{
        host_ipv4[0],
        host_ipv4[1],
        host_ipv4[2],
        host_ipv4[3],
        srv.boundPort(),
        srv.modelId(),
        cfg.arch.label(),
        cfg.precisionLabel(),
        if (use_continuous) "continuous" else "serial",
        batch_slots,
        cli.gpu_prefill_mode.label(),
        cli.decode_share,
        if (cli.adaptive_prefill_narrow) "on" else "off",
        if (cli.message_boundary_cache) "on" else "off",
        if (use_continuous) cli.prefix_cache_chunks else 0,
        cli.kv_retire_q8_after_chunks,
        cli.ngram_pack_count,
        if (use_spec) mode.label() else "none",
        cli.pearl_mode.label(),
        if (spec_eng_ptr) |se| (if (se.stats.pearl_enabled) "yes" else "no") else "no",
        if (ngram_tbl_ptr != null or ngram_reg_ptr != null) "on" else "off",
        total_wired_mib,
        cli.max_conns,
        cli.max_active_requests,
        max_seq_len,
    });
    try out.flush();

    // Run inference worker loop on the main scheduler thread so Metal 4 / ANE
    // contexts remain bound to a single deterministic thread.
    if (batch_runner_ptr) |br| {
        srv.runBatchedWorkerLoop(br.handler());
    } else {
        srv.runWorkerLoop(runner_ptr.?.handler());
    }
    srv.stop();

    const vm_after = try aminfer.vmstat.Snapshot.capture();
    const st = srv.stats();
    const steady_hd = if (batch_runner_ptr) |br| br.steadyHeapDelta() else runner_ptr.?.steadyHeapDelta();
    const batch_stats_opt: ?aminfer.qwen3.BatchRuntimeStats = if (batch_rt_ptr) |br| br.stats() else null;
    try out.print(
        \\serve.status                     stopped
        \\serve.total_requests             {d}
        \\serve.rejected_503               {d}
        \\serve.cancelled                  {d}
        \\serve.max_observed_batch_size    {d}
        \\serve.mean_batch_size            {d:.2}
        \\serve.total_batch_steps          {d}
        \\serve.prefix_hit_tokens          {d}
        \\serve.prefix_hit_requests        {d}
        \\serve.prefix_cow_copies          {d}
        \\serve.boundary_checkpoints       {d} (boundary_hits={d}, evict_first_evictions={d})
        \\serve.qos_steps                  decode={d} prefill={d} narrowed_chunks={d}
        \\serve.kv_retire_events           {d}
        \\serve.steady_malloc_blocks       {d}
        \\serve.steady_malloc_bytes        {d}
        \\serve.swapouts_delta             {d}
        \\
    , .{
        st.total_requests,
        st.rejected_503_count,
        st.cancelled_count,
        st.max_observed_batch_size,
        st.mean_batch_size,
        st.total_batch_steps,
        st.prefix_hit_tokens,
        st.prefix_hit_requests,
        if (batch_stats_opt) |bs| bs.prefix_cow_copies else 0,
        st.message_boundary_checkpoints,
        if (batch_stats_opt) |bs| bs.boundary_hits else 0,
        if (batch_stats_opt) |bs| bs.evict_first_evictions else 0,
        st.qos_decode_steps,
        st.qos_prefill_steps,
        st.narrowed_prefill_chunks,
        if (batch_stats_opt) |bs| bs.kv_retire_events else model.kvRetireEvents(),
        steady_hd.blocks_in_use,
        steady_hd.size_in_use,
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    });
    try aminfer.vmstat.checkNoSwap(vm_before, vm_after);
}

fn verifyAmf(allocator: std.mem.Allocator, path: [:0]const u8, out: *Io.Writer) !void {
    var file = try aminfer.amf.AmfFile.open(allocator, path);
    defer file.close(allocator);
    try file.verifyAllBlobs(allocator);

    var total_wired_bytes: u64 = 0;
    var cold_bytes: u64 = 0;
    var sparse_table_bytes: u64 = 0;
    var metadata_bytes: u64 = 0;
    var lut16_tensors: u64 = 0;
    var loop_adapter_tensors: u64 = 0;
    for (file.parsed.entries) |*entry| {
        const kind = try entry.tensorKind();
        const pk = try entry.quantParamKind();
        if (pk == .lut16_codebook) lut16_tensors += 1;
        if (kind == .loop_adapter) loop_adapter_tensors += 1;
        if (kind == .cold_expert) {
            cold_bytes += entry.blob_bytes;
        } else if (kind == .sparse_table) {
            sparse_table_bytes += entry.blob_bytes;
        } else if (kind == .metadata) {
            metadata_bytes += entry.blob_bytes;
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
    if (metadata_bytes > 0) {
        try out.print("amf.meta_bytes   {d}\n", .{metadata_bytes});
    }
    if (sparse_table_bytes > 0) {
        try out.print("amf.sparse_bytes {d}\n", .{sparse_table_bytes});
    }
    if (lut16_tensors > 0) {
        try out.print("amf.lut16_count  {d}\n", .{lut16_tensors});
    }
    if (loop_adapter_tensors > 0) {
        try out.print("amf.loop_tensors {d}\n", .{loop_adapter_tensors});
    }
}

fn runLs(allocator: std.mem.Allocator, raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const dir_path: []const u8 = if (raw_args.len >= 1) raw_args[0] else ".";
    const items = try aminfer.amf.scanAmfDirectory(allocator, dir_path);
    defer aminfer.amf.freeAmfInspections(allocator, items);

    try out.print("ls.dir                           {s}\n", .{dir_path});
    try out.print("ls.models_found                  {d}\n", .{items.len});
    try out.print(
        "  {s:<32}  {s:<18}  {s:>8}  {s:>9}  {s:>10}  {s:<10}  {s:<14}\n",
        .{ "model_id / file", "arch", "bpw", "file_mib", "wired_mib", "meta", "min_tier" },
    );
    for (items) |*item| {
        const base = std.fs.path.basename(item.path);
        const file_mib = @as(f64, @floatFromInt(item.file_bytes)) / (1024.0 * 1024.0);
        const wired_mib = @as(f64, @floatFromInt(item.estimated_total_wired_bytes)) / (1024.0 * 1024.0);
        var meta_buf: [16]u8 = undefined;
        const meta_tag = std.fmt.bufPrint(
            &meta_buf,
            "{s}/{s}/{s}",
            .{
                if (item.config_meta.present) "c" else "-",
                if (item.tokenizer_meta.present) "t" else "-",
                if (item.adapter_config_meta.present) "l" else "-",
            },
        ) catch "-/-/-";
        try out.print(
            "  {s:<32}  {s:<18}  {d:>8.2}  {d:>9.1}  {d:>10.1}  {s:<10}  {s:<14}\n",
            .{
                base,
                if (item.archSlice().len > 0) item.archSlice() else item.modelIdSlice(),
                item.effective_bits_per_weight,
                file_mib,
                wired_mib,
                meta_tag,
                item.tier_fit.recommended_min_tier.label(),
            },
        );
    }
}

fn runInspect(allocator: std.mem.Allocator, path: [:0]const u8, out: *Io.Writer) !void {
    const vm_before = try aminfer.vmstat.Snapshot.capture();
    var info = try aminfer.amf.inspectAmfPath(allocator, path);
    defer info.deinit(allocator);
    const vm_after = try aminfer.vmstat.Snapshot.capture();

    const file_mib = @as(f64, @floatFromInt(info.file_bytes)) / (1024.0 * 1024.0);
    const wired_weight_mib = @as(f64, @floatFromInt(info.wired_weight_bytes)) / (1024.0 * 1024.0);
    const est_total_mib = @as(f64, @floatFromInt(info.estimated_total_wired_bytes)) / (1024.0 * 1024.0);
    const has_emb_flag = (info.header_flags & aminfer.amf.FLAG_EMBEDDED_METADATA) != 0;

    try out.print(
        \\inspect.path                     {s}
        \\inspect.model_id                 {s}
        \\inspect.file_bytes               {d} ({d:.2} MiB)
        \\inspect.header_crc32             0x{x:0>8}
        \\inspect.directory_crc32          0x{x:0>8}
        \\inspect.header_flags             0x{x:0>8} (embedded_metadata={s})
        \\inspect.entries                  total={d} weights={d} metadata={d}
        \\inspect.wired_weight_bytes       {d} ({d:.2} MiB)
        \\inspect.cold_weight_bytes        {d}
        \\inspect.sparse_table_bytes       {d}
        \\inspect.metadata_blob_bytes      {d}
        \\inspect.estimated_kv_scratch     {d}
        \\inspect.estimated_total_wired    {d} ({d:.2} MiB)
        \\inspect.effective_bpw            {d:.2}
        \\
    , .{
        info.path,
        info.modelIdSlice(),
        info.file_bytes,
        file_mib,
        info.header_crc32,
        info.directory_crc32,
        info.header_flags,
        if (has_emb_flag) "yes" else "no",
        info.total_entries,
        info.weight_tensors,
        info.metadata_entries,
        info.wired_weight_bytes,
        wired_weight_mib,
        info.cold_weight_bytes,
        info.sparse_table_bytes,
        info.metadata_blob_bytes,
        info.estimated_runtime_kv_scratch_bytes,
        info.estimated_total_wired_bytes,
        est_total_mib,
        info.effective_bits_per_weight,
    });
    try out.print(
        \\inspect.config_source            {s} (embedded={s}, crc_valid={s})
        \\inspect.tokenizer_meta           embedded={s} (bytes={d}, crc_valid={s})
        \\inspect.adapter_config_meta      embedded={s} (bytes={d}, crc_valid={s})
        \\inspect.arch                     {s} (attn={s}, ffn={s}, kv={s}, sparse={s})
        \\inspect.dims                     layers={d} hidden={d} heads={d}/{d} head_dim={d} vocab={d}
        \\inspect.tier_fit                 min_tier="{s}" (m4_16gb={s}, m4_24gb={s}, pro_48gb={s}, max_128gb={s})
        \\inspect.swapouts_delta           {d}
        \\
    , .{
        info.config_source.label(),
        if (info.config_meta.present) "yes" else "no",
        if (info.config_meta.crc_valid) "yes" else "no",
        if (info.tokenizer_meta.present) "yes" else "no",
        info.tokenizer_meta.data_bytes,
        if (info.tokenizer_meta.crc_valid) "yes" else "no",
        if (info.adapter_config_meta.present) "yes" else "no",
        info.adapter_config_meta.data_bytes,
        if (info.adapter_config_meta.crc_valid) "yes" else "no",
        if (info.archSlice().len > 0) info.archSlice() else "unknown",
        info.arch.attentionSlice(),
        info.arch.ffnSlice(),
        info.arch.kvSlice(),
        info.arch.sparseSlice(),
        info.arch.num_hidden_layers,
        info.arch.hidden_size,
        info.arch.num_attention_heads,
        info.arch.num_key_value_heads,
        info.arch.head_dim,
        info.arch.vocab_size,
        info.tier_fit.recommended_min_tier.label(),
        if (info.tier_fit.fits_m4_16gb) "yes" else "no",
        if (info.tier_fit.fits_m4_24gb) "yes" else "no",
        if (info.tier_fit.fits_pro_48gb) "yes" else "no",
        if (info.tier_fit.fits_max_128gb) "yes" else "no",
        aminfer.vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    });
}

fn runMonitor(allocator: std.mem.Allocator, raw_args: []const [:0]const u8) !void {
    const plain = try allocator.alloc([]const u8, raw_args.len);
    defer allocator.free(plain);
    for (raw_args, 0..) |a, idx| plain[idx] = a;
    try aminfer.tui.runCli(allocator, plain);
}

const ChatCliArgs = struct {
    model_path: ?[:0]const u8 = null,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
    lora_specs: [aminfer.lora.MAX_ADAPTERS]LoraSpecArg = undefined,
    lora_spec_count: usize = 0,
    disk_prefix_cache_dir: ?[]const u8 = null,
    max_seq_len: u32 = 4096,
    prefix_cache_chunks: u32 = 8,
    wired_cap_mib: ?u64 = null,
    verify_crcs: bool = true,
    chat_cfg: aminfer.chat_repl.ChatConfig = .{},
};

fn parseChatCliArgs(args: []const [:0]const u8) !ChatCliArgs {
    var out: ChatCliArgs = .{};
    var repl_args_buf: [64][]const u8 = undefined;
    var repl_args_len: usize = 0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--model") or std.mem.eql(u8, flag, "-m") or std.mem.eql(u8, flag, "--amf")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.model_path = args[i];
            if (repl_args_len + 2 > repl_args_buf.len) return error.TooManyFlags;
            repl_args_buf[repl_args_len] = "--model";
            repl_args_buf[repl_args_len + 1] = args[i];
            repl_args_len += 2;
        } else if (std.mem.eql(u8, flag, "--config")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.config_path = args[i];
        } else if (std.mem.eql(u8, flag, "--tokenizer")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.tokenizer_path = args[i];
        } else if (std.mem.eql(u8, flag, "--lora")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (out.lora_spec_count >= out.lora_specs.len) return error.TooManyLoraAdapters;
            out.lora_specs[out.lora_spec_count] = try parseLoraSpecArg(args[i]);
            out.lora_spec_count += 1;
        } else if (std.mem.eql(u8, flag, "--disk-prefix-cache")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (args[i].len == 0) return error.InvalidFlagValue;
            out.disk_prefix_cache_dir = args[i];
        } else if (std.mem.eql(u8, flag, "--max-seq-len")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.max_seq_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--prefix-cache-chunks")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.prefix_cache_chunks = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--wired-cap-mib")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            out.wired_cap_mib = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, flag, "--no-verify-crc")) {
            out.verify_crcs = false;
        } else {
            if (repl_args_len >= repl_args_buf.len) return error.TooManyFlags;
            repl_args_buf[repl_args_len] = flag;
            repl_args_len += 1;
        }
    }
    out.chat_cfg = try aminfer.chat_repl.ChatConfig.parseArgs(repl_args_buf[0..repl_args_len]);
    return out;
}

const ChatEngineRunner = struct {
    model: *aminfer.qwen3.Qwen3Model,
    batch_rt: *aminfer.qwen3.Qwen3BatchRuntime,
    tok: *const aminfer.tokenizer.Tokenizer,
    max_seq_len: u32,
    bos_token_id: u32,
    stop_tokens: [4]u32,
    stop_token_count: usize,
    encode_scratch: aminfer.tokenizer.EncodeScratch,
    prompt_ids_buf: []u32,
    boundary_verify_buf: []u32,

    fn init(
        allocator: std.mem.Allocator,
        model: *aminfer.qwen3.Qwen3Model,
        batch_rt: *aminfer.qwen3.Qwen3BatchRuntime,
        tok: *const aminfer.tokenizer.Tokenizer,
        max_seq_len: u32,
    ) !ChatEngineRunner {
        var encode_scratch = try aminfer.tokenizer.EncodeScratch.init(
            allocator,
            @max(@as(usize, 8192), @as(usize, max_seq_len) * 2),
        );
        errdefer encode_scratch.deinit(allocator);
        const prompt_ids_buf = try allocator.alloc(u32, max_seq_len);
        errdefer allocator.free(prompt_ids_buf);
        const boundary_verify_buf = try allocator.alloc(u32, max_seq_len);
        errdefer allocator.free(boundary_verify_buf);

        var stop_tokens: [4]u32 = @splat(0);
        var stop_cnt: usize = 0;
        const candidates = [_]?u32{
            model.config.eos_token_id,
            tok.addedTokenId("<|im_end|>"),
            tok.addedTokenId("<|endoftext|>"),
        };
        for (candidates) |maybe_id| {
            const id = maybe_id orelse continue;
            var dup = false;
            for (stop_tokens[0..stop_cnt]) |existing| {
                if (existing == id) {
                    dup = true;
                    break;
                }
            }
            if (!dup and stop_cnt < stop_tokens.len) {
                stop_tokens[stop_cnt] = id;
                stop_cnt += 1;
            }
        }

        return .{
            .model = model,
            .batch_rt = batch_rt,
            .tok = tok,
            .max_seq_len = max_seq_len,
            .bos_token_id = model.config.bos_token_id orelse 0,
            .stop_tokens = stop_tokens,
            .stop_token_count = stop_cnt,
            .encode_scratch = encode_scratch,
            .prompt_ids_buf = prompt_ids_buf,
            .boundary_verify_buf = boundary_verify_buf,
        };
    }

    fn deinit(self: *ChatEngineRunner, allocator: std.mem.Allocator) void {
        allocator.free(self.boundary_verify_buf);
        allocator.free(self.prompt_ids_buf);
        self.encode_scratch.deinit(allocator);
    }

    fn isStopToken(self: *const ChatEngineRunner, tid: u32) bool {
        for (self.stop_tokens[0..self.stop_token_count]) |st| {
            if (tid == st) return true;
        }
        return false;
    }

    fn turnRunner(self: *ChatEngineRunner) aminfer.chat_repl.TurnRunner {
        return .{
            .ctx = self,
            .runTurnFn = runTurnCallback,
            .onClearFn = onClearCallback,
            .loadAttachmentFn = loadAttachmentCallback,
        };
    }

    fn onClearCallback(ctx: *anyopaque) void {
        const self: *ChatEngineRunner = @ptrCast(@alignCast(ctx));
        if (self.batch_rt.prefix_cache) |*pc| {
            pc.reset();
        }
    }

    fn loadAttachmentCallback(
        ctx: *anyopaque,
        kind: aminfer.chat_repl.AttachmentKind,
        path: []const u8,
    ) aminfer.chat_repl.Error!?*anyopaque {
        const self: *ChatEngineRunner = @ptrCast(@alignCast(ctx));
        if (path.len == 0 or path.len >= std.fs.max_path_bytes) return error.BackendFailed;
        var path_z_buf: [std.fs.max_path_bytes]u8 = undefined;
        @memcpy(path_z_buf[0..path.len], path);
        path_z_buf[path.len] = 0;
        const path_z: [:0]const u8 = path_z_buf[0..path.len :0];

        const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.BackendFailed;
        defer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0 or st.size <= 0) return error.BackendFailed;

        // If `kind == .image` and ImageIO can decode it into a 16 KiB-aligned G14 IOSurface
        // VisionFrame, verify zero-copy ingress and release the temporary surface cleanly.
        if (kind == .image) {
            if (aminfer.metal.VisionFrame.loadImage(self.model.device, path_z)) |vf_val| {
                var vf = vf_val;
                vf.close();
            } else |_| {}
        }
        return @ptrFromInt(1);
    }

    fn runTurnCallback(
        ctx: *anyopaque,
        req: *const aminfer.chat_repl.TurnRequest,
        stream: *aminfer.chat_repl.TurnStream,
    ) aminfer.chat_repl.Error!aminfer.chat_repl.TurnStats {
        const self: *ChatEngineRunner = @ptrCast(@alignCast(ctx));
        const t0 = try aminfer.vmstat.nowNs();

        var prompt_ids = self.tok.encodeWithScratch(
            &self.encode_scratch,
            req.chatml_prompt,
            self.prompt_ids_buf,
        ) catch return error.PromptTooLarge;
        if (prompt_ids.len == 0) {
            self.prompt_ids_buf[0] = self.bos_token_id;
            prompt_ids = self.prompt_ids_buf[0..1];
        }
        if (prompt_ids.len + 1 >= self.max_seq_len) return error.PromptTooLarge;

        var boundary_tok: ?u32 = null;
        if (req.last_user_msg_byte_offset) |b_off| {
            if (b_off > 0 and b_off < req.chatml_prompt.len and prompt_ids.len <= self.boundary_verify_buf.len) {
                if (self.tok.encodeWithScratch(&self.encode_scratch, req.chatml_prompt[0..b_off], self.boundary_verify_buf)) |prefix_ids| {
                    if (prefix_ids.len > 0 and prefix_ids.len < prompt_ids.len) {
                        if (self.tok.encodeWithScratch(
                            &self.encode_scratch,
                            req.chatml_prompt[b_off..],
                            self.boundary_verify_buf[prefix_ids.len..],
                        )) |suffix_ids| {
                            if (prefix_ids.len + suffix_ids.len == prompt_ids.len and
                                std.mem.eql(u32, self.boundary_verify_buf[0..prompt_ids.len], prompt_ids))
                            {
                                boundary_tok = @intCast(prefix_ids.len);
                            }
                        } else |_| {}
                    }
                } else |_| {}
            }
        }

        var req_rng = aminfer.rng.RequestRng.init(req.seed) catch return error.BackendFailed;
        const lookup = self.batch_rt.lookupPrefix(0, prompt_ids);
        var cur_prefix_handle: u64 = lookup.prefix_handle;
        defer self.batch_rt.batchBackend().releaseSlot(0, cur_prefix_handle);

        const chunk_cap: usize = aminfer.batch_scheduler.DEFAULT_MAX_PREFILL_CHUNK_TOKENS;
        var off: usize = @min(@as(usize, lookup.matched_tokens), prompt_ids.len - 1);
        var first_tok: u32 = 0;
        while (off < prompt_ids.len) {
            var take: usize = @min(chunk_cap, prompt_ids.len - off);
            var is_bndry_ckpt = false;
            if (boundary_tok) |b_tok| {
                const b_usize: usize = b_tok;
                if (b_usize > off and b_usize < prompt_ids.len and off + take >= b_usize) {
                    take = b_usize - off;
                    is_bndry_ckpt = true;
                }
            }
            const is_last = (off + take == prompt_ids.len);
            const pres = self.batch_rt.prefillSlot(&.{
                .slot_id = 0,
                .request_id = @as(u64, req.turn_index) + 1,
                .user_tag = 0,
                .tokens = prompt_ids[off .. off + take],
                .start_pos = @intCast(off),
                .total_prompt_len = @intCast(prompt_ids.len),
                .is_final_chunk = is_last,
                .prefix_handle = cur_prefix_handle,
                .temperature = req.temperature,
                .rng = &req_rng,
                .ngram_domain = "",
                .message_boundary_token = boundary_tok,
                .is_message_boundary_checkpoint = is_bndry_ckpt,
            }) catch return error.BackendFailed;
            if (pres.update_prefix_handle) {
                cur_prefix_handle = pres.prefix_handle;
            }
            if (is_last) {
                first_tok = pres.first_token;
            }
            off += take;
        }

        const prompt_len_u32: u32 = @intCast(prompt_ids.len);
        const room: u32 = self.max_seq_len - prompt_len_u32 - 1;
        const max_gen: u32 = @min(req.max_tokens, room);
        var emitted: u32 = 0;
        var cur_tok: u32 = first_tok;

        if (max_gen > 0 and !self.isStopToken(cur_tok)) {
            if (!self.tok.isSpecialToken(cur_tok)) {
                const piece = self.tok.tokenBytes(cur_tok) catch "";
                try stream.emitText(piece);
            }
            emitted = 1;
            var pos: u32 = prompt_len_u32;
            const slot_ids = [_]u16{0};
            const req_ids = [_]u64{@as(u64, req.turn_index) + 1};
            const user_tags = [_]u64{0};
            const temps = [_]f32{req.temperature};
            const rngs = [_]*aminfer.rng.RequestRng{&req_rng};
            const ngram_domains = [_][]const u8{""};
            const prefix_handles = [_]u64{cur_prefix_handle};
            var next_out = [_]u32{0};

            while (emitted < max_gen) {
                const in_toks = [_]u32{cur_tok};
                const seq_pos = [_]u32{pos};
                const view: aminfer.batch_scheduler.BatchStepView = .{
                    .slot_ids = &slot_ids,
                    .request_ids = &req_ids,
                    .user_tags = &user_tags,
                    .input_tokens = &in_toks,
                    .seq_positions = &seq_pos,
                    .temperatures = &temps,
                    .rngs = &rngs,
                    .ngram_domains = &ngram_domains,
                    .prefix_handles = &prefix_handles,
                };
                self.batch_rt.stepDecodeBatch(&view, &next_out) catch return error.BackendFailed;
                cur_tok = next_out[0];
                pos += 1;
                if (self.isStopToken(cur_tok)) break;
                if (!self.tok.isSpecialToken(cur_tok)) {
                    const piece = self.tok.tokenBytes(cur_tok) catch "";
                    try stream.emitText(piece);
                }
                emitted += 1;
            }
        }

        const t1 = try aminfer.vmstat.nowNs();
        const ttft_ns: u64 = if (stream.first_token_ns) |ft| (ft -| t0) else (t1 -| t0);
        return .{
            .prompt_tokens = prompt_len_u32,
            .generated_tokens = emitted,
            .prefix_hit_tokens = lookup.matched_tokens,
            .ttft_ns = ttft_ns,
            .total_ns = t1 -| t0,
        };
    }
};

fn runChat(allocator: std.mem.Allocator, raw_args: []const [:0]const u8) !void {
    const cli = try parseChatCliArgs(raw_args);
    if (cli.chat_cfg.show_help) {
        try aminfer.tui.writeAllFd(1, aminfer.chat_repl.HELP_TEXT);
        return;
    }
    const amf_path = cli.model_path orelse return error.MissingRequiredFlag;

    var meta = try resolveConfigAndTokenizerAlloc(allocator, amf_path, cli.config_path, cli.tokenizer_path);
    defer meta.deinit(allocator);
    const cfg = try aminfer.qwen3.ModelConfig.fromJson(allocator, meta.config.bytes);

    var tok = try aminfer.tokenizer.Tokenizer.fromHuggingFaceJson(allocator, meta.tokenizer.bytes);
    defer tok.deinit(allocator);

    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(@as(u32, 512), cli.max_seq_len));
    const budget_override = try buildBudgetOverride(cli.wired_cap_mib);

    var model = try aminfer.qwen3.Qwen3Model.loadFromFile(allocator, amf_path, cfg, .{
        .max_seq_len = max_seq_len,
        .max_tree_nodes = 32,
        .verify_crcs = cli.verify_crcs,
        .budget_override = budget_override,
        .continuous_batch_only = true,
    });
    defer model.deinit(allocator);

    var lora_reg_storage: aminfer.lora.LoraRegistry = undefined;
    var lora_reg_ptr: ?*aminfer.lora.LoraRegistry = null;
    defer if (lora_reg_ptr) |lr| lr.deinit(allocator);
    var active_lora_slot: u32 = aminfer.lora.NO_ADAPTER;
    if (cli.lora_spec_count > 0) {
        lora_reg_storage = try aminfer.lora.LoraRegistry.init(.{
            .pool_bytes = aminfer.lora.MAX_LORA_POOL_BYTES,
            .base_shape = model.baseLoraShape(),
            .sme_dispatcher = if (model.sme_rt) |srt| &srt.dispatcher else null,
            .wired_budget = budget_override,
        });
        lora_reg_ptr = &lora_reg_storage;
        for (cli.lora_specs[0..cli.lora_spec_count]) |ls| {
            active_lora_slot = try lora_reg_storage.loadAdapterFromFile(allocator, ls.name, ls.path);
        }
        model.attachLoraRegistry(lora_reg_ptr);
        model.setActiveLoraSlot(active_lora_slot);
    }

    var disk_prefix_storage: aminfer.prefix_cache.DiskPrefixCache = undefined;
    var disk_prefix_ptr: ?*aminfer.prefix_cache.DiskPrefixCache = null;
    defer if (disk_prefix_ptr) |dp| dp.deinit(allocator);
    if (cli.disk_prefix_cache_dir) |dpc_dir| {
        const kv_bytes_per_layer: usize = @as(usize, aminfer.prefix_cache.DEFAULT_BLOCK_TOKENS) *
            @as(usize, cfg.num_key_value_heads) * @as(usize, cfg.head_dim) * 2;
        const max_kv_art = @max(@as(usize, aminfer.page.APPLE_PAGE_SIZE), kv_bytes_per_layer * @as(usize, cfg.num_hidden_layers));
        disk_prefix_storage = try aminfer.prefix_cache.DiskPrefixCache.init(allocator, .{
            .cache_dir = dpc_dir,
            .amf_header_crc32 = meta.amf_header_crc32,
            .amf_dir_crc32 = meta.amf_dir_crc32,
            .max_tokens_per_artifact = aminfer.prefix_cache.DEFAULT_BLOCK_TOKENS,
            .max_kv_bytes_per_artifact = max_kv_art,
            .max_gdn_bytes_per_artifact = 0,
            .enable_async_worker = true,
        });
        disk_prefix_ptr = &disk_prefix_storage;
    }

    var batch_rt = try aminfer.qwen3.Qwen3BatchRuntime.init(allocator, &model, .{
        .max_slots = 1,
        .prefix_cache_chunks = @max(@as(u32, 4), cli.prefix_cache_chunks),
        .budget_override = budget_override,
    });
    defer batch_rt.deinit(allocator);
    batch_rt.attachLoraRegistry(lora_reg_ptr);
    batch_rt.setSlotLora(0, active_lora_slot);

    var chat_runner = try ChatEngineRunner.init(allocator, &model, &batch_rt, &tok, max_seq_len);
    defer chat_runner.deinit(allocator);

    try aminfer.chat_repl.runReplOrPipe(allocator, cli.chat_cfg, chat_runner.turnRunner(), 0, 1, 2, null);
}

/// Auto-clamps default `batch_slots` (when `--max-active-requests` was not
/// explicitly forced on the CLI) so `(batch_slots * active_chunks_per_slot +
/// prefix_cache_chunks + 1) * num_kv_layers * 2` stays safely below macOS's
/// per-process `IOSurface` ceiling (`4,096`).
fn autoClampServeBatchSlots(
    requested_slots: usize,
    explicit: bool,
    max_seq_len: u32,
    kv_retire_q8_after_chunks: u32,
    prefix_cache_chunks: u32,
    num_kv_layers: u32,
) usize {
    if (explicit or requested_slots <= 1 or num_kv_layers == 0) return requested_slots;
    const chunk_tokens: u32 = aminfer.attention.DEFAULT_CHUNK_TOKENS;
    const total_chunks: u32 = @max(@as(u32, 1), (max_seq_len + chunk_tokens - 1) / chunk_tokens);
    const active_per_slot: u32 = if (kv_retire_q8_after_chunks > 0 and total_chunks > kv_retire_q8_after_chunks)
        @max(kv_retire_q8_after_chunks, 2)
    else
        total_chunks;
    const max_iosurfaces: u64 = 4096;
    const surf_per_group: u64 = @as(u64, num_kv_layers) * 2;
    const max_groups: u64 = max_iosurfaces / surf_per_group;
    const reserved: u64 = 1 + @as(u64, prefix_cache_chunks);
    if (max_groups <= reserved) return 1;
    const slots_fit: usize = @intCast(@max(@as(u64, 1), (max_groups - reserved) / @as(u64, active_per_slot)));
    return @min(requested_slots, slots_fit);
}

fn probe(raw_args: []const [:0]const u8, out: *Io.Writer) !void {
    const cli = try parseM69Args(raw_args);
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

    var device = try aminfer.metal.Device.open();
    defer device.close();
    const info = try device.info();
    const chip = aminfer.chip_profile.ChipProfile.resolve(
        facts,
        info.name(),
        info.appleGpuFamily(),
        cli.bus_bandwidth_gbs,
    );
    const wired_budget = aminfer.budget.Budget.fromHost(facts, info, .{});
    const vm_snap = try aminfer.vmstat.Snapshot.capture();
    const ps: u64 = if (vm_snap.page_size > 0) vm_snap.page_size else aminfer.page.APPLE_PAGE_SIZE;
    const lib = try aminfer.metal.Library.openEmbedded(device);
    defer lib.close();
    const has_dense_nax = try lib.hasFunction("ami_dense_gemm_nax");
    const has_dequant_nax = try lib.hasFunction("ami_dequant_gemm_nax");
    const has_fp4_nax = try lib.hasFunction("ami_fp4_e2m1_gemm_nax");
    var dq_pipes = try aminfer.dequant.DequantPipelines.initWithOptions(device, lib, .{
        .backend_mode = cli.prefill_backend_mode,
        .strict_nax = cli.strict_nax,
    });
    defer dq_pipes.close();
    const canary_prof = try dq_pipes.runStartupNumericalCanaryProfileMut(&device);
    const canary = canary_prof.cold;
    try out.print(
        \\chip.generation                  {s}
        \\chip.tier                        {s}
        \\chip.cpu_brand                   {s}
        \\chip.hw_model                    {s}
        \\chip.spec_bus_gbs                {d:.2}
        \\chip.bandwidth_source            {s}
        \\gpu.name                         {s}
        \\gpu.apple_family                 {d}
        \\gpu.supports_apple10             {s}
        \\gpu.supports_apple11             {s}
        \\gpu.supports_nax                 {s}
        \\gpu.prefill_matmul_backend       {s}
        \\gpu.nax_fallback_reason          {s}
        \\gpu.metal4                       {}
        \\gpu.unified_memory               {}
        \\gpu.recommended_working_set_mib  {d}
        \\gpu.max_buffer_mib               {d}
        \\metallib.has_dense_gemm_nax      {s}
        \\metallib.has_dequant_gemm_nax    {s}
        \\metallib.has_fp4_e2m1_gemm_nax   {s}
        \\
    , .{
        chip.generation.label(),
        chip.tier.label(),
        facts.cpuBrand(),
        facts.hwModel(),
        chip.spec_bus_gbs,
        chip.bandwidth_source.label(),
        info.name(),
        info.appleGpuFamily(),
        if (info.supportsApple10()) "yes" else "no",
        if (info.supportsApple11()) "yes" else "no",
        if (info.supportsNaxTensorOps()) "yes" else "no",
        dq_pipes.activeBackend().label(),
        dq_pipes.naxFallbackReason().label(),
        info.supportsMetal4(),
        info.hasUnifiedMemory(),
        info.recommendedMaxWorkingSetSize() >> 20,
        info.maxBufferLength() >> 20,
        if (has_dense_nax) "yes" else "no",
        if (has_dequant_nax) "yes" else "no",
        if (has_fp4_nax) "yes" else "no",
    });
    try out.print(
        \\ane.cores                        {d}
        \\budget.wired_limit_mib           {d}
        \\budget.binding_limit             {s}
        \\vm.live_zero_swap_reclaimable_mib {d} (free={d}, speculative={d}, purgeable={d}, external={d}; excluded: inactive={d}, internal={d}, compressor={d}, wired={d})
        \\canary.status                    pass
        \\canary.cold_elapsed_us           {d:.1}
        \\canary.warm_inbuf_elapsed_us     {d:.1}
        \\canary.max_abs_err_f16           {d:.6}
        \\canary.max_abs_err_bf16          {d:.6}
        \\canary.max_abs_err_q4            {d:.6}
        \\
    , .{
        info.aneCoreCount(),
        wired_budget.limit_bytes >> 20,
        wired_budget.binding_limit.label(),
        vm_snap.liveZeroSwapReclaimableBytes() >> 20,
        (vm_snap.free_pages *| ps) >> 20,
        (vm_snap.speculative_pages *| ps) >> 20,
        (vm_snap.purgeable_pages *| ps) >> 20,
        (vm_snap.external_pages *| ps) >> 20,
        (vm_snap.inactive_pages *| ps) >> 20,
        (vm_snap.internal_pages *| ps) >> 20,
        (vm_snap.compressor_pages *| ps) >> 20,
        (vm_snap.wire_pages *| ps) >> 20,
        canary.elapsed_us,
        canary_prof.warm_elapsed_us,
        canary.max_abs_err_f16,
        canary.max_abs_err_bf16,
        canary.max_abs_err_q4,
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
    try std.testing.expectEqualStrings("c.json", ok.config_path.?);
    try std.testing.expectEqualStrings("t.json", ok.tokenizer_path.?);
    try std.testing.expectEqualStrings("hello world", ok.prompt);
    try std.testing.expectEqual(@as(u32, 8), ok.max_tokens);
    try std.testing.expect(ok.ignore_eos);
    try std.testing.expect(ok.verify_crcs);

    const sf = try parseDecodeArgs(&.{ "--amf", "m.amf" });
    try std.testing.expectEqual(@as(?[:0]const u8, null), sf.config_path);
    try std.testing.expectEqual(@as(?[:0]const u8, null), sf.tokenizer_path);

    try std.testing.expectError(error.MissingRequiredFlag, parseDecodeArgs(&.{}));
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

test "M5 CLI parsers handle --pearl, --ngram-nvme on spec-decode and aminfer serve flags" {
    const spec = try parseSpecDecodeArgs(&.{
        "--amf",                 "14b.amf",
        "--config",              "14b.json",
        "--tokenizer",           "tok.json",
        "--draft-amf",           "0.6b.amf",
        "--draft-config",        "0.6b.json",
        "--draft-mode",          "small",
        "--pearl",               "on",
        "--pearl-min-target-ms", "35.5",
        "--ngram-nvme",          "repo.ang",
        "--ngram-cache-pages",   "64",
    });
    try std.testing.expectEqual(aminfer.engine.PearlMode.on, spec.pearl_mode);
    try std.testing.expectApproxEqAbs(@as(f64, 35.5), spec.pearl_min_target_ms, 1e-6);
    try std.testing.expectEqualStrings("repo.ang", spec.ngram_nvme_path.?);
    try std.testing.expectEqual(@as(usize, 64), spec.ngram_cache_pages);

    const srv = try parseServeArgs(&.{
        "--amf",                 "0.6b.amf",
        "--config",              "0.6b.json",
        "--tokenizer",           "tok.json",
        "--host",                "127.0.0.1",
        "--port",                "0",
        "--model-id",            "qwen3-0.6b",
        "--max-seq-len",         "2048",
        "--max-conns",           "16",
        "--max-active-requests", "8",
        "--draft-mode",          "l1",
        "--pearl",               "auto",
        "--ngram-nvme",          "repo.ang",
    });
    try std.testing.expectEqualStrings("0.6b.amf", srv.amf_path);
    try std.testing.expectEqualStrings("127.0.0.1", srv.host);
    try std.testing.expectEqual(@as(u16, 0), srv.port);
    try std.testing.expectEqualStrings("qwen3-0.6b", srv.model_id.?);
    try std.testing.expectEqual(@as(u32, 2048), srv.max_seq_len);
    try std.testing.expectEqual(@as(usize, 16), srv.max_conns);
    try std.testing.expectEqual(@as(usize, 8), srv.max_active_requests);
    try std.testing.expectEqual(aminfer.draft.DraftMode.l1_only, srv.draft_mode.?);
    try std.testing.expectEqual(aminfer.engine.PearlMode.auto, srv.pearl_mode);
    try std.testing.expectEqualStrings("repo.ang", srv.ngram_nvme_path.?);

    try std.testing.expectError(error.InvalidPearlMode, parseServeArgs(&.{
        "--amf",       "0.6b.amf",
        "--config",    "0.6b.json",
        "--tokenizer", "tok.json",
        "--pearl",     "always",
    }));
}

test "M6 CLI parsers handle --ple-amf, --max-loops, --loop-adapter-amf, --loop-early-exit, self-recurrent & sme2" {
    const dec = try parseDecodeArgs(&.{
        "--amf",              "qwen4.amf",
        "--config",           "qwen4.json",
        "--tokenizer",        "tok.json",
        "--prompt",           "hello",
        "--ple-amf",          "ple_0.amf",
        "--ple-amf",          "ple_1.amf",
        "--max-loops",        "3",
        "--loop-adapter-amf", "steplora.amf",
        "--loop-early-exit",
    });
    try std.testing.expectEqual(@as(usize, 2), dec.m6.ple_amf_count);
    try std.testing.expectEqualStrings("ple_0.amf", dec.m6.pleSlice()[0]);
    try std.testing.expectEqualStrings("ple_1.amf", dec.m6.pleSlice()[1]);
    try std.testing.expectEqual(@as(?u32, 3), dec.m6.max_loops);
    try std.testing.expectEqualStrings("steplora.amf", dec.m6.loop_adapter_amf_path.?);
    try std.testing.expect(dec.m6.loop_early_exit);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",        "looplm.amf",
        "--config",     "looplm.json",
        "--tokenizer",  "tok.json",
        "--draft-mode", "self-recurrent",
        "--placement",  "sme2",
        "--max-loops",  "4",
    });
    try std.testing.expectEqual(aminfer.draft.DraftMode.self_recurrent, spec.draft_mode.?);
    try std.testing.expectEqual(aminfer.draft.Placement.cpu_sme2, spec.placement);
    try std.testing.expectEqual(@as(?u32, 4), spec.m6.max_loops);

    const srv = try parseServeArgs(&.{
        "--amf",        "qwen4.amf",
        "--config",     "qwen4.json",
        "--tokenizer",  "tok.json",
        "--draft-mode", "self-recurrent",
        "--placement",  "sme2",
        "--ple-amf",    "ple_0.amf",
        "--max-loops",  "2",
    });
    try std.testing.expectEqual(aminfer.draft.DraftMode.self_recurrent, srv.draft_mode.?);
    try std.testing.expectEqual(aminfer.draft.Placement.cpu_sme2, srv.placement);
    try std.testing.expectEqual(@as(usize, 1), srv.m6.ple_amf_count);
    try std.testing.expectEqual(@as(?u32, 2), srv.m6.max_loops);
}

test "M6.5 CLI parsers handle --batch-mode, --prefix-cache-chunks, --kv-retire-q8-after-chunks, --ngram-pack, --image and --vt-test-video" {
    const dec = try parseDecodeArgs(&.{
        "--amf",                       "qwen4.amf",
        "--config",                    "qwen4.json",
        "--tokenizer",                 "tok.json",
        "--prompt",                    "describe frame",
        "--kv-retire-q8-after-chunks", "2",
        "--image",                     "sample.png",
        "--vt-test-video",             "hevc",
    });
    try std.testing.expectEqual(@as(u32, 2), dec.kv_retire_q8_after_chunks);
    try std.testing.expectEqualStrings("sample.png", dec.image_path.?);
    try std.testing.expectEqual(aminfer.metal.VtCodec.hevc, dec.vt_test_video.?);

    const bdec = try parseBenchDecodeArgs(&.{
        "--amf",                       "30b.amf",
        "--config",                    "config.json",
        "--kv-retire-q8-after-chunks", "4",
    });
    try std.testing.expectEqual(@as(u32, 4), bdec.kv_retire_q8_after_chunks);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",                       "0.6b.amf",
        "--config",                    "0.6b.json",
        "--tokenizer",                 "tok.json",
        "--kv-retire-q8-after-chunks", "1",
    });
    try std.testing.expectEqual(@as(u32, 1), spec.kv_retire_q8_after_chunks);

    const srv = try parseServeArgs(&.{
        "--amf",                       "0.6b.amf",
        "--config",                    "0.6b.json",
        "--tokenizer",                 "tok.json",
        "--batch-mode",                "continuous",
        "--prefix-cache-chunks",       "24",
        "--kv-retire-q8-after-chunks", "2",
        "--ngram-pack",                "repo=/models/repo.ang",
        "--ngram-pack",                "docs=/models/docs.ang",
    });
    try std.testing.expectEqual(ServeBatchMode.continuous, srv.batch_mode);
    try std.testing.expectEqual(@as(u32, 24), srv.prefix_cache_chunks);
    try std.testing.expectEqual(@as(u32, 2), srv.kv_retire_q8_after_chunks);
    try std.testing.expectEqual(@as(usize, 2), srv.ngram_pack_count);
    try std.testing.expectEqualStrings("repo", srv.ngram_packs[0].domain);
    try std.testing.expectEqualStrings("/models/repo.ang", srv.ngram_packs[0].path);
    try std.testing.expectEqualStrings("docs", srv.ngram_packs[1].domain);
    try std.testing.expectEqualStrings("/models/docs.ang", srv.ngram_packs[1].path);

    try std.testing.expectError(error.InvalidBatchMode, parseServeArgs(&.{
        "--amf",        "0.6b.amf",
        "--config",     "0.6b.json",
        "--tokenizer",  "tok.json",
        "--batch-mode", "parallel",
    }));
    try std.testing.expectError(error.InvalidNgramPackSpec, parseServeArgs(&.{
        "--amf",        "0.6b.amf",
        "--config",     "0.6b.json",
        "--tokenizer",  "tok.json",
        "--ngram-pack", "missing_equals.ang",
    }));
    try std.testing.expectError(error.InvalidVtCodec, parseDecodeArgs(&.{
        "--amf",           "qwen4.amf",
        "--config",        "qwen4.json",
        "--tokenizer",     "tok.json",
        "--vt-test-video", "vp9",
    }));
}

test "M6.8 CLI parsers handle --gpu-prefill-mode, --strict-live-ram, --decode-share, --adaptive-prefill-narrow and --message-boundary-cache" {
    const dec = try parseDecodeArgs(&.{
        "--amf",              "m.amf",
        "--config",           "c.json",
        "--tokenizer",        "t.json",
        "--gpu-prefill-mode", "tree64",
        "--strict-live-ram",
    });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.tree64, dec.gpu_prefill_mode);
    try std.testing.expect(dec.strict_live_ram);

    const dec_def = try parseDecodeArgs(&.{ "--amf", "m.amf", "--config", "c.json", "--tokenizer", "t.json" });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.wide, dec_def.gpu_prefill_mode);
    try std.testing.expect(!dec_def.strict_live_ram);

    const bdec = try parseBenchDecodeArgs(&.{
        "--amf",              "m.amf",
        "--config",           "c.json",
        "--gpu-prefill-mode", "wide",
        "--strict-live-ram",
    });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.wide, bdec.gpu_prefill_mode);
    try std.testing.expect(bdec.strict_live_ram);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",              "m.amf",
        "--config",           "c.json",
        "--tokenizer",        "t.json",
        "--gpu-prefill-mode", "tree64",
        "--strict-live-ram",
    });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.tree64, spec.gpu_prefill_mode);
    try std.testing.expect(spec.strict_live_ram);

    const srv = try parseServeArgs(&.{
        "--amf",              "m.amf",
        "--config",           "c.json",
        "--tokenizer",        "t.json",
        "--gpu-prefill-mode", "wide",
        "--strict-live-ram",  "--decode-share",
        "0.65",               "--adaptive-prefill-narrow",
        "off",                "--message-boundary-cache",
        "off",
    });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.wide, srv.gpu_prefill_mode);
    try std.testing.expect(srv.strict_live_ram);
    try std.testing.expectApproxEqAbs(@as(f32, 0.65), srv.decode_share, 1e-6);
    try std.testing.expect(!srv.adaptive_prefill_narrow);
    try std.testing.expect(!srv.message_boundary_cache);

    const srv_def = try parseServeArgs(&.{ "--amf", "m.amf", "--config", "c.json", "--tokenizer", "t.json" });
    try std.testing.expectEqual(aminfer.qwen3.GpuPrefillMode.wide, srv_def.gpu_prefill_mode);
    try std.testing.expectApproxEqAbs(@as(f32, 0.50), srv_def.decode_share, 1e-6);
    try std.testing.expect(srv_def.adaptive_prefill_narrow);
    try std.testing.expect(srv_def.message_boundary_cache);
    try std.testing.expect(!srv_def.max_active_requests_explicit);

    const srv_exp = try parseServeArgs(&.{
        "--amf",                 "m.amf",
        "--config",              "c.json",
        "--tokenizer",           "t.json",
        "--max-active-requests", "3",
    });
    try std.testing.expect(srv_exp.max_active_requests_explicit);
    try std.testing.expectEqual(@as(usize, 3), srv_exp.max_active_requests);
    try std.testing.expectEqual(@as(usize, 2), autoClampServeBatchSlots(8, false, 10240, 0, 16, 28));
    try std.testing.expectEqual(@as(usize, 8), autoClampServeBatchSlots(8, false, 2048, 0, 16, 28));
    try std.testing.expectEqual(@as(usize, 3), autoClampServeBatchSlots(3, true, 10240, 0, 16, 28));

    try std.testing.expectError(error.InvalidGpuPrefillMode, parseDecodeArgs(&.{
        "--amf",              "m.amf",
        "--config",           "c.json",
        "--tokenizer",        "t.json",
        "--gpu-prefill-mode", "chunk128",
    }));
    try std.testing.expectError(error.InvalidDecodeShare, parseServeArgs(&.{
        "--amf",          "m.amf",
        "--config",       "c.json",
        "--tokenizer",    "t.json",
        "--decode-share", "1.25",
    }));
    try std.testing.expectError(error.InvalidOnOff, parseServeArgs(&.{
        "--amf",                     "m.amf",
        "--config",                  "c.json",
        "--tokenizer",               "t.json",
        "--adaptive-prefill-narrow", "maybe",
    }));
}

test "M6.9 CLI parsers handle --prefill-matmul-backend, --strict-nax, and --bus-bandwidth-gbs across all subcommands" {
    // Default M6.9 flags (`probe` / `bench-m69`).
    const m69_def = try parseM69Args(&.{});
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.auto, m69_def.prefill_backend_mode);
    try std.testing.expect(!m69_def.strict_nax);
    try std.testing.expectEqual(@as(?f64, null), m69_def.bus_bandwidth_gbs);

    const m69_exp = try parseM69Args(&.{
        "--prefill-matmul-backend", "nax",
        "--strict-nax",             "--bus-bandwidth-gbs",
        "546.5",
    });
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.nax, m69_exp.prefill_backend_mode);
    try std.testing.expect(m69_exp.strict_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 546.5), m69_exp.bus_bandwidth_gbs.?, 1e-6);

    const dec = try parseDecodeArgs(&.{
        "--amf",                    "m.amf",
        "--config",                 "c.json",
        "--tokenizer",              "t.json",
        "--prefill-matmul-backend", "simdgroup",
        "--bus-bandwidth-gbs",      "273.0",
    });
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.simdgroup, dec.m69.prefill_backend_mode);
    try std.testing.expect(!dec.m69.strict_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 273.0), dec.m69.bus_bandwidth_gbs.?, 1e-6);

    const bdec = try parseBenchDecodeArgs(&.{
        "--amf",                    "m.amf",
        "--config",                 "c.json",
        "--prefill-matmul-backend", "nax",
        "--strict-nax",             "--bus-bandwidth-gbs",
        "800",
    });
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.nax, bdec.m69.prefill_backend_mode);
    try std.testing.expect(bdec.m69.strict_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 800.0), bdec.m69.bus_bandwidth_gbs.?, 1e-6);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",                    "m.amf",
        "--config",                 "c.json",
        "--tokenizer",              "t.json",
        "--prefill-matmul-backend", "auto",
        "--strict-nax",             "--bus-bandwidth-gbs",
        "153.6",
    });
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.auto, spec.m69.prefill_backend_mode);
    try std.testing.expect(spec.m69.strict_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 153.6), spec.m69.bus_bandwidth_gbs.?, 1e-6);

    const srv = try parseServeArgs(&.{
        "--amf",                    "m.amf",
        "--config",                 "c.json",
        "--tokenizer",              "t.json",
        "--prefill-matmul-backend", "simdgroup",
        "--strict-nax",             "--bus-bandwidth-gbs",
        "400.0",
    });
    try std.testing.expectEqual(aminfer.dequant.PrefillBackendMode.simdgroup, srv.m69.prefill_backend_mode);
    try std.testing.expect(srv.m69.strict_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 400.0), srv.m69.bus_bandwidth_gbs.?, 1e-6);

    try std.testing.expectError(error.InvalidPrefillMatmulBackend, parseM69Args(&.{
        "--prefill-matmul-backend", "amx",
    }));
    try std.testing.expectError(error.InvalidBusBandwidthGbs, parseM69Args(&.{
        "--bus-bandwidth-gbs", "0",
    }));
    try std.testing.expectError(error.InvalidBusBandwidthGbs, parseM69Args(&.{
        "--bus-bandwidth-gbs", "-120.0",
    }));
    try std.testing.expectError(error.UnknownFlag, parseM69Args(&.{"--unknown"}));
}

test "M7 CLI parsers handle peer-worker, --peer/--rank/--world-size/--partition/--cluster-transport/--strict-tb5-rdma, and --draft-mode dspark" {
    const pw = try parsePeerWorkerArgs(&.{
        "--listen",            "127.0.0.1:19410",
        "--amf",               "model.amf",
        "--config",            "config.json",
        "--rank",              "1",
        "--world-size",        "2",
        "--partition",         "round-robin",
        "--cluster-transport", "tcp",
        "--max-requests",      "12",
    });
    try std.testing.expectEqual([4]u8{ 127, 0, 0, 1 }, pw.host);
    try std.testing.expectEqual(@as(u16, 19410), pw.port);
    try std.testing.expectEqual(@as(u16, 1), pw.rank);
    try std.testing.expectEqual(@as(u16, 2), pw.world_size);
    try std.testing.expectEqual(aminfer.expert_parallel.PartitionStrategy.round_robin, pw.partition);
    try std.testing.expectEqual(aminfer.transport.TransportKind.tb4_tcp, pw.transport_kind);
    try std.testing.expectEqual(@as(u32, 12), pw.max_requests);

    const dec = try parseDecodeArgs(&.{
        "--amf",               "model.amf",
        "--config",            "config.json",
        "--tokenizer",         "tok.json",
        "--peer",              "127.0.0.1:19410",
        "--peer",              "127.0.0.1:19411",
        "--rank",              "0",
        "--world-size",        "3",
        "--partition",         "contiguous",
        "--cluster-transport", "auto",
        "--shutdown-peers",
    });
    try std.testing.expect(dec.m7.hasPeers());
    try std.testing.expectEqual(@as(u16, 3), dec.m7.effectiveWorldSize());
    try std.testing.expect(dec.m7.shutdown_peers);
    var ep_buf: [aminfer.expert_parallel.MAX_WORLD_SIZE - 1]aminfer.expert_parallel.PeerEndpoint = undefined;
    const eps = try dec.m7.resolveEndpoints(&ep_buf);
    try std.testing.expectEqual(@as(usize, 2), eps.len);
    try std.testing.expectEqual(@as(u16, 1), eps[0].rank);
    try std.testing.expectEqual(@as(u16, 19410), eps[0].port);
    try std.testing.expectEqual(@as(u16, 2), eps[1].rank);
    try std.testing.expectEqual(@as(u16, 19411), eps[1].port);

    const spec = try parseSpecDecodeArgs(&.{
        "--amf",             "model.amf",
        "--config",          "config.json",
        "--tokenizer",       "tok.json",
        "--draft-mode",      "dspark",
        "--strict-tb5-rdma",
    });
    try std.testing.expectEqual(aminfer.draft.DraftMode.dspark, spec.draft_mode.?);
    try std.testing.expectEqual(aminfer.transport.TransportKind.tb5_rdma, spec.m7.effectiveTransport());

    const bdec = try parseBenchDecodeArgs(&.{
        "--amf",        "model.amf",
        "--config",     "config.json",
        "--peer",       "1=127.0.0.1:19412",
        "--world-size", "2",
    });
    try std.testing.expect(bdec.m7.hasPeers());
    const eps_b = try bdec.m7.resolveEndpoints(&ep_buf);
    try std.testing.expectEqual(@as(u16, 1), eps_b[0].rank);
    try std.testing.expectEqual(@as(u16, 19412), eps_b[0].port);

    try std.testing.expectError(error.InvalidPeerAddress, parsePeerWorkerArgs(&.{
        "--listen",     "not-an-ip",
        "--amf",        "m.amf",
        "--config",     "c.json",
        "--rank",       "1",
        "--world-size", "2",
    }));
    try std.testing.expectError(error.InvalidRank, parsePeerWorkerArgs(&.{
        "--listen",     "127.0.0.1:9000",
        "--amf",        "m.amf",
        "--config",     "c.json",
        "--rank",       "2",
        "--world-size", "2",
    }));
}

test "M7.5 CLI parsers handle W18 optional metadata flags, C18 --api-key/--unix-socket, W19 --lora, R10 --disk-prefix-cache, and C20 aminfer chat" {
    const srv = try parseServeArgs(&.{
        "--amf",               "single_file.amf",
        "--api-key",           "secret-token-123",
        "--unix-socket",       "/tmp/aminfer_m75.sock",
        "--lora",              "coder=coder_r8.amlora",
        "--lora",              "math=math_r8.amlora",
        "--disk-prefix-cache", "/tmp/aminfer_kv_cache",
    });
    try std.testing.expectEqualStrings("single_file.amf", srv.amf_path);
    try std.testing.expectEqual(@as(?[:0]const u8, null), srv.config_path);
    try std.testing.expectEqual(@as(?[:0]const u8, null), srv.tokenizer_path);
    try std.testing.expectEqualStrings("secret-token-123", srv.api_key.?);
    try std.testing.expectEqualStrings("/tmp/aminfer_m75.sock", srv.unix_socket_path.?);
    try std.testing.expectEqual(@as(usize, 2), srv.lora_spec_count);
    try std.testing.expectEqualStrings("coder", srv.lora_specs[0].name);
    try std.testing.expectEqualStrings("coder_r8.amlora", srv.lora_specs[0].path);
    try std.testing.expectEqualStrings("math", srv.lora_specs[1].name);
    try std.testing.expectEqualStrings("math_r8.amlora", srv.lora_specs[1].path);
    try std.testing.expectEqualStrings("/tmp/aminfer_kv_cache", srv.disk_prefix_cache_dir.?);

    const chat = try parseChatCliArgs(&.{
        "--model",             "single_file.amf",
        "--system",            "You are concise.",
        "--lora",              "coder=coder_r8.amlora",
        "--disk-prefix-cache", "/tmp/aminfer_chat_kv",
        "--no-think",          "--max-tokens",
        "128",                 "--temperature",
        "0.2",                 "--seed",
        "42",
    });
    try std.testing.expectEqualStrings("single_file.amf", chat.model_path.?);
    try std.testing.expectEqual(@as(?[:0]const u8, null), chat.config_path);
    try std.testing.expectEqual(@as(?[:0]const u8, null), chat.tokenizer_path);
    try std.testing.expectEqual(@as(usize, 1), chat.lora_spec_count);
    try std.testing.expectEqualStrings("coder", chat.lora_specs[0].name);
    try std.testing.expectEqualStrings("You are concise.", chat.chat_cfg.system_prompt);
    try std.testing.expect(!chat.chat_cfg.enable_thinking);
    try std.testing.expectEqual(@as(u32, 128), chat.chat_cfg.max_tokens);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), chat.chat_cfg.temperature, 1e-6);
    try std.testing.expectEqual(@as(?u64, 42), chat.chat_cfg.seed);

    try std.testing.expectError(error.InvalidLoraSpec, parseLoraSpecArg("no_equals"));
    try std.testing.expectError(error.InvalidLoraSpec, parseLoraSpecArg("=missing_name.amlora"));
}
