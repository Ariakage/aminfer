//! aminfer command-line entry point.

const std = @import("std");
const Io = std.Io;
const aminfer = @import("aminfer");

const usage =
    \\usage: aminfer <command>
    \\
    \\commands:
    \\  probe             print the kernel, Metal and Neural Engine facts of this machine
    \\  verify-amf <path> validate an .amf v0 file's alignment, CRCs and wired budget fit
    \\  bench             run C9 kernel bandwidth calibration and Spike S2 hot-path probe
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    if (args.len == 2 and std.mem.eql(u8, args[1], "probe")) {
        try probe(out);
    } else if (args.len == 3 and std.mem.eql(u8, args[1], "verify-amf")) {
        try verifyAmf(arena, args[2], out);
    } else if (args.len == 2 and std.mem.eql(u8, args[1], "bench")) {
        try runBench(arena, out);
    } else {
        try out.writeAll(usage);
    }
    try out.flush();
}

fn runBench(allocator: std.mem.Allocator, out: *Io.Writer) !void {
    var report = try aminfer.bench.run(allocator, .{});
    defer report.deinit(allocator);
    try report.write(out);
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
