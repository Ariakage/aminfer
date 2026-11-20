//! aminfer command-line entry point.

const std = @import("std");
const Io = std.Io;
const aminfer = @import("aminfer");

const usage =
    \\usage: aminfer <command>
    \\
    \\commands:
    \\  probe    print the Metal and Neural Engine facts of this machine
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    if (args.len == 2 and std.mem.eql(u8, args[1], "probe")) {
        try probe(out);
    } else {
        try out.writeAll(usage);
    }
    try out.flush();
}

fn probe(out: *Io.Writer) !void {
    const device = try aminfer.metal.Device.open();
    defer device.close();
    const info = try device.info();
    try out.print(
        \\gpu.name                         {s}
        \\gpu.apple_family                 {d}
        \\gpu.metal4                       {}
        \\gpu.unified_memory               {}
        \\gpu.recommended_working_set_mib  {d}
        \\gpu.max_buffer_mib               {d}
        \\ane.cores                        {d}
        \\
    , .{
        info.name(),
        info.appleGpuFamily(),
        info.supportsMetal4(),
        info.hasUnifiedMemory(),
        info.recommendedMaxWorkingSetSize() >> 20,
        info.maxBufferLength() >> 20,
        info.aneCoreCount(),
    });
}
