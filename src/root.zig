//! aminfer — bare-metal LLM inference engine for Apple Silicon.
//! Library root: the engine modules, re-exported for the CLI and tests.

pub const metal = @import("bridge/metal.zig");
pub const expert = @import("expert/source.zig");
pub const page = @import("mem/page.zig");
pub const sysinfo = @import("platform/sysinfo.zig");

test {
    _ = metal;
    _ = expert;
    _ = page;
    _ = sysinfo;
}
