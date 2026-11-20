//! aminfer — bare-metal LLM inference engine for Apple Silicon.
//! Library root: the engine modules, re-exported for the CLI and tests.

pub const metal = @import("bridge/metal.zig");
pub const expert = @import("expert/source.zig");
pub const page = @import("mem/page.zig");
pub const budget = @import("mem/budget.zig");
pub const arena = @import("mem/arena.zig");
pub const quant = @import("quant/spec.zig");
pub const amf = @import("format/amf.zig");
pub const dequant = @import("gpu/dequant.zig");
pub const attention = @import("gpu/attention.zig");
pub const moe = @import("gpu/moe.zig");
pub const ring = @import("runtime/ring.zig");
pub const qos = @import("platform/qos.zig");
pub const sysinfo = @import("platform/sysinfo.zig");
pub const vmstat = @import("platform/vmstat.zig");
pub const tokenizer = @import("model/tokenizer.zig");
pub const qwen3 = @import("model/qwen3.zig");
pub const bench = @import("bench/runner.zig");

test {
    _ = metal;
    _ = expert;
    _ = page;
    _ = budget;
    _ = arena;
    _ = quant;
    _ = amf;
    _ = dequant;
    _ = attention;
    _ = moe;
    _ = ring;
    _ = qos;
    _ = sysinfo;
    _ = vmstat;
    _ = tokenizer;
    _ = qwen3;
    _ = bench;
}
