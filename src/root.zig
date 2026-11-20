//! aminfer — bare-metal LLM inference engine for Apple Silicon.
//! Library root: the engine modules, re-exported for the CLI and tests.

pub const metal = @import("bridge/metal.zig");

test {
    _ = metal;
}
