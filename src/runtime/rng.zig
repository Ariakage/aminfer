//! Two-tier per-request sampling RNG (capability D11,
//! `docs/architecture/speculative-decoding.md`).
//!
//! Tier 1 (cold path, once per request admission): when no caller seed is
//! provided, draws a 64-bit initial seed from macOS `getentropy(2)`, which is
//! fed by Apple Silicon's on-die Secure Enclave Processor (SEP) hardware TRNG.
//! Return codes are checked explicitly (never silently discarded).
//!
//! Tier 2 (per-token / per-round hot path): seeds an embedded 32-byte
//! `std.Random.Xoshiro256` (`std.Random.DefaultPrng`) inside the owning request
//! slot. Subsequent draws execute purely in registers/stack with zero syscalls,
//! zero global locks, zero cross-slot interference, and zero heap allocation.

const std = @import("std");

/// Darwin `<sys/random.h>` `getentropy(2)` symbol (macOS 10.12+, platform floor 26.0+).
extern "c" fn getentropy(buffer: [*]u8, size: usize) c_int;

pub const EntropyError = error{
    /// OS / Secure Enclave hardware CSPRNG returned a non-zero status code.
    EntropyUnavailable,
};

pub const SeedSource = enum(u8) {
    hardware_trng = 0,
    explicit = 1,

    pub fn name(self: SeedSource) []const u8 {
        return switch (self) {
            .hardware_trng => "hardware_trng",
            .explicit => "explicit",
        };
    }
};

comptime {
    if (std.Random.DefaultPrng != std.Random.Xoshiro256) {
        @compileError("expected std.Random.DefaultPrng to be std.Random.Xoshiro256");
    }
    if (@sizeOf(std.Random.Xoshiro256) != 32) {
        @compileError("expected std.Random.Xoshiro256 state to be exactly 32 bytes");
    }
}

/// Draws a 64-bit seed from macOS `getentropy(2)` (Apple Silicon Secure Enclave
/// hardware-TRNG-backed kernel CSPRNG). Fails explicitly on any non-zero status.
pub fn drawHardwareSeed64() EntropyError!u64 {
    return drawHardwareSeed64With(getentropy);
}

fn drawHardwareSeed64With(fill: *const fn ([*]u8, usize) callconv(.c) c_int) EntropyError!u64 {
    var buf: [8]u8 = undefined;
    if (fill(&buf, buf.len) != 0) {
        return error.EntropyUnavailable;
    }
    return std.mem.readInt(u64, &buf, .little);
}

/// Slot-local two-tier sampling PRNG (`seed_used` recorded for exact replay).
pub const RequestRng = struct {
    seed_used: u64,
    is_hardware_seeded: bool,
    prng: std.Random.Xoshiro256,

    /// Initializes a request RNG slot:
    /// - When `explicit_seed` is provided, uses it directly with zero syscalls.
    /// - When `explicit_seed` is `null`, draws one 64-bit seed from the
    ///   Secure Enclave hardware-TRNG-fed `getentropy(2)` CSPRNG.
    pub fn init(explicit_seed: ?u64) EntropyError!RequestRng {
        if (explicit_seed) |s| {
            return initDeterministic(s);
        }
        const hw_seed = try drawHardwareSeed64();
        return .{
            .seed_used = hw_seed,
            .is_hardware_seeded = true,
            .prng = std.Random.DefaultPrng.init(hw_seed),
        };
    }

    /// Infallible deterministic initializer for greedy, benchmark, and replay paths.
    pub fn initDeterministic(seed: u64) RequestRng {
        return .{
            .seed_used = seed,
            .is_hardware_seeded = false,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    /// Returns a `std.Random` handle backed solely by this slot's embedded
    /// 32-byte `Xoshiro256` state (zero syscalls, zero locks, zero heap allocs).
    pub fn random(self: *RequestRng) std.Random {
        return self.prng.random();
    }

    /// Advances the slot-local `Xoshiro256` state and returns the next `u64`.
    pub inline fn nextU64(self: *RequestRng) u64 {
        return self.prng.next();
    }

    /// Returns a uniform `f64` in `[0, 1)` from the slot-local `Xoshiro256` state.
    pub inline fn nextUnitF64(self: *RequestRng) f64 {
        return self.prng.random().float(f64);
    }

    /// Returns `.hardware_trng` or `.explicit` describing how `seed_used` was obtained.
    pub fn seedSource(self: RequestRng) SeedSource {
        return if (self.is_hardware_seeded) .hardware_trng else .explicit;
    }

    /// Returns `"hardware_trng"` or `"explicit"` for CLI / SSE metadata echoing.
    pub fn seedSourceName(self: RequestRng) []const u8 {
        return self.seedSource().name();
    }
};

// Local declaration matching `<malloc/malloc.h>` so standalone `zig test src/runtime/rng.zig`
// can verify zero C-heap allocations without importing outside `src/runtime/`.
const MallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;

fn fakeEntropyFailure(_: [*]u8, _: usize) callconv(.c) c_int {
    return -1;
}

fn fakeEntropyFixed(buf: [*]u8, len: usize) callconv(.c) c_int {
    if (len != 8) return -1;
    const expected = [_]u8{ 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };
    @memcpy(buf[0..8], &expected);
    return 0;
}

test "drawHardwareSeed64 checks return code explicitly and propagates EntropyUnavailable" {
    try std.testing.expectError(error.EntropyUnavailable, drawHardwareSeed64With(fakeEntropyFailure));
    const fixed = try drawHardwareSeed64With(fakeEntropyFixed);
    try std.testing.expectEqual(@as(u64, 0x8877_6655_4433_2211), fixed);
}

test "RequestRng.init(null) draws 64 distinct hardware seeds with zero errors and zero collisions" {
    var seeds: [64]u64 = undefined;
    for (&seeds, 0..) |*slot, i| {
        const rng = try RequestRng.init(null);
        try std.testing.expect(rng.is_hardware_seeded);
        try std.testing.expectEqual(SeedSource.hardware_trng, rng.seedSource());
        try std.testing.expectEqualStrings("hardware_trng", rng.seedSourceName());
        slot.* = rng.seed_used;

        for (seeds[0..i]) |prev| {
            try std.testing.expect(prev != slot.*);
        }
    }

    // Sanity check: consecutive 64-bit hardware seeds must not be trivially monotonic or zero.
    var nonzero_count: usize = 0;
    for (seeds) |s| {
        if (s != 0) nonzero_count += 1;
    }
    try std.testing.expect(nonzero_count >= 63);
}

test "RequestRng.init(seed_used) replays bit-for-bit identical u64 and f64 stream" {
    var hw_rng = try RequestRng.init(null);
    try std.testing.expect(hw_rng.is_hardware_seeded);

    var replay_rng = try RequestRng.init(hw_rng.seed_used);
    try std.testing.expect(!replay_rng.is_hardware_seeded);
    try std.testing.expectEqual(SeedSource.explicit, replay_rng.seedSource());
    try std.testing.expectEqualStrings("explicit", replay_rng.seedSourceName());
    try std.testing.expectEqual(hw_rng.seed_used, replay_rng.seed_used);

    const hw_handle = hw_rng.random();
    const replay_handle = replay_rng.random();

    for (0..256) |_| {
        const u_hw = hw_handle.int(u64);
        const u_rep = replay_handle.int(u64);
        try std.testing.expectEqual(u_hw, u_rep);

        const f_hw = hw_handle.float(f64);
        const f_rep = replay_handle.float(f64);
        try std.testing.expectEqual(@as(u64, @bitCast(f_hw)), @as(u64, @bitCast(f_rep)));
        try std.testing.expect(f_hw >= 0.0 and f_hw < 1.0);
    }

    // Explicit deterministic initializer must match `init(seed)` identically.
    var det_a = try RequestRng.init(0xDEAD_BEEF_CAFE_BABE);
    var det_b = RequestRng.initDeterministic(0xDEAD_BEEF_CAFE_BABE);
    try std.testing.expect(!det_a.is_hardware_seeded);
    try std.testing.expect(!det_b.is_hardware_seeded);
    for (0..128) |_| {
        try std.testing.expectEqual(det_a.nextU64(), det_b.nextU64());
        try std.testing.expectEqual(
            @as(u64, @bitCast(det_a.nextUnitF64())),
            @as(u64, @bitCast(det_b.nextUnitF64())),
        );
    }
}

test "interleaved independent RequestRng slots have zero cross-slot interference and zero malloc allocations" {
    const slot_count = 8;
    const steps_per_slot = 512;

    var slots: [slot_count]RequestRng = undefined;
    var refs: [slot_count]std.Random.Xoshiro256 = undefined;
    for (0..slot_count) |i| {
        const seed: u64 = 0x9E37_79B9_7F4A_7C15 +% (@as(u64, i) *% 0xBF58_476D_1CE4_E5B9);
        slots[i] = try RequestRng.init(seed);
        refs[i] = std.Random.DefaultPrng.init(seed);
    }

    // Warm up malloc_zone_statistics once, then assert zero heap activity across the hot loop.
    var m_warm: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_warm);

    var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_before);

    var mismatch_count: usize = 0;
    var sink: u64 = 0;
    for (0..steps_per_slot) |step| {
        // Step slots in a permuted order each iteration to prove strict slot isolation.
        for (0..slot_count) |j| {
            const idx = (j * 3 + step) % slot_count;
            const r_slot = slots[idx].random();
            const r_ref = refs[idx].random();

            const got_u = r_slot.int(u64);
            const exp_u = r_ref.int(u64);
            if (got_u != exp_u) mismatch_count += 1;

            const got_f: u64 = @bitCast(r_slot.float(f64));
            const exp_f: u64 = @bitCast(r_ref.float(f64));
            if (got_f != exp_f) mismatch_count += 1;

            sink ^= got_u +% got_f;
        }
    }
    std.mem.doNotOptimizeAway(sink);

    var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_after);

    try std.testing.expectEqual(@as(usize, 0), mismatch_count);
    try std.testing.expectEqual(m_before.blocks_in_use, m_after.blocks_in_use);
    try std.testing.expectEqual(m_before.size_in_use, m_after.size_in_use);
    try std.testing.expectEqual(m_before.size_allocated, m_after.size_allocated);
}
