//! Multi-generation Apple Silicon chip profile, published/estimated memory-bus
//! bandwidth lookup (`spec_bus_gbs`), and per-chip runtime crossover gates
//! (`C16`, `ADR-0021`).
//!
//! Pure stack-allocated value types: zero heap allocations on resolution or
//! routing decisions (`ADR-0011`).

const std = @import("std");
const sysinfo = @import("sysinfo.zig");

/// Default `C13` SME2 + GPU co-execution speedup gate (`ADR-0018`).
pub const DEFAULT_C13_MIN_COEXEC_SPEEDUP: f64 = 1.05;

/// Default prompt-token threshold above which ANE hybrid wavefront prefill is
/// preferred on non-`NAX` (`M1–M4`, `Apple7–Apple9`) silicon when no per-chip
/// measured chunk latency pair is supplied (`A1`, `ADR-0010`, `ADR-0021`).
pub const DEFAULT_ANE_WAVEFRONT_MIN_TOKENS_NON_NAX: u32 = 513;

/// Default prompt-token threshold on `Apple10+` (`M5+`) silicon with GPU `NAX`
/// TensorOps, where wide GPU prefill raises GPU TFLOP/s substantially relative
/// to ANE (`ADR-0021`).
pub const DEFAULT_ANE_WAVEFRONT_MIN_TOKENS_NAX: u32 = 2049;

/// Required latency margin (`5%` faster) for measured ANE wavefront chunk
/// latency to override GPU wide prefill (`ADR-0021`, `C16`).
pub const ANE_WAVEFRONT_WIN_RATIO: f64 = 0.95;

pub const ChipGeneration = enum(u8) {
    unknown = 0,
    m1 = 1,
    m2 = 2,
    m3 = 3,
    m4 = 4,
    m5 = 5,
    post_m5 = 6,

    pub fn label(self: ChipGeneration) []const u8 {
        return switch (self) {
            .unknown => "unknown",
            .m1 => "m1",
            .m2 => "m2",
            .m3 => "m3",
            .m4 => "m4",
            .m5 => "m5",
            .post_m5 => "post_m5",
        };
    }
};

pub const ChipTier = enum(u8) {
    base = 0,
    pro = 1,
    max = 2,
    ultra = 3,

    pub fn label(self: ChipTier) []const u8 {
        return switch (self) {
            .base => "base",
            .pro => "pro",
            .max => "max",
            .ultra => "ultra",
        };
    }
};

pub const PrefillRoute = enum(u8) {
    gpu_wide = 0,
    ane_wavefront = 1,

    pub fn label(self: PrefillRoute) []const u8 {
        return switch (self) {
            .gpu_wide => "gpu_wide",
            .ane_wavefront => "ane_wavefront",
        };
    }
};

pub const BandwidthSource = enum(u8) {
    user_override = 0,
    apple_spec_table = 1,
    family_fallback_estimate = 2,

    pub fn label(self: BandwidthSource) []const u8 {
        return switch (self) {
            .user_override => "user_override",
            .apple_spec_table => "apple_spec_table",
            .family_fallback_estimate => "family_fallback_estimate",
        };
    }
};

pub const ChipProfile = struct {
    generation: ChipGeneration,
    tier: ChipTier,
    gpu_apple_family: u8,
    p_cores: u32,
    e_cores: u32,
    has_sme2: bool,
    has_nax: bool,
    spec_bus_gbs: f64,
    bandwidth_source: BandwidthSource,
    c13_min_coexec_speedup: f64,
    ane_wavefront_min_tokens_default: u32,

    /// Resolves the host Apple Silicon profile from `sysctl` facts, Metal device
    /// name (`MTLDevice.name`), probed Apple GPU family (`7..11+`), and an
    /// optional explicit `--bus-bandwidth-gbs` override (`> 0`).
    pub fn resolve(
        facts: sysinfo.Facts,
        device_name: []const u8,
        gpu_apple_family: u8,
        bus_gbs_override: ?f64,
    ) ChipProfile {
        const dev_gen = parseGenerationFromName(device_name, facts.has_sme2);
        const brand_slice = facts.cpuBrand();
        const generation: ChipGeneration = if (dev_gen != .unknown)
            dev_gen
        else blk: {
            const brand_gen = parseGenerationFromName(brand_slice, facts.has_sme2);
            if (brand_gen != .unknown) break :blk brand_gen;
            break :blk generationFromGpuFamily(gpu_apple_family, facts.has_sme2);
        };

        const tier: ChipTier = if (parseTierFromName(device_name)) |t|
            t
        else if (dev_gen != .unknown)
            .base
        else
            parseTierFromName(brand_slice) orelse .base;

        const has_nax: bool = if (gpu_apple_family > 0)
            gpu_apple_family >= 10
        else
            (generation == .m5 or generation == .post_m5);

        const looked_up = lookupBandwidth(generation, tier);
        var spec_bus_gbs = looked_up.gbs;
        var bandwidth_source = looked_up.source;
        if (bus_gbs_override) |ov| {
            if (std.math.isFinite(ov) and ov > 0.0) {
                spec_bus_gbs = ov;
                bandwidth_source = .user_override;
            }
        }

        return .{
            .generation = generation,
            .tier = tier,
            .gpu_apple_family = gpu_apple_family,
            .p_cores = @intCast(@min(facts.performance_cores, std.math.maxInt(u32))),
            .e_cores = @intCast(@min(facts.efficiency_cores, std.math.maxInt(u32))),
            .has_sme2 = facts.has_sme2,
            .has_nax = has_nax,
            .spec_bus_gbs = spec_bus_gbs,
            .bandwidth_source = bandwidth_source,
            .c13_min_coexec_speedup = DEFAULT_C13_MIN_COEXEC_SPEEDUP,
            .ane_wavefront_min_tokens_default = if (has_nax)
                DEFAULT_ANE_WAVEFRONT_MIN_TOKENS_NAX
            else
                DEFAULT_ANE_WAVEFRONT_MIN_TOKENS_NON_NAX,
        };
    }

    /// Selects between wide GPU prefill (`G15`/`G17`) and ANE hybrid wavefront
    /// prefill (`A1`) for a prompt of `prompt_tokens` tokens (`ADR-0021`, `C16`):
    /// 1) If `!ane_wavefront_ready`, always returns `.gpu_wide`.
    /// 2) If both `measured_gpu_chunk_ms` and `measured_ane_wavefront_chunk_ms`
    ///    are provided (`> 0`, finite), chooses `.ane_wavefront` iff
    ///    `measured_ane_wavefront_chunk_ms < measured_gpu_chunk_ms * 0.95`.
    /// 3) Otherwise falls back to `prompt_tokens >= self.ane_wavefront_min_tokens_default`.
    pub fn selectPrefillRoute(
        self: ChipProfile,
        prompt_tokens: u32,
        ane_wavefront_ready: bool,
        measured_gpu_chunk_ms: ?f64,
        measured_ane_wavefront_chunk_ms: ?f64,
    ) PrefillRoute {
        if (!ane_wavefront_ready) return .gpu_wide;
        if (measured_gpu_chunk_ms) |gpu_ms| {
            if (measured_ane_wavefront_chunk_ms) |ane_ms| {
                if (std.math.isFinite(gpu_ms) and gpu_ms > 0.0 and
                    std.math.isFinite(ane_ms) and ane_ms > 0.0)
                {
                    return if (ane_ms < gpu_ms * ANE_WAVEFRONT_WIN_RATIO)
                        .ane_wavefront
                    else
                        .gpu_wide;
                }
            }
        }
        return if (prompt_tokens >= self.ane_wavefront_min_tokens_default)
            .ane_wavefront
        else
            .gpu_wide;
    }

    /// Computes percentage of this chip's `spec_bus_gbs` achieved by `effective_gbs`.
    pub fn percentOfBus(self: ChipProfile, effective_gbs: f64) f64 {
        if (!(self.spec_bus_gbs > 0.0) or !std.math.isFinite(self.spec_bus_gbs)) return 0.0;
        return (effective_gbs / self.spec_bus_gbs) * 100.0;
    }

    /// Evaluates the `C13` SME2 + GPU co-execution speedup gate (`ADR-0018`).
    pub fn passesC13CoexecGate(self: ChipProfile, net_speedup: f64) bool {
        return std.math.isFinite(net_speedup) and net_speedup > self.c13_min_coexec_speedup;
    }
};

const BandwidthLookup = struct {
    gbs: f64,
    source: BandwidthSource,
};

fn lookupBandwidth(gen: ChipGeneration, tier: ChipTier) BandwidthLookup {
    return switch (gen) {
        .m1 => .{
            .gbs = switch (tier) {
                .base => 68.25,
                .pro => 200.0,
                .max => 400.0,
                .ultra => 800.0,
            },
            .source = .apple_spec_table,
        },
        .m2 => .{
            .gbs = switch (tier) {
                .base => 100.0,
                .pro => 200.0,
                .max => 400.0,
                .ultra => 800.0,
            },
            .source = .apple_spec_table,
        },
        .m3 => .{
            .gbs = switch (tier) {
                .base => 100.0,
                .pro => 150.0,
                .max => 400.0,
                .ultra => 819.0,
            },
            .source = .apple_spec_table,
        },
        .m4 => switch (tier) {
            .base => .{ .gbs = 120.0, .source = .apple_spec_table },
            .pro => .{ .gbs = 273.0, .source = .apple_spec_table },
            .max => .{ .gbs = 546.0, .source = .apple_spec_table },
            // No Apple-published M4 Ultra SKU; extrapolate conservatively from M4 Max.
            .ultra => .{ .gbs = 1092.0, .source = .family_fallback_estimate },
        },
        .m5, .post_m5 => .{
            .gbs = switch (tier) {
                .base => 153.6,
                .pro => 307.2,
                .max => 614.4,
                .ultra => 1228.8,
            },
            .source = .family_fallback_estimate,
        },
        .unknown => .{
            .gbs = switch (tier) {
                .base => 120.0,
                .pro => 200.0,
                .max => 400.0,
                .ultra => 800.0,
            },
            .source = .family_fallback_estimate,
        },
    };
}

fn generationFromGpuFamily(gpu_apple_family: u8, has_sme2: bool) ChipGeneration {
    return switch (gpu_apple_family) {
        7 => .m1,
        8 => .m2,
        9 => if (has_sme2) .m4 else .m3,
        10 => .m5,
        11...255 => .post_m5,
        else => .unknown,
    };
}

fn parseGenerationFromName(name: []const u8, has_sme2: bool) ChipGeneration {
    const m_gen = parseMTokenGeneration(name);
    if (m_gen != .unknown) return m_gen;
    if (parseAppleFamilyToken(name)) |fam| {
        return generationFromGpuFamily(fam, has_sme2);
    }
    return .unknown;
}

/// Parses a word-bounded `M<digits>` token (e.g. `"Apple M4 Max"` -> `.m4`,
/// `"Apple M10"` -> `.post_m5`) without matching substrings like `"Mac16,12"`.
fn parseMTokenGeneration(s: []const u8) ChipGeneration {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const ch = s[i];
        if (ch != 'm' and ch != 'M') continue;
        if (i > 0 and std.ascii.isAlphanumeric(s[i - 1])) continue;
        if (i + 1 >= s.len or !std.ascii.isDigit(s[i + 1])) continue;

        var end = i + 1;
        var val: u32 = 0;
        while (end < s.len and std.ascii.isDigit(s[end])) : (end += 1) {
            const digit: u32 = s[end] - '0';
            val = if (val > 1000) 1000 else val * 10 + digit;
        }
        if (end < s.len and std.ascii.isAlphanumeric(s[end])) continue;

        return switch (val) {
            0 => .unknown,
            1 => .m1,
            2 => .m2,
            3 => .m3,
            4 => .m4,
            5 => .m5,
            else => .post_m5,
        };
    }
    return .unknown;
}

/// Parses a word-bounded `Apple<digits>` token (e.g. `"Apple11"` -> `11`).
fn parseAppleFamilyToken(s: []const u8) ?u8 {
    const prefix = "apple";
    if (s.len <= prefix.len) return null;
    var i: usize = 0;
    while (i + prefix.len < s.len) : (i += 1) {
        if (i > 0 and std.ascii.isAlphanumeric(s[i - 1])) continue;
        if (!std.ascii.eqlIgnoreCase(s[i .. i + prefix.len], prefix)) continue;

        const digit_start = i + prefix.len;
        if (!std.ascii.isDigit(s[digit_start])) continue;

        var end = digit_start;
        var val: u32 = 0;
        while (end < s.len and std.ascii.isDigit(s[end])) : (end += 1) {
            const digit: u32 = s[end] - '0';
            val = if (val > 255) 255 else val * 10 + digit;
        }
        if (end < s.len and std.ascii.isAlphanumeric(s[end])) continue;
        if (val > 0) return @intCast(@min(val, 255));
    }
    return null;
}

fn containsWordIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (i > 0 and std.ascii.isAlphanumeric(haystack[i - 1])) continue;
        const end = i + needle.len;
        if (end < haystack.len and std.ascii.isAlphanumeric(haystack[end])) continue;
        if (std.ascii.eqlIgnoreCase(haystack[i..end], needle)) return true;
    }
    return false;
}

fn parseTierFromName(name: []const u8) ?ChipTier {
    if (containsWordIgnoreCase(name, "ultra")) return .ultra;
    if (containsWordIgnoreCase(name, "max")) return .max;
    if (containsWordIgnoreCase(name, "pro")) return .pro;
    return null;
}

fn syntheticFacts(p_cores: u64, e_cores: u64, has_sme2: bool, brand: []const u8) sysinfo.Facts {
    const base: sysinfo.Facts = .{
        .page_size = 16384,
        .memory_bytes = 24 * 1024 * 1024 * 1024,
        .performance_cores = p_cores,
        .efficiency_cores = e_cores,
        .user_wire_limit = 16 * 1024 * 1024 * 1024,
        .global_user_wire_limit = 16 * 1024 * 1024 * 1024,
        .gpu_wired_limit_mib = 0,
        .pcore_physical_cpu = p_cores,
        .pcore_cpus_per_l2 = p_cores,
        .pcore_l2_cache_bytes = 16 * 1024 * 1024,
        .pcore_clusters = 1,
        .has_sme = has_sme2,
        .has_sme2 = has_sme2,
        .has_sme_b16f32 = has_sme2,
        .has_sme_f16f32 = has_sme2,
        .sme_max_svl_bytes = if (has_sme2) 64 else 0,
    };
    return base.withIdentity(brand, "");
}

test "ChipProfile.resolve matches live host Facts.collect() and Metal DeviceInfo" {
    const metal = @import("../bridge/metal.zig");
    const facts = try sysinfo.Facts.collect();
    const dev = try metal.Device.open();
    defer dev.close();
    const dev_info = try dev.info();

    const profile = ChipProfile.resolve(
        facts,
        dev_info.name(),
        @intCast(dev_info.appleGpuFamily()),
        null,
    );

    try std.testing.expect(profile.p_cores > 0);
    try std.testing.expect(profile.e_cores > 0);
    try std.testing.expect(profile.spec_bus_gbs > 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.05), profile.c13_min_coexec_speedup, 1e-12);
    try std.testing.expectEqual(facts.has_sme2, profile.has_sme2);

    // On this baseline Apple M4 (`Apple9`, `FEAT_SME2=1`) host, verify exact M4 facts.
    if (std.mem.eql(u8, dev_info.name(), "Apple M4")) {
        try std.testing.expectEqual(ChipGeneration.m4, profile.generation);
        try std.testing.expectEqual(ChipTier.base, profile.tier);
        try std.testing.expectEqual(@as(u8, 9), profile.gpu_apple_family);
        try std.testing.expectApproxEqAbs(@as(f64, 120.0), profile.spec_bus_gbs, 1e-9);
        try std.testing.expectEqual(BandwidthSource.apple_spec_table, profile.bandwidth_source);
        try std.testing.expect(profile.has_sme2);
        try std.testing.expect(!profile.has_nax);
        try std.testing.expectEqual(@as(u32, 513), profile.ane_wavefront_min_tokens_default);
    }
}

test "ChipProfile.resolve maps M1 through M5+ generations, tiers, fallbacks, and user override" {
    // Apple M1 base (68.25 GB/s, Apple7, no SME2, no NAX).
    const m1 = ChipProfile.resolve(syntheticFacts(4, 4, false, "Apple M1"), "Apple M1", 7, null);
    try std.testing.expectEqual(ChipGeneration.m1, m1.generation);
    try std.testing.expectEqual(ChipTier.base, m1.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 68.25), m1.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m1.bandwidth_source);
    try std.testing.expect(!m1.has_sme2);
    try std.testing.expect(!m1.has_nax);
    try std.testing.expectEqual(@as(u32, 513), m1.ane_wavefront_min_tokens_default);

    // Apple M1 Pro / Max / Ultra table checks.
    try std.testing.expectApproxEqAbs(@as(f64, 200.0), ChipProfile.resolve(syntheticFacts(8, 2, false, ""), "Apple M1 Pro", 7, null).spec_bus_gbs, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 400.0), ChipProfile.resolve(syntheticFacts(8, 2, false, ""), "Apple M1 Max", 7, null).spec_bus_gbs, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 800.0), ChipProfile.resolve(syntheticFacts(16, 4, false, ""), "Apple M1 Ultra", 7, null).spec_bus_gbs, 1e-9);

    // Apple M2 Max (400.0 GB/s, Apple8).
    const m2_max = ChipProfile.resolve(syntheticFacts(8, 4, false, "Apple M2 Max"), "Apple M2 Max", 8, null);
    try std.testing.expectEqual(ChipGeneration.m2, m2_max.generation);
    try std.testing.expectEqual(ChipTier.max, m2_max.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 400.0), m2_max.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m2_max.bandwidth_source);
    try std.testing.expect(!m2_max.has_sme2 and !m2_max.has_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), ChipProfile.resolve(syntheticFacts(4, 4, false, ""), "Apple M2", 8, null).spec_bus_gbs, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 200.0), ChipProfile.resolve(syntheticFacts(8, 4, false, ""), "Apple M2 Pro", 8, null).spec_bus_gbs, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 800.0), ChipProfile.resolve(syntheticFacts(16, 8, false, ""), "Apple M2 Ultra", 8, null).spec_bus_gbs, 1e-9);

    // Apple M3 Pro (150.0 GB/s) and M3 Ultra (819.0 GB/s).
    const m3_pro = ChipProfile.resolve(syntheticFacts(6, 6, false, "Apple M3 Pro"), "Apple M3 Pro", 9, null);
    try std.testing.expectEqual(ChipGeneration.m3, m3_pro.generation);
    try std.testing.expectEqual(ChipTier.pro, m3_pro.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), m3_pro.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m3_pro.bandwidth_source);
    try std.testing.expect(!m3_pro.has_sme2 and !m3_pro.has_nax);

    const m3_ultra = ChipProfile.resolve(syntheticFacts(24, 8, false, "Apple M3 Ultra"), "Apple M3 Ultra", 9, null);
    try std.testing.expectEqual(ChipGeneration.m3, m3_ultra.generation);
    try std.testing.expectEqual(ChipTier.ultra, m3_ultra.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 819.0), m3_ultra.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m3_ultra.bandwidth_source);

    // Apple M4 Pro (273.0 GB/s) and M4 Max (546.0 GB/s).
    const m4_pro = ChipProfile.resolve(syntheticFacts(10, 4, true, "Apple M4 Pro"), "Apple M4 Pro", 9, null);
    try std.testing.expectEqual(ChipGeneration.m4, m4_pro.generation);
    try std.testing.expectEqual(ChipTier.pro, m4_pro.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 273.0), m4_pro.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m4_pro.bandwidth_source);
    try std.testing.expect(m4_pro.has_sme2 and !m4_pro.has_nax);

    const m4_max = ChipProfile.resolve(syntheticFacts(12, 4, true, "Apple M4 Max"), "Apple M4 Max", 9, null);
    try std.testing.expectEqual(ChipGeneration.m4, m4_max.generation);
    try std.testing.expectEqual(ChipTier.max, m4_max.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 546.0), m4_max.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, m4_max.bandwidth_source);
    try std.testing.expect(m4_max.has_sme2 and !m4_max.has_nax);

    // Apple M5 (gpu_apple_family = 10, has_nax = true, family_fallback_estimate).
    const m5 = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M5"), "Apple M5", 10, null);
    try std.testing.expectEqual(ChipGeneration.m5, m5.generation);
    try std.testing.expectEqual(ChipTier.base, m5.tier);
    try std.testing.expectEqual(@as(u8, 10), m5.gpu_apple_family);
    try std.testing.expect(m5.has_sme2 and m5.has_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 153.6), m5.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.family_fallback_estimate, m5.bandwidth_source);
    try std.testing.expectEqual(@as(u32, 2049), m5.ane_wavefront_min_tokens_default);

    const m5_pro = ChipProfile.resolve(syntheticFacts(10, 4, true, ""), "Apple M5 Pro", 10, null);
    try std.testing.expectApproxEqAbs(@as(f64, 307.2), m5_pro.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.family_fallback_estimate, m5_pro.bandwidth_source);

    const m5_max = ChipProfile.resolve(syntheticFacts(12, 4, true, ""), "Apple M5 Max", 10, null);
    try std.testing.expectApproxEqAbs(@as(f64, 614.4), m5_max.spec_bus_gbs, 1e-9);
    const m5_ultra = ChipProfile.resolve(syntheticFacts(24, 8, true, ""), "Apple M5 Ultra", 10, null);
    try std.testing.expectApproxEqAbs(@as(f64, 1228.8), m5_ultra.spec_bus_gbs, 1e-9);

    // Apple11 (gpu_apple_family = 11 -> .post_m5, has_nax = true).
    const apple11 = ChipProfile.resolve(syntheticFacts(6, 6, true, ""), "Apple11", 11, null);
    try std.testing.expectEqual(ChipGeneration.post_m5, apple11.generation);
    try std.testing.expectEqual(ChipTier.base, apple11.tier);
    try std.testing.expectEqual(@as(u8, 11), apple11.gpu_apple_family);
    try std.testing.expect(apple11.has_nax);
    try std.testing.expectApproxEqAbs(@as(f64, 153.6), apple11.spec_bus_gbs, 1e-9);
    try std.testing.expectEqual(BandwidthSource.family_fallback_estimate, apple11.bandwidth_source);

    // Apple11 resolves to .post_m5 even when facts carry a different cpuBrand.
    const apple11_override_name = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M4"), "Apple11", 11, null);
    try std.testing.expectEqual(ChipGeneration.post_m5, apple11_override_name.generation);

    // Fallback to facts.cpuBrand() when device_name is generic/empty.
    const brand_fb = ChipProfile.resolve(syntheticFacts(12, 4, true, "Apple M4 Max"), "", 9, null);
    try std.testing.expectEqual(ChipGeneration.m4, brand_fb.generation);
    try std.testing.expectEqual(ChipTier.max, brand_fb.tier);
    try std.testing.expectApproxEqAbs(@as(f64, 546.0), brand_fb.spec_bus_gbs, 1e-9);

    // Fallback to gpu_apple_family when both names are empty (Apple9 disambiguated by has_sme2).
    const fam9_m3 = ChipProfile.resolve(syntheticFacts(4, 4, false, ""), "", 9, null);
    try std.testing.expectEqual(ChipGeneration.m3, fam9_m3.generation);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), fam9_m3.spec_bus_gbs, 1e-9);

    const fam9_m4 = ChipProfile.resolve(syntheticFacts(4, 6, true, ""), "", 9, null);
    try std.testing.expectEqual(ChipGeneration.m4, fam9_m4.generation);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), fam9_m4.spec_bus_gbs, 1e-9);

    // Explicit --bus-bandwidth-gbs 320.0 sets .user_override regardless of table.
    const overridden = ChipProfile.resolve(syntheticFacts(10, 4, true, "Apple M4 Pro"), "Apple M4 Pro", 9, 320.0);
    try std.testing.expectEqual(BandwidthSource.user_override, overridden.bandwidth_source);
    try std.testing.expectApproxEqAbs(@as(f64, 320.0), overridden.spec_bus_gbs, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), overridden.percentOfBus(160.0), 1e-9);

    // Non-positive or NaN overrides are ignored cleanly.
    const zero_ov = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M4"), "Apple M4", 9, 0.0);
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, zero_ov.bandwidth_source);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), zero_ov.spec_bus_gbs, 1e-9);
    const nan_ov = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M4"), "Apple M4", 9, std.math.nan(f64));
    try std.testing.expectEqual(BandwidthSource.apple_spec_table, nan_ov.bandwidth_source);

    // C13 co-execution gate check.
    try std.testing.expect(!m4_pro.passesC13CoexecGate(1.05));
    try std.testing.expect(m4_pro.passesC13CoexecGate(1.06));
}

test "ChipProfile.selectPrefillRoute applies per-chip thresholds and measured latency gates" {
    const apple9 = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M4"), "Apple M4", 9, null);
    const apple10 = ChipProfile.resolve(syntheticFacts(4, 6, true, "Apple M5"), "Apple M5", 10, null);

    // ANE not ready always forces GPU wide.
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple9.selectPrefillRoute(4096, false, 10.0, 2.0));
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple10.selectPrefillRoute(4096, false, 10.0, 2.0));

    // Heuristic threshold without measurements: Apple9 (`513`) vs Apple10+ (`2049`).
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple9.selectPrefillRoute(512, true, null, null));
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple9.selectPrefillRoute(513, true, null, null));
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple9.selectPrefillRoute(1024, true, null, null));
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple9.selectPrefillRoute(2048, true, null, null));

    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple10.selectPrefillRoute(512, true, null, null));
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple10.selectPrefillRoute(1024, true, null, null));
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple10.selectPrefillRoute(2048, true, null, null));
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple10.selectPrefillRoute(2049, true, null, null));

    // Measured latency override (`ane_ms < gpu_ms * 0.95`):
    // Beats 5% threshold -> ANE wavefront even on Apple10 at 512 tokens.
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple10.selectPrefillRoute(512, true, 10.0, 9.4));
    // Within 5% margin or slower -> GPU wide even on Apple9 at 4096 tokens.
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple9.selectPrefillRoute(4096, true, 10.0, 9.5));
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple9.selectPrefillRoute(4096, true, 10.0, 10.2));

    // Non-positive measurements are ignored and fall back to token threshold.
    try std.testing.expectEqual(PrefillRoute.ane_wavefront, apple9.selectPrefillRoute(1024, true, 0.0, 5.0));
    try std.testing.expectEqual(PrefillRoute.gpu_wide, apple10.selectPrefillRoute(1024, true, 10.0, 0.0));
}
