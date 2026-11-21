//! Read-only kernel facts the engine budgets against, read through
//! `sysctlbyname` (docs/architecture/hardware.md).

const std = @import("std");
const page = @import("../mem/page.zig");

pub const Error = std.posix.UnexpectedError || error{
    /// The kernel does not know this sysctl name.
    UnknownName,
    /// The value is not a 32- or 64-bit integer.
    UnexpectedSize,
};

/// Reads an unsigned 32- or 64-bit integer sysctl.
pub fn readUnsigned(name: [:0]const u8) Error!u64 {
    var value: u64 = 0;
    var size: usize = @sizeOf(u64);
    switch (std.c.errno(std.c.sysctlbyname(name.ptr, &value, &size, null, 0))) {
        .SUCCESS => {},
        .NOENT => return error.UnknownName,
        .NOMEM => return error.UnexpectedSize,
        else => |err| return std.posix.unexpectedErrno(err),
    }
    return switch (size) {
        // aarch64 is little-endian: a 4-byte value fills the low half.
        @sizeOf(u32), @sizeOf(u64) => value,
        else => error.UnexpectedSize,
    };
}

/// Reads an optional unsigned 32- or 64-bit integer sysctl.
/// Returns `null` only when the kernel reports `ENOENT` (`UnknownName`);
/// other errors propagate explicitly.
pub fn readOptionalUnsigned(name: [:0]const u8) Error!?u64 {
    return readUnsigned(name) catch |err| switch (err) {
        error.UnknownName => null,
        else => err,
    };
}

/// Reads an optional boolean sysctl flag (`ENOENT` or `0` -> `false`, non-zero -> `true`).
pub fn readOptionalFlag(name: [:0]const u8) Error!bool {
    const val = (try readOptionalUnsigned(name)) orelse return false;
    return val != 0;
}

/// Reads an optional boolean feature that Apple kernels expose under either a
/// primary or alias sysctl name (e.g. `SME_B16F32` vs `FEAT_SME_B16F32`).
fn readOptionalFlagAny(primary: [:0]const u8, fallback: [:0]const u8) Error!bool {
    if (try readOptionalUnsigned(primary)) |v| return v != 0;
    if (try readOptionalUnsigned(fallback)) |v| return v != 0;
    return false;
}

pub const Facts = struct {
    /// `hw.pagesize`
    page_size: u64,
    /// `hw.memsize`
    memory_bytes: u64,
    /// `hw.perflevel0.logicalcpu`
    performance_cores: u64,
    /// `hw.perflevel1.logicalcpu`
    efficiency_cores: u64,
    /// `vm.user_wire_limit`: per-process `mlock` limit in bytes.
    user_wire_limit: u64,
    /// `vm.global_user_wire_limit`: system-wide `mlock` limit in bytes.
    global_user_wire_limit: u64,
    /// `iogpu.wired_limit_mb`: GPU wired limit in MiB; 0 means the system default.
    gpu_wired_limit_mib: u64,
    /// `hw.perflevel0.physicalcpu` (falls back to `performance_cores`).
    pcore_physical_cpu: u64 = 0,
    /// `hw.perflevel0.cpusperl2`: P-cores sharing one L2 cache cluster.
    pcore_cpus_per_l2: u64 = 0,
    /// `hw.perflevel0.l2cachesize`: L2 cache size per P-core cluster in bytes.
    pcore_l2_cache_bytes: u64 = 0,
    /// Derived physical P-core cluster count (`ceil(pcore_physical_cpu / pcore_cpus_per_l2)`).
    pcore_clusters: u64 = 1,
    /// `hw.optional.arm.FEAT_SME`
    has_sme: bool = false,
    /// `hw.optional.arm.FEAT_SME2`
    has_sme2: bool = false,
    /// `hw.optional.arm.SME_B16F32` (or `FEAT_SME_B16F32`)
    has_sme_b16f32: bool = false,
    /// `hw.optional.arm.SME_F16F32` (or `FEAT_SME_F16F32`)
    has_sme_f16f32: bool = false,
    /// `hw.optional.arm.SME_F32F32` (or `FEAT_SME_F32F32`)
    has_sme_f32f32: bool = false,
    /// `hw.optional.arm.FEAT_SME_I16I64`
    has_sme_i16i64: bool = false,
    /// `hw.optional.arm.sme_max_svl_b`: maximum Streaming Vector Length in bytes (64 on M4).
    sme_max_svl_bytes: u64 = 0,

    pub fn pcoreL2Kib(self: Facts) u64 {
        return self.pcore_l2_cache_bytes / 1024;
    }

    /// True when the host exposes the full Apple M4+ ARMv9.2 SME2 engine ISA
    /// (`FEAT_SME`, `FEAT_SME2`, `SME_B16F32`, `SME_F16F32`, `SVL == 64 B`).
    pub fn supportsSme2Engine(self: Facts) bool {
        return self.has_sme and
            self.has_sme2 and
            self.has_sme_b16f32 and
            self.has_sme_f16f32 and
            self.sme_max_svl_bytes == 64;
    }

    pub fn read() Error!Facts {
        const perf_logical = try readUnsigned("hw.perflevel0.logicalcpu");
        const pcore_phys = (try readOptionalUnsigned("hw.perflevel0.physicalcpu")) orelse perf_logical;
        const cpus_per_l2 = (try readOptionalUnsigned("hw.perflevel0.cpusperl2")) orelse pcore_phys;
        const pcore_l2 = (try readOptionalUnsigned("hw.perflevel0.l2cachesize")) orelse 0;
        const clusters: u64 = if (cpus_per_l2 > 0 and pcore_phys > 0)
            (pcore_phys + cpus_per_l2 - 1) / cpus_per_l2
        else
            1;

        return .{
            .page_size = try readUnsigned("hw.pagesize"),
            .memory_bytes = try readUnsigned("hw.memsize"),
            .performance_cores = perf_logical,
            .efficiency_cores = try readUnsigned("hw.perflevel1.logicalcpu"),
            .user_wire_limit = try readUnsigned("vm.user_wire_limit"),
            .global_user_wire_limit = try readUnsigned("vm.global_user_wire_limit"),
            .gpu_wired_limit_mib = try readUnsigned("iogpu.wired_limit_mb"),
            .pcore_physical_cpu = pcore_phys,
            .pcore_cpus_per_l2 = cpus_per_l2,
            .pcore_l2_cache_bytes = pcore_l2,
            .pcore_clusters = clusters,
            .has_sme = try readOptionalFlag("hw.optional.arm.FEAT_SME"),
            .has_sme2 = try readOptionalFlag("hw.optional.arm.FEAT_SME2"),
            .has_sme_b16f32 = try readOptionalFlagAny("hw.optional.arm.SME_B16F32", "hw.optional.arm.FEAT_SME_B16F32"),
            .has_sme_f16f32 = try readOptionalFlagAny("hw.optional.arm.SME_F16F32", "hw.optional.arm.FEAT_SME_F16F32"),
            .has_sme_f32f32 = try readOptionalFlagAny("hw.optional.arm.SME_F32F32", "hw.optional.arm.FEAT_SME_F32F32"),
            .has_sme_i16i64 = try readOptionalFlag("hw.optional.arm.FEAT_SME_I16I64"),
            .sme_max_svl_bytes = (try readOptionalUnsigned("hw.optional.arm.sme_max_svl_b")) orelse 0,
        };
    }
};

test "the kernel page size is the Apple Silicon page size" {
    const facts = try Facts.read();
    try std.testing.expectEqual(@as(u64, page.APPLE_PAGE_SIZE), facts.page_size);
}

test "memory, core counts and wire limits are reported" {
    const facts = try Facts.read();
    try std.testing.expect(facts.memory_bytes > 0);
    try std.testing.expect(facts.performance_cores > 0);
    try std.testing.expect(facts.efficiency_cores > 0);
    try std.testing.expect(facts.user_wire_limit > 0 and facts.user_wire_limit <= facts.memory_bytes);
    try std.testing.expect(facts.global_user_wire_limit > 0 and facts.global_user_wire_limit <= facts.memory_bytes);
}

test "an unknown sysctl name is an error, not a zero" {
    try std.testing.expectError(error.UnknownName, readUnsigned("aminfer.no_such_name"));
    try std.testing.expectEqual(@as(?u64, null), try readOptionalUnsigned("aminfer.no_such_name"));
    try std.testing.expectEqual(false, try readOptionalFlag("aminfer.no_such_name"));
}

test "P-core L2 cluster topology and SME/SME2 capability probes are consistent" {
    const facts = try Facts.read();
    try std.testing.expect(facts.pcore_physical_cpu > 0);
    try std.testing.expect(facts.pcore_cpus_per_l2 > 0);
    try std.testing.expect(facts.pcore_clusters >= 1);
    try std.testing.expect(facts.pcore_l2_cache_bytes >= 1024 * 1024);
    try std.testing.expectEqual(facts.pcore_l2_cache_bytes / 1024, facts.pcoreL2Kib());
    if (facts.has_sme2) {
        try std.testing.expect(facts.has_sme);
        try std.testing.expect(facts.sme_max_svl_bytes >= 64);
    }
    if (facts.supportsSme2Engine()) {
        try std.testing.expect(facts.has_sme and facts.has_sme2);
        try std.testing.expect(facts.has_sme_b16f32 and facts.has_sme_f16f32);
        try std.testing.expectEqual(@as(u64, 64), facts.sme_max_svl_bytes);
    }
}
