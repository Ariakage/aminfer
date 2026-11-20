//! Where an expert's weights live (ADR-0003). Kernels and the scheduler only
//! consume resident bytes; reading a cold expert into a slot is an explicit,
//! asynchronous step owned by the prefetcher.

const std = @import("std");
const page = @import("../mem/page.zig");

pub const ExpertSource = union(enum) {
    /// Resident in a wired region; already GPU-visible through that region's
    /// `MTLBuffer`.
    local_wired_ram: WiredExpert,
    /// A 16 KiB-aligned blob in an `.amf` file on the internal SSD; it must be
    /// read into a slot before use.
    local_nvme_ssd: NvmeExpert,
    /// Reserved for expert memory on a Thunderbolt-connected Mac (future).
    thunderbolt_peer: PeerExpert,

    pub const ResidencyError = error{
        /// The expert lives on NVMe and has not been read into a slot.
        NotResident,
        /// The Thunderbolt transport is reserved and not implemented.
        TransportUnavailable,
    };

    /// The expert's bytes, if they are resident in wired memory. Never loads.
    pub fn residentBytes(self: ExpertSource) ResidencyError![]align(page.APPLE_PAGE_SIZE) const u8 {
        return switch (self) {
            .local_wired_ram => |wired| wired.bytes,
            .local_nvme_ssd => error.NotResident,
            .thunderbolt_peer => error.TransportUnavailable,
        };
    }
};

pub const WiredExpert = struct {
    bytes: []align(page.APPLE_PAGE_SIZE) const u8,

    pub fn init(bytes: []align(page.APPLE_PAGE_SIZE) const u8) page.ShareError!WiredExpert {
        try page.checkShareable(bytes.ptr, bytes.len);
        return .{ .bytes = bytes };
    }
};

pub const NvmeExpert = struct {
    /// Index of the `.amf` file in the model's file table.
    file: u16,
    /// Byte offset of the blob in that file; a multiple of 16 KiB.
    offset: u64,
    /// Blob length; a non-zero multiple of 16 KiB.
    len: u64,

    pub const InitError = error{ ZeroLength, MisalignedOffset, MisalignedLength };

    pub fn init(file: u16, offset: u64, len: u64) InitError!NvmeExpert {
        if (len == 0) return error.ZeroLength;
        if (offset % page.APPLE_PAGE_SIZE != 0) return error.MisalignedOffset;
        if (len % page.APPLE_PAGE_SIZE != 0) return error.MisalignedLength;
        return .{ .file = file, .offset = offset, .len = len };
    }
};

/// Placeholder location on a Thunderbolt peer; no transport exists yet.
pub const PeerExpert = struct {
    peer: u16,
    remote_offset: u64,
    len: u64,
};

test "ExpertSource has exactly the three charter variants" {
    const names = @typeInfo(ExpertSource).@"union".field_names;
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expectEqualStrings("local_wired_ram", names[0]);
    try std.testing.expectEqualStrings("local_nvme_ssd", names[1]);
    try std.testing.expectEqualStrings("thunderbolt_peer", names[2]);
}

test "a wired expert exposes its resident bytes" {
    const region = try page.WiredRegion.init(2 * page.APPLE_PAGE_SIZE);
    defer region.deinit();
    const blob = region.bytes[page.APPLE_PAGE_SIZE..];
    const source: ExpertSource = .{ .local_wired_ram = try .init(blob) };
    const bytes = try source.residentBytes();
    try std.testing.expectEqual(blob.ptr, bytes.ptr);
    try std.testing.expectEqual(blob.len, bytes.len);
}

test "a wired expert must span whole pages" {
    const region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer region.deinit();
    try std.testing.expectError(error.MisalignedLength, WiredExpert.init(region.bytes[0..100]));
}

test "an NVMe expert is not resident" {
    const source: ExpertSource = .{ .local_nvme_ssd = try .init(0, page.APPLE_PAGE_SIZE, 2 * page.APPLE_PAGE_SIZE) };
    try std.testing.expectError(error.NotResident, source.residentBytes());
}

test "NVMe locations must be non-empty and 16 KiB aligned" {
    try std.testing.expectError(error.MisalignedOffset, NvmeExpert.init(0, 100, page.APPLE_PAGE_SIZE));
    try std.testing.expectError(error.MisalignedLength, NvmeExpert.init(0, 0, 100));
    try std.testing.expectError(error.ZeroLength, NvmeExpert.init(0, 0, 0));
}

test "the Thunderbolt peer source is reserved" {
    const source: ExpertSource = .{ .thunderbolt_peer = .{ .peer = 1, .remote_offset = 0, .len = page.APPLE_PAGE_SIZE } };
    try std.testing.expectError(error.TransportUnavailable, source.residentBytes());
}
