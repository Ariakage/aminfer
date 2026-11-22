//! Thunderbolt-peer transport layer (Milestone M7, capability X1, ADR-0003,
//! ADR-0015):
//! - Zero-allocation `AMI7` 32-byte fixed binary framing protocol with CRC32
//!   header & payload verification over non-blocking Darwin `kqueue` TCP
//!   (`TCP_NODELAY`, `SO_NOSIGPIPE`, `O_NONBLOCK`, `FD_CLOEXEC`).
//! - `16 KiB`-aligned (`APPLE_PAGE_SIZE = 16384`) wired send/receive slabs
//!   (`WiredSlab`, backed by `page.WiredRegion` or borrowed from a `WiredArena`).
//! - Runtime macOS 26.2+ Thunderbolt 5 RDMA probe (`probeTb5Rdma`) via dynamic
//!   `std.DynLib.openZ` / `dlopen` lookup of `RDMA.framework` / `libibverbs.dylib`
//!   + `ibv_get_device_list` with zero compile-time link dependencies, plus a
//!   runtime `TransportVTable` abstraction. On TB4 Macs without an RDMA HCA,
//!   `.auto` cleanly falls back to `.tb4_tcp` while explicit `.tb5_rdma` fails
//!   fast with `error.Tb5RdmaUnavailable`.

const std = @import("std");
const page = @import("../mem/page.zig");
const vmstat = @import("../platform/vmstat.zig");

pub const APPLE_PAGE_SIZE: usize = page.APPLE_PAGE_SIZE;

/// Wire magic `"AMI7"` encoded little-endian (`0x37494D41`).
pub const MAGIC: u32 = std.mem.readInt(u32, "AMI7", .little);
pub const PROTOCOL_VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 32;
pub const MAX_EXPERTS_BITMAP_WORDS: usize = 16;
pub const MAX_EXPERTS_BITMAP_BITS: usize = MAX_EXPERTS_BITMAP_WORDS * 64;
pub const DEFAULT_TIMEOUT_MS: u32 = 5_000;

/// Requested or resolved Thunderbolt cluster transport backend.
pub const TransportKind = enum(u8) {
    auto = 0,
    tb4_tcp = 1,
    tb5_rdma = 2,
};

/// Result of runtime probing for macOS 26.2+ Thunderbolt 5 RDMA (Apple TN3205).
pub const RdmaProbeResult = struct {
    available: bool,
    driver_lib_found: bool,
    device_count: u32,
    reason: []const u8,
};

/// Effective transport selected after probing hardware/OS support.
pub const ResolvedTransport = struct {
    kind: TransportKind,
    probe: RdmaProbeResult,
};

pub const TransportError = error{
    Tb5RdmaUnavailable,
    InvalidAddress,
    SocketFailed,
    FcntlFailed,
    SetSockOptFailed,
    GetSockOptFailed,
    BindFailed,
    ListenFailed,
    GetSockNameFailed,
    AcceptFailed,
    ConnectFailed,
    KqueueFailed,
    KeventFailed,
    SendFailed,
    RecvFailed,
    ConnectionReset,
    UnexpectedEof,
    Timeout,
    ClockFailed,
    BufferTooSmall,
    Overflow,
};

pub const ProtocolError = error{
    ShortFrame,
    BadMagic,
    UnsupportedVersion,
    InvalidMsgType,
    HeaderCrcMismatch,
    PayloadCrcMismatch,
    PayloadTooLarge,
    MalformedPayload,
};

pub const SlabError = page.WiredRegion.InitError || page.ShareError || error{
    InvalidAlignment,
    OutOfSlabSpace,
};

/// Dynamically probed `libibverbs` / `RDMA.framework` symbols (no link dep).
const IbvGetDeviceListFn = *const fn (num_devices: ?*c_int) callconv(.c) ?[*]?*anyopaque;
const IbvFreeDeviceListFn = *const fn (list: ?[*]?*anyopaque) callconv(.c) void;

/// Candidate dynamic library paths for macOS 26.2+ RDMA over Thunderbolt 5.
const rdma_candidate_paths = [_][:0]const u8{
    "/System/Library/Frameworks/RDMA.framework/RDMA",
    "libibverbs.dylib",
    "/usr/lib/libibverbs.dylib",
};

/// Probes the running macOS host for Thunderbolt 5 RDMA support without any
/// compile-time link dependency on `RDMA.framework` or `libibverbs`.
pub fn probeTb5Rdma() RdmaProbeResult {
    var any_lib_found = false;
    for (rdma_candidate_paths) |path| {
        var lib = std.DynLib.openZ(path) catch |err| switch (err) {
            error.FileNotFound => continue,
            error.AccessDenied, error.PermissionDenied => return .{
                .available = false,
                .driver_lib_found = true,
                .device_count = 0,
                .reason = "rdma_permission_denied",
            },
            else => continue,
        };
        defer lib.close();
        any_lib_found = true;

        const get_list = lib.lookup(IbvGetDeviceListFn, "ibv_get_device_list") orelse continue;
        const free_list = lib.lookup(IbvFreeDeviceListFn, "ibv_free_device_list");

        var num_devices: c_int = 0;
        const list = get_list(&num_devices);
        defer if (list != null) {
            if (free_list) |free_fn| free_fn(list);
        };

        if (list != null and num_devices > 0) {
            return .{
                .available = true,
                .driver_lib_found = true,
                .device_count = @intCast(num_devices),
                .reason = "tb5_rdma_ready",
            };
        }
        return .{
            .available = false,
            .driver_lib_found = true,
            .device_count = 0,
            .reason = "no_tb5_rdma_hca",
        };
    }

    if (any_lib_found) {
        return .{
            .available = false,
            .driver_lib_found = true,
            .device_count = 0,
            .reason = "no_tb5_rdma_hca",
        };
    }

    return .{
        .available = false,
        .driver_lib_found = false,
        .device_count = 0,
        .reason = "rdma_framework_absent",
    };
}

/// Pure selection helper taking an explicit `RdmaProbeResult` (deterministic in tests).
pub fn selectTransportWithProbe(
    requested: TransportKind,
    probe: RdmaProbeResult,
) error{Tb5RdmaUnavailable}!ResolvedTransport {
    return switch (requested) {
        .tb4_tcp => .{ .kind = .tb4_tcp, .probe = probe },
        .auto => .{
            .kind = if (probe.available) .tb5_rdma else .tb4_tcp,
            .probe = probe,
        },
        .tb5_rdma => blk: {
            if (!probe.available) return error.Tb5RdmaUnavailable;
            break :blk .{ .kind = .tb5_rdma, .probe = probe };
        },
    };
}

/// Resolves `requested` against the live host's TB5 RDMA probe.
pub fn selectTransport(requested: TransportKind) error{Tb5RdmaUnavailable}!ResolvedTransport {
    if (requested == .tb4_tcp) {
        return .{
            .kind = .tb4_tcp,
            .probe = .{
                .available = false,
                .driver_lib_found = false,
                .device_count = 0,
                .reason = "tb4_tcp_explicit",
            },
        };
    }
    return selectTransportWithProbe(requested, probeTb5Rdma());
}

// ---------------------------------------------------------------------------
// AMI7 Binary Wire Protocol (zero-allocation fixed 32-byte header)
// ---------------------------------------------------------------------------

pub const MsgType = enum(u8) {
    hello_req = 1,
    hello_resp = 2,
    eval_experts_req = 3,
    eval_experts_resp = 4,
    ping = 5,
    pong = 6,
    shutdown = 7,

    pub fn fromByte(raw: u8) ProtocolError!MsgType {
        return switch (raw) {
            1 => .hello_req,
            2 => .hello_resp,
            3 => .eval_experts_req,
            4 => .eval_experts_resp,
            5 => .ping,
            6 => .pong,
            7 => .shutdown,
            else => error.InvalidMsgType,
        };
    }
};

/// 32-byte fixed wire header (`AMI7` v1) with CRC32 over header and payload.
pub const FrameHeader = extern struct {
    magic: u32 = MAGIC,
    version: u16 = PROTOCOL_VERSION,
    msg_type: MsgType,
    flags: u8 = 0,
    seq_id: u64 = 0,
    payload_len: u32 = 0,
    header_crc32: u32 = 0,
    payload_crc32: u32 = 0,
    reserved: u32 = 0,

    pub fn computeHeaderCrc32(self: FrameHeader) u32 {
        var buf: [HEADER_BYTES]u8 = undefined;
        writeHeaderFields(self, 0, &buf);
        return std.hash.Crc32.hash(&buf);
    }

    /// Serializes `self` into `out` in little-endian wire order with a freshly
    /// computed `header_crc32`, returning the finalized header.
    pub fn encode(self: FrameHeader, out: *[HEADER_BYTES]u8) FrameHeader {
        var copy = self;
        copy.magic = MAGIC;
        copy.version = PROTOCOL_VERSION;
        copy.header_crc32 = copy.computeHeaderCrc32();
        writeHeaderFields(copy, copy.header_crc32, out);
        return copy;
    }

    /// Validates magic, protocol version, message type, and header CRC32 from
    /// raw wire bytes.
    pub fn decode(raw: *const [HEADER_BYTES]u8) ProtocolError!FrameHeader {
        const magic = std.mem.readInt(u32, raw[0..4], .little);
        if (magic != MAGIC) return error.BadMagic;
        const version = std.mem.readInt(u16, raw[4..6], .little);
        if (version != PROTOCOL_VERSION) return error.UnsupportedVersion;
        const msg_type = try MsgType.fromByte(raw[6]);
        const flags = raw[7];
        const seq_id = std.mem.readInt(u64, raw[8..16], .little);
        const payload_len = std.mem.readInt(u32, raw[16..20], .little);
        const header_crc32 = std.mem.readInt(u32, raw[20..24], .little);
        const payload_crc32 = std.mem.readInt(u32, raw[24..28], .little);
        const reserved = std.mem.readInt(u32, raw[28..32], .little);

        const hdr: FrameHeader = .{
            .magic = magic,
            .version = version,
            .msg_type = msg_type,
            .flags = flags,
            .seq_id = seq_id,
            .payload_len = payload_len,
            .header_crc32 = header_crc32,
            .payload_crc32 = payload_crc32,
            .reserved = reserved,
        };
        if (hdr.computeHeaderCrc32() != header_crc32) return error.HeaderCrcMismatch;
        return hdr;
    }

    fn writeHeaderFields(self: FrameHeader, crc: u32, out: *[HEADER_BYTES]u8) void {
        std.mem.writeInt(u32, out[0..4], self.magic, .little);
        std.mem.writeInt(u16, out[4..6], self.version, .little);
        out[6] = @backingInt(self.msg_type);
        out[7] = self.flags;
        std.mem.writeInt(u64, out[8..16], self.seq_id, .little);
        std.mem.writeInt(u32, out[16..20], self.payload_len, .little);
        std.mem.writeInt(u32, out[20..24], crc, .little);
        std.mem.writeInt(u32, out[24..28], self.payload_crc32, .little);
        std.mem.writeInt(u32, out[28..32], self.reserved, .little);
    }
};

/// Handshake payload carried by `hello_req` and `hello_resp`.
/// Includes a 1024-bit `expert_bitmap` (covering up to 1024 experts for both
/// contiguous and round-robin partitions) plus contiguous range bounds.
pub const HelloPayload = extern struct {
    expert_bitmap: [MAX_EXPERTS_BITMAP_WORDS]u64 = @splat(0),
    hidden_size: u32 = 0,
    rank: u16 = 0,
    world_size: u16 = 0,
    num_layers: u16 = 0,
    num_experts: u16 = 0,
    expert_range_start: u16 = 0,
    expert_range_end: u16 = 0,
    transport_kind: u8 = @backingInt(TransportKind.tb4_tcp),
    status: u8 = 0,
    reserved0: u16 = 0,
    reserved1: u32 = 0,

    pub const WIRE_BYTES: usize = 152;

    pub fn setExpert(self: *HelloPayload, expert_id: u16) void {
        if (expert_id >= MAX_EXPERTS_BITMAP_BITS) return;
        const word_idx: usize = expert_id / 64;
        const bit_idx: u6 = @intCast(expert_id % 64);
        self.expert_bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
    }

    pub fn hasExpert(self: *const HelloPayload, expert_id: u16) bool {
        if (expert_id >= MAX_EXPERTS_BITMAP_BITS) return false;
        const word_idx: usize = expert_id / 64;
        const bit_idx: u6 = @intCast(expert_id % 64);
        return (self.expert_bitmap[word_idx] & (@as(u64, 1) << bit_idx)) != 0;
    }

    pub fn encode(self: *const HelloPayload, out: *[WIRE_BYTES]u8) void {
        for (self.expert_bitmap, 0..) |w, i| {
            std.mem.writeInt(u64, out[i * 8 ..][0..8], w, .little);
        }
        std.mem.writeInt(u32, out[128..132], self.hidden_size, .little);
        std.mem.writeInt(u16, out[132..134], self.rank, .little);
        std.mem.writeInt(u16, out[134..136], self.world_size, .little);
        std.mem.writeInt(u16, out[136..138], self.num_layers, .little);
        std.mem.writeInt(u16, out[138..140], self.num_experts, .little);
        std.mem.writeInt(u16, out[140..142], self.expert_range_start, .little);
        std.mem.writeInt(u16, out[142..144], self.expert_range_end, .little);
        out[144] = self.transport_kind;
        out[145] = self.status;
        std.mem.writeInt(u16, out[146..148], self.reserved0, .little);
        std.mem.writeInt(u32, out[148..152], self.reserved1, .little);
    }

    pub fn decode(raw: []const u8) ProtocolError!HelloPayload {
        if (raw.len != WIRE_BYTES) return error.MalformedPayload;
        var p: HelloPayload = .{};
        for (&p.expert_bitmap, 0..) |*w, i| {
            w.* = std.mem.readInt(u64, raw[i * 8 ..][0..8], .little);
        }
        p.hidden_size = std.mem.readInt(u32, raw[128..132], .little);
        p.rank = std.mem.readInt(u16, raw[132..134], .little);
        p.world_size = std.mem.readInt(u16, raw[134..136], .little);
        p.num_layers = std.mem.readInt(u16, raw[136..138], .little);
        p.num_experts = std.mem.readInt(u16, raw[138..140], .little);
        p.expert_range_start = std.mem.readInt(u16, raw[140..142], .little);
        p.expert_range_end = std.mem.readInt(u16, raw[142..144], .little);
        p.transport_kind = raw[144];
        p.status = raw[145];
        p.reserved0 = std.mem.readInt(u16, raw[146..148], .little);
        p.reserved1 = std.mem.readInt(u32, raw[148..152], .little);
        return p;
    }
};

/// Packed routed expert assignment `(token_idx, expert_id, weight)` (8 bytes).
pub const Assignment = extern struct {
    token_idx: u16,
    expert_id: u16,
    weight: f32,
};

/// Fixed 16-byte prefix of `eval_experts_req` payload.
/// Followed by `[num_assignments]Assignment` and `[num_tokens * hidden_size]f16`.
pub const EvalExpertsReqHeader = extern struct {
    layer_idx: u16,
    num_tokens: u16,
    hidden_size: u32,
    num_assignments: u32,
    flags: u32 = 0,

    pub const WIRE_BYTES: usize = 16;

    pub fn encode(self: EvalExpertsReqHeader, out: *[WIRE_BYTES]u8) void {
        std.mem.writeInt(u16, out[0..2], self.layer_idx, .little);
        std.mem.writeInt(u16, out[2..4], self.num_tokens, .little);
        std.mem.writeInt(u32, out[4..8], self.hidden_size, .little);
        std.mem.writeInt(u32, out[8..12], self.num_assignments, .little);
        std.mem.writeInt(u32, out[12..16], self.flags, .little);
    }

    pub fn decode(raw: []const u8) ProtocolError!EvalExpertsReqHeader {
        if (raw.len < WIRE_BYTES) return error.MalformedPayload;
        return .{
            .layer_idx = std.mem.readInt(u16, raw[0..2], .little),
            .num_tokens = std.mem.readInt(u16, raw[2..4], .little),
            .hidden_size = std.mem.readInt(u32, raw[4..8], .little),
            .num_assignments = std.mem.readInt(u32, raw[8..12], .little),
            .flags = std.mem.readInt(u32, raw[12..16], .little),
        };
    }
};

pub const EvalStatus = enum(u32) {
    ok = 0,
    invalid_layer = 1,
    invalid_shape = 2,
    invalid_expert = 3,
    buffer_too_small = 4,
    backend_failed = 5,
};

/// Fixed 24-byte prefix of `eval_experts_resp` payload.
/// Followed by `[num_tokens * hidden_size]f32` accumulated outputs when `status == 0`.
pub const EvalExpertsRespHeader = extern struct {
    status: u32 = 0,
    num_tokens: u16 = 0,
    reserved: u16 = 0,
    hidden_size: u32 = 0,
    assignments_evaluated: u32 = 0,
    peer_compute_ns: u64 = 0,

    pub const WIRE_BYTES: usize = 24;

    pub fn encode(self: EvalExpertsRespHeader, out: *[WIRE_BYTES]u8) void {
        std.mem.writeInt(u32, out[0..4], self.status, .little);
        std.mem.writeInt(u16, out[4..6], self.num_tokens, .little);
        std.mem.writeInt(u16, out[6..8], self.reserved, .little);
        std.mem.writeInt(u32, out[8..12], self.hidden_size, .little);
        std.mem.writeInt(u32, out[12..16], self.assignments_evaluated, .little);
        std.mem.writeInt(u64, out[16..24], self.peer_compute_ns, .little);
    }

    pub fn decode(raw: []const u8) ProtocolError!EvalExpertsRespHeader {
        if (raw.len < WIRE_BYTES) return error.MalformedPayload;
        return .{
            .status = std.mem.readInt(u32, raw[0..4], .little),
            .num_tokens = std.mem.readInt(u16, raw[4..6], .little),
            .reserved = std.mem.readInt(u16, raw[6..8], .little),
            .hidden_size = std.mem.readInt(u32, raw[8..12], .little),
            .assignments_evaluated = std.mem.readInt(u32, raw[12..16], .little),
            .peer_compute_ns = std.mem.readInt(u64, raw[16..24], .little),
        };
    }
};

comptime {
    if (@sizeOf(FrameHeader) != HEADER_BYTES) @compileError("FrameHeader must be 32 bytes");
    if (@sizeOf(HelloPayload) != HelloPayload.WIRE_BYTES) @compileError("HelloPayload must be 152 bytes");
    if (@sizeOf(Assignment) != 8) @compileError("Assignment must be 8 bytes");
    if (@sizeOf(EvalExpertsReqHeader) != EvalExpertsReqHeader.WIRE_BYTES) @compileError("EvalExpertsReqHeader must be 16 bytes");
    if (@sizeOf(EvalExpertsRespHeader) != EvalExpertsRespHeader.WIRE_BYTES) @compileError("EvalExpertsRespHeader must be 24 bytes");
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("aminfer cluster wire views assume little-endian Apple Silicon");
    }
}

pub fn evalReqPayloadBytes(num_tokens: u16, hidden_size: u32, num_assignments: u32) error{Overflow}!usize {
    const assign_bytes = try std.math.mul(usize, num_assignments, @sizeOf(Assignment));
    const hidden_elems = try std.math.mul(usize, num_tokens, hidden_size);
    const hidden_bytes = try std.math.mul(usize, hidden_elems, @sizeOf(f16));
    const total1 = try std.math.add(usize, EvalExpertsReqHeader.WIRE_BYTES, assign_bytes);
    return try std.math.add(usize, total1, hidden_bytes);
}

pub fn evalRespPayloadBytes(num_tokens: u16, hidden_size: u32) error{Overflow}!usize {
    const out_elems = try std.math.mul(usize, num_tokens, hidden_size);
    const out_bytes = try std.math.mul(usize, out_elems, @sizeOf(f32));
    return try std.math.add(usize, EvalExpertsRespHeader.WIRE_BYTES, out_bytes);
}

/// Encodes an `eval_experts_req` payload directly into `dst` with zero heap allocation.
pub fn encodeEvalReqInto(
    dst: []u8,
    layer_idx: u16,
    num_tokens: u16,
    hidden_size: u32,
    assignments: []const Assignment,
    hidden_f16: []const f16,
) (error{ BufferTooSmall, Overflow })!usize {
    const num_assignments: u32 = std.math.cast(u32, assignments.len) orelse return error.Overflow;
    const expected_elems = try std.math.mul(usize, num_tokens, hidden_size);
    if (hidden_f16.len < expected_elems) return error.BufferTooSmall;
    const total = try evalReqPayloadBytes(num_tokens, hidden_size, num_assignments);
    if (dst.len < total) return error.BufferTooSmall;

    const hdr: EvalExpertsReqHeader = .{
        .layer_idx = layer_idx,
        .num_tokens = num_tokens,
        .hidden_size = hidden_size,
        .num_assignments = num_assignments,
        .flags = 0,
    };
    hdr.encode(dst[0..EvalExpertsReqHeader.WIRE_BYTES]);

    const assign_bytes = std.mem.sliceAsBytes(assignments);
    const assign_off = EvalExpertsReqHeader.WIRE_BYTES;
    @memcpy(dst[assign_off .. assign_off + assign_bytes.len], assign_bytes);

    const hidden_off = assign_off + assign_bytes.len;
    const hidden_bytes = std.mem.sliceAsBytes(hidden_f16[0..expected_elems]);
    @memcpy(dst[hidden_off .. hidden_off + hidden_bytes.len], hidden_bytes);
    return total;
}

/// Zero-copy view of a validated `eval_experts_req` payload inside an aligned RX slab.
pub const EvalReqView = struct {
    header: EvalExpertsReqHeader,
    assignments: []const Assignment,
    hidden_f16: []const f16,
};

pub fn decodeEvalReqView(payload: []align(8) const u8) ProtocolError!EvalReqView {
    const hdr = try EvalExpertsReqHeader.decode(payload);
    const expected_len = evalReqPayloadBytes(hdr.num_tokens, hdr.hidden_size, hdr.num_assignments) catch {
        return error.MalformedPayload;
    };
    if (payload.len != expected_len) return error.MalformedPayload;

    const assign_off = EvalExpertsReqHeader.WIRE_BYTES;
    const assign_bytes = @as(usize, hdr.num_assignments) * @sizeOf(Assignment);
    const hidden_off = assign_off + assign_bytes;
    const hidden_bytes = @as(usize, hdr.num_tokens) * @as(usize, hdr.hidden_size) * @sizeOf(f16);

    const assign_slice: []align(4) const u8 = @alignCast(payload[assign_off .. assign_off + assign_bytes]);
    const hidden_slice: []align(2) const u8 = @alignCast(payload[hidden_off .. hidden_off + hidden_bytes]);

    return .{
        .header = hdr,
        .assignments = std.mem.bytesAsSlice(Assignment, assign_slice),
        .hidden_f16 = std.mem.bytesAsSlice(f16, hidden_slice),
    };
}

/// Zero-copy view of a validated `eval_experts_resp` payload inside an aligned RX slab.
pub const EvalRespView = struct {
    header: EvalExpertsRespHeader,
    accum_out_f32: []const f32,
};

pub fn decodeEvalRespView(payload: []align(8) const u8) ProtocolError!EvalRespView {
    const hdr = try EvalExpertsRespHeader.decode(payload);
    if (hdr.status != @backingInt(EvalStatus.ok)) {
        if (payload.len != EvalExpertsRespHeader.WIRE_BYTES) return error.MalformedPayload;
        return .{ .header = hdr, .accum_out_f32 = &.{} };
    }
    const expected_len = evalRespPayloadBytes(hdr.num_tokens, hdr.hidden_size) catch {
        return error.MalformedPayload;
    };
    if (payload.len != expected_len) return error.MalformedPayload;
    const out_slice: []align(4) const u8 = @alignCast(payload[EvalExpertsRespHeader.WIRE_BYTES..]);
    return .{
        .header = hdr,
        .accum_out_f32 = std.mem.bytesAsSlice(f32, out_slice),
    };
}

// ---------------------------------------------------------------------------
// 16 KiB-aligned WiredSlab (owned WiredRegion or borrowed WiredArena pages)
// ---------------------------------------------------------------------------

/// Linear bumper allocator over a single `16 KiB`-aligned, `mlock`ed region.
/// Startup carving only; zero dynamic heap allocations during steady state.
pub const WiredSlab = struct {
    bytes: []align(APPLE_PAGE_SIZE) u8,
    used: usize,
    owns_region: bool,

    pub fn initOwned(min_bytes: usize) SlabError!WiredSlab {
        const region = try page.WiredRegion.init(min_bytes);
        return .{
            .bytes = region.bytes,
            .used = 0,
            .owns_region = true,
        };
    }

    pub fn initBorrowed(slice: []align(APPLE_PAGE_SIZE) u8) SlabError!WiredSlab {
        try page.checkShareable(slice.ptr, slice.len);
        return .{
            .bytes = slice,
            .used = 0,
            .owns_region = false,
        };
    }

    pub fn deinit(self: *WiredSlab) void {
        if (self.owns_region) {
            const region: page.WiredRegion = .{ .bytes = self.bytes };
            region.deinit();
            self.owns_region = false;
        }
        self.used = 0;
    }

    pub fn carvePages(self: *WiredSlab, min_len: usize) SlabError![]align(APPLE_PAGE_SIZE) u8 {
        if (min_len == 0) return error.ZeroLength;
        const aligned_len = try page.alignUp(min_len);
        const start = try page.alignUp(self.used);
        const end = try std.math.add(usize, start, aligned_len);
        if (end > self.bytes.len) return error.OutOfSlabSpace;
        self.used = end;
        return @alignCast(self.bytes[start..end]);
    }

    pub fn carveAligned(self: *WiredSlab, len: usize, alignment: usize) SlabError![]u8 {
        if (len == 0) return error.ZeroLength;
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment) or alignment > APPLE_PAGE_SIZE) {
            return error.InvalidAlignment;
        }
        const mask = alignment - 1;
        const padded = try std.math.add(usize, self.used, mask);
        const start = padded & ~mask;
        const end = try std.math.add(usize, start, len);
        if (end > self.bytes.len) return error.OutOfSlabSpace;
        self.used = end;
        return self.bytes[start..end];
    }
};

// ---------------------------------------------------------------------------
// Darwin Socket, IPv4 & kqueue Primitives
// ---------------------------------------------------------------------------

pub const Endpoint = struct {
    host: [4]u8 = .{ 127, 0, 0, 1 },
    port: u16 = 0,
};

/// Parses a dotted IPv4 address (e.g. `"127.0.0.1"`, `"169.254.10.2"`) with
/// zero heap allocation.
pub fn parseIpv4(text: []const u8) error{InvalidAddress}![4]u8 {
    var octets: [4]u8 = undefined;
    var octet_idx: usize = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |part| {
        if (octet_idx >= 4 or part.len == 0 or part.len > 3) return error.InvalidAddress;
        const val = std.fmt.parseInt(u8, part, 10) catch return error.InvalidAddress;
        octets[octet_idx] = val;
        octet_idx += 1;
    }
    if (octet_idx != 4) return error.InvalidAddress;
    return octets;
}

pub fn setCloexec(fd: std.c.fd_t) TransportError!void {
    if (std.c.fcntl(fd, std.c.F.SETFD, @as(c_int, std.c.FD_CLOEXEC)) != 0) {
        return error.FcntlFailed;
    }
}

pub fn setNonBlocking(fd: std.c.fd_t) TransportError!void {
    const raw = std.c.fcntl(fd, std.c.F.GETFL, @as(c_int, 0));
    if (raw < 0) return error.FcntlFailed;
    var flags: std.c.O = @bitCast(@as(u32, @intCast(raw)));
    flags.NONBLOCK = true;
    const updated: c_int = @intCast(@as(u32, @bitCast(flags)));
    if (std.c.fcntl(fd, std.c.F.SETFL, updated) != 0) {
        return error.FcntlFailed;
    }
}

pub fn setSockOptInt(fd: std.c.fd_t, level: i32, optname: u32, value: c_int) TransportError!void {
    var v = value;
    if (std.c.setsockopt(fd, level, optname, &v, @sizeOf(c_int)) != 0) {
        return error.SetSockOptFailed;
    }
}

/// Applies mandatory aminfer TCP socket invariants: `FD_CLOEXEC`, `O_NONBLOCK`,
/// Darwin `SO_NOSIGPIPE`, and `TCP_NODELAY`.
pub fn configureTcpSocket(fd: std.c.fd_t) TransportError!void {
    try setCloexec(fd);
    try setNonBlocking(fd);
    try setSockOptInt(fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, 1);
    try setSockOptInt(fd, std.c.IPPROTO.TCP, std.c.TCP.NODELAY, 1);
}

pub fn nsToTimespec(timeout_ns: u64) std.c.timespec {
    const secs = timeout_ns / std.time.ns_per_s;
    const rem_ns = timeout_ns % std.time.ns_per_s;
    const max_sec: u64 = @intCast(std.math.maxInt(isize));
    return .{
        .sec = @intCast(@min(secs, max_sec)),
        .nsec = @intCast(rem_ns),
    };
}

pub fn kqueueCreate() TransportError!std.c.fd_t {
    const kq = std.c.kqueue();
    if (kq < 0) return error.KqueueFailed;
    errdefer _ = std.c.close(kq);
    try setCloexec(kq);
    return kq;
}

pub fn kqueueRegisterUser(kq_fd: std.c.fd_t, ident: u64, udata: u64) TransportError!void {
    const change = [1]std.c.kevent64_s{.{
        .ident = ident,
        .filter = std.c.EVFILT.USER,
        .flags = std.c.EV.ADD | std.c.EV.CLEAR,
        .fflags = 0,
        .data = 0,
        .udata = udata,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
    if (rc < 0) return error.KeventFailed;
}

pub fn kqueueTriggerUser(kq_fd: std.c.fd_t, ident: u64, udata: u64) void {
    const change = [1]std.c.kevent64_s{.{
        .ident = ident,
        .filter = std.c.EVFILT.USER,
        .flags = 0,
        .fflags = std.c.NOTE.TRIGGER,
        .data = 0,
        .udata = udata,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    while (true) {
        const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        if (rc < 0 and std.c.errno(rc) == .INTR) continue;
        return;
    }
}

pub fn kqueueSetFilter(
    kq_fd: std.c.fd_t,
    fd: std.c.fd_t,
    filter: i16,
    flags: u16,
    udata: u64,
) TransportError!void {
    const change = [1]std.c.kevent64_s{.{
        .ident = @intCast(fd),
        .filter = filter,
        .flags = flags,
        .fflags = 0,
        .data = 0,
        .udata = udata,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    while (true) {
        const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.KeventFailed;
        }
        return;
    }
}

pub fn kqueueWait(
    kq_fd: std.c.fd_t,
    out_events: []std.c.kevent64_s,
    timeout_ns: ?u64,
) TransportError!usize {
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    var ts: std.c.timespec = undefined;
    const ts_ptr: ?*const std.c.timespec = if (timeout_ns) |ns| blk: {
        ts = nsToTimespec(ns);
        break :blk &ts;
    } else null;

    while (true) {
        const rc = std.c.kevent64(
            kq_fd,
            &empty,
            0,
            out_events.ptr,
            @intCast(out_events.len),
            std.c.KEVENT.FLAG.NONE,
            ts_ptr,
        );
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.KeventFailed;
        }
        return @intCast(rc);
    }
}

/// Non-blocking TCP listener bound with `SO_REUSEADDR`, `SO_NOSIGPIPE`, and `TCP_NODELAY`.
pub const TcpListener = struct {
    fd: std.c.fd_t,
    bound_port: u16,

    pub fn bind(endpoint: Endpoint, backlog: u31) TransportError!TcpListener {
        const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        errdefer _ = std.c.close(fd);

        try configureTcpSocket(fd);
        try setSockOptInt(fd, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, 1);

        var addr = std.c.sockaddr.in{
            .port = std.mem.nativeToBig(u16, endpoint.port),
            .addr = @bitCast(endpoint.host),
        };
        if (std.c.bind(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
            return error.BindFailed;
        }
        if (std.c.listen(fd, @as(c_uint, backlog)) != 0) {
            return error.ListenFailed;
        }

        var bound_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
        var bound_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
        if (std.c.getsockname(fd, @ptrCast(&bound_addr), &bound_len) != 0) {
            return error.GetSockNameFailed;
        }
        return .{
            .fd = fd,
            .bound_port = std.mem.bigToNative(u16, bound_addr.port),
        };
    }

    pub fn acceptNonBlocking(self: *const TcpListener) TransportError!?std.c.fd_t {
        while (true) {
            const client_fd = std.c.accept(self.fd, null, null);
            if (client_fd < 0) {
                const err = std.c.errno(client_fd);
                if (err == .INTR) continue;
                if (err == .AGAIN) return null;
                return error.AcceptFailed;
            }
            errdefer _ = std.c.close(client_fd);
            try configureTcpSocket(client_fd);
            return client_fd;
        }
    }

    pub fn close(self: *TcpListener) void {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
    }
};

/// Connects non-blockingly to `endpoint` with `TCP_NODELAY` and `SO_NOSIGPIPE`,
/// bounded by `timeout_ms` via a dedicated `kqueue` wait.
pub fn connectNonBlocking(endpoint: Endpoint, timeout_ms: u32) TransportError!std.c.fd_t {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);

    try configureTcpSocket(fd);

    var addr = std.c.sockaddr.in{
        .port = std.mem.nativeToBig(u16, endpoint.port),
        .addr = @bitCast(endpoint.host),
    };
    const rc = std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in));
    if (rc == 0) return fd;

    const err = std.c.errno(rc);
    if (err != .INPROGRESS and err != .AGAIN) {
        return error.ConnectFailed;
    }

    const kq = try kqueueCreate();
    defer _ = std.c.close(kq);
    try kqueueSetFilter(kq, fd, std.c.EVFILT.WRITE, std.c.EV.ADD | std.c.EV.ENABLE | std.c.EV.ONESHOT, 1);

    const deadline_ns = (vmstat.nowNs() catch return error.ClockFailed) +
        @as(u64, timeout_ms) * std.time.ns_per_ms;

    while (true) {
        const now = vmstat.nowNs() catch return error.ClockFailed;
        if (now >= deadline_ns) return error.Timeout;
        var evs: [1]std.c.kevent64_s = undefined;
        const n = try kqueueWait(kq, &evs, deadline_ns - now);
        if (n == 0) return error.Timeout;

        var so_err: c_int = 0;
        var so_len: std.c.socklen_t = @sizeOf(c_int);
        if (std.c.getsockopt(fd, std.c.SOL.SOCKET, std.c.SO.ERROR, &so_err, &so_len) != 0) {
            return error.GetSockOptFailed;
        }
        if (so_err != 0) return error.ConnectFailed;
        return fd;
    }
}

// ---------------------------------------------------------------------------
// Zero-allocation Non-blocking FrameChannel & TransportVTable
// ---------------------------------------------------------------------------

pub const SendStep = union(enum) {
    /// Entire queued frame was handed to the kernel socket buffer.
    complete: usize,
    /// Socket send buffer is full (`EAGAIN`); caller should arm `EVFILT_WRITE`.
    would_block: usize,
};

pub const ReceivedFrame = struct {
    header: FrameHeader,
    payload: []align(8) const u8,
    bytes_read: usize,
};

pub const RecvStep = union(enum) {
    frame_ready: ReceivedFrame,
    would_block: usize,
    peer_closed: void,
};

/// Runtime transport vtable abstracting frame send/recv across TB4 TCP and
/// future macOS 26.2+ TB5 RDMA verbs implementations without refactoring callers.
pub const TransportVTable = struct {
    ctx: *anyopaque,
    kind: TransportKind,
    sendSyncFn: *const fn (ctx: *anyopaque, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!u64,
    recvSyncFn: *const fn (ctx: *anyopaque, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!ReceivedFrame,
    closeFn: *const fn (ctx: *anyopaque) void,

    pub fn sendSync(self: TransportVTable, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!u64 {
        return self.sendSyncFn(self.ctx, kq_fd, timeout_ms);
    }

    pub fn recvSync(self: TransportVTable, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!ReceivedFrame {
        return self.recvSyncFn(self.ctx, kq_fd, timeout_ms);
    }

    pub fn close(self: TransportVTable) void {
        self.closeFn(self.ctx);
    }
};

/// Non-blocking TCP framed channel backed by preallocated `16 KiB`-aligned
/// `tx_buf` and `rx_buf` slices. Performs zero heap allocations and zero
/// extra payload copies on the hot path.
pub const FrameChannel = struct {
    fd: std.c.fd_t,
    tx_buf: []align(APPLE_PAGE_SIZE) u8,
    rx_buf: []align(APPLE_PAGE_SIZE) u8,
    tx_len: usize = 0,
    tx_sent: usize = 0,
    rx_len: usize = 0,
    rx_expected: usize = 0,
    rx_header: ?FrameHeader = null,
    write_armed: bool = false,

    pub fn init(
        fd: std.c.fd_t,
        tx_buf: []align(APPLE_PAGE_SIZE) u8,
        rx_buf: []align(APPLE_PAGE_SIZE) u8,
    ) page.ShareError!FrameChannel {
        try page.checkShareable(tx_buf.ptr, tx_buf.len);
        try page.checkShareable(rx_buf.ptr, rx_buf.len);
        return .{
            .fd = fd,
            .tx_buf = tx_buf,
            .rx_buf = rx_buf,
        };
    }

    pub fn resetState(self: *FrameChannel) void {
        self.tx_len = 0;
        self.tx_sent = 0;
        self.rx_len = 0;
        self.rx_expected = 0;
        self.rx_header = null;
        self.write_armed = false;
    }

    pub fn resetRx(self: *FrameChannel) void {
        self.rx_len = 0;
        self.rx_expected = 0;
        self.rx_header = null;
    }

    pub fn hasPendingTx(self: *const FrameChannel) bool {
        return self.tx_sent < self.tx_len;
    }

    pub fn close(self: *FrameChannel) void {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
        self.resetState();
    }

    /// Reserves an 8-byte-aligned payload region inside `tx_buf` directly after
    /// the 32-byte frame header so callers encode payloads with zero extra copy.
    pub fn preparePayload(self: *FrameChannel, payload_len: usize) error{ BufferTooSmall, Overflow }![]align(8) u8 {
        const total = try std.math.add(usize, HEADER_BYTES, payload_len);
        if (total > self.tx_buf.len) return error.BufferTooSmall;
        const slice: []align(8) u8 = @alignCast(self.tx_buf[HEADER_BYTES .. HEADER_BYTES + payload_len]);
        return slice;
    }

    /// Finalizes the 32-byte `AMI7` header (with payload & header CRC32) for a
    /// payload previously written into `tx_buf[HEADER_BYTES..]`.
    pub fn commitFrame(
        self: *FrameChannel,
        msg_type: MsgType,
        flags: u8,
        seq_id: u64,
        payload_len: usize,
    ) FrameHeader {
        const payload = self.tx_buf[HEADER_BYTES .. HEADER_BYTES + payload_len];
        const raw_hdr: FrameHeader = .{
            .msg_type = msg_type,
            .flags = flags,
            .seq_id = seq_id,
            .payload_len = @intCast(payload_len),
            .payload_crc32 = std.hash.Crc32.hash(payload),
        };
        const encoded = raw_hdr.encode(self.tx_buf[0..HEADER_BYTES]);
        self.tx_len = HEADER_BYTES + payload_len;
        self.tx_sent = 0;
        return encoded;
    }

    /// Non-blocking drain of queued TX bytes.
    pub fn stepSend(self: *FrameChannel) TransportError!SendStep {
        var written: usize = 0;
        while (self.tx_sent < self.tx_len) {
            const rem = self.tx_buf[self.tx_sent..self.tx_len];
            const rc = std.c.send(self.fd, rem.ptr, rem.len, 0);
            if (rc < 0) {
                const err = std.c.errno(rc);
                if (err == .INTR) continue;
                if (err == .AGAIN) return .{ .would_block = written };
                if (err == .PIPE or err == .CONNRESET) return error.ConnectionReset;
                return error.SendFailed;
            }
            if (rc == 0) return error.ConnectionReset;
            const n: usize = @intCast(rc);
            self.tx_sent += n;
            written += n;
        }
        self.tx_len = 0;
        self.tx_sent = 0;
        return .{ .complete = written };
    }

    /// Non-blocking read advancing at most one complete `AMI7` frame boundary
    /// so back-to-back frames never mix or require buffer compaction.
    pub fn stepRecv(self: *FrameChannel) (TransportError || ProtocolError)!RecvStep {
        var total_read: usize = 0;
        while (true) {
            if (self.rx_header == null) {
                while (self.rx_len < HEADER_BYTES) {
                    const dst = self.rx_buf[self.rx_len..HEADER_BYTES];
                    const rc = std.c.recv(self.fd, dst.ptr, dst.len, 0);
                    if (rc < 0) {
                        const err = std.c.errno(rc);
                        if (err == .INTR) continue;
                        if (err == .AGAIN) return .{ .would_block = total_read };
                        if (err == .CONNRESET) return error.ConnectionReset;
                        return error.RecvFailed;
                    }
                    if (rc == 0) {
                        if (self.rx_len == 0 and total_read == 0) return .peer_closed;
                        return error.UnexpectedEof;
                    }
                    const n: usize = @intCast(rc);
                    self.rx_len += n;
                    total_read += n;
                }
                const hdr = try FrameHeader.decode(self.rx_buf[0..HEADER_BYTES]);
                const total_frame = std.math.add(usize, HEADER_BYTES, hdr.payload_len) catch {
                    return error.PayloadTooLarge;
                };
                if (total_frame > self.rx_buf.len) return error.PayloadTooLarge;
                self.rx_header = hdr;
                self.rx_expected = total_frame;
            }

            while (self.rx_len < self.rx_expected) {
                const dst = self.rx_buf[self.rx_len..self.rx_expected];
                const rc = std.c.recv(self.fd, dst.ptr, dst.len, 0);
                if (rc < 0) {
                    const err = std.c.errno(rc);
                    if (err == .INTR) continue;
                    if (err == .AGAIN) return .{ .would_block = total_read };
                    if (err == .CONNRESET) return error.ConnectionReset;
                    return error.RecvFailed;
                }
                if (rc == 0) return error.UnexpectedEof;
                const n: usize = @intCast(rc);
                self.rx_len += n;
                total_read += n;
            }

            const hdr = self.rx_header.?;
            const payload: []align(8) const u8 = @alignCast(self.rx_buf[HEADER_BYTES..self.rx_expected]);
            if (std.hash.Crc32.hash(payload) != hdr.payload_crc32) {
                return error.PayloadCrcMismatch;
            }
            return .{
                .frame_ready = .{
                    .header = hdr,
                    .payload = payload,
                    .bytes_read = total_read,
                },
            };
        }
    }

    /// Drives a queued TX frame to completion via `kq_fd` within `timeout_ms`.
    pub fn sendFrameSync(self: *FrameChannel, kq_fd: std.c.fd_t, timeout_ms: u32) TransportError!u64 {
        const deadline_ns = (vmstat.nowNs() catch return error.ClockFailed) +
            @as(u64, timeout_ms) * std.time.ns_per_ms;
        var total_sent: u64 = 0;

        while (true) {
            switch (try self.stepSend()) {
                .complete => |n| {
                    total_sent += n;
                    return total_sent;
                },
                .would_block => |n| {
                    total_sent += n;
                    const now = vmstat.nowNs() catch return error.ClockFailed;
                    if (now >= deadline_ns) return error.Timeout;
                    try kqueueSetFilter(
                        kq_fd,
                        self.fd,
                        std.c.EVFILT.WRITE,
                        std.c.EV.ADD | std.c.EV.ENABLE | std.c.EV.ONESHOT,
                        @intCast(self.fd),
                    );
                    var evs: [2]std.c.kevent64_s = undefined;
                    const nev = try kqueueWait(kq_fd, &evs, deadline_ns - now);
                    if (nev == 0) return error.Timeout;
                },
            }
        }
    }

    /// Drives RX until one full CRC-verified `AMI7` frame arrives within `timeout_ms`.
    pub fn recvFrameSync(
        self: *FrameChannel,
        kq_fd: std.c.fd_t,
        timeout_ms: u32,
    ) (TransportError || ProtocolError)!ReceivedFrame {
        const deadline_ns = (vmstat.nowNs() catch return error.ClockFailed) +
            @as(u64, timeout_ms) * std.time.ns_per_ms;
        var total_read: usize = 0;

        while (true) {
            switch (try self.stepRecv()) {
                .frame_ready => |frame| {
                    var res = frame;
                    res.bytes_read += total_read;
                    return res;
                },
                .peer_closed => return error.UnexpectedEof,
                .would_block => |n| {
                    total_read += n;
                    const now = vmstat.nowNs() catch return error.ClockFailed;
                    if (now >= deadline_ns) return error.Timeout;
                    try kqueueSetFilter(
                        kq_fd,
                        self.fd,
                        std.c.EVFILT.READ,
                        std.c.EV.ADD | std.c.EV.ENABLE | std.c.EV.ONESHOT,
                        @intCast(self.fd),
                    );
                    var evs: [2]std.c.kevent64_s = undefined;
                    const nev = try kqueueWait(kq_fd, &evs, deadline_ns - now);
                    if (nev == 0) return error.Timeout;
                },
            }
        }
    }

    pub fn vtable(self: *FrameChannel) TransportVTable {
        return .{
            .ctx = self,
            .kind = .tb4_tcp,
            .sendSyncFn = vtableSendSync,
            .recvSyncFn = vtableRecvSync,
            .closeFn = vtableClose,
        };
    }

    fn vtableSendSync(ctx: *anyopaque, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!u64 {
        const self: *FrameChannel = @ptrCast(@alignCast(ctx));
        return self.sendFrameSync(kq_fd, timeout_ms);
    }

    fn vtableRecvSync(ctx: *anyopaque, kq_fd: std.c.fd_t, timeout_ms: u32) (TransportError || ProtocolError)!ReceivedFrame {
        const self: *FrameChannel = @ptrCast(@alignCast(ctx));
        return self.recvFrameSync(kq_fd, timeout_ms);
    }

    fn vtableClose(ctx: *anyopaque) void {
        const self: *FrameChannel = @ptrCast(@alignCast(ctx));
        self.close();
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "probeTb5Rdma and transport selection handle TB4 fallback and strict TB5 error" {
    const live = probeTb5Rdma();
    try std.testing.expect(live.reason.len > 0);
    if (!live.available) {
        try std.testing.expectEqual(@as(u32, 0), live.device_count);
        const auto_res = try selectTransport(.auto);
        try std.testing.expectEqual(TransportKind.tb4_tcp, auto_res.kind);
        try std.testing.expectError(error.Tb5RdmaUnavailable, selectTransport(.tb5_rdma));
    }

    const explicit_tcp = try selectTransport(.tb4_tcp);
    try std.testing.expectEqual(TransportKind.tb4_tcp, explicit_tcp.kind);

    // Deterministic coverage of both unavailable and available TB5 RDMA probes.
    const absent_probe: RdmaProbeResult = .{
        .available = false,
        .driver_lib_found = false,
        .device_count = 0,
        .reason = "rdma_framework_absent",
    };
    const fallback = try selectTransportWithProbe(.auto, absent_probe);
    try std.testing.expectEqual(TransportKind.tb4_tcp, fallback.kind);
    try std.testing.expectError(error.Tb5RdmaUnavailable, selectTransportWithProbe(.tb5_rdma, absent_probe));

    const ready_probe: RdmaProbeResult = .{
        .available = true,
        .driver_lib_found = true,
        .device_count = 2,
        .reason = "tb5_rdma_ready",
    };
    const auto_rdma = try selectTransportWithProbe(.auto, ready_probe);
    try std.testing.expectEqual(TransportKind.tb5_rdma, auto_rdma.kind);
    const strict_rdma = try selectTransportWithProbe(.tb5_rdma, ready_probe);
    try std.testing.expectEqual(TransportKind.tb5_rdma, strict_rdma.kind);
    try std.testing.expectEqual(@as(u32, 2), strict_rdma.probe.device_count);
}

test "AMI7 FrameHeader and HelloPayload encode, decode, and catch CRC corruption" {
    var raw: [HEADER_BYTES]u8 = undefined;
    const hdr: FrameHeader = .{
        .msg_type = .eval_experts_req,
        .flags = 0x03,
        .seq_id = 42,
        .payload_len = 128,
        .payload_crc32 = 0xDEADBEEF,
    };
    const encoded = hdr.encode(&raw);
    try std.testing.expectEqual(MAGIC, encoded.magic);
    try std.testing.expectEqual(PROTOCOL_VERSION, encoded.version);
    const decoded = try FrameHeader.decode(&raw);
    try std.testing.expectEqual(encoded.seq_id, decoded.seq_id);
    try std.testing.expectEqual(encoded.msg_type, decoded.msg_type);
    try std.testing.expectEqual(encoded.payload_len, decoded.payload_len);
    try std.testing.expectEqual(encoded.payload_crc32, decoded.payload_crc32);
    try std.testing.expectEqual(encoded.header_crc32, decoded.header_crc32);

    // Corrupt a payload length byte -> header CRC must reject.
    var bad_crc = raw;
    bad_crc[16] ^= 0x01;
    try std.testing.expectError(error.HeaderCrcMismatch, FrameHeader.decode(&bad_crc));

    // Corrupt magic -> BadMagic.
    var bad_magic = raw;
    bad_magic[0] ^= 0xFF;
    try std.testing.expectError(error.BadMagic, FrameHeader.decode(&bad_magic));

    // Corrupt msg_type -> InvalidMsgType.
    var bad_type = raw;
    bad_type[6] = 99;
    try std.testing.expectError(error.InvalidMsgType, FrameHeader.decode(&bad_type));

    // HelloPayload 1024-bit expert bitmap round-trip (including expert 511 and 1023).
    var hello: HelloPayload = .{
        .hidden_size = 4096,
        .rank = 3,
        .world_size = 4,
        .num_layers = 61,
        .num_experts = 512,
        .expert_range_start = 384,
        .expert_range_end = 512,
        .transport_kind = @backingInt(TransportKind.tb4_tcp),
        .status = 0,
    };
    hello.setExpert(0);
    hello.setExpert(511);
    hello.setExpert(1023);
    var hello_raw: [HelloPayload.WIRE_BYTES]u8 = undefined;
    hello.encode(&hello_raw);
    const hello_dec = try HelloPayload.decode(&hello_raw);
    try std.testing.expectEqual(@as(u32, 4096), hello_dec.hidden_size);
    try std.testing.expectEqual(@as(u16, 3), hello_dec.rank);
    try std.testing.expect(hello_dec.hasExpert(0));
    try std.testing.expect(hello_dec.hasExpert(511));
    try std.testing.expect(hello_dec.hasExpert(1023));
    try std.testing.expect(!hello_dec.hasExpert(128));
}

test "parseIpv4 parses Thunderbolt Bridge and loopback addresses without allocation" {
    try std.testing.expectEqual([4]u8{ 127, 0, 0, 1 }, try parseIpv4("127.0.0.1"));
    try std.testing.expectEqual([4]u8{ 169, 254, 42, 7 }, try parseIpv4("169.254.42.7"));
    try std.testing.expectError(error.InvalidAddress, parseIpv4("169.254.1"));
    try std.testing.expectError(error.InvalidAddress, parseIpv4("256.0.0.1"));
    try std.testing.expectError(error.InvalidAddress, parseIpv4("not-an-ip"));
}

test "WiredSlab carves 16 KiB aligned frames and drives loopback FrameChannel round-trip" {
    var slab = try WiredSlab.initOwned(4 * APPLE_PAGE_SIZE);
    defer slab.deinit();

    const c_tx = try slab.carvePages(APPLE_PAGE_SIZE);
    const c_rx = try slab.carvePages(APPLE_PAGE_SIZE);
    const s_tx = try slab.carvePages(APPLE_PAGE_SIZE);
    const s_rx = try slab.carvePages(APPLE_PAGE_SIZE);

    var listener = try TcpListener.bind(.{ .host = .{ 127, 0, 0, 1 }, .port = 0 }, 4);
    defer listener.close();
    try std.testing.expect(listener.bound_port > 0);

    const client_fd = try connectNonBlocking(.{ .host = .{ 127, 0, 0, 1 }, .port = listener.bound_port }, 2000);
    var client_chan = try FrameChannel.init(client_fd, c_tx, c_rx);
    defer client_chan.close();

    // Wait briefly until listener accepts.
    var server_fd: std.c.fd_t = -1;
    for (0..200) |_| {
        if (try listener.acceptNonBlocking()) |fd| {
            server_fd = fd;
            break;
        }
        const ts = nsToTimespec(1 * std.time.ns_per_ms);
        _ = std.c.nanosleep(&ts, null);
    }
    try std.testing.expect(server_fd >= 0);
    var server_chan = try FrameChannel.init(server_fd, s_tx, s_rx);
    defer server_chan.close();

    const kq_c = try kqueueCreate();
    defer _ = std.c.close(kq_c);
    const kq_s = try kqueueCreate();
    defer _ = std.c.close(kq_s);

    // Send Ping through TransportVTable and echo Pong back.
    const ping_payload = try client_chan.preparePayload(8);
    std.mem.writeInt(u64, ping_payload[0..8], 0x123456789ABCDEF0, .little);
    _ = client_chan.commitFrame(.ping, 0, 77, 8);

    const vt = client_chan.vtable();
    try std.testing.expectEqual(TransportKind.tb4_tcp, vt.kind);
    const sent = try vt.sendSync(kq_c, 2000);
    try std.testing.expectEqual(@as(u64, HEADER_BYTES + 8), sent);

    const req = try server_chan.recvFrameSync(kq_s, 2000);
    try std.testing.expectEqual(MsgType.ping, req.header.msg_type);
    try std.testing.expectEqual(@as(u64, 77), req.header.seq_id);
    try std.testing.expectEqual(@as(u64, 0x123456789ABCDEF0), std.mem.readInt(u64, req.payload[0..8], .little));
    server_chan.resetRx();

    const pong_payload = try server_chan.preparePayload(8);
    std.mem.writeInt(u64, pong_payload[0..8], 0x123456789ABCDEF0, .little);
    _ = server_chan.commitFrame(.pong, 0, 77, 8);
    _ = try server_chan.sendFrameSync(kq_s, 2000);

    const resp = try vt.recvSync(kq_c, 2000);
    try std.testing.expectEqual(MsgType.pong, resp.header.msg_type);
    try std.testing.expectEqual(@as(u64, 77), resp.header.seq_id);
    client_chan.resetRx();
}
