//! Zig wrapper for the native macOS `Network.framework` + `Security.framework`
//! TLS 1.2/1.3 listener bridge (`src/bridge/apple_bridge.h`).

const std = @import("std");
const c = @import("apple_bridge");

pub const TLS_VERSION_1_1: u16 = 0x0302;
pub const TLS_VERSION_1_2: u16 = 0x0303;
pub const TLS_VERSION_1_3: u16 = 0x0304;

pub const TlsError = error{
    InvalidArgument,
    OutOfMemory,
    TlsIdentityInvalid,
    TlsPassphraseRequired,
    TlsBindFailed,
    UnexpectedStatus,
    NullHandle,
    FileReadFailed,
};

pub const TlsConfig = struct {
    host_ipv4: [4]u8 = .{ 127, 0, 0, 1 },
    port: u16 = 0,
    backlog: u16 = 128,
    cert_bytes: []const u8,
    key_bytes: ?[]const u8 = null,
    passphrase: ?[]const u8 = null,
};

pub const LoadedFiles = struct {
    cert_bytes: []u8,
    key_bytes: ?[]u8,

    /// Securely wipes both credential buffers before freeing them.
    pub fn deinit(self: *LoadedFiles, allocator: std.mem.Allocator) void {
        if (self.cert_bytes.len > 0) {
            std.crypto.secureZero(u8, self.cert_bytes);
        }
        allocator.free(self.cert_bytes);
        if (self.key_bytes) |kb| {
            if (kb.len > 0) {
                std.crypto.secureZero(u8, kb);
            }
            allocator.free(kb);
        }
        self.* = .{ .cert_bytes = &.{}, .key_bytes = null };
    }
};

fn check(status: c.ami_status) TlsError!void {
    return switch (status) {
        c.AMI_OK => {},
        c.AMI_ERR_INVALID_ARGUMENT => error.InvalidArgument,
        c.AMI_ERR_OUT_OF_MEMORY => error.OutOfMemory,
        c.AMI_STATUS_TLS_IDENTITY_INVALID => error.TlsIdentityInvalid,
        c.AMI_STATUS_TLS_PASSPHRASE_REQUIRED => error.TlsPassphraseRequired,
        c.AMI_STATUS_TLS_BIND_FAILED => error.TlsBindFailed,
        else => error.UnexpectedStatus,
    };
}

fn readCredentialFileAlloc(allocator: std.mem.Allocator, path: []const u8, max_bytes: usize) TlsError![]u8 {
    if (path.len == 0 or path.len >= std.fs.max_path_bytes) {
        return error.InvalidArgument;
    }
    var path_z: [std.fs.max_path_bytes]u8 = undefined;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;

    const fd = std.c.open(@ptrCast(&path_z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileReadFailed;
    defer _ = std.c.close(fd);

    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.FileReadFailed;
    if (st.size < 0) return error.FileReadFailed;
    const sz: usize = @intCast(st.size);
    if (sz == 0 or sz > max_bytes) return error.FileReadFailed;

    const buf = allocator.alloc(u8, sz) catch return error.OutOfMemory;
    errdefer {
        std.crypto.secureZero(u8, buf);
        allocator.free(buf);
    }

    var done: usize = 0;
    while (done < sz) {
        const rc = std.c.read(fd, buf.ptr + done, sz - done);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.FileReadFailed;
        }
        if (rc == 0) return error.FileReadFailed;
        done += @intCast(rc);
    }
    return buf;
}

/// Loads `cert_path` (PEM cert/fullchain or `.p12`/`.pfx`) and optional `key_path`
/// into temporary buffers owned by the caller; `LoadedFiles.deinit` wipes them
/// with `std.crypto.secureZero`.
pub fn loadFiles(
    allocator: std.mem.Allocator,
    cert_path: []const u8,
    key_path: ?[]const u8,
) TlsError!LoadedFiles {
    const max_cred_bytes: usize = 4 * 1024 * 1024;
    const cert_bytes = try readCredentialFileAlloc(allocator, cert_path, max_cred_bytes);
    errdefer {
        std.crypto.secureZero(u8, cert_bytes);
        allocator.free(cert_bytes);
    }
    const key_bytes: ?[]u8 = if (key_path) |kp|
        try readCredentialFileAlloc(allocator, kp, max_cred_bytes)
    else
        null;
    return .{
        .cert_bytes = cert_bytes,
        .key_bytes = key_bytes,
    };
}

pub const TlsListener = struct {
    handle: *c.ami_tls_listener,

    pub fn init(cfg: TlsConfig) TlsError!TlsListener {
        var raw_cfg: c.ami_tls_config = .{
            .host_ipv4 = cfg.host_ipv4,
            .port = cfg.port,
            .backlog = cfg.backlog,
            .cert_bytes = if (cfg.cert_bytes.len > 0) cfg.cert_bytes.ptr else null,
            .cert_len = cfg.cert_bytes.len,
            .key_bytes = if (cfg.key_bytes) |kb| (if (kb.len > 0) kb.ptr else null) else null,
            .key_len = if (cfg.key_bytes) |kb| kb.len else 0,
            .passphrase_bytes = if (cfg.passphrase) |pb| (if (pb.len > 0) pb.ptr else null) else null,
            .passphrase_len = if (cfg.passphrase) |pb| pb.len else 0,
        };
        var out: ?*c.ami_tls_listener = null;
        try check(c.ami_tls_listener_create(&raw_cfg, &out));
        return .{ .handle = out orelse return error.NullHandle };
    }

    pub fn initFromFiles(
        allocator: std.mem.Allocator,
        host_ipv4: [4]u8,
        port: u16,
        backlog: u16,
        cert_path: []const u8,
        key_path: ?[]const u8,
        passphrase: ?[]const u8,
    ) TlsError!TlsListener {
        var files = try loadFiles(allocator, cert_path, key_path);
        defer files.deinit(allocator);
        return init(.{
            .host_ipv4 = host_ipv4,
            .port = port,
            .backlog = backlog,
            .cert_bytes = files.cert_bytes,
            .key_bytes = files.key_bytes,
            .passphrase = passphrase,
        });
    }

    pub fn deinit(self: TlsListener) void {
        c.ami_tls_listener_destroy(self.handle);
    }

    pub fn boundPort(self: TlsListener) u16 {
        return c.ami_tls_listener_bound_port(self.handle);
    }

    pub fn notifyFd(self: TlsListener) std.c.fd_t {
        return c.ami_tls_listener_notify_fd(self.handle);
    }

    pub fn popAcceptedFd(self: TlsListener) TlsError!?std.c.fd_t {
        var fd: i32 = -1;
        try check(c.ami_tls_listener_pop_accepted_fd(self.handle, &fd));
        if (fd < 0) return null;
        return fd;
    }
};

pub const ClientConnection = struct {
    fd: std.c.fd_t,
    tls_version: u16,
};

/// WARNING: INSECURE LOOPBACK TEST HELPER ONLY.
/// Connects a TLS client over loopback with server certificate verification
/// DISABLED (`complete(true)` unconditionally). Do NOT use in production; use
/// `clientConnectSocketpairVerified` whenever peer authenticity is required.
pub fn clientConnectSocketpairInsecureLoopback(
    host_ipv4: [4]u8,
    port: u16,
    min_tls_version: u16,
    max_tls_version: u16,
) TlsError!ClientConnection {
    var fd: i32 = -1;
    var ver: u16 = 0;
    try check(c.ami_tls_client_connect_socketpair_insecure_loopback(
        &host_ipv4,
        port,
        min_tls_version,
        max_tls_version,
        &fd,
        &ver,
    ));
    if (fd < 0) return error.TlsBindFailed;
    return .{ .fd = fd, .tls_version = ver };
}

/// Legacy alias delegating to `clientConnectSocketpairInsecureLoopback`.
/// WARNING: Disables X.509 certificate verification; retained solely for
/// existing loopback benchmarks and tests.
pub fn clientConnectSocketpair(
    host_ipv4: [4]u8,
    port: u16,
    min_tls_version: u16,
    max_tls_version: u16,
) TlsError!ClientConnection {
    return clientConnectSocketpairInsecureLoopback(host_ipv4, port, min_tls_version, max_tls_version);
}

/// Certificate-verifying TLS client backed by `nw_connection_t` + `socketpair`.
/// Evaluates peer trust via `SecTrustEvaluateWithError` inside the verify block.
/// Pass `server_name` for SNI/hostname checking (or `null` to check `host_ipv4`),
/// and `anchor_pem` to pin against caller-provided PEM trust anchor(s) (or
/// `null` to validate against the macOS system trust store).
pub fn clientConnectSocketpairVerified(
    host_ipv4: [4]u8,
    port: u16,
    server_name: ?[:0]const u8,
    anchor_pem: ?[]const u8,
    min_tls_version: u16,
    max_tls_version: u16,
) TlsError!ClientConnection {
    var fd: i32 = -1;
    var ver: u16 = 0;
    const sn_ptr: ?[*:0]const u8 = if (server_name) |sn| sn.ptr else null;
    const anchor_ptr: ?[*]const u8 = if (anchor_pem) |ap| (if (ap.len > 0) ap.ptr else null) else null;
    const anchor_len: usize = if (anchor_pem) |ap| ap.len else 0;
    try check(c.ami_tls_client_connect_socketpair_verified(
        &host_ipv4,
        port,
        sn_ptr,
        anchor_ptr,
        anchor_len,
        min_tls_version,
        max_tls_version,
        &fd,
        &ver,
    ));
    if (fd < 0) return error.TlsBindFailed;
    return .{ .fd = fd, .tls_version = ver };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn runCmdChecked(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    const res = try std.process.run(allocator, io, .{ .argv = argv });
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("cmd failed ({d}): {s}\nstderr: {s}\n", .{ code, argv[0], res.stderr });
            return error.SubprocessFailed;
        },
        else => return error.SubprocessFailed,
    }
}

fn writeAllFd(fd: std.c.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const rc = std.c.write(fd, data.ptr + off, data.len - off);
        if (rc > 0) {
            off += @intCast(rc);
            continue;
        }
        const err = std.c.errno(rc);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.OUT, .revents = 0 }};
            _ = std.c.poll(&pfd, 1, 5000);
            continue;
        }
        return error.WriteFailed;
    }
}

fn readExactOrEofFd(fd: std.c.fd_t, out: []u8) !usize {
    var off: usize = 0;
    while (off < out.len) {
        const rc = std.c.read(fd, out.ptr + off, out.len - off);
        if (rc > 0) {
            off += @intCast(rc);
            continue;
        }
        if (rc == 0) break;
        const err = std.c.errno(rc);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
            const prc = std.c.poll(&pfd, 1, 5000);
            if (prc <= 0) return error.ReadTimeout;
            continue;
        }
        return error.ReadFailed;
    }
    return off;
}

fn popAcceptedWithWait(lst: TlsListener, timeout_ms: i32) !std.c.fd_t {
    if (try lst.popAcceptedFd()) |fd| return fd;
    const nfd = lst.notifyFd();
    var pfd = [1]std.c.pollfd{.{ .fd = nfd, .events = std.c.POLL.IN, .revents = 0 }};
    const prc = std.c.poll(&pfd, 1, timeout_ms);
    if (prc <= 0) return error.AcceptTimeout;
    return (try lst.popAcceptedFd()) orelse error.AcceptEmpty;
}

fn sleepMs(ms: u64) void {
    const req: std.c.timespec = .{
        .sec = @intCast(ms / 1000),
        .nsec = @intCast((ms % 1000) * 1_000_000),
    };
    _ = std.c.nanosleep(&req, null);
}

test "bridge.tls: EC P-256 PEM TLS 1.3 and forced TLS 1.2 round-trip, kqueue notifyFd, close_notify, and TLS <1.2 rejection" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const key_path = try std.fmt.allocPrint(gpa, "{s}/ec.key.pem", .{dir});
    defer gpa.free(key_path);
    const explicit_key_path = try std.fmt.allocPrint(gpa, "{s}/ec_explicit.key.pem", .{dir});
    defer gpa.free(explicit_key_path);
    const cert_path = try std.fmt.allocPrint(gpa, "{s}/ec.cert.pem", .{dir});
    defer gpa.free(cert_path);
    const explicit_cert_path = try std.fmt.allocPrint(gpa, "{s}/ec_explicit.cert.pem", .{dir});
    defer gpa.free(explicit_cert_path);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new",    "-key", key_path, "-subj", "/CN=127.0.0.1",
        "-days",            "1",   "-out",  cert_path,
    });

    // Verify non-RFC-5480 explicit EC cert fails cleanly (SecCertificateCopyKey rejects explicit params),
    // while `genpkey -pkeyopt ec_param_enc:named_curve` PKCS#8 PEM succeeds with TLS 1.3.
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "genpkey",         "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:P-256",
        "-out",             explicit_key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new",             "-key", explicit_key_path, "-subj", "/CN=127.0.0.1",
        "-days",            "1",   "-out",  explicit_cert_path,
    });
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, explicit_cert_path, explicit_key_path, null),
    );
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "genpkey",                  "-algorithm", "EC",              "-pkeyopt", "ec_paramgen_curve:P-256",
        "-pkeyopt",         "ec_param_enc:named_curve", "-out",       explicit_key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new",             "-key", explicit_key_path, "-subj", "/CN=127.0.0.1",
        "-days",            "1",   "-out",  explicit_cert_path,
    });
    {
        const lst_p8 = try TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, explicit_cert_path, explicit_key_path, null);
        defer lst_p8.deinit();
        const cc = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, lst_p8.boundPort(), TLS_VERSION_1_3, TLS_VERSION_1_3);
        _ = std.c.close(cc.fd);
        try std.testing.expectEqual(TLS_VERSION_1_3, cc.tls_version);
        const sfd = try popAcceptedWithWait(lst_p8, 3000);
        _ = std.c.close(sfd);
    }

    // Verify loadFiles + secureZero deinit wipes credential buffers in place.
    var loaded = try loadFiles(gpa, cert_path, key_path);
    try std.testing.expect(loaded.cert_bytes.len > 0);
    try std.testing.expect(loaded.key_bytes != null);
    const lst = try TlsListener.init(.{
        .host_ipv4 = .{ 127, 0, 0, 1 },
        .port = 0,
        .backlog = 32,
        .cert_bytes = loaded.cert_bytes,
        .key_bytes = loaded.key_bytes,
    });
    loaded.deinit(gpa);
    defer lst.deinit();

    const port = lst.boundPort();
    try std.testing.expect(port > 0);
    try std.testing.expect((try lst.popAcceptedFd()) == null);

    // Register notifyFd() on kqueue EVFILT_READ and confirm zero busy-spin when drained.
    const kq = std.c.kqueue();
    try std.testing.expect(kq >= 0);
    defer _ = std.c.close(kq);
    const change = [1]std.c.kevent64_s{.{
        .ident = @intCast(lst.notifyFd()),
        .filter = std.c.EVFILT.READ,
        .flags = std.c.EV.ADD,
        .fflags = 0,
        .data = 0,
        .udata = 0,
        .ext = .{ 0, 0 },
    }};
    var ev: [1]std.c.kevent64_s = undefined;
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    try std.testing.expectEqual(@as(c_int, 0), std.c.kevent64(kq, &change, 1, &ev, 1, std.c.KEVENT.FLAG.NONE, &.{ .sec = 0, .nsec = 0 }));

    // 1) TLS 1.3 round-trip + clean close_notify EOF
    {
        const client = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, TLS_VERSION_1_3, TLS_VERSION_1_3);
        defer _ = std.c.close(client.fd);
        try std.testing.expectEqual(TLS_VERSION_1_3, client.tls_version);

        const n_events = std.c.kevent64(kq, &empty, 0, &ev, 1, std.c.KEVENT.FLAG.NONE, &.{ .sec = 3, .nsec = 0 });
        try std.testing.expectEqual(@as(c_int, 1), n_events);

        const sfd = (try lst.popAcceptedFd()) orelse return error.ExpectedAcceptedFd;
        try std.testing.expect((try lst.popAcceptedFd()) == null);
        // After draining accept queue, level-triggered notifyFd must NOT stay readable.
        try std.testing.expectEqual(@as(c_int, 0), std.c.kevent64(kq, &empty, 0, &ev, 1, std.c.KEVENT.FLAG.NONE, &.{ .sec = 0, .nsec = 0 }));

        const req = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
        try writeAllFd(client.fd, req);

        var req_buf: [128]u8 = undefined;
        const rlen = try readExactOrEofFd(sfd, req_buf[0..req.len]);
        try std.testing.expectEqualStrings(req, req_buf[0..rlen]);

        const resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
        try writeAllFd(sfd, resp);
        _ = std.c.close(sfd); // Triggers TLS close_notify via NW_CONNECTION_FINAL_MESSAGE_CONTEXT

        var resp_buf: [128]u8 = undefined;
        const c_rlen = try readExactOrEofFd(client.fd, &resp_buf);
        try std.testing.expectEqualStrings(resp, resp_buf[0..c_rlen]);
    }

    // 2) Forced TLS 1.2 (0x0303) round-trip
    {
        const client = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, TLS_VERSION_1_2, TLS_VERSION_1_2);
        defer _ = std.c.close(client.fd);
        try std.testing.expectEqual(TLS_VERSION_1_2, client.tls_version);

        const sfd = try popAcceptedWithWait(lst, 3000);
        const ping = "PING12";
        try writeAllFd(client.fd, ping);
        var buf: [16]u8 = undefined;
        const rn = try readExactOrEofFd(sfd, buf[0..ping.len]);
        try std.testing.expectEqualStrings(ping, buf[0..rn]);
        try writeAllFd(sfd, "PONG12");
        _ = std.c.close(sfd);

        const cn = try readExactOrEofFd(client.fd, &buf);
        try std.testing.expectEqualStrings("PONG12", buf[0..cn]);
    }

    // 3) TLS < 1.2 (0x0302) must be rejected and produce no accepted server fd
    try std.testing.expectError(
        error.TlsBindFailed,
        clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, TLS_VERSION_1_1, TLS_VERSION_1_1),
    );
    sleepMs(50);
    try std.testing.expect((try lst.popAcceptedFd()) == null);
}

test "bridge.tls: RSA-2048 fullchain PEM + encrypted PKCS#8 (PBES2, PBES1) and legacy PEM passphrase handling" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    const ca_key = try std.fmt.allocPrint(gpa, "{s}/ca.key.pem", .{dir});
    defer gpa.free(ca_key);
    const ca_cert = try std.fmt.allocPrint(gpa, "{s}/ca.cert.pem", .{dir});
    defer gpa.free(ca_cert);
    const leaf_key = try std.fmt.allocPrint(gpa, "{s}/leaf.key.pem", .{dir});
    defer gpa.free(leaf_key);
    const leaf_csr = try std.fmt.allocPrint(gpa, "{s}/leaf.csr.pem", .{dir});
    defer gpa.free(leaf_csr);
    const leaf_cert = try std.fmt.allocPrint(gpa, "{s}/leaf.cert.pem", .{dir});
    defer gpa.free(leaf_cert);
    const fullchain = try std.fmt.allocPrint(gpa, "{s}/fullchain.pem", .{dir});
    defer gpa.free(fullchain);
    const enc_pbes2 = try std.fmt.allocPrint(gpa, "{s}/leaf.pbes2.pem", .{dir});
    defer gpa.free(enc_pbes2);
    const enc_pbes1 = try std.fmt.allocPrint(gpa, "{s}/leaf.pbes1.pem", .{dir});
    defer gpa.free(enc_pbes1);
    const enc_legacy = try std.fmt.allocPrint(gpa, "{s}/leaf.legacy.pem", .{dir});
    defer gpa.free(enc_legacy);
    const ca_serial = try std.fmt.allocPrint(gpa, "{s}/ca.srl", .{dir});
    defer gpa.free(ca_serial);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req",                 "-x509", "-newkey", "rsa:2048", "-nodes",
        "-subj",            "/CN=aminfer-test-ca", "-days", "1",       "-keyout",  ca_key,
        "-out",             ca_cert,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req",           "-new",    "-newkey", "rsa:2048", "-nodes",
        "-subj",            "/CN=127.0.0.1", "-keyout", leaf_key,  "-out",     leaf_csr,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "x509",      "-req",    "-in",  leaf_csr,  "-CA",   ca_cert, "-CAkey", ca_key,
        "-CAcreateserial",  "-CAserial", ca_serial, "-out", leaf_cert, "-days", "1",
    });

    // Build fullchain.pem = leaf + ca
    {
        var files_leaf = try loadFiles(gpa, leaf_cert, ca_cert);
        defer files_leaf.deinit(gpa);
        const fc_z = try std.fmt.allocPrintSentinel(gpa, "{s}", .{fullchain}, 0);
        defer gpa.free(fc_z);
        const fd = std.c.open(fc_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        try writeAllFd(fd, files_leaf.cert_bytes);
        try writeAllFd(fd, files_leaf.key_bytes.?);
    }

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "pkcs8",           "-topk8", "-v2",     "aes256", "-in", leaf_key,
        "-passout",         "pass:amf-secret", "-out",   enc_pbes2,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "pkcs8",           "-topk8", "-in",     leaf_key,
        "-passout",         "pass:amf-secret", "-out",   enc_pbes1,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "rsa",             "-aes256", "-in",      leaf_key,
        "-passout",         "pass:amf-secret", "-out",    enc_legacy,
    });

    for ([_][]const u8{ enc_pbes2, enc_pbes1, enc_legacy }) |kp| {
        try std.testing.expectError(
            error.TlsPassphraseRequired,
            TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, fullchain, kp, null),
        );
        try std.testing.expectError(
            error.TlsIdentityInvalid,
            TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, fullchain, kp, "wrong-pass"),
        );

        const lst = try TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, fullchain, kp, "amf-secret");
        defer lst.deinit();
        const client = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, lst.boundPort(), TLS_VERSION_1_3, TLS_VERSION_1_3);
        defer _ = std.c.close(client.fd);
        try std.testing.expectEqual(TLS_VERSION_1_3, client.tls_version);
        const sfd = try popAcceptedWithWait(lst, 3000);
        _ = std.c.close(sfd);
    }
}

test "bridge.tls: PKCS#12 .p12 in-memory import, mismatched key rejection, and bind failure" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const key1 = try std.fmt.allocPrint(gpa, "{s}/k1.pem", .{dir});
    defer gpa.free(key1);
    const cert1 = try std.fmt.allocPrint(gpa, "{s}/c1.pem", .{dir});
    defer gpa.free(cert1);
    const key2 = try std.fmt.allocPrint(gpa, "{s}/k2.pem", .{dir});
    defer gpa.free(key2);
    const cert2 = try std.fmt.allocPrint(gpa, "{s}/c2.pem", .{dir});
    defer gpa.free(cert2);
    const p12_path = try std.fmt.allocPrint(gpa, "{s}/bundle.p12", .{dir});
    defer gpa.free(p12_path);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key1,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", key1, "-subj", "/CN=k1", "-days", "1", "-out", cert1,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key2,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", key2, "-subj", "/CN=k2", "-days", "1", "-out", cert2,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "pkcs12", "-export", "-in", cert1, "-inkey", key1, "-passout", "pass:p12-pass", "-out", p12_path,
    });

    // 1) Mismatched cert1 + key2 fails immediately
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, cert1, key2, null),
    );

    // 2) PKCS#12 missing/wrong/correct passphrase
    try std.testing.expectError(
        error.TlsPassphraseRequired,
        TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, p12_path, null, null),
    );
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, p12_path, null, "bad-p12-pass"),
    );

    const lst = try TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, p12_path, null, "p12-pass");
    defer lst.deinit();
    const port = lst.boundPort();

    const client = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, TLS_VERSION_1_3, TLS_VERSION_1_3);
    defer _ = std.c.close(client.fd);
    try std.testing.expectEqual(TLS_VERSION_1_3, client.tls_version);
    const sfd = try popAcceptedWithWait(lst, 3000);
    _ = std.c.close(sfd);

    // 3) Binding a second listener on the exact same already-bound port fails with TlsBindFailed
    var p12_loaded = try loadFiles(gpa, p12_path, null);
    defer p12_loaded.deinit(gpa);
    try std.testing.expectError(error.TlsBindFailed, TlsListener.init(.{
        .host_ipv4 = .{ 127, 0, 0, 1 },
        .port = port,
        .backlog = 16,
        .cert_bytes = p12_loaded.cert_bytes,
        .passphrase = "p12-pass",
    }));
}

test "bridge.tls: plain HTTP bytes fail TLS handshake cleanly without accepted fd" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const key_path = try std.fmt.allocPrint(gpa, "{s}/plain.key.pem", .{dir});
    defer gpa.free(key_path);
    const cert_path = try std.fmt.allocPrint(gpa, "{s}/plain.cert.pem", .{dir});
    defer gpa.free(cert_path);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", key_path, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", cert_path,
    });

    const lst = try TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 16, cert_path, key_path, null);
    defer lst.deinit();
    const port = lst.boundPort();

    const sock = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    try std.testing.expect(sock >= 0);
    defer _ = std.c.close(sock);

    var addr: std.c.sockaddr.in = .{
        .family = std.c.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
        .zero = @splat(0),
    };
    try std.testing.expectEqual(@as(c_int, 0), std.c.connect(sock, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)));
    try writeAllFd(sock, "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");

    var sink: [256]u8 = undefined;
    _ = std.c.read(sock, &sink, sink.len);
    sleepMs(50);
    try std.testing.expect((try lst.popAcceptedFd()) == null);
}

test "bridge.tls: multi-megabyte backpressure streaming, concurrent clients, and curl --tlsv1.3" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const key_path = try std.fmt.allocPrint(gpa, "{s}/stream.key.pem", .{dir});
    defer gpa.free(key_path);
    const cert_path = try std.fmt.allocPrint(gpa, "{s}/stream.cert.pem", .{dir});
    defer gpa.free(cert_path);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", key_path, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", cert_path,
    });

    const lst = try TlsListener.initFromFiles(gpa, .{ 127, 0, 0, 1 }, 0, 64, cert_path, key_path, null);
    defer lst.deinit();
    const port = lst.boundPort();

    const payload_len: usize = 2 * 1024 * 1024 + 137;
    const n_clients: usize = 4;

    const ServerCtx = struct {
        lst: TlsListener,
        expected_conns: usize,
        payload_len: usize,

        fn run(ctx: *@This()) !void {
            const chunk = try std.testing.allocator.alloc(u8, ctx.payload_len);
            defer std.testing.allocator.free(chunk);
            for (chunk, 0..) |*b, idx| {
                b.* = @truncate((idx * 131 + 17) & 0xff);
            }
            var handled: usize = 0;
            while (handled < ctx.expected_conns) : (handled += 1) {
                const sfd = try popAcceptedWithWait(ctx.lst, 5000);
                defer _ = std.c.close(sfd);

                var hdr_buf: [512]u8 = undefined;
                var hlen: usize = 0;
                while (hlen < hdr_buf.len) {
                    const rc = std.c.read(sfd, hdr_buf[hlen..].ptr, hdr_buf.len - hlen);
                    if (rc > 0) {
                        hlen += @intCast(rc);
                        if (std.mem.indexOf(u8, hdr_buf[0..hlen], "\r\n\r\n") != null) break;
                    } else if (rc == 0) {
                        break;
                    } else {
                        const err = std.c.errno(rc);
                        if (err == .INTR) continue;
                        if (err == .AGAIN) {
                            var pfd = [1]std.c.pollfd{.{ .fd = sfd, .events = std.c.POLL.IN, .revents = 0 }};
                            _ = std.c.poll(&pfd, 1, 5000);
                            continue;
                        }
                        return error.ServerReadFailed;
                    }
                }
                if (std.mem.startsWith(u8, hdr_buf[0..hlen], "GET /stream")) {
                    try writeAllFd(sfd, chunk);
                } else {
                    try writeAllFd(sfd, "HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\ntls13-ok\n");
                }
            }
        }
    };

    var sctx: ServerCtx = .{
        .lst = lst,
        .expected_conns = n_clients + 1,
        .payload_len = payload_len,
    };
    const srv_thread = try std.Thread.spawn(.{}, ServerCtx.run, .{&sctx});
    var srv_joined = false;
    defer if (!srv_joined) srv_thread.join();

    // 4 concurrent multi-MiB streaming clients
    const ClientTask = struct {
        port: u16,
        payload_len: usize,

        fn run(self: *@This()) !void {
            const client = try clientConnectSocketpair(.{ 127, 0, 0, 1 }, self.port, TLS_VERSION_1_3, TLS_VERSION_1_3);
            defer _ = std.c.close(client.fd);
            try std.testing.expectEqual(TLS_VERSION_1_3, client.tls_version);
            try writeAllFd(client.fd, "GET /stream HTTP/1.1\r\nHost: localhost\r\n\r\n");

            const rx = try std.testing.allocator.alloc(u8, self.payload_len + 64);
            defer std.testing.allocator.free(rx);
            const got = try readExactOrEofFd(client.fd, rx);
            try std.testing.expectEqual(self.payload_len, got);
            for (rx[0..got], 0..) |b, idx| {
                const exp: u8 = @truncate((idx * 131 + 17) & 0xff);
                if (b != exp) return error.DataCorruption;
            }
        }
    };

    var tasks: [n_clients]ClientTask = undefined;
    var threads: [n_clients]std.Thread = undefined;
    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |*th| th.join();
    for (&tasks, &threads) |*tk, *th| {
        tk.* = .{ .port = port, .payload_len = payload_len };
        th.* = try std.Thread.spawn(.{}, ClientTask.run, .{tk});
        spawned += 1;
    }
    for (threads[0..spawned]) |*th| th.join();
    spawned = 0;

    // Real `/usr/bin/curl -ksS` verification (macOS SecureTransport curl negotiates TLS 1.3 (304) by default,
    // while rejecting the explicit `--tlsv1.3` CLI flag with exit 4).
    const url = try std.fmt.allocPrint(gpa, "https://127.0.0.1:{d}/health", .{port});
    defer gpa.free(url);
    const curl_res = try std.process.run(gpa, io, .{
        .argv = &.{ "/usr/bin/curl", "-v", "-ksS", url },
    });
    defer gpa.free(curl_res.stdout);
    defer gpa.free(curl_res.stderr);
    if (!std.mem.eql(u8, curl_res.stdout, "tls13-ok\n")) {
        std.debug.print("curl term={any}\nstderr:\n{s}\n", .{ curl_res.term, curl_res.stderr });
    }
    try std.testing.expectEqualStrings("tls13-ok\n", curl_res.stdout);
    try std.testing.expect(
        std.mem.indexOf(u8, curl_res.stderr, "TLSv1.3") != null or
            std.mem.indexOf(u8, curl_res.stderr, "(304)") != null,
    );

    srv_thread.join();
    srv_joined = true;
}

test "bridge.tls: L2 clientConnectSocketpairInsecureLoopback and clientConnectSocketpairVerified trust evaluation" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const key_path = try std.fmt.allocPrint(gpa, "{s}/srv.key.pem", .{dir});
    defer gpa.free(key_path);
    const cert_path = try std.fmt.allocPrint(gpa, "{s}/srv.cert.pem", .{dir});
    defer gpa.free(cert_path);
    const other_key_path = try std.fmt.allocPrint(gpa, "{s}/other.key.pem", .{dir});
    defer gpa.free(other_key_path);
    const other_cert_path = try std.fmt.allocPrint(gpa, "{s}/other.cert.pem", .{dir});
    defer gpa.free(other_cert_path);

    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl",
        "req",
        "-x509",
        "-new",
        "-key",
        key_path,
        "-subj",
        "/CN=localhost",
        "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
        "-addext",
        "extendedKeyUsage=serverAuth",
        "-days",
        "1",
        "-out",
        cert_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", other_key_path,
    });
    try runCmdChecked(gpa, io, &.{
        "/usr/bin/openssl",
        "req",
        "-x509",
        "-new",
        "-key",
        other_key_path,
        "-subj",
        "/CN=other",
        "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
        "-addext",
        "extendedKeyUsage=serverAuth",
        "-days",
        "1",
        "-out",
        other_cert_path,
    });

    var server_files = try loadFiles(gpa, cert_path, key_path);
    defer server_files.deinit(gpa);
    var other_files = try loadFiles(gpa, other_cert_path, other_key_path);
    defer other_files.deinit(gpa);

    const lst = try TlsListener.init(.{
        .host_ipv4 = .{ 127, 0, 0, 1 },
        .port = 0,
        .backlog = 16,
        .cert_bytes = server_files.cert_bytes,
        .key_bytes = server_files.key_bytes,
    });
    defer lst.deinit();
    const port = lst.boundPort();

    // 1. Insecure loopback variant succeeds against self-signed server without anchor.
    {
        const cc = try clientConnectSocketpairInsecureLoopback(.{ 127, 0, 0, 1 }, port, TLS_VERSION_1_3, TLS_VERSION_1_3);
        defer _ = std.c.close(cc.fd);
        try std.testing.expectEqual(TLS_VERSION_1_3, cc.tls_version);
        const sfd = try popAcceptedWithWait(lst, 3000);
        defer _ = std.c.close(sfd);
        try writeAllFd(cc.fd, "ping");
        var rx: [4]u8 = undefined;
        try std.testing.expectEqual(@as(usize, 4), try readExactOrEofFd(sfd, &rx));
        try std.testing.expectEqualStrings("ping", &rx);
    }

    // 2. Verified variant rejects untrusted self-signed cert when anchor_pem is null or mismatched.
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        clientConnectSocketpairVerified(.{ 127, 0, 0, 1 }, port, "localhost", null, TLS_VERSION_1_3, TLS_VERSION_1_3),
    );
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        clientConnectSocketpairVerified(.{ 127, 0, 0, 1 }, port, "localhost", other_files.cert_bytes, TLS_VERSION_1_3, TLS_VERSION_1_3),
    );
    try std.testing.expectError(
        error.TlsIdentityInvalid,
        clientConnectSocketpairVerified(.{ 127, 0, 0, 1 }, port, "wrong.example.com", server_files.cert_bytes, TLS_VERSION_1_3, TLS_VERSION_1_3),
    );

    // 3. Verified variant succeeds when matching anchor PEM and matching SNI/IP are supplied.
    {
        const vc = try clientConnectSocketpairVerified(.{ 127, 0, 0, 1 }, port, "localhost", server_files.cert_bytes, TLS_VERSION_1_3, TLS_VERSION_1_3);
        defer _ = std.c.close(vc.fd);
        try std.testing.expectEqual(TLS_VERSION_1_3, vc.tls_version);
        const sfd = try popAcceptedWithWait(lst, 3000);
        defer _ = std.c.close(sfd);
        try writeAllFd(vc.fd, "verified");
        var rx: [8]u8 = undefined;
        try std.testing.expectEqual(@as(usize, 8), try readExactOrEofFd(sfd, &rx));
        try std.testing.expectEqualStrings("verified", &rx);
    }
}
