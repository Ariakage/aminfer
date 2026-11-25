//! Zero-allocation bounded path helpers for scratch directories, model roots,
//! and metadata directories.

const std = @import("std");

pub const ENV_TMPDIR: [:0]const u8 = "TMPDIR";
pub const ENV_MODELS_DIR: [:0]const u8 = "AMINFER_MODELS_DIR";
pub const ENV_METADATA_DIR: [:0]const u8 = "AMINFER_METADATA_DIR";
pub const DEFAULT_TMP_DIR: []const u8 = "/tmp";

fn trimDir(raw: []const u8) ?[]const u8 {
    if (raw.len == 0) return null;
    const trimmed = std.mem.trimEnd(u8, raw, "/");
    if (trimmed.len == 0) return "/";
    return trimmed;
}

fn readEnvDir(name: [:0]const u8) ?[]const u8 {
    const ptr = std.c.getenv(name.ptr) orelse return null;
    return trimDir(std.mem.sliceTo(ptr, 0));
}

/// Resolves the temporary root directory from `TMPDIR` (trimmed of trailing `/`),
/// falling back to `"/tmp"` when unset or empty.
pub fn tmpDir() []const u8 {
    return readEnvDir(ENV_TMPDIR) orelse DEFAULT_TMP_DIR;
}

/// Resolves a scratch directory override when provided and non-empty, otherwise
/// falling back to `tmpDir()`.
pub fn resolveTmpDir(override: ?[]const u8) []const u8 {
    if (override) |raw| {
        if (trimDir(raw)) |dir| return dir;
    }
    return tmpDir();
}

/// Formats `<base_dir>/<subpath>` into a caller-provided stack buffer with zero
/// heap allocations.
pub fn joinPath(buf: []u8, base_dir: []const u8, subpath: []const u8) error{NoSpaceLeft}![]const u8 {
    const norm_base = trimDir(base_dir) orelse "";
    const clean_sub = std.mem.trimStart(u8, subpath, "/");
    if (norm_base.len == 0) {
        return std.fmt.bufPrint(buf, "{s}", .{clean_sub});
    }
    if (std.mem.eql(u8, norm_base, "/")) {
        return std.fmt.bufPrint(buf, "/{s}", .{clean_sub});
    }
    if (clean_sub.len == 0) {
        return std.fmt.bufPrint(buf, "{s}", .{norm_base});
    }
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ norm_base, clean_sub });
}

/// Formats NUL-terminated `<base_dir>/<subpath>` into a caller-provided buffer.
pub fn joinPathZ(buf: []u8, base_dir: []const u8, subpath: []const u8) error{NoSpaceLeft}![:0]const u8 {
    if (buf.len == 0) return error.NoSpaceLeft;
    const s = try joinPath(buf[0 .. buf.len - 1], base_dir, subpath);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

/// Formats NUL-terminated `<base_dir>/<fmt...>` into a caller-provided buffer.
pub fn joinPathFmtZ(buf: []u8, base_dir: []const u8, comptime fmt: []const u8, args: anytype) error{NoSpaceLeft}![:0]const u8 {
    if (buf.len == 0) return error.NoSpaceLeft;
    const usable = buf[0 .. buf.len - 1];
    const norm_base = trimDir(base_dir) orelse "";
    const prefix = if (norm_base.len == 0)
        ""
    else if (std.mem.eql(u8, norm_base, "/"))
        try std.fmt.bufPrint(usable, "/", .{})
    else
        try std.fmt.bufPrint(usable, "{s}/", .{norm_base});
    const rest = try std.fmt.bufPrint(usable[prefix.len..], fmt, args);
    const total_len = prefix.len + rest.len;
    buf[total_len] = 0;
    return buf[0..total_len :0];
}

/// Formats `<tmpdir>/<subpath>` into a caller-provided stack buffer.
pub fn tmpPath(buf: []u8, subpath: []const u8) error{NoSpaceLeft}![]const u8 {
    return joinPath(buf, tmpDir(), subpath);
}

/// Formats NUL-terminated `<tmpdir>/<subpath>` into a caller-provided buffer.
pub fn tmpPathZ(buf: []u8, subpath: []const u8) error{NoSpaceLeft}![:0]const u8 {
    return joinPathZ(buf, tmpDir(), subpath);
}

/// Formats NUL-terminated `<tmpdir>/<fmt...>` into a caller-provided buffer.
pub fn tmpPathFmtZ(buf: []u8, comptime fmt: []const u8, args: anytype) error{NoSpaceLeft}![:0]const u8 {
    return joinPathFmtZ(buf, tmpDir(), fmt, args);
}

/// Resolves the model root directory in precedence order:
/// explicit CLI `--models-dir` -> `AMINFER_MODELS_DIR` env var -> `null`.
pub fn resolveModelsDir(cli_override: ?[]const u8) ?[]const u8 {
    if (cli_override) |raw| {
        if (trimDir(raw)) |dir| return dir;
    }
    return readEnvDir(ENV_MODELS_DIR);
}

/// Resolves the metadata root directory in precedence order:
/// `AMINFER_METADATA_DIR` -> `<models_dir>/metadata` -> `null`.
pub fn resolveMetadataDir(cli_models_override: ?[]const u8, buf: []u8) error{NoSpaceLeft}!?[]const u8 {
    if (readEnvDir(ENV_METADATA_DIR)) |meta_dir| {
        return meta_dir;
    }
    const models_dir = resolveModelsDir(cli_models_override) orelse return null;
    return try joinPath(buf, models_dir, "metadata");
}

/// Formats NUL-terminated `<models_dir>/<subpath>` into `buf`, or returns
/// `null` if no model root is configured.
pub fn resolveModelFileZ(cli_models_override: ?[]const u8, buf: []u8, subpath: []const u8) error{NoSpaceLeft}!?[:0]const u8 {
    const models_dir = resolveModelsDir(cli_models_override) orelse return null;
    return try joinPathZ(buf, models_dir, subpath);
}

/// Formats NUL-terminated `<metadata_dir>/<subpath>` into `buf`, or returns
/// `null` if neither `AMINFER_METADATA_DIR` nor a models root is configured.
pub fn resolveMetadataFileZ(cli_models_override: ?[]const u8, buf: []u8, subpath: []const u8) error{NoSpaceLeft}!?[:0]const u8 {
    var meta_buf: [std.fs.max_path_bytes]u8 = undefined;
    const meta_dir = (try resolveMetadataDir(cli_models_override, &meta_buf)) orelse return null;
    return try joinPathZ(buf, meta_dir, subpath);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const SavedEnv = struct {
    name: [:0]const u8,
    prev: ?[:0]u8,

    fn captureAndClear(allocator: std.mem.Allocator, name: [:0]const u8) !SavedEnv {
        const prev = if (std.c.getenv(name.ptr)) |p|
            try std.fmt.allocPrintSentinel(allocator, "{s}", .{std.mem.sliceTo(p, 0)}, 0)
        else
            null;
        if (unsetenv(name.ptr) != 0) return error.Unexpected;
        return .{ .name = name, .prev = prev };
    }

    fn restore(self: *SavedEnv, allocator: std.mem.Allocator) void {
        if (self.prev) |val| {
            _ = setenv(self.name.ptr, val.ptr, 1);
            allocator.free(val);
            self.prev = null;
        } else {
            _ = unsetenv(self.name.ptr);
        }
    }
};

test "paths: tmpDir trims trailing slash, falls back to /tmp, and bounds stack buffers" {
    const gpa = std.testing.allocator;
    var saved_tmp = try SavedEnv.captureAndClear(gpa, ENV_TMPDIR);
    defer saved_tmp.restore(gpa);

    try std.testing.expectEqualStrings("/tmp", tmpDir());

    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("/tmp/sample.bin", try tmpPathZ(&buf, "sample.bin"));
    try std.testing.expectEqualStrings("/tmp/item_42.amf", try tmpPathFmtZ(&buf, "item_{d}.amf", .{42}));

    try std.testing.expectEqual(@as(c_int, 0), setenv(ENV_TMPDIR.ptr, "/var/folders/test///", 1));
    try std.testing.expectEqualStrings("/var/folders/test", tmpDir());
    try std.testing.expectEqualStrings("/var/folders/test/a.ang", try tmpPathZ(&buf, "/a.ang"));
    try std.testing.expectEqualStrings("/custom/scratch", resolveTmpDir("/custom/scratch/"));

    try std.testing.expectEqual(@as(c_int, 0), setenv(ENV_TMPDIR.ptr, "", 1));
    try std.testing.expectEqualStrings("/tmp", tmpDir());

    var tiny: [8]u8 = undefined;
    try std.testing.expectError(error.NoSpaceLeft, tmpPathZ(&tiny, "too_long_name.bin"));
}

test "paths: model and metadata root precedence respects CLI override and env vars" {
    const gpa = std.testing.allocator;
    var saved_models = try SavedEnv.captureAndClear(gpa, ENV_MODELS_DIR);
    defer saved_models.restore(gpa);
    var saved_meta = try SavedEnv.captureAndClear(gpa, ENV_METADATA_DIR);
    defer saved_meta.restore(gpa);

    var buf: [256]u8 = undefined;
    try std.testing.expectEqual(@as(?[]const u8, null), resolveModelsDir(null));
    try std.testing.expectEqual(@as(?[]const u8, null), try resolveMetadataDir(null, &buf));
    try std.testing.expectEqual(@as(?[:0]const u8, null), try resolveModelFileZ(null, &buf, "amf/a.amf"));
    try std.testing.expectEqual(@as(?[:0]const u8, null), try resolveMetadataFileZ(null, &buf, "m/tokenizer.json"));

    try std.testing.expectEqual(@as(c_int, 0), setenv(ENV_MODELS_DIR.ptr, "/opt/models///", 1));
    try std.testing.expectEqualStrings("/opt/models", resolveModelsDir(null).?);
    try std.testing.expectEqualStrings("/cli/models", resolveModelsDir("/cli/models/").?);
    try std.testing.expectEqualStrings("/cli/models/metadata", (try resolveMetadataDir("/cli/models/", &buf)).?);
    try std.testing.expectEqualStrings(
        "/cli/models/amf/q.amf",
        (try resolveModelFileZ("/cli/models/", &buf, "amf/q.amf")).?,
    );
    try std.testing.expectEqualStrings(
        "/opt/models/metadata/m/tokenizer.json",
        (try resolveMetadataFileZ(null, &buf, "m/tokenizer.json")).?,
    );

    try std.testing.expectEqual(@as(c_int, 0), setenv(ENV_METADATA_DIR.ptr, "/opt/meta/", 1));
    try std.testing.expectEqualStrings("/opt/meta", (try resolveMetadataDir("/cli/models/", &buf)).?);
    try std.testing.expectEqualStrings(
        "/opt/meta/m/tokenizer.json",
        (try resolveMetadataFileZ("/cli/models/", &buf, "m/tokenizer.json")).?,
    );
}
