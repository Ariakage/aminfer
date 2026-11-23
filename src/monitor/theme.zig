//! TrueColor palette and `btop` `.theme` parser for `aminfer monitor` (`C19`, ADR-0022).
//!
//! Built-in default palette: **`trans-pride`** ("soft pink, white and blue on deep navy"),
//! matching all 15 base chrome hex keys and all 9 3-stop smooth RGB gradients in
//! `ADR-0022` verbatim. Custom standard `btop` `.theme` files (`theme[key]="#RRGGBB"`)
//! load at startup via `Theme.loadFile(path)` with zero heap allocations.

const std = @import("std");

pub const Error = error{
    InvalidHexColor,
    InvalidSyntax,
    ThemeFileOpenFailed,
    ThemeFileReadFailed,
    ThemeFileTooLarge,
};

pub const MAX_THEME_FILE_BYTES: usize = 32 * 1024;

pub const Rgb = extern struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(r: u8, g: u8, b: u8) Rgb {
        return .{ .r = r, .g = g, .b = b };
    }

    pub fn eql(a: Rgb, b: Rgb) bool {
        return a.r == b.r and a.g == b.g and a.b == b.b;
    }

    /// Parses `#RRGGBB`, `RRGGBB`, `#RGB`, or `RGB` case-insensitively.
    pub fn parseHex(raw: []const u8) Error!Rgb {
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        const hex = if (trimmed.len > 0 and trimmed[0] == '#') trimmed[1..] else trimmed;
        if (hex.len == 6) {
            return .{
                .r = try parseHexByte(hex[0..2]),
                .g = try parseHexByte(hex[2..4]),
                .b = try parseHexByte(hex[4..6]),
            };
        }
        if (hex.len == 3) {
            const rn = try parseHexNibble(hex[0]);
            const gn = try parseHexNibble(hex[1]);
            const bn = try parseHexNibble(hex[2]);
            return .{
                .r = (rn << 4) | rn,
                .g = (gn << 4) | gn,
                .b = (bn << 4) | bn,
            };
        }
        return error.InvalidHexColor;
    }

    pub fn formatHex(self: Rgb, out: *[7]u8) []const u8 {
        const digits = "0123456789ABCDEF";
        out[0] = '#';
        out[1] = digits[self.r >> 4];
        out[2] = digits[self.r & 0x0F];
        out[3] = digits[self.g >> 4];
        out[4] = digits[self.g & 0x0F];
        out[5] = digits[self.b >> 4];
        out[6] = digits[self.b & 0x0F];
        return out[0..7];
    }

    /// Linear RGB interpolation (`t` clamped to `[0.0, 1.0]`).
    pub fn lerp(a: Rgb, b: Rgb, t: f32) Rgb {
        const u: f32 = if (std.math.isNan(t)) 0.0 else std.math.clamp(t, 0.0, 1.0);
        const inv: f32 = 1.0 - u;
        const rf: f32 = @as(f32, @floatFromInt(a.r)) * inv + @as(f32, @floatFromInt(b.r)) * u;
        const gf: f32 = @as(f32, @floatFromInt(a.g)) * inv + @as(f32, @floatFromInt(b.g)) * u;
        const bf: f32 = @as(f32, @floatFromInt(a.b)) * inv + @as(f32, @floatFromInt(b.b)) * u;
        return .{
            .r = @intFromFloat(@round(std.math.clamp(rf, 0.0, 255.0))),
            .g = @intFromFloat(@round(std.math.clamp(gf, 0.0, 255.0))),
            .b = @intFromFloat(@round(std.math.clamp(bf, 0.0, 255.0))),
        };
    }
};

fn parseHexNibble(ch: u8) Error!u8 {
    return switch (ch) {
        '0'...'9' => ch - '0',
        'a'...'f' => ch - 'a' + 10,
        'A'...'F' => ch - 'A' + 10,
        else => error.InvalidHexColor,
    };
}

fn parseHexByte(pair: *const [2]u8) Error!u8 {
    const hi = try parseHexNibble(pair[0]);
    const lo = try parseHexNibble(pair[1]);
    return (hi << 4) | lo;
}

/// 3-stop smooth RGB gradient (`start -> mid -> end`).
pub const Gradient3 = extern struct {
    start: Rgb,
    mid: Rgb,
    end: Rgb,

    pub fn init(start: Rgb, mid: Rgb, end: Rgb) Gradient3 {
        return .{ .start = start, .mid = mid, .end = end };
    }

    /// Evaluates the 3-stop gradient at `ratio` in `[0.0, 1.0]`.
    pub fn sample(self: Gradient3, ratio: f32) Rgb {
        const r: f32 = if (std.math.isNan(ratio)) 0.0 else std.math.clamp(ratio, 0.0, 1.0);
        if (r <= 0.5) {
            return Rgb.lerp(self.start, self.mid, r * 2.0);
        }
        return Rgb.lerp(self.mid, self.end, (r - 0.5) * 2.0);
    }

    /// Evaluates the 3-stop gradient at percentage `pct` in `[0.0, 100.0]`.
    pub fn samplePct(self: Gradient3, pct: f32) Rgb {
        return self.sample(pct / 100.0);
    }
};

pub const Theme = struct {
    name_buf: [64]u8 = initName("trans-pride"),
    name_len: u8 = "trans-pride".len,

    // 15 Base Chrome Keys (`ADR-0022`)
    main_bg: Rgb,
    main_fg: Rgb,
    title: Rgb,
    hi_fg: Rgb,
    selected_bg: Rgb,
    selected_fg: Rgb,
    inactive_fg: Rgb,
    graph_text: Rgb,
    meter_bg: Rgb,
    proc_misc: Rgb,
    cpu_box: Rgb,
    mem_box: Rgb,
    net_box: Rgb,
    proc_box: Rgb,
    div_line: Rgb,

    // 9 3-Stop Smooth RGB Gradients (`ADR-0022`)
    temp: Gradient3,
    cpu: Gradient3,
    free: Gradient3,
    cached: Gradient3,
    available: Gradient3,
    used: Gradient3,
    download: Gradient3,
    upload: Gradient3,
    process: Gradient3,

    fn initName(comptime s: []const u8) [64]u8 {
        var b: [64]u8 = @splat(0);
        @memcpy(b[0..s.len], s);
        return b;
    }

    pub fn name(self: *const Theme) []const u8 {
        const n: usize = @min(@as(usize, self.name_len), self.name_buf.len);
        return self.name_buf[0..n];
    }

    pub fn setName(self: *Theme, label: []const u8) void {
        self.name_buf = @splat(0);
        const n = @min(label.len, self.name_buf.len);
        @memcpy(self.name_buf[0..n], label[0..n]);
        self.name_len = @intCast(n);
    }

    /// Built-in default TrueColor palette: `trans-pride` ("soft pink, white and blue
    /// on deep navy"), matching `ADR-0022` verbatim.
    pub const TRANS_PRIDE: Theme = .{
        .main_bg = Rgb.init(0x11, 0x13, 0x1D), // #11131D
        .main_fg = Rgb.init(0xF7, 0xF7, 0xFC), // #F7F7FC
        .title = Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        .hi_fg = Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
        .selected_bg = Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        .selected_fg = Rgb.init(0x11, 0x13, 0x1D), // #11131D
        .inactive_fg = Rgb.init(0x92, 0x97, 0xAC), // #9297AC
        .graph_text = Rgb.init(0xF7, 0xF7, 0xFC), // #F7F7FC
        .meter_bg = Rgb.init(0x34, 0x38, 0x49), // #343849
        .proc_misc = Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        .cpu_box = Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
        .mem_box = Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        .net_box = Rgb.init(0xDA, 0xEA, 0xF4), // #DAEAF4
        .proc_box = Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
        .div_line = Rgb.init(0x77, 0x7F, 0x94), // #777F94

        .temp = Gradient3.init(
            Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
            Rgb.init(0xF7, 0xF7, 0xFC), // #F7F7FC
            Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        ),
        .cpu = Gradient3.init(
            Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
            Rgb.init(0xC1, 0xE5, 0xF7), // #C1E5F7
            Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        ),
        .free = Gradient3.init(
            Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
            Rgb.init(0xA5, 0xDF, 0xF5), // #A5DFF5
            Rgb.init(0xE6, 0xF6, 0xFF), // #E6F6FF
        ),
        .cached = Gradient3.init(
            Rgb.init(0xE9, 0xB7, 0xD0), // #E9B7D0
            Rgb.init(0xD0, 0xCC, 0xED), // #D0CCED
            Rgb.init(0xA1, 0xDD, 0xF5), // #A1DDF5
        ),
        .available = Gradient3.init(
            Rgb.init(0x88, 0xD5, 0xF4), // #88D5F4
            Rgb.init(0xC5, 0xE9, 0xF9), // #C5E9F9
            Rgb.init(0xF1, 0xFA, 0xFF), // #F1FAFF
        ),
        .used = Gradient3.init(
            Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
            Rgb.init(0xE8, 0xBF, 0xDD), // #E8BFDD
            Rgb.init(0xB9, 0xDF, 0xF3), // #B9DFF3
        ),
        .download = Gradient3.init(
            Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
            Rgb.init(0xB6, 0xE6, 0xF8), // #B6E6F8
            Rgb.init(0xF7, 0xF7, 0xFC), // #F7F7FC
        ),
        .upload = Gradient3.init(
            Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
            Rgb.init(0xF7, 0xC8, 0xD8), // #F7C8D8
            Rgb.init(0xF7, 0xF7, 0xFC), // #F7F7FC
        ),
        .process = Gradient3.init(
            Rgb.init(0x5B, 0xCE, 0xFA), // #5BCEFA
            Rgb.init(0xD2, 0xE7, 0xF4), // #D2E7F4
            Rgb.init(0xF5, 0xA9, 0xB8), // #F5A9B8
        ),
    };

    /// Parses a standard `btop` `.theme` buffer (`theme[key]="#RRGGBB"`) with zero
    /// heap allocations. Unspecified keys inherit `Theme.TRANS_PRIDE`; empty `_mid`
    /// stops (`theme[cpu_mid]=""`) smoothly interpolate between `_start` and `_end`.
    pub fn parseBtop(source: []const u8, theme_name: []const u8) Error!Theme {
        var out: Theme = Theme.TRANS_PRIDE;
        if (theme_name.len > 0) {
            out.setName(theme_name);
        }

        // Track explicit two-stop gradients where `theme[<grad>_mid]=""`.
        var empty_mid: [9]bool = @splat(false);

        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;

            if (!std.mem.startsWith(u8, line, "theme[")) {
                return error.InvalidSyntax;
            }
            const after_prefix = line["theme[".len..];
            const close_bracket = std.mem.indexOfScalar(u8, after_prefix, ']') orelse return error.InvalidSyntax;
            const key = std.mem.trim(u8, after_prefix[0..close_bracket], " \t");
            if (key.len == 0) return error.InvalidSyntax;

            const rest_after_key = std.mem.trimStart(u8, after_prefix[close_bracket + 1 ..], " \t");
            if (rest_after_key.len == 0 or rest_after_key[0] != '=') return error.InvalidSyntax;

            const val_slice = try extractThemeValue(rest_after_key[1..]);
            try applyThemeEntry(&out, &empty_mid, key, val_slice);
        }

        // Resolve any two-stop gradients (`_mid == ""`) as exact midpoint `lerp(start, end, 0.5)`.
        if (empty_mid[0]) out.temp.mid = Rgb.lerp(out.temp.start, out.temp.end, 0.5);
        if (empty_mid[1]) out.cpu.mid = Rgb.lerp(out.cpu.start, out.cpu.end, 0.5);
        if (empty_mid[2]) out.free.mid = Rgb.lerp(out.free.start, out.free.end, 0.5);
        if (empty_mid[3]) out.cached.mid = Rgb.lerp(out.cached.start, out.cached.end, 0.5);
        if (empty_mid[4]) out.available.mid = Rgb.lerp(out.available.start, out.available.end, 0.5);
        if (empty_mid[5]) out.used.mid = Rgb.lerp(out.used.start, out.used.end, 0.5);
        if (empty_mid[6]) out.download.mid = Rgb.lerp(out.download.start, out.download.end, 0.5);
        if (empty_mid[7]) out.upload.mid = Rgb.lerp(out.upload.start, out.upload.end, 0.5);
        if (empty_mid[8]) out.process.mid = Rgb.lerp(out.process.start, out.process.end, 0.5);

        return out;
    }

    /// Loads a standard `btop` `.theme` file from `path` using a stack buffer
    /// (`0` heap allocations).
    pub fn loadFile(path: []const u8) Error!Theme {
        if (path.len == 0 or path.len >= std.fs.max_path_bytes) {
            return error.ThemeFileOpenFailed;
        }
        var path_z: [std.fs.max_path_bytes]u8 = undefined;
        @memcpy(path_z[0..path.len], path);
        path_z[path.len] = 0;

        const fd = std.c.open(@ptrCast(&path_z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.ThemeFileOpenFailed;
        defer _ = std.c.close(fd);

        var file_buf: [MAX_THEME_FILE_BYTES]u8 = undefined;
        var total: usize = 0;
        while (total < file_buf.len) {
            const rc = std.c.read(fd, file_buf[total..].ptr, file_buf.len - total);
            if (rc < 0) {
                if (std.c.errno(rc) == .INTR) continue;
                return error.ThemeFileReadFailed;
            }
            if (rc == 0) break;
            total += @intCast(rc);
        }
        // Check whether the file exceeded MAX_THEME_FILE_BYTES.
        if (total == file_buf.len) {
            var extra: [1]u8 = undefined;
            const extra_rc = std.c.read(fd, &extra, 1);
            if (extra_rc > 0) return error.ThemeFileTooLarge;
        }

        const base = std.fs.path.basename(path);
        const stem = if (std.mem.endsWith(u8, base, ".theme"))
            base[0 .. base.len - ".theme".len]
        else
            base;

        return parseBtop(file_buf[0..total], stem);
    }
};

fn extractThemeValue(raw_rhs: []const u8) Error![]const u8 {
    const rhs = std.mem.trim(u8, raw_rhs, " \t\r");
    if (rhs.len == 0) return "";

    if (rhs[0] == '"' or rhs[0] == '\'') {
        const quote = rhs[0];
        const close_idx = std.mem.indexOfScalarPos(u8, rhs, 1, quote) orelse return error.InvalidSyntax;
        const trailing = std.mem.trim(u8, rhs[close_idx + 1 ..], " \t\r");
        if (trailing.len > 0 and trailing[0] != '#') {
            return error.InvalidSyntax;
        }
        return std.mem.trim(u8, rhs[1..close_idx], " \t");
    }

    // Unquoted value (`theme[main_bg]=#11131D` or `theme[cpu_mid]= # comment`):
    // Note that a leading `#` in an unquoted RHS is a hex color (`#RRGGBB`), while
    // any subsequent `#` preceded by whitespace starts an inline comment.
    var end: usize = rhs.len;
    for (rhs, 0..) |ch, idx| {
        if (idx > 0 and ch == '#' and (rhs[idx - 1] == ' ' or rhs[idx - 1] == '\t')) {
            end = idx;
            break;
        }
    }
    return std.mem.trim(u8, rhs[0..end], " \t");
}

fn applyThemeEntry(
    out: *Theme,
    empty_mid: *[9]bool,
    key: []const u8,
    val: []const u8,
) Error!void {
    // Base chrome keys (empty string keeps default fallback in btop).
    if (std.mem.eql(u8, key, "main_bg")) {
        if (val.len > 0) out.main_bg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "main_fg")) {
        if (val.len > 0) out.main_fg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "title")) {
        if (val.len > 0) out.title = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "hi_fg")) {
        if (val.len > 0) out.hi_fg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "selected_bg")) {
        if (val.len > 0) out.selected_bg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "selected_fg")) {
        if (val.len > 0) out.selected_fg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "inactive_fg")) {
        if (val.len > 0) out.inactive_fg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "graph_text")) {
        if (val.len > 0) out.graph_text = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "meter_bg")) {
        if (val.len > 0) out.meter_bg = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "proc_misc")) {
        if (val.len > 0) out.proc_misc = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "cpu_box")) {
        if (val.len > 0) out.cpu_box = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "mem_box")) {
        if (val.len > 0) out.mem_box = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "net_box")) {
        if (val.len > 0) out.net_box = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "proc_box")) {
        if (val.len > 0) out.proc_box = try Rgb.parseHex(val);
        return;
    }
    if (std.mem.eql(u8, key, "div_line")) {
        if (val.len > 0) out.div_line = try Rgb.parseHex(val);
        return;
    }

    // 9 3-stop gradients (`_start`, `_mid`, `_end`).
    if (try applyGradientStop(&out.temp, &empty_mid[0], "temp", key, val)) return;
    if (try applyGradientStop(&out.cpu, &empty_mid[1], "cpu", key, val)) return;
    if (try applyGradientStop(&out.free, &empty_mid[2], "free", key, val)) return;
    if (try applyGradientStop(&out.cached, &empty_mid[3], "cached", key, val)) return;
    if (try applyGradientStop(&out.available, &empty_mid[4], "available", key, val)) return;
    if (try applyGradientStop(&out.used, &empty_mid[5], "used", key, val)) return;
    if (try applyGradientStop(&out.download, &empty_mid[6], "download", key, val)) return;
    if (try applyGradientStop(&out.upload, &empty_mid[7], "upload", key, val)) return;
    if (try applyGradientStop(&out.process, &empty_mid[8], "process", key, val)) return;

    // Unknown extra `btop` theme keys (e.g. `proc_filter`, `clock_bg`) are ignored
    // for forward compatibility, provided their value (if `#...`) is valid hex.
    if (val.len > 0 and val[0] == '#') {
        _ = try Rgb.parseHex(val);
    }
}

fn applyGradientStop(
    grad: *Gradient3,
    is_empty_mid: *bool,
    comptime prefix: []const u8,
    key: []const u8,
    val: []const u8,
) Error!bool {
    if (std.mem.eql(u8, key, prefix ++ "_start")) {
        if (val.len > 0) grad.start = try Rgb.parseHex(val);
        return true;
    }
    if (std.mem.eql(u8, key, prefix ++ "_mid")) {
        if (val.len == 0) {
            is_empty_mid.* = true;
        } else {
            is_empty_mid.* = false;
            grad.mid = try Rgb.parseHex(val);
        }
        return true;
    }
    if (std.mem.eql(u8, key, prefix ++ "_end")) {
        if (val.len > 0) grad.end = try Rgb.parseHex(val);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "built-in trans-pride theme matches all 15 base chrome colors and 9 3-stop gradients from ADR-0022" {
    const t = Theme.TRANS_PRIDE;
    var hex: [7]u8 = undefined;

    try std.testing.expectEqualStrings("trans-pride", t.name());
    try std.testing.expectEqualStrings("#11131D", t.main_bg.formatHex(&hex));
    try std.testing.expectEqualStrings("#F7F7FC", t.main_fg.formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.title.formatHex(&hex));
    try std.testing.expectEqualStrings("#5BCEFA", t.hi_fg.formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.selected_bg.formatHex(&hex));
    try std.testing.expectEqualStrings("#11131D", t.selected_fg.formatHex(&hex));
    try std.testing.expectEqualStrings("#9297AC", t.inactive_fg.formatHex(&hex));
    try std.testing.expectEqualStrings("#F7F7FC", t.graph_text.formatHex(&hex));
    try std.testing.expectEqualStrings("#343849", t.meter_bg.formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.proc_misc.formatHex(&hex));
    try std.testing.expectEqualStrings("#5BCEFA", t.cpu_box.formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.mem_box.formatHex(&hex));
    try std.testing.expectEqualStrings("#DAEAF4", t.net_box.formatHex(&hex));
    try std.testing.expectEqualStrings("#5BCEFA", t.proc_box.formatHex(&hex));
    try std.testing.expectEqualStrings("#777F94", t.div_line.formatHex(&hex));

    // Verify all 9 3-stop gradients endpoints + midpoints.
    try std.testing.expectEqualStrings("#5BCEFA", t.temp.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#F7F7FC", t.temp.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.temp.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#5BCEFA", t.cpu.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#C1E5F7", t.cpu.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.cpu.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#5BCEFA", t.free.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#A5DFF5", t.free.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#E6F6FF", t.free.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#E9B7D0", t.cached.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#D0CCED", t.cached.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#A1DDF5", t.cached.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#88D5F4", t.available.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#C5E9F9", t.available.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F1FAFF", t.available.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#F5A9B8", t.used.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#E8BFDD", t.used.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#B9DFF3", t.used.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#5BCEFA", t.download.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#B6E6F8", t.download.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F7F7FC", t.download.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#F5A9B8", t.upload.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#F7C8D8", t.upload.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F7F7FC", t.upload.sample(1.0).formatHex(&hex));

    try std.testing.expectEqualStrings("#5BCEFA", t.process.sample(0.0).formatHex(&hex));
    try std.testing.expectEqualStrings("#D2E7F4", t.process.sample(0.5).formatHex(&hex));
    try std.testing.expectEqualStrings("#F5A9B8", t.process.sample(1.0).formatHex(&hex));
}

test "parseBtop parses standard btop .theme syntax, inline comments, empty _mid stops, and rejects bad hex" {
    const sample_btop =
        \\# Custom btop theme file
        \\theme[main_bg]="#0A0B10" # deep background
        \\theme[main_fg]='#E0E4F0'
        \\theme[title] = "#FF88AA"
        \\theme[cpu_start]="#00AAFF"
        \\theme[cpu_mid]=""
        \\theme[cpu_end]="#FF00AA"
        \\theme[unknown_future_key]="#123456"
    ;

    const parsed = try Theme.parseBtop(sample_btop, "custom-neon");
    var hex: [7]u8 = undefined;
    try std.testing.expectEqualStrings("custom-neon", parsed.name());
    try std.testing.expectEqualStrings("#0A0B10", parsed.main_bg.formatHex(&hex));
    try std.testing.expectEqualStrings("#E0E4F0", parsed.main_fg.formatHex(&hex));
    try std.testing.expectEqualStrings("#FF88AA", parsed.title.formatHex(&hex));
    // Unspecified keys inherit trans-pride defaults.
    try std.testing.expectEqualStrings("#5BCEFA", parsed.hi_fg.formatHex(&hex));
    // Empty `cpu_mid` interpolates midpoint of `#00AAFF` and `#FF00AA` -> `#8055D5`.
    try std.testing.expectEqual(Rgb.lerp(Rgb.init(0x00, 0xAA, 0xFF), Rgb.init(0xFF, 0x00, 0xAA), 0.5), parsed.cpu.mid);

    // Invalid hex or malformed lines fail explicitly without swallowing errors.
    try std.testing.expectError(
        error.InvalidHexColor,
        Theme.parseBtop("theme[main_bg]=\"#ZZZZZZ\"\n", "bad"),
    );
    try std.testing.expectError(
        error.InvalidSyntax,
        Theme.parseBtop("not_a_theme_entry = 123\n", "bad"),
    );
}

test "Theme.loadFile loads a custom btop .theme file from disk with zero heap allocations" {
    const tmp_dir: []const u8 = if (std.c.getenv("TMPDIR")) |p|
        std.mem.sliceTo(p, 0)
    else
        ".zig-cache";
    const trimmed_dir = std.mem.trimEnd(u8, tmp_dir, "/");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], "{s}/aminfer_test_custom_btop.theme", .{trimmed_dir});
    path_buf[path.len] = 0;
    const path_z: [:0]const u8 = path_buf[0..path.len :0];

    const fd = std.c.open(path_z.ptr, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
        .CLOEXEC = true,
    }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.unlink(path_z.ptr);
    const content =
        \\# Disk theme test
        \\theme[main_bg]="#101218"
        \\theme[cpu_box]="#88CCFF"
        \\theme[temp_start]="#112233"
        \\theme[temp_mid]="#445566"
        \\theme[temp_end]="#778899"
        \\
    ;
    const wrc = std.c.write(fd, content.ptr, content.len);
    _ = std.c.close(fd);
    try std.testing.expectEqual(@as(isize, @intCast(content.len)), wrc);

    const loaded = try Theme.loadFile(path);
    var hex: [7]u8 = undefined;
    try std.testing.expectEqualStrings("aminfer_test_custom_btop", loaded.name());
    try std.testing.expectEqualStrings("#101218", loaded.main_bg.formatHex(&hex));
    try std.testing.expectEqualStrings("#88CCFF", loaded.cpu_box.formatHex(&hex));
    try std.testing.expectEqualStrings("#112233", loaded.temp.start.formatHex(&hex));
    try std.testing.expectEqualStrings("#445566", loaded.temp.mid.formatHex(&hex));
    try std.testing.expectEqualStrings("#778899", loaded.temp.end.formatHex(&hex));
    // Unspecified keys inherit trans-pride defaults.
    try std.testing.expectEqualStrings("#F5A9B8", loaded.title.formatHex(&hex));
}
