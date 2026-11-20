//! The single build hub of aminfer (ADR-0001): Zig engine core, the
//! Objective-C hardware bridge, the Metal kernel library and all tests.

const std = @import("std");

/// Platform floor: macOS 26.0 on Apple Silicon (ADR-0011).
const macos_floor: std.SemanticVersion = .{ .major = 26, .minor = 0, .patch = 0 };

/// Apple frameworks the bridge may call (ADR-0001, ADR-0004).
const frameworks = [_][]const u8{ "Foundation", "Metal", "CoreML", "IOSurface", "CoreVideo" };

const objc_flags = [_][]const u8{ "-fobjc-arc", "-Wall", "-Wextra", "-Werror" };

const kernel_sources = [_][]const u8{
    "src/kernels/smoke.metal",
    "src/kernels/dequant_gemm.metal",
    "src/kernels/kv_attention.metal",
    "src/kernels/moe_ffn.metal",
};

pub fn build(b: *std.Build) void {
    // Native CPU features (Apple M-series), deployment floor macOS 26.0.
    const target = b.standardTargetOptions(.{ .default_target = .{
        .cpu_model = .native,
        .os_version_min = .{ .semver = macos_floor },
    } });
    const optimize = b.standardOptimizeOption(.{});

    if (target.result.cpu.arch != .aarch64 or target.result.os.tag != .macos) {
        std.process.fatal("aminfer only targets aarch64-macos (Apple Silicon)", .{});
    }

    // Pure C ABI of the bridge, translated for Zig (Zig 0.17 has no @cImport).
    const bridge_c = b.addTranslateC(.{
        .root_source_file = b.path("src/bridge/apple_bridge.h"),
        .target = target,
        .optimize = optimize,
    });

    // All Metal kernels compiled into one library that the binary embeds.
    const metal = b.addSystemCommand(&.{
        "xcrun",         "-sdk",  "macosx",  "metal",
        "-std=metal4.0", "-Wall", "-Werror", "-mmacosx-version-min=26.0",
    });
    for (kernel_sources) |src| metal.addFileArg(b.path(src));
    metal.addArg("-o");
    const metallib = metal.addOutputFileArg("aminfer.metallib");

    const mod = b.addModule("aminfer", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("apple_bridge", bridge_c.createModule());
    mod.addAnonymousImport("aminfer_metallib", .{ .root_source_file = metallib });
    mod.addIncludePath(b.path("src/bridge"));
    addMacosSdkPaths(b, mod);
    mod.addCSourceFiles(.{ .files = &.{"src/bridge/apple_bridge.m"}, .flags = &objc_flags });
    for (frameworks) |name| mod.linkFramework(name, .{});

    const exe = b.addExecutable(.{
        .name = "aminfer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "aminfer", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the aminfer CLI").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module })).step);
}

/// Overriding the minimum OS version makes the target non-native, and Zig then
/// stops locating the macOS SDK on its own. Resolve it with `xcrun` instead.
fn addMacosSdkPaths(b: *std.Build, mod: *std.Build.Module) void {
    const sdk = std.mem.trimEnd(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), "\n");
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "System/Library/Frameworks" }) });
    mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/include" }) });
    mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/lib" }) });
}
