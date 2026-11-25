//! The single build hub of aminfer (ADR-0001): Zig engine core, the
//! Objective-C hardware bridge, the Metal kernel library and all tests.

const std = @import("std");

/// Platform floor: macOS 26.0 on Apple Silicon (ADR-0011).
const macos_floor: std.SemanticVersion = .{ .major = 26, .minor = 0, .patch = 0 };

/// Apple frameworks the bridge may call (ADR-0001, ADR-0004, ADR-0018, ADR-0019).
const frameworks = [_][]const u8{
    "Foundation",
    "Metal",
    "CoreML",
    "IOSurface",
    "CoreVideo",
    "Accelerate",
    "VideoToolbox",
    "CoreMedia",
    "CoreGraphics",
    "ImageIO",
};

const objc_flags = [_][]const u8{ "-fobjc-arc", "-Wall", "-Wextra", "-Werror" };

/// Scoped strictly to `src/cpu/sme2_kernels.s` (ADR-0018, ADR-0001): never applied
/// to global Zig target flags so M1–M3 builds stay free of implicit SME instructions.
/// Note: LLVM/Clang includes SME f16f32/b16f32 in `+sme+sme2`.
const sme2_asm_flags = [_][]const u8{
    "-Wa,-march=armv9.2-a+sme+sme2+sme-i16i64+sme-f64f64",
};

const kernel_sources = [_][]const u8{
    "src/kernels/smoke.metal",
    "src/kernels/dequant_gemm.metal",
    "src/kernels/quant_formats.metal",
    "src/kernels/kv_attention.metal",
    "src/kernels/moe_ffn.metal",
    "src/kernels/hybrid_ops.metal",
    "src/kernels/vision_ingress.metal",
    "src/kernels/nax_tensorops.metal",
    "src/kernels/sampling.metal",
    "src/kernels/inkling2_hy4.metal",
    "src/kernels/dflare_gpu.metal",
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
    const bridge_mod = bridge_c.createModule();

    // Public C ABI header (`include/aminfer.h`), translated so `c_api.zig`
    // verifies struct sizes and ABI constants at compile time (`C17`).
    const sdk_c = b.addTranslateC(.{
        .root_source_file = b.path("include/aminfer.h"),
        .target = target,
        .optimize = optimize,
    });

    // All Metal kernels compiled into one library that the binary embeds
    // (including Apple10+ MetalPerformancePrimitives NAX TensorOps, ADR-0021).
    const sdk = sdkPath(b);
    const metal = b.addSystemCommand(&.{
        "xcrun",         "-sdk",  "macosx",  "metal",
        "-std=metal4.0", "-Wall", "-Werror", "-mmacosx-version-min=26.0",
        "-F",
    });
    metal.addArg(b.fmt("{s}/System/Library/Frameworks", .{sdk}));
    for (kernel_sources) |src| metal.addFileArg(b.path(src));
    metal.addArg("-o");
    const metallib = metal.addOutputFileArg("aminfer.metallib");

    const mod = b.addModule("aminfer", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("apple_bridge", bridge_mod);
    mod.addAnonymousImport("aminfer_metallib", .{ .root_source_file = metallib });
    mod.addImport("aminfer", mod);
    mod.addIncludePath(b.path("src/bridge"));
    addMacosSdkPaths(b, mod, sdk);
    mod.addCSourceFiles(.{ .files = &.{"src/bridge/apple_bridge.m"}, .flags = &objc_flags });
    mod.addCSourceFiles(.{ .files = &.{"src/cpu/sme2_kernels.s"}, .flags = &sme2_asm_flags });
    for (frameworks) |name| mod.linkFramework(name, .{});

    // Public C-ABI shared library (`zig-out/lib/libaminfer.dylib` + `zig-out/include/aminfer.h`,
    // ADR-0022 `C17`); keeps the standalone `aminfer` CLI binary completely unchanged.
    const sdk_mod = b.createModule(.{
        .root_source_file = b.path("src/sdk/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "aminfer", .module = mod },
            .{ .name = "apple_bridge", .module = bridge_mod },
            .{ .name = "aminfer_sdk_c", .module = sdk_c.createModule() },
        },
    });
    addMacosSdkPaths(b, sdk_mod, sdk);
    for (frameworks) |name| sdk_mod.linkFramework(name, .{});
    sdk_mod.linkFramework("IOKit", .{});

    const dylib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "aminfer",
        .root_module = sdk_mod,
    });
    dylib.installHeader(b.path("include/aminfer.h"), "aminfer.h");
    b.installArtifact(dylib);

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

    const aux_wf = b.addWriteFiles();

    // Dedicated SME2-only test step (`zig build test-sme`), filtering the real
    // `cpu.sme.*` tests from the root library module without recompiling duplicates.
    const sme_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{"cpu.sme."},
    });
    const run_sme_test = b.addRunArtifact(sme_test);
    b.step("test-sme", "Run SME2 microkernel, canary, parity and fallback tests").dependOn(&run_sme_test.step);

    // Dedicated M6 hybrid GPU test step (`zig build test-hybrid`), filtering the
    // real `gpu.hybrid.*` tests from the root library module.
    const hybrid_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{"gpu.hybrid."},
    });
    const run_hybrid_test = b.addRunArtifact(hybrid_test);
    b.step("test-hybrid", "Run M6 hybrid GPU (GDN, QSA, 4-branch GR) parity tests").dependOn(&run_hybrid_test.step);

    // Dedicated M6.5 G14 vision/video ingress test step (`zig build test-vision`),
    // filtering the real `gpu.vision.*` tests from the root library module.
    const vision_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{"gpu.vision."},
    });
    const run_vision_test = b.addRunArtifact(vision_test);
    b.step("test-vision", "Run M6.5 G14 zero-copy VideoToolbox/IOSurface vision ingress tests").dependOn(&run_vision_test.step);

    // Dedicated M7 cluster transport & expert-parallel test step (`zig build test-cluster`),
    // filtering `cluster.*` and `expert.source.*` tests from the root library module.
    const cluster_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{ "cluster.", "expert.source." },
    });
    const run_cluster_test = b.addRunArtifact(cluster_test);
    b.step("test-cluster", "Run M7 cluster transport (TB4 TCP / TB5 RDMA probe) and expert-parallel tests").dependOn(&run_cluster_test.step);

    const s7_main = aux_wf.add("s7_standalone_main.zig",
        \\const std = @import("std");
        \\const aminfer = @import("aminfer");
        \\
        \\pub fn main() !void {
        \\    const gpa = std.heap.c_allocator;
        \\    const report = try aminfer.sme.runSpikeS7Standalone(gpa);
        \\    var aw: std.Io.Writer.Allocating = .init(gpa);
        \\    defer aw.deinit();
        \\    try report.write(&aw.writer);
        \\    var written: usize = 0;
        \\    const buf = aw.written();
        \\    while (written < buf.len) {
        \\        const rc = std.c.write(1, buf.ptr + written, buf.len - written);
        \\        if (rc <= 0) return error.WriteFailed;
        \\        written += @intCast(rc);
        \\    }
        \\}
        \\
    );
    const s7_exe = b.addExecutable(.{
        .name = "bench-s7-standalone",
        .root_module = b.createModule(.{
            .root_source_file = s7_main,
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "aminfer", .module = mod }},
        }),
    });
    const run_s7 = b.addRunArtifact(s7_exe);
    b.step("bench-s7-standalone", "Run standalone Spike S7 SME2 vs Accelerate benchmark").dependOn(&run_s7.step);

    const sdk_test = b.addTest(.{ .root_module = sdk_mod });
    const run_sdk_test = b.addRunArtifact(sdk_test);
    b.step("test-sdk", "Run M7.5 C17 libaminfer C-ABI unit tests").dependOn(&run_sdk_test.step);

    // Dedicated M8.1 G23/G24 multi-format quantization test step (`zig build test-quant-formats`),
    // filtering `gpu.quant_formats.*` and `quant.formats.*` from the unified root library module.
    const quant_formats_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{ "gpu.quant_formats.", "quant.formats." },
    });
    const run_quant_formats_test = b.addRunArtifact(quant_formats_test);
    b.step("test-quant-formats", "Run M8.1 G23/G24 multi-format Metal 4 quantization tests").dependOn(&run_quant_formats_test.step);

    // Dedicated M8.2 G25 Inkling 2 and Hy4 GPU operator test step (`zig build test-m82-gpu`),
    // filtering `gpu.inkling2_hy4.` from the unified root library module.
    const m82_gpu_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{"gpu.inkling2_hy4."},
    });
    const run_m82_gpu_test = b.addRunArtifact(m82_gpu_test);
    b.step("test-m82-gpu", "Run M8.2 G25 Inkling 2 and Hy4 fused Metal 4 operator tests").dependOn(&run_m82_gpu_test.step);

    // Dedicated M8.2 D14 + C26 test step (`zig build test-m82-draft`), filtering
    // `cpu.inkling2_ctrl.*`, `spec.dflare_dfly.*`, and `storage.ple.*` from `mod`.
    const m82_draft_test = b.addTest(.{
        .root_module = mod,
        .filters = &.{ "cpu.inkling2_ctrl.", "spec.dflare_dfly.", "storage.ple." },
    });
    const run_m82_draft_test = b.addRunArtifact(m82_draft_test);
    b.step("test-m82-draft", "Run M8.2 D14 DFlare/DFly2, C26 SME2 loop controllers, and FP8 Engram tests").dependOn(&run_m82_draft_test.step);

    const run_mod_test = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    const run_exe_test = b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module }));
    run_exe_test.step.dependOn(&run_mod_test.step);
    run_sdk_test.step.dependOn(&run_exe_test.step);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_mod_test.step);
    test_step.dependOn(&run_exe_test.step);
    test_step.dependOn(&run_sdk_test.step);
}

/// Resolves the active macOS SDK path once per build invocation.
fn sdkPath(b: *std.Build) []const u8 {
    return std.mem.trimEnd(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), "\r\n");
}

/// Overriding the minimum OS version makes the target non-native, and Zig then
/// stops locating the macOS SDK on its own. Resolve it with `xcrun` instead.
fn addMacosSdkPaths(b: *std.Build, mod: *std.Build.Module, sdk: []const u8) void {
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "System/Library/Frameworks" }) });
    mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/include" }) });
    mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/lib" }) });
}
