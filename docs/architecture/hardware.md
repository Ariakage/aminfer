# Hardware model

> Design reference; status lives in [../status.md](../status.md).

## Baseline machine: measured facts

Measured on 2026-11-20 on the baseline machine: Apple M4, 24 GiB, macOS 27.2,
Xcode 27.1, Metal Toolchain 27A266a, Zig 0.17.1. The probes were throwaway
programs outside the repository; `aminfer probe` (status rows F3 and F9)
re-measures these facts from the repository.

| Fact | Value | Source |
| --- | --- | --- |
| CPU | 4 P-cores + 6 E-cores | `hw.perflevel0/1.logicalcpu` |
| P-cluster L2 | 16 MiB | `hw.perflevel0.l2cachesize` |
| Unified memory | 24 GiB (25,769,803,776 B) | `hw.memsize` |
| Page size | 16,384 B | `hw.pagesize` |
| `mlock` limits | `vm.user_wire_limit` = `vm.global_user_wire_limit` = 19,585,050,869 B (≈ 18.24 GiB); `RLIMIT_MEMLOCK` unlimited | `sysctl`, `ulimit -l` |
| GPU wired limit | `iogpu.wired_limit_mb` = 0 (system default) | `sysctl` |
| Metal working set | `recommendedMaxWorkingSetSize` = 18,186 MiB | `MTLDevice` |
| Largest buffer | `maxBufferLength` = 13,639 MiB | `MTLDevice` |
| GPU family | Apple9 (not Apple10) | `supportsFamily:` |
| Neural Engine | 16 cores | `MLNeuralEngineComputeDevice.totalCoreCount` |
| Metal 4 | `MTL4CommandQueue`, `MTL4CommandAllocator` available; `MTLResidencySet` available | `respondsToSelector:`, SDK headers (`API_AVAILABLE(macos(26.0))`) |
| Linear texture alignment | 16 B for R16Float, RG16Float, RGBA16Float | `minimumLinearTextureAlignmentForPixelFormat:` |
| IOSurface fp16 tile (128 × 512, `OneComponent16Half`) | `bytesPerRow` 256 (multiple of 64 B); base address 16 KiB aligned; `IOSurfaceGetAllocSize` 131,136 B — not a 16 KiB multiple | CoreVideo / IOSurface |
| Texture views of that surface | R16Float 128 × 512 and RGBA16Float 32 × 512 both created, also with `MTL_DEBUG_LAYER=1` | `newTextureWithDescriptor:iosurface:plane:` |
| Buffer alias of that surface | `newBufferWithBytesNoCopy` over the surface base succeeded; an RGBA16Float linear texture view of that buffer succeeded | Metal |
| NoCopy length check | Metal accepted a NoCopy length that is not a 16 KiB multiple, also with `MTL_DEBUG_LAYER=1` | Metal — so our code must enforce the rule |
| `MLMultiArray(pixelBuffer:shape:)` | Accepted the fp16 surface with shape `[4, 128, 128]`, strides `[16384, 128, 1]` | CoreML |
| Async file I/O | `aio_read` with `SIGEV_KEVENT` delivers `EVFILT_AIO`; `kevent64.ext[0]` = error, `ext[1]` = bytes; `aio_return` afterwards fails with `EINVAL` | probe |
| AIO limits | `kern.aiothreads` = 4, `kern.aioprocmax` = 256, `kern.aiomax` = 1024 | `sysctl` |
| Direct I/O | `fcntl(fd, F_NOCACHE, 1)` succeeds | probe |
| SSD | APPLE SSD AP0512Z, 500 GB | `system_profiler` |
| Thunderbolt | Up to 40 Gb/s (Thunderbolt 4 class) | `system_profiler` |
| Metal compiler | metalfe-32023.921; `-std=metal4.0 -mmacosx-version-min=26.0` compiles | `xcrun metal` |

## Bandwidth and compute

- **One memory system.** CPU, GPU and ANE share the unified LPDDR memory
  (120 GB/s on M4 per Apple's specification). Every engine's weight traffic
  counts against the same budget.
- **ANE memory behaviour.** The ANE's on-chip SRAM is megabyte-scale: a 2 MB
  working-set threshold was measured on M1, and an M4 benchmark places a
  throughput cliff near 32 MB. A 0.5B-class draft model (≈ 300 MB at 4 bits)
  must therefore stream from DRAM. Reported measurements also show that int8
  weights do not raise ANE compute throughput (they are expanded to fp16), so
  quantization helps the ANE through bandwidth, not FLOPs.
- **Consequence.** "L2 on the ANE does not touch GPU bandwidth" does not hold.
  The ANE adds parallel compute; the scheduler accounts for its DRAM bytes
  ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)) and spike
  S4 measures the contention ([../process.md](../process.md#spikes)).

## Core steering

- macOS offers no hard CPU affinity on Apple Silicon. Placement is steered
  with QoS classes: `QOS_CLASS_USER_INTERACTIVE` / `USER_INITIATED` favour
  P-cores; `QOS_CLASS_UTILITY` / `BACKGROUND` favour or restrict to E-cores.
- Low QoS classes also lower disk I/O priority (background work is throttled).
  Prefetch threads therefore keep an E-core QoS but raise their own I/O policy
  with `setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_IMPORTANT)`.
  Whether this policy also covers requests served by the kernel AIO workers is
  unverified — spike S3.

## Scaling notes

- Larger chips raise memory bandwidth and the wired budget; the cost models
  (tree pruning, placement gate, and GPU-vs-ANE prefill crossover) and
  `spec_bus_gbs` telemetry are calibrated per chip, not hard-coded to M4's
  `120.0 GB/s` ([ADR-0021](../decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md)).
- One `arm64` binary and one embedded `aminfer.metallib` target `M1–M3`
  (NEON/Accelerate + `simdgroup_matrix`), `M4` (`SME2` + `Apple9`
  `simdgroup_matrix`), and `M5+` (`SME2` + `Apple10+` GPU `NAX`
  `MetalPerformancePrimitives` `mpp::tensor_ops::matmul2d`) concurrently via
  runtime `SmeCaps.probe()`, lazy per-entry-point `MTL4Compiler` pipeline
  specialization, and `G16` startup canary fallback
  ([ADR-0021](../decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md)).
- Thunderbolt 5 Macs on macOS 26.2+ support RDMA over Thunderbolt (Apple
  TN3205); the Thunderbolt 4 baseline is limited to IP over Thunderbolt Bridge.
  Both only matter for the future `thunderbolt_peer` expert source.

## Sources

- Apple, *init(pixelBuffer:shape:)* — fp16 pixel buffers must be
  `kCVPixelFormatType_OneComponent16Half`:
  <https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:)>
- Apple, *TN3205: Low-latency communication with RDMA over Thunderbolt*:
  <https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt>
- *Apple Neural Engine* (arXiv 2606.22283) — M1 roofline, 2 MB on-chip working
  set: <https://arxiv.org/html/2606.22283>
- maderix, *Inside the M4 Apple Neural Engine, Part 2: ANE Benchmarks* — SRAM
  cliff, int8 vs fp16 throughput:
  <https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615>
