# Benchmark Results and Raw Telemetry

The single canonical source of truth for measured performance metrics,
benchmark methodology, environment baselines, privileged system-tuning
verification, and verbatim raw run outputs ([ADR-0007](../decisions/0007-docs-single-source-of-truth.md)).
Capability status lives in [../status.md](../status.md); milestone progress
lives in [../process.md](../process.md).

## Environment and hardware baseline

| Field | Value | Source |
| --- | --- | --- |
| Measurement date | 2026-11-20 | local run |
| Git commit SHA | `16322279e9e29b96942257c0cee1546f3c3183da` | `git rev-parse HEAD` |
| Machine model | `Mac16,12` | `powermetrics` |
| SoC / GPU | Apple M4 (`Apple9` GPU family, Metal 4, unified memory) | `aminfer probe` |
| CPU topology | 4 P-cores (16 MiB P-cluster L2) + 6 E-cores | `aminfer probe`, [hardware.md](../architecture/hardware.md) |
| Neural Engine | 16 cores | `aminfer probe` |
| Unified memory | 24 GiB (24,576 MiB / 25,769,803,776 B), 120.0 GB/s nominal LPDDR bus | `aminfer probe`, [hardware.md](../architecture/hardware.md) |
| Page size | 16,384 B (16 KiB) | `aminfer probe` |
| SSD | APPLE SSD AP0512Z, 500 GB | [hardware.md](../architecture/hardware.md) |
| OS | macOS 27.2 (Build `26B123`) | `sw_vers` |
| Zig | `0.17.1` | `zig version` |
| Xcode | `27.1` (Build `27A9275`) | `xcodebuild -version` |
| Metal Toolchain | `27A266a` (`MetalToolchain-v27.1.266.1`, `metalfe-32023.921`, deployment floor `macOS 26.0`) | `xcrun --find metal`, [hardware.md](../architecture/hardware.md) |

### Verbatim baseline `aminfer probe` output

Captured non-root under default kernel limits (`iogpu.wired_limit_mb = 0`,
`vm.user_wire_limit = vm.global_user_wire_limit = 19585050869`):

```text
sys.page_size                    16384
sys.memory_mib                   24576
cpu.performance_cores            4
cpu.efficiency_cores             6
wire.user_limit_bytes            19585050869
wire.global_limit_bytes          19585050869
wire.gpu_limit_mib               0
gpu.name                         Apple M4
gpu.apple_family                 9
gpu.metal4                       true
gpu.unified_memory               true
gpu.recommended_working_set_mib  18186
gpu.max_buffer_mib               13639
ane.cores                        16
budget.wired_limit_mib           18186
budget.binding_limit             MTLDevice.recommendedMaxWorkingSetSize
```

## Methodology

1. **Build and safety gates**: All code and kernel changes pass `zig build`
   (Debug), `zig build test` (Debug), `zig build test -Doptimize=ReleaseSafe`
   (alignment and safety assertions enabled), and direct test execution under
   `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` before building the benchmark
   binary with `zig build -Doptimize=ReleaseFast`.
2. **Single wired arena and SLC busting**: `src/bench/runner.zig` validates a
   single 81 MiB `WiredArena` (`mlock`-pinned, 16 KiB page-aligned) against
   `Budget.fromHost` (`18,186 MiB` default wired ceiling), wraps it once with
   zero-copy `newBufferWithBytesNoCopy`, and registers it in an
   `MTLResidencySet`. Weights rotate across 64-byte-aligned slots spanning a
   `>= 64 MiB` `qweight` pool plus a `16 MiB` `scale_bias` pool filled with
   deterministic non-zero PRNG data so repeated dispatches exceed Apple M4
   L2/SLC capacity and stream from unified DRAM.
3. **Exact byte accounting (C9)**: Per-dispatch algorithmic memory traffic
   (`bytesMovedPerDispatch` in [runner.zig](../../src/bench/runner.zig)) is
   the sum of weight, affine scale/bias, input, and output bytes:
   - `qweight_bytes = N * (K / 2)` for 4-bit (`q4`) or `N * (K * 3 / 8)` for
     3-bit (`q3`, group size 64);
   - `scale_bias_bytes = N * (K / 64) * 4` (`half2(scale, bias)`);
   - `x_bytes = M * K * 2` (`f16`);
   - `y_bytes = M * N * 2` (`f16`);
   - `bytes_per_dispatch = qweight_bytes + scale_bias_bytes + x_bytes + y_bytes`.
   Effective bandwidth is reported in decimal GB/s
   (`beta = (bytes_per_dispatch * dispatches) / elapsed_ns`, where
   `elapsed_ns` is measured with `CLOCK_UPTIME_RAW`), and bus utilization is
   `(beta / 120.0) * 100%`. Case 0 executes `16` extra DVFS warmup batches
   plus `4` standard warmup batches before timing; each subsequent case runs
   `4` warmup batches. Timed windows run `8 × 32 = 256` dispatches for GEMV
   (`M = 1`) and `4 × 16 = 64` dispatches for GEMM (`M = 16, 64`), with
   intra-frame barriers (`insert_barrier_before`) between dispatches and a
   finite-output check on `y` after each case.
4. **Spike S2 Metal 4 steady-state protocol**: A 2-frame Metal 4 `Compute`
   ring (`MTL4CommandQueue`, 2 reused `MTL4CommandBuffer` +
   `MTL4CommandAllocator` pairs, uniform ring streaming, `MTLSharedEvent`)
   executes `2,048` warmup rounds (`32` dispatches/round of `gemv_q4`
   `1×16×64`) to fill Apple's bounded internal Metal 4 driver pool (~2,000
   entries, ~128 KiB), followed by `1,000` steady-state timed rounds (`32,000`
   dispatches). Prior work on the slot being reused is drained outside the
   CPU encode timer while the sibling slot remains in flight on the GPU.
   `malloc_zone_statistics(null, &stats)` deltas (`blocks_in_use`,
   `size_in_use`, `size_allocated`) are sampled immediately before and after
   the 1,000 steady-state rounds.
5. **Zero-swap gate**: Native Mach
   `host_statistics64(mach_host_self(), HOST_VM_INFO64, ...)` (`swapouts`,
   `pageouts`) and `sysctlbyname("vm.swapusage")` wrap the full benchmark run;
   `vmstat.checkNoSwap` aborts with `error.SwapOccurred` if `swapouts` grows.
6. **`powermetrics` sampling protocol**: Under default kernel tunables
   (`iogpu.wired_limit_mb = 0`), `./zig-out/bin/aminfer bench` is started
   non-root in the background while
   `/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 6`
   samples CPU/GPU/ANE frequencies, residencies, and power over a ~3.0 s
   window (six 500 ms intervals). Because `powermetrics` polling and active
   host desktop threads perturb latency, standalone unperturbed `aminfer bench`
   is recorded separately from the concurrent telemetry run.

## M1 C9 kernel bandwidth and Spike S2 (standalone ReleaseFast)

Measured on 2026-11-20 at commit `16322279` via standalone non-root
`./zig-out/bin/aminfer bench` (ReleaseFast, default kernel tunables,
`wired_budget_mib = 18186`, `arena_mib = 81`, `swapouts_delta = 0`,
`pageouts_delta = 0`).

### C9 kernel bandwidth calibration summary

| Kernel | Shape (`M×N×K`) | Bytes/dispatch | Weight slots | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gemv_q4` | `1×2048×2048` | 2,367,488 | 32 | 29.38 | 80.58 | 67.2% |
| `gemv_q4` | `1×6144×2048` | 7,094,272 | 10 | 90.86 | 78.08 | 65.1% |
| `gemv_q4` | `1×2048×768` | 890,368 | 85 | 14.31 | 62.21 | 51.8% |
| `gemv_q4` | `1×4096×4096` | 9,453,568 | 8 | 95.74 | 98.74 | 82.3% |
| `gemv_q3` | `1×2048×2048` | 1,843,200 | 42 | 24.41 | 75.51 | 62.9% |
| `gemv_q3` | `1×6144×2048` | 5,521,408 | 14 | 63.22 | 87.34 | 72.8% |
| `gemv_q3` | `1×2048×768` | 693,760 | 113 | 11.04 | 62.83 | 52.4% |
| `gemv_q3` | `1×4096×4096` | 7,356,416 | 10 | 86.89 | 84.67 | 70.6% |
| `gemm_q4` | `16×2048×2048` | 2,490,368 | 32 | 105.82 | 23.54 | 19.6% |
| `gemm_q4` | `16×4096×4096` | 9,699,328 | 8 | 401.20 | 24.18 | 20.1% |
| `gemm_q4` | `64×2048×2048` | 2,883,584 | 32 | 210.40 | 13.71 | 11.4% |
| `gemm_q4` | `64×4096×4096` | 10,485,760 | 8 | 812.83 | 12.90 | 10.8% |
| `gemm_q3` | `16×2048×2048` | 1,966,080 | 42 | 103.55 | 18.99 | 15.8% |
| `gemm_q3` | `16×4096×4096` | 7,602,176 | 10 | 403.27 | 18.85 | 15.7% |
| `gemm_q3` | `64×2048×2048` | 2,359,296 | 42 | 209.73 | 11.25 | 9.4% |
| `gemm_q3` | `64×4096×4096` | 8,388,608 | 10 | 828.26 | 10.13 | 8.4% |

Notes:
- Peak decode GEMV bandwidth reached **98.74 GB/s (82.3% of the 120 GB/s bus)**
  for `gemv_q4` (`1×4096×4096`) and **87.34 GB/s (72.8%)** for `gemv_q3`
  (`1×6144×2048`).
- Batched `gemm_q4`/`gemm_q3` (`M = 16, 64`) reuse weights across `M` rows, so
  effective *memory* bandwidth drops as compute intensity rises (each loaded
  weight byte feeds 16–64× more FMAs).

### Spike S2 steady-state hot-path summary

| Metric | Value |
| --- | --- |
| Measured rounds × dispatches/round | 1,000 × 32 (after 2,048 warmup rounds) |
| CPU encode time | 7.78 us/round (243.3 ns/dispatch) |
| Full pipelined round time | 98.23 us/round |
| `malloc_zone_statistics` `blocks_in_use` delta | 0 |
| `malloc_zone_statistics` `size_in_use` delta | 0 B |
| `malloc_zone_statistics` `size_allocated` delta | 0 B |
| System `swapouts` / `pageouts` delta | 0 / 0 |
| System `swap.used_mib` at end of run | 3,014 MiB (pre-existing macOS background swap; zero new swapouts during run) |

### Verbatim standalone `aminfer bench` raw stdout

```text
bench.gpu                        Apple M4
bench.wired_budget_mib           18186
bench.arena_mib                  81
bench.bus_peak_gbs               120.0

C9 Kernel Bandwidth Calibration
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q4   1x2048x2048           2367488      32         29.38        80.58     67.2
gemv_q4   1x6144x2048           7094272      10         90.86        78.08     65.1
gemv_q4   1x2048x768             890368      85         14.31        62.21     51.8
gemv_q4   1x4096x4096           9453568       8         95.74        98.74     82.3
gemv_q3   1x2048x2048           1843200      42         24.41        75.51     62.9
gemv_q3   1x6144x2048           5521408      14         63.22        87.34     72.8
gemv_q3   1x2048x768             693760     113         11.04        62.83     52.4
gemv_q3   1x4096x4096           7356416      10         86.89        84.67     70.6
gemm_q4   16x2048x2048          2490368      32        105.82        23.54     19.6
gemm_q4   16x4096x4096          9699328       8        401.20        24.18     20.1
gemm_q4   64x2048x2048          2883584      32        210.40        13.71     11.4
gemm_q4   64x4096x4096         10485760       8        812.83        12.90     10.8
gemm_q3   16x2048x2048          1966080      42        103.55        18.99     15.8
gemm_q3   16x4096x4096          7602176      10        403.27        18.85     15.7
gemm_q3   64x2048x2048          2359296      42        209.73        11.25      9.4
gemm_q3   64x4096x4096          8388608      10        828.26        10.13      8.4

Spike S2 Metal 4 Steady-State Hot Path
s2.rounds                        1000
s2.dispatches_per_round          32
s2.cpu_encode_us_per_round       7.78
s2.cpu_encode_ns_per_dispatch    243.3
s2.full_round_us                 98.23
s2.malloc_blocks_in_use_delta    0
s2.malloc_size_in_use_delta      0
s2.malloc_size_allocated_delta   0

Swap Check
swap.status                      ok
swap.swapouts_delta              0
swap.pageouts_delta              0
swap.used_mib                    3014
```

## Privileged tunables (`sysctl`) verification

Tested on 2026-11-20 on Apple M4 24 GiB, macOS 27.2 (Build `26B123`) for the
21 GiB "everyday raised limits" profile in
[../guides/system-tuning.md](../guides/system-tuning.md).

### Summary of `sysctl` writes and `aminfer probe` delta

| Parameter / probe field | Default baseline | Under 21 GiB profile | After restore | Result |
| --- | ---: | ---: | ---: | --- |
| `iogpu.wired_limit_mb` | `0` | `21504` | `0` | Write & restore exit code `0` |
| `vm.global_user_wire_limit` | `19585050869` | `22548578304` | `19585050869` | Write & restore exit code `0` |
| `vm.user_wire_limit` | `19585050869` | `22548578304` | `19585050869` | Write & restore exit code `0` |
| `gpu.recommended_working_set_mib` | `18186` | `21504` (+3,318 MiB) | `18186` | Responded immediately in new non-root process |
| `gpu.max_buffer_mib` | `13639` | `13639` (unchanged) | `13639` | Single-buffer cap unchanged; large models require multi-buffer sharding |
| `budget.wired_limit_mib` | `18186` | `21504` (+3,318 MiB) | `18186` | Wired ceiling expanded by 3,318 MiB |
| `budget.binding_limit` | `MTLDevice.recommendedMaxWorkingSetSize` | `physical_headroom` | `MTLDevice.recommendedMaxWorkingSetSize` | ADR-0012 3 GiB headroom clamp (`24,576 - 3,072 = 21,504 MiB`) engaged cleanly |

Key operational notes verified on macOS 27.2:
- All three tunables (`iogpu.wired_limit_mb`, `vm.global_user_wire_limit`,
  `vm.user_wire_limit`) are writable by root on macOS 27.2 with standard
  `sysctl -w` (no SIP/entitlement rejection).
- Ordering: raise `vm.global_user_wire_limit` before `vm.user_wire_limit`, and
  lower `vm.user_wire_limit` before `vm.global_user_wire_limit` on restore.
- An independent non-root `sysctl` check after the privileged session exited
  re-confirmed `iogpu.wired_limit_mb: 0`,
  `vm.global_user_wire_limit: 19585050869`, and
  `vm.user_wire_limit: 19585050869`.

### Verbatim `sysctl` and `aminfer probe` verification output

```text
=== INITIAL SYSCTL ===
iogpu.wired_limit_mb: 0
vm.global_user_wire_limit: 19585050869
vm.user_wire_limit: 19585050869
=== ATTEMPT SYSCTL WRITES ===
--- sysctl -w iogpu.wired_limit_mb=21504 ---
iogpu.wired_limit_mb: 0 -> 21504
exit_code=0
--- sysctl -w vm.global_user_wire_limit=22548578304 ---
vm.global_user_wire_limit: 19585050869 -> 22548578304
exit_code=0
--- sysctl -w vm.user_wire_limit=22548578304 ---
vm.user_wire_limit: 19585050869 -> 22548578304
exit_code=0
=== SYSCTL AFTER WRITE ATTEMPTS ===
iogpu.wired_limit_mb: 21504
vm.global_user_wire_limit: 22548578304
vm.user_wire_limit: 22548578304
=== AMINFER PROBE UNDER RAISED SYSCTL ===
sys.page_size                    16384
sys.memory_mib                   24576
cpu.performance_cores            4
cpu.efficiency_cores             6
wire.user_limit_bytes            22548578304
wire.global_limit_bytes          22548578304
wire.gpu_limit_mib               21504
gpu.name                         Apple M4
gpu.apple_family                 9
gpu.metal4                       true
gpu.unified_memory               true
gpu.recommended_working_set_mib  21504
gpu.max_buffer_mib               13639
ane.cores                        16
budget.wired_limit_mib           21504
budget.binding_limit             physical_headroom
probe_su_exit_code=0
=== RESTORE SYSCTL ===
--- sysctl -w vm.user_wire_limit=19585050869 ---
vm.user_wire_limit: 22548578304 -> 19585050869
exit_code=0
--- sysctl -w vm.global_user_wire_limit=19585050869 ---
vm.global_user_wire_limit: 22548578304 -> 19585050869
exit_code=0
--- sysctl -w iogpu.wired_limit_mb=0 ---
iogpu.wired_limit_mb: 21504 -> 0
exit_code=0
=== SYSCTL AFTER RESTORE ===
iogpu.wired_limit_mb: 0
vm.global_user_wire_limit: 19585050869
vm.user_wire_limit: 19585050869
=== AMINFER PROBE AFTER RESTORE ===
sys.page_size                    16384
sys.memory_mib                   24576
cpu.performance_cores            4
cpu.efficiency_cores             6
wire.user_limit_bytes            19585050869
wire.global_limit_bytes          19585050869
wire.gpu_limit_mib               0
gpu.name                         Apple M4
gpu.apple_family                 9
gpu.metal4                       true
gpu.unified_memory               true
gpu.recommended_working_set_mib  18186
gpu.max_buffer_mib               13639
ane.cores                        16
budget.wired_limit_mib           18186
budget.binding_limit             MTLDevice.recommendedMaxWorkingSetSize
```

## Power and frequency telemetry (`powermetrics`)

Captured on 2026-11-20 after restoring default `sysctl` values, running
`./zig-out/bin/aminfer bench` (non-root) in the background while executing
`/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 6`
(`powermetrics_exit_code=0`, `bench_exit_code=0`).

### Summary across the six 500 ms samples

| Sample | Timestamp (`+0800`) | Elapsed (ms) | E-Cluster active (freq / res) | P-Cluster active (freq / res) | GPU active (freq / res / 1470 MHz) | CPU Power | GPU Power (Proc / GPU block) | ANE Power | Combined Power | Observed phase |
| ---: | --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| 1 | `20:16:17` | 510.32 | 2867 MHz / 100.00% | 4257 MHz / 70.97% | 1459 MHz / 70.98% / 70% | 4244 mW | 1513 / 1513 mW | 0 mW | 5757 mW | Active `aminfer bench` GPU burst |
| 2 | `20:16:18` | 509.83 | 2221 MHz / 86.87% | 4372 MHz / 52.12% | 807 MHz / 26.43% / 2.8% | 2501 mW | 357 / 357 mW | 0 mW | 2858 mW | `aminfer bench` tail / ramp-down |
| 3 | `20:16:18` | 516.84 | 1356 MHz / 58.34% | 4369 MHz / 49.85% | 563 MHz / 29.76% / 0% | 1983 mW | 0 / 408 mW | 0 mW | 1983 mW | Post-bench desktop baseline |
| 4 | `20:16:19` | 516.26 | 2594 MHz / 92.87% | 4367 MHz / 49.48% | 550 MHz / 29.52% / 0% | 2443 mW | 794 / 384 mW | 0 mW | 3237 mW | Post-bench desktop baseline |
| 5 | `20:16:19` | 510.78 | 2857 MHz / 100.00% | 4358 MHz / 53.86% | 496 MHz / 28.50% / 0% | 2937 mW | 480 / 499 mW | 0 mW | 3416 mW | Post-bench desktop baseline |
| 6 | `20:16:20` | 509.66 | 2863 MHz / 100.00% | 4310 MHz / 65.26% | 524 MHz / 35.10% / 0.35% | 3926 mW | 420 / 400 mW | 0 mW | 4346 mW | Post-bench desktop baseline |

Telemetry takeaways:
- During `aminfer bench` (Sample 1), M4 GPU clocks pinned at the **1470 MHz**
  top P-state (`1459 MHz` weighted active average, `70.98%` HW active
  residency) at **1,513 mW** GPU power (~1.5 W), while ANE remained completely
  idle (**0 mW** across all 6 samples, as expected for GPU-only M1 kernels).
- Concurrent 500 ms `powermetrics` polling and active background desktop/agent
  load (`E-Cluster` 58–100% active, `P-Cluster` 49–71% active across the whole
  window) noticeably perturbed the concurrent `aminfer bench` run below
  (peak `gemv_q4` `86.49 GB/s` vs `98.74 GB/s` standalone; Spike S2 CPU encode
  `21.70 us/round` with `1` malloc block / `96 B` in-use delta and
  `0 B` `size_allocated` delta vs `7.78 us/round` and `0` delta standalone).
  Swap remained strictly zero (`swapouts_delta = 0`, `pageouts_delta = 0`).
- In Samples 3–6, `powermetrics` reports slightly different instantaneous
  `GPU Power` readings in its `Processor usage` summary block versus its
  `GPU usage` block (an artifact of Apple's per-sampler window alignment);
  both values are tabulated above and preserved verbatim below.

### Verbatim concurrent `aminfer bench` raw stdout (during `powermetrics`)

```text
bench.gpu                        Apple M4
bench.wired_budget_mib           18186
bench.arena_mib                  81
bench.bus_peak_gbs               120.0

C9 Kernel Bandwidth Calibration
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q4   1x2048x2048           2367488      32         27.95        84.70     70.6
gemv_q4   1x6144x2048           7094272      10         82.02        86.49     72.1
gemv_q4   1x2048x768             890368      85         11.94        74.57     62.1
gemv_q4   1x4096x4096           9453568       8        110.12        85.85     71.5
gemv_q3   1x2048x2048           1843200      42         26.05        70.75     59.0
gemv_q3   1x6144x2048           5521408      14         73.65        74.97     62.5
gemv_q3   1x2048x768             693760     113         11.20        61.93     51.6
gemv_q3   1x4096x4096           7356416      10         89.86        81.87     68.2
gemm_q4   16x2048x2048          2490368      32        116.82        21.32     17.8
gemm_q4   16x4096x4096          9699328       8        437.62        22.16     18.5
gemm_q4   64x2048x2048          2883584      32        218.79        13.18     11.0
gemm_q4   64x4096x4096         10485760       8        894.34        11.72      9.8
gemm_q3   16x2048x2048          1966080      42        120.17        16.36     13.6
gemm_q3   16x4096x4096          7602176      10        436.11        17.43     14.5
gemm_q3   64x2048x2048          2359296      42        225.19        10.48      8.7
gemm_q3   64x4096x4096          8388608      10        904.87         9.27      7.7

Spike S2 Metal 4 Steady-State Hot Path
s2.rounds                        1000
s2.dispatches_per_round          32
s2.cpu_encode_us_per_round       21.70
s2.cpu_encode_ns_per_dispatch    678.2
s2.full_round_us                 114.00
s2.malloc_blocks_in_use_delta    1
s2.malloc_size_in_use_delta      96
s2.malloc_size_allocated_delta   0

Swap Check
swap.status                      ok
swap.swapouts_delta              0
swap.pageouts_delta              0
swap.used_mib                    3014
```

### Verbatim `powermetrics` raw output

```text
Machine model: Mac16,12
OS version: 26B123
Boot arguments: 
Boot time: Mon Nov  2 16:13:44 2026



*** Sampled system activity (Fri Nov 20 20:16:17 2026 +0800) (510.32ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 2867 MHz
E-Cluster HW active residency: 100.00% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:  30% 2892 MHz:  70%)
E-Cluster idle residency:   0.00%
CPU 0 frequency: 2892 MHz
CPU 0 active residency:  38.53% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  39%)
CPU 0 idle residency:  61.47%
CPU 1 frequency: 2892 MHz
CPU 1 active residency:  44.88% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  45%)
CPU 1 idle residency:  55.12%
CPU 2 frequency: 2892 MHz
CPU 2 active residency:  33.42% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  33%)
CPU 2 idle residency:  66.58%
CPU 3 frequency: 2892 MHz
CPU 3 active residency:  35.11% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  35%)
CPU 3 idle residency:  64.89%
CPU 4 frequency: 2892 MHz
CPU 4 active residency:  24.26% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  24%)
CPU 4 idle residency:  75.74%
CPU 5 frequency: 2892 MHz
CPU 5 active residency:  16.92% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  17%)
CPU 5 idle residency:  83.08%

P-Cluster HW active frequency: 4257 MHz
P-Cluster HW active residency:  70.97% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz: .00% 3936 MHz:  24% 3984 MHz: 5.8% 4044 MHz: 5.1% 4104 MHz:   0% 4416 MHz:  62% 4464 MHz: 2.5%)
P-Cluster idle residency:  29.03%
CPU 6 frequency: 4428 MHz
CPU 6 active residency:  37.47% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  28% 4464 MHz: 9.4%)
CPU 6 idle residency:  62.53%
CPU 7 frequency: 4426 MHz
CPU 7 active residency:  33.88% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  27% 4464 MHz: 6.7%)
CPU 7 idle residency:  66.12%
CPU 8 frequency: 4429 MHz
CPU 8 active residency:  29.81% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  22% 4464 MHz: 7.9%)
CPU 8 idle residency:  70.19%
CPU 9 frequency: 4431 MHz
CPU 9 active residency:  29.15% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  20% 4464 MHz: 8.9%)
CPU 9 idle residency:  70.85%

CPU Power: 4244 mW
GPU Power: 1513 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 5757 mW

**** GPU usage ****

GPU HW active frequency: 1459 MHz
GPU HW active residency:  70.98% (338 MHz: .66% 618 MHz:   0% 796 MHz:   0% 836 MHz:   0% 928 MHz:   0% 952 MHz:   0% 1056 MHz:   0% 1053 MHz:   0% 1170 MHz:   0% 1152 MHz:   0% 1278 MHz:   0% 1204 MHz:   0% 1338 MHz: .27% 1326 MHz:   0% 1470 MHz:  70%)
GPU SW requested state: (P1 :   0% P2 :   0% P3 :   0% P4 :   0% P5 :   0% P6 :   0% P7 :   0% P8 :  99% P9 : .62% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 : .66% SW_P2 :   0% SW_P3 :   0% SW_P4 :   0% SW_P5 :   0% SW_P6 :   0% SW_P7 :   0% SW_P8 : .27% SW_P9 :  70% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  29.02%
GPU Power: 1513 mW



*** Sampled system activity (Fri Nov 20 20:16:18 2026 +0800) (509.83ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 2221 MHz
E-Cluster HW active residency:  86.87% (900 MHz: .78% 1080 MHz:  32% 1476 MHz: 1.1% 1860 MHz: 3.7% 2172 MHz: .86% 2496 MHz: .64% 2808 MHz:  24% 2892 MHz:  37%)
E-Cluster idle residency:  13.13%
CPU 0 frequency: 2173 MHz
CPU 0 active residency:  48.29% (900 MHz: .19% 1080 MHz:  17% 1476 MHz: .87% 1860 MHz: 2.1% 2172 MHz: .49% 2496 MHz: .44% 2808 MHz: .56% 2892 MHz:  27%)
CPU 0 idle residency:  51.71%
CPU 1 frequency: 2301 MHz
CPU 1 active residency:  48.16% (900 MHz: .06% 1080 MHz:  14% 1476 MHz: .34% 1860 MHz: 1.7% 2172 MHz: .52% 2496 MHz: .51% 2808 MHz: .64% 2892 MHz:  30%)
CPU 1 idle residency:  51.84%
CPU 2 frequency: 2434 MHz
CPU 2 active residency:  41.37% (900 MHz: .11% 1080 MHz: 9.3% 1476 MHz: .44% 1860 MHz: .87% 2172 MHz: .14% 2496 MHz: .35% 2808 MHz: .57% 2892 MHz:  30%)
CPU 2 idle residency:  58.63%
CPU 3 frequency: 2248 MHz
CPU 3 active residency:  26.51% (900 MHz: .08% 1080 MHz: 8.5% 1476 MHz: .22% 1860 MHz: .99% 2172 MHz: .05% 2496 MHz: .36% 2808 MHz: .19% 2892 MHz:  16%)
CPU 3 idle residency:  73.49%
CPU 4 frequency: 2261 MHz
CPU 4 active residency:  21.34% (900 MHz: .03% 1080 MHz: 7.0% 1476 MHz: .03% 1860 MHz: .58% 2172 MHz: .01% 2496 MHz: .14% 2808 MHz: .27% 2892 MHz:  13%)
CPU 4 idle residency:  78.66%
CPU 5 frequency: 2528 MHz
CPU 5 active residency:  19.76% (900 MHz: .00% 1080 MHz: 3.4% 1476 MHz: .17% 1860 MHz: .58% 2172 MHz: .02% 2496 MHz: .29% 2808 MHz: .18% 2892 MHz:  15%)
CPU 5 idle residency:  80.24%

P-Cluster HW active frequency: 4372 MHz
P-Cluster HW active residency:  52.12% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz: 9.3% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  91% 4464 MHz:   0%)
P-Cluster idle residency:  47.88%
CPU 6 frequency: 4416 MHz
CPU 6 active residency:  27.49% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  27% 4464 MHz:   0%)
CPU 6 idle residency:  72.51%
CPU 7 frequency: 4416 MHz
CPU 7 active residency:  27.35% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  27% 4464 MHz:   0%)
CPU 7 idle residency:  72.65%
CPU 8 frequency: 4416 MHz
CPU 8 active residency:  10.42% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  10% 4464 MHz:   0%)
CPU 8 idle residency:  89.58%
CPU 9 frequency: 4416 MHz
CPU 9 active residency:   8.04% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 8.0% 4464 MHz:   0%)
CPU 9 idle residency:  91.96%

CPU Power: 2501 mW
GPU Power: 357 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 2858 mW

**** GPU usage ****

GPU HW active frequency: 807 MHz
GPU HW active residency:  26.43% (338 MHz: 7.4% 618 MHz: 4.3% 796 MHz: 3.6% 836 MHz:   0% 928 MHz: 3.2% 952 MHz:   0% 1056 MHz: 1.4% 1053 MHz:   0% 1170 MHz: 1.2% 1152 MHz:   0% 1278 MHz: 1.2% 1204 MHz:   0% 1338 MHz: 1.4% 1326 MHz:   0% 1470 MHz: 2.8%)
GPU SW requested state: (P1 :  36% P2 : 7.3% P3 :  17% P4 : 9.4% P5 : 4.8% P6 : 4.9% P7 : 5.3% P8 :  15% P9 : 1.1% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 : 7.3% SW_P2 : 4.3% SW_P3 : 3.6% SW_P4 : 3.2% SW_P5 : 1.4% SW_P6 : 1.2% SW_P7 : 1.2% SW_P8 : 1.4% SW_P9 : 2.8% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  73.57%
GPU Power: 357 mW



*** Sampled system activity (Fri Nov 20 20:16:18 2026 +0800) (516.84ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 1356 MHz
E-Cluster HW active residency:  58.34% (900 MHz: 3.0% 1080 MHz:  73% 1476 MHz: 4.6% 1860 MHz: 8.2% 2172 MHz: .93% 2496 MHz:   0% 2808 MHz: 5.6% 2892 MHz: 5.1%)
E-Cluster idle residency:  41.66%
CPU 0 frequency: 1377 MHz
CPU 0 active residency:  47.35% (900 MHz: .95% 1080 MHz:  34% 1476 MHz: 2.7% 1860 MHz: 4.5% 2172 MHz: .43% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 5.0%)
CPU 0 idle residency:  52.65%
CPU 1 frequency: 1377 MHz
CPU 1 active residency:  40.22% (900 MHz: .32% 1080 MHz:  29% 1476 MHz: 2.2% 1860 MHz: 3.5% 2172 MHz: .32% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 4.4%)
CPU 1 idle residency:  59.78%
CPU 2 frequency: 1390 MHz
CPU 2 active residency:  29.77% (900 MHz: .18% 1080 MHz:  21% 1476 MHz: 1.7% 1860 MHz: 3.1% 2172 MHz: .34% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 3.2%)
CPU 2 idle residency:  70.23%
CPU 3 frequency: 1432 MHz
CPU 3 active residency:  22.12% (900 MHz: .21% 1080 MHz:  15% 1476 MHz: 1.2% 1860 MHz: 2.3% 2172 MHz: .07% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 3.0%)
CPU 3 idle residency:  77.88%
CPU 4 frequency: 1482 MHz
CPU 4 active residency:  18.66% (900 MHz: .02% 1080 MHz:  13% 1476 MHz: 1.1% 1860 MHz: 2.0% 2172 MHz: .08% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 3.0%)
CPU 4 idle residency:  81.34%
CPU 5 frequency: 1518 MHz
CPU 5 active residency:  14.09% (900 MHz: .05% 1080 MHz: 8.9% 1476 MHz: .81% 1860 MHz: 1.9% 2172 MHz: .03% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz: 2.4%)
CPU 5 idle residency:  85.91%

P-Cluster HW active frequency: 4369 MHz
P-Cluster HW active residency:  49.85% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz: .01% 3936 MHz: 9.7% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  90% 4464 MHz:   0%)
P-Cluster idle residency:  50.15%
CPU 6 frequency: 4416 MHz
CPU 6 active residency:  30.52% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  31% 4464 MHz:   0%)
CPU 6 idle residency:  69.48%
CPU 7 frequency: 4416 MHz
CPU 7 active residency:  28.61% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  29% 4464 MHz:   0%)
CPU 7 idle residency:  71.39%
CPU 8 frequency: 4416 MHz
CPU 8 active residency:   9.17% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 9.2% 4464 MHz:   0%)
CPU 8 idle residency:  90.83%
CPU 9 frequency: 4416 MHz
CPU 9 active residency:   4.93% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 4.9% 4464 MHz:   0%)
CPU 9 idle residency:  95.07%

CPU Power: 1983 mW
GPU Power: 0 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 1983 mW

**** GPU usage ****

GPU HW active frequency: 563 MHz
GPU HW active residency:  29.76% (338 MHz:  14% 618 MHz: 6.5% 796 MHz: 3.1% 836 MHz:   0% 928 MHz: 5.9% 952 MHz:   0% 1056 MHz:   0% 1053 MHz:   0% 1170 MHz:   0% 1152 MHz:   0% 1278 MHz:   0% 1204 MHz:   0% 1338 MHz:   0% 1326 MHz:   0% 1470 MHz:   0%)
GPU SW requested state: (P1 :  65% P2 : 3.9% P3 :  27% P4 : 4.9% P5 : .09% P6 :   0% P7 :   0% P8 :   0% P9 :   0% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 :  14% SW_P2 : 6.5% SW_P3 : 3.1% SW_P4 : 5.9% SW_P5 :   0% SW_P6 :   0% SW_P7 :   0% SW_P8 :   0% SW_P9 :   0% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  70.24%
GPU Power: 408 mW



*** Sampled system activity (Fri Nov 20 20:16:19 2026 +0800) (516.26ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 2594 MHz
E-Cluster HW active residency:  92.87% (900 MHz: .55% 1080 MHz:  12% 1476 MHz: 1.4% 1860 MHz: 1.7% 2172 MHz: .81% 2496 MHz: .55% 2808 MHz:  23% 2892 MHz:  60%)
E-Cluster idle residency:   7.13%
CPU 0 frequency: 2570 MHz
CPU 0 active residency:  41.17% (900 MHz: .13% 1080 MHz: 6.3% 1476 MHz: .55% 1860 MHz: .56% 2172 MHz: .06% 2496 MHz: .13% 2808 MHz: .77% 2892 MHz:  33%)
CPU 0 idle residency:  58.83%
CPU 1 frequency: 2500 MHz
CPU 1 active residency:  47.75% (900 MHz: .10% 1080 MHz: 8.0% 1476 MHz: 1.3% 1860 MHz: 1.4% 2172 MHz: .76% 2496 MHz: .45% 2808 MHz: .02% 2892 MHz:  36%)
CPU 1 idle residency:  52.25%
CPU 2 frequency: 2608 MHz
CPU 2 active residency:  35.58% (900 MHz: .02% 1080 MHz: 4.8% 1476 MHz: .53% 1860 MHz: .50% 2172 MHz: .01% 2496 MHz: .03% 2808 MHz: .02% 2892 MHz:  30%)
CPU 2 idle residency:  64.42%
CPU 3 frequency: 2577 MHz
CPU 3 active residency:  21.12% (900 MHz:   0% 1080 MHz: 3.2% 1476 MHz: .45% 1860 MHz: .22% 2172 MHz: .00% 2496 MHz: .00% 2808 MHz: .01% 2892 MHz:  17%)
CPU 3 idle residency:  78.88%
CPU 4 frequency: 2584 MHz
CPU 4 active residency:  17.25% (900 MHz: .01% 1080 MHz: 2.4% 1476 MHz: .46% 1860 MHz: .15% 2172 MHz: .03% 2496 MHz: .09% 2808 MHz: .05% 2892 MHz:  14%)
CPU 4 idle residency:  82.75%
CPU 5 frequency: 2663 MHz
CPU 5 active residency:  23.64% (900 MHz: .09% 1080 MHz: 2.4% 1476 MHz: .52% 1860 MHz: .22% 2172 MHz:   0% 2496 MHz: .00% 2808 MHz:   0% 2892 MHz:  20%)
CPU 5 idle residency:  76.36%

P-Cluster HW active frequency: 4367 MHz
P-Cluster HW active residency:  49.48% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz: .00% 3936 MHz:  10% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  90% 4464 MHz:   0%)
P-Cluster idle residency:  50.52%
CPU 6 frequency: 4416 MHz
CPU 6 active residency:  29.97% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  30% 4464 MHz:   0%)
CPU 6 idle residency:  70.03%
CPU 7 frequency: 4416 MHz
CPU 7 active residency:  28.56% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  29% 4464 MHz:   0%)
CPU 7 idle residency:  71.44%
CPU 8 frequency: 4416 MHz
CPU 8 active residency:   8.87% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 8.9% 4464 MHz:   0%)
CPU 8 idle residency:  91.13%
CPU 9 frequency: 4416 MHz
CPU 9 active residency:   4.85% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 4.8% 4464 MHz:   0%)
CPU 9 idle residency:  95.15%

CPU Power: 2443 mW
GPU Power: 794 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 3237 mW

**** GPU usage ****

GPU HW active frequency: 550 MHz
GPU HW active residency:  29.52% (338 MHz:  15% 618 MHz: 6.2% 796 MHz: 2.8% 836 MHz:   0% 928 MHz: 4.0% 952 MHz:   0% 1056 MHz: .34% 1053 MHz:   0% 1170 MHz: .47% 1152 MHz:   0% 1278 MHz: .27% 1204 MHz:   0% 1338 MHz:   0% 1326 MHz:   0% 1470 MHz:   0%)
GPU SW requested state: (P1 :  69% P2 : 3.6% P3 :  21% P4 : 2.8% P5 : 1.9% P6 : 1.9% P7 : .11% P8 :   0% P9 :   0% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 :  16% SW_P2 : 6.2% SW_P3 : 2.8% SW_P4 : 4.0% SW_P5 : .34% SW_P6 : .47% SW_P7 : .27% SW_P8 :   0% SW_P9 :   0% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  70.48%
GPU Power: 384 mW



*** Sampled system activity (Fri Nov 20 20:16:19 2026 +0800) (510.78ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 2857 MHz
E-Cluster HW active residency: 100.00% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:  37% 2892 MHz:  63%)
E-Cluster idle residency:   0.00%
CPU 0 frequency: 2883 MHz
CPU 0 active residency:  52.88% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  53%)
CPU 0 idle residency:  47.12%
CPU 1 frequency: 2882 MHz
CPU 1 active residency:  44.42% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  44%)
CPU 1 idle residency:  55.58%
CPU 2 frequency: 2882 MHz
CPU 2 active residency:  46.09% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  46%)
CPU 2 idle residency:  53.91%
CPU 3 frequency: 2880 MHz
CPU 3 active residency:  37.11% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  37%)
CPU 3 idle residency:  62.89%
CPU 4 frequency: 2877 MHz
CPU 4 active residency:  30.62% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  30%)
CPU 4 idle residency:  69.38%
CPU 5 frequency: 2873 MHz
CPU 5 active residency:  24.42% (900 MHz:   0% 1080 MHz: .25% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  24%)
CPU 5 idle residency:  75.58%

P-Cluster HW active frequency: 4358 MHz
P-Cluster HW active residency:  53.86% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:  12% 3984 MHz: .07% 4044 MHz: .01% 4104 MHz:   0% 4416 MHz:  88% 4464 MHz:   0%)
P-Cluster idle residency:  46.14%
CPU 6 frequency: 4416 MHz
CPU 6 active residency:  31.95% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  32% 4464 MHz:   0%)
CPU 6 idle residency:  68.05%
CPU 7 frequency: 4416 MHz
CPU 7 active residency:  30.06% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  30% 4464 MHz:   0%)
CPU 7 idle residency:  69.94%
CPU 8 frequency: 4416 MHz
CPU 8 active residency:  11.93% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  12% 4464 MHz:   0%)
CPU 8 idle residency:  88.07%
CPU 9 frequency: 4416 MHz
CPU 9 active residency:   8.21% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz: 8.2% 4464 MHz:   0%)
CPU 9 idle residency:  91.79%

CPU Power: 2937 mW
GPU Power: 480 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 3416 mW

**** GPU usage ****

GPU HW active frequency: 496 MHz
GPU HW active residency:  28.50% (338 MHz:  16% 618 MHz: 5.9% 796 MHz: 5.7% 836 MHz:   0% 928 MHz: .39% 952 MHz:   0% 1056 MHz:   0% 1053 MHz:   0% 1170 MHz:   0% 1152 MHz:   0% 1278 MHz:   0% 1204 MHz:   0% 1338 MHz:   0% 1326 MHz:   0% 1470 MHz:   0%)
GPU SW requested state: (P1 :  70% P2 :  11% P3 :  18% P4 : .61% P5 :   0% P6 :   0% P7 :   0% P8 :   0% P9 :   0% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 :  16% SW_P2 : 5.9% SW_P3 : 5.7% SW_P4 : .39% SW_P5 :   0% SW_P6 :   0% SW_P7 :   0% SW_P8 :   0% SW_P9 :   0% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  71.50%
GPU Power: 499 mW



*** Sampled system activity (Fri Nov 20 20:16:20 2026 +0800) (509.66ms elapsed) ***


**** Processor usage ****

E-Cluster Online: 100%
E-Cluster HW active frequency: 2863 MHz
E-Cluster HW active residency: 100.00% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:  34% 2892 MHz:  66%)
E-Cluster idle residency:   0.00%
CPU 0 frequency: 2892 MHz
CPU 0 active residency:  54.62% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  55%)
CPU 0 idle residency:  45.38%
CPU 1 frequency: 2892 MHz
CPU 1 active residency:  49.90% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  50%)
CPU 1 idle residency:  50.10%
CPU 2 frequency: 2892 MHz
CPU 2 active residency:  42.93% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  43%)
CPU 2 idle residency:  57.07%
CPU 3 frequency: 2892 MHz
CPU 3 active residency:  30.50% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  31%)
CPU 3 idle residency:  69.50%
CPU 4 frequency: 2892 MHz
CPU 4 active residency:  27.31% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  27%)
CPU 4 idle residency:  72.69%
CPU 5 frequency: 2892 MHz
CPU 5 active residency:  18.66% (900 MHz:   0% 1080 MHz:   0% 1476 MHz:   0% 1860 MHz:   0% 2172 MHz:   0% 2496 MHz:   0% 2808 MHz:   0% 2892 MHz:  19%)
CPU 5 idle residency:  81.34%

P-Cluster HW active frequency: 4310 MHz
P-Cluster HW active residency:  65.26% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz: .01% 3936 MHz:  15% 3984 MHz: 4.9% 4044 MHz: 3.3% 4104 MHz:   0% 4416 MHz:  74% 4464 MHz: 2.3%)
P-Cluster idle residency:  34.74%
CPU 6 frequency: 4425 MHz
CPU 6 active residency:  32.95% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  27% 4464 MHz: 6.0%)
CPU 6 idle residency:  67.05%
CPU 7 frequency: 4426 MHz
CPU 7 active residency:  29.71% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  24% 4464 MHz: 5.9%)
CPU 7 idle residency:  70.29%
CPU 8 frequency: 4426 MHz
CPU 8 active residency:  25.68% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  20% 4464 MHz: 5.4%)
CPU 8 idle residency:  74.32%
CPU 9 frequency: 4435 MHz
CPU 9 active residency:  19.07% (912 MHz:   0% 1260 MHz:   0% 1512 MHz:   0% 1800 MHz:   0% 2088 MHz:   0% 2352 MHz:   0% 2616 MHz:   0% 2868 MHz:   0% 3096 MHz:   0% 3300 MHz:   0% 3468 MHz:   0% 3624 MHz:   0% 3756 MHz:   0% 3876 MHz:   0% 3936 MHz:   0% 3984 MHz:   0% 4044 MHz:   0% 4104 MHz:   0% 4416 MHz:  12% 4464 MHz: 7.4%)
CPU 9 idle residency:  80.93%

CPU Power: 3926 mW
GPU Power: 420 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 4346 mW

**** GPU usage ****

GPU HW active frequency: 524 MHz
GPU HW active residency:  35.10% (338 MHz:  19% 618 MHz: 6.3% 796 MHz: 8.0% 836 MHz:   0% 928 MHz: .83% 952 MHz:   0% 1056 MHz:   0% 1053 MHz:   0% 1170 MHz:   0% 1152 MHz:   0% 1278 MHz:   0% 1204 MHz:   0% 1338 MHz: .22% 1326 MHz:   0% 1470 MHz: .35%)
GPU SW requested state: (P1 :  66% P2 :  13% P3 :  19% P4 : .31% P5 :   0% P6 :   0% P7 :   0% P8 : 1.7% P9 :   0% P10 :   0% P11 :   0% P12 :   0% P13 :   0% P14 :   0% P15 :   0%)
GPU SW state: (SW_P1 :  19% SW_P2 : 6.3% SW_P3 : 8.0% SW_P4 : .83% SW_P5 :   0% SW_P6 :   0% SW_P7 :   0% SW_P8 : .22% SW_P9 : .35% SW_P10 :   0% SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)
GPU idle residency:  64.90%
GPU Power: 400 mW

```

## M2 Spike S1, G3 MoE, G4 GQA and end-to-end `.amf` decode (standalone ReleaseFast)

Measured on 2026-11-20 on the Apple M4 24 GiB baseline (`Mac16,12`, macOS 27.2
`26B123`, default kernel tunables `wired_budget_mib = 18186`) at git commit
`517ba0da9094849813575a9520da28daaea78721` via standalone non-root
`./zig-out/bin/aminfer bench` (ReleaseFast) and end-to-end
`aminfer-tools synth-qwen3` → `./zig-out/bin/aminfer decode`.

### M2 methodology notes

1. **Spike S1 (IOSurface KV-tile read bandwidth)**: Allocates 2 rotated context
   banks × 32 chunks × 2 tiles (K, V) = 128 IOSurface-backed `CVPixelBuffer`s
   (`OneComponent16Half`, explicit 16 KiB-aligned `kIOSurfaceAllocSize = 524,288 B`,
   `bytesPerRow = 256 B`, 64 MiB total K+V across both banks to bust M4 L2/SLC)
   representing a 16,384-token Qwen3-30B-A3B KV context (`q=32, kv=4, d=128,
   chunk=512`), registered in `MTLResidencySet`. Measures 8 × 8 = 64 dispatches
   per mode across `{token_major, channel_major}` × `{RGBA16Float, R16Float,
   alias MTLBuffer}` (`33,554,432 B/dispatch`) and probes `mlock` support on
   IOSurface-mapped pages.
2. **G3 MoE deduplicated gather / SwiGLU**: Allocates all 128 distinct 3-bit
   Qwen3-30B-A3B experts (`hidden=2048, moe_intermediate=768`, 252 MiB total
   resident weights in one `Budget`-checked `WiredArena`, naturally exceeding
   SLC) and rotates router-logit banks across all 128 experts for `M=1`
   (`U=8/8` slots) and `M=8` (`U=24/64` and `U=40/64` slots). Reports exact
   GPU-verified `ScheduleHeader.num_unique_experts` (`U`), deduplicated weight
   bytes (`expertLayerBytes(q3, U, 2048, 768)`), naive per-slot weight bytes
   (`expertLayerBytes(q3, M*k, 2048, 768)`), percentage bytes saved, and
   effective decimal GB/s.
3. **G4 GQA KV-tile decode attention**: Benchmarks production
   `AttentionPipelines.encodeGqaDecodeAttn` (FlashDecoding split-KV over
   bindless `RGBA16Float` token-major `KvTilePair` arrays) on Qwen3-30B-A3B
   geometry (`q=32, kv=4, d=128, chunk=512`) at `seq_len = 512, 4096, 16384`
   using rotated tile banks totaling `>= 64 MiB`.
4. **M2 end-to-end greedy decode from `.amf`**: Streams a valid 4-layer `.amf`
   v0 file (with `F_NOCACHE` write + `F_NOCACHE` load + full CRC-32 check,
   full `vocab_size = 151,936`, real per-layer Qwen3-30B-A3B-3bit and
   Qwen3-14B-4bit dimensions) into a `Budget.fromHost`-checked `WiredArena` +
   IOSurface `KvTile`s + `MTLResidencySet`. Genuinely prefills 511 KV positions
   via `prefillToken`, warms up 4 steps, then times 16 steady-state `stepDecode`
   calls (`T_full`) and 16 `prefillToken` body-only calls (`T_body`) at exact
   `seq_len = 512`.
   - **Direct measurement** is the measured 4-layer `.amf` decode step (`T_4L`,
     `T_layer = T_body / 4`, `T_head = T_4L - T_body`, `bytes_4L`).
   - **Linear full-depth scaling** (`N = 48` layers for Qwen3-30B-A3B-3bit;
     `N = 40` layers for Qwen3-14B-4bit) is computed transparently as
     `T_NL = N * T_layer + T_head` and `bytes_NL = N * bytes_per_layer + bytes_global_head`
     so the global `lm_head` + `argmax` step is counted exactly once rather than
     multiplied by `N/4`.

### Spike S1 IOSurface KV-tile read bandwidth summary

Geometry: `q=32, kv=4, d=128, chunk=512, 32 chunks (ctx=16,384), 2 rotated banks (64 MiB K+V)`.
IOSurface facts: `bytes_per_row = 256 B`, `data_bytes = 524,288 B`,
`alloc_bytes = 524,288 B`, `IOSurfaceGetAllocSize = 524,288 B` (exact 16 KiB
multiple via explicit `kIOSurfaceAllocSize`), `iosurface_mlock_supported = yes`.

| Layout | Access mode | Bytes/dispatch | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| --- | --- | ---: | ---: | ---: | ---: |
| `token_major` | `rgba16f` (`tex.read(half4)`) | 33,554,432 | 418.22 | 80.23 | 66.9% |
| `token_major` | `r16f` (`tex.read(half)`) | 33,554,432 | 365.73 | 91.75 | 76.5% |
| `token_major` | `alias_buf` (`MTLBuffer`) | 33,554,432 | 408.48 | 82.14 | 68.5% |
| `channel_major` | `rgba16f` (`tex.read(half4)`) | 33,554,432 | 421.37 | 79.63 | 66.4% |
| `channel_major` | `r16f` (`tex.read(half)`) | 33,554,432 | 416.37 | 80.59 | 67.2% |
| `channel_major` | `alias_buf` (`MTLBuffer`) | 33,554,432 | 426.41 | 78.69 | 65.6% |

Decision (`R4` settled): **token-major `RGBA16Float`** (`width = head_dim / 4`,
`height = num_kv_heads * 512`, one 512-token chunk per layer). Texture views
match or outperform aliasing `MTLBuffer` reads with zero copy penalty, while
token-major `RGBA16Float` enables single-instruction `tex.write(half4)` decode
KV append without read-modify-write, clean partial-chunk `seq_len` bounds, and
**85.61 GB/s (71.3% bus)** in production FlashDecoding GQA below.

### G3 MoE deduplicated gather / SwiGLU summary

Geometry: Qwen3-30B-A3B 3-bit (`E=128, k=8, hidden=2048, moe_intermediate=768`,
all 128 experts resident = 252 MiB).

| `M` | Unique experts `U / slots` | Dedup bytes | Naive bytes | Weight bytes saved | Latency (us) | Dedup `beta` (GB/s) | % of 120 GB/s bus | Naive-equiv (GB/s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `8/8` | 16,515,072 | 16,515,072 | 0.0% | 228.34 | 72.33 | 60.3% | 72.33 |
| 8 | `24/64` | 49,545,216 | 132,120,576 | 62.5% | 2,096.27 | 23.63 | 19.7% | 63.03 |
| 8 | `40/64` | 82,575,360 | 132,120,576 | 37.5% | 2,124.23 | 38.87 | 32.4% | 62.20 |

Note: `M=1` uses the fast SIMD32 GEMV path (`72.33 GB/s`), whereas `M=8`
deduplicates 37.5%–62.5% of expert weight bytes while currently running a
scalar-accumulating multi-token loop (compute-bound at `2.1 ms`); upgrading the
`count_u > 1` path to `simdgroup_matrix` is a prime M3 (`G5` / `S5`) speedup
opportunity.

### G4 GQA KV-tile decode attention summary

Geometry: Qwen3-30B-A3B (`q=32, kv=4, d=128, chunk=512`, `RGBA16Float` token-major
IOSurface tiles, rotated banks `>= 64 MiB`).

| `seq_len` | Chunks | Splits | Banks | Bytes/dispatch | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 8 | 64 | 1,048,576 | 24.55 | 42.72 | 35.6% |
| 4,096 | 8 | 32 | 8 | 8,388,608 | 106.98 | 78.42 | 65.3% |
| 16,384 | 32 | 32 | 2 | 33,554,432 | 391.96 | 85.61 | 71.3% |

### M2 end-to-end greedy decode from `.amf` summary (`seq_len = 512`, `vocab = 151,936`)

| Target model | Measurement basis | Layers | Wired (MiB) | `ms/tok` | `tok/s` | Bytes/tok | Effective `beta` (GB/s) | % bus | Hot-path `malloc` delta | `swapouts` delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **Qwen3-30B-A3B (3-bit MoE)** | Measured 4-layer `.amf` (`0.477 ms/L + 1.598 ms head`) | 4 | 1,310 | 3.508 | 285.05 | 241,564,544 | 68.86 | 57.4% | 0 blocks / 0 B | 0 |
| **Qwen3-30B-A3B (3-bit MoE)** | Extrapolated linear scaling (`48 * layer_ms + head_ms`) | 48 | — | 24.515 | **40.79** | **1,401,238,400** | 57.16 | 47.6% | — | — |
| **Qwen3-14B (4-bit Dense)** | Measured 4-layer `.amf` (`2.436 ms/L + 5.182 ms head`) | 4 | 1,561 | 14.926 | 67.00 | 1,189,256,000 | 79.68 | 66.4% | 0 blocks / 0 B | 0 |
| **Qwen3-14B (4-bit Dense)** | Extrapolated linear scaling (`40 * layer_ms + head_ms`) | 40 | — | 102.622 | **9.74** | **7,954,260,800** | 77.51 | 64.6% | — | — |

### Verbatim standalone `aminfer bench` M1+M2 raw stdout (commit `517ba0da`)

```text
bench.gpu                        Apple M4
bench.wired_budget_mib           18186
bench.arena_mib                  81
bench.bus_peak_gbs               120.0

C9 Kernel Bandwidth Calibration
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q4   1x2048x2048           2367488      32         28.68        82.55     68.8
gemv_q4   1x6144x2048           7094272      10         80.88        87.71     73.1
gemv_q4   1x2048x768             890368      85         17.14        51.95     43.3
gemv_q4   1x4096x4096           9453568       8        106.03        89.16     74.3
gemv_q3   1x2048x2048           1843200      42         23.48        78.50     65.4
gemv_q3   1x6144x2048           5521408      14         68.79        80.27     66.9
gemv_q3   1x2048x768             693760     113         10.56        65.67     54.7
gemv_q3   1x4096x4096           7356416      10         83.90        87.68     73.1
gemm_q4   16x2048x2048          2490368      32        102.12        24.39     20.3
gemm_q4   16x4096x4096          9699328       8        399.72        24.27     20.2
gemm_q4   64x2048x2048          2883584      32        208.64        13.82     11.5
gemm_q4   64x4096x4096         10485760       8        817.72        12.82     10.7
gemm_q3   16x2048x2048          1966080      42        105.22        18.69     15.6
gemm_q3   16x4096x4096          7602176      10        406.40        18.71     15.6
gemm_q3   64x2048x2048          2359296      42        210.30        11.22      9.3
gemm_q3   64x4096x4096          8388608      10        829.09        10.12      8.4

Spike S2 Metal 4 Steady-State Hot Path
s2.rounds                        1000
s2.dispatches_per_round          32
s2.cpu_encode_us_per_round       9.52
s2.cpu_encode_ns_per_dispatch    297.5
s2.full_round_us                 100.20
s2.malloc_blocks_in_use_delta    0
s2.malloc_size_in_use_delta      0
s2.malloc_size_allocated_delta   0

Swap Check
swap.status                      ok
swap.swapouts_delta              0
swap.pageouts_delta              0
swap.used_mib                    4624

Spike S1 IOSurface KV-Tile Read Bandwidth
s1.geometry                      q=32,kv=4,d=128,chunk=512,chunks=32 (ctx=16384),banks=2
s1.tile_bytes_per_row            256
s1.tile_data_bytes               524288
s1.tile_alloc_bytes              524288
s1.tile_surface_alloc_size       524288
s1.iosurface_mlock_supported     yes
layout          access      bytes/disp   latency(us)   beta(GB/s)   bus(%)
token_major     rgba16f       33554432        418.22        80.23     66.9
token_major     r16f          33554432        365.73        91.75     76.5
token_major     alias_buf     33554432        408.48        82.14     68.5
channel_major   rgba16f       33554432        421.37        79.63     66.4
channel_major   r16f          33554432        416.37        80.59     67.2
channel_major   alias_buf     33554432        426.41        78.69     65.6
s1.swapouts_delta                0
s1.pageouts_delta                0

G3 MoE Deduplicated Gather / SwiGLU FFN (3-bit Qwen3-30B-A3B Geometry)
m    E    k   U/slots   dedup_bytes   naive_bytes   saved(%)   latency(us)   beta(GB/s)   bus(%)   naive_eq(GB/s)
1    128  8   8/8          16515072      16515072        0.0        228.34        72.33     60.3            72.33
8    128  8   24/64        49545216     132120576       62.5       2096.27        23.63     19.7            63.03
8    128  8   40/64        82575360     132120576       37.5       2124.23        38.87     32.4            62.20
g3.swapouts_delta                0
g3.pageouts_delta                0

G4 GQA Decode Attention over RGBA16Float IOSurface KV Tiles
seq_len   chunks   splits   banks   bytes/disp   latency(us)   beta(GB/s)   bus(%)
512            1        8      64      1048576         24.55        42.72     35.6
4096           8       32       8      8388608        106.98        78.42     65.3
16384         32       32       2     33554432        391.96        85.61     71.3
g4.swapouts_delta                0
g4.pageouts_delta                0

M2 End-to-End Greedy Decode from .amf (Primary & Secondary Targets)
decode_bench.target              qwen3-30b-a3b-3bit
decode_bench.arch                qwen3_moe (q3)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           1310
decode_bench.load_ms             3343.7
decode_bench.measured_layers     4
decode_bench.measured_ms_per_tok 3.508
decode_bench.measured_tok_per_s  285.05
decode_bench.measured_layer_ms   0.477
decode_bench.measured_head_ms    1.598
decode_bench.measured_bytes_tok  241564544
decode_bench.measured_beta_gbs   68.86 (57.4% bus)
decode_bench.scaled_layers       48 (linear: 48 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   24.515
decode_bench.scaled_tok_per_s    40.79
decode_bench.scaled_bytes_tok    1401238400
decode_bench.scaled_beta_gbs     57.16 (47.6% bus)
decode_bench.malloc_blocks_delta 0
decode_bench.malloc_bytes_delta  0
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      1

decode_bench.target              qwen3-14b-4bit
decode_bench.arch                qwen3 (q4)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           1561
decode_bench.load_ms             4351.0
decode_bench.measured_layers     4
decode_bench.measured_ms_per_tok 14.926
decode_bench.measured_tok_per_s  67.00
decode_bench.measured_layer_ms   2.436
decode_bench.measured_head_ms    5.182
decode_bench.measured_bytes_tok  1189256000
decode_bench.measured_beta_gbs   79.68 (66.4% bus)
decode_bench.scaled_layers       40 (linear: 40 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   102.622
decode_bench.scaled_tok_per_s    9.74
decode_bench.scaled_bytes_tok    7954260800
decode_bench.scaled_beta_gbs     77.51 (64.6% bus)
decode_bench.malloc_blocks_delta 0
decode_bench.malloc_bytes_delta  0
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      81

M2 Overall Swap Check
m2.swap.status                   ok
m2.swap.swapouts_delta           0
m2.swap.pageouts_delta           87
m2.swap.used_mib                 4616
```

### Verbatim end-to-end `synth-qwen3` → `aminfer decode` raw stdout (commit `517ba0da`)

```text
=== 1. Generate & Decode tiny-moe (3-bit) ===
synthesized 69 tensors (1196032 bytes) into /tmp/aminfer-decode-smoke-kvnRG0/tiny-moe
amf.status       ok
amf.model_id     qwen3-tiny-moe-3bit
amf.tensors      69
amf.file_bytes   1196032
amf.wired_bytes  1163264
amf.cold_bytes   0
decode.arch                      qwen3_moe (q3)
decode.layers                    2
decode.shards                    1
decode.wired_mib                 1
decode.prompt_tokens             8
decode.generated_tokens          16
decode.prefill_ms                5.120
decode.decode_ms_per_tok         0.414
decode.decode_tok_per_s          2414.62
decode.bytes_per_tok             231224
decode.effective_gbs             0.56
decode.swapouts_delta            0
decode.token_ids                 [236, 407, 107, 327, 107, 407, 359, 407, 359, 336, 407, 17, 327, 17, 327, 17]
decode.text                       v000407k v000327k v000407 v000359 v000407 v000359 v000336 v000407  v000327  v000327 
=== 2. Generate & Decode tiny-dense (4-bit) ===
synthesized 25 tensors (671744 bytes) into /tmp/aminfer-decode-smoke-kvnRG0/tiny-dense
amf.status       ok
amf.model_id     qwen3-tiny-dense-4bit
amf.tensors      25
amf.file_bytes   671744
amf.wired_bytes  638976
amf.cold_bytes   0
decode.arch                      qwen3 (q4)
decode.layers                    2
decode.shards                    1
decode.wired_mib                 1
decode.prompt_tokens             6
decode.generated_tokens          16
decode.prefill_ms                3.869
decode.decode_ms_per_tok         0.386
decode.decode_tok_per_s          2590.90
decode.bytes_per_tok             282440
decode.effective_gbs             0.73
decode.swapouts_delta            0
decode.token_ids                 [264, 301, 301, 336, 264, 9, 93, 411, 114, 114, 264, 301, 336, 28, 63, 355]
decode.text                      hell v000301 v000301 v000336hell	] v000411rrhell v000301 v000336 ? v000355
=== 3. Generate & Decode 4-layer qwen3-30b-a3b-3bit (full 151936 vocab) ===
synthesized 1575 tensors (1364869120 bytes) into /tmp/aminfer-decode-smoke-kvnRG0/q30b-4l
decode.arch                      qwen3_moe (q3)
decode.layers                    4
decode.shards                    1
decode.wired_mib                 1306
decode.prompt_tokens             5
decode.generated_tokens          16
decode.prefill_ms                72.887
decode.decode_ms_per_tok         3.576
decode.decode_tok_per_s          279.61
decode.bytes_per_tok             237534080
decode.effective_gbs             66.42
decode.swapouts_delta            0
decode.token_ids                 [121296, 151333, 151333, 151333, 106281, 53955, 53955, 24113, 52463, 23579, 13999, 7871, 18364, 18364, 7871, 23579]
decode.text                       v121296 v151333 v151333 v151333 v106281 v053955 v053955 v024113 v052463 v023579 v013999 v007871 v018364 v018364 v007871 v023579
```

## M2.5 Native FP16 / BF16 kernels and real-checkpoint benchmarks (standalone ReleaseFast)

Measured on 2026-11-20–2026-11-21 on the Apple M4 24 GiB baseline (`Mac16,12`,
macOS 27.2 `26B123`, default kernel tunables `wired_budget_mib = 18186`) at git
commit `1893ab6` (incorporating `2a48c66` W12 `.amf` `bf16` + multi-shard
streaming packer and `32dc321` G12 Metal 4 native `FP16`/`BF16` kernels) via
standalone non-root ReleaseFast `./zig-out/bin/aminfer`. All `.amf` files and
temporary artifacts were placed strictly on external NVMe
(`/Volumes/Akari/models/amf/`, `TMPDIR=/Volumes/Akari/.tmp`).

### M2.5 methodology notes

1. **Multi-shard streaming `.amf` conversion (`W12`)**: All three local
   Hugging Face / MLX checkpoints (`Qwen/Qwen3-0.6B` BF16,
   `mlx-community/Qwen3-14B-4bit`, and `mlx-community/Qwen3-30B-A3B-3bit`) were
   converted via `aminfer-tools pack-amf --dtype auto` (two-pass header-first
   layout + bounded-RAM streaming with incremental CRC-32; peak RSS measured via
   `/usr/bin/time -l`) and validated end-to-end with ReleaseFast
   `./zig-out/bin/aminfer verify-amf` (`F_NOCACHE`, `<= 8 MiB` streaming CRC
   buffer).
2. **Native `FP16` / `BF16` GEMV & GEMM calibration (`G12`, `aminfer bench-m25`)**:
   Mirrors M1's 16 C9 shapes (`1×2048×2048`, `1×6144×2048`, `1×2048×768`,
   `1×4096×4096`, plus `M = 16` and `M = 64` on `2048×2048` and `4096×4096`)
   over a single 97 MiB `Budget`-checked `WiredArena` with a 96 MiB rotating
   weight pool (3–32 slots per shape so even 32 MiB `4096×4096` matrices bust
   M4 L2/SLC). Algorithmic traffic per dispatch is `weight + x + y`
   (`(N*K + M*K + M*N) * 2` bytes); compute throughput is reported as
   `TFLOP/s = (2 * M * N * K * dispatches) / (elapsed_ns * 1000)`.
3. **Real-checkpoint full-depth decode & `bench-decode` (`W13`)**: Executes the
   complete real 28-layer (`Qwen3-0.6B` BF16), 40-layer (`Qwen3-14B-4bit`), and
   48-layer (`Qwen3-30B-A3B-3bit`) models end-to-end through `aminfer decode`
   (English & Chinese ChatML prompts, `--max-tokens 32`, checking coherent
   readable text and `swapouts_delta = 0`) and `aminfer bench-decode
   --prefill-tokens 511 --decode-steps 16` (measured at exact `seq_len = 512`).
4. **Hot-path `malloc_zone_statistics` note**: As documented in
   [ADR-0011](../decisions/0011-metal4-zero-allocation-hot-path.md), a fresh
   `MTL4CommandQueue` lazily fills Apple's bounded internal Metal 4 driver pool
   (~2,000 × 64 B ≈ 128 KiB) across early commits before flattening to `0` in
   steady state. Accordingly, standalone short `decode` runs (6–32 steps) show
   `2 × 64 B` per commit inside the driver pool fill window, while full
   `bench-decode` (`515` warmup commits) reaches `0 blocks / 0 B` on
   `Qwen3-0.6B` BF16 and `2 blocks / 352 B` on `Qwen3-14B-4bit` /
   `Qwen3-30B-A3B-3bit` with zero allocations from aminfer and `swapouts_delta = 0`
   across every run.

### Multi-shard `.amf` conversion and verification summary (`W12`)

| Real checkpoint | Source shards | `.amf` tensors | Precision mix in `.amf` (`--dtype auto`) | `.amf` file size (B) | Pack time (s) | Pack peak RSS | `verify-amf` time (s) | `verify-amf` RSS |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **`Qwen/Qwen3-0.6B`** | 1 | 310 | Pure `BF16` (`lm_head` deduped via `tie_word_embeddings = true`) | 1,193,885,696 | 4.7 | 35 MB | 4.6 | 21 MB |
| **`mlx-community/Qwen3-14B-4bit`** | 2 | 443 | 282 `q4_affine` (incl. `embed_tokens`, `lm_head`) + 161 `bf16` norms | 8,309,669,888 | 50.4 | 116 MB | 28.3 | 21 MB |
| **`mlx-community/Qwen3-30B-A3B-3bit`** | 3 | 18,867 | 18,432 per-expert `q3_affine` + 192 `q3_affine` attn + 48 `q3_affine` routers + `q3_affine` embed/head + 193 `bf16` norms | 13,363,314,688 | 111.6 | 127 MB | 50.9 | 25 MB |

Peak Python RSS stayed well below the 512 MiB target (**35–127 MB** even when
streaming and slicing the 12.44 GiB, 3-shard `Qwen3-30B-A3B-3bit` checkpoint
into 18,867 16 KiB-aligned blobs), and Zig `verify-amf` stayed at **21–25 MB**
RSS across all three files.

### M2.5 native `FP16` / `BF16` GEMV & GEMM calibration summary (`G12`)

Measured via standalone ReleaseFast `./zig-out/bin/aminfer bench-m25`
(`wired_budget_mib = 18186`, `arena_mib = 97`, `swapouts_delta = 0`,
`pageouts_delta = 0`):

| Kernel | Shape (`M×N×K`) | Bytes/dispatch | Weight slots | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus | Compute (TFLOP/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemv_f16` | `1×2048×2048` | 8,396,800 | 12 | 110.39 | 76.07 | 63.4% | 0.076 |
| `gemv_f16` | `1×6144×2048` | 25,182,208 | 4 | 319.71 | 78.77 | 65.6% | 0.079 |
| `gemv_f16` | `1×2048×768` | 3,151,360 | 32 | 40.98 | 76.90 | 64.1% | 0.077 |
| `gemv_f16` | `1×4096×4096` | 33,570,816 | 3 | 386.94 | **86.76** | **72.3%** | 0.087 |
| `gemv_bf16` | `1×2048×2048` | 8,396,800 | 12 | 108.40 | 77.46 | 64.5% | 0.077 |
| `gemv_bf16` | `1×6144×2048` | 25,182,208 | 4 | 320.88 | 78.48 | 65.4% | 0.078 |
| `gemv_bf16` | `1×2048×768` | 3,151,360 | 32 | 36.70 | 85.86 | 71.6% | 0.086 |
| `gemv_bf16` | `1×4096×4096` | 33,570,816 | 3 | 380.39 | **88.25** | **73.5%** | 0.088 |
| `gemm_f16` | `16×2048×2048` | 8,519,680 | 12 | 121.76 | 69.97 | 58.3% | 1.102 |
| `gemm_f16` | `16×4096×4096` | 33,816,576 | 3 | 444.96 | 76.00 | 63.3% | 1.207 |
| `gemm_f16` | `64×2048×2048` | 8,912,896 | 12 | 199.55 | 44.67 | 37.2% | **2.690** |
| `gemm_f16` | `64×4096×4096` | 34,603,008 | 3 | 913.26 | 37.89 | 31.6% | 2.351 |
| `gemm_bf16` | `16×2048×2048` | 8,519,680 | 12 | 153.73 | 55.42 | 46.2% | 0.873 |
| `gemm_bf16` | `16×4096×4096` | 33,816,576 | 3 | 489.96 | 69.02 | 57.5% | 1.096 |
| `gemm_bf16` | `64×2048×2048` | 8,912,896 | 12 | 214.01 | 41.65 | 34.7% | **2.509** |
| `gemm_bf16` | `64×4096×4096` | 34,603,008 | 3 | 868.59 | 39.84 | 33.2% | 2.472 |

Notes:
- Native `BF16` GEMV matches or slightly edges `FP16` GEMV (**88.25 GB/s / 73.5%
  bus** vs **86.76 GB/s / 72.3% bus** on `1×4096×4096`), confirming zero penalty
  for preserving upstream `BF16` weights natively on Apple9 Metal 4.
- Batched `simdgroup_half8x8` and `simdgroup_bfloat8x8` GEMM (accumulating into
  `simdgroup_float8x8` `FP32`) sustain **2.47–2.69 TFLOP/s** at `M = 64` and
  **1.10–1.21 TFLOP/s** (`69–76 GB/s`) at `M = 16`, directly benefiting M3
  speculative tree verification and M4 prefill.

### Real-checkpoint full-layer decode summary vs M2 4-layer projections (`seq_len = 512`)

| Real model (`.amf`) | Layers | Shards | Wired (MiB) | `load_ms` | Measured `ms/tok` | Measured `tok/s` | Bytes/tok | Effective `beta` (GB/s / % bus) | M2 4L linear projection (`tok/s`) | Delta vs M2 projection | `swapouts` delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| **`Qwen3-0.6B` (BF16, tied)** | 28 | 1 | 1,251 | 4,433.8 | 20.884 | **47.88** | 1,250,936,832 | 59.90 (49.9%) | 55.20 *(M2.5 4L synth)* | -13.3% *(short 32-tok EN/ZH: 62.98–66.17 tok/s, 75.5–79.3 GB/s)* | 0 |
| **`Qwen3-14B-4bit` (q4 + bf16 norms)** | 40 | 1 | 8,086 | 29,793.1 | 104.997 | **9.52** | 7,954,260,800 | 75.76 (63.1%) | 9.74 *(M2 4L synth)* | **-2.3%** (`2.423` vs `2.436 ms/L`, exact bytes match; short EN/ZH: 10.56–10.67 tok/s, 83.1–84.0 GB/s) | 0 |
| **`Qwen3-30B-A3B-3bit` (q3 + bf16 norms)** | 48 | 1 | 12,839 | 53,596.4 | 19.282 | **51.86** | 1,381,577,600 | 71.65 (59.7%) | 40.79 *(M2 4L synth)* | **+27.1%** (`0.402` vs `0.477 ms/L`; quantized `q3` routers save 19.7 MB/tok + 48L command-buffer amortization + real expert locality) | 0 |

Real-checkpoint interactive `aminfer decode` summary (English & Chinese ChatML
prompts, `--max-tokens 32`, `swapouts_delta = 0` on all runs):

| Real model | Prompt | Generated text | `decode_tok_per_s` | Effective `beta` (GB/s) | `swapouts` delta |
| --- | --- | --- | ---: | ---: | ---: |
| `Qwen3-0.6B` (`bf16`) | EN capital | `The capital of France is Paris.<\|im_end\|>` | 51.27 | 61.31 | 0 |
| `Qwen3-0.6B` (`bf16`) | ZH capital | `中国的首都是北京。<\|im_end\|>` | 50.31 | 60.14 | 0 |
| `Qwen3-0.6B` (`bf16`) | EN UMA (32 tok) | `Unified Memory is a feature in Apple Silicon that allows the system to share data between the CPU and the GPU, improving performance by reducing the need for frequent memory swaps` | **62.98** | 75.48 | 0 |
| `Qwen3-0.6B` (`bf16`) | ZH UMA (32 tok) | `苹果的统一内存架构（Unified Memory Architecture, UMA）通过将多个核心的内存共享为一个单一的内存池，提高了内存的利用率和` | **66.17** | 79.29 | 0 |
| `Qwen3-14B-4bit` (`q4`) | EN capital | `The capital of France is Paris.<\|im_end\|>` | 10.67 | 84.00 | 0 |
| `Qwen3-14B-4bit` (`q4`) | ZH capital | `中国的首都是北京。<\|im_end\|>` | 10.56 | 83.13 | 0 |
| `Qwen3-30B-A3B-3bit` (`q3`) | EN capital | `The capital of France is Paris.<\|im_end\|>` | 50.49 | 67.36 | 0 |
| `Qwen3-30B-A3B-3bit` (`q3`) | ZH capital | `中国的首都是北京。<\|im_end\|>` | 42.61 | 56.84 | 0 |

### Verbatim standalone `aminfer bench-m25` raw stdout (commit `1893ab6`)

```text
M2.5 Native FP16 / BF16 GEMV & GEMM Bandwidth Calibration
m25.wired_budget_mib             18186
m25.arena_mib                    97
kernel     shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)   tflops
gemv_f16   1x2048x2048           8396800      12        110.39        76.07     63.4    0.076
gemv_f16   1x6144x2048          25182208       4        319.71        78.77     65.6    0.079
gemv_f16   1x2048x768            3151360      32         40.98        76.90     64.1    0.077
gemv_f16   1x4096x4096          33570816       3        386.94        86.76     72.3    0.087
gemv_bf16  1x2048x2048           8396800      12        108.40        77.46     64.5    0.077
gemv_bf16  1x6144x2048          25182208       4        320.88        78.48     65.4    0.078
gemv_bf16  1x2048x768            3151360      32         36.70        85.86     71.6    0.086
gemv_bf16  1x4096x4096          33570816       3        380.39        88.25     73.5    0.088
gemm_f16   16x2048x2048          8519680      12        121.76        69.97     58.3    1.102
gemm_f16   16x4096x4096         33816576       3        444.96        76.00     63.3    1.207
gemm_f16   64x2048x2048          8912896      12        199.55        44.67     37.2    2.690
gemm_f16   64x4096x4096         34603008       3        913.26        37.89     31.6    2.351
gemm_bf16  16x2048x2048          8519680      12        153.73        55.42     46.2    0.873
gemm_bf16  16x4096x4096         33816576       3        489.96        69.02     57.5    1.096
gemm_bf16  64x2048x2048          8912896      12        214.01        41.65     34.7    2.509
gemm_bf16  64x4096x4096         34603008       3        868.59        39.84     33.2    2.472
m25.dense.swapouts_delta         0
m25.dense.pageouts_delta         0

M2.5 Synthetic BF16 End-to-End Greedy Decode from .amf (Qwen3-0.6B Geometry)
decode_bench.target              qwen3-0.6b-bf16-synth
decode_bench.arch                qwen3 (bf16)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           433
decode_bench.load_ms             1545.9
decode_bench.measured_layers     4
decode_bench.measured_ms_per_tok 5.539
decode_bench.measured_tok_per_s  180.54
decode_bench.measured_layer_ms   0.524
decode_bench.measured_head_ms    3.443
decode_bench.measured_bytes_tok  445421568
decode_bench.measured_beta_gbs   80.42 (67.0% bus)
decode_bench.scaled_layers       28 (linear: 28 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   18.115
decode_bench.scaled_tok_per_s    55.20
decode_bench.scaled_bytes_tok    1250936832
decode_bench.scaled_beta_gbs     69.06 (57.5% bus)
decode_bench.malloc_blocks_delta 32
decode_bench.malloc_bytes_delta  2048
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      0

M2.5 Overall Swap Check
m25.swap.status                  ok
m25.swap.swapouts_delta          0
m25.swap.pageouts_delta          0
m25.swap.used_mib                10410
```

### Verbatim real-checkpoint `verify-amf`, `decode`, and `bench-decode` raw stdout (commit `1893ab6`)

```text
=== 1. Qwen3-0.6B BF16 (28L, tied embeddings) ===
amf.status       ok
amf.model_id     qwen3-0.6b-bf16
amf.tensors      310
amf.file_bytes   1193885696
amf.wired_bytes  1193820160
amf.cold_bytes   0

decode.arch                      qwen3 (bf16)
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             24
decode.generated_tokens          8
decode.prefill_ms                441.664
decode.decode_ms_per_tok         19.505
decode.decode_tok_per_s          51.27
decode.bytes_per_tok             1195771904
decode.effective_gbs             61.31
decode.malloc_blocks_delta       14
decode.malloc_bytes_delta        896
decode.swapouts_delta            0
decode.token_ids                 [785, 6722, 315, 9625, 374, 12095, 13, 151645]
decode.text                      The capital of France is Paris.<|im_end|>

decode.arch                      qwen3 (bf16)
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             22
decode.generated_tokens          6
decode.prefill_ms                395.189
decode.decode_ms_per_tok         19.876
decode.decode_tok_per_s          50.31
decode.bytes_per_tok             1195313152
decode.effective_gbs             60.14
decode.malloc_blocks_delta       10
decode.malloc_bytes_delta        640
decode.swapouts_delta            0
decode.token_ids                 [105538, 59975, 100132, 68990, 1773, 151645]
decode.text                      中国的首都是北京。<|im_end|>

decode.arch                      qwen3 (bf16)
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             23
decode.generated_tokens          32
decode.prefill_ms                307.262
decode.decode_ms_per_tok         15.877
decode.decode_tok_per_s          62.98
decode.bytes_per_tok             1198409728
decode.effective_gbs             75.48
decode.malloc_blocks_delta       62
decode.malloc_bytes_delta        3968
decode.swapouts_delta            0
decode.token_ids                 [85379, 13850, 374, 264, 4565, 304, 8162, 37150, 429, 6147, 279, 1849, 311, 4332, 821, 1948, 279, 13940, 323, 279, 22670, 11, 18392, 5068, 553, 17719, 279, 1184, 369, 20757, 4938, 73151]
decode.text                      Unified Memory is a feature in Apple Silicon that allows the system to share data between the CPU and the GPU, improving performance by reducing the need for frequent memory swaps

decode.arch                      qwen3 (bf16)
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             23
decode.generated_tokens          32
decode.prefill_ms                291.474
decode.decode_ms_per_tok         15.114
decode.decode_tok_per_s          66.17
decode.bytes_per_tok             1198409728
decode.effective_gbs             79.29
decode.malloc_blocks_delta       62
decode.malloc_bytes_delta        3968
decode.swapouts_delta            0
decode.token_ids                 [104167, 9370, 103967, 111390, 106379, 9909, 85379, 13850, 37843, 11, 547, 4835, 7552, 67338, 44063, 101213, 100185, 9370, 111390, 101203, 17714, 46944, 106215, 9370, 111390, 100230, 3837, 105953, 111390, 9370, 113766, 33108]
decode.text                      苹果的统一内存架构（Unified Memory Architecture, UMA）通过将多个核心的内存共享为一个单一的内存池，提高了内存的利用率和

decode_bench.target              qwen3
decode_bench.arch                qwen3 (bf16)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           1251
decode_bench.load_ms             4433.8
decode_bench.measured_layers     28
decode_bench.measured_ms_per_tok 20.884
decode_bench.measured_tok_per_s  47.88
decode_bench.measured_layer_ms   0.574
decode_bench.measured_head_ms    4.815
decode_bench.measured_bytes_tok  1250936832
decode_bench.measured_beta_gbs   59.90 (49.9% bus)
decode_bench.scaled_layers       28 (linear: 28 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   20.884
decode_bench.scaled_tok_per_s    47.88
decode_bench.scaled_bytes_tok    1250936832
decode_bench.scaled_beta_gbs     59.90 (49.9% bus)
decode_bench.malloc_blocks_delta 0
decode_bench.malloc_bytes_delta  0
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      1

=== 2. Qwen3-14B-4bit (40L, q4_affine + bf16 norms) ===
amf.status       ok
amf.model_id     qwen3-14b-4bit
amf.tensors      443
amf.file_bytes   8309669888
amf.wired_bytes  8309587968
amf.cold_bytes   0

decode.arch                      qwen3 (q4)
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             24
decode.generated_tokens          8
decode.prefill_ms                2364.012
decode.decode_ms_per_tok         93.758
decode.decode_tok_per_s          10.67
decode.bytes_per_tok             7875453760
decode.effective_gbs             84.00
decode.malloc_blocks_delta       16
decode.malloc_bytes_delta        1504
decode.swapouts_delta            0
decode.token_ids                 [785, 6722, 315, 9625, 374, 12095, 13, 151645]
decode.text                      The capital of France is Paris.<|im_end|>

decode.arch                      qwen3 (q4)
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             22
decode.generated_tokens          6
decode.prefill_ms                2059.952
decode.decode_ms_per_tok         94.730
decode.decode_tok_per_s          10.56
decode.bytes_per_tok             7874798400
decode.effective_gbs             83.13
decode.malloc_blocks_delta       12
decode.malloc_bytes_delta        1184
decode.swapouts_delta            0
decode.token_ids                 [105538, 59975, 100132, 68990, 1773, 151645]
decode.text                      中国的首都是北京。<|im_end|>

decode_bench.target              qwen3
decode_bench.arch                qwen3 (q4)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           8086
decode_bench.load_ms             29793.1
decode_bench.measured_layers     40
decode_bench.measured_ms_per_tok 104.997
decode_bench.measured_tok_per_s  9.52
decode_bench.measured_layer_ms   2.423
decode_bench.measured_head_ms    8.074
decode_bench.measured_bytes_tok  7954260800
decode_bench.measured_beta_gbs   75.76 (63.1% bus)
decode_bench.scaled_layers       40 (linear: 40 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   104.997
decode_bench.scaled_tok_per_s    9.52
decode_bench.scaled_bytes_tok    7954260800
decode_bench.scaled_beta_gbs     75.76 (63.1% bus)
decode_bench.malloc_blocks_delta 2
decode_bench.malloc_bytes_delta  352
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      7

=== 3. Qwen3-30B-A3B-3bit (48L, 128 experts, q3_affine + bf16 norms) ===
amf.status       ok
amf.model_id     qwen3-30b-a3b-3bit
amf.tensors      18867
amf.file_bytes   13363314688
amf.wired_bytes  13360873472
amf.cold_bytes   0

decode.arch                      qwen3_moe (q3)
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             24
decode.generated_tokens          8
decode.prefill_ms                782.584
decode.decode_ms_per_tok         19.808
decode.decode_tok_per_s          50.49
decode.bytes_per_tok             1334293376
decode.effective_gbs             67.36
decode.malloc_blocks_delta       16
decode.malloc_bytes_delta        1248
decode.swapouts_delta            0
decode.token_ids                 [785, 6722, 315, 9625, 374, 12095, 13, 151645]
decode.text                      The capital of France is Paris.<|im_end|>

decode.arch                      qwen3_moe (q3)
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             22
decode.generated_tokens          6
decode.prefill_ms                635.024
decode.decode_ms_per_tok         23.469
decode.decode_tok_per_s          42.61
decode.bytes_per_tok             1333900160
decode.effective_gbs             56.84
decode.malloc_blocks_delta       10
decode.malloc_bytes_delta        640
decode.swapouts_delta            0
decode.token_ids                 [105538, 59975, 100132, 68990, 1773, 151645]
decode.text                      中国的首都是北京。<|im_end|>

decode_bench.target              qwen3_moe
decode_bench.arch                qwen3_moe (q3)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           12839
decode_bench.load_ms             53596.4
decode_bench.measured_layers     48
decode_bench.measured_ms_per_tok 19.282
decode_bench.measured_tok_per_s  51.86
decode_bench.measured_layer_ms   0.402
decode_bench.measured_head_ms    0.000
decode_bench.measured_bytes_tok  1381577600
decode_bench.measured_beta_gbs   71.65 (59.7% bus)
decode_bench.scaled_layers       48 (linear: 48 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   19.301
decode_bench.scaled_tok_per_s    51.81
decode_bench.scaled_bytes_tok    1381577600
decode_bench.scaled_beta_gbs     71.58 (59.7% bus)
decode_bench.malloc_blocks_delta 2
decode_bench.malloc_bytes_delta  352
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      16
```


## M3 speculative decoding and Spikes S4 & S5

Measured on 2026-11-21 on the Apple M4 24 GiB baseline at commit `034492f`
(standalone ReleaseFast `./zig-out/bin/aminfer bench-m3`, `decode`, and
`spec-decode`, default kernel tunables, `wired_budget_mib = 18186`,
`swapouts_delta = 0` and `prefill_swapouts_delta = 0` across all archived runs).

All real-checkpoint `T = 0` comparisons were machine-verified for **100%
bit-for-bit identical generated `token_ids`** between non-speculative
`aminfer decode` and `aminfer spec-decode`. Hot-path engine data structures
(L1 suffix automaton, draft trie, D7 pruner, GPU verification workspaces, and
acceptance buffers) perform **zero Zig dynamic heap allocations** after startup;
small non-zero `malloc_blocks_delta` values on short CLI decode runs (~64 B per
Metal 4 command-buffer submission) reflect Apple's bounded internal Metal 4
driver pool warming up below the 2,048-round steady-state threshold documented
in Spike S2 (`sam.malloc_blocks_delta = 0` on CPU, `s2.malloc_blocks_in_use_delta = 0`
in steady state).

Real EAGLE-3 draft checkpoint packed into `.amf` v0 (`W12`/`D2`) via
`aminfer-tools pack-amf` and validated with `aminfer verify-amf`:
- `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` → `qwen3-30b-a3b-eagle3.amf`:
  15 tensors (`bf16` weights + `u32` `d2t` 32,000-entry reduced-vocab map),
  `1,044,217,856 B` (`996 MiB`), `wired_bytes = 1,044,185,088 B`, `cold_bytes = 0`.

### Spike S4: CoreML ANE zero-copy handoff, D5 placement gate, and DRAM contention

| Measurement | Workload | ANE (`us`) | ANE + handoff (`us`) | GPU in-CB (`us`) | `uses_ane` / `zero_copy` | Decision / Delta |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Round-trip handoff | `GPU → CPU → ANE → CPU → GPU` tiny IOSurface | 92.46 *(ANE predict)* | **574.64** *(total round-trip)* | 332.33 *(GPU baseline)* | `yes` / `yes` | `574.64 us` handoff overhead per round |
| Draft head (`M = 1`) | `2L × 2048×2048 fp16` | 366.93 | 941.57 | **201.87** | `yes` / `yes` | **`d5_gate = gpu`** (`4.66×` faster) |
| Draft head (`batch = 32`) | `2L × 2048×2048 fp16` | 363.20 | 937.84 | **277.36** | `yes` / `yes` | **`d5_gate = gpu`** (`3.38×` faster) |
| DRAM contention (`q3`) | Concurrent 40 MiB ANE stream (`50.00 GB/s`) + GPU `gemv_q3` | — | — | solo `70.91 GB/s` → concurrent `49.09 GB/s` | `yes` | **`-30.8%`** GPU bandwidth |
| DRAM contention (`q4`) | Concurrent 40 MiB ANE stream (`50.00 GB/s`) + GPU `gemv_q4` | — | — | solo `77.92 GB/s` → concurrent `52.57 GB/s` | `yes` | **`-32.5%`** GPU bandwidth |

Takeaways (closing **Spike S4** and settling **D5** on Apple M4):
- Zero-copy IOSurface backing between Metal and CoreML/ANE (`uses_ane = yes`,
  `zero_copy_backing = yes`) works without host copies, but a full round-trip
  handoff costs `574.64 us`. Because EAGLE/MTP drafting is serial with
  verification ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)),
  in-command-buffer GPU execution beats ANE+handoff by `3.38–4.66×`, so the
  `D5` startup calibration gate selects `.gpu` on Apple M4.
- Concurrent 40 MiB ANE weight streaming sustains `50.00 GB/s` on unified DRAM
  and trims concurrent GPU GEMV bandwidth by `30.8–32.5%`, providing concrete
  contention parameters for M4/M5 scheduling (`A1`, `A2`, `D8`).

### Spike S5: Unique active experts vs tree size and CSR-dedup vs per-slot MoE

Measured on `Qwen3-30B-A3B` MoE geometry (`q3`, `E = 128`, `top_k = 8`,
`H = 2048`, `I = 768`, 252.0 MiB rotating expert weight pool):

| Routing | `M` | Mean `\|U(T)\|` | Total slots | Weight reads saved | CSR-dedup (`us`) | Per-slot (`us`) | Dedup `beta` (GB/s) | Per-slot unique `beta` (GB/s) | Per-slot raw `beta` (GB/s) | Dedup speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correlated | 1 | 8.0 | 8 | 0.0% | 258.24 | 237.32 | 63.95 | 69.59 | 69.59 | 0.92× |
| Correlated | 2 | 9.3 | 16 | 41.8% | 365.90 | 367.26 | 52.54 | 52.35 | 89.94 | **1.00×** *(crossover)* |
| Correlated | 4 | 11.2 | 32 | 65.0% | 512.37 | 695.38 | 45.08 | 33.21 | 95.00 | **1.36×** |
| Correlated | 8 | 13.6 | 64 | 78.7% | 764.80 | 1,360.09 | **36.78** | 20.68 | 97.14 | **1.78×** |
| Correlated | 16 | 17.4 | 128 | 86.4% | 1,475.00 | 2,758.62 | 24.32 | 13.00 | 95.79 | **1.87×** |
| Correlated | 32 | 23.0 | 256 | 91.0% | 2,942.96 | 5,431.57 | 16.13 | 8.74 | 97.30 | **1.85×** |
| Uniform | 1 | 8.0 | 8 | 0.0% | 278.25 | 251.84 | 59.35 | 65.58 | 65.58 | 0.91× |
| Uniform | 2 | 15.4 | 16 | 3.5% | 481.17 | 434.38 | 66.23 | 73.37 | 76.04 | 0.90× |
| Uniform | 4 | 29.2 | 32 | 8.8% | 881.16 | 788.50 | 68.38 | 76.42 | 83.78 | 0.89× |
| Uniform | 8 | 52.5 | 64 | 18.0% | 1,596.99 | 1,408.13 | 67.87 | 76.97 | 93.83 | 0.88× |
| Uniform | 16 | 82.0 | 128 | 35.9% | 2,518.45 | 2,719.27 | 67.22 | 62.25 | 97.17 | **1.08×** *(crossover)* |
| Uniform | 32 | 112.6 | 256 | 56.0% | 4,164.71 | 5,618.64 | 55.83 | 41.38 | 94.06 | **1.35×** |

Takeaways (closing **Spike S5** and calibrating **D7**):
- Under correlated tree routing, CSR deduplication matches per-slot at `M = 2`
  and beats it by **`1.36–1.87×`** for `M = 4..32` (`65.0–91.0%` fewer expert
  weight reads). Under uniform routing, crossover occurs at `M = 16` (`1.08×`).
- Calibrated `beta_dedup = 36.78 GB/s` at `M = 8` (and the steeper per-token
  cost curve as `|U(T)|` grows) feeds the `D7` cost model directly
  ([ADR-0009](../decisions/0009-expert-byte-budget-tree-pruning.md)).

### L1 suffix automaton latency (`D1`, 32,768-token context)

| Metric | Measured value | Charter target |
| --- | ---: | --- |
| Preallocated arena size (`max_tokens = 32,768`) | `5.14 MiB` | Fixed preallocated |
| Online append latency | `0.108 us/tok` | Amortized O(1) |
| Active-state query latency (10,000 queries) | **`0.041 us`** (`41 ns`) | **`< 50.00 us`** (`1,220×` headroom) |
| Suffix-walk query latency | `0.154 us` | `< 50.00 us` |
| Max observed single-query latency | `5.917 us` | `< 50.00 us` |
| Hot-path `malloc` delta (`blocks` / `bytes`) | **`0` / `0 B`** | `0` |

### Real-checkpoint end-to-end speculative decoding summary (`D1`–`D3`, `D5`–`D7`, `D9`, `G5`)

| Real model & workload | Mode | `T` | Prompt → Gen | `prefill_ms` | `decode tok/s` | Speedup vs base | Mean `tau` | Drafted → Verified nodes | L1 seed rounds | Unique experts / L | Bytes / acc tok (vs base) | Exact `token_ids` match | `swapouts` delta |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |
| **`Qwen3-0.6B` (BF16, 28L)** TypeScript edit | Baseline `decode` | 0.0 | 106 → 55 | 1,179.3 | 69.20 | 1.00× | 1.000 | — | — | — | 1,210.6 MB | baseline | 0 |
| **`Qwen3-0.6B` (BF16, 28L)** TypeScript edit | `L1` (`cap=8`) | 0.0 | 106 → 55 | 107.2 *(11.0×)* | **327.49** | **4.73×** | **5.600** | 11.80 → 6.40 | 6 / 10 | — | 215.6 MB (`-82.2%`) | **55 / 55 (100%)** | 0 |
| **`Qwen3-0.6B` (BF16, 28L)** Fibonacci prompt | `L1` stochastic (`seed=42`) | 0.7 | 13 → 48 | 39.3 | 69.28 | — | 1.175 | 3.90 → 2.23 | 1 / 40 | — | 1,018.1 MB | stochastic `D9` | 0 |
| **`Qwen3-14B-4bit` (q4, 40L)** TypeScript edit | Baseline `decode` | 0.0 | 106 → 60 | 10,198.8 | 10.23 | 1.00× | 1.000 | — | — | — | 7,897.4 MB | baseline | 0 |
| **`Qwen3-14B-4bit` (q4, 40L)** TypeScript edit | `L1` (`cap=8`) | 0.0 | 106 → 60 | 3,262.6 *(3.13×)* | **17.45** | **1.71×** | **4.357** | 12.14 → 6.43 | 8 / 14 | — | 1,811.6 MB (`-77.1%`) | **60 / 60 (100%)** | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** C++20 reasoning (`<think>`) | Baseline `decode` | 0.0 | 44 → 64 | 1,039.9 | 46.00 | 1.00× | 1.000 | — | — | 8.00 | 1,341.8 MB | baseline | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** C++20 reasoning (`<think>`) | `EAGLE-3` `--no-prune` (`cap=16`) | 0.0 | 44 → 64 | 580.4 | 24.93 | 0.54× | 3.150 | 12.65 → 12.65 | 1 / 20 | 31.28 | 1,158.0 MB (`-13.7%`) | **64 / 64 (100%)** | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** C++20 reasoning (`<think>`) | **`EAGLE-3 + L1 + D7`** (`cap=16`) | 0.0 | 44 → 64 | 597.5 *(1.74×)* | **48.38** | **1.05×** *(**1.94×** vs unpruned)* | 2.333 | 13.44 → 2.93 | 3 / 27 | **15.83** (`-49.4%`) | **906.4 MB (`-32.4%`)** | **64 / 64 (100%)** | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** TypeScript edit (`/no_think`) | Baseline `decode` | 0.0 | 82 → 59 | 1,754.3 | 50.15 | 1.00× | 1.000 | — | — | 8.00 | 1,345.0 MB | baseline | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** TypeScript edit (`/no_think`) | **`L1 + D7`** (`cap=16`) | 0.0 | 82 → 59 | 832.5 *(2.11×)* | **59.24** | **1.18×** | 2.500 | 11.83 → 2.92 | 17 / 24 | 15.82 | 846.8 MB (`-37.0%`) | **59 / 59 (100%)** | 0 |
| **`Qwen3-30B-A3B-3bit` (48L MoE)** TypeScript edit (`/no_think`) | `EAGLE-3 + L1 + D7` (`cap=16`) | 0.0 | 82 → 59 | 1,042.4 | 33.82 | 0.67× | **3.412** | 14.65 → 3.88 | **14 / 17** | 19.79 | **736.0 MB (`-45.3%`)** | **59 / 59 (100%)** | 0 |

Key engineering findings from M3 real-checkpoint benchmarks:
1. **ADR-0009 validated empirically on 48-layer MoE**: Verifying an unpruned
   `cap = 16` EAGLE-3 tree on `Qwen3-30B-A3B-3bit` explodes active experts from
   `8.00` to `31.28/layer` (`verify = 109.12 ms/round`) and slows decode to
   `24.93 tok/s` (`0.54×` baseline). `D7` submodular expert-byte-budget pruning
   runs in `0.390 ms/round`, trims `13.44 → 2.93` verified nodes, halves active
   experts to `15.83/layer` (`-49.4%`), cuts DRAM traffic `-32.4%`, and achieves
   **`48.38 tok/s`** (**`1.94×`** vs unpruned, **`1.05×`** vs baseline).
2. **Dual-source `D6` synergy vs unquantized BF16 draft-head cost**: On
   repetitive code editing, combining `L1` (`14/17` seed rounds) with `EAGLE-3`
   lifts `tau` from `2.500` to **`3.412`** (`accept_hist = [1:2, 2:1, 3:3, 4:10, 5:1]`)
   and achieves the lowest DRAM traffic (**`736.0 MB/acc_tok`, `-45.3%`**).
   However, because the upstream EAGLE-3 checkpoint is unquantized `BF16`
   (`996 MiB`), sequential `syncAccepted` (`8.31 ms/round`) plus the wider
   `3.88`-node (`19.79`-expert) verification pass makes `L1-only + D7` faster
   on repetitive edits (**`59.24 tok/s`, `1.18×`** vs baseline), while
   `EAGLE-3 + L1 + D7` wins on non-repetitive reasoning (`48.38 tok/s`).
   Quantizing the EAGLE-3 head to `q4` (~250 MiB) and adding learned cross-layer
   expert prediction (`A2`) + `simdgroup_matrix` batched MoE tiling are clear
   M4/M5 follow-ups.
3. **Dense targets & batched prefill win immediately**: Because dense targets
   have zero expert-union explosion, `L1` speculative decoding delivers
   **`4.73×`** (`327.49 tok/s`, `-82.2%` bytes) on `Qwen3-0.6B` BF16 and
   **`1.71×`** (`17.45 tok/s`, `-77.1%` bytes) on `Qwen3-14B-4bit`, while
   batched `prefillChunk` accelerates prompt prefill by **`1.74–11.0×`**.
4. **Numeric boundary note (`seq_len > 128` on 48L MoE)**: Unifying the
   weight-stationary `dequant8 + dot` path in `moe_ffn.metal` and gating
   8-token tile-max to `num_splits > 1` in `kv_attention.metal` yields 100%
   bit-for-bit parity between `decode` and `spec-decode` across all four
   archived suites above (including dense `Qwen3-14B-4bit` at `seq_len = 166`
   and 48L MoE `Qwen3-30B-A3B-3bit` at `seq_len = 141`). On longer 48L MoE
   prompts where `chooseNumSplits(seq_len)` vs `chooseTreeNumSplits(base_pos)`
   crosses the `128`-token FlashDecoding split boundary mid-tree, 1-ULP FP32
   attention reduction order differences can occasionally flip a near-tie MoE
   router gate across 48 layers.

### Verbatim standalone `aminfer bench-m3` raw stdout (commit `034492f`)

```text
=== M3 Speculative Decoding & Placement Gate Benchmarks ===

Spike S4 CoreML ANE Zero-Copy Handoff, D5 Gate & Bandwidth Contention
s4.ane_cores                     16
s4.handoff.uses_ane              yes
s4.handoff.zero_copy_backing     yes
s4.handoff.roundtrip_us          574.64
s4.handoff.ane_tiny_predict_us   92.46
s4.handoff.gpu_baseline_us       332.33
s4.draft.shape                   2L x 2048x2048 fp16 (batch=32)
s4.draft.m1.ane_uses_ne          yes (zero_copy=yes)
s4.draft.m1.ane_us               366.93 (with_handoff=941.57)
s4.draft.m1.gpu_in_cb_us         201.87
s4.draft.m1.d5_gate              gpu
s4.draft.batch.ane_uses_ne       yes (zero_copy=yes)
s4.draft.batch.ane_us            363.20 (with_handoff=937.84)
s4.draft.batch.gpu_in_cb_us      277.36
s4.draft.batch.d5_gate           gpu
s4.contention.ane_stream_mib     40.0 (uses_ane=yes, iters=77, ane_bw=50.00 GB/s)
s4.contention.gemv_q3            solo=70.91 GB/s  concurrent=49.09 GB/s  delta=30.8%
s4.contention.gemv_q4            solo=77.92 GB/s  concurrent=52.57 GB/s  delta=32.5%
s4.swapouts_delta                0
s4.pageouts_delta                0

Spike S5 Unique Active Experts vs Tree Size & CSR-Dedup vs Per-Slot MoE
s5.geometry                      q3, E=128, top_k=8, H=2048, I=768, pool=252.0 MiB
s5.beta_dedup_gbs                36.78
s5.crossover_m.correlated        2
s5.crossover_m.uniform           16
routing      M    |U(T)|  slots  saved%  dedup(us)  slot(us)  dedup_GB/s  slot_uGB/s  slot_rGB/s  speedup
correlated   1      8.0      8     0.0     258.24    237.32       63.95       69.59       69.59    0.92x
correlated   2      9.3     16    41.8     365.90    367.26       52.54       52.35       89.94    1.00x
correlated   4     11.2     32    65.0     512.37    695.38       45.08       33.21       95.00    1.36x
correlated   8     13.6     64    78.7     764.80   1360.09       36.78       20.68       97.14    1.78x
correlated  16     17.4    128    86.4    1475.00   2758.62       24.32       13.00       95.79    1.87x
correlated  32     23.0    256    91.0    2942.96   5431.57       16.13        8.74       97.30    1.85x
uniform      1      8.0      8     0.0     278.25    251.84       59.35       65.58       65.58    0.91x
uniform      2     15.4     16     3.5     481.17    434.38       66.23       73.37       76.04    0.90x
uniform      4     29.2     32     8.8     881.16    788.50       68.38       76.42       83.78    0.89x
uniform      8     52.5     64    18.0    1596.99   1408.13       67.87       76.97       93.83    0.88x
uniform     16     82.0    128    35.9    2518.45   2719.27       67.22       62.25       97.17    1.08x
uniform     32    112.6    256    56.0    4164.71   5618.64       55.83       41.38       94.06    1.35x
s5.swapouts_delta                0
s5.pageouts_delta                0

L1 Suffix Automaton Latency (D1, target < 50.00 us)
sam.context_tokens               32768
sam.arena_mib                    5.14
sam.append_us_per_tok            0.108
sam.query_active_us              0.041
sam.query_suffix_us              0.154
sam.max_query_us                 5.917
sam.queries                      10000
sam.seed_hits                    10000
sam.mean_draft_len               8.00
sam.malloc_blocks_delta          0
sam.malloc_bytes_delta           0

Synthetic End-to-End Speculative Decode Check (MoE q3 + EAGLE-3 + L1 + D7)
spec_synth.mode                  eagle3
spec_synth.placement             gpu
spec_synth.lossless_match        yes
spec_synth.generated_tokens      12
spec_synth.rounds                7
spec_synth.mean_accept_len       1.571
spec_synth.drafted_nodes         7.57
spec_synth.verified_nodes        1.57
spec_synth.unique_experts_layer  2.07
spec_synth.bytes_per_acc_tok     94022
spec_synth.malloc_blocks_delta   92
spec_synth.malloc_bytes_delta    5888
spec_synth.swapouts_delta        0

M3 Swap Check
m3.swap.status                   ok
m3.swap.swapouts_delta           0
m3.swap.pageouts_delta           0
```

### Verbatim real-checkpoint `verify-amf`, `decode`, and `spec-decode` raw stdout (commit `034492f`)

```text
=== Real EAGLE-3 Draft Checkpoint Verification ===
amf.status       ok
amf.model_id     qwen3-30b-a3b-eagle3
amf.tensors      15
amf.file_bytes   1044217856
amf.wired_bytes  1044185088
amf.cold_bytes   0

=== 1. Qwen3-0.6B BF16: BASELINE DECODE (T=0) ===
decode.arch                      qwen3 (bf16)
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             106
decode.generated_tokens          55
decode.prefill_ms                1179.288
decode.decode_ms_per_tok         14.450
decode.decode_tok_per_s          69.20
decode.bytes_per_tok             1210566656
decode.effective_gbs             83.78
decode.malloc_blocks_delta       108
decode.malloc_bytes_delta        6912
decode.swapouts_delta            0
decode.token_ids                 [4970, 55099, 341, 220, 877, 25, 914, 280, 220, 5934, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 48831, 4820, 914, 280, 220, 33051, 25, 914, 280, 630, 4970, 5787, 2648, 341, 220, 877, 25, 914, 280, 220, 72516, 25, 914, 280, 220, 96923, 25, 914, 280, 220, 33051, 25, 914, 280, 92, 151645]
decode.text                      interface UserProfile {
  id: string;
  username: string;
  email: string;
  updatedAt?: string;
  createdAt: string;
}

interface ProjectConfig {
  id: string;
  projectName: string;
  ownerId: string;
  createdAt: string;
}<|im_end|>

=== 1B. Qwen3-0.6B BF16: SPEC-DECODE L1 (T=0) ===
spec.arch                        qwen3 (bf16)
spec.mode                        l1
spec.placement                   gpu (requested=auto)
spec.pruning                     enabled (cap=8, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      28
spec.shards                      1
spec.wired_mib                   1217
spec.prompt_tokens               106
spec.generated_tokens            55
spec.rounds                      10
spec.mean_accept_len             5.600
spec.accept_hist                 [1:1, 2:2, 5:1, 6:1, 8:5]
spec.drafted_nodes_per_round     11.80
spec.verified_nodes_per_round    6.40
spec.l1_seed_rounds              6
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  107.225
spec.decode_ms_per_tok           3.054
spec.decode_tok_per_s            327.49
spec.draft_ms_per_round          0.000
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         16.166
spec.commit_sync_ms_per_round    0.320
spec.bytes_per_accepted_tok      215635968
spec.effective_gbs               73.23
spec.malloc_blocks_delta         41
spec.malloc_bytes_delta          2656
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4970, 55099, 341, 220, 877, 25, 914, 280, 220, 5934, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 48831, 4820, 914, 280, 220, 33051, 25, 914, 280, 630, 4970, 5787, 2648, 341, 220, 877, 25, 914, 280, 220, 72516, 25, 914, 280, 220, 96923, 25, 914, 280, 220, 33051, 25, 914, 280, 92, 151645]
spec.text                        interface UserProfile {
  id: string;
  username: string;
  email: string;
  updatedAt?: string;
  createdAt: string;
}

interface ProjectConfig {
  id: string;
  projectName: string;
  ownerId: string;
  createdAt: string;
}<|im_end|>

=== 1C. Qwen3-0.6B BF16: SPEC-DECODE L1 STOCHASTIC (T=0.7, seed=42) ===
spec.arch                        qwen3 (bf16)
spec.mode                        l1
spec.placement                   gpu (requested=auto)
spec.pruning                     enabled (cap=8, depth=4, width=2)
spec.temperature                 0.70
spec.layers                      28
spec.shards                      1
spec.wired_mib                   1217
spec.prompt_tokens               13
spec.generated_tokens            48
spec.rounds                      40
spec.mean_accept_len             1.175
spec.accept_hist                 [1:36, 2:1, 3:3]
spec.drafted_nodes_per_round     3.90
spec.verified_nodes_per_round    2.23
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  39.276
spec.decode_ms_per_tok           14.434
spec.decode_tok_per_s            69.28
spec.draft_ms_per_round          0.000
spec.prune_ms_per_round          0.000
spec.verify_ms_per_round         15.323
spec.commit_sync_ms_per_round    0.425
spec.bytes_per_accepted_tok      1018136663
spec.effective_gbs               70.54
spec.malloc_blocks_delta         162
spec.malloc_bytes_delta          10400
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [729, 15801, 1445, 11, 1932, 761, 579, 8, 11397, 1140, 320, 1607, 8, 1380, 308, 374, 279, 1372, 315, 5424, 311, 470, 11, 323, 1932, 761, 579, 374, 279, 7192, 897, 315, 279, 79683, 8500, 13, 1416, 1932, 761, 579, 374, 537, 3897, 11, 9658, 429, 1932, 761]
spec.text                         function fib(n, max_fib) → list (list) where n is the number of elements to return, and max_fib is the maximum value of the Fibonacci sequence. If max_fib is not provided, assume that max_f

=== 2A. Qwen3-14B-4bit: BASELINE DECODE (T=0) ===
decode.arch                      qwen3 (q4)
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             106
decode.generated_tokens          60
decode.prefill_ms                10198.800
decode.decode_ms_per_tok         97.708
decode.decode_tok_per_s          10.23
decode.bytes_per_tok             7897408320
decode.effective_gbs             80.83
decode.malloc_blocks_delta       120
decode.malloc_bytes_delta        8096
decode.swapouts_delta            0
decode.token_ids                 [4970, 55099, 341, 220, 877, 25, 914, 280, 220, 5934, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 4820, 914, 280, 630, 4970, 5787, 2648, 341, 220, 877, 25, 914, 280, 220, 72516, 25, 914, 280, 220, 96923, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 4820, 914, 280, 92, 151645]
decode.text                      interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: string;
  updatedAt?: string;
}

interface ProjectConfig {
  id: string;
  projectName: string;
  ownerId: string;
  createdAt: string;
  updatedAt?: string;
}<|im_end|>

=== 2B. Qwen3-14B-4bit: SPEC-DECODE L1 cap=8 (T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        l1
spec.placement                   gpu (requested=auto)
spec.pruning                     enabled (cap=8, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   8050
spec.prompt_tokens               106
spec.generated_tokens            60
spec.rounds                      14
spec.mean_accept_len             4.357
spec.accept_hist                 [1:3, 2:2, 3:1, 4:1, 5:3, 8:4]
spec.drafted_nodes_per_round     12.14
spec.verified_nodes_per_round    6.43
spec.l1_seed_rounds              8
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  3262.574
spec.decode_ms_per_tok           57.320
spec.decode_tok_per_s            17.45
spec.draft_ms_per_round          0.001
spec.prune_ms_per_round          0.002
spec.verify_ms_per_round         240.941
spec.commit_sync_ms_per_round    0.618
spec.bytes_per_accepted_tok      1811583465
spec.effective_gbs               32.68
spec.malloc_blocks_delta         58
spec.malloc_bytes_delta          4128
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4970, 55099, 341, 220, 877, 25, 914, 280, 220, 5934, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 4820, 914, 280, 630, 4970, 5787, 2648, 341, 220, 877, 25, 914, 280, 220, 72516, 25, 914, 280, 220, 96923, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 4820, 914, 280, 92, 151645]
spec.text                        interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: string;
  updatedAt?: string;
}

interface ProjectConfig {
  id: string;
  projectName: string;
  ownerId: string;
  createdAt: string;
  updatedAt?: string;
}<|im_end|>

=== 3A. Qwen3-30B-A3B-3bit: BASELINE DECODE (C++20 Reasoning, T=0) ===
decode.arch                      qwen3_moe (q3)
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             44
decode.generated_tokens          64
decode.prefill_ms                1039.859
decode.decode_ms_per_tok         21.741
decode.decode_tok_per_s          46.00
decode.bytes_per_tok             1341764480
decode.effective_gbs             61.72
decode.malloc_blocks_delta       126
decode.malloc_bytes_delta        8064
decode.swapouts_delta            0
decode.token_ids                 [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 13, 6771, 594, 1191, 553, 88646, 1128, 264, 10058, 4147, 374, 13, 1084, 594, 264, 26931, 1334, 1380, 279, 4147, 702, 264, 8356, 1379, 11, 323, 582, 990, 14937, 311, 3754, 279, 1968, 323, 9787, 13, 8704, 432, 594, 5296]
decode.text                      <think>
Okay, I need to implement a lock-free single-producer single-consumer ring buffer in C++20. Let's start by recalling what a ring buffer is. It's a circular array where the buffer has a fixed size, and we use indices to track the head and tail. Since it's lock

=== 3B. Qwen3-30B-A3B-3bit: EAGLE-3 + L1 + D7 PRUNED (C++20 Reasoning, T=0) ===
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=gpu)
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      48
spec.shards                      1
spec.wired_mib                   13876
spec.prompt_tokens               44
spec.generated_tokens            64
spec.rounds                      27
spec.mean_accept_len             2.333
spec.accept_hist                 [1:7, 2:6, 3:12, 4:2]
spec.drafted_nodes_per_round     13.44
spec.verified_nodes_per_round    2.93
spec.l1_seed_rounds              3
spec.unique_experts_per_layer    15.83
spec.prefill_ms                  597.451
spec.decode_ms_per_tok           20.671
spec.decode_tok_per_s            48.38
spec.draft_ms_per_round          5.111
spec.prune_ms_per_round          0.390
spec.verify_ms_per_round         37.148
spec.commit_sync_ms_per_round    5.554
spec.bytes_per_accepted_tok      906427004
spec.effective_gbs               43.85
spec.malloc_blocks_delta         358
spec.malloc_bytes_delta          22912
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 13, 6771, 594, 1191, 553, 88646, 1128, 264, 10058, 4147, 374, 13, 1084, 594, 264, 26931, 1334, 1380, 279, 4147, 702, 264, 8356, 1379, 11, 323, 582, 990, 14937, 311, 3754, 279, 1968, 323, 9787, 13, 8704, 432, 594, 5296]
spec.text                        <think>
Okay, I need to implement a lock-free single-producer single-consumer ring buffer in C++20. Let's start by recalling what a ring buffer is. It's a circular array where the buffer has a fixed size, and we use indices to track the head and tail. Since it's lock

=== 3C. Qwen3-30B-A3B-3bit: EAGLE-3 --no-prune (C++20 Reasoning, T=0) ===
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=gpu)
spec.pruning                     disabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      48
spec.shards                      1
spec.wired_mib                   13832
spec.prompt_tokens               44
spec.generated_tokens            64
spec.rounds                      20
spec.mean_accept_len             3.150
spec.accept_hist                 [1:4, 2:5, 3:6, 4:2, 5:2, 13:1]
spec.drafted_nodes_per_round     12.65
spec.verified_nodes_per_round    12.65
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    31.28
spec.prefill_ms                  580.399
spec.decode_ms_per_tok           40.107
spec.decode_tok_per_s            24.93
spec.draft_ms_per_round          6.960
spec.prune_ms_per_round          0.000
spec.verify_ms_per_round         109.120
spec.commit_sync_ms_per_round    10.191
spec.bytes_per_accepted_tok      1157991170
spec.effective_gbs               28.87
spec.malloc_blocks_delta         178
spec.malloc_bytes_delta          11392
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 13, 6771, 594, 1191, 553, 88646, 1128, 264, 10058, 4147, 374, 13, 1084, 594, 264, 26931, 1334, 1380, 279, 4147, 702, 264, 8356, 1379, 11, 323, 582, 990, 14937, 311, 3754, 279, 1968, 323, 9787, 13, 8704, 432, 594, 5296]
spec.text                        <think>
Okay, I need to implement a lock-free single-producer single-consumer ring buffer in C++20. Let's start by recalling what a ring buffer is. It's a circular array where the buffer has a fixed size, and we use indices to track the head and tail. Since it's lock

=== 4A. Qwen3-30B-A3B-3bit: BASELINE DECODE (TypeScript Code Edit, T=0) ===
decode.arch                      qwen3_moe (q3)
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             82
decode.generated_tokens          59
decode.prefill_ms                1754.316
decode.decode_ms_per_tok         19.939
decode.decode_tok_per_s          50.15
decode.bytes_per_tok             1345008512
decode.effective_gbs             67.46
decode.malloc_blocks_delta       116
decode.malloc_bytes_delta        7424
decode.swapouts_delta            0
decode.token_ids                 [4970, 2657, 341, 220, 877, 25, 914, 280, 220, 829, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 532, 4970, 7909, 341, 220, 877, 25, 914, 280, 220, 2265, 25, 914, 280, 220, 6372, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 92, 151645]
decode.text                      interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
  updatedAt: string;
}
interface Team {
  id: string;
  title: string;
  owner: string;
  createdAt: string;
  updatedAt: string;
}<|im_end|>

=== 4B. Qwen3-30B-A3B-3bit: L1-ONLY + D7 PRUNED (TypeScript Code Edit, T=0) ===
spec.arch                        qwen3_moe (q3)
spec.mode                        l1
spec.placement                   gpu (requested=auto)
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      48
spec.shards                      1
spec.wired_mib                   12839
spec.prompt_tokens               82
spec.generated_tokens            59
spec.rounds                      24
spec.mean_accept_len             2.500
spec.accept_hist                 [1:10, 2:3, 3:1, 4:9, 5:1]
spec.drafted_nodes_per_round     11.83
spec.verified_nodes_per_round    2.92
spec.l1_seed_rounds              17
spec.unique_experts_per_layer    15.82
spec.prefill_ms                  832.505
spec.decode_ms_per_tok           16.881
spec.decode_tok_per_s            59.24
spec.draft_ms_per_round          0.000
spec.prune_ms_per_round          0.484
spec.verify_ms_per_round         39.292
spec.commit_sync_ms_per_round    1.007
spec.bytes_per_accepted_tok      846832661
spec.effective_gbs               51.89
spec.malloc_blocks_delta         162
spec.malloc_bytes_delta          14528
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4970, 2657, 341, 220, 877, 25, 914, 280, 220, 829, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 532, 4970, 7909, 341, 220, 877, 25, 914, 280, 220, 2265, 25, 914, 280, 220, 6372, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 92, 151645]
spec.text                        interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
  updatedAt: string;
}
interface Team {
  id: string;
  title: string;
  owner: string;
  createdAt: string;
  updatedAt: string;
}<|im_end|>

=== 4C. Qwen3-30B-A3B-3bit: EAGLE-3 + L1 D6 SEED + D7 PRUNED (TypeScript Code Edit, T=0) ===
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=gpu)
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      48
spec.shards                      1
spec.wired_mib                   13876
spec.prompt_tokens               82
spec.generated_tokens            59
spec.rounds                      17
spec.mean_accept_len             3.412
spec.accept_hist                 [1:2, 2:1, 3:3, 4:10, 5:1]
spec.drafted_nodes_per_round     14.65
spec.verified_nodes_per_round    3.88
spec.l1_seed_rounds              14
spec.unique_experts_per_layer    19.79
spec.prefill_ms                  1042.391
spec.decode_ms_per_tok           29.572
spec.decode_tok_per_s            33.82
spec.draft_ms_per_round          1.223
spec.prune_ms_per_round          0.837
spec.verify_ms_per_round         90.498
spec.commit_sync_ms_per_round    8.306
spec.bytes_per_accepted_tok      735950738
spec.effective_gbs               24.89
spec.malloc_blocks_delta         154
spec.malloc_bytes_delta          9856
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4970, 2657, 341, 220, 877, 25, 914, 280, 220, 829, 25, 914, 280, 220, 2551, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 532, 4970, 7909, 341, 220, 877, 25, 914, 280, 220, 2265, 25, 914, 280, 220, 6372, 25, 914, 280, 220, 33051, 25, 914, 280, 220, 48831, 25, 914, 280, 92, 151645]
spec.text                        interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
  updatedAt: string;
}
interface Team {
  id: string;
  title: string;
  owner: string;
  createdAt: string;
  updatedAt: string;
}<|im_end|>
```
