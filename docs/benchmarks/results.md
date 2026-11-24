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
  15 tensors (`bf16` weights + `u32` `d2t` 64,000-entry reduced-vocab map),
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

## M4 Neural Engine, hybrid prefill, cold-expert NVMe streaming and D4 small draft

Measured on 2026-11-21 on the Apple M4 24 GiB baseline at commits `6b6fb09`
(`bench-m4`) and `8701fdb` (`bench-a2`, real-checkpoint cold-expert `decode` and
`D4` `spec-decode`), ReleaseFast `./zig-out/bin/aminfer`, default kernel
tunables (`kern.aiothreads = 4`, `kern.aioprocmax = 256`, host
`wired_budget_mib = 18186`, `swapouts_delta = 0` across every archived run).

Offline M4 artifacts produced by `tools/` (`W7`, `W8`, 40/40 Python unittests
passing) and stored under `/Volumes/Akari/models/`:
- **W8 expert heat profile (`qwen3-30b-a3b-heat.json`)**: `schema_version = 1`,
  `48 × 128` experts (`hot = 3,072`, `cold = 3,072` at `64 cold/layer`),
  `estimated_wired_bytes = 7,019,085,824 B` (`6,693.9 MiB`),
  `estimated_cold_bytes = 6,341,787,648 B` (`6,048.0 MiB`).
- **W7 pure-stdlib CoreML `.mlpackage` exports** (verified on macOS CoreML / ANE):
  `qwen3-30b-a3b-expert-predictor-all48.mlpackage` (`hidden_in[32,2048] -> router_logits[32,6144]`),
  `qwen3-30b-a3b-expert-predictor-la2.mlpackage`,
  `qwen3-30b-a3b-attn-qkv-l0.mlpackage`, and `qwen3-0.6b-draft-ffn-l0.mlpackage`.

### Spike S3: Cold-expert `F_NOCACHE` direct I/O (`kqueue_aio` vs `ecore_pread`, `C3`–`C4`)

Measured on a 192 MiB `F_NOCACHE` file on `/Volumes/Akari/.tmp` (external SSD
volume capped at `~0.64 GB/s` sustained throughput; internal Apple Silicon NVMe
reaches 3–7 GB/s). Swept real Qwen3-30B-A3B 3-bit (`2,064,384 B`) and 4-bit
(`2,654,208 B`) expert blobs across `QD = 1..16`, `1/2/4` E-core `pread`
workers, and `IOPOL_IMPORTANT` vs `IOPOL_STANDARD` (`64` timed random reads/case):

| Blob size | Backend | Workers | I/O Policy | Peak QD | Peak GB/s | QD=1 p50 (us) | Peak-QD p50 / p99 (us) | `swapouts` delta |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `2,064,384 B` (Q3) | **`kqueue_aio`** | 0 (kernel `4`) | `important` | 16 | **0.638** | 3,563.4 | 43,448.1 / 84,186.7 | 0 |
| `2,064,384 B` (Q3) | `kqueue_aio` | 0 (kernel `4`) | `standard` | 16 | 0.637 | 3,562.4 | 43,944.3 / 82,428.1 | 0 |
| `2,064,384 B` (Q3) | `ecore_pread` | 4 | `important` | 8 | 0.624 | 3,558.5 | 26,446.7 / 26,681.3 | 0 |
| `2,064,384 B` (Q3) | `ecore_pread` | 4 | `standard` | 4 | 0.622 | 3,592.2 | 13,249.1 / 13,552.1 | 0 |
| `2,654,208 B` (Q4) | **`kqueue_aio`** | 0 (kernel `4`) | `important` | 16 | **0.636** | 4,543.8 | 68,481.9 / 71,774.4 | 0 |
| `2,654,208 B` (Q4) | `ecore_pread` | 4 | `important` | 4 | 0.626 | 4,488.6 | 16,930.0 / 17,220.3 | 0 |

Takeaways (closing **Spike S3** and confirming **C3** + **C4**):
- Both `F_NOCACHE` backends (`kqueue_aio` via `SIGEV_KEVENT` → `EVFILT_AIO`
  auto-reap and `ecore_pread` via E-core `utility` + `IOPOL_IMPORTANT` workers)
  saturate the `/Volumes/Akari` external SSD link with **`swapouts_delta = 0`**
  and exact page-sentinel integrity verification.
- `kqueue_aio` edges out 4-worker `ecore_pread` on peak throughput (`0.638` vs
  `0.626 GB/s`, `+1.9%`), while `ecore_pread` exhibits tighter p99 tail latency
  at high queue depth (`QD = 16`: `53.7 ms` vs `84.2 ms`) because `kern.aiothreads = 4`
  queues bursts inside the kernel AIO pool. Both backends remain first-class
  selectable via `--io-backend {auto,aio,pread}`.

### Spike S6 & `A1`/`A3`/`G6`: Static KV buckets and hybrid ANE/GPU wavefront prefill

Measured on `Qwen3-30B-A3B` attention geometry (`hidden = 2048, q_heads = 32,
kv_heads = 4, head_dim = 128, chunk = 512`) plus synthetic end-to-end `A1`/`G6`
`Qwen3Model.prefillHybridWavefront` vs GPU-only `prefillChunk`:

| Static KV bucket (`kv_len`) | Compile (ms) | ANE layers | Zero-copy | `bucketed_full_ane` (ms) | `split_ane_qkv_gpu_attn` (ms) | `gpu_only` (ms) | ANE tok/s | GPU tok/s | Fastest |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 524.0 | 25 / 25 | `yes` | **5.130** | 8.232 | 11.392 | **99,805** | 44,946 | `bucketed_full_ane` (`2.22×`) |
| 1024 | 391.5 | 27 / 27 | `yes` | **8.002** | 12.684 | 16.165 | **63,981** | 31,673 | `bucketed_full_ane` (`2.02×`) |
| 2048 | 565.0 | 27 / 27 | `yes` | **13.589** | 21.482 | 24.834 | **37,677** | 20,617 | `bucketed_full_ane` (`1.83×`) |
| 4096 | 431.7 | 27 / 27 | `yes` | **24.716** | 37.713 | 41.021 | **20,715** | 12,481 | `bucketed_full_ane` (`1.66×`) |

FP16 numerical precision vs CPU FP64 reference at `kv_len = 1024` (`32` sampled
query rows, exercising `k_past`/`v_past` concat and causal masking):
- `attn_max_abs_err = 0.002404`, `attn_rmse = 0.000515`
- `kv_max_abs_err = 0.000968`, `kv_rmse = 0.000212`

Wavefront overlap (`A1` + `G6`, zero-copy IOSurface `KvTile`s, `USER_INITIATED` ANE driver thread):

| Workload | Prompt / chunks | Hybrid wavefront (ms / tok/s) | GPU-only (ms / tok/s) | Overlap efficiency | Speedup vs GPU-only | ANE / Zero-copy | `swapouts` delta |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Spike S6 (`30B-A3B` attn + MoE FFN, 6L) | 2 × 512 | `289.818 ms/chunk` (**1,767 tok/s**) | `344.682 ms/chunk` (`1,485 tok/s`) | **1.198×** (`11/12` steps) | **1.189×** | `yes` / `yes` | 0 |
| E2E `Qwen3Model` (`2L, H=512, E=16, q3`) | 1024 (2 × 512) | `14.892 ms` (**68,761.2 tok/s**) | `27.831 ms` (`36,792.9 tok/s`) | **1.35×** | **1.87×** | `yes` / `yes` | 0 |
| E2E `Qwen3Model` (`2L, H=512, E=16, q3`) | 2048 (4 × 512) | `32.819 ms` (**62,401.9 tok/s**) | `57.603 ms` (`35,553.5 tok/s`) | **1.59×** | **1.76×** | `yes` / `yes` | 0 |

Takeaways (closing **Spike S6** and settling **A3**):
- **`bucketed_full_ane`** (`B ∈ {512, 1024, 2048, 4096}`) places 100% of graph
  ops on the 16-core ANE (`25/25`–`27/27`), preserves zero-copy IOSurface
  output backings (`zero_copy = yes`), stays within `0.0024` max abs error vs
  FP64, and beats GPU-only chunked attention by **`1.66–2.22×`** across every
  bucket length.
- Overlapping ANE attention `A(l, c_next)` with Metal 4 GPU FFN/MoE `F(l, c_cur)`
  via `WavefrontPrefill` (`G6` + `A1`) achieves **`1.189×`** on `Qwen3-30B-A3B`
  6-layer geometry and **`1.76–1.87×`** on end-to-end 1024/2048-token `Qwen3Model`
  prefill with `swapouts_delta = 0`.

### Real-checkpoint `A2` ANE cross-layer MoE expert predictor (`Qwen3-30B-A3B-3bit`)

Evaluated on real 48-layer `/Volumes/Akari/models/amf/qwen3-30b-a3b-3bit.amf`
across `16` real decode steps (`752` layer transitions, `128` experts, `top_k = 8`,
`hidden = 2048`), comparing:
1. **In-memory folded ANE predictor** (`M = 1`, `post_attention_layernorm`
   scales folded into router gate weights, 6 overlapped ANE blocks, compiled in
   `214.6 ms`),
2. **GPU router lookahead baseline** (`post_attention_layernorm` + `q3` router
   GEMV on Metal 4), and
3. **Offline W7 `.mlpackage`** (`qwen3-30b-a3b-expert-predictor-all48.mlpackage`,
   `M = 32` static batch, raw unfolded router weights, loaded in `225.0 ms`).

Measured real 48-layer GPU layer time: **`401.42 us/layer`**.
Steady-state predictor latency: **in-memory ANE `156.44 us`** (`uses_ne = yes`,
`zero_copy = yes`, **`17.85×` faster** than GPU 3-layer lookahead `2,792.99 us`),
offline W7 `.mlpackage` `516.95 us` (`uses_ne = yes`, `zero_copy = yes`),
`swapouts_delta = 0`:

| Lookahead `delta` | Available lead time (`delta × 401.42 - 156.44 us`) | In-memory folded ANE `R@8 / R@12 / R@16` | GPU lookahead baseline `R@8 / R@12 / R@16` | Offline W7 `.mlpackage` (unfolded) `R@8 / R@12 / R@16` |
| ---: | ---: | --- | --- | --- |
| `1` layer | `245.0 us` | **0.861 / 0.958 / 0.982** | 0.860 / 0.958 / 0.982 | 0.705 / 0.826 / 0.882 |
| `2` layers | **`646.4 us`** | **0.809 / 0.927 / 0.960** | 0.809 / 0.927 / 0.960 | 0.684 / 0.810 / 0.869 |
| `3` layers | **`1,047.8 us`** | **0.774 / 0.891 / 0.933** | 0.774 / 0.890 / 0.933 | 0.664 / 0.789 / 0.851 |

Key finding:
- Folding each layer's `post_attention_layernorm` channel scales into the linear
  ANE router bank raises real 48-layer `Recall@8` by **`+15.6 pp`** (`70.5% → 86.1%`)
  and `Recall@12` by **`+13.2 pp`** (`82.6% → 95.8%`), matching exact GPU
  RMSNorm + router lookahead to three decimal places (`0.861/0.958/0.982` vs
  `0.860/0.958/0.982`) while executing asynchronously on ANE in `156.44 us` and
  yielding **`646–1,048 us`** of cross-layer NVMe prefetch lead time at `delta = 2..3`.

### Real-checkpoint `Qwen3-30B-A3B-3bit` cold-expert NVMe streaming decode (`C3`–`C5` + `A2` + `W8`)

All runs execute greedy `T = 0` decode (`--max-tokens 32 --ignore-eos`) on real
48-layer `qwen3-30b-a3b-3bit.amf` with the W8 heat profile
`/Volumes/Akari/models/amf/qwen3-30b-a3b-heat.json`, marking `64` cold experts
per layer (`3,072` cold experts = `6.05 GiB` kept on disk) under an explicit
`--wired-cap-mib 8192` wired RAM cap (`5,922 MiB` / `-46.3%` lower wired RAM at
`64` slots; `5,040 MiB` / `-39.4%` lower at `512` slots). Every cold configuration
matched the all-hot `12.8 GiB` baseline on **32 / 32 `token_ids` (100% bit-for-bit)**
with **`swapouts_delta = 0`**:

| Configuration | Wired RAM (MiB) | `decode tok/s` | `ms/tok` | Hot / Cache / Prefetch / Sync-miss | Prefetch hit rate | Total cold hit rate | Disk bytes read | Unused prefetched | Exact `token_ids` | `swapouts` delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| **All-hot baseline** (`0` cold) | 12,791 | **48.47** | 20.633 | all hot | — | — | 0 B | 0 | baseline (32/32) | 0 |
| `64 cold/L`, `64 slots`, `pread` + ANE | **6,869** (`-46.3%`) | 0.91 | 1,095.947 | `5992 / 681 / 2884 / 2347` | **48.8%** | 60.3% | 19.28 GB | 4,097 | **32 / 32 (100%)** | 0 |
| `64 cold/L`, `64 slots`, `aio` + ANE | **6,869** (`-46.3%`) | 0.94 | 1,058.703 | `5992 / 663 / 2885 / 2364` | **48.8%** | 60.0% | 19.45 GB | 4,174 | **32 / 32 (100%)** | 0 |
| `64 cold/L`, `64 slots`, `aio`, no ANE | 6,869 (`-46.3%`) | 1.18 | 844.948 | `5992 / 0 / 0 / 5912` | 0.0% | 0.0% | 12.20 GB | 0 | **32 / 32 (100%)** | 0 |
| `64 cold/L`, `512 slots`, `aio`, no ANE | 7,751 (`-39.4%`) | 2.95 | 338.785 | `5992 / 4033 / 0 / 1879` | 0.0% | 68.2% | 3.88 GB | 0 | **32 / 32 (100%)** | 0 |
| **`64 cold/L`, `512 slots`, `aio` + ANE** | **7,751** (`-39.4%`) | **3.02** | **331.076** | `5992 / 3779 / 1257 / 876` | **21.3%** | **85.2%** | 5.48 GB | 526 | **32 / 32 (100%)** | 0 |

Key engineering findings from the 2×2 cold-expert ablation:
1. **Budget reduction & lossless bit-parity**: Halving wired experts reduces
   wired DRAM from `12,791 MiB` down to `6,869–7,751 MiB`, cleanly passing an
   `8,192 MiB` wired cap with `swapouts_delta = 0` and 100% bit-for-bit exact
   decode output.
2. **Cache sizing vs speculative prefetch bandwidth**: Because `/Volumes/Akari`
   is an external SSD capped at `~0.64 GB/s`, a tiny 64-slot cache (`~1.3` slots
   per layer across 48 layers) thrashes under aggressive 12-candidate lookahead
   prefetching (`4,174` unused prefetches = `8.6 GB` extra reads), making
   on-demand `aio` slightly faster (`1.18` vs `0.94 tok/s`). Expanding the slot
   pool to **`512` slots** (`7,751 MiB`, still `< 8,192 MiB`) eliminates slot
   thrashing: LFU + ANE prefetch raises total cold hit rate to **`85.2%`**
   (`+17.0 pp` over LFU-only `68.2%`), cuts synchronous disk misses by
   **`-53.4%`** (`1,879 → 876`), reduces disk reads by `3.55×` (`19.45 → 5.48 GB`),
   and beats both 64-slot modes and 512-slot no-ANE (**`3.02 tok/s`** even on
   the `0.64 GB/s` external SSD).

### Real-checkpoint `D4` same-family small draft (`Qwen3-14B-4bit` + `Qwen3-0.6B` BF16)

All runs execute greedy `T = 0` (`--max-tokens 64 --ignore-eos`) on real 40-layer
`qwen3-14b-4bit.amf` (`8,006 MiB`) paired with standalone 28-layer
`qwen3-0.6b-bf16.amf` (combined wired `9,274–9,347 MiB`). Startup `D5`
calibration selects `.gpu` (`gate_gpu_us = 52.4–145.9 us` vs `gate_ane_us = 482.4–990.9 us`).
Every run matched baseline `aminfer decode` on **64 / 64 `token_ids` (100% bit-for-bit)**
with **`swapouts_delta = 0`**:

| Prompt & workload | Mode / Tree config | `decode tok/s` | Speedup vs base | Mean `tau` | Drafted → Verified nodes | L1 seed rounds | `draft / verify / sync` (ms/rnd) | Bytes / acc tok (vs base) | Exact `token_ids` | `swapouts` delta |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| **Python LRU cache** (20 → 64 tok) | Baseline `decode` | 10.82 | 1.00× | 1.000 | — | — | — | 7,884.0 MB | baseline | 0 |
| **Python LRU cache** (20 → 64 tok) | **`D4 small` (`cap=8, d=5, w=1, k=2`)** | **15.13** | **1.40×** | **3.938** | 10.13 → 6.94 | 2 / 16 | `40.6 / 203.0 / 16.6` | **2,001.2 MB (`-74.6%`)** | **64 / 64 (100%)** | 0 |
| **B-tree + binary search** (24 → 64 tok) | Baseline `decode` | 10.88 | 1.00× | 1.000 | — | — | — | 7,884.6 MB | baseline | 0 |
| **B-tree + binary search** (24 → 64 tok) | `D4 small` default (`cap=16, d=4, w=2, k=3`) | **12.48** | **1.15×** | **3.706** | 17.76 → 8.71 | 1 / 17 | `34.5 / 246.2 / 16.2` | 2,126.4 MB (`-73.0%`) | **64 / 64 (100%)** | 0 |
| **B-tree + binary search** (24 → 64 tok) | **`D4 small` tuned (`cap=8, d=5, w=1, k=2`)** | **13.81** | **1.27×** | **3.500** | 10.89 → 6.44 | 1 / 18 | `41.6 / 195.3 / 16.6` | 2,251.2 MB (`-71.4%`) | **64 / 64 (100%)** | 0 |
| **Rust `Pin<P>` reasoning** (21 → 64 tok, **0 L1**) | Baseline `decode` | 10.83 | 1.00× | 1.000 | — | — | — | 7,884.1 MB | baseline | 0 |
| **Rust `Pin<P>` reasoning** (21 → 64 tok, **0 L1**) | `D4 small` chain (`cap=6, d=5, w=1, k=1`) | 10.55 | 0.97× | 1.909 | 4.21 → 4.21 | **0 / 33** | `24.5 / 140.3 / 16.1` | 4,127.4 MB (`-47.6%`) | **64 / 64 (100%)** | 0 |
| **Rust `Pin<P>` reasoning** (21 → 64 tok, **0 L1**) | `D4 small` tree (`cap=10, d=5, w=2, k=2`) | 9.55 | 0.88× | **2.100** | 9.57 → 6.10 | **0 / 30** | `19.0 / 185.0 / 15.9` | 3,752.2 MB (`-52.4%`) | **64 / 64 (100%)** | 0 |

Takeaways:
- `D4` (`Qwen3-0.6B` BF16) delivers **`tau = 3.50–3.94`**, **`-71%` to `-75%`
  DRAM bytes per accepted token**, and **`13.81–15.13 tok/s` (`1.27–1.40×` over
  the `10.8 tok/s` baseline; `1.35–1.48×` over M2.5's `10.2 tok/s`)** on general
  coding/reasoning prompts where L1 alone triggers on only 1–2 rounds.
- On zero-L1-seed Rust prose (`l1_seed_rounds = 0`), EAGLE-2 tree branching
  raises pure-`D4` acceptance from `tau = 1.909` to `2.100`, while exposing the
  remaining wall-clock bottleneck: unquantized BF16 drafting (`~35–42 ms/round`)
  plus bit-exact `encodeGemvBatch` verification (`~185–203 ms/round`, ~`2.0×` of
  `M = 1` `92 ms`), motivating quantized 4-bit small drafts and pipelined `D8`
  drafting in M5.

### Verbatim standalone `aminfer bench-m4` raw stdout (commit `6b6fb09`)

```text
=== M4 Hybrid ANE/GPU Wavefront, Cold-Expert NVMe Streaming, A2 & D4 Benchmarks ===

spike_s3.file_mib                192
spike_s3.kern_aiothreads         4
spike_s3.kern_aioprocmax         256
spike_s3.peak_aio_gbs            0.638
spike_s3.peak_pread_gbs          0.626
spike_s3.recommended_backend     kqueue_aio
spike_s3.recommended_workers     0
spike_s3.recommended_io_policy   important
spike_s3.swapouts_delta          0
spike_s3.pageouts_delta          0

Spike S3 NVMe Direct-I/O Sweep (F_NOCACHE)
blob_B     backend      wk  iopol      qd   reads   GB/s     p50(us)   p99(us)   cpu(us)
2064384    kqueue_aio   0   important  1    64      0.575    3563.4    4722.0    6904
2064384    kqueue_aio   0   important  2    64      0.607    6790.2    7032.5    6454
2064384    kqueue_aio   0   important  4    64      0.622    13266.6   13462.1   8022
2064384    kqueue_aio   0   important  8    64      0.631    26170.9   26346.7   8249
2064384    kqueue_aio   0   important  16   64      0.638    43448.1   84186.7   7735
2064384    ecore_pread  1   important  1    64      0.580    3544.9    3803.2    6257
2064384    ecore_pread  1   important  2    64      0.579    7109.4    7585.3    6266
2064384    ecore_pread  1   important  4    64      0.578    14251.8   14669.7   6273
2064384    ecore_pread  1   important  8    64      0.572    28876.6   29289.7   7481
2064384    ecore_pread  1   important  16   64      0.569    58035.7   58595.1   8292
2064384    ecore_pread  2   important  1    64      0.575    3577.4    3919.8    6563
2064384    ecore_pread  2   important  2    64      0.608    6786.5    6904.1    6215
2064384    ecore_pread  2   important  4    64      0.604    13626.3   13950.3   6825
2064384    ecore_pread  2   important  8    64      0.606    27224.5   27527.0   7393
2064384    ecore_pread  2   important  16   64      0.608    54150.0   54699.4   7046
2064384    ecore_pread  4   important  1    64      0.578    3558.5    3743.3    6837
2064384    ecore_pread  4   important  2    64      0.607    6804.5    6963.7    7147
2064384    ecore_pread  4   important  4    64      0.622    13265.8   13524.8   8222
2064384    ecore_pread  4   important  8    64      0.624    26446.7   26681.3   8711
2064384    ecore_pread  4   important  16   64      0.620    53013.5   53733.2   8820
2064384    kqueue_aio   0   standard   1    64      0.574    3562.4    6073.6    7973
2064384    kqueue_aio   0   standard   2    64      0.610    6762.5    6848.8    6369
2064384    kqueue_aio   0   standard   4    64      0.622    13244.3   13550.0   7774
2064384    kqueue_aio   0   standard   8    64      0.630    26233.5   29331.3   8260
2064384    kqueue_aio   0   standard   16   64      0.637    43944.3   82428.1   8066
2064384    ecore_pread  1   standard   1    64      0.578    3564.6    3829.1    6550
2064384    ecore_pread  1   standard   2    64      0.580    7083.5    7488.8    6694
2064384    ecore_pread  1   standard   4    64      0.581    14173.9   14641.7   6172
2064384    ecore_pread  1   standard   8    64      0.576    28521.0   29302.8   6775
2064384    ecore_pread  1   standard   16   64      0.582    56714.7   57013.8   6272
2064384    ecore_pread  2   standard   1    64      0.581    3545.9    3732.7    6105
2064384    ecore_pread  2   standard   2    64      0.609    6772.3    7015.2    6231
2064384    ecore_pread  2   standard   4    64      0.608    13555.2   13928.5   6908
2064384    ecore_pread  2   standard   8    64      0.607    27156.0   27384.2   6998
2064384    ecore_pread  2   standard   16   64      0.609    54090.5   54624.8   6422
2064384    ecore_pread  4   standard   1    64      0.572    3592.2    3898.7    6733
2064384    ecore_pread  4   standard   2    64      0.608    6780.4    6952.1    6941
2064384    ecore_pread  4   standard   4    64      0.622    13249.1   13552.1   7726
2064384    ecore_pread  4   standard   8    64      0.620    26669.2   26971.5   8360
2064384    ecore_pread  4   standard   16   64      0.618    53378.4   53755.5   9816
2654208    kqueue_aio   0   important  1    64      0.574    4543.8    8684.6    9835
2654208    kqueue_aio   0   important  2    64      0.608    8713.7    8935.0    10278
2654208    kqueue_aio   0   important  4    64      0.624    17001.5   17329.4   11667
2654208    kqueue_aio   0   important  8    64      0.629    33669.9   34444.7   10764
2654208    kqueue_aio   0   important  16   64      0.636    68481.9   71774.4   9256
2654208    ecore_pread  1   important  1    64      0.590    4481.7    4821.2    7181
2654208    ecore_pread  1   important  2    64      0.589    8984.1    9320.4    7690
2654208    ecore_pread  1   important  4    64      0.590    17952.4   18237.9   8134
2654208    ecore_pread  1   important  8    64      0.589    36046.1   36482.6   8085
2654208    ecore_pread  1   important  16   64      0.588    71916.5   73371.2   8403
2654208    ecore_pread  2   important  1    64      0.590    4480.1    4752.8    7436
2654208    ecore_pread  2   important  2    64      0.613    8644.5    8898.1    8628
2654208    ecore_pread  2   important  4    64      0.614    17275.3   17482.0   9137
2654208    ecore_pread  2   important  8    64      0.612    34623.3   35112.4   8859
2654208    ecore_pread  2   important  16   64      0.611    69371.2   69911.1   10068
2654208    ecore_pread  4   important  1    64      0.590    4488.6    4752.3    7237
2654208    ecore_pread  4   important  2    64      0.613    8640.4    8994.5    8795
2654208    ecore_pread  4   important  4    64      0.626    16930.0   17220.3   9572
2654208    ecore_pread  4   important  8    64      0.625    33934.6   34449.3   11312
2654208    ecore_pread  4   important  16   64      0.625    67742.9   68686.7   10168
2654208    kqueue_aio   0   standard   1    64      0.590    4491.5    4786.3    7491
2654208    kqueue_aio   0   standard   2    64      0.612    8640.1    13091.6   8515
2654208    kqueue_aio   0   standard   4    64      0.626    16974.3   17121.6   9767
2654208    kqueue_aio   0   standard   8    64      0.630    33714.2   34165.2   12582
2654208    kqueue_aio   0   standard   16   64      0.636    67514.3   97771.9   8706
2654208    ecore_pread  1   standard   1    64      0.584    4529.4    4826.3    8729
2654208    ecore_pread  1   standard   2    64      0.584    9085.5    9423.4    9186
2654208    ecore_pread  1   standard   4    64      0.590    17990.7   18186.2   8875
2654208    ecore_pread  1   standard   8    64      0.591    35772.5   36530.5   7504
2654208    ecore_pread  1   standard   16   64      0.586    72207.0   73878.0   8565
2654208    ecore_pread  2   standard   1    64      0.588    4496.6    4865.8    7916
2654208    ecore_pread  2   standard   2    64      0.612    8671.3    8838.0    8822
2654208    ecore_pread  2   standard   4    64      0.613    17313.7   17515.4   9542
2654208    ecore_pread  2   standard   8    64      0.611    34719.0   35054.6   10518
2654208    ecore_pread  2   standard   16   64      0.612    69324.3   69964.1   9429
2654208    ecore_pread  4   standard   1    64      0.589    4488.2    4755.2    7871
2654208    ecore_pread  4   standard   2    64      0.613    8648.4    8881.0    8392
2654208    ecore_pread  4   standard   4    64      0.624    16987.1   17451.5   10625
2654208    ecore_pread  4   standard   8    64      0.624    33996.8   34372.5   10298
2654208    ecore_pread  4   standard   16   64      0.625    67901.1   68393.4   11117

Spike S6 Hybrid ANE/GPU Wavefront Prefill & Static KV Bucket Study (A1, A3, ADR-0010)
s6.ane_cores                     16
s6.geometry                      hidden=2048 q_heads=32 kv_heads=4 head_dim=128 chunk=512
s6.qkv_only.compile_ms           254.4 (ane=11/11, uses_ane=yes, zero_copy=yes)

Static KV Bucket Comparison (per 1 layer, 512-token chunk)
kv_len  compile(ms)  ane_layers  zc   ane_bucket(ms)  split_qkv+gpu(ms)  gpu_only(ms)  ane_tok/s  gpu_tok/s  fastest
   512        524.0    25/25     yes           5.130              8.232        11.392      99805      44946  bucketed_full_ane
  1024        391.5    27/27     yes           8.002             12.684        16.165      63981      31673  bucketed_full_ane
  2048        565.0    27/27     yes          13.589             21.482        24.834      37677      20617  bucketed_full_ane
  4096        431.7    27/27     yes          24.716             37.713        41.021      20715      12481  bucketed_full_ane

FP16 Numerical Precision vs CPU FP64 Reference (kv_len=1024, sampled_rows=32)
s6.precision.attn_max_abs_err    0.002404
s6.precision.attn_rmse           0.000515
s6.precision.kv_max_abs_err      0.000968
s6.precision.kv_rmse             0.000212

2-Chunk Hybrid Wavefront Overlap (A1: 2 chunks x 6 layers, chunk=512 tokens)
s6.wf.ane_thread_qos             user_initiated
s6.wf.overlapped_steps           11/12
s6.wf.zero_copy_backing          yes
s6.wf.t_ane_layer_chunk_ms       10.311
s6.wf.t_gpu_ffn_layer_chunk_ms   47.562
s6.wf.t_gpu_attn_layer_chunk_ms  13.778
s6.wf.sequential_ms_per_chunk    347.236 (1475 tok/s)
s6.wf.wavefront_ms_per_chunk     289.818 (1767 tok/s)
s6.wf.gpu_only_ms_per_chunk      344.682 (1485 tok/s)
s6.wf.overlap_efficiency         1.198x
s6.wf.speedup_vs_sequential      1.198x
s6.wf.speedup_vs_gpu_only        1.189x
s6.a3.recommended_strategy       bucketed_full_ane (max_bucket=4096)
s6.swapouts_delta                0
s6.pageouts_delta                0

A2 ANE Cross-Layer MoE Expert Predictor Evaluation
a2.ane_cores                     16
a2.shape                         layers=16 experts=128 top_k=8 hidden=2048 lookahead=2 M=1
a2.blocks_compiled               2 (compile_ms=227.2, uses_ane=yes, zero_copy=yes)
a2.ane_thread_qos                user_initiated
a2.ane_predict_us                169.25
a2.cpu_baseline_us               364.42
a2.lead_time_us                  750.75 (gpu_layer=460.0us x 2, nvme_target=400.0us, hides_nvme=yes)
a2.ane_recall                    R@8=0.871  R@12=0.983  R@16=0.996
a2.baseline_recall               R@8=0.871  R@12=0.983  R@16=0.996
a2.unused_prefetch_ratio_k12     0.345
a2.swapouts_delta                0
a2.pageouts_delta                0

Synthetic End-to-End A1/G6 Hybrid Wavefront Prefill vs GPU-Only Prefill
hybrid_prefill.geometry          layers=2 hidden=512 experts=16 top_k=4 chunk=512
prompt_toks   hybrid(ms)   hybrid(tok/s)   gpu_only(ms)   gpu(tok/s)   speedup   overlap   ane   zc
       1024       14.892         68761.2         27.831      36792.9     1.87x     1.35x   yes   yes
       2048       32.819         62401.9         57.603      35553.5     1.76x     1.59x   yes   yes
hybrid_prefill.swapouts_delta    0

Synthetic End-to-End Cold-Expert NVMe Streaming Decode (C3-C5 + A2)
cold_synth.geometry              layers=4 experts=16 cold_per_layer=8 slots=16
cold_synth.io_backend            ecore_pread (ane_predictor=on, uses_ne=yes)
cold_synth.wired_bytes           all_hot=7536640 cold_mode=5963776
cold_synth.bit_exact_vs_hot      yes (gen_tokens=12)
cold_synth.hits                  hot=83 cache=77 prefetch=3 sync_miss=13
cold_synth.hit_rates             prefetch=0.032 cold_total=0.860 bytes_read=1867776 evictions=18
cold_synth.malloc_blocks_delta   143
cold_synth.malloc_bytes_delta    9152
cold_synth.swapouts_delta        0

Synthetic End-to-End D4 Same-Family Small Draft Lossless Check
d4_synth.placement               gpu (gate_ane_us=428.3, gate_gpu_us=38.6)
d4_synth.lossless_match          yes
d4_synth.generated_tokens        12
d4_synth.rounds                  4
d4_synth.mean_accept_len         2.750
d4_synth.drafted_nodes           6.25
d4_synth.verified_nodes          2.75
d4_synth.malloc_blocks_delta     32
d4_synth.malloc_bytes_delta      2048
d4_synth.swapouts_delta          0

M4 Swap Check
m4.swap.status                   ok
m4.swap.swapouts_delta           0
m4.swap.pageouts_delta           0
```

### Verbatim real-checkpoint `bench-a2`, cold-expert NVMe `decode`, and `D4` `spec-decode` raw stdout (commit `8701fdb`)

```text
=== 1. Real Qwen3-30B-A3B-3bit A2 ANE Predictor Evaluation (bench-a2) ===
Real-Checkpoint A2 ANE Cross-Layer MoE Expert Predictor Evaluation
real_a2.shape                    layers=48 experts=128 top_k=8 hidden=2048 steps=16
real_a2.gpu_layer_time_us        401.42
real_a2.inmem_ane                compile_ms=214.6 latency_us=156.44 uses_ne=yes zero_copy=yes
real_a2.gpu_lookahead_us         2792.99 (for 3 lookahead layers)
real_a2.offline_mlpackage        compile_ms=225.0 latency_us=516.95 uses_ne=yes zero_copy=yes

delta  lead_us   inmem_ANE(R@8/12/16)       gpu_base(R@8/12/16)        w7_mlpkg(R@8/12/16)
    1    245.0   0.861 / 0.958 / 0.982      0.860 / 0.958 / 0.982      0.705 / 0.826 / 0.882
    2    646.4   0.809 / 0.927 / 0.960      0.809 / 0.927 / 0.960      0.684 / 0.810 / 0.869
    3   1047.8   0.774 / 0.891 / 0.933      0.774 / 0.890 / 0.933      0.664 / 0.789 / 0.851
real_a2.swapouts_delta           0
real_a2.pageouts_delta           27

=== 2A. Qwen3-30B-A3B-3bit All-Hot Baseline (12.8 GiB wired) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                742.274
decode.decode_ms_per_tok         20.633
decode.decode_tok_per_s          48.47
decode.bytes_per_tok             1336652672
decode.effective_gbs             64.78
decode.malloc_blocks_delta       62
decode.malloc_bytes_delta        3968
decode.swapouts_delta            0
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 2B. Qwen3-30B-A3B-3bit Cold-Expert NVMe Streaming (64 cold/L, 64 slots, 8192 MiB cap, pread + ANE) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 6869
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                27695.749
decode.decode_ms_per_tok         1095.947
decode.decode_tok_per_s          0.91
decode.bytes_per_tok             1336652672
decode.effective_gbs             1.22
decode.malloc_blocks_delta       -1
decode.malloc_bytes_delta        -96
decode.swapouts_delta            0
decode.cold_experts              3072
decode.cold_slots                64
decode.cold_io_backend           ecore_pread
decode.cold_ane_predictor        on (uses_ne=yes, compile_ms=198.4)
decode.cold_hot_hits             5992
decode.cold_cache_hits           681
decode.cold_prefetch_hits        2884
decode.cold_sync_misses          2347
decode.cold_prefetch_hit_rate    0.488
decode.cold_hit_rate             0.603
decode.cold_bytes_read           19275153408
decode.cold_evictions            9336
decode.cold_unused_prefetched    4097
decode.cold_prefetch_wait_us     7850881.4
decode.cold_sync_miss_wait_us    22275016.0
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 2C. Qwen3-30B-A3B-3bit Cold-Expert NVMe Streaming (64 cold/L, 64 slots, 8192 MiB cap, aio + ANE) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 6869
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                26370.115
decode.decode_ms_per_tok         1058.703
decode.decode_tok_per_s          0.94
decode.bytes_per_tok             1336652672
decode.effective_gbs             1.26
decode.malloc_blocks_delta       4500
decode.malloc_bytes_delta        308160
decode.swapouts_delta            0
decode.cold_experts              3072
decode.cold_slots                64
decode.cold_io_backend           kqueue_aio
decode.cold_ane_predictor        on (uses_ne=yes, compile_ms=184.1)
decode.cold_hot_hits             5992
decode.cold_cache_hits           663
decode.cold_prefetch_hits        2885
decode.cold_sync_misses          2364
decode.cold_prefetch_hit_rate    0.488
decode.cold_hit_rate             0.600
decode.cold_bytes_read           19447185408
decode.cold_evictions            9430
decode.cold_unused_prefetched    4174
decode.cold_prefetch_wait_us     9255550.3
decode.cold_sync_miss_wait_us    19661013.2
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 2D. Qwen3-30B-A3B-3bit Cold-Expert Ablation (64 cold/L, 64 slots, 8192 MiB cap, aio, NO ANE) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 6869
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                21481.619
decode.decode_ms_per_tok         844.948
decode.decode_tok_per_s          1.18
decode.bytes_per_tok             1336652672
decode.effective_gbs             1.58
decode.malloc_blocks_delta       2279
decode.malloc_bytes_delta        145856
decode.swapouts_delta            0
decode.cold_experts              3072
decode.cold_slots                64
decode.cold_io_backend           kqueue_aio
decode.cold_ane_predictor        off (uses_ne=no, compile_ms=0.0)
decode.cold_hot_hits             5992
decode.cold_cache_hits           0
decode.cold_prefetch_hits        0
decode.cold_sync_misses          5912
decode.cold_prefetch_hit_rate    0.000
decode.cold_hit_rate             0.000
decode.cold_bytes_read           12204638208
decode.cold_evictions            5912
decode.cold_unused_prefetched    0
decode.cold_prefetch_wait_us     0.0
decode.cold_sync_miss_wait_us    22474541.4
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 2E. Qwen3-30B-A3B-3bit Cold-Expert 512-Slot Cache (64 cold/L, 512 slots, 8192 MiB cap, aio + ANE) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 7751
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                9606.303
decode.decode_ms_per_tok         331.076
decode.decode_tok_per_s          3.02
decode.bytes_per_tok             1336652672
decode.effective_gbs             4.04
decode.malloc_blocks_delta       4494
decode.malloc_bytes_delta        287488
decode.swapouts_delta            0
decode.cold_experts              3072
decode.cold_slots                512
decode.cold_io_backend           kqueue_aio
decode.cold_ane_predictor        on (uses_ne=yes, compile_ms=189.7)
decode.cold_hot_hits             5992
decode.cold_cache_hits           3779
decode.cold_prefetch_hits        1257
decode.cold_sync_misses          876
decode.cold_prefetch_hit_rate    0.213
decode.cold_hit_rate             0.852
decode.cold_bytes_read           5480939520
decode.cold_evictions            2661
decode.cold_unused_prefetched    526
decode.cold_prefetch_wait_us     1816611.2
decode.cold_sync_miss_wait_us    5306989.7
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 2F. Qwen3-30B-A3B-3bit Cold-Expert 512-Slot Ablation (64 cold/L, 512 slots, 8192 MiB cap, aio, NO ANE) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.layers                    48
decode.shards                    1
decode.wired_mib                 7751
decode.prompt_tokens             24
decode.generated_tokens          32
decode.prefill_ms                10262.628
decode.decode_ms_per_tok         338.785
decode.decode_tok_per_s          2.95
decode.bytes_per_tok             1336652672
decode.effective_gbs             3.95
decode.malloc_blocks_delta       2411
decode.malloc_bytes_delta        154304
decode.swapouts_delta            0
decode.cold_experts              3072
decode.cold_slots                512
decode.cold_io_backend           kqueue_aio
decode.cold_ane_predictor        off (uses_ne=no, compile_ms=0.0)
decode.cold_hot_hits             5992
decode.cold_cache_hits           4033
decode.cold_prefetch_hits        0
decode.cold_sync_misses          1879
decode.cold_prefetch_hit_rate    0.000
decode.cold_hit_rate             0.682
decode.cold_bytes_read           3878977536
decode.cold_evictions            1879
decode.cold_unused_prefetched    0
decode.cold_prefetch_wait_us     0.0
decode.cold_sync_miss_wait_us    7099237.8
decode.token_ids                 [220, 16, 13, 81917, 279, 1882, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 13, 220, 17, 13, 31075, 264, 13027, 7868, 2711, 729, 382, 32313, 11, 358, 1184]
decode.text                       1. Explain the process of how B-trees maintain balance during insertion and deletion. 2. Implement a Python binary search function.

Okay, I need

=== 3A. Qwen3-14B-4bit Baseline Decode (Python LRU cache, T=0, 64 tok) ===
decode.arch                      qwen3 (q4)
decode.prefill_mode              auto
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             20
decode.generated_tokens          64
decode.prefill_ms                1910.918
decode.decode_ms_per_tok         92.446
decode.decode_tok_per_s          10.82
decode.bytes_per_tok             7883973440
decode.effective_gbs             85.28
decode.malloc_blocks_delta       128
decode.malloc_bytes_delta        8608
decode.swapouts_delta            0
decode.token_ids                 [4710, 32313, 11, 358, 1184, 311, 4211, 264, 4516, 55119, 444, 36003, 6500, 304, 13027, 448, 633, 323, 2182, 7525, 304, 506, 7, 16, 8, 882, 13, 88190, 11, 1077, 594, 1744, 911, 1246, 311, 5486, 419, 382, 5338, 11, 358, 6099, 429, 458, 444, 36003, 6500, 11136, 5711, 264, 10601, 315, 264, 5175, 2415, 320, 1958, 506, 7, 16, 8, 1401, 8602, 8]
decode.text                       

Okay, I need to implement a thread-safe LRU cache in Python with get and put operations in O(1) time. Hmm, let's think about how to approach this.

First, I remember that an LRU cache typically uses a combination of a hash map (for O(1) lookups)

=== 3B. Qwen3-14B-4bit + Qwen3-0.6B BF16 D4 Small Draft (Python LRU cache, cap=8 d=5 w=1 k=2, T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       990.9
spec.placement_gate_gpu_us       145.9
spec.pruning                     enabled (cap=8, depth=5, width=1)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   9324
spec.prompt_tokens               20
spec.generated_tokens            64
spec.rounds                      16
spec.mean_accept_len             3.938
spec.accept_hist                 [1:2, 2:2, 3:4, 4:2, 5:1, 6:4, 8:1]
spec.drafted_nodes_per_round     10.13
spec.verified_nodes_per_round    6.94
spec.l1_seed_rounds              2
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  712.409
spec.decode_ms_per_tok           66.080
spec.decode_tok_per_s            15.13
spec.draft_ms_per_round          40.600
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         203.027
spec.commit_sync_ms_per_round    16.561
spec.bytes_per_accepted_tok      2001224853
spec.effective_gbs               30.28
spec.malloc_blocks_delta         190
spec.malloc_bytes_delta          14464
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4710, 32313, 11, 358, 1184, 311, 4211, 264, 4516, 55119, 444, 36003, 6500, 304, 13027, 448, 633, 323, 2182, 7525, 304, 506, 7, 16, 8, 882, 13, 88190, 11, 1077, 594, 1744, 911, 1246, 311, 5486, 419, 382, 5338, 11, 358, 6099, 429, 458, 444, 36003, 6500, 11136, 5711, 264, 10601, 315, 264, 5175, 2415, 320, 1958, 506, 7, 16, 8, 1401, 8602, 8]
spec.text                         

Okay, I need to implement a thread-safe LRU cache in Python with get and put operations in O(1) time. Hmm, let's think about how to approach this.

First, I remember that an LRU cache typically uses a combination of a hash map (for O(1) lookups)

=== 3C. Qwen3-14B-4bit Baseline Decode (B-tree + binary search, T=0, 64 tok) ===
decode.arch                      qwen3 (q4)
decode.prefill_mode              auto
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             24
decode.generated_tokens          64
decode.prefill_ms                2264.177
decode.decode_ms_per_tok         91.939
decode.decode_tok_per_s          10.88
decode.bytes_per_tok             7884628800
decode.effective_gbs             85.76
decode.malloc_blocks_delta       126
decode.malloc_bytes_delta        8256
decode.swapouts_delta            0
decode.token_ids                 [4710, 750, 7868, 10716, 10939, 11, 2169, 982, 262, 671, 697, 2038, 1588, 271, 32313, 11, 1077, 594, 21403, 419, 3491, 13, 576, 1196, 6801, 1378, 2513, 25, 1156, 11, 458, 16148, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 11, 323, 2086, 11, 264, 13027, 8129, 315, 264, 7868, 2711, 729, 382, 24617, 448, 279, 425, 40438, 16148, 13, 358]
decode.text                       

def binary_search(arr, target):
    # your code here

Okay, let's tackle this problem. The user wants two things: first, an explanation of how B-trees maintain balance during insertion and deletion, and second, a Python implementation of a binary search function.

Starting with the B-tree explanation. I

=== 3D. Qwen3-14B-4bit + Qwen3-0.6B BF16 D4 Small Draft (B-tree + binary search, cap=16 d=4 w=2 k=3, T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       631.9
spec.placement_gate_gpu_us       69.5
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   9347
spec.prompt_tokens               24
spec.generated_tokens            64
spec.rounds                      17
spec.mean_accept_len             3.706
spec.accept_hist                 [1:1, 2:5, 3:2, 4:3, 5:5, 9:1]
spec.drafted_nodes_per_round     17.76
spec.verified_nodes_per_round    8.71
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  749.683
spec.decode_ms_per_tok           80.141
spec.decode_tok_per_s            12.48
spec.draft_ms_per_round          34.508
spec.prune_ms_per_round          0.003
spec.verify_ms_per_round         246.239
spec.commit_sync_ms_per_round    16.243
spec.bytes_per_accepted_tok      2126420947
spec.effective_gbs               26.53
spec.malloc_blocks_delta         59
spec.malloc_bytes_delta          2048
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4710, 750, 7868, 10716, 10939, 11, 2169, 982, 262, 671, 697, 2038, 1588, 271, 32313, 11, 1077, 594, 21403, 419, 3491, 13, 576, 1196, 6801, 1378, 2513, 25, 1156, 11, 458, 16148, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 11, 323, 2086, 11, 264, 13027, 8129, 315, 264, 7868, 2711, 729, 382, 24617, 448, 279, 425, 40438, 16148, 13, 358]
spec.text                         

def binary_search(arr, target):
    # your code here

Okay, let's tackle this problem. The user wants two things: first, an explanation of how B-trees maintain balance during insertion and deletion, and second, a Python implementation of a binary search function.

Starting with the B-tree explanation. I

=== 3E. Qwen3-14B-4bit + Qwen3-0.6B BF16 D4 Small Draft (B-tree + binary search, cap=8 d=5 w=1 k=2, T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       549.1
spec.placement_gate_gpu_us       88.2
spec.pruning                     enabled (cap=8, depth=5, width=1)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   9324
spec.prompt_tokens               24
spec.generated_tokens            64
spec.rounds                      18
spec.mean_accept_len             3.500
spec.accept_hist                 [1:4, 2:3, 3:3, 4:1, 5:4, 6:2, 8:1]
spec.drafted_nodes_per_round     10.89
spec.verified_nodes_per_round    6.44
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  752.305
spec.decode_ms_per_tok           72.413
spec.decode_tok_per_s            13.81
spec.draft_ms_per_round          41.611
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         195.259
spec.commit_sync_ms_per_round    16.574
spec.bytes_per_accepted_tok      2251241996
spec.effective_gbs               31.09
spec.malloc_blocks_delta         67
spec.malloc_bytes_delta          2480
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [4710, 750, 7868, 10716, 10939, 11, 2169, 982, 262, 671, 697, 2038, 1588, 271, 32313, 11, 1077, 594, 21403, 419, 3491, 13, 576, 1196, 6801, 1378, 2513, 25, 1156, 11, 458, 16148, 315, 1246, 425, 2385, 7858, 10306, 8172, 2337, 35927, 323, 36066, 11, 323, 2086, 11, 264, 13027, 8129, 315, 264, 7868, 2711, 729, 382, 24617, 448, 279, 425, 40438, 16148, 13, 358]
spec.text                         

def binary_search(arr, target):
    # your code here

Okay, let's tackle this problem. The user wants two things: first, an explanation of how B-trees maintain balance during insertion and deletion, and second, a Python implementation of a binary search function.

Starting with the B-tree explanation. I

=== 3F. Qwen3-14B-4bit Baseline Decode (Rust Pin<P>, T=0, 64 tok) ===
decode.arch                      qwen3 (q4)
decode.prefill_mode              auto
decode.layers                    40
decode.shards                    1
decode.wired_mib                 8006
decode.prompt_tokens             21
decode.generated_tokens          64
decode.prefill_ms                1999.143
decode.decode_ms_per_tok         92.327
decode.decode_tok_per_s          10.83
decode.bytes_per_tok             7884137280
decode.effective_gbs             85.39
decode.malloc_blocks_delta       128
decode.malloc_bytes_delta        8608
decode.swapouts_delta            0
decode.token_ids                 [279, 2874, 374, 429, 279, 3853, 1231, 387, 7726, 311, 264, 2155, 4938, 3728, 2337, 11320, 11, 323, 279, 5306, 27454, 315, 279, 3853, 1231, 3635, 8318, 13, 17471, 25351, 429, 279, 3853, 374, 537, 7726, 11, 773, 279, 5306, 27454, 7146, 2697, 13, 4354, 11, 979, 1667, 3312, 14, 11421, 11, 279, 19415, 9463, 28872, 8983, 1229, 369, 601, 13, 2055, 11, 979]
decode.text                       the reason is that the future may be moved to a different memory location during execution, and the internal pointers of the future may become invalid. Pin ensures that the future is not moved, so the internal pointers remain valid. However, when using async/await, the compiler automatically manages pinning for us. So, when

=== 3G. Qwen3-14B-4bit + Qwen3-0.6B BF16 D4 Small Draft (Rust Pin<P>, 0 L1 seeds, cap=6 d=5 w=1 k=1 no-prune, T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       482.4
spec.placement_gate_gpu_us       80.3
spec.pruning                     disabled (cap=6, depth=5, width=1)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   9274
spec.prompt_tokens               21
spec.generated_tokens            64
spec.rounds                      33
spec.mean_accept_len             1.909
spec.accept_hist                 [1:17, 2:7, 3:4, 4:5]
spec.drafted_nodes_per_round     4.21
spec.verified_nodes_per_round    4.21
spec.l1_seed_rounds              0
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  800.963
spec.decode_ms_per_tok           94.767
spec.decode_tok_per_s            10.55
spec.draft_ms_per_round          24.516
spec.prune_ms_per_round          0.000
spec.verify_ms_per_round         140.278
spec.commit_sync_ms_per_round    16.125
spec.bytes_per_accepted_tok      4127432264
spec.effective_gbs               43.55
spec.malloc_blocks_delta         148
spec.malloc_bytes_delta          7744
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [279, 2874, 374, 429, 279, 3853, 1231, 387, 7726, 311, 264, 2155, 4938, 3728, 2337, 11320, 11, 323, 279, 5306, 27454, 315, 279, 3853, 1231, 3635, 8318, 13, 17471, 25351, 429, 279, 3853, 374, 537, 7726, 11, 773, 279, 5306, 27454, 7146, 2697, 13, 4354, 11, 979, 1667, 3312, 14, 11421, 11, 279, 19415, 9463, 28872, 8983, 1229, 369, 601, 13, 2055, 11, 979]
spec.text                         the reason is that the future may be moved to a different memory location during execution, and the internal pointers of the future may become invalid. Pin ensures that the future is not moved, so the internal pointers remain valid. However, when using async/await, the compiler automatically manages pinning for us. So, when

=== 3H. Qwen3-14B-4bit + Qwen3-0.6B BF16 D4 Small Draft (Rust Pin<P>, 0 L1 seeds, cap=10 d=5 w=2 k=2, T=0) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       769.5
spec.placement_gate_gpu_us       52.4
spec.pruning                     enabled (cap=10, depth=5, width=2)
spec.temperature                 0.00
spec.layers                      40
spec.shards                      1
spec.wired_mib                   9324
spec.prompt_tokens               21
spec.generated_tokens            64
spec.rounds                      30
spec.mean_accept_len             2.100
spec.accept_hist                 [1:12, 2:5, 3:11, 4:2]
spec.drafted_nodes_per_round     9.57
spec.verified_nodes_per_round    6.10
spec.l1_seed_rounds              0
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  714.155
spec.decode_ms_per_tok           104.748
spec.decode_tok_per_s            9.55
spec.draft_ms_per_round          19.031
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         184.994
spec.commit_sync_ms_per_round    15.943
spec.bytes_per_accepted_tok      3752202154
spec.effective_gbs               35.82
spec.malloc_blocks_delta         249
spec.malloc_bytes_delta          18368
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [279, 2874, 374, 429, 279, 3853, 1231, 387, 7726, 311, 264, 2155, 4938, 3728, 2337, 11320, 11, 323, 279, 5306, 27454, 315, 279, 3853, 1231, 3635, 8318, 13, 17471, 25351, 429, 279, 3853, 374, 537, 7726, 11, 773, 279, 5306, 27454, 7146, 2697, 13, 4354, 11, 979, 1667, 3312, 14, 11421, 11, 279, 19415, 9463, 28872, 8983, 1229, 369, 601, 13, 2055, 11, 979]
spec.text                         the reason is that the future may be moved to a different memory location during execution, and the internal pointers of the future may become invalid. Pin ensures that the future is not moved, so the internal pointers remain valid. However, when using async/await, the compiler automatically manages pinning for us. So, when
```


## M4.5 bare-metal ARMv9 SME2 engine, Spike S7, C13 bus contention and D11 TRNG

Measured on 2026-11-21 on the Apple M4 24 GiB baseline at commit `b23f787`
(ReleaseFast `./zig-out/bin/aminfer probe`, `bench-m45`, `verify-amf`, `decode`,
and `spec-decode`, default kernel tunables, `wired_budget_mib = 18186`,
`swapouts_delta = 0` and `prefill_swapouts_delta = 0` across all archived runs).

### Binary ABI, M1–M3 backward compatibility, and `otool -tvV` zero-spill audit (`C11`, `C12`)

| Check | Command / scope | Measured result |
| --- | --- | --- |
| Deployment floor | `otool -l ./zig-out/bin/aminfer` (`LC_BUILD_VERSION`) | `minos 26.0`, `sdk 27.0` |
| M1–M3 zero-leak check | `zig build -Doptimize=ReleaseFast -Dcpu=apple_m1` + `otool -tvV` | `34` `_ami_sme2_*` symbols; **`0`** SME/SVE instructions outside `_ami_sme2_*` |
| Pre-`smstart` callee-save ABI | Native ReleaseFast `otool -tvV` across all `30` `_ami_sme2_*` entry kernels | `smstart = 1`, `smstop = 1`, `pre_saves = 10` (`x19–x30` + `d8–d15` saved before `smstart`), `post_restores = 10` (restored after `smstop`) |
| Inner K-loop register spill audit | Native ReleaseFast `otool -tvV` between `smstart` and `smstop` | **`0`** stack-pointer (`[sp`) loads/stores across all `34` `_ami_sme2_*` symbols |

### Spike S7: 3-way ARMv9.2 `.s` SME2 microkernels vs Apple `Accelerate` (`cblas_sgemm`)

Measured via `./zig-out/bin/aminfer bench-m45` (`host.sme2=true`, `svl_bytes=64`,
`pcore_clusters=1`, `cpus_per_l2=4`, `pcore_l2_kib=16384`, `swapouts_delta = 0`).
`vs_acc` compares SME2 against pre-dequantized FP32 `cblas_sgemm`; `vs_dq+ac`
compares SME2 against scalar dequantization + `cblas_sgemm` for quantized types.

| Scenario | Dtype | Hint | `M×N×K` | `T` | `sme2_us` | Eff. `GB/s` | `GFLOP/s` | `acc_us` | `vs_acc` | `vs_dq+ac` |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `l2_router` | `bf16` | `ld1` | `1×128×2048` | 1 | 4.33 | 122.06 | 121.00 | 3.08 | 0.71× | 0.71× |
| `l2_router` | `bf16` | `ld1` | `16×128×2048` | 1 | **11.58** | 51.63 | 724.22 | 24.33 | **2.10×** | **2.10×** |
| `l2_router` | `q4` | `ld1` | `1×128×2048` | 1 | 4.17 | 40.42 | 125.82 | 3.67 | 0.88× | **18.92×** |
| `l2_router` | `q4` | `ld1` | `16×128×2048` | 1 | **16.63** | 14.29 | 504.58 | 22.42 | **1.35×** | **5.93×** |
| `l2_proj` | `bf16` | `ld1` | `8×1024×2048` | 4 | **49.58** | 85.91 | 676.73 | 198.29 | **4.00×** | **4.00×** |
| `l2_proj` | `bf16` | `ld1` | `16×1024×2048` | 4 | **46.13** | 93.78 | **1,454.93** | 175.33 | **3.80×** | **3.80×** |
| `l2_proj` | `bf16` | `ld1` | `32×1024×2048` | 4 | **90.75** | 49.11 | 1,478.98 | 183.88 | **2.03×** | **2.03×** |
| `l2_proj` | `bf16` | `ld1` | `128×1024×2048` | 4 | 345.71 | 15.17 | **1,552.96** | 446.17 | **1.29×** | **1.29×** |
| `l2_proj` | `f16` | `ld1` | `16×1024×2048` | 4 | **48.00** | 90.11 | 1,398.10 | 182.21 | **3.80×** | **3.80×** |
| `l2_proj` | `f16` | `ld1` | `512×1024×2048` | 4 | 1,355.92 | 6.19 | **1,583.79** | 1,436.13 | **1.06×** | **1.06×** |
| `l2_proj` | `q4` | `ld1` | `1×1024×2048` | 4 | 31.25 | 42.21 | 134.22 | 31.63 | 1.01× | **18.78×** |
| `l2_proj` | `q4` | `ld1` | `16×1024×2048` | 4 | **80.92** | 17.82 | 829.35 | 170.67 | **2.11×** | **9.31×** |
| `l2_proj` | `q4` | `ld1` | `32×1024×2048` | 4 | **87.96** | 17.88 | **1,525.93** | 172.92 | **1.97×** | **8.38×** |
| `l2_proj` | `q4` | `ld1` | `128×1024×2048` | 4 | 342.33 | 6.89 | **1,568.27** | 437.25 | **1.28×** | **2.98×** |
| `l2_proj` | `q3` | `ld1` | `1×1024×2048` | 4 | 42.92 | 24.62 | 97.73 | 31.17 | 0.73× | **14.06×** |
| `l2_proj` | `q3` | `ld1` | `128×1024×2048` | 4 | 347.00 | 6.04 | **1,547.18** | 443.21 | **1.28×** | **2.86×** |
| `l2_proj` | `q2` | `ld1` | `1×1024×2048` | 4 | 28.71 | 27.68 | 146.10 | 24.08 | 0.84× | **20.43×** |
| `l2_proj` | `q2` | `ld1` | `128×1024×2048` | 4 | 364.17 | 5.04 | **1,474.24** | 426.21 | **1.17×** | **2.72×** |
| `dram_slc_bust` | `bf16` | `ld1` | `1×4096×4096` | 4 | 522.17 | 64.31 | 64.26 | 1,016.96 | **1.95×** | **1.95×** |
| `dram_slc_bust` | `bf16` | `ldnt` | `1×4096×4096` | 4 | **473.67** | **70.89** | 70.84 | 1,036.92 | **2.19×** | **2.19×** |
| `dram_slc_bust` | `bf16` | `ld1` | `16×4096×4096` | 4 | **488.92** | 69.43 | 1,098.08 | 2,188.63 | **4.48×** | **4.48×** |
| `dram_slc_bust` | `bf16` | `ld1` | `128×4096×4096` | 4 | 2,588.83 | 14.18 | **1,659.04** | 3,762.33 | **1.45×** | **1.45×** |
| `dram_slc_bust` | `q4` | `ld1` | `1×4096×4096` | 4 | **218.38** | 48.13 | 153.66 | 1,003.46 | **4.60×** | **26.12×** |
| `dram_slc_bust` | `q4` | `ldnt` | `1×4096×4096` | 4 | 219.54 | 47.87 | 152.84 | 997.83 | **4.55×** | **25.41×** |
| `dram_slc_bust` | `q3` | `ldnt` | `1×4096×4096` | 4 | 313.04 | 26.88 | 107.19 | 986.33 | **3.15×** | **17.81×** |
| `dram_slc_bust` | `q2` | `ldnt` | `1×4096×4096` | 4 | **208.50** | 30.29 | 160.93 | 995.42 | **4.77×** | **26.56×** |

P-cluster thread scaling sweep (`1 / 2 / 4` threads on M4's single 4-core P-cluster):
- **L2-hot compute** (`bf16 16×2048×2048`, `bf16 128×2048×2048`, `q4 1×2048×2048`): 1 thread already saturates `95–100%` of M4's shared SME2 matrix unit (`1.00–1.05×` at 2–4T, peaking at **`1,716.73 GFLOP/s`** at 2T `M=128`, dropping to `0.91×` at 4T), empirically confirming [ADR-0018](../decisions/0018-bare-metal-armv9-sme2-engine.md)'s 1-worker-per-P-cluster rule for compute-bound workloads.
- **DRAM-bound `q4` `1×4096×4096`**: scales `1.00× → 1.17× → 1.24×` (`54.49 → 63.73 → 67.74 GB/s`) across `1 / 2 / 4` threads as additional P-cores expose more DRAM memory-level parallelism.

### `C13` concurrent Metal 4 GPU + CPU SME2 bus gate, `ldnt1b` L2 preservation, and router calibration

| Scenario | SME2 workload | GPU solo (`GB/s`) | GPU concurrent (`GB/s`) | GPU delta | SME2 solo (`us`) | SME2 conc (`us`) | Aggregate (`GB/s`) | Net speedup | Gate (`> 1.05×`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gpu+sme2_l2_router` | `bf16 16×1024×2048` (L2-hot) | 78.04 | **79.62** | **+2.0%** | 44.4 | 56.9 | **155.57** | **1.80×** | **PASS** |
| `gpu+sme2_dram_ld1` | `q4 1×4096×4096` (DRAM `ld1b`) | 78.04 | 68.29 | -12.5% | 212.1 | 470.1 | 90.65 | **1.33×** | **PASS** |
| `gpu+sme2_dram_ldnt1` | `q4 1×4096×4096` (DRAM `ldnt1b`) | 78.04 | **71.64** | **-8.2%** | 198.0 | 480.9 | **93.49** | **1.33×** | **PASS** |

- **`ldnt1b` cold-expert L2 cache preservation (`4 MiB` hot router + `16.88 MiB` cold `q4` SwiGLU stream)**:
  - Hot router baseline: **`44.17 us`**.
  - After sweeping `6` cold `q4` experts with temporal **`ld1b`**: hot router degrades **`+40.9%`** to **`62.25 us`** (`cold = 86.31 us/expert`).
  - After sweeping `6` cold `q4` experts with non-temporal **`ldnt1b`**: hot router holds at **`44.08 us` (`-0.2%`, zero L2 eviction)** and cold experts also run faster at **`81.51 us/expert` (`-5.6%`)**.
- **L2-resident SME2 vs Metal 4 GPU speculative tree router prediction (`8×128×2048`, `4 MiB` bank)**:
  - `M = 1`: GPU `573.46 us` vs SME2 **`32.38 us`** (**`17.71×`**, `100.00%` top-k agreement).
  - `M = 8`: GPU `857.92 us` vs SME2 **`74.75 us`** (**`11.48×`**, `100.00%` top-k agreement).
  - `M = 16`: GPU `759.17 us` vs SME2 **`106.13 us`** (**`7.15×`**, `99.90%` top-k agreement).

### Real-checkpoint M4.5 verification (`W15`, `C13`, `D11`)

1. **100% `.amf` v0 backward compatibility (`W15`)**: All four existing real checkpoints (`qwen3-0.6b-bf16.amf`, `qwen3-14b-4bit.amf`, `qwen3-30b-a3b-3bit.amf`, `qwen3-30b-a3b-eagle3.amf`) pass `aminfer verify-amf` unchanged. Repacking real `qwen3-0.6b-bf16.amf` with `aminfer-tools repack-sme2 --targets all-linear` converts `196` linear tensors to `sme2_panel64` in `5.50 s` at the exact same file size (`1,193,885,696 B`, `1,195 MiB` wired, zero RAM duplication, `swapouts_delta = 0`). Full-model `M = 1` decode on CPU SME2 runs at `21.94 tok/s` (`26.26 GB/s`, `malloc_blocks_delta = 1`) vs Metal 4 GPU `68.67 tok/s` (`82.19 GB/s`) — matching the first 10 tokens verbatim before a 1-ULP `BFMOPA`/`BFDOT` vs GPU FP32-FMA reduction tie-break at token 11 across 196 BF16 projections — empirically confirming `C13`'s rule to keep full-model dense `M = 1` decode on GPU while steering L2-resident routers and cold-expert digestion to SME2.
2. **Real `Qwen3-30B-A3B-3bit` + `EAGLE-3` GPU-router vs SME2 L2-router A/B (`C13`, C++20 SPSC ring buffer, `34 → 64` tokens, `T = 0`, `swapouts_delta = 0`)**:

| Mode | Router backend | Wired (`MiB`) | `prefill_ms` | `prune_ms/round` | `decode tok/s` | Speedup vs base | `malloc_blocks_delta` | Exact `token_ids` | `swapouts` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Baseline `decode` | — | 12,791 | 1,040.5 | — | 49.73 | 1.000× | 126 | baseline | 0 |
| `EAGLE-3 + L1 + D7` (`--sme-router gpu`) | `gpu` | 13,876 | 287.9 | 0.396 | 52.49 | 1.055× | 320 | **64 / 64 (100%)** | 0 |
| **`EAGLE-3 + L1 + D7` (`--sme-router auto`)** | **`sme2`** (`78.0` vs `975.5 us`, `12.51×`) | 13,880 (`+4`) | **266.3** | **0.217 (`-45.2%`)** | **52.97** | **1.065×** | **194 (`-39.4%`)** | **64 / 64 (100%)** | 0 |

3. **Real `Qwen3-0.6B` BF16 `D11` Secure Enclave TRNG + bit-exact explicit replay (`T = 0.7`, `32` tokens, `swapouts_delta = 0`)**:

| CLI mode | Run | `seed_source` | `seed_used` | `decode tok/s` | Verification |
| --- | --- | --- | ---: | ---: | --- |
| `decode` | Run 1 (no `--seed`) | `hardware_trng` | `15577856275329892283` | 61.85 | distinct 64-bit SEP TRNG seed |
| `decode` | Run 2 (no `--seed`) | `hardware_trng` | `8891266525397946930` | 64.79 | distinct seed & diverse text vs Run 1 |
| `decode` | Replay (`--seed <Run 1>`) | `explicit` | `15577856275329892283` | 66.23 | **32 / 32 (100%) bit-exact `token_ids` vs Run 1** |
| `spec-decode` | Run 1 (no `--seed`) | `hardware_trng` | `4723189805597218849` | 62.64 | distinct 64-bit SEP TRNG seed |
| `spec-decode` | Run 2 (no `--seed`) | `hardware_trng` | `3910625554483478406` | 65.49 | distinct seed & diverse text vs Run 1 |
| `spec-decode` | Replay (`--seed <Run 1>`) | `explicit` | `4723189805597218849` | 66.18 | **32 / 32 (100%) bit-exact `token_ids` vs Run 1** |

### Verbatim binary audit, `aminfer probe`, and `aminfer bench-m45` raw stdout (commit `b23f787`)

```text
=== 1. M1 BINARY COMPATIBILITY AUDIT (-Dcpu=apple_m1) ===
m1_build.ami_sme2_symbols = 34
m1_build.non_sme2_sme_instruction_leaks = 0
=== 2. NATIVE RELEASEFAST BUILD & OTOOL SPILL AUDIT ===
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 26.0
      sdk 27.0
native_build.ami_sme2_symbols = 34
  _ami_sme2_drain_1x4: insns=75 smstart=0 smstop=0 pre_saves=0 post_restores=0 streaming_sp=0
  _ami_sme2_drain_2x2: insns=90 smstart=0 smstop=0 pre_saves=0 post_restores=0 streaming_sp=0
  _ami_sme2_epilogue_store_vec: insns=65 smstart=0 smstop=0 pre_saves=0 post_restores=0 streaming_sp=0
  _ami_sme2_m1_4_bf16_nt: insns=180 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_bf16_t: insns=180 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_f16_nt: insns=180 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_f16_t: insns=180 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q2_nt: insns=392 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q2_t: insns=392 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q3_nt: insns=292 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q3_t: insns=292 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q4_nt: insns=400 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m1_4_q4_t: insns=400 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_bf16_nt: insns=100 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_bf16_t: insns=100 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_f16_nt: insns=100 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_f16_t: insns=100 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q2_nt: insns=529 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q2_t: insns=532 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q3_nt: insns=524 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q3_t: insns=524 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q4_nt: insns=200 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m32_q4_t: insns=200 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_bf16_nt: insns=108 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_bf16_t: insns=108 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_f16_nt: insns=108 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_f16_t: insns=108 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q2_nt: insns=652 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q2_t: insns=652 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q3_nt: insns=672 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q3_t: insns=672 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q4_nt: insns=248 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_m5_16_q4_t: insns=248 smstart=1 smstop=1 pre_saves=10 post_restores=10 streaming_sp=0
  _ami_sme2_store_m1_4_panel: insns=49 smstart=0 smstop=0 pre_saves=0 post_restores=0 streaming_sp=0
native_build.total_streaming_sp_accesses = 0

=== 3. AMINFER PROBE ===
sys.page_size                    16384
sys.memory_mib                   24576
cpu.performance_cores            4
cpu.efficiency_cores             6
cpu.pcore_clusters               1
cpu.pcore_l2_kib                 16384
cpu.sme2                         true
cpu.sme_bf16f32                  true
cpu.sme_f16f32                   true
cpu.sme_svl_bytes                64
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

=== 4. AMINFER BENCH-M45 ===

=== M4.5 Bare-Metal ARMv9 SME2 Engine & C13 Concurrent Bus Suite ===

=== Spike S7 Standalone: ARMv9 SME2 vs Apple Accelerate (cblas_sgemm) ===
host.sme2=true sme_b16f32=true sme_f16f32=true svl_bytes=64 pcore_clusters=1 cpus_per_l2=4 pcore_l2_kib=16384 swapouts_delta=0

scenario         dtype hint    M     N     K  T |   sme2_us     GB/s   GFLOP/s |    acc_us    acc_GF |  vs_acc vs_dq+ac
l2_router        bf16  ld1     1   128  2048  1 |      4.33   122.06    121.00 |      3.08    170.06 |   0.71x    0.71x
l2_router        bf16  ld1    16   128  2048  1 |     11.58    51.63    724.22 |     24.33    344.74 |   2.10x    2.10x
l2_router        q4    ld1     1   128  2048  1 |      4.17    40.42    125.82 |      3.67    143.01 |   0.88x   18.92x
l2_router        q4    ld1    16   128  2048  1 |     16.63    14.29    504.58 |     22.42    374.21 |   1.35x    5.93x
l2_proj          bf16  ld1     1  1024  2048  4 |     25.21   166.71    166.38 |     30.75    136.40 |   1.22x    1.22x
l2_proj          bf16  ld1     4  1024  2048  4 |     92.13    45.88    182.11 |    180.42     92.99 |   1.96x    1.96x
l2_proj          bf16  ld1     8  1024  2048  4 |     49.58    85.91    676.73 |    198.29    169.22 |   4.00x    4.00x
l2_proj          bf16  ld1    16  1024  2048  4 |     46.13    93.78   1454.93 |    175.33    382.75 |   3.80x    3.80x
l2_proj          bf16  ld1    32  1024  2048  4 |     90.75    49.11   1478.98 |    183.88    729.94 |   2.03x    2.03x
l2_proj          bf16  ld1   128  1024  2048  4 |    345.71    15.17   1552.96 |    446.17   1203.30 |   1.29x    1.29x
l2_proj          bf16  ld1   512  1024  2048  4 |   1447.96     5.79   1483.11 |   1453.88   1477.08 |   1.00x    1.00x
l2_proj          f16   ld1     1  1024  2048  4 |     31.67   132.71    132.45 |     35.25    118.99 |   1.11x    1.11x
l2_proj          f16   ld1    16  1024  2048  4 |     48.00    90.11   1398.10 |    182.21    368.31 |   3.80x    3.80x
l2_proj          f16   ld1   128  1024  2048  4 |    341.75    15.34   1570.95 |    449.00   1195.70 |   1.31x    1.31x
l2_proj          f16   ld1   512  1024  2048  4 |   1355.92     6.19   1583.79 |   1436.13   1495.33 |   1.06x    1.06x
l2_proj          q4    ld1     1  1024  2048  4 |     31.25    42.21    134.22 |     31.63    132.63 |   1.01x   18.78x
l2_proj          q4    ld1     4  1024  2048  4 |     98.79    13.60    169.83 |    179.79     93.31 |   1.82x    7.39x
l2_proj          q4    ld1    16  1024  2048  4 |     80.92    17.82    829.35 |    170.67    393.22 |   2.11x    9.31x
l2_proj          q4    ld1    32  1024  2048  4 |     87.96    17.88   1525.93 |    172.92    776.20 |   1.97x    8.38x
l2_proj          q4    ld1   128  1024  2048  4 |    342.33     6.89   1568.27 |    437.25   1227.84 |   1.28x    2.98x
l2_proj          q3    ld1     1  1024  2048  4 |     42.92    24.62     97.73 |     31.17    134.58 |   0.73x   14.06x
l2_proj          q3    ld1    16  1024  2048  4 |     83.33    14.16    805.31 |    169.29    396.41 |   2.03x    8.78x
l2_proj          q3    ld1   128  1024  2048  4 |    347.00     6.04   1547.18 |    443.21   1211.33 |   1.28x    2.86x
l2_proj          q2    ld1     1  1024  2048  4 |     28.71    27.68    146.10 |     24.08    174.16 |   0.84x   20.43x
l2_proj          q2    ld1    16  1024  2048  4 |     79.88    11.49    840.17 |    167.13    401.55 |   2.09x    9.09x
l2_proj          q2    ld1   128  1024  2048  4 |    364.17     5.04   1474.24 |    426.21   1259.65 |   1.17x    2.72x
dram_slc_bust    bf16  ld1     1  4096  4096  4 |    522.17    64.31     64.26 |   1016.96     32.99 |   1.95x    1.95x
dram_slc_bust    bf16  ldnt    1  4096  4096  4 |    473.67    70.89     70.84 |   1036.92     32.36 |   2.19x    2.19x
dram_slc_bust    bf16  ld1    16  4096  4096  4 |    488.92    69.43   1098.08 |   2188.63    245.30 |   4.48x    4.48x
dram_slc_bust    bf16  ld1   128  4096  4096  4 |   2588.83    14.18   1659.04 |   3762.33   1141.57 |   1.45x    1.45x
dram_slc_bust    q4    ld1     1  4096  4096  4 |    218.38    48.13    153.66 |   1003.46     33.44 |   4.60x   26.12x
dram_slc_bust    q4    ldnt    1  4096  4096  4 |    219.54    47.87    152.84 |    997.83     33.63 |   4.55x   25.41x
dram_slc_bust    q4    ld1    16  4096  4096  4 |    572.54    19.00    937.70 |   2167.33    247.71 |   3.79x   12.14x
dram_slc_bust    q4    ldnt   16  4096  4096  4 |    571.21    19.05    939.89 |   2214.83    242.40 |   3.88x   12.17x
dram_slc_bust    q3    ldnt    1  4096  4096  4 |    313.04    26.88    107.19 |    986.33     34.02 |   3.15x   17.81x
dram_slc_bust    q2    ldnt    1  4096  4096  4 |    208.50    30.29    160.93 |    995.42     33.71 |   4.77x   26.56x

--- P-Cluster Thread Scaling Sweep (1 / 2 / 4 threads) ---
scenario         dtype    M     N     K  T |   sme2_us     GB/s   GFLOP/s |   scale
sweep_l2_q4      q4       1  2048  2048  1 |     50.50    52.15    166.11 |   1.00x
sweep_l2_q4      q4       1  2048  2048  2 |     49.54    53.16    169.32 |   1.02x
sweep_l2_q4      q4       1  2048  2048  4 |     51.92    50.73    161.58 |   0.97x
sweep_l2_bf16    bf16    16  2048  2048  1 |     82.33   104.27   1630.18 |   1.00x
sweep_l2_bf16    bf16    16  2048  2048  2 |     80.79   106.26   1661.27 |   1.02x
sweep_l2_bf16    bf16    16  2048  2048  4 |     78.67   109.13   1706.15 |   1.05x
sweep_l2_bf16    bf16   128  2048  2048  1 |    641.71    15.52   1673.26 |   1.00x
sweep_l2_bf16    bf16   128  2048  2048  2 |    625.46    15.93   1716.73 |   1.03x
sweep_l2_bf16    bf16   128  2048  2048  4 |    706.54    14.10   1519.71 |   0.91x
sweep_dram_q4    q4       1  4096  4096  1 |    192.88    54.49    173.97 |   1.00x
sweep_dram_q4    q4       1  4096  4096  2 |    164.92    63.73    203.46 |   1.17x
sweep_dram_q4    q4       1  4096  4096  4 |    155.17    67.74    216.25 |   1.24x

--- C13 Concurrent Metal 4 GPU + CPU SME2 Unified-Bus Contention Gate (M4 120 GB/s) ---
scenario              sme_shape         gpu_solo   gpu_conc  gpu_d% | sme_solo   sme_conc   agg_GB/s  net_spd  gate
gpu+sme2_l2_router    bf16:16x1024x2048  78.04GB/s  79.62GB/s    2.0% |    44.4us    56.9us   155.57    1.80x  PASS
gpu+sme2_dram_ld1     q4:1x4096x4096     78.04GB/s  68.29GB/s  -12.5% |   212.1us   470.1us    90.65    1.33x  PASS
gpu+sme2_dram_ldnt1   q4:1x4096x4096     78.04GB/s  71.64GB/s   -8.2% |   198.0us   480.9us    93.49    1.33x  PASS

--- C12/C13 ldnt1b Non-Temporal Cold-Expert L2 Cache Preservation ---
ldnt_l2.hot_router_kib           4096.0
ldnt_l2.cold_stream_mib          16.88 (6 q4 SwiGLU experts)
ldnt_l2.hot_baseline_us          44.17
ldnt_l2.hot_after_ld1_us         62.25 (40.9% vs baseline)
ldnt_l2.hot_after_ldnt1_us       44.08 (-0.2% vs baseline)
ldnt_l2.cold_swiglu_ld1_us       86.31 /expert
ldnt_l2.cold_swiglu_ldnt1_us     81.51 /expert

--- C13 L2-Resident SME2 vs Metal 4 GPU Speculative Tree Router Prediction ---
router_cal.bank_kib              4096.0
M    shape (L x E x H)       gpu_us    sme2_us   speedup   topk_agree
1    8x128x2048              573.46      32.38    17.71x   100.00%
8    8x128x2048              857.92      74.75    11.48x   100.00%
16   8x128x2048              759.17     106.13     7.15x    99.90%

M4.5 Swap Check
m45.swap.status                  ok
m45.swap.swapouts_delta          0
m45.swap.pageouts_delta          0
```

### Verbatim real-checkpoint `verify-amf`, `repack-sme2`, `Qwen3-30B-A3B-3bit` + `EAGLE-3` C13 A/B, and `Qwen3-0.6B` D11 TRNG raw stdout (commit `b23f787`)

```text
=== 5. VERIFY-AMF ON ALL 4 EXISTING REAL CHECKPOINTS ===
--- verify-amf /Volumes/Akari/models/amf/qwen3-0.6b-bf16.amf ---
amf.status       ok
amf.model_id     qwen3-0.6b-bf16
amf.tensors      310
amf.file_bytes   1193885696
amf.wired_bytes  1193820160
amf.cold_bytes   0
--- verify-amf /Volumes/Akari/models/amf/qwen3-14b-4bit.amf ---
amf.status       ok
amf.model_id     qwen3-14b-4bit
amf.tensors      443
amf.file_bytes   8309669888
amf.wired_bytes  8309587968
amf.cold_bytes   0
--- verify-amf /Volumes/Akari/models/amf/qwen3-30b-a3b-3bit.amf ---
amf.status       ok
amf.model_id     qwen3-30b-a3b-3bit
amf.tensors      18867
amf.file_bytes   13363314688
amf.wired_bytes  13360873472
amf.cold_bytes   0
--- verify-amf /Volumes/Akari/models/amf/qwen3-30b-a3b-eagle3.amf ---
amf.status       ok
amf.model_id     qwen3-30b-a3b-eagle3
amf.tensors      15
amf.file_bytes   1044217856
amf.wired_bytes  1044185088
amf.cold_bytes   0

=== 6. REPACK-SME2 REAL QWEN3-0.6B-BF16 SINGLE-OWNER CHECK ===
repacked /Volumes/Akari/.tmp/qwen3-0.6b-sme2.amf (1193885696 bytes, targets=all-linear): sme2_panel64=196, row_major=114 in 5.50 s
amf.status       ok
amf.model_id     qwen3-0.6b-bf16
amf.tensors      310
amf.file_bytes   1193885696
amf.wired_bytes  1193820160
amf.cold_bytes   0
--- decode original row_major ---
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             10
decode.generated_tokens          32
decode.prefill_ms                164.506
decode.decode_ms_per_tok         14.563
decode.decode_tok_per_s          68.67
decode.bytes_per_tok             1196918784
decode.effective_gbs             82.19
decode.malloc_blocks_delta       62
decode.malloc_bytes_delta        3968
decode.swapouts_delta            0
decode.token_ids                 [8162, 37150, 72434, 13850, 374, 264, 3175, 4938, 3550, 429, 374, 1483, 369, 678, 279, 8357, 323, 11364, 4303, 389, 264, 7401, 13, 1084, 374, 264, 3175, 4938, 3550, 429, 374, 1483]
decode.text                       Apple Silicon Unified Memory is a single memory space that is used for all the applications and processes running on a Mac. It is a single memory space that is used
--- decode repacked sme2_panel64 ---
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             10
decode.generated_tokens          32
decode.prefill_ms                485.986
decode.decode_ms_per_tok         45.572
decode.decode_tok_per_s          21.94
decode.bytes_per_tok             1196918784
decode.effective_gbs             26.26
decode.malloc_blocks_delta       1
decode.malloc_bytes_delta        192
decode.swapouts_delta            0
decode.sme2_weights              yes
decode.token_ids                 [8162, 37150, 72434, 13850, 374, 264, 3175, 4938, 3550, 429, 5610, 678, 279, 821, 323, 11221, 429, 264, 7401, 61594, 6366, 5711, 11, 2670, 279, 10350, 1849, 11, 8357, 11, 323, 1008]
decode.text                       Apple Silicon Unified Memory is a single memory space that contains all the data and instructions that a Macintosh computer uses, including the operating system, applications, and other

=== 7. REAL QWEN3-0.6B BF16 D11 TRNG + REPLAY (DECODE & SPEC-DECODE) ===
--- decode TRNG run 1 (no --seed) ---
decode.temperature               0.70
decode.seed_used                 15577856275329892283
decode.seed_source               hardware_trng
decode.decode_tok_per_s          61.85
decode.swapouts_delta            0
decode.token_ids                 [279, 10085, 11, 12941, 6945, 11, 323, 30128, 24231, 13, 5443, 220, 20, 23992, 962, 4244, 13, 8886, 1555, 702, 311, 614, 220, 20, 4244, 2790, 13, 576, 6386, 38242, 1265, 387]
--- decode TRNG run 2 (no --seed) ---
decode.temperature               0.70
decode.seed_used                 8891266525397946930
decode.seed_source               hardware_trng
decode.decode_tok_per_s          64.79
decode.swapouts_delta            0
decode.token_ids                 [220, 16, 13, 8162, 37150, 374, 264, 35821, 6200, 11, 220, 17, 13, 8162, 37150, 374, 264, 26917, 1849, 11, 220, 18, 13, 8162, 37150, 374, 264, 6200, 315, 279, 3853, 382]
--- decode REPLAY run 3 (--seed 15577856275329892283) ---
decode.temperature               0.70
decode.seed_used                 15577856275329892283
decode.seed_source               explicit
decode.decode_tok_per_s          66.23
decode.swapouts_delta            0
decode.token_ids                 [279, 10085, 11, 12941, 6945, 11, 323, 30128, 24231, 13, 5443, 220, 20, 23992, 962, 4244, 13, 8886, 1555, 702, 311, 614, 220, 20, 4244, 2790, 13, 576, 6386, 38242, 1265, 387]
VERIFY decode: seed1=15577856275329892283 seed2=8891266525397946930 distinct=YES diverse=YES replay_bit_exact=YES
--- spec-decode TRNG run 1 (no --seed) ---
spec.temperature                 0.70
spec.seed_used                   4723189805597218849
spec.seed_source                 hardware_trng
spec.decode_tok_per_s            62.64
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [264, 3671, 429, 374, 7988, 323, 11050, 11, 7881, 553, 8162, 4848, 13, 304, 279, 2274, 11, 323, 429, 374, 1483, 304, 279, 8162, 24982, 13, 7281, 11, 2924, 279, 829, 315]
--- spec-decode TRNG run 2 (no --seed) ---
spec.temperature                 0.70
spec.seed_used                   3910625554483478406
spec.seed_source                 hardware_trng
spec.decode_tok_per_s            65.49
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [264, 5662, 624, 32942, 38242, 374, 264, 75434, 1352, 429, 5711, 2326, 5128, 11, 1817, 448, 264, 47604, 480, 1760, 315, 220, 20, 624, 785, 1156, 1555, 1265, 614, 220, 20, 47604]
--- spec-decode REPLAY run 3 (--seed 4723189805597218849) ---
spec.temperature                 0.70
spec.seed_used                   4723189805597218849
spec.seed_source                 explicit
spec.decode_tok_per_s            66.18
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [264, 3671, 429, 374, 7988, 323, 11050, 11, 7881, 553, 8162, 4848, 13, 304, 279, 2274, 11, 323, 429, 374, 1483, 304, 279, 8162, 24982, 13, 7281, 11, 2924, 279, 829, 315]
VERIFY spec-decode: seed1=4723189805597218849 seed2=3910625554483478406 distinct=YES diverse=YES replay_bit_exact=YES

=== 8. REAL QWEN3-30B-A3B-3BIT + EAGLE-3 GPU-ROUTER vs SME2-ROUTER A/B (T=0) ===
--- 8A. Qwen3-30B-A3B-3bit baseline decode (T=0) ---
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    48
decode.shards                    1
decode.wired_mib                 12791
decode.prompt_tokens             34
decode.generated_tokens          64
decode.prefill_ms                1040.538
decode.decode_ms_per_tok         20.110
decode.decode_tok_per_s          49.73
decode.bytes_per_tok             1340781440
decode.effective_gbs             66.67
decode.malloc_blocks_delta       126
decode.malloc_bytes_delta        8064
decode.swapouts_delta            0
decode.token_ids                 [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 1667, 1460, 486, 6618, 448, 21256, 88648, 21391, 13, 6771, 752, 1744, 911, 1246, 311, 5486, 419, 382, 5338, 11, 279, 10058, 4147, 374, 264, 8356, 7174, 1334, 13, 576, 1376, 374, 311, 990, 24510, 7525, 311, 5978, 4516, 7149]
decode.text                      <think>
Okay, I need to implement a lock-free single-producer single-consumer ring buffer in C++20 using std::atomic with acquire/release ordering. Let me think about how to approach this.

First, the ring buffer is a fixed-size array. The key is to use atomic operations to ensure thread safety

--- 8B. Qwen3-30B-A3B-3bit + EAGLE-3 spec-decode (--sme-router gpu, T=0) ---
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       1383.6
spec.placement_gate_gpu_us       292.2
spec.router_pred_backend         gpu (requested=gpu)
spec.router_pred_cal             sme2_us=0.0 gpu_us=0.0 bank_kib=0.0
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      48
spec.shards                      1
spec.wired_mib                   13876
spec.prompt_tokens               34
spec.generated_tokens            64
spec.rounds                      26
spec.mean_accept_len             2.423
spec.accept_hist                 [1:7, 2:7, 3:6, 4:6]
spec.drafted_nodes_per_round     10.69
spec.verified_nodes_per_round    2.88
spec.l1_seed_rounds              6
spec.unique_experts_per_layer    15.31
spec.prefill_ms                  287.852
spec.decode_ms_per_tok           19.050
spec.decode_tok_per_s            52.49
spec.draft_ms_per_round          4.184
spec.prune_ms_per_round          0.396
spec.verify_ms_per_round         36.177
spec.commit_sync_ms_per_round    5.402
spec.bytes_per_accepted_tok      851223595
spec.effective_gbs               44.68
spec.malloc_blocks_delta         320
spec.malloc_bytes_delta          20640
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 1667, 1460, 486, 6618, 448, 21256, 88648, 21391, 13, 6771, 752, 1744, 911, 1246, 311, 5486, 419, 382, 5338, 11, 279, 10058, 4147, 374, 264, 8356, 7174, 1334, 13, 576, 1376, 374, 311, 990, 24510, 7525, 311, 5978, 4516, 7149]

--- 8C. Qwen3-30B-A3B-3bit + EAGLE-3 spec-decode (--sme-router auto -> SME2 L2, T=0) ---
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       1317.5
spec.placement_gate_gpu_us       240.8
spec.router_pred_backend         sme2 (requested=auto)
spec.router_pred_cal             sme2_us=78.0 gpu_us=975.5 bank_kib=4096.0
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      48
spec.shards                      1
spec.wired_mib                   13880
spec.prompt_tokens               34
spec.generated_tokens            64
spec.rounds                      26
spec.mean_accept_len             2.423
spec.accept_hist                 [1:6, 2:8, 3:7, 4:5]
spec.drafted_nodes_per_round     10.04
spec.verified_nodes_per_round    2.85
spec.l1_seed_rounds              6
spec.unique_experts_per_layer    15.20
spec.prefill_ms                  266.302
spec.decode_ms_per_tok           18.878
spec.decode_tok_per_s            52.97
spec.draft_ms_per_round          3.774
spec.prune_ms_per_round          0.217
spec.verify_ms_per_round         36.147
spec.commit_sync_ms_per_round    5.604
spec.bytes_per_accepted_tok      846664147
spec.effective_gbs               44.85
spec.malloc_blocks_delta         194
spec.malloc_bytes_delta          12576
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.token_ids                   [151667, 198, 32313, 11, 358, 1184, 311, 4211, 264, 5296, 12577, 3175, 45254, 3828, 3175, 14859, 11761, 10058, 4147, 304, 356, 1027, 17, 15, 1667, 1460, 486, 6618, 448, 21256, 88648, 21391, 13, 6771, 752, 1744, 911, 1246, 311, 5486, 419, 382, 5338, 11, 279, 10058, 4147, 374, 264, 8356, 7174, 1334, 13, 576, 1376, 374, 311, 990, 24510, 7525, 311, 5978, 4516, 7149]
VERIFY 30B: base_eq_gpu=YES base_eq_sme2=YES gpu_eq_sme2=YES
```

## M5 serving, pipelining, C6 NVMe N-gram and ADR-0017 GSQ+RCO hybrid quantization

Measured on 2026-11-21 on the idle Apple M4 24 GiB baseline (`aminfer bench-m5`,
`verify-amf`, `bench-decode`, `decode`, `spec-decode`, and live `aminfer serve`,
ReleaseFast at commits `b78fc8e`–`f9d8c11`, `swapouts_delta = 0` across every
archived run).

> **Run context:** Idle-GPU measurements below (`bench-m5`, `W14`/`G13` `4.35b`
> GSQ+RCO vs MinMax vs BF16, and `D8` off-GPU SME2 + same-GPU multi-prefix PEARL)
> were captured at commit `f9d8c11` with an idle M4 GPU. The initial `C6`
> `aminfer-repo.ang` multi-prompt sweep, `3.25b` GSQ stress sweep, and live
> `aminfer serve` HTTP/SSE load runs (commits `b78fc8e`–`d5069a2`) ran during an
> earlier window while an unrelated user GPU process was active (idle-GPU `C6`
> re-check at `f9d8c11` is also included below); every archived run passed the
> strict `vmstat.checkNoSwap` (`swapouts_delta = 0`) gate.

### G13 `q2_affine` Metal 4 calibration & scheme-bucketed heterogeneous MoE (`aminfer bench-m5`, idle GPU, `f9d8c11`)

| Kernel / Profile | Shape / Geometry | Bytes/disp | vs `q4` bytes | Buckets | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemv_q2` | `1×2048×2048` | 1,318,912 | — | — | 48.43 | 27.23 | 22.7% |
| `gemv_q2` | `1×6144×2048` | 3,948,544 | — | — | 58.56 | 67.43 | 56.2% |
| `gemv_q2` | `1×2048×768` | 497,152 | — | — | 9.24 | 53.79 | 44.8% |
| `gemv_q2` | `1×4096×4096` | 5,259,264 | — | — | 72.90 | **72.15** | **60.1%** |
| `gemm_q2` | `16×2048×2048` | 1,441,792 | — | — | 130.05 | 11.09 | 9.2% |
| `gemm_q2` | `16×4096×4096` | 5,505,024 | — | — | 422.76 | 13.02 | 10.9% |
| `gemm_q2` | `64×2048×2048` | 1,835,008 | — | — | 231.90 | 7.91 | 6.6% |
| `gemm_q2` | `64×4096×4096` | 6,291,456 | — | — | 903.36 | 6.96 | 5.8% |
| MoE `uniform_q4` | `M=1, E=128, k=8, U=8` | 21,242,368 | `0.0%` | 1 | 253.81 | 83.69 | 69.7% |
| MoE `uniform_q3` | `M=1, E=128, k=8, U=8` | 16,523,776 | `-22.2%` | 1 | 260.10 | 63.53 | 52.9% |
| MoE `uniform_q2` | `M=1, E=128, k=8, U=8` | 11,805,184 | `-44.4%` | 1 | 242.45 | 48.69 | 40.6% |
| **MoE `hetero_q2_q3_q4`** | `M=1, E=128, k=8, U=8` | 15,344,128 | **`-27.8%`** | 3 | **246.64 (`1.03×` vs `q4`)** | 62.21 | 51.8% |
| MoE `uniform_q4` | `M=8, E=128, k=8, U=24` | 63,770,624 | `0.0%` | 1 | 1,012.56 | 62.98 | 52.5% |
| MoE `uniform_q3` | `M=8, E=128, k=8, U=24` | 49,614,848 | `-22.2%` | 1 | 1,054.58 | 47.05 | 39.2% |
| MoE `uniform_q2` | `M=8, E=128, k=8, U=24` | 35,459,072 | `-44.4%` | 1 | 950.55 | 37.30 | 31.1% |
| **MoE `hetero_q2_q3_q4`** | `M=8, E=128, k=8, U=24` | 46,075,904 | **`-27.7%`** | 3 | **909.14 (`1.11×` vs `q4`)** | 50.68 | 42.2% |

Notes:
- Zero-sync, zero-allocation 3-bucket CSR dispatch (`ami_moe_route_topk_bucketed`
  + per-scheme specialized Metal 4 pipelines) cuts MoE weight traffic by
  `-27.7%` to `-27.8%` while running within `1.00×`–`1.11×` of 1-bucket uniform
  `q4` (`246.64–255.53 us` vs `253.81–256.46 us` at `M = 1`; `909.14–989.76 us`
  vs `991.97–1,012.56 us` at `M = 8`) with `0` hot-path heap allocations and
  `0` CPU-GPU syncs; `SpecEngine` wires `Qwen3Model`'s per-expert static byte
  and per-scheme bandwidth tables into `D7` `PruneCostModel`
  ([ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md)).

### W14 + G13 real `Qwen3-0.6B` GSQ + RCO vs naive min-max vs uniform BF16 (`ADR-0017`, idle GPU, `f9d8c11`)

Side-by-side comparison on real `Qwen/Qwen3-0.6B` (28 layers, tied embeddings,
310 tensors; scale-invariant layer-norm channel-peakiness proxy + boundary-layer
`L0`/`L27` & attention protection from commit `19b7372`; all files verified with
`aminfer verify-amf`, `swapouts_delta = 0`):

| Profile (`target-bits`) | Tensor mix | File size (MB) | `wired_mib` (`seq=512` / `decode`) | `bytes/tok` (`seq=512`) | Mean quant MSE (RCO vs MinMax) | `bench-decode` (`seq=512`) | ZH (`kqueue`) / ZH (`M4`) / EN (`zero-copy`) `decode` (`tok/s`) | Output coherence |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Uniform `bf16` baseline | `bf16=310` | 1,193.89 | 1,251 / 1,195 | 1,251,051,520 | `0` (ref) | 61.09 tok/s (`16.37 ms`) | 64.24 / 66.51 / 67.50 | Fluent EN & ZH |
| **`GSQ + RCO` (`4.35` eff-b, archived)** | `bf16=113, q3=29, q4=168` | **325.76** (`-72.7%`) | **423 (`-66.2%`) / 367 (`-69.3%`)** | **382,919,232 (`-69.4%`)** | **`8.4796e-6`** (`-16.82%` vs MinMax, `1.20×` lower) | **171.05 tok/s (`2.80×`)** | **203.90 (`3.17×`) / 205.52 (`3.09×`) / 195.80 (`2.90×`)** | Fluent EN & ZH |
| `GSQ MinMax` (`4.35` eff-b, `--no-rco`) | `bf16=113, q3=29, q4=168` | 325.76 | 423 / 367 | 382,919,232 | `1.0194e-5` (baseline) | 179.33 tok/s (`2.94×`) | 197.02 / 200.31 / 199.89 | Fluent EN & ZH |
| `GSQ + RCO` (`3.25` eff-b, earlier stress sweep) | `bf16=113, q2=149, q3=24, q4=24` | 243.84 (`-79.6%`) | 345 (`-72.4%`) / 289 (`-75.8%`) | 300,884,544 (`-75.9%`) | **`6.7565e-5`** (`-46.86%` vs MinMax, `1.88×` lower) | 59.34 tok/s (`1.76×` vs `33.71`) | 69.72 / 96.64 (under load) | Degraded (58.4% 2-bit on 0.6B dense) |
| `GSQ MinMax` (`3.25` eff-b, `--no-rco`) | `bf16=113, q2=146, q3=31, q4=20` | 243.84 | 345 / 289 | 300,884,544 | `1.2715e-4` (baseline) | 69.00 tok/s (`2.05×` vs `33.71`) | 96.55 / 98.61 (under load) | Collapsed (`tr tr`, `Aتم...`) |

Notes ([ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md)):
- Replacing raw depth-growing RMSNorm energy (`ln_in[26] / ln_in[0] > 10,000×`
  on Qwen3, which starved `L0..L16` and broke Chinese output) with a
  scale-invariant gamma channel-peakiness proxy (`max(|w|) / rms(w)`) plus
  boundary-layer (`L0`/`L27`) and attention `>= q4_affine` floors preserves
  coherent Chinese and English output across the tested prompts while reducing
  file size by `72.7%` and wired RAM by `66.2%–69.3%`.
- Offline RCO alternating ridge-OLS scale/bias refinement reduces quantization
  MSE by `16.82%` (`1.20×` lower) at `4.35` eff-bits (`17.03%` on matched-scheme
  tensors) and `46.86%` (`1.88×` lower) at `3.25` eff-bits with no runtime
  changes. Notice in the table above that `GSQ MinMax` (`179.33 tok/s` at
  `seq=512`) and `GSQ + RCO` (`171.05–177.45 tok/s` across runs) execute the
  same tensor bit-widths; small `tok/s` differences between them reflect normal
  run-to-run clock variance rather than RCO runtime cost.
- At `4.35` eff-bits (`/Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf`),
  idle-GPU `Qwen3-0.6B` runs at `171.05–177.45 tok/s` (`2.80–2.90×` vs BF16) at
  `seq=512` and `195.80–205.52 tok/s` (`2.90–3.17×`) on short EN/ZH chat decode
  with `swapouts_delta = 0`.
- Compressing the `0.6B` dense model down to `3.25` eff-bits assigns `149/197`
  matrices (`58.4%` weights) to `q2_affine` and degrades output coherence
  (collapsing completely without RCO), showing that `q2_affine` should be
  confined to low-sensitivity or cold MoE experts rather than tiny dense backbones.

### C6 repo-scale `F_NOCACHE` NVMe `.ang` N-gram table (`aminfer-repo.ang`)

Built `/Volumes/Akari/models/amf/aminfer-repo.ang` from
`/Volumes/Akari/workflow/aminfer/{src,docs,tools}` using
`aminfer-tools build-ngram` + `Qwen3-0.6B/tokenizer.json`:
**`134,234,112 B` (`128.02 MiB`)**, `8,192` × 16 KiB bucket pages, `879,949`
`2..4`-gram entries (`0` dropped), `88` files, `1,194,964` tokens,
`table_crc32 = 0xfe3a0eb2`, `header_crc32 = 0x4523e5c0`, `512 KiB` wired LRU
slot cache (`32` × 16 KiB pages, `F_NOCACHE`).

- **Idle-GPU `bench-m5` synthetic `.ang` lookup latency** (`256` buckets / `4.02 MiB` freshly written temp file in `$TMPDIR`):
  - Fresh temp-file `F_NOCACHE` `pread` p50: `1.13–1.54 us` (p99: `1.63–1.96 us`; note: `F_NOCACHE` bypasses the macOS unified buffer cache, not SSD controller volatile DRAM on a newly written `4 MiB` file — true external-SSD flash latency is reflected in the `396–1,208 us` `128 MiB` runs below, consistent with Spike S3)
  - Wired LRU cache hit p50: `0.04 us` (`40 ns`)
  - Async `kqueue_aio` `prefetch()` + `query()` hit p50: `0.13 us`
- **Real-checkpoint `T = 0` lossless `spec-decode` with vs without `--ngram-nvme aminfer-repo.ang`**:
  - Note on workload sensitivity: the `APPLE_PAGE_SIZE` raw-completion prompt triggered a repetitive numeric continuation (`= 1024 + 1024 + ...`) during the earlier GPU-loaded window (`41.48 tok/s` base → `106.33` L1 → `216.28 tok/s` with `.ang` seed hits). On non-repetitive code prompts (`alignUp`, `page.zig`), `396–1,208 us` external-SSD `.ang` queries yield `0.95×`–`1.01×` of in-RAM L1 SAM on `0.6B` (and `3.72` vs `5.98 tok/s` on `14B` when wider trees execute under host GPU load) while maintaining bit-exact `token_ids`.

| Model & Prompt | Mode | Bit-exact vs `decode` | `tau` (mean accept) | Rounds | `ngram` queries / hits / seed hits | Mean `.ang` query (us) | `bytes/acc tok` | Decode `tok/s` | `swapouts_delta` |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig) | `decode` baseline | ref (`32/32`) | `1.000` | 31 | — | — | 1,197.8 MB | 41.48 | 0 |
| `Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig) | `spec-decode l1` | `YES (32/32)` | `2.818` | 11 | — | — | 424.2 MB | 106.33 (`2.56×`) | 0 |
| **`Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig)** | **`spec-decode l1 + C6 .ang`** | **`YES (32/32)`** | **`5.167`** | **6** | **`3 / 3 / 2`** | **`396.10`** | **231.5 MB (`-45.4%` vs L1)** | **216.28 (`2.03×` vs L1, `5.21×` vs base)** | **0** |
| `Qwen3-0.6B` BF16 (`alignUp` Zig, idle GPU) | `decode` / `l1` / `l1 + C6 .ang` | `YES (32/32)` | `1.000` / `2.818` / `2.818` | 31 / 11 / 11 | `9 / 2 / 0` | `840.33` | 1,201.6 / 425.8 / 425.8 MB | 63.85 / 170.58 / 162.35 (`2.54×` vs base) | 0 |
| `Qwen3-0.6B` BF16 (`alignUp` 64-tok Zig) | `decode` baseline | ref (`64/64`) | `1.000` | 63 | — | — | 1,204.1 MB | 36.70 | 0 |
| `Qwen3-0.6B` BF16 (`alignUp` 64-tok Zig) | `spec-decode l1` | `YES (64/64)` | `1.500` | 42 | — | — | 800.6 MB | 55.82 (`1.52×`) | 0 |
| **`Qwen3-0.6B` BF16 (`alignUp` 64-tok Zig)** | **`spec-decode l1 + C6 .ang`** | **`YES (64/64)`** | **`1.615`** | **39** | **`34 / 18 / 1`** | **`913.44`** | **743.4 MB (`-7.1%` vs L1)** | **56.13 (`1.53×` vs base)** | **0** |
| `Qwen3-14B-4bit` (`page.zig` Zig test) | `decode` baseline | ref (`48/48`) | `1.000` | 47 | — | — | 7,887.6 MB | 5.95 | 0 |
| `Qwen3-14B-4bit` (`page.zig` Zig test) | `spec-decode l1` | `YES (48/48)` | `1.469` | 32 | — | — | 5,367.4 MB | 5.98 | 0 |
| **`Qwen3-14B-4bit` (`page.zig` Zig test)** | **`spec-decode l1 + C6 .ang`** | **`YES (48/48)`** | **`1.621`** | **29** | **`27 / 16 / 3`** | **`1,207.78`** | **4,864.4 MB (`-9.4%` vs L1)** | 3.72 (larger tree `4.86` vs `3.16` nodes under host GPU load) | **0** |

### D8 off-GPU SME2 + multi-prefix PEARL async pipelined small draft (`ADR-0008`, `ADR-0018`, idle GPU, `f9d8c11` / `a599b6f`)

- **Gate behavior (`--pearl auto`, default `pearl_min_target_us = 40 ms` + `ADR-0008` median contention probe)**:
  - Off-GPU ARMv9 SME2 `Qwen3-14B-4bit + Qwen3-0.6B-GSQ-RCO-SME2` (`ldnt1b`, `0` Metal CBs from the draft worker): stays `enabled = yes` (`verify_ms` did not increase relative to `--pearl off`).
  - Fast `Qwen3-30B-A3B-3bit + EAGLE-3` (48L MoE, `--cold-experts-per-layer 64`): stays `enabled = no` (`0` PEARL rounds, `swapouts_delta = 0`), per [ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md).
  - Same-GPU `Qwen3-14B-4bit + Qwen3-0.6B` (`row_major`): probes 4 rounds against round-0 baseline `verify_us/node` and disables PEARL on longer runs when single-GPU 10-core shader contention exceeds overlapped draft savings (`13.06 tok/s` vs `11.83` forced `--pearl on` on 96-token `0.6B-GSQ-RCO`; `13.33 tok/s` vs `11.07` forced `--pearl on` on `0.6B-BF16`).
- **Idle-GPU `bench-m5` synthetic PEARL check**: at `f9d8c11`, `pearl_rounds = 7`, `pearl_hits = 2`, `pearl_saved_ms = 1.029 ms`; after `a599b6f` (completing depth-1 root expansion before early-stop), `pearl_rounds = 8`, `pearl_hits = 7`, `pearl_rollbacks = 1`, `pearl_saved_ms = 5.552 ms`, `bit_exact_off = yes`, `bit_exact_on = yes`, `swapouts_delta = 0`. Note that on the tiny synthetic 2-layer model (`base_tok_per_s = 2,341.91`), verify time is `< 1 ms`, so worker synchronization lowers synthetic `tok/s` (`1,468.01` off / `1,261.78` on) as expected.
- **Real `Qwen3-14B-4bit + Qwen3-0.6B` (`T = 0`, 64 tokens, M4 tuned small-draft tree `--max-tree-nodes 8 --max-draft-depth 5 --branch-width 1 --top-k 2`, bit-exact `token_ids`, idle M4 GPU, `swapouts_delta = 0`; default `cap=16, d=4, w=2` 32-token SME2 run at `a599b6f` also archived verbatim below at `10.22 tok/s`)**:

| Run | Draft backend & `.amf` | PEARL mode | `wired_mib` | Bit-exact vs `decode` | `tau` | `pearl_hits` / `prefix` / `rollbacks` | `pearl_saved_ms` | `draft_ms` / `verify_ms` / `sync_ms` | `bytes/acc tok` | Decode `tok/s` |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |
| `D8-base` | None (`14B` `decode`) | — | 8,006 | ref (`64/64`) | `1.000` | — | — | `—` / `92.7` / `—` | 7,885.6 MB | 10.79 |
| `D8-SME2-off` | Off-GPU SME2 `0.6B-GSQ-RCO-SME2` (`ldnt1b`) | `off` | 8,545 | `YES (64/64)` | `3.316` | `0 / 0 / 0` | `0.0` | `41.2 / 247.3 / 22.2` | 2,377.0 MB (`-69.9%`) | 10.67 |
| **`D8-SME2-on`** | **Off-GPU SME2 `0.6B-GSQ-RCO-SME2` (`ldnt1b`)** | **`on`** | **8,545** | **`YES (64/64)`** | **`3.316`** | **`7 / 11 / 12` (`36.8%` full hit)** | **`1,305.5`** | **`29.9` (`-27.4%`) / `225.7` (`0%` contention) / `25.2`** | **2,376.9 MB (`-69.9%`)** | **11.81 (`+10.7%` vs off, `+9.5%` vs base)** |
| **`D8-SME2-auto`** | **Off-GPU SME2 `0.6B-GSQ-RCO-SME2` (`ldnt1b`)** | **`auto -> ON`** | **8,545** | **`YES (64/64)`** | **`3.316`** | **`7 / 9 / 12` (`36.8%` full hit)** | **`1,195.9`** | **`29.4` (`-28.5%`) / `225.2` (`0%` contention) / `19.4` (`-12.6%`)** | **2,377.0 MB (`-69.9%`)** | **12.10 (`+13.4%` vs off, `+12.1%` vs base)** |
| `D8-GPU-GSQ-off` | Same-GPU `0.6B-GSQ-RCO` (`4.35b`) | `off` | **8,519** | `YES (64/64)` | `3.500` | `0 / 0 / 0` | `0.0` | `15.0 / 235.6 / 7.7` | 2,251.9 MB (`-71.4%`) | **13.55 (`1.26×` vs base)** |
| `D8-GPU-GSQ-on` | Same-GPU `0.6B-GSQ-RCO` (`4.35b`) | `on` | 8,519 | `YES (64/64)` | `3.500` | `6 / 10 / 12` | `1,190.4` | `11.9 / 241.7 / 8.4` | 2,251.8 MB (`-71.4%`) | 13.36 (`+6.0 ms` GPU contention) |
| `D8-GPU-GSQ-auto` | Same-GPU `0.6B-GSQ-RCO` (`4.35b`) | `auto` | 8,519 | `YES (64/64)` | `3.500` | `6 / 10 / 11` | `1,214.6` | `11.7 / 243.3 / 6.0` | 2,251.8 MB (`-71.4%`) | 13.41 (96-tok auto→OFF: **13.06** vs 12.73 off / 11.83 on) |
| `D8-GPU-BF16-off` | Same-GPU `0.6B-BF16` (`1.2 GB`) | `off` | 9,347 | `YES (64/64)` | `3.706` | `0 / 0 / 0` | `0.0` | `41.3 / 243.9 / 16.8` | 2,126.8 MB (`-73.0%`) | 12.27 (`1.14×` vs base) |
| `D8-GPU-BF16-on` | Same-GPU `0.6B-BF16` (`1.2 GB`) | `on` | 9,347 | `YES (64/64)` | `3.706` | `6 / 4 / 11` | `1,474.1` | `27.8 / 288.7 / 16.0` | 2,126.8 MB (`-73.0%`) | 11.15 (`+44.8 ms` GPU contention) |
| `D8-GPU-BF16-auto` | Same-GPU `0.6B-BF16` (`1.2 GB`) | `auto` | 9,347 | `YES (64/64)` | `3.938` | `5 / 5 / 10` | `1,150.5` | `31.6 / 287.5 / 14.6` | 2,001.7 MB (`-74.6%`) | 11.80 (`auto -> OFF` reaches **13.33** when probe triggers) |

Measured trade-offs (`ADR-0008` + `ADR-0018`):
- Running `SmallDraftModel` off-GPU on ARMv9 SME2 (`sme2_panel64` + `ldnt1b`, `0`
  draft Metal CBs) avoids the `+6.0` to `+44.8 ms/round` single-GPU `verify_ms`
  slowdown (`247.3 ms` off vs `225.2–225.7 ms` on/auto), lifting SME2-drafted
  throughput from `10.67 tok/s` (`--pearl off`) to `11.81–12.10 tok/s` (`+10.7%`
  to `+13.4%`, vs `10.79 tok/s` `14B` greedy baseline) with 64/64 bit-exact
  `token_ids` and `0` swapouts.
- On this single-GPU 10-core M4, however, **sequential same-GPU `0.6B-GSQ-RCO`
  (`--pearl off`, `13.55 tok/s`, `1.26×` vs `14B` greedy)** remains the fastest
  overall `14B + 0.6B` configuration because GPU `q4`/`q3` drafting (`15.0 ms` +
  `7.7 ms` sync) is shorter than CPU SME2 full-model drafting (`29.4 ms` +
  `19.4 ms` sync). The `--pearl auto` contention probe disables same-GPU overlap
  when `verify_ms/node` growth outweighs overlapped draft time while keeping
  off-GPU SME2 PEARL enabled.

### C7 + C8 live `aminfer serve` HTTP/1.1 + SSE & concurrency load test

- **Idle-GPU `bench-m5` synthetic SSE framing loopback microbenchmark (in-memory token emitter stub, no LLM forward pass)**:
  - `GET /health` p50: `72.33–77.33 us`
  - Single-client `POST /v1/chat/completions` (`stream: true`) TTFT p50: `106.21–109.08 us`, E2E p50: `184.88–186.17 us`
  - `8` concurrent SSE clients: TTFT p50 `791.21–806.08 us`, p99 `1,022.25–1,048.83 us`, framing-only emitter rate `100,277.5–101,245.8 tok/s`
  - Steady-state `malloc_zone_statistics` delta: `0` blocks / `0` B (`all_ok = yes`, `swapouts_delta = 0`)
- **Live `aminfer serve` backed by real `Qwen3-0.6B` BF16 (`1,251 MiB` wired, loopback ephemeral port)**:
  - `GET /health` -> `200 OK` (`{"status":"ok","model":"qwen3-0.6b-bf16","wired_mib":1251,"active_requests":0}`)
  - `GET /v1/models` -> `200 OK` (OpenAI `list` object with `qwen3-0.6b-bf16`)
  - `POST /v1/chat/completions` (`stream: true`, `enable_thinking: true`): cleanly separated 448 chars of `<think>...</think>` into `choices[0].delta.reasoning_content` and `"\n\n4"` into `choices[0].delta.content`, returning `usage`, `aminfer_timings` (`36.74 tok/s`, `tpot_ms = 27.22`), `X-Aminfer-Seed-Used`, and `data: [DONE]`
  - `POST /v1/chat/completions` (`stream: false`): `200 OK`, `"Two planets are **Mars** and **Saturn**."`, `36.16 tok/s`
  - **`D11` SSE TRNG + replay** (`temperature: 0.7`, `stream: true`): two unseeded calls drew distinct Secure Enclave TRNG seeds (`6438769541196058799` vs `13003324638593368964`, `seed_source = "hardware_trng"`), and replaying with `"seed": 6438769541196058799` (`seed_source = "explicit"`) produced **100% bit-exact SSE content replay**
  - Thin stateless adapters: `POST /v1/responses` (`response.created`, `response.output_text.delta`, `response.completed`) and `POST /v1/messages` (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`) both returned `200 OK` streaming `"Hello! How can I assist you today?"`
  - **Multi-client concurrent SSE load (`1, 4, 8` concurrent clients × 24 tokens each)**:

| Concurrent clients | Requests OK | Total tokens | Wall time (s) | TTFT p50 (ms) | TTFT p99 (ms) | Mean TPOT (ms/tok) | Aggregate `tok/s` | `rejected_503` / `cancelled` | `serve.swapouts_delta` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | `1/1` (`200 OK`) | 24 | 1.204 | 557.7 | 557.7 | 27.08 | 19.93 | `0 / 0` | 0 |
| 4 | `4/4` (`200 OK`) | 96 | 4.210 | 2,567.7 | 3,611.0 | 27.38 | 22.80 | `0 / 0` | 0 |
| 8 | `8/8` (`200 OK`) | 192 | 8.239 | 4,526.5 | 7,644.8 | 26.46 | 23.30 | `0 / 0` | 0 |

### Verbatim M5 CLI raw stdout (`bench-m5`, `verify-amf`, `bench-decode`, `decode`, `spec-decode`, `build-ngram`)

Full unedited CLI stdout at post-remediation code (`a599b6f` / `8497fb7`):

```text
=== M5 Serving, Pipelining, C6 NVMe N-Gram & ADR-0017 GSQ+RCO Hybrid Quant Suite ===

--- G13 Metal 4 q2_affine Kernel Bandwidth Calibration ---
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q2   1x2048x2048           1318912      64         87.11        15.14     12.6
gemv_q2   1x6144x2048           3948544      21        109.32        36.12     30.1
gemv_q2   1x2048x768             497152     170         16.56        30.02     25.0
gemv_q2   1x4096x4096           5259264      16        112.78        46.63     38.9
gemm_q2   16x2048x2048          1441792      64        146.24         9.86      8.2
gemm_q2   16x4096x4096          5505024      16        504.82        10.90      9.1
gemm_q2   64x2048x2048          1835008      64        230.45         7.96      6.6
gemm_q2   64x4096x4096          6291456      16        882.31         7.13      5.9

--- G13 Scheme-Bucketed Heterogeneous MoE ({q2, q3, q4}) vs Uniform CSR Gather ---
profile          m    E    k   U    bkts    bytes/disp   vs_q4(%)   latency(us)   beta(GB/s)   bus(%)
uniform_q4       1    128  8   8    1         21242368        0.0        256.46        82.83     69.0
uniform_q3       1    128  8   8    1         16523776      -22.2        270.87        61.00     50.8
uniform_q2       1    128  8   8    1         11805184      -44.4        241.66        48.85     40.7
hetero_q2_q3_q4  1    128  8   8    3         15344128      -27.8        255.53        60.05     50.0
uniform_q4       8    128  8   24   1         63770624        0.0        991.97        64.29     53.6
uniform_q3       8    128  8   24   1         49614848      -22.2       1014.77        48.89     40.7
uniform_q2       8    128  8   24   1         35459072      -44.4        938.16        37.80     31.5
hetero_q2_q3_q4  8    128  8   24   3         46075904      -27.7        989.76        46.55     38.8

--- C6 F_NOCACHE NVMe .ang N-Gram Lookup Latency ---
ngram.file_bytes                 4210688
ngram.buckets                    256
ngram.total_entries              1024
ngram.wired_cache_kib            512
ngram.cold_nvme_p50_us           1.54
ngram.cold_nvme_p99_us           1.96
ngram.cache_hit_p50_us           0.04
ngram.prefetch_hit_p50_us        0.13
ngram.queries                    345
ngram.hits                       345
ngram.seed_hits                  345
ngram.nvme_reads                 87

--- D8 PEARL Async Pipelined Small-Draft Synthetic Parity & Overlap ---
pearl_synth.own_compute_queue    yes
pearl_synth.bit_exact_off        yes
pearl_synth.bit_exact_on         yes
pearl_synth.base_tok_per_s       2341.91
pearl_synth.off_tok_per_s        1468.01
pearl_synth.on_tok_per_s         1261.78
pearl_synth.mean_accept_len      2.375
pearl_synth.pearl_rounds         8
pearl_synth.pearl_hits           7
pearl_synth.pearl_rollbacks      1
pearl_synth.pearl_saved_ms       5.552

--- C7 + C8 kqueue HTTP/1.1 + SSE Server Loopback Microbenchmark ---
server.health_p50_us             77.33
server.single_ttft_p50_us        109.08
server.single_e2e_p50_us         186.17
server.concurrent_clients        8
server.concurrent_ttft_p50_us    791.21
server.concurrent_ttft_p99_us    1048.83
server.concurrent_tok_per_s      100277.5
server.steady_malloc_blocks      0
server.steady_malloc_bytes       0
server.all_ok                    yes

M5 Swap Check
m5.swap.status                   ok
m5.swap.swapouts_delta           0
m5.swap.pageouts_delta           0

--- verify-amf qwen3-0.6b-gsq-rco.amf & qwen3-0.6b-gsq-rco-sme2.amf ---
amf.status       ok
amf.model_id     qwen3-0.6b-gsq-rco
amf.tensors      310
amf.file_bytes   325763072
amf.wired_bytes  325697536
amf.cold_bytes   0
amf.status       ok
amf.model_id     qwen3-0.6b-gsq-rco
amf.tensors      310
amf.file_bytes   353288192
amf.wired_bytes  353222656
amf.cold_bytes   0

--- bench-decode qwen3-0.6b-gsq-rco.amf (prefill=512, steps=32) ---
decode_bench.target              qwen3
decode_bench.arch                qwen3 (bf16)
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           423
decode_bench.load_ms             1268.5
decode_bench.measured_layers     28
decode_bench.measured_ms_per_tok 5.635
decode_bench.measured_tok_per_s  177.45
decode_bench.measured_layer_ms   0.172
decode_bench.measured_head_ms    0.833
decode_bench.measured_bytes_tok  382919232
decode_bench.measured_beta_gbs   67.95 (56.6% bus)
decode_bench.scaled_layers       28 (linear: 28 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   5.635
decode_bench.scaled_tok_per_s    177.45
decode_bench.scaled_bytes_tok    382919216
decode_bench.scaled_beta_gbs     67.95 (56.6% bus)
decode_bench.malloc_blocks_delta 64
decode_bench.malloc_bytes_delta  4096
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      0

--- decode qwen3-0.6b-gsq-rco.amf (ZH ChatML) ---
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 367
decode.prompt_tokens             26
decode.generated_tokens          24
decode.prefill_ms                107.584
decode.decode_ms_per_tok         4.954
decode.decode_tok_per_s          201.86
decode.bytes_per_tok             329704000
decode.effective_gbs             66.55
decode.malloc_blocks_delta       48
decode.malloc_bytes_delta        3296
decode.swapouts_delta            0
decode.token_ids                 [26567, 386, 19, 43589, 103967, 111390, 70500, 32555, 13940, 58143, 22670, 43589, 100768, 101991, 115350, 3837, 91572, 101940, 17404, 100293, 33108, 108976, 1773, 151645]
decode.text                      Apple M4 的统一内存设计使 CPU 和 GPU 的计算效率大幅提升，同时减少功耗和发热。<|im_end|>

--- spec-decode C6 aminfer-repo.ang on qwen3-0.6b-bf16.amf ---
spec.arch                        qwen3 (bf16)
spec.mode                        l1
spec.placement                   gpu (requested=auto)
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      28
spec.shards                      1
spec.wired_mib                   1225
spec.prompt_tokens               14
spec.generated_tokens            32
spec.rounds                      22
spec.mean_accept_len             1.409
spec.accept_hist                 [1:16, 2:4, 3:1, 4:1]
spec.drafted_nodes_per_round     6.55
spec.verified_nodes_per_round    3.55
spec.l1_seed_rounds              2
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  66.885
spec.decode_ms_per_tok           12.855
spec.decode_tok_per_s            77.79
spec.draft_ms_per_round          0.754
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         16.931
spec.commit_sync_ms_per_round    0.421
spec.bytes_per_accepted_tok      848623418
spec.effective_gbs               66.01
spec.malloc_blocks_delta         89
spec.malloc_bytes_delta          5696
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.ngram_nvme                  on (buckets=8192, entries=879949, wired_kib=512)
spec.ngram_queries               21
spec.ngram_hits                  8
spec.ngram_seed_hits             2
spec.ngram_cache_hits            2
spec.ngram_nvme_reads            58
spec.ngram_mean_query_us         785.42
spec.token_ids                   [198, 262, 1077, 5206, 897, 284, 897, 280, 262, 1077, 5206, 17189, 284, 17189, 280, 262, 1077, 5206, 1102, 284, 220, 15, 401, 262, 1393, 897, 861, 220, 15, 341, 286, 421]
spec.text                        
    let mut value = value;
    let mut alignment = alignment;
    let mut result = 0;

    while value > 0 {
        if

--- spec-decode D8 off-GPU SME2 PEARL (default tree cap=16, d=4, w=2, 32 tokens) ---
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       842.1
spec.placement_gate_gpu_us       66.7
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      40
spec.shards                      1
spec.wired_mib                   8545
spec.prompt_tokens               22
spec.generated_tokens            32
spec.rounds                      10
spec.mean_accept_len             3.100
spec.accept_hist                 [1:1, 2:3, 3:2, 4:2, 5:2]
spec.drafted_nodes_per_round     15.10
spec.verified_nodes_per_round    8.10
spec.l1_seed_rounds              0
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  810.215
spec.decode_ms_per_tok           97.851
spec.decode_tok_per_s            10.22
spec.draft_ms_per_round          33.718
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         243.986
spec.commit_sync_ms_per_round    25.626
spec.bytes_per_accepted_tok      2541173192
spec.effective_gbs               25.97
spec.malloc_blocks_delta         40
spec.malloc_bytes_delta          2560
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.pearl_mode                  auto (enabled=yes, own_queue=yes)
spec.pearl_rounds                10
spec.pearl_hits                  2
spec.pearl_partial_prefix_hits   5
spec.pearl_rollbacks             8
spec.pearl_saved_ms              382.805
spec.pearl_worker_ms             1783.011
spec.token_ids                   [8420, 594, 458, 8129, 315, 264, 3070, 81816, 40996, 12199, 320, 20117, 52, 8, 19479, 334, 304, 13027, 1667, 279, 5798, 3419, 1565, 51137, 19664, 291, 13448, 28587, 1096, 5486, 374, 14452]
spec.text                        Here's an implementation of a **Least Recently Used (LRU) Cache** in Python using the built-in `collections.OrderedDict`. This approach is suitable

--- aminfer-tools build-ngram JSON stdout (commit b78fc8e) ---
{
  "output": "/Volumes/Akari/models/amf/aminfer-repo.ang",
  "file_bytes": 134234112,
  "num_buckets": 8192,
  "total_entries": 879949,
  "dropped_entries": 0,
  "source_files": 88,
  "total_tokens": 1194964,
  "min_ngram": 2,
  "max_ngram": 4,
  "max_cont_len": 8,
  "vocab_size": 151669,
  "table_crc32": "0xfe3a0eb2",
  "header_crc32": "0x4523e5c0",
  "corpus_id": "aminfer-repo-v1"
}
```

### Archived M5 multi-run sweep logs and `aminfer serve` harness output (commits `b78fc8e`–`f9d8c11`)

Preserved raw `gsq-convert` / `repack-sme2` stdout and multi-configuration
harness logs from commits `b78fc8e`–`f9d8c11`:

```text
=== IDLE-GPU AMINFER BENCH-M5 (commit f9d8c11) ===
--- G13 Metal 4 q2_affine Kernel Bandwidth Calibration ---
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q2   1x2048x2048           1318912      64         48.43        27.23     22.7
gemv_q2   1x6144x2048           3948544      21         58.56        67.43     56.2
gemv_q2   1x2048x768             497152     170          9.24        53.79     44.8
gemv_q2   1x4096x4096           5259264      16         72.90        72.15     60.1
gemm_q2   16x2048x2048          1441792      64        130.05        11.09      9.2
gemm_q2   16x4096x4096          5505024      16        422.76        13.02     10.9
gemm_q2   64x2048x2048          1835008      64        231.90         7.91      6.6
gemm_q2   64x4096x4096          6291456      16        903.36         6.96      5.8

--- G13 Scheme-Bucketed Heterogeneous MoE ({q2, q3, q4}) vs Uniform CSR Gather ---
profile          m    E    k   U    bkts    bytes/disp   vs_q4(%)   latency(us)   beta(GB/s)   bus(%)
uniform_q4       1    128  8   8    1         21242368        0.0        253.81        83.69     69.7
uniform_q3       1    128  8   8    1         16523776      -22.2        260.10        63.53     52.9
uniform_q2       1    128  8   8    1         11805184      -44.4        242.45        48.69     40.6
hetero_q2_q3_q4  1    128  8   8    3         15344128      -27.8        246.64        62.21     51.8
uniform_q4       8    128  8   24   1         63770624        0.0       1012.56        62.98     52.5
uniform_q3       8    128  8   24   1         49614848      -22.2       1054.58        47.05     39.2
uniform_q2       8    128  8   24   1         35459072      -44.4        950.55        37.30     31.1
hetero_q2_q3_q4  8    128  8   24   3         46075904      -27.7        909.14        50.68     42.2

--- C6 F_NOCACHE NVMe .ang N-Gram Lookup Latency ---
ngram.cold_nvme_p50_us=1.13  ngram.cold_nvme_p99_us=1.63  ngram.cache_hit_p50_us=0.04  ngram.prefetch_hit_p50_us=0.13

--- D8 PEARL Async Pipelined Small-Draft Synthetic Parity & Overlap ---
pearl_synth.own_compute_queue=yes bit_exact_off=yes bit_exact_on=yes base_tok_per_s=2979.65 off_tok_per_s=1740.26 on_tok_per_s=1680.57 mean_accept_len=2.714 pearl_rounds=7 pearl_hits=2 pearl_rollbacks=5 pearl_saved_ms=1.029

--- C7 + C8 kqueue HTTP/1.1 + SSE Server Loopback Microbenchmark ---
server.health_p50_us=72.33 single_ttft_p50_us=106.21 single_e2e_p50_us=184.88 concurrent_clients=8 concurrent_ttft_p50_us=806.08 concurrent_ttft_p99_us=1022.25 concurrent_tok_per_s=101245.8 steady_malloc_blocks=0 steady_malloc_bytes=0 all_ok=yes
m5.swap.status=ok m5.swap.swapouts_delta=0 m5.swap.pageouts_delta=0

=== BUILD & VERIFY QWEN3-0.6B-GSQ-RCO.AMF (4.35 eff-b, scale-invariant LN proxy, commit 19b7372) ===
gsq-converted 310 tensors into /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf (325763072 bytes, target=4.35b, effective=4.348b, rco=on) in 199.3 s [bf16=113, q3_affine=29, q4_affine=168]
repacked /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco-sme2.amf (353288192 bytes, targets=all-linear): sme2_panel64=196, row_major=114 in 12.32 s
RCO   : target=4.35 eff=4.3482 alg_bytes=323969024 mean_quant_mse=8.479575e-06 counts={'q3_affine': 29, 'q4_affine': 168}
MinMax: target=4.35 eff=4.3482 alg_bytes=323969024 mean_quant_mse=1.019441e-05 counts={'q3_affine': 29, 'q4_affine': 168}
Overall RCO MSE reduction: -16.82% (1.20x lower MSE); Matched-scheme: -17.03%
verify-amf qwen3-0.6b-gsq-rco.amf:      amf.status=ok tensors=310 file_bytes=325763072 wired_bytes=325697536 cold_bytes=0
verify-amf qwen3-0.6b-gsq-rco-sme2.amf: amf.status=ok tensors=310 file_bytes=353288192 wired_bytes=353222656 cold_bytes=0

=== W14/G13 IDLE-GPU BENCH-DECODE (seq=512, 32 steps) & EN/ZH COHERENCE ===
BF16:         wired_mib=1251 measured_ms_per_tok=16.368 measured_tok_per_s=61.09  measured_bytes_tok=1251051520 swapouts_delta=0
GSQ-RCO-4.35: wired_mib=423  measured_ms_per_tok=5.846  measured_tok_per_s=171.05 measured_bytes_tok=382919232  swapouts_delta=0
GSQ-MM-4.35:  wired_mib=423  measured_ms_per_tok=5.576  measured_tok_per_s=179.33 measured_bytes_tok=382919232  swapouts_delta=0
GSQ-RCO-4.35 ZH (kqueue):    203.90 tok/s (367 MiB) -> 现代操作系统内核中的 `kqueue` 事件驱动机制的两项核心优势包括：1. **高效性**...
GSQ-RCO-4.35 ZH (M4 统一内存): 205.52 tok/s (367 MiB) -> 1. Apple M4芯片采用统一内存技术，有效减少内存占用，提升计算效率。 2. Apple M4芯片通过统一内存技术，显著降低功耗，提高计算性能。<|im_end|>
GSQ-RCO-4.35 EN (zero-copy): 195.80 tok/s (367 MiB) -> Here are three key benefits of zero-copy shared memory on Apple Silicon: 1. **Improved performance and reduced latency**...

=== D8 IDLE-GPU PEARL A/B ON QWEN3-14B-4BIT + QWEN3-0.6B (T=0, 64/64 bit-exact, swapouts_delta=0) ===
D8-base (14B decode):   wired_mib=8006 decode_ms_per_tok=92.682 decode_tok_per_s=10.79 bytes_per_tok=7885611840 swapouts_delta=0
D8-SME2-off:            wired_mib=8545 rounds=19 tau=3.316 draft_ms=41.154 verify_ms=247.299 sync_ms=22.238 decode_tok_per_s=10.67 pearl=off swapouts_delta=0
D8-SME2-on:             wired_mib=8545 rounds=19 tau=3.316 draft_ms=29.892 verify_ms=225.728 sync_ms=25.227 decode_tok_per_s=11.81 pearl=on(hits=7,prefix=11,rb=12,saved=1305.542ms) swapouts_delta=0
D8-SME2-auto:           wired_mib=8545 rounds=19 tau=3.316 draft_ms=29.425 verify_ms=225.193 sync_ms=19.440 decode_tok_per_s=12.10 pearl=auto(enabled=yes,hits=7,prefix=9,rb=12,saved=1195.865ms) swapouts_delta=0
D8-GPU-GSQ-off:         wired_mib=8519 rounds=18 tau=3.500 draft_ms=15.024 verify_ms=235.649 sync_ms=7.710  decode_tok_per_s=13.55 pearl=off swapouts_delta=0
D8-GPU-GSQ-on:          wired_mib=8519 rounds=18 tau=3.500 draft_ms=11.900 verify_ms=241.681 sync_ms=8.365  decode_tok_per_s=13.36 pearl=on(hits=6,prefix=10,rb=12,saved=1190.394ms) swapouts_delta=0
D8-GPU-GSQ-auto:        wired_mib=8519 rounds=18 tau=3.500 draft_ms=11.669 verify_ms=243.318 sync_ms=6.032  decode_tok_per_s=13.41 pearl=auto(enabled=yes,hits=6,prefix=10,rb=11,saved=1214.637ms) swapouts_delta=0
D8-GPU-BF16-off:        wired_mib=9347 rounds=17 tau=3.706 draft_ms=41.256 verify_ms=243.872 sync_ms=16.837 decode_tok_per_s=12.27 pearl=off swapouts_delta=0
D8-GPU-BF16-on:         wired_mib=9347 rounds=17 tau=3.706 draft_ms=27.766 verify_ms=288.695 sync_ms=15.967 decode_tok_per_s=11.15 pearl=on(hits=6,prefix=4,rb=11,saved=1474.146ms) swapouts_delta=0
D8-GPU-BF16-auto:       wired_mib=9347 rounds=16 tau=3.938 draft_ms=31.644 verify_ms=287.522 sync_ms=14.618 decode_tok_per_s=11.80 pearl=auto(enabled=yes,hits=5,prefix=5,rb=10,saved=1150.483ms) swapouts_delta=0
30B+EAGLE-3 --pearl auto: wired_mib=8084 (64 cold/layer) mode=eagle3 pearl=OFF swapouts_delta=0

=== BUILD REPO-SCALE C6 AMINFER-REPO.ANG & C6/SERVE RAW TELEMETRY ===
aminfer-repo.ang: file_bytes=134234112 num_buckets=8192 total_entries=879949 dropped_entries=0 source_files=88 total_tokens=1194964 table_crc32=0xfe3a0eb2 header_crc32=0x4523e5c0 corpus_id=aminfer-repo-v1
C6 Qwen3-0.6B (APPLE_PAGE_SIZE, 32/32 bit-exact): base=41.48 tok/s -> l1=106.33 tok/s (tau=2.818) -> l1+c6=216.28 tok/s (tau=5.167, queries=3, hits=3, seed_hits=2, mean_query_us=396.10, swapouts_delta=0)
C6 Qwen3-0.6B (alignUp idle GPU, 32/32 bit-exact): base=63.85 tok/s -> l1=170.58 tok/s -> l1+c6=162.35 tok/s (queries=9, hits=2, mean_query_us=840.33, swapouts_delta=0)
GET /health -> 200 {"status": "ok", "model": "qwen3-0.6b-bf16", "wired_mib": 1251, "active_requests": 0}
GET /v1/models -> 200 {"object": "list", "data": [{"id": "qwen3-0.6b-bf16", "object": "model", "created": 1795252432, "owned_by": "aminfer"}]}
POST /v1/chat/completions (stream=true, think=true): done=True reasoning_chars=448 content_chars=3 usage={'prompt_tokens': 22, 'completion_tokens': 123, 'total_tokens': 145} timings={'ttft_ms': 378.43, 'tpot_ms': 27.22, 'tok_per_s': 36.74} content='\n\n4'
POST /v1/chat/completions (stream=false): status=200 seed=0(greedy) content='Two planets are **Mars** and **Saturn**.' usage={'prompt_tokens': 16, 'completion_tokens': 13, 'total_tokens': 29} timings={'ttft_ms': 352.624, 'tpot_ms': 27.653, 'tok_per_s': 36.16}
D11 SSE TRNG: s1=6438769541196058799(hardware_trng) s2=13003324638593368964(hardware_trng) replay=6438769541196058799(explicit) distinct=True replay_exact=True
POST /v1/responses (stream=true): status=200 events=['response.completed', 'response.created', 'response.output_text.delta'] content='Hello! How can I assist you today?'
POST /v1/messages (stream=true): status=200 events=['content_block_delta', 'content_block_start', 'content_block_stop', 'message_delta', 'message_start', 'message_stop'] content='Hello! How can I assist you today?'
LOAD conc=1: ok=True total_toks=24  wall_s=1.204 ttft_p50_ms=557.7  ttft_p99_ms=557.7  tpot_ms=27.08 agg_tok_per_s=19.93
LOAD conc=4: ok=True total_toks=96  wall_s=4.210 ttft_p50_ms=2567.7 ttft_p99_ms=3611.0 tpot_ms=27.38 agg_tok_per_s=22.80
LOAD conc=8: ok=True total_toks=192 wall_s=8.239 ttft_p50_ms=4526.5 ttft_p99_ms=7644.8 tpot_ms=26.46 agg_tok_per_s=23.30
serve.status=stopped serve.total_requests=20 serve.rejected_503=0 serve.cancelled=0 serve.swapouts_delta=0
```

## M6 frontier hybrid sparse models, recurrent depth, large-MoE GSQ+RCO and off-GPU SME2 draft heads

Measured on 2026-11-22 on the Apple M4 24 GiB baseline (`Mac16,12`, macOS 27.2,
`18,186 MiB` wired budget, ReleaseFast at commit `8a4bc68`, re-verified after
audit remediation commits `750b56c` and `f6b4705` with `swapouts_delta = 0`
across all timed decode and benchmark runs). All M6 code, Python packers, and
real-checkpoint `.amf` builds landed across commits `1c75816`, `be75050`,
`5baa49e`, `c93a5fa`, `7651cad`, `1af0f8d`, `8a4bc68`, `750b56c`, and `f6b4705`.

### 1. Hybrid-sparse operators (`G8` GDN, `G9` QSA, `G10` 4-branch GR, `G7` 512E MoE) and `C10` PLE

| Operator / subsystem | Measurement | Latency | Throughput / metric | Notes |
| --- | --- | ---: | ---: | --- |
| `G8` Gated DeltaNet (`16K/32V x 128`) | Prefill (`seq = 256`) | `3,269.5 us` | `78,300.2 tok/s` (`3.23 GB/s`) | Chunked Metal 4 kernel |
| `G8` Gated DeltaNet (`16K/32V x 128`) | Decode (`M = 1`, `S_t -> S_t`) | `30.1 us` | `140.30 GB/s` | SLC-resident single-layer state |
| `G8` Gated DeltaNet loop isolation | `S_t -> S_tmp` eval / `S_tmp -> S_t` commit | `76.5 us` / `27.2 us` | `stmp_isolated = true` | `S_t` byte-identical before commit |
| `G9` QSA (`16Q/4KV x 128`, `rot = 64`) | `[11,11,10]` MRoPE + YaRN (`M = 1`) | `16.8 us` | — | Interleaved 3D MRoPE |
| `G9` QSA `full_chunk_512` (`kv = 512`) | `r = 0` index+attn vs `r >= 1` reuse | `38.7 us` -> `25.0 us` | `1.55x` speedup | Skips `ami_qsa_index_blocks` on `r >= 1` |
| `G9` QSA `sliding_window_128` (`kv = 128`) | `r = 0` index+attn vs `r >= 1` reuse | `77.7 us` -> `25.6 us` | `3.04x` speedup | Circular 128-token `HybridKvRing` |
| `G10` 4-branch GR (`H = 2048`) | `M = 1` vs `M = 16` full cycle | `31.5 us` / `23.1 us` | `4.56` / `67.44 GB/s` | Launch-bound at `M = 1`; amortized at `M = 16` |
| `G7` 512-expert MoE (`top-10 + shared`, `217 MiB`) | `M = 1` vs `M = 16` (Zipf CSR) | `239.1 us` / `850.7 us` | `22.22` / `13.60 GB/s` | `10` / `24` unique experts (`6.67x` CSR dedup) |
| `C10` 4-shard NVMe PLE (`8,192 x 256`, `144 KiB`) | Cold `F_NOCACHE` p50/p99 vs wired LRU hit | `1.54 / 2.04 us` vs `0.17 us` | `r >= 1` NVMe reads: `0` / `32` reuses | Fresh small file hits drive DRAM cache; multi-GiB uncached SSD reads remain in the `0.4-1.2 ms` range observed in M5 `C6` |
| End-to-end `scale-qwen4-512e` (`98 MiB` wired) | Serial vs `MTP` (`Placement.cpu_sme2`) | `193.66` vs `285.32 tok/s` | `tau = 1.000`, `T = 0` bit-exact | `3:1` GDN:QSA, 4-branch GR, `512E+shared`, `32` cold/L (`50` hits / `129` misses), 4-shard PLE (`0` `r >= 1` NVMe reads) |

### 2. `D12` `SelfSpecRecurrent` sweep (`R = 1..4`, `43.88 MiB q4 Core` exceeding L2/SLC)

| Loops `R` | Serial `M = 1` (`ms/tok`) | Serial (`tok/s`) | vs `R = 1` | `verifyTree` `M = 4` / `8` / `12` / `16` (`ms`) | Raw verify break-even `M` | `SelfSpecRecurrent` `M = 8` (`tok/s`, `tau`) | `SelfSpecRecurrent` `M = 16` (`tok/s`, `tau`) | `T = 0` exact |
| ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `1` | `1.449` | `690.15` | `1.00x` | `1.55 / 2.44 / 3.45 / 4.34` | `1` | `588.98` (`2.50`) | `588.78` (`2.50`) | `true` |
| `2` | `1.900` | `526.39` | `0.76x` | `2.71 / 4.13 / 5.90 / 7.40` | `4` | `392.96` (`2.14`) | `386.34` (`2.14`) | `true` |
| `3` | `2.644` | `378.23` | `0.55x` | `3.54 / 5.50 / 8.23 / 10.11` | `4` | `190.45` (`1.25`) | `182.04` (`1.25`) | `true` |
| `4` | `3.322` | `301.00` | `0.44x` | `4.55 / 7.09 / 10.58 / 12.70` | `4` | `141.59` (`1.00`) | `132.88` (`1.00`) | `true` |

Trade-offs observed on M4 (`120 GB/s`):

- With a `43.88 MiB` `q4` Core exceeding M4's `16 MiB` P-core L2 + `8 MiB` SLC,
  serial `M = 1` decode slows down roughly linearly with `R` (`1.00x -> 0.76x ->
  0.55x -> 0.44x`), confirming the bandwidth-bound `~1/R` scaling noted in
  [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md).
- Batched `verifyTree` amortizes the `R` Core weight reads across candidate
  nodes: at `R = 4`, per-node verify time drops from `3.322 ms` (`M = 1`) to
  `1.138 ms` (`M = 4`, below the `1.449 ms` `R = 1` serial baseline) and
  `0.794 ms` (`M = 16`, `4.18x` higher raw node throughput than `M = 1` `R = 4`).
- End-to-end `SelfSpecRecurrent` matches serial `R`-loop `stepDecode`
  bit-for-bit at `T = 0` for all `R = 1..4` via `r = 0` shared KV across loop
  steps, with zero extra draft-model RAM. On untrained random weights, shallow
  `r = 1` predictions diverge from deep `Core x R` outputs as `R` increases
  (`tau = 2.50` at `R = 1`, `2.14` at `R = 2`, `1.25` at `R = 3`, `1.00` at
  `R = 4`), making end-to-end `SelfSpecRecurrent` slower than serial decode on
  random weights (`141.59` vs `301.00 tok/s` at `R = 4`). Real end-to-end gains
  require trained `LoopLM` checkpoints whose `r = 1` pass achieves `tau >= 3-4`
  against `Core x R`.

### 3. `D13` `Placement.cpu_sme2` vs `.gpu` on L2-resident draft projections

| Subgraph | Dtype | Shape (`N x K`) | Size (`KiB`) | Fits L2 | `M = 1` SME2 / GPU (`us`) | `M = 4` SME2 / GPU (`us`) | `M = 8` SME2 / GPU (`us`) | Winners (`M = 1 / 4 / 8`) |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `eagle3_fc_q4_2048x6144` | `q4` | `2048 x 6144` | `7,680.0` | `yes` | `117.2 / 203.4` (`1.74x`) | `475.2 / 524.5` (`1.10x`) | `429.5 / 533.5` (`1.24x`) | `cpu_sme2 / cpu_sme2 / cpu_sme2` |
| `eagle3_fc_f16_1024x3072` | `f16` | `1024 x 3072` | `6,144.0` | `yes` | `40.1 / 215.8` (`5.38x`) | `83.8 / 267.4` (`3.19x`) | `62.3 / 268.5` (`4.31x`) | `cpu_sme2 / cpu_sme2 / cpu_sme2` |
| `mtp_eh_proj_q4_2048x4096` | `q4` | `2048 x 4096` | `5,120.0` | `yes` | `79.0 / 240.4` (`3.04x`) | `401.4 / 386.1` (`0.96x`) | `219.4 / 384.0` (`1.75x`) | `cpu_sme2 / gpu / cpu_sme2` |
| `mtp_eh_proj_f16_1024x2048` | `f16` | `1024 x 2048` | `4,096.0` | `yes` | `21.0 / 226.5` (`10.81x`) | `70.9 / 231.5` (`3.27x`) | `44.8 / 225.6` (`5.04x`) | `cpu_sme2 / cpu_sme2 / cpu_sme2` |

`Placement.cpu_sme2` wins `11/12` L2-resident configurations (`1.10x` to `10.81x`
lower latency, `0` Metal command buffers), while `.gpu` wins
`mtp_eh_proj_q4_2048x4096` at `M = 4` (`386.1 us` vs `401.4 us`, `0.96x`),
confirming the need for runtime `ADR-0008` median calibration. For real
`qwen3-30b-a3b-eagle3.amf`, `fc` is `24 MiB` BF16 (above the `16 MiB` P-cluster
L2 cap), so `D13` rejects `cpu_sme2` and retains `.gpu` (`252.6 us` vs ANE
`1,222.5 us`).

### 4. Real-checkpoint `W17` hardware verification (24 GiB M4, `swapouts_delta = 0`)

1. **`Qwen3-30B-A3B` heat-guided `2.700b` GSQ + RCO (`qwen3-30b-a3b-gsq-rco.amf`) vs uniform `3-bit` baseline (`qwen3-30b-a3b-3bit.amf`)**:
   - **Footprint & validation**: `verify-amf` passes (`18,867` tensors, `9.599 GiB`
     wired vs `12.443 GiB` baseline, saving `2.844 GiB` / `-22.9%` wired RAM;
     built from `Qwen3-30B-A3B-3bit` with `--cold-experts-per-layer 0`, exact
     composition `bf16 = 193` norms, `q3_affine = 3,140` covering `embed_tokens`,
     `lm_head`, `192` attention projections, `48` routers, and the `966` hottest
     experts / `2,898` expert tensors kept lossless at `q3_affine`, and
     `q2_affine = 15,534` covering `5,178` low-heat wired experts transcoded
     with RCO; `0` `q4_affine` tensors and `0` cold NVMe experts).
   - **`bench-decode` (`seq_len = 512`, `16` steps)**: `9,927 MiB` wired vs
     `12,839 MiB` (`-2,912 MiB`, `-22.7%`), `1,190.7 MB/tok` vs `1,381.6 MB/tok`
     (`-13.8%`), `49.59 tok/s` (`20.164 ms/tok`, `0.390 ms/layer`,
     `swapouts_delta = 0`) vs `51.86 tok/s` (`19.282 ms/tok`, `0.402 ms/layer`)
     on uniform 1-bucket `3-bit`. The `-4.4%` throughput delta comes from
     scheme-bucketed MoE dispatch (`ami_moe_route_topk_bucketed` plus 2 active
     `{q2, q3}` scheme buckets, since `scheme_counts[q4_affine] == 0` and
     `encodeGateUpDownHetero` skips `q4`) and `1.456 ms` `lm_head` timing versus
     the single-bucket `q3` baseline.
   - **Bilingual EN/ZH output quality & `EAGLE-3` `spec-decode`**: coherent
     outputs on EN capital (`53.06 tok/s`: `Paris is the capital of France.`),
     ZH capital (`49.25 tok/s`: `北京是中国的首都。`), and 38-token EN M4 explanation
     (`48.93 tok/s`, `9,879 MiB` wired). Paired with `qwen3-30b-a3b-eagle3.amf`
     (`10,968 MiB` wired, L2 SME2 router `70.2 us` vs GPU `987.8 us`),
     `spec-decode` produces `38/38` bit-exact tokens at `tau = 1.682` and
     `38.66 tok/s` (slower than `48.93 tok/s` serial on this 38-token prompt
     because 2-bucket `{q2, q3}` `verify_ms = 31.736 ms` plus `draft_ms =
     5.646 ms` and `commit_sync = 5.794 ms` exceeds the `1.682x` acceptance
     factor; `swapouts_delta = 0`).

2. **`Qwen3-0.6B` `0x0101` `ZT0` `LUT16` SME2 (`qwen3-0.6b-lut16-sme2.amf`) vs affine `qwen3-0.6b-gsq-rco-sme2.amf` and `D8` `14B` draft**:
   - **Offline quantization trade-off**: on clustered or bimodal synthetic
     weights, `LUT16` reduces quantization MSE by `-44%` to `-62.5%` versus
     affine. On real `Qwen3-0.6B`, sampled MSE is `1.325x` higher than
     per-column RCO affine (`9.770e-06` vs `7.374e-06`) because `QuantParamKind
     = 1` (`0x0101`) shares one unscaled 16-entry `ZT0` codebook across 16 panel
     rows whose weight standard deviations span `1.5x` to `8.6x`.
   - **Hardware CPU SME2 decode & `D8` PEARL**: `verify-amf` reports
     `amf.lut16_count = 196` (`336.9 MiB`, `393 MiB` wired). CPU SME2 decode
     runs at `26.01-26.09 tok/s` (`38.33-38.44 ms/tok`, `0` malloc delta, `0`
     swapouts) versus `26.38-26.62 tok/s` for affine `qwen3-0.6b-gsq-rco-sme2.amf`
     (`-1.1%` to `-2.3%`), producing identical `8/8` EN capital token IDs and
     fluent ZH output. As an off-GPU CPU SME2 small draft for `Qwen3-14B-4bit`
     (`8,545 MiB` total wired, `swapouts_delta = 0`), `--pearl on` hides
     `1,879.5 ms` of CPU drafting (`draft_ms` `46.843 -> 13.246 ms`) without GPU
     contention (`verify_ms` `256.673 -> 238.194 ms`), reaching `14.12 tok/s`
     (`tau = 3.917`, `10/12` PEARL hits, `48/48` bit-exact) versus `11.89 tok/s`
     with `--pearl off` (`+18.8%`) and `10.67 tok/s` `14B` serial (`1.32x`).

### Verbatim raw outputs (`aminfer bench-m6`, ReleaseFast)

```text
=== Milestone M6 Benchmark Suite (Qwen3.8-Flash-Next + LoopSchedule + D12/D13) ===

--- 1. Hybrid-Sparse Operators (G8 GDN, G9 QSA, G10 4-Branch GR, G7 512E MoE) ---
g8.gdn.prefill_seq               256
g8.gdn.prefill_us                3269.5
g8.gdn.prefill_tok_per_s         78300.2
g8.gdn.prefill_gbs               3.23
g8.gdn.decode_m1_us              30.1
g8.gdn.decode_m1_gbs             140.30
g8.gdn.stmp_eval_us              76.5
g8.gdn.stmp_commit_us            27.2
g8.gdn.stmp_isolated             true
g9.qsa.mrope_yarn_us             16.8
g9.qsa.full512_r0_us             38.7
g9.qsa.full512_reuse_us          25.0
g9.qsa.full512_reuse_speedup     1.55x
g9.qsa.swa128_r0_us              77.7
g9.qsa.swa128_reuse_us           25.6
g9.qsa.swa128_reuse_speedup      3.04x
g10.gr.hidden                    2048
g10.gr.m1_us                     31.5
g10.gr.m1_gbs                    4.56
g10.gr.m16_us                    23.1
g10.gr.m16_gbs                   67.44
g7.moe512.experts                512 (top-10 + shared_expert)
g7.moe512.wired_mib              217
g7.moe512.m1_us                  239.1
g7.moe512.m1_gbs                 22.22
g7.moe512.m1_unique_experts      10
g7.moe512.m16_us                 850.7
g7.moe512.m16_gbs                13.60
g7.moe512.m16_unique_experts     24
g7.moe512.m16_dedup_ratio        6.67x

--- 2. C10 F_NOCACHE NVMe PLE Store & Loop Isolation (r >= 1) ---
c10.ple.shards                   4
c10.ple.total_rows               8192
c10.ple.embed_dim                256
c10.ple.wired_kib                144
c10.ple.cold_nvme_p50_us         1.54
c10.ple.cold_nvme_p99_us         2.04
c10.ple.wired_lru_hit_p50_us     0.17
c10.ple.loop_r0_nvme_reads       16
c10.ple.loop_r_ge1_nvme_reads    0
c10.ple.loop_r_ge1_reuses        32

--- 3. End-to-End scale-qwen4-512e (3:1 GDN:QSA, 4-Branch GR, 512E+Shared, PLE, Cold, MTP) ---
qwen4_512e.experts               512 (top-10 + shared, 32 cold/layer)
qwen4_512e.ple_shards            4
qwen4_512e.wired_mib             98
qwen4_512e.gen_tokens            16
qwen4_512e.serial_tok_per_s      193.66
qwen4_512e.mtp_tok_per_s         285.32
qwen4_512e.mtp_mean_accept_tau   1.000
qwen4_512e.mtp_placement         sme2
qwen4_512e.mtp_bit_exact_t0      true
qwen4_512e.cold_hits             50
qwen4_512e.cold_misses           129
qwen4_512e.ple_nvme_reads        16
qwen4_512e.ple_r_ge1_nvme_reads  0

--- 4. D12 SelfSpecRecurrent Sweep (R = 1, 2, 3, 4; Core > L2/SLC) ---
d12.core_layers                  6 (43.88 MiB q4 Core, 72 MiB total wired)
R   serial_ms   serial_tok/s  vs_R1   vM4_ms  vM8_ms  vM12_ms vM16_ms break_M  specM8_tok/s (tau)   specM16_tok/s (tau)  t0_exact
1      1.449        690.15   1.00x    1.55    2.44    3.45    4.34      1       588.98 (2.50)       588.78 (2.50)   true
2      1.900        526.39   0.76x    2.71    4.13    5.90    7.40      4       392.96 (2.14)       386.34 (2.14)   true
3      2.644        378.23   0.55x    3.54    5.50    8.23   10.11      4       190.45 (1.25)       182.04 (1.25)   true
4      3.322        301.00   0.44x    4.55    7.09   10.58   12.70      4       141.59 (1.00)       132.88 (1.00)   true

--- 5. D13 Placement.cpu_sme2 vs .gpu (L2-Resident Draft Projections) ---
subgraph                    dtype  N     K     M   w_kib    l2    sme2_us   gpu_us  gpu/sme  winner
eagle3_fc_q4_2048x6144      q4     2048  6144  1  7680.0  yes      117.2    203.4    1.74x  cpu_sme2
eagle3_fc_q4_2048x6144      q4     2048  6144  4  7680.0  yes      475.2    524.5    1.10x  cpu_sme2
eagle3_fc_q4_2048x6144      q4     2048  6144  8  7680.0  yes      429.5    533.5    1.24x  cpu_sme2
eagle3_fc_f16_1024x3072     f16    1024  3072  1  6144.0  yes       40.1    215.8    5.38x  cpu_sme2
eagle3_fc_f16_1024x3072     f16    1024  3072  4  6144.0  yes       83.8    267.4    3.19x  cpu_sme2
eagle3_fc_f16_1024x3072     f16    1024  3072  8  6144.0  yes       62.3    268.5    4.31x  cpu_sme2
mtp_eh_proj_q4_2048x4096    q4     2048  4096  1  5120.0  yes       79.0    240.4    3.04x  cpu_sme2
mtp_eh_proj_q4_2048x4096    q4     2048  4096  4  5120.0  yes      401.4    386.1    0.96x  gpu
mtp_eh_proj_q4_2048x4096    q4     2048  4096  8  5120.0  yes      219.4    384.0    1.75x  cpu_sme2
mtp_eh_proj_f16_1024x2048   f16    1024  2048  1  4096.0  yes       21.0    226.5   10.81x  cpu_sme2
mtp_eh_proj_f16_1024x2048   f16    1024  2048  4  4096.0  yes       70.9    231.5    3.27x  cpu_sme2
mtp_eh_proj_f16_1024x2048   f16    1024  2048  8  4096.0  yes       44.8    225.6    5.04x  cpu_sme2

m6.swap.swapouts_delta           0
m6.swap.pageouts_delta           0
m6.swap.status                   PASS (no swap)
```

### Verbatim raw outputs (`qwen3-30b-a3b-gsq-rco.amf`, ReleaseFast)

```text
=== 1. verify-amf qwen3-30b-a3b-gsq-rco.amf ===
amf.status       ok
amf.model_id     qwen3-30b-a3b-gsq-rco
amf.tensors      18867
amf.file_bytes   10309206016
amf.wired_bytes  10306764800
amf.cold_bytes   0

=== 2. bench-decode qwen3-30b-a3b-gsq-rco.amf (seq=512, steps=16) ===
decode_bench.target              qwen3_moe
decode_bench.arch                qwen3_moe (q3)
decode_bench.seq_len             512
decode_bench.shards              1
decode_bench.wired_mib           9927
decode_bench.load_ms             26629.8
decode_bench.measured_layers     48
decode_bench.measured_ms_per_tok 20.164
decode_bench.measured_tok_per_s  49.59
decode_bench.measured_layer_ms   0.390
decode_bench.measured_head_ms    1.456
decode_bench.measured_bytes_tok  1190695808
decode_bench.measured_beta_gbs   59.05 (49.2% bus)
decode_bench.scaled_layers       48 (linear: 48 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   20.164
decode_bench.scaled_tok_per_s    49.59
decode_bench.scaled_bytes_tok    1190695808
decode_bench.scaled_beta_gbs     59.05 (49.2% bus)
decode_bench.malloc_blocks_delta 34
decode_bench.malloc_bytes_delta  2400
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      12

=== 3. decode qwen3-30b-a3b-gsq-rco.amf (EN ChatML) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    48
decode.shards                    1
decode.wired_mib                 9879
decode.prompt_tokens             25
decode.generated_tokens          8
decode.prefill_ms                658.443
decode.decode_ms_per_tok         18.846
decode.decode_tok_per_s          53.06
decode.bytes_per_tok             1143509888
decode.effective_gbs             60.68
decode.malloc_blocks_delta       12
decode.malloc_bytes_delta        544
decode.swapouts_delta            0
decode.hetero_moe                yes
decode.token_ids                 [59604, 374, 279, 6722, 315, 9625, 13, 151645]
decode.text                      Paris is the capital of France.<|im_end|>

=== 4. decode qwen3-30b-a3b-gsq-rco.amf (ZH ChatML) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    48
decode.shards                    1
decode.wired_mib                 9879
decode.prompt_tokens             22
decode.generated_tokens          6
decode.prefill_ms                727.423
decode.decode_ms_per_tok         20.303
decode.decode_tok_per_s          49.25
decode.bytes_per_tok             1143018368
decode.effective_gbs             56.30
decode.malloc_blocks_delta       8
decode.malloc_bytes_delta        288
decode.swapouts_delta            0
decode.hetero_moe                yes
decode.token_ids                 [68990, 105196, 9370, 106114, 1773, 151645]
decode.text                      北京是中国的首都。<|im_end|>

=== 5. decode qwen3-30b-a3b-gsq-rco.amf (EN longer 48-tok) ===
decode.arch                      qwen3_moe (q3)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    48
decode.shards                    1
decode.wired_mib                 9879
decode.prompt_tokens             29
decode.generated_tokens          38
decode.prefill_ms                729.307
decode.decode_ms_per_tok         20.438
decode.decode_tok_per_s          48.93
decode.bytes_per_tok             1146852224
decode.effective_gbs             56.11
decode.malloc_blocks_delta       72
decode.malloc_bytes_delta        4384
decode.swapouts_delta            0
decode.hetero_moe                yes
decode.token_ids                 [26567, 386, 19, 42690, 4938, 7567, 444, 10994, 44378, 553, 8241, 11050, 11, 3347, 98414, 2251, 2615, 311, 279, 2567, 54480, 4963, 13, 576, 42690, 4938, 17646, 25351, 46870, 17590, 323, 22823, 5068, 369, 44378, 9079, 13, 151645]
decode.text                      Apple M4 unified memory benefits LLM inference by providing efficient, low-latency access to the required computational resources. The unified memory architecture ensures seamless integration and optimal performance for inference tasks.<|im_end|>

=== 6. spec-decode qwen3-30b-a3b-gsq-rco.amf + qwen3-30b-a3b-eagle3.amf ===
spec.arch                        qwen3_moe (q3)
spec.mode                        eagle3
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       1222.5
spec.placement_gate_gpu_us       252.6
spec.router_pred_backend         sme2 (requested=auto)
spec.router_pred_cal             sme2_us=70.2 gpu_us=987.8 bank_kib=4096.0
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      48
spec.shards                      1
spec.wired_mib                   10968
spec.prompt_tokens               29
spec.generated_tokens            38
spec.rounds                      22
spec.mean_accept_len             1.682
spec.accept_hist                 [1:11, 2:7, 3:4]
spec.drafted_nodes_per_round     10.59
spec.verified_nodes_per_round    2.36
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    13.64
spec.prefill_ms                  249.893
spec.decode_ms_per_tok           25.868
spec.decode_tok_per_s            38.66
spec.draft_ms_per_round          5.646
spec.prune_ms_per_round          0.327
spec.verify_ms_per_round         31.736
spec.commit_sync_ms_per_round    5.794
spec.bytes_per_accepted_tok      933317462
spec.effective_gbs               36.08
spec.malloc_blocks_delta         201
spec.malloc_bytes_delta          22880
spec.prefill_swapouts_delta      1148
spec.swapouts_delta              0
spec.hetero_moe                  yes
spec.token_ids                   [26567, 386, 19, 42690, 4938, 7567, 444, 10994, 44378, 553, 8241, 11050, 11, 3347, 98414, 2251, 2615, 311, 279, 2567, 54480, 4963, 13, 576, 42690, 4938, 17646, 25351, 46870, 17590, 323, 22823, 5068, 369, 44378, 9079, 13, 151645]
spec.text                        Apple M4 unified memory benefits LLM inference by providing efficient, low-latency access to the required computational resources. The unified memory architecture ensures seamless integration and optimal performance for inference tasks.<|im_end|>
```

### Verbatim raw outputs (`qwen3-0.6b-lut16-sme2.amf` & `D8` `14B`, ReleaseFast)

```text
=== 1. verify-amf qwen3-0.6b-lut16-sme2.amf ===
amf.status       ok
amf.model_id     qwen3-0.6b-lut16-sme2
amf.tensors      310
amf.file_bytes   353288192
amf.wired_bytes  353222656
amf.cold_bytes   0
amf.lut16_count  196

=== 2. decode qwen3-0.6b-lut16-sme2.amf (EN ChatML, CPU SME2) ===
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 393
decode.prompt_tokens             25
decode.generated_tokens          8
decode.prefill_ms                869.118
decode.decode_ms_per_tok         38.441
decode.decode_tok_per_s          26.01
decode.bytes_per_tok             355279424
decode.effective_gbs             9.24
decode.malloc_blocks_delta       0
decode.malloc_bytes_delta        0
decode.swapouts_delta            0
decode.sme2_weights              yes
decode.token_ids                 [785, 6722, 315, 9625, 374, 12095, 13, 151645]
decode.text                      The capital of France is Paris.<|im_end|>

=== 3. decode qwen3-0.6b-lut16-sme2.amf (ZH ChatML, CPU SME2) ===
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 393
decode.prompt_tokens             26
decode.generated_tokens          23
decode.prefill_ms                881.424
decode.decode_ms_per_tok         38.330
decode.decode_tok_per_s          26.09
decode.bytes_per_tok             357114432
decode.effective_gbs             9.32
decode.malloc_blocks_delta       0
decode.malloc_bytes_delta        0
decode.swapouts_delta            0
decode.sme2_weights              yes
decode.token_ids                 [26567, 386, 19, 43589, 103967, 111390, 70500, 17714, 102202, 5373, 99464, 33108, 30440, 106375, 9370, 42140, 43268, 38507, 75117, 104257, 100143, 1773, 151645]
decode.text                      Apple M4 的统一内存设计为高效、安全和可扩展的多线程执行提供了支持。<|im_end|>

=== 4. decode qwen3-0.6b-gsq-rco-sme2.amf (EN ChatML, CPU SME2 affine baseline) ===
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 393
decode.prompt_tokens             25
decode.generated_tokens          8
decode.prefill_ms                877.373
decode.decode_ms_per_tok         37.568
decode.decode_tok_per_s          26.62
decode.bytes_per_tok             355279424
decode.effective_gbs             9.46
decode.malloc_blocks_delta       0
decode.malloc_bytes_delta        0
decode.swapouts_delta            0
decode.sme2_weights              yes
decode.token_ids                 [785, 6722, 315, 9625, 374, 12095, 13, 151645]
decode.text                      The capital of France is Paris.<|im_end|>

=== 5. decode qwen3-0.6b-gsq-rco-sme2.amf (ZH ChatML, CPU SME2 affine baseline) ===
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 393
decode.prompt_tokens             26
decode.generated_tokens          32
decode.prefill_ms                890.987
decode.decode_ms_per_tok         37.903
decode.decode_tok_per_s          26.38
decode.bytes_per_tok             358146624
decode.effective_gbs             9.45
decode.malloc_blocks_delta       -1
decode.malloc_bytes_delta        -96
decode.swapouts_delta            0
decode.sme2_weights              yes
decode.token_ids                 [26567, 386, 19, 9370, 103967, 111390, 70500, 100661, 101321, 41146, 101910, 42140, 43268, 38507, 106379, 3837, 67338, 44063, 20074, 105653, 18493, 106215, 9370, 111390, 99922, 15946, 3837, 101982, 104656, 100341, 100768, 101991]
decode.text                      Apple M4的统一内存设计优势在于其采用多线程架构，通过将数据存储在单一的内存块中，从而显著提升计算效率

=== 6. decode qwen3-0.6b-bf16.amf (ZH ChatML, GPU BF16 reference) ===
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1195
decode.prompt_tokens             26
decode.generated_tokens          26
decode.prefill_ms                325.632
decode.decode_ms_per_tok         16.069
decode.decode_tok_per_s          62.23
decode.bytes_per_tok             1198065664
decode.effective_gbs             74.56
decode.malloc_blocks_delta       50
decode.malloc_bytes_delta        3200
decode.swapouts_delta            0
decode.token_ids                 [26567, 386, 19, 9370, 103967, 111390, 70500, 67338, 44063, 35568, 111390, 33108, 100519, 24360, 106201, 3837, 108456, 20074, 54542, 101991, 33108, 72448, 102808, 101149, 1773, 151645]
decode.text                      Apple M4的统一内存设计通过将主内存和辅存分离，提升了数据处理效率和系统响应速度。<|im_end|>

=== 7. D8 spec-decode Qwen3-14B-4bit + qwen3-0.6b-lut16-sme2.amf (PEARL off) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       664.8
spec.placement_gate_gpu_us       62.7
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      40
spec.shards                      1
spec.wired_mib                   8545
spec.prompt_tokens               25
spec.generated_tokens            48
spec.rounds                      12
spec.mean_accept_len             3.917
spec.accept_hist                 [2:3, 3:2, 5:7]
spec.drafted_nodes_per_round     19.25
spec.verified_nodes_per_round    9.00
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  848.059
spec.decode_ms_per_tok           84.114
spec.decode_tok_per_s            11.89
spec.draft_ms_per_round          46.843
spec.prune_ms_per_round          0.002
spec.verify_ms_per_round         256.673
spec.commit_sync_ms_per_round    25.929
spec.bytes_per_accepted_tok      2011729130
spec.effective_gbs               23.92
spec.malloc_blocks_delta         48
spec.malloc_bytes_delta          3072
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.pearl_mode                  off (enabled=no, own_queue=no)
spec.pearl_rounds                0
spec.pearl_hits                  0
spec.pearl_partial_prefix_hits   0
spec.pearl_rollbacks             0
spec.pearl_saved_ms              0.000
spec.pearl_worker_ms             0.000
spec.token_ids                   [8420, 594, 264, 63594, 13027, 729, 429, 5169, 3070, 25891, 2711, 334, 389, 264, 3070, 28298, 1140, 334, 1447, 73594, 12669, 198, 750, 7868, 10716, 10939, 11, 2169, 982, 262, 2115, 11, 1290, 284, 220, 15, 11, 2422, 10939, 8, 481, 220, 16, 198, 262, 1393, 2115, 2651]
spec.text                        Here's a concise Python function that implements **binary search** on a **sorted list**:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <=

=== 8. D8 spec-decode Qwen3-14B-4bit + qwen3-0.6b-lut16-sme2.amf (PEARL on) ===
spec.arch                        qwen3 (q4)
spec.mode                        small
spec.placement                   gpu (requested=auto)
spec.placement_gate_ane_us       691.1
spec.placement_gate_gpu_us       31.0
spec.pruning                     enabled (cap=16, depth=4, width=2)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      40
spec.shards                      1
spec.wired_mib                   8545
spec.prompt_tokens               25
spec.generated_tokens            48
spec.rounds                      12
spec.mean_accept_len             3.917
spec.accept_hist                 [1:1, 2:1, 3:2, 4:2, 5:6]
spec.drafted_nodes_per_round     15.83
spec.verified_nodes_per_round    6.75
spec.l1_seed_rounds              1
spec.unique_experts_per_layer    0.00
spec.prefill_ms                  919.325
spec.decode_ms_per_tok           70.837
spec.decode_tok_per_s            14.12
spec.draft_ms_per_round          13.246
spec.prune_ms_per_round          0.001
spec.verify_ms_per_round         238.194
spec.commit_sync_ms_per_round    25.998
spec.bytes_per_accepted_tok      2011737934
spec.effective_gbs               28.40
spec.malloc_blocks_delta         48
spec.malloc_bytes_delta          3072
spec.prefill_swapouts_delta      0
spec.swapouts_delta              0
spec.pearl_mode                  on (enabled=yes, own_queue=yes)
spec.pearl_rounds                12
spec.pearl_hits                  10
spec.pearl_partial_prefix_hits   1
spec.pearl_rollbacks             2
spec.pearl_saved_ms              1879.548
spec.pearl_worker_ms             2635.331
spec.token_ids                   [8420, 594, 264, 63594, 13027, 729, 429, 5169, 3070, 25891, 2711, 334, 389, 264, 3070, 28298, 1140, 334, 1447, 73594, 12669, 198, 750, 7868, 10716, 10939, 11, 2169, 982, 262, 2115, 11, 1290, 284, 220, 15, 11, 2422, 10939, 8, 481, 220, 16, 198, 262, 1393, 2115, 2651]
spec.text                        Here's a concise Python function that implements **binary search** on a **sorted list**:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <=
```

---

## M6.5 continuous batching, long-context KV tiering, shared prefix cache, and zero-copy multimodal ingress

Captured on 2026-11-22 on Apple M4 24 GiB (macOS 27.2, `ReleaseFast`, commits `9f1efa7..990d2a4`, `18,186 MiB` wired budget, `swapouts_delta = 0` across all runs). Capabilities verified: `R5`, `R8`, `G14`, and `C14` ([ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md), [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md)).

### Summary tables

#### 1. `G14` hardware VideoToolbox to zero-copy IOSurface and Metal 4 vision ingress (`448x448`, `256` merged tokens)

| Path | HW VT | Zero-copy | 16 KiB aligned | VT decode (`us`) | GPU vision + MRoPE (`us`) | Total (`us`) | Throughput (`fps`) | GPU traffic (`GB/s`) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vt_h264_nv12` | yes | yes | yes | 591.8 | 9,355.3 | 9,947.1 | 100.5 | 0.60 |
| `vt_hevc_nv12` | yes | yes | yes | 461.3 | 6,816.3 | 7,277.6 | 137.4 | 0.83 |
| `vt_h264_bgra8` | yes | yes | yes | 805.0 | 6,513.3 | 7,318.3 | 136.6 | 0.94 |
| `vt_hevc_bgra8` | yes | yes | yes | 883.8 | 6,598.6 | 7,482.5 | 133.6 | 0.93 |

End-to-end `qwen4_exp` (`synth-qwen4 --preset tiny-qwen4`) with `--vt-test-video hevc` decodes 2 `128x128` NV12 frames via hardware VideoToolbox (`zero_copy = yes`, `page16k = yes`), produces `32` merged visual tokens (`grid = 2x4x4`, `vision_encode_us = 2,400.38 us`), advances interleaved 3D MRoPE to `vision_next_mrope_pos = 25` (`21` prefix text tokens + `max(2, 4, 4)`), and decodes at `865.82 tok/s` (`swapouts_delta = 0`). The patch embedding linear projection (`2352 -> 1024`) over `256` tokens is compute-bound on the M4 GPU, which accounts for the sub-1 GB/s reported weight/activation memory traffic metric.

#### 2. `R5` 32,768-token retired group-64 `q8` KV tier (`Qwen3-30B-A3B` geometry, 1 layer) and real `Qwen3-0.6B` BF16 decode

| Benchmark | Pure `fp16` IOSurface | Tiered `q8` retired + active `fp16` | Delta / Notes |
| --- | --- | --- | --- |
| 32K 1-layer wired KV (`q=32, kv=4, d=128`) | `64.00 MiB` (`64` tiles) | `35.88 MiB` (`4.00` active + `31.88` retired `q8`) | `-28.13 MiB/layer` (`-43.95%`), retire `22.04 us/chunk` |
| 32K decode attention | `1,143.17 us` (`58.70 GB/s`) | `1,076.40 us` (`34.95 GB/s`) | `-5.8%` latency, `max_abs_err = 0.000006` |
| 32K `M = 16` tree verify attention | `13,403.57 us` (`5.01 GB/s`) | `11,421.67 us` (`3.29 GB/s`) | `-14.8%` latency, `max_abs_err = 0.000006` |
| Real `Qwen3-0.6B` BF16 (`1,358` prompt tok, `--kv-retire-q8-after-chunks 1`) | `1,307 MiB` wired, `62.74 tok/s`, `84.75 GB/s`, `0` malloc delta | `1,255 MiB` wired (`active=56.00`, `retired_q8=59.50`), `65.12 tok/s`, `84.38 GB/s`, `0` malloc delta | `-52 MiB` wired (`-31.25%` KV at 3 chunks), `+3.8%` tok/s, `25/25` identical `token_ids` |

Trade-offs: `R5` lowers effective GB/s during attention because group-64 `q8` dequantization adds arithmetic per element and introduces a small numerical error (`max_abs_err = 0.000006`), while cutting per-layer wired KV bytes by `43.95%` at 32K tokens (`46.875%` per retired chunk) and preserving exact greedy output tokens on the 1,358-token `Qwen3-0.6B` test prompt.

#### 3. `R8` shared `PrefixCache`, GPU CoW partial tail, and `GDN` snapshot clone

| Measurement | Uncached baseline | Cached (`PrefixCache`) | Speedup / Savings |
| --- | --- | --- | --- |
| `bench-m65` exact `1,024`-tok prefix + `16`-tok suffix | `26.660 ms` | `3.500 ms` (`matched=1024, cow=0`) | `7.62x` faster prefill |
| `bench-m65` partial `1,200`-tok prefix + `16`-tok suffix | `44.669 ms` | `1.407 ms` (`matched=1200, cow=1`, `bit_exact=yes`) | `31.76x` faster prefill (`cow_tail_176tok = 101.80 us`) |
| `bench-m65` wired KV saved (`1,024`-tok shared prefix) | — | `4` slots: `12.00 MiB`, `8` slots: `28.00 MiB` (`4L` synth) | `Qwen3-30B-A3B` 48L, `8` slots: `672.0 MiB` saved |
| `bench-m65` 36-layer `GDN` snapshot clone (`73.69 MiB`) | — | `2,100.15 us` (`73.58 GB/s`, `bit_exact=yes`) | Single GPU dispatch (`ami_gdn_clone_state`) |
| Live `aminfer serve` (`qwen3-0.6b-gsq-rco.amf`, `~728`-tok prompt, `4` concurrent clients) | Cold 1-client TTFT `566.75 ms` | Warm 4-client TTFT p50 `63.93 ms`, p99 `83.91 ms`, `297.49 agg tok/s` | `8.87x` lower TTFT p50 (`2,849` prefix hit tokens across `4` clients, `0` swapouts) |

Trade-offs: prefix caching reserves additional wired KV tiles for cached entries (raising `qwen3-0.6b-gsq-rco.amf` wired RAM from `2,361 MiB` to `3,257 MiB` with `--prefix-cache-chunks 16`) and incurs a GPU copy-on-write pass (`101.80 us` for 176 tokens across 4 layers) whenever a shared prefix does not end on a 512-token tile boundary.

#### 4. `C14` continuous batching (`B = 1..8`) and `DomainNgramRegistry` multi-pack `.ang` routing

Synthetic `bench-m65` scaling (`B = 1, 2, 4, 8`):

| Backend | `B = 1` agg (`tok/s`) | `B = 2` agg (`tok/s`) | `B = 4` agg (`tok/s`) | `B = 8` agg (`tok/s`) | `B = 8` scaling |
| --- | ---: | ---: | ---: | ---: | ---: |
| GQA + MoE `q4` (`Qwen3BatchRuntime`) | 1,413.88 | 2,411.58 (`1.71x`) | 4,469.23 (`3.16x`) | 7,052.57 | `4.99x` |
| Hybrid `GDN + QSA + MoE q4` (`GraphBatchRuntime`) | 1,297.61 | 2,376.49 (`1.83x`) | 3,723.90 (`2.87x`) | 8,752.01 | `6.74x` |

Live `aminfer serve` (`--batch-mode serial` vs `--batch-mode continuous`, `--max-seq-len 2048 --max-active-requests 8`, `swapouts_delta = 0`, `D11` explicit `seed = 12345` at `T = 0.7` and `T = 0` greedy bit-exact across serial and continuous):

| Model | Concurrency | Serial agg (`tok/s`) | Continuous agg (`tok/s`) | Agg speedup | Serial TTFT p50 / p99 (`ms`) | Continuous TTFT p50 / p99 (`ms`) | Serial TPOT p50 (`ms`) | Continuous TPOT p50 (`ms`) | Max batch |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `Qwen3-0.6B` BF16 (`24` tok/req) | 1 | 38.92 | 50.60 | `1.30x` | 264.15 / 264.15 | 25.22 / 25.22 | 14.53 | 19.50 | 1 |
| `Qwen3-0.6B` BF16 (`24` tok/req) | 4 | 38.56 | 201.76 | `5.23x` | 1,410.36 / 2,126.75 | 83.18 / 104.62 | 15.75 | 17.94 | 4 |
| `Qwen3-0.6B` BF16 (`24` tok/req) | 8 | 40.07 | 304.35 | `7.60x` | 2,632.84 / 4,460.87 | 131.30 / 196.59 | 14.36 | 22.64 | 8 |
| `Qwen3-0.6B` GSQ+RCO (`32` tok/req) | 1 | 132.45 | 181.64 | `1.37x` | 90.80 / 90.80 | 16.64 / 16.64 | 4.72 | 5.14 | 1 |
| `Qwen3-0.6B` GSQ+RCO (`32` tok/req) | 4 | 139.06 | 426.39 | `3.07x` | 536.90 / 765.71 | 47.33 / 69.60 | 4.64 | 8.45 | 4 |
| `Qwen3-0.6B` GSQ+RCO (`32` tok/req) | 8 | 131.68 | 518.75 | `3.94x` | 1,035.62 / 1,786.04 | 83.01 / 130.08 | 4.95 | 13.72 | 8 |

Trade-offs:
- Continuous batching increases per-slot TPOT under load (`14.36 -> 22.64 ms` on BF16 and `4.95 -> 13.72 ms` on GSQ+RCO at `C = 8`) because each decode step computes `M = 8` projections and attention across 8 active KV sequences, and increases preallocated wired KV pool size (`1,363 -> 3,189 MiB` on BF16; `535 -> 2,361 MiB` on GSQ+RCO at `max_seq_len = 2048, max_active_requests = 8`). In exchange, aggregate token throughput rises `3.94x–7.60x` and queue-induced TTFT p50/p99 drops `12.5x–22.7x`.
- Zig scheduler, HTTP/SSE, and `DomainNgramRegistry` hot paths perform `0` heap allocations (`ngram.steady_malloc_blocks = 0` in `bench-m65`, `0` net blocks in `http_sse.zig` continuous-batching and endpoint tests, and `decode.malloc_blocks_delta = 0` on `Qwen3-0.6B` after 1,358 prefill frames). During short Metal 4 serving bursts (`< ~1,000` GPU commits), macOS 26.0 Metal 4 driver-internal `64 B` tracking slabs grow by `~2` blocks/frame (`206–1,596` blocks in the `16`-request serve sweeps) until the driver pool stabilizes.
- Multi-pack `DomainNgramRegistry` built with real `/Volumes/Akari/models/amf/aminfer-repo.ang` (`128.02 MiB`, `879,949` entries) and `/Volumes/Akari/models/amf/aminfer-docs.ang` (`32.02 MiB`, `198,113` entries, `table_crc32 = 0xc1f2f6ed`): `540,672 B` (`528 KiB`) wired LRU cache, uncached `F_NOCACHE` lookup (`verify_table_crc = false`, key probe also opened via `F_NOCACHE` to avoid warming XNU UBC) `p50 = 257.88 us` / `p99 = 418.92 us`, LRU cache hit `0.05 us`, `selectDefaultDomain` `3.6 ns`, live atomic `swapDomainPack` `305.75 us`, `steady_malloc_blocks = 0`. Note that because the benchmark extracts sample keys from the same 32 `.ang` buckets immediately prior to timing, `257.88–418.92 us` can hit external-SSD controller DRAM rather than unaccessed NAND flash (`396–1,208 us` in M5 cold external-SSD queries). Live `aminfer serve` verified `GET /v1/ngram-packs`, per-request `"ngram_domain"` routing (`repo`, `docs`, and HTTP `400` on unknown `"nonexistent"`), and `POST /v1/ngram-packs/select`.

### Verbatim `aminfer bench-m65` output (`ReleaseFast`)

```text
=== M6.5 G14 Zero-Copy Hardware VideoToolbox + Metal 4 Vision Ingress ===
path             res      tokens  hw_vt  zero_copy  page16k  vt_us     gpu_us    total_us  fps      gpu_gbs  malloc_blk
vt_h264_nv12     448x448  256     yes    yes        yes      591.8     9355.3    9947.1    100.5    0.60     34
vt_hevc_nv12     448x448  256     yes    yes        yes      461.3     6816.3    7277.6    137.4    0.83     32
vt_h264_bgra8    448x448  256     yes    yes        yes      805.0     6513.3    7318.3    136.6    0.94     15
vt_hevc_bgra8    448x448  256     yes    yes        yes      883.8     6598.6    7482.5    133.6    0.93     -19

=== M6.5 R5 32K Retired Group-64 q8 KV Tier (1 Layer) ===
r5.geometry                      q=32 kv=4 d=128 chunk=512 seq_len=32768 (60 retired q8 + 4 active fp16)
r5.pure_fp16_mib_per_layer       64.00 (67108864 bytes)
r5.tiered_q8_mib_per_layer       35.88 (active=4.00 MiB, retired_q8=31.88 MiB, 37617664 bytes)
r5.wired_saved_per_layer         28.13 MiB (-43.95%)
r5.retire_per_chunk_us           22.04 (total 60 chunks: 1322.7 us)
r5.decode_fp16_vs_tiered_us      fp16=1143.17 us (58.70 GB/s) vs tiered_q8=1076.40 us (34.95 GB/s)
r5.tree_m16_fp16_vs_tiered_us    fp16=13403.57 us (5.01 GB/s) vs tiered_q8=11421.67 us (3.29 GB/s)
r5.max_abs_err                   decode=0.000006 tree=0.000006

=== M6.5 R8 Shared PrefixCache + GPU CoW Tail + GDN Snapshot Clone ===
r8.exact_prefix_1024tok           uncached=26.660 ms vs cached=3.500 ms (7.62x, matched=1024, cow=0)
r8.partial_prefix_1200tok         uncached=44.669 ms vs cached=1.407 ms (31.76x, matched=1200, cow=1, bit_exact=yes)
r8.cow_tail_176tok_us             101.80 us
r8.kv_wired_saved_mib            4_slots=12.00 MiB 8_slots=28.00 MiB (Qwen3-30B-48L 8_slots=672.0 MiB)
r8.gdn_snapshot_clone            layers=36 bytes=77266944 (73.69 MiB) clone_us=2100.15 (73.58 GB/s, bit_exact=yes)

=== M6.5 C14 Continuous Batching Scaling (GQA + MoE q4) ===
batch  step_ms   slot_tok_s  agg_tok_s   scaling_vs_b1  malloc_blk
B=1    0.707     1413.88     1413.88     1.00           16
B=2    0.829     1205.79     2411.58     1.71           16
B=4    0.895     1117.31     4469.23     3.16           272
B=8    1.134     881.57      7052.57     4.99           16

=== M6.5 C14 Continuous Batching Scaling (Hybrid GDN + QSA + MoE q4) ===
batch  step_ms   slot_tok_s  agg_tok_s   scaling_vs_b1  malloc_blk
B=1    0.771     1297.61     1297.61     1.00           15
B=2    0.842     1188.24     2376.49     1.83           16
B=4    1.074     930.97      3723.90     2.87           14
B=8    0.914     1094.00     8752.01     6.74           16

=== M6.5 C14 DomainNgramRegistry Multi-Pack Lookup & Atomic Hot-Swap ===
ngram.packs                      source=real-ang count=2 total_entries=1078062 wired_bytes=540672
ngram.cold_query_us              p50=257.88 p99=418.92
ngram.cache_hit_us               0.05
ngram.select_default_ns          3.6
ngram.atomic_swap_pack_us        305.75
ngram.domain_hits                repo=32 docs=32
ngram.steady_malloc_blocks       0

m65.swap.status                  ok (swapouts_delta=0, pageouts_delta=0)
```

### Verbatim `G14` end-to-end `qwen4_exp` hardware VideoToolbox HEVC decode (`ReleaseFast`)

```text
decode.arch                      graph (hybrid_gdn_qsa, moe_swiglu, gated_4branch, loop=single_pass R=1)
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    4
decode.shards                    1
decode.wired_mib                 7
decode.prompt_tokens             21
decode.generated_tokens          16
decode.prefill_ms                132.151
decode.decode_ms_per_tok         1.155
decode.decode_tok_per_s          865.82
decode.bytes_per_tok             1472576
decode.effective_gbs             1.27
decode.malloc_blocks_delta       -34
decode.malloc_bytes_delta        -2832
decode.swapouts_delta            0
decode.loop_total_steps             15
decode.loop_early_exit_tokens       0
decode.loop_gdn_scratch_evals       0
decode.loop_gdn_final_commits       45
decode.loop_ple_r0_fills            60
decode.loop_ple_r_ge1_reuses        0
decode.loop_ple_r_ge1_nvme_reads    0
decode.loop_qsa_index_computes      15
decode.loop_qsa_index_reuses        0
decode.loop_moe_r_ge1_pinned_routes 0
decode.loop_moe_r_ge1_cold_stalls   0
decode.loop_steplora_zero_skips     0
decode.loop_steplora_active_evals   0
decode.ple_nvme_reads            16
decode.ple_cache_hits            272
decode.vision_source             vt-hevc-nv12
decode.vision_frames             2 (128x128)
decode.vision_zero_copy          yes (page16k=yes)
decode.vision_tokens             32 (grid=2x4x4, total_prompt=53)
decode.vision_next_mrope_pos     25
decode.vision_encode_us          2400.38
decode.token_ids                 [170, 297, 421, 405, 493, 288, 472, 278, 157, 489, 114, 401, 493, 171, 444, 210]
decode.text                      � v000297 v000421 v000405 v000493� v000472en� v000489r v000401 v000493� v000444�
```

### Verbatim real `Qwen3-0.6B` BF16 `R5` pure `fp16` vs tiered `q8` decode (`1,358` prompt tokens, `ReleaseFast`)

Pure `fp16` baseline (`--max-tokens 32 --no-verify-crc`):

```text
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1307
decode.prompt_tokens             1358
decode.generated_tokens          25
decode.prefill_ms                15793.902
decode.decode_ms_per_tok         15.938
decode.decode_tok_per_s          62.74
decode.bytes_per_tok             1350715392
decode.effective_gbs             84.75
decode.malloc_blocks_delta       0
decode.malloc_bytes_delta        0
decode.swapouts_delta            0
decode.token_ids                 [271, 785, 8162, 37150, 2150, 1379, 264, 1065, 802, 662, 48796, 369, 19206, 323, 64080, 10509, 27389, 374, 220, 16, 21, 29458, 33, 13, 151645]
decode.text                      

The Apple Silicon page size aminfer enforces for Metal and IOSurface buffers is 16 KiB.<|im_end|>
```

Tiered `q8` retired KV (`--kv-retire-q8-after-chunks 1 --max-tokens 32 --no-verify-crc`):

```text
decode.arch                      qwen3 (bf16)
decode.prefill_mode              auto
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    28
decode.shards                    1
decode.wired_mib                 1255
decode.prompt_tokens             1358
decode.generated_tokens          25
decode.prefill_ms                15356.153
decode.decode_ms_per_tok         15.355
decode.decode_tok_per_s          65.12
decode.bytes_per_tok             1295665152
decode.effective_gbs             84.38
decode.malloc_blocks_delta       0
decode.malloc_bytes_delta        0
decode.swapouts_delta            0
decode.retired_kv_chunks         2 (events=2)
decode.kv_wired_mib              active=56.00 retired_q8=59.50
decode.token_ids                 [271, 785, 8162, 37150, 2150, 1379, 264, 1065, 802, 662, 48796, 369, 19206, 323, 64080, 10509, 27389, 374, 220, 16, 21, 29458, 33, 13, 151645]
decode.text                      

The Apple Silicon page size aminfer enforces for Metal and IOSurface buffers is 16 KiB.<|im_end|>
```

### Verbatim live `aminfer serve` concurrency sweeps (`ReleaseFast`)

`Qwen3-0.6B` BF16 client harness (`1/4/8` concurrent SSE clients, `24` tokens/client, `D11` explicit `seed = 12345`):

```text
qwen3-0.6b-bf16 mode=serial C=1 max_tok=24 wall_s=0.617 agg_tok_s=38.92 ttft_p50_ms=264.15 ttft_p99_ms=264.15 tpot_p50_ms=14.53 ok=1/1
qwen3-0.6b-bf16 mode=serial C=4 max_tok=24 wall_s=2.490 agg_tok_s=38.56 ttft_p50_ms=1410.36 ttft_p99_ms=2126.75 tpot_p50_ms=15.75 ok=4/4
qwen3-0.6b-bf16 mode=serial C=8 max_tok=24 wall_s=4.792 agg_tok_s=40.07 ttft_p50_ms=2632.84 ttft_p99_ms=4460.87 tpot_p50_ms=14.36 ok=8/8
qwen3-0.6b-bf16 mode=serial d11_seed=12345 T=0.7 seed_used=(12345,12345) exact_match=True
qwen3-0.6b-bf16 mode=continuous C=1 max_tok=24 wall_s=0.474 agg_tok_s=50.60 ttft_p50_ms=25.22 ttft_p99_ms=25.22 tpot_p50_ms=19.50 ok=1/1
qwen3-0.6b-bf16 mode=continuous C=4 max_tok=24 wall_s=0.476 agg_tok_s=201.76 ttft_p50_ms=83.18 ttft_p99_ms=104.62 tpot_p50_ms=17.94 ok=4/4
qwen3-0.6b-bf16 mode=continuous C=8 max_tok=24 wall_s=0.631 agg_tok_s=304.35 ttft_p50_ms=131.30 ttft_p99_ms=196.59 tpot_p50_ms=22.64 ok=8/8
qwen3-0.6b-bf16 mode=continuous d11_seed=12345 T=0.7 seed_used=(12345,12345) exact_match=True
qwen3-0.6b-bf16 serial_vs_continuous_t0_match=True d11_cross_mode_match=True
```

`Qwen3-0.6B` BF16 server stdout (`--batch-mode serial`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:50402
serve.model_id                   qwen3-0.6b-bf16
serve.arch                       qwen3 (bf16)
serve.batch_mode                 serial (max_batch_size=1)
serve.prefix_cache_chunks        0
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  1363
serve.max_conns                  32
serve.max_active_requests        8
serve.max_seq_len                2048
serve.status                     stopped
serve.total_requests             16
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    0
serve.mean_batch_size            0.00
serve.total_batch_steps          0
serve.prefix_hit_tokens          0
serve.prefix_hit_requests        0
serve.prefix_cow_copies          0
serve.kv_retire_events           0
serve.steady_malloc_blocks       1388
serve.steady_malloc_bytes        88608
serve.swapouts_delta             0
```

`Qwen3-0.6B` BF16 server stdout (`--batch-mode continuous`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:51066
serve.model_id                   qwen3-0.6b-bf16
serve.arch                       qwen3 (bf16)
serve.batch_mode                 continuous (max_batch_size=8)
serve.prefix_cache_chunks        0
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  3189
serve.max_conns                  32
serve.max_active_requests        8
serve.max_seq_len                2048
serve.status                     stopped
serve.total_requests             16
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    8
serve.mean_batch_size            3.69
serve.total_batch_steps          90
serve.prefix_hit_tokens          0
serve.prefix_hit_requests        0
serve.prefix_cow_copies          0
serve.kv_retire_events           0
serve.steady_malloc_blocks       206
serve.steady_malloc_bytes        13376
serve.swapouts_delta             0
```

`Qwen3-0.6B` GSQ+RCO client harness (`1/4/8` concurrent SSE clients, `32` tokens/client, `D11` explicit `seed = 12345`):

```text
qwen3-0.6b-gsq-rco mode=serial C=1 max_tok=32 wall_s=0.242 agg_tok_s=132.45 ttft_p50_ms=90.80 ttft_p99_ms=90.80 tpot_p50_ms=4.72 ok=1/1
qwen3-0.6b-gsq-rco mode=serial C=4 max_tok=32 wall_s=0.920 agg_tok_s=139.06 ttft_p50_ms=536.90 ttft_p99_ms=765.71 tpot_p50_ms=4.64 ok=4/4
qwen3-0.6b-gsq-rco mode=serial C=8 max_tok=32 wall_s=1.944 agg_tok_s=131.68 ttft_p50_ms=1035.62 ttft_p99_ms=1786.04 tpot_p50_ms=4.95 ok=8/8
qwen3-0.6b-gsq-rco mode=serial d11_seed=12345 T=0.7 seed_used=(12345,12345) exact_match=True
qwen3-0.6b-gsq-rco mode=continuous C=1 max_tok=32 wall_s=0.176 agg_tok_s=181.64 ttft_p50_ms=16.64 ttft_p99_ms=16.64 tpot_p50_ms=5.14 ok=1/1
qwen3-0.6b-gsq-rco mode=continuous C=4 max_tok=32 wall_s=0.300 agg_tok_s=426.39 ttft_p50_ms=47.33 ttft_p99_ms=69.60 tpot_p50_ms=8.45 ok=4/4
qwen3-0.6b-gsq-rco mode=continuous C=8 max_tok=32 wall_s=0.493 agg_tok_s=518.75 ttft_p50_ms=83.01 ttft_p99_ms=130.08 tpot_p50_ms=13.72 ok=8/8
qwen3-0.6b-gsq-rco mode=continuous d11_seed=12345 T=0.7 seed_used=(12345,12345) exact_match=True
qwen3-0.6b-gsq-rco serial_vs_continuous_t0_match=True d11_cross_mode_match=True
```

`Qwen3-0.6B` GSQ+RCO server stdout (`--batch-mode serial`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:51632
serve.model_id                   qwen3-0.6b-gsq-rco
serve.arch                       qwen3 (bf16)
serve.batch_mode                 serial (max_batch_size=1)
serve.prefix_cache_chunks        0
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  535
serve.max_conns                  32
serve.max_active_requests        8
serve.max_seq_len                2048
serve.status                     stopped
serve.total_requests             16
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    0
serve.mean_batch_size            0.00
serve.total_batch_steps          0
serve.prefix_hit_tokens          0
serve.prefix_hit_requests        0
serve.prefix_cow_copies          0
serve.kv_retire_events           0
serve.steady_malloc_blocks       1596
serve.steady_malloc_bytes        101920
serve.swapouts_delta             0
```

`Qwen3-0.6B` GSQ+RCO server stdout (`--batch-mode continuous`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:51734
serve.model_id                   qwen3-0.6b-gsq-rco
serve.arch                       qwen3 (bf16)
serve.batch_mode                 continuous (max_batch_size=8)
serve.prefix_cache_chunks        0
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  2361
serve.max_conns                  32
serve.max_active_requests        8
serve.max_seq_len                2048
serve.status                     stopped
serve.total_requests             16
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    8
serve.mean_batch_size            3.86
serve.total_batch_steps          113
serve.prefix_hit_tokens          0
serve.prefix_hit_requests        0
serve.prefix_cow_copies          0
serve.kv_retire_events           0
serve.steady_malloc_blocks       252
serve.steady_malloc_bytes        16320
serve.swapouts_delta             0
```

### Verbatim live `aminfer serve` `R8` `PrefixCache` and multi-pack `.ang` verification (`ReleaseFast`)

Client harness output (`~728`-tok shared system prompt cold miss vs warm 4-client hit, plus `/v1/ngram-packs`, `/v1/ngram-packs/select`, and per-request `"ngram_domain"`):

```text
r8.cold_miss prompt_tokens=728 ttft_ms=566.75 e2e_ms=699.96 tpot_ms=5.79
r8.warm_4_clients prompt_tokens=[731, 731, 729, 728] ttft_p50_ms=63.93 ttft_p99_ms=83.91 ttft_speedup=8.87x agg_tok_s=297.49
ngram.get_initial status=200 body={"object":"list","default_domain":"repo","pack_count":2,"cache_pages":32,"wired_bytes":540672,"data":[{"domain":"repo","corpus_id":"aminfer-repo-v1","is_default":true,"entries":879949,"buckets":8192,"min_ngram":2,"max_ngram":4,"generation":1,"swaps":0,"stats":{"queries":0,"hits":0,"seed_hits":0,"branch_hits":0,"cache_hits":0,"prefetch_submitted":0,"prefetch_hits":0,"nvme_reads":0,"hit_rate":0.0000}},{"domain":"docs","corpus_id":"aminfer-docs-v1","is_default":false,"entries":198113,"buckets":2048,"min_ngram":2,"max_ngram":4,"generation":2,"swaps":0,"stats":{"queries":0,"hits":0,"seed_hits":0,"branch_hits":0,"cache_hits":0,"prefetch_submitted":0,"prefetch_hits":0,"nvme_reads":0,"hit_rate":0.0000}}]}
ngram.req_default_repo status=200 tokens=8
ngram.req_explicit_docs status=200 tokens=8
ngram.req_unknown status=400 body={"error":{"message":"unknown ngram_domain","type":"invalid_request_error","code":"domain_not_found"}}
ngram.select_docs status=200 body={"status":"ok","default_domain":"docs"}
ngram.req_after_select_docs status=200 tokens=8
ngram.get_final status=200 body={"object":"list","default_domain":"docs","pack_count":2,"cache_pages":32,"wired_bytes":540672,"data":[{"domain":"repo","corpus_id":"aminfer-repo-v1","is_default":false,"entries":879949,"buckets":8192,"min_ngram":2,"max_ngram":4,"generation":1,"swaps":0,"stats":{"queries":1,"hits":0,"seed_hits":0,"branch_hits":0,"cache_hits":0,"prefetch_submitted":0,"prefetch_hits":0,"nvme_reads":3,"hit_rate":0.0000}},{"domain":"docs","corpus_id":"aminfer-docs-v1","is_default":true,"entries":198113,"buckets":2048,"min_ngram":2,"max_ngram":4,"generation":2,"swaps":0,"stats":{"queries":2,"hits":1,"seed_hits":0,"branch_hits":1,"cache_hits":0,"prefetch_submitted":0,"prefetch_hits":0,"nvme_reads":6,"hit_rate":0.5000}}]}
```

Server stdout (`R8` `PrefixCache`, `--prefix-cache-chunks 16`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:52283
serve.model_id                   qwen3-0.6b-gsq-rco
serve.arch                       qwen3 (bf16)
serve.batch_mode                 continuous (max_batch_size=8)
serve.prefix_cache_chunks        16
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  3257
serve.max_conns                  32
serve.max_active_requests        8
serve.max_seq_len                2048
serve.status                     stopped
serve.total_requests             5
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    4
serve.mean_batch_size            2.50
serve.total_batch_steps          46
serve.prefix_hit_tokens          2849
serve.prefix_hit_requests        4
serve.prefix_cow_copies          9
serve.kv_retire_events           0
serve.steady_malloc_blocks       116
serve.steady_malloc_bytes        8256
serve.swapouts_delta             0
```

Server stdout (multi-pack `.ang`, `--ngram-pack repo=... --ngram-pack docs=...`):

```text
serve.status                     listening
serve.url                        http://127.0.0.1:52340
serve.model_id                   qwen3-0.6b-gsq-rco
serve.arch                       qwen3 (bf16)
serve.batch_mode                 continuous (max_batch_size=4)
serve.prefix_cache_chunks        8
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                2
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 on
serve.wired_mib                  1352
serve.max_conns                  16
serve.max_active_requests        4
serve.max_seq_len                1024
serve.status                     stopped
serve.total_requests             3
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    1
serve.mean_batch_size            1.00
serve.total_batch_steps          21
serve.prefix_hit_tokens          7
serve.prefix_hit_requests        2
serve.prefix_cow_copies          5
serve.kv_retire_events           0
serve.steady_malloc_blocks       64
serve.steady_malloc_bytes        4512
serve.swapouts_delta             0
```

## M6.8: Wide-chunk GPU prefill, prompt_tail pruning, serving QoS, live-RAM admission and metallib canary

Measured on Apple M4 24 GiB (`macOS 27.2`, `18,186 MiB` static wired budget,
`ReleaseFast`, `swapouts_delta = 0` across every run including 3/3 consecutive
`bench-m68` runs) for
[ADR-0020](../decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md)
(`R9`, `G15`, `G16`, `C15`). All real-checkpoint benchmarks ran against the
converted `.amf` bundles under `/Volumes/Akari/models/amf/`.

### G16 startup `.metallib` numerical canary and `Apple10` (`NAX`) probe

| Metric | `aminfer probe` | `aminfer bench-m68` | Notes |
| --- | --- | --- | --- |
| `gpu.supports_apple10` | `no` | `no` | M4 reports `Apple9`; `Apple10` (`NAX`) dispatch hook reserved for M5+ hardware |
| `gpu.prefill_matmul_backend` | `simdgroup_apple9` | `simdgroup_apple9` | Dispatches `BM=64, BN=64` wide `simdgroup_matrix` kernels on M4 |
| Startup canary cold first-submit (`us`) | `2,132.5` | `2,223.5` | Cold first-submit wall time on fresh `MTLDevice` including Metal 4 queue/residency/AGX wake (`1,325.2–3,780.0 us` across `bench-decode`); runs once at load, never on hot path |
| Startup canary warm in-buffer (`us`) | `219.0` | `231.2` | Repeat `runStartupNumericalCanaryInBuffer` on same resident `16 KiB` buffer and warm `MTL4CommandQueue` (`187.0–237.3 us` across runs; covers 6 GPU GEMM dispatches + CPU fp32 verification) |
| `max_abs_err_f16` / `bf16` / `q4` | `0.000023` / `0.001953` / `0.000028` | `0.000023` / `0.001953` / `0.000028` | All three dtypes pass tolerance against CPU FP32 reference |
| Negative corrupted-slice rejection | n/a | `yes` | Corrupted output slice reliably triggers `error.MetallibNumericalCanaryFailed` |

### G15 and R9 wide-chunk GPU prefill, `prompt_tail` pruning and scratch aliasing A/B

Single-load A/B comparison in `aminfer bench-m68` (`max_tree_nodes = 64`, `lm_head` remains 100% wired; on this host snapshot `r9.live_reclaimable_mib = 9,624 MiB`, so the `R9` zero-swap guard automatically offloaded cold experts on the `30B-A3B` steps via `qwen3-30b-a3b-heat.json` while keeping `swapouts_delta = 0`):

| Model | Prompt tokens | Real checkpoint | `tree64` (ms) | `wide` unpruned (ms) | `wide` + `prompt_tail` (ms) | Speedup (`tree64 / wide`) | `wide` throughput (tok/s) | Max logit diff | Greedy match | Aliased scratch saved (MiB) | Wired (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b-bf16` | 512 | yes | 405.81 | 229.20 | 212.88 | 1.91x | 2,405.1 | 0.03516 | yes | 18.02 | 1,423 |
| `qwen3-0.6b-bf16` | 2,048 | yes | 1,880.96 | 1,102.51 | 1,092.75 | 1.72x | 1,874.2 | 0.05859 | yes | 18.02 | 1,423 |
| `qwen3-0.6b-gsq-rco` | 512 | yes | 556.59 | 216.66 | 206.86 | 2.69x | 2,475.2 | 0.02734 | yes | 18.02 | 595 |
| `qwen3-0.6b-gsq-rco` | 2,048 | yes | 2,481.34 | 1,150.43 | 1,066.57 | 2.33x | 1,920.2 | 0.03125 | yes | 18.02 | 595 |
| `qwen3-14b-4bit` | 512 | yes | 14,529.71 | 6,491.80 | 6,331.50 | 2.29x | 80.9 | 0.03125 | yes | 78.02 | 8,125 |
| `qwen3-30b-a3b-gsq-rco` | 512 | yes | 18,020.11 | 4,588.77 | 4,603.94 | 3.91x | 111.2 | 1.64063 | yes | 53.34 | 8,809 |
| `qwen4_exp-512e-R2` | 512 | no (synth) | 73.12 | 40.06 | 33.87 | 2.16x | 15,116.2 | 0.08008 | yes | 2.06 | 209 |

- **`EAGLE-3` wide prefill + aliased `verifyTree` transition (`qwen3-30b-a3b-gsq-rco` + `qwen3-30b-a3b-eagle3`, `prompt = 480`, `gen = 16`, `R9` live-RAM gated)**: `wide_prefill_ms = 8,605.43`, `spec_tok_s = 2.22`, `mean_accept = 2.50`, `alias_mib = 19.63`, `cold_exp = 66`, `wired_mib = 6,770`, `greedy_match = yes`, `malloc_blk = 0` (`R9` scaled `cold_exp` dynamically across 3/3 consecutive runs: `66` at `9,624 MiB`, `41` at `11,466 MiB` with `4.35 tok/s` / `8,387 MiB` wired, and `96` at `3,848 MiB` with `4,702 MiB` wired — `100%` `greedy_match = yes`, `malloc_blk = 0`, `infer_swapouts_delta = total_swapouts_delta = 0`; on an unpressured host where all `128` experts remain hot, the same `EAGLE-3` path runs at `~84.7 tok/s` and `~2,504 ms` prefill).

Fresh-process standalone `aminfer bench-decode` A/B (`--prefill-tokens 512 --decode-steps 8 --no-verify-crc`, default `max_tree_nodes = 16`, `100%` hot weights, `swapouts_delta = 0` on all 6 runs):

| Model | Mode | Prefill (ms) | Prefill (tok/s) | Prefill speedup | Wired (MiB) | Wired delta (MiB) | Aliased scratch saved (MiB) | Decode (tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b-gsq-rco` | `tree64` | 541.835 | 944.94 | 1.00x | 483 | 0 | 0.00 | 195.76 |
| `qwen3-0.6b-gsq-rco` | `wide` | 222.487 | 2,301.26 | 2.44x | 445 | -38 | 10.45 | 189.11 |
| `qwen3-14b-4bit` | `tree64` | 12,537.840 | 40.84 | 1.00x | 8,205 | 0 | 0.00 | 11.53 |
| `qwen3-14b-4bit` | `wide` | 5,604.319 | 91.36 | 2.24x | 8,168 | -37 | 25.27 | 10.36 |
| `qwen3-30b-a3b-gsq-rco` | `tree64` | 3,469.696 | 147.56 | 1.00x | 10,027 | 0 | 0.00 | 50.83 |
| `qwen3-30b-a3b-gsq-rco` | `wide` | 2,729.462 | 187.58 | 1.27x | 9,988 | -39 | 19.44 | 48.73 |

Honest trade-offs and observations:

- **Full-hot vs live-RAM-gated `30B-A3B` prefill & `EAGLE-3`**: when all `128` experts fit in wired RAM (`standalone bench-decode` and `bench-m68` Run 2 at `11,466 MiB` live reclaimable RAM), `qwen3-30b-a3b-gsq-rco` wide prefill speeds up `1.14x–1.28x` (`125.6–187.6 tok/s`, `9,979–9,988 MiB` wired) because sparse MoE gather dominates over dense GEMM. When desktop live reclaimable RAM dips below `10.6 GiB` (`bench-m68` Run 1 at `9,624 MiB`), the `R9` live-RAM guard offloads cold experts to NVMe (`8,809 MiB` wired); under cold-expert streaming, `wide` reduces NVMe passes by `8x` relative to `tree64` (`18,020.11 -> 4,603.94 ms`, `3.91x` faster), while `EAGLE-3` decode trades throughput (`2.22–4.35 tok/s` when `41–66` experts/layer stream over NVMe) to guarantee `0` XNU swapouts.
- **`qwen3-14b-4bit` thermal sensitivity**: inside single-process `bench-m68`, `wide` runs immediately after a `14.5 s` `tree64` GPU burn and settles at `6,331.50 ms` (`80.9 tok/s`, `2.29x`), whereas a fresh standalone `bench-decode` process reaches `5,604.32 ms` (`91.36 tok/s`, `2.24x` over `12,537.84 ms` `tree64`).
- **`prompt_tail` pruning magnitude**: pruning eliminates `O_proj` + FFN + `lm_head` work on layer `L - 1` (`~1/L` on a single 512-token chunk, plus skipped final-layer blocks on non-final chunks). Across the dense checkpoints (`28–40` layers), pruning saves `2.5–7.3%`; on `qwen4_exp-512e-R2` (4 layers, `R = 2`) it saves `15.5%` (`40.06 -> 33.87 ms`); on cold-expert `30B-A3B` (`48` layers) the `~2%` compute savings sits within NVMe I/O variance (`4,588.77` vs `4,603.94 ms`).
- **MoE routing sensitivity and decode variance**: on `qwen3-30b-a3b-gsq-rco`, different fp16/bf16 accumulation order in `BM=64, BN=64` GEMM flips one close top-k router boundary across 48 layers (`max_diff = 1.64063`), while greedy argmax remains identical (`greedy = yes`) for both base prefill and `EAGLE-3` speculative decode. Minor standalone decode differences (`195.76` vs `189.11`, `11.53` vs `10.36`, `50.83` vs `48.73 tok/s`) reflect thermal state right after prefill; decode kernels did not change.
- **Wired RAM savings without unloading `lm_head`**: in `wide` mode, eliminating per-node `tree_k`/`tree_v` copies and aliasing wide-prefill scratch with `verifyTree` scratch via `WiredArena.allocAliasedPair` reduces total wired memory by `37–39 MiB` on standalone `bench-decode` (`aliased_saved_mib = 10.45–25.27 MiB` at `max_tree_nodes = 16`; `18.02–78.02 MiB` at `max_tree_nodes = 64` in `bench-m68`).

### C15 serving QoS, adaptive prefill narrowing and ChatML message-boundary `PrefixCache`

| Benchmark | Mode / Phase | Foreground decode TPOT p50 (ms) | Foreground decode TPOT p99 (ms) | Background prefill TTFT (ms) | Boundary / QoS telemetry |
| --- | --- | --- | --- | --- | --- |
| `bench-m68` QoS (`0.6b-gsq-rco`, `bg = 10,240` tok, periodic 2-slot fg bursts) | `qos_off` (`decode_share = 0.0`, narrow off) | 562.94 | 987.28 | 12,432.5 | `qos_steps = d:0/p:0`, `narrowed_chunks = 0`, `samples = 42` |
| `bench-m68` QoS (`0.6b-gsq-rco`, `bg = 10,240` tok, periodic 2-slot fg bursts) | `qos_on` (`decode_share = 0.50`, narrow on) | 5.43 (`103.7x` lower) | 673.20 (`1.47x` lower) | 14,455.8 (`+16.3%`) | `qos_steps = d:270/p:6`, `narrowed_chunks = 3`, `samples = 576` |
| `bench-m68` ChatML boundary (`0.6b-gsq-rco`, `boundary_pos = 359`) | `boundary_off` vs `boundary_on` | n/a | n/a | `183.05 -> 28.74` (`6.37x` faster) | `hit_tokens = 0/383 -> 359/383`, `boundary_hits = 1`, `greedy_match = yes`, `evict_first = 1`, `canonical_512_survived = yes` |
| Live `aminfer serve` HTTP/SSE (`0.6b-gsq-rco`, default `--max-active-requests`, `--max-seq-len 10240 --prefix-cache-chunks 16`) | Multi-turn ChatML Turn 1 -> Turn 2 | n/a | n/a | `337.05 -> 30.48` (`11.06x` lower TTFT) | `boundary_checkpoints = 1`, `boundary_hits = 1`, `evict_first_evictions = 1` |
| Live `aminfer serve` HTTP/SSE (`0.6b-gsq-rco`, sustained `~9.4K` bg + 8 fg reqs) | `decode_share = 0.50`, narrow on | 4.83 | 413.62 | 11,925.20 | `max_batch_size = 2` (auto-clamped), `qos_steps = decode=210 prefill=4 narrowed_chunks=1`, `wired_mib = 3,534`, `swapouts_delta = 0` |

Honest trade-offs and scope notes:

- **Deep-context p99 cap vs background TTFT cost under QoS**: in `bench-m68`, re-seeding foreground bursts across the entire `10,240`-token background prefill verifies both `decode-share` scheduling (`qos_steps = d:270/p:6`, slashing median TPOT `103.7x` from `562.94` to `5.43 ms`) and depth-scaled adaptive chunk narrowing at `>= 8,192` tokens (`narrowed_chunks = 3`, capping p99 TPOT across `576` samples at `673.20 ms` vs `987.28 ms`, `1.47x` lower), at a `+16.3%` background TTFT trade-off (`12,432.5 -> 14,455.8 ms`). Adaptive narrowing triggers only when an active decode slot overlaps a deep prefill chunk so solo long-context prefills keep full `512`-token chunks.
- **Default `serve` IOSurface auto-clamping**: omitting `--max-active-requests` at `--max-seq-len 10240 --prefix-cache-chunks 16` skips unused single-request KV chunks on the base model (`continuous_batch_only = true`) and auto-clamps default `max_batch_size` to `2` (`3,192` IOSurfaces, `3,534 MiB` wired RAM, `max_active_requests = 16` HTTP queue slots), launching cleanly without hitting macOS's per-process `IOSurface` ceiling.
- **ChatML boundary cache scope vs M6.5 `R8`**: for pure-KV models (`require_gdn_snapshot = false`), `PrefixCache.matchAndPin` already performs partial-LCP CoW matching via `first_child` once a root `512`-token tile exists. `C15` message-boundary checkpoints specifically cover histories whose edit boundary sits below the first `512`-token tile (rescuing `0/383 -> 359/383` hits in `bench-m68` and `11.06x` Turn 2 TTFT in live `serve`), retired `q8` blocks, and hybrid `GDN + QSA` exact-snapshot boundaries (`require_gdn_snapshot = true`), while `evict_first = true` ensures canonical `512`-token tiles survive eviction under cache pressure.

### R9 live non-anonymous RAM admission and soft cache auto-sizing

- `HOST_VM_INFO64` live zero-swap reclaimable RAM reported by `probe` and `bench-m68`: `10,053 MiB` (`probe`) and `9,624 MiB` (`bench-m68` Run 1: `free = 3,749`, `speculative = 729`, `purgeable = 543`, `external = 4,602`), explicitly excluding `inactive` (`6,333`), `internal` (`9,114`), `compressor` (`3,097`), and `wired` (`3,408`).
- Soft cache auto-sizing under tight simulated live headroom cleanly scales optional `prefix_cache_chunks` from `64 -> 16` (`cache_scaled_down = yes`), while opt-in `--strict-live-ram` refuses oversubscribed admissions (`strict_live_ram_rejected = yes`).
- Zero swapouts across every M6.8 benchmark (`bench-m68`: 3/3 consecutive runs with `infer_swapouts_delta = 0` and `total_swapouts_delta = 0`; all 6 standalone `bench-decode` runs: `swapouts_delta = 0`; live `serve`: `swapouts_delta = 0`).

### Verbatim `aminfer probe` output (`ReleaseFast`, M6.8)

```text
sys.page_size                    16384
sys.memory_mib                   24576
cpu.performance_cores            4
cpu.efficiency_cores             6
cpu.pcore_clusters               1
cpu.pcore_l2_kib                 16384
cpu.sme2                         true
cpu.sme_bf16f32                  true
cpu.sme_f16f32                   true
cpu.sme_svl_bytes                64
wire.user_limit_bytes            19585050869
wire.global_limit_bytes          19585050869
wire.gpu_limit_mib               0
gpu.name                         Apple M4
gpu.apple_family                 9
gpu.supports_apple10             no
gpu.prefill_matmul_backend       simdgroup_apple9
gpu.metal4                       true
gpu.unified_memory               true
gpu.recommended_working_set_mib  18186
gpu.max_buffer_mib               13639
ane.cores                        16
budget.wired_limit_mib           18186
budget.binding_limit             MTLDevice.recommendedMaxWorkingSetSize
vm.live_zero_swap_reclaimable_mib 10053 (free=2023, speculative=367, purgeable=139, external=7522; excluded: inactive=7838, internal=8508, compressor=2482, wired=3434)
canary.status                    pass
canary.cold_elapsed_us           2132.5
canary.warm_inbuf_elapsed_us     219.0
canary.max_abs_err_f16           0.000023
canary.max_abs_err_bf16          0.001953
canary.max_abs_err_q4            0.000028
```

### Verbatim `aminfer bench-m68` output (`ReleaseFast`)

```text
=== M6.8 G16 Startup .metallib Numerical Canary & Apple10 (NAX) Probe ===
g16.supports_apple10             no
g16.prefill_matmul_backend       simdgroup_apple9
g16.canary_cold_elapsed_us       2223.5
g16.canary_warm_inbuf_elapsed_us 231.2
g16.canary_max_abs_err           f16=0.000023 bf16=0.001953 q4=0.000028
g16.corrupted_slice_rejected     yes

=== M6.8 G15+R9 Wide-Chunk GPU Prefill, prompt_tail Pruning & Scratch Aliasing A/B ===
model                 toks  real  tree64_ms  wide_nop_ms  wide_ms    speedup  wide_tok_s  max_diff  greedy  alias_mib  wired_mib
qwen3-0.6b-bf16       512   yes   405.81     229.20       212.88     1.91   x 2405.1      0.03516   yes     18.02      1423
qwen3-0.6b-bf16       2048  yes   1880.96    1102.51      1092.75    1.72   x 1874.2      0.05859   yes     18.02      1423
qwen3-0.6b-gsq-rco    512   yes   556.59     216.66       206.86     2.69   x 2475.2      0.02734   yes     18.02      595
qwen3-0.6b-gsq-rco    2048  yes   2481.34    1150.43      1066.57    2.33   x 1920.2      0.03125   yes     18.02      595
qwen3-14b-4bit        512   yes   14529.71   6491.80      6331.50    2.29   x 80.9        0.03125   yes     78.02      8125
qwen3-30b-a3b-gsq     512   yes   18020.11   4588.77      4603.94    3.91   x 111.2       1.64063   yes     53.34      8809
qwen4_exp-512e-R2     512   no    73.12      40.06        33.87      2.16   x 15116.2     0.08008   yes     2.06       209
g15.eagle3_wide_spec             prompt=480 wide_prefill_ms=8605.43 spec_tokens=16 spec_tok_s=2.22 mean_accept=2.50 alias_mib=19.63 cold_exp=66 wired_mib=6770 greedy_match=yes malloc_blk=0

=== M6.8 C15 Concurrent Serving QoS & Multi-Turn ChatML Message-Boundary PrefixCache ===
c15.workload                     real=yes bg_prefill_tokens=10240
c15.qos_off                      decode_tpot_p50_ms=562.94 decode_tpot_p99_ms=987.28 bg_prefill_ttft_ms=12432.5 narrowed_chunks=0 qos_steps=d:0/p:0 samples=42
c15.qos_on                       decode_tpot_p50_ms=5.43 decode_tpot_p99_ms=673.20 bg_prefill_ttft_ms=14455.8 narrowed_chunks=3 qos_steps=d:270/p:6 samples=576 decode_share=0.50
c15.qos_p99_ratio                1.47x (qos_off_p99 / qos_on_p99)
c15.boundary_off                 hit_tokens=0/383 boundary_pos=359 ttft_ms=183.05
c15.boundary_on                  hit_tokens=359/383 boundary_pos=359 ttft_ms=28.74 ttft_speedup=6.37x boundary_hits=1 greedy_match=yes
c15.evict_first                  evictions=1 canonical_512_survived=yes

=== M6.8 R9 Live Non-Anonymous RAM Admission & Soft Cache Auto-Sizing ===
r9.live_reclaimable_mib          9624 (free=3749, speculative=729, purgeable=543, external=4602; excluded: inactive=6333, internal=9114, compressor=3097, wired=3408)
r9.soft_cache_autosize           requested_chunks=64 -> granted_chunks=16 cache_scaled_down=yes
r9.strict_live_ram_rejected      yes
m68.swap.status                  infer_swapouts_delta=0 infer_pageouts_delta=8 total_swapouts_delta=0 total_pageouts_delta=27
```

### Verbatim standalone `aminfer bench-decode` `tree64` vs `wide` A/B (`ReleaseFast`)

```text
=== qwen3-0.6b-gsq-rco (tree64) ===
decode_bench.target              qwen3
decode_bench.arch                qwen3 (bf16)
decode_bench.gpu_prefill_mode    tree64
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           483
decode_bench.aliased_saved_mib   0.00
decode_bench.canary_us           1932.5
decode_bench.load_ms             684.6
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          541.835
decode_bench.prefill_tok_per_s   944.94
decode_bench.measured_layers     28
decode_bench.measured_ms_per_tok 5.108
decode_bench.measured_tok_per_s  195.76
decode_bench.measured_layer_ms   0.156
decode_bench.measured_head_ms    0.729
decode_bench.measured_bytes_tok  382919232
decode_bench.measured_beta_gbs   74.96 (62.5% bus)
decode_bench.scaled_layers       28 (linear: 28 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   5.108
decode_bench.scaled_tok_per_s    195.76
decode_bench.scaled_bytes_tok    382919216
decode_bench.scaled_beta_gbs     74.96 (62.5% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      0
=== qwen3-0.6b-gsq-rco (wide) ===
decode_bench.target              qwen3
decode_bench.arch                qwen3 (bf16)
decode_bench.gpu_prefill_mode    wide
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           445
decode_bench.aliased_saved_mib   10.45
decode_bench.canary_us           1325.2
decode_bench.load_ms             699.3
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          222.487
decode_bench.prefill_tok_per_s   2301.26
decode_bench.measured_layers     28
decode_bench.measured_ms_per_tok 5.288
decode_bench.measured_tok_per_s  189.11
decode_bench.measured_layer_ms   0.158
decode_bench.measured_head_ms    0.870
decode_bench.measured_bytes_tok  382919232
decode_bench.measured_beta_gbs   72.41 (60.3% bus)
decode_bench.scaled_layers       28 (linear: 28 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   5.288
decode_bench.scaled_tok_per_s    189.11
decode_bench.scaled_bytes_tok    382919216
decode_bench.scaled_beta_gbs     72.41 (60.3% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      0
=== qwen3-14b-4bit (tree64) ===
decode_bench.target              qwen3
decode_bench.arch                qwen3 (q4)
decode_bench.gpu_prefill_mode    tree64
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           8205
decode_bench.aliased_saved_mib   0.00
decode_bench.canary_us           1431.3
decode_bench.load_ms             13943.7
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          12537.840
decode_bench.prefill_tok_per_s   40.84
decode_bench.measured_layers     40
decode_bench.measured_ms_per_tok 86.734
decode_bench.measured_tok_per_s  11.53
decode_bench.measured_layer_ms   2.139
decode_bench.measured_head_ms    1.185
decode_bench.measured_bytes_tok  7954424640
decode_bench.measured_beta_gbs   91.71 (76.4% bus)
decode_bench.scaled_layers       40 (linear: 40 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   86.734
decode_bench.scaled_tok_per_s    11.53
decode_bench.scaled_bytes_tok    7954424640
decode_bench.scaled_beta_gbs     91.71 (76.4% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      0
=== qwen3-14b-4bit (wide) ===
decode_bench.target              qwen3
decode_bench.arch                qwen3 (q4)
decode_bench.gpu_prefill_mode    wide
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           8168
decode_bench.aliased_saved_mib   25.27
decode_bench.canary_us           1542.1
decode_bench.load_ms             14033.2
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          5604.319
decode_bench.prefill_tok_per_s   91.36
decode_bench.measured_layers     40
decode_bench.measured_ms_per_tok 96.529
decode_bench.measured_tok_per_s  10.36
decode_bench.measured_layer_ms   2.222
decode_bench.measured_head_ms    7.642
decode_bench.measured_bytes_tok  7954424640
decode_bench.measured_beta_gbs   82.40 (68.7% bus)
decode_bench.scaled_layers       40 (linear: 40 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   96.529
decode_bench.scaled_tok_per_s    10.36
decode_bench.scaled_bytes_tok    7954424640
decode_bench.scaled_beta_gbs     82.40 (68.7% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      48
=== qwen3-30b-a3b-gsq-rco (tree64) ===
decode_bench.target              qwen3_moe
decode_bench.arch                qwen3_moe (q3)
decode_bench.gpu_prefill_mode    tree64
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           10027
decode_bench.aliased_saved_mib   0.00
decode_bench.canary_us           3780.0
decode_bench.load_ms             24432.4
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          3469.696
decode_bench.prefill_tok_per_s   147.56
decode_bench.measured_layers     48
decode_bench.measured_ms_per_tok 19.673
decode_bench.measured_tok_per_s  50.83
decode_bench.measured_layer_ms   0.385
decode_bench.measured_head_ms    1.184
decode_bench.measured_bytes_tok  1190794112
decode_bench.measured_beta_gbs   60.53 (50.4% bus)
decode_bench.scaled_layers       48 (linear: 48 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   19.673
decode_bench.scaled_tok_per_s    50.83
decode_bench.scaled_bytes_tok    1190794112
decode_bench.scaled_beta_gbs     60.53 (50.4% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      34
=== qwen3-30b-a3b-gsq-rco (wide) ===
decode_bench.target              qwen3_moe
decode_bench.arch                qwen3_moe (q3)
decode_bench.gpu_prefill_mode    wide
decode_bench.seq_len             513
decode_bench.shards              1
decode_bench.wired_mib           9988
decode_bench.aliased_saved_mib   19.44
decode_bench.canary_us           1506.8
decode_bench.load_ms             23887.0
decode_bench.prefill_tokens      512
decode_bench.prefill_ms          2729.462
decode_bench.prefill_tok_per_s   187.58
decode_bench.measured_layers     48
decode_bench.measured_ms_per_tok 20.522
decode_bench.measured_tok_per_s  48.73
decode_bench.measured_layer_ms   0.398
decode_bench.measured_head_ms    1.414
decode_bench.measured_bytes_tok  1190794112
decode_bench.measured_beta_gbs   58.03 (48.4% bus)
decode_bench.scaled_layers       48 (linear: 48 * layer_ms + head_ms)
decode_bench.scaled_ms_per_tok   20.522
decode_bench.scaled_tok_per_s    48.73
decode_bench.scaled_bytes_tok    1190794112
decode_bench.scaled_beta_gbs     58.03 (48.4% bus)
decode_bench.malloc_blocks_delta 16
decode_bench.malloc_bytes_delta  1024
decode_bench.swapouts_delta      0
decode_bench.pageouts_delta      1
```

### Verbatim live `aminfer serve` `C15` ChatML boundary cache and concurrent QoS output (`ReleaseFast`)

Command: `./zig-out/bin/aminfer serve --amf /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf --config /Volumes/Akari/models/Qwen3-0.6B/config.json --tokenizer /Volumes/Akari/models/Qwen3-0.6B/tokenizer.json --port 0 --batch-mode continuous --prefix-cache-chunks 16 --max-seq-len 10240 --decode-share 0.50 --adaptive-prefill-narrow on --message-boundary-cache on --gpu-prefill-mode wide --no-verify-crc` (default `--max-active-requests` omitted; auto-clamps `max_batch_size=2`).

Client harness output:

```text
serve_test.health_initial status=200 body={"status":"ok","model":"qwen3-0.6b-gsq-rco","wired_mib":3534,"active_requests":0,"active_batch_size":0,"mean_batch_size":0.00,"prefix_hit_tokens":0}
serve_test.chatml_turn1 ttft_ms=337.05 e2e_ms=418.34 chunks=15
serve_test.chatml_turn2 ttft_ms=30.48 e2e_ms=108.24 chunks=15 ttft_speedup=11.06x
serve_test.concurrent_qos bg_ttft_ms=11925.20 bg_e2e_ms=11959.17 fg_reqs=8 fg_tokens=248 fg_tpot_samples=240 fg_tpot_p50_ms=4.83 fg_tpot_p99_ms=413.62
serve_test.health_final status=200 body={"status":"ok","model":"qwen3-0.6b-gsq-rco","wired_mib":3534,"active_requests":0,"active_batch_size":0,"mean_batch_size":1.01,"prefix_hit_tokens":515}
serve_test.exit_code=0
```

Server stdout:

```text
serve.status                     listening
serve.url                        http://127.0.0.1:51018
serve.model_id                   qwen3-0.6b-gsq-rco
serve.arch                       qwen3 (bf16)
serve.batch_mode                 continuous (max_batch_size=2)
serve.gpu_prefill_mode           wide
serve.decode_share               0.50 (adaptive_prefill_narrow=on, message_boundary_cache=on)
serve.prefix_cache_chunks        16
serve.kv_retire_q8_after_chunks  0
serve.ngram_packs                0
serve.spec_mode                  none
serve.pearl_mode                 auto (enabled=no)
serve.ngram_nvme                 off
serve.wired_mib                  3534
serve.max_conns                  32
serve.max_active_requests        16
serve.max_seq_len                10240
serve.status                     stopped
serve.total_requests             11
serve.rejected_503               0
serve.cancelled                  0
serve.max_observed_batch_size    2
serve.mean_batch_size            1.01
serve.total_batch_steps          278
serve.prefix_hit_tokens          515
serve.prefix_hit_requests        8
serve.prefix_cow_copies          20
serve.boundary_checkpoints       1 (boundary_hits=1, evict_first_evictions=1)
serve.qos_steps                  decode=210 prefill=4 narrowed_chunks=1
serve.kv_retire_events           0
serve.steady_malloc_blocks       650
serve.steady_malloc_bytes        42432
serve.swapouts_delta             0
```

## M7: Ultra and cluster scale-out (deepseek41 CED, MLA/CSA2 FP4, Mega-mHC, Engram, DSpark, and loopback TCP cluster)

Measured on 2026-11-23 on the Apple M4 24 GiB (`Apple9`, TB4) baseline at commit
`69d30afd9b53c2199f07046795899b06a1e95795` (`ReleaseFast`, `swapouts_delta = 0`,
`pageouts_delta = 0`).

Hardware scope notice: all measurements below ran locally on the single 24 GiB
M4 (`Apple9`, Thunderbolt 4) host using loopback TCP (`127.0.0.1`) and compact
synthetic `deepseek41` bundles. Physical multi-Mac Thunderbolt 5 RDMA HCA
transport (`X1`–`X3`) and full-scale `552B + 196B` `DeepSeek-V4.1-Flash-1017`
execution (`W11`) require a 512 GB Apple Silicon Ultra node or a multi-Mac TB5
RDMA cluster and were not executed on this host
(`tier3_full_552b_status=deferred_requires_512gb_ultra_or_multi_mac_tb5_rdma_cluster`).

### Summary metrics

| Area | Metric | Measured result |
| --- | --- | --- |
| `R7` FP4 KV codec | `deepseek41` full geometry (`kv_lora_rank=1536`, `qk_rope_dim=26`, `E4M3` RoPE) | `890 B/tok` (`3.51x` vs `3,124 B/tok` FP16 MLA; `81.1x` vs `72,192 B/tok` unabsorbed FP16 K+V); latent `max_abs=0.2778`, `rmse=0.0822`; RoPE `max_abs=0.0624` |
| `G11` MLA + CSA2 GPU | 3-layer `[Full, Reindex, Reuse]` sharing 1 FP4 KV slab (`kv_len=2048`) | `868,352 B` shared vs `2,605,056 B` unshared (`66.7%` saved); latency `full=891.32 us`, `reindex=922.37 us`, `reuse=788.24 us` |
| `G10` Single-Pass Mega-mHC | 4 streams, `hidden=4096`, Sinkhorn-Knopp Birkhoff (`4` iters) | `m=1`: `215.85 us` (`0.57 GB/s`); `m=64`: `230.70 us` (`31.85 GB/s`, `26.5%` of `120.0 GB/s` bus) |
| `C10` Engram `F_NOCACHE` | 2 shards, `2,048` rows, `2` heads (`orders=[2,3]`) | Cold SSD: `7.47 us` (`24` NVMe reads); `PlePageCache` hit: `0.82 us` (`0` reads); loop `r >= 1` scratch reuse: `0.49 us` (`0` reads); `malloc_blocks_delta = 0` |
| `W11` + `D10` CED + DSpark | Synthetic `deepseek41` (`2` enc + `4` dec, `R=2`, `8` experts, 2 Engram shards) | CED decode `341.26 tok/s` (`2.930 ms/tok`, `0` encoder decode dispatches, `0` `r >= 1` Engram SSD reads); `DSpark` `greedy_match=yes`, `mean_accept=1.07`, `0` encoder decode dispatches |
| `X1` TB5 RDMA probe | Runtime `probeTb5Rdma()` on M4 TB4 | `tb5_rdma_hw=unavailable` (`driver_lib_found=no`, `devices=0`, `reason=rdma_framework_absent`); `.auto -> tb4_tcp`; `--strict-tb5-rdma -> error: Tb5RdmaUnavailable` |
| `X2` 2-rank loopback CED | In-process 2-rank loopback TCP `deepseek41` CED (`world_size=2`) | `greedy_match=yes` vs 1-rank, `0` encoder decode dispatches, `40` dispatches, `46` remote assignments, `118.73 us` avg RPC, `59,408 B` transferred |
| `X3` 4-rank loopback MoE | 4-rank loopback TCP (`3` `PeerWorkerServer`s + Rank 0 coordinator) | `m1_max_err=0.000000`, `m8_max_err=0.000000`, `16` dispatches, `48` remote assignments, `121.20 us` avg RPC, `79,104 B`, `malloc_blocks_delta = 0` |
| Live 2-OS-process TCP cluster | `/Volumes/Akari/models/amf/synth-deepseek41/` (`7.0 MiB`): 1-process `decode` vs 2-OS-process `peer-worker` (Rank 1) + `decode --peer` (Rank 0) vs `spec-decode --draft-mode dspark` | `15/15` bit-exact greedy `token_ids` match across all three; `0` encoder decode dispatches; `32` TCP RPCs / `123` remote assignments (`285.87 us` avg RPC, `130,264 B`); `swapouts_delta = 0` |

Note on `malloc_blocks_delta`: pure-CPU/IO/TCP paths (`m7.c10.malloc_blocks_delta = 0`,
`m7.x3.cluster_4rank_malloc_delta = 0`) perform zero steady-state heap
allocations. Short Metal 4 GPU bursts (`< 1,000` submits) exhibit the macOS
26.0 Metal 4 driver-internal `64 B` tracking-slab growth (`~2` blocks/commit)
previously documented in M6.5 and M6.8. On the random-weight synthetic
`deepseek41` bundle (`vocab = 2048`), the CPU `f32` MoE worker and GPU `f16`
MoE path agree bit-for-bit on `15/15` greedy tokens for `--max-tokens 15`
(on longer random-weight runs, near-tied uniform logits within `1 ULP` of
`f16` intermediate rounding can flip argmax).

### Verbatim `aminfer bench-m7` output (ReleaseFast)

Command: `./zig-out/bin/aminfer bench-m7`

```text
== aminfer M7 benchmark (R7/G11 MLA+CSA2 FP4, G10 Mega-mHC, C10 Engram, W11/D10 deepseek41 CED+DSpark, X1-X3 Cluster) ==
m7.host.gpu                      Apple M4 (m4)
m7.host.spec_bus_gbs             120.0

-- Section 1: R7 + G11 MLA FP4 KV (890 B/tok) & CSA2 (Full/Reindex/Reuse + SWA) --
m7.r7.deepseek41_fp4_b_per_tok   890 (ratio=3.51x vs 3124 B/tok FP16-MLA; 81.1x vs 72192 B/tok unabsorbed FP16 K+V)
m7.r7.fp4_latent_err             max_abs=0.2778 rmse=0.0822
m7.r7.fp4_rope_e4m3_err          max_abs=0.0624
m7.g11.mla_csa2_kv_len           2048
m7.g11.mla_csa2_kv_bytes_3layer  shared=868352 unshared=2605056 saved=1736704 (66.7%)
m7.g11.mla_csa2_latency_us       full=891.32 reindex=922.37 reuse=788.24
m7.g11.malloc_blocks_delta       191

-- Section 2: G10 Single-Pass Mega-mHC (4 streams, Sinkhorn-Knopp Birkhoff) --
m7.g10.mhc_shape                 streams=4 hidden=4096
m7.g10.mhc_m1_decode             latency_us=215.85 effective_gbs=0.57 (0.5% of bus)
m7.g10.mhc_m64_prefill           latency_us=230.70 effective_gbs=31.85 (26.5% of bus)
m7.g10.malloc_blocks_delta       128
-- Section 3: C10 Engram Multi-Head F_NOCACHE Lookup & Loop Reuse --
m7.c10.engram_config             shards=2 rows=2048 heads=2
m7.c10.engram_cold_ssd           latency_us=7.47 nvme_reads=24
m7.c10.engram_cache_hit          latency_us=0.82 nvme_reads=0
m7.c10.engram_loop_r_ge1         latency_us=0.49 nvme_reads=0
m7.c10.malloc_blocks_delta       0

-- Section 4: W11 + D10 Synthetic deepseek41 CED 2-Phase + DSpark Speculative Decode --
m7.w11.ced_layers                encoder=2 decoder=4
m7.w11.ced_prefill_ms            6.406
m7.w11.ced_decode                ms_per_tok=2.930 tok_s=341.26 encoder_decode_dispatches=0 engram_r_ge1_nvme_reads=0
m7.w11.malloc_blocks_delta       234
m7.d10.dspark_decode             ms_per_tok=3.661 tok_s=273.17 mean_accept=1.07 speedup=0.80x greedy_match=yes encoder_decode_dispatches=0
m7.d10.malloc_blocks_delta       310
-- Section 5: X1-X3 TB5 RDMA Probe & Live Loopback TCP Cluster --
m7.x1.tb5_rdma_hw                unavailable (driver_lib_found=no devices=0 reason=rdma_framework_absent)
m7.x1.transport_selection        auto=tb4_tcp strict_tb5_reject=pass
m7.x2.cluster_2rank_ced          greedy_match=yes encoder_decode_dispatches=0 dispatches=40 remote_assignments=46 rpc_us=118.73 bytes_transferred=59408
m7.x2.cluster_2rank_malloc_delta 170
m7.x3.cluster_4rank_moe          m1_max_err=0.000000 m8_max_err=0.000000 dispatches=16 remote_assignments=48 rpc_us=121.20 bytes_transferred=79104
m7.x3.cluster_4rank_malloc_delta 0
tier3_full_552b_status=deferred_requires_512gb_ultra_or_multi_mac_tb5_rdma_cluster
m7.vm.swapouts_delta             0
m7.vm.pageouts_delta             0
```

### Verbatim synthetic `deepseek41` bundle verification and live 2-OS-process TCP cluster runs (ReleaseFast)

Bundle generator: `cd tools && uv run aminfer-tools synth-deepseek41 --preset tiny-deepseek41 --output-dir /Volumes/Akari/models/amf/synth-deepseek41` (`7.0 MiB` total).

Command: `./zig-out/bin/aminfer verify-amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf`

```text
amf.status       ok
amf.model_id     deepseek41-tiny-synth
amf.tensors      174
amf.file_bytes   6111232
amf.wired_bytes  6062080
amf.cold_bytes   0
```

Command: `./zig-out/bin/aminfer verify-amf /Volumes/Akari/models/amf/synth-deepseek41/engram-shard-00-of-02.amf`

```text
amf.status       ok
amf.model_id     deepseek41-tiny-synth-engram-s0
amf.tensors      4
amf.file_bytes   163840
amf.wired_bytes  0
amf.cold_bytes   0
amf.sparse_bytes 131072
```

Command: `./zig-out/bin/aminfer verify-amf /Volumes/Akari/models/amf/synth-deepseek41/engram-shard-01-of-02.amf`

```text
amf.status       ok
amf.model_id     deepseek41-tiny-synth-engram-s1
amf.tensors      4
amf.file_bytes   163840
amf.wired_bytes  0
amf.cold_bytes   0
amf.sparse_bytes 131072
```

Command: `./zig-out/bin/aminfer verify-amf /Volumes/Akari/models/amf/synth-deepseek41/dspark.amf`

```text
amf.status       ok
amf.model_id     deepseek41-tiny-synth-dspark
amf.tensors      11
amf.file_bytes   901120
amf.wired_bytes  868352
amf.cold_bytes   0
```

Baseline 1-process greedy decode:
`./zig-out/bin/aminfer decode --amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf --config /Volumes/Akari/models/amf/synth-deepseek41/config.json --tokenizer /Volumes/Akari/models/amf/synth-deepseek41/tokenizer.json --prompt "Hello Apple Silicon cluster" --max-tokens 15 --ignore-eos`

```text
decode.arch                      graph (mla_csa2, moe_swiglu, mhc_single_pass, loop=single_pass R=1)
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    4
decode.shards                    1
decode.wired_mib                 11
decode.prompt_tokens             26
decode.generated_tokens          15
decode.prefill_ms                6.090
decode.decode_ms_per_tok         1.141
decode.decode_tok_per_s          876.10
decode.bytes_per_tok             1564132
decode.effective_gbs             1.37
decode.malloc_blocks_delta       84
decode.malloc_bytes_delta        5376
decode.swapouts_delta            0
decode.loop_total_steps             14
decode.loop_early_exit_tokens       0
decode.loop_gdn_scratch_evals       0
decode.loop_gdn_final_commits       0
decode.loop_ple_r0_fills            28
decode.loop_ple_r_ge1_reuses        0
decode.loop_ple_r_ge1_nvme_reads    0
decode.loop_qsa_index_computes      0
decode.loop_qsa_index_reuses        0
decode.loop_moe_r_ge1_pinned_routes 0
decode.loop_moe_r_ge1_cold_stalls   0
decode.loop_steplora_zero_skips     0
decode.loop_steplora_active_evals   0
decode.m7.enc_prefill_dispatches 2
decode.m7.enc_decode_dispatches  0
decode.m7.dec_decode_dispatches  28
decode.ple_nvme_reads            16
decode.ple_cache_hits            492
decode.token_ids                 [670, 1924, 825, 1521, 1983, 1140, 76, 1991, 1702, 1173, 382, 1602, 1037, 527, 896]
decode.text                       v000670 v001924 v000825 v001521 v001983 v001140L v001991 v001702 v001173 v000382 v001602 v001037 v000527 v000896
```

Live 2-OS-process TCP cluster — Process 1 (`peer-worker`, Rank 1):
`./zig-out/bin/aminfer peer-worker --listen 127.0.0.1:19471 --amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf --config /Volumes/Akari/models/amf/synth-deepseek41/config.json --rank 1 --world-size 2 --partition contiguous`

```text
peer_worker.listen               127.0.0.1:19471
peer_worker.rank                 1/2 (partition=contiguous)
peer_worker.local_experts        4..8 (of 8, layers=4)
peer_worker.transport            tb4_tcp (rdma_reason=rdma_framework_absent)
peer_worker.max_requests         0
peer_worker.status               ready
peer_worker.requests_served      32
peer_worker.assignments_eval     123
peer_worker.bytes_rx             44720
peer_worker.bytes_tx             85944
peer_worker.swapouts_delta       0
```

Live 2-OS-process TCP cluster — Process 2 (`decode --peer`, Rank 0):
`./zig-out/bin/aminfer decode --amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf --config /Volumes/Akari/models/amf/synth-deepseek41/config.json --tokenizer /Volumes/Akari/models/amf/synth-deepseek41/tokenizer.json --prompt "Hello Apple Silicon cluster" --max-tokens 15 --ignore-eos --peer 127.0.0.1:19471 --rank 0 --world-size 2 --partition contiguous --shutdown-peers`

```text
decode.arch                      graph (mla_csa2, moe_swiglu, mhc_single_pass, loop=single_pass R=1)
decode.temperature               0.00
decode.seed_used                 0
decode.seed_source               greedy
decode.layers                    4
decode.shards                    1
decode.wired_mib                 9
decode.prompt_tokens             26
decode.generated_tokens          15
decode.prefill_ms                12.414
decode.decode_ms_per_tok         2.647
decode.decode_tok_per_s          377.85
decode.bytes_per_tok             1564132
decode.effective_gbs             0.59
decode.malloc_blocks_delta       140
decode.malloc_bytes_delta        8960
decode.swapouts_delta            0
decode.loop_total_steps             14
decode.loop_early_exit_tokens       0
decode.loop_gdn_scratch_evals       0
decode.loop_gdn_final_commits       0
decode.loop_ple_r0_fills            28
decode.loop_ple_r_ge1_reuses        0
decode.loop_ple_r_ge1_nvme_reads    0
decode.loop_qsa_index_computes      0
decode.loop_qsa_index_reuses        0
decode.loop_moe_r_ge1_pinned_routes 0
decode.loop_moe_r_ge1_cold_stalls   0
decode.loop_steplora_zero_skips     0
decode.loop_steplora_active_evals   0
decode.m7.enc_prefill_dispatches 2
decode.m7.enc_decode_dispatches  0
decode.m7.dec_decode_dispatches  28
decode.cluster.world_size        2 (rank=0, partition=contiguous)
decode.cluster.transport         tb4_tcp (rdma_reason=rdma_framework_absent)
decode.cluster.dispatches        32 (local=101, remote=123)
decode.cluster.rpc_us            285.87
decode.cluster.bytes_transferred 130264
decode.ple_nvme_reads            16
decode.ple_cache_hits            492
decode.token_ids                 [670, 1924, 825, 1521, 1983, 1140, 76, 1991, 1702, 1173, 382, 1602, 1037, 527, 896]
decode.text                       v000670 v001924 v000825 v001521 v001983 v001140L v001991 v001702 v001173 v000382 v001602 v001037 v000527 v000896
```

`DSpark` speculative decode (`--draft-mode dspark`):
`./zig-out/bin/aminfer spec-decode --amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf --config /Volumes/Akari/models/amf/synth-deepseek41/config.json --tokenizer /Volumes/Akari/models/amf/synth-deepseek41/tokenizer.json --prompt "Hello Apple Silicon cluster" --max-tokens 15 --ignore-eos --draft-mode dspark`

```text
spec.arch                        graph (mla_csa2, moe_swiglu, mhc_single_pass, loop=single_pass R=1)
spec.mode                        dspark
spec.placement                   gpu (requested=auto)
spec.temperature                 0.00
spec.seed_used                   0
spec.seed_source                 greedy
spec.layers                      4
spec.wired_mib                   11
spec.prompt_tokens               26
spec.generated_tokens            15
spec.rounds                      14
spec.mean_accept_len             1.000
spec.self_recurrent_draft_steps  0
spec.prefill_ms                  7.982
spec.decode_ms_per_tok           2.394
spec.decode_tok_per_s            417.67
spec.malloc_blocks_delta         139
spec.malloc_bytes_delta          8928
spec.swapouts_delta              0
spec.loop_total_steps             14
spec.loop_early_exit_tokens       0
spec.loop_gdn_scratch_evals       0
spec.loop_gdn_final_commits       0
spec.loop_ple_r0_fills            28
spec.loop_ple_r_ge1_reuses        0
spec.loop_ple_r_ge1_nvme_reads    0
spec.loop_qsa_index_computes      0
spec.loop_qsa_index_reuses        0
spec.loop_moe_r_ge1_pinned_routes 0
spec.loop_moe_r_ge1_cold_stalls   0
spec.loop_steplora_zero_skips     0
spec.loop_steplora_active_evals   0
spec.m7.enc_prefill_dispatches   2
spec.m7.enc_decode_dispatches    0
spec.m7.dec_decode_dispatches    28
spec.ple_nvme_reads              16
spec.ple_cache_hits              600
spec.token_ids                   [670, 1924, 825, 1521, 1983, 1140, 76, 1991, 1702, 1173, 382, 1602, 1037, 527, 896]
spec.text                         v000670 v001924 v000825 v001521 v001983 v001140L v001991 v001702 v001173 v000382 v001602 v001037 v000527 v000896
```

Strict TB5 RDMA negative gate (`--strict-tb5-rdma` on M4 TB4):
`./zig-out/bin/aminfer decode --amf /Volumes/Akari/models/amf/synth-deepseek41/model.amf --config /Volumes/Akari/models/amf/synth-deepseek41/config.json --tokenizer /Volumes/Akari/models/amf/synth-deepseek41/tokenizer.json --prompt "hi" --max-tokens 2 --strict-tb5-rdma`

```text
error: Tb5RdmaUnavailable
```

## M7.5: Native C/Python SDK, full HTTP/SSE protocol parity, rootless TUI monitor and local ergonomics

Recorded on 2026-11-23 (`ReleaseFast` on Apple M4 24 GB unified memory, macOS 26.0, commits `996782f..ddd90a4`, [ADR-0022](../decisions/0022-native-c-python-sdk-and-full-protocol-parity.md)).

### Summary metrics

| Component | Capability | Measured M4 metric (`ReleaseFast`) | Invariant check |
| --- | --- | --- | --- |
| `W18` Single-file `.amf` metadata & header-only inspect | `aminfer inspect` / `aminfer ls` & `loadFromFileAutoConfig` | `127.96 us` header-only inspect; `9.778 ms` single-file auto-config load (`parity=yes`) | `0 B` wired delta during inspect; `0` swapouts |
| `W19` Multi-LoRA hot-loading & per-request routing | `LoraRegistry` (`512 KiB` wired pool `<= 4 MiB` L2 cap, rank-8 `.amlora` `128 KiB`) | `304.82 us` base vs `273.47 us` `active_adapters == 0` bypass (`zero_bypass_match=yes`, `unload_match=yes`); `1684.67 us` active rank-8 LoRA | `0` instructions added inside base layer loop when `active_adapters == 0` |
| `C22` Preallocated `json_schema` / GBNF bitmask engine | `compileJsonSchema` + `VocabIndex.fillAllowedTokenBitmask` (`vocab=4096`) | `43.54 us` PDA compile; `6.01 us` per-step token bitmask evaluation (`schema_match=yes`) | `0` steady-state heap blocks (`malloc_blocks_delta=0`, `malloc_bytes_delta=0`) |
| `C19` Rootless `aminfer monitor` telemetry | Mach `PROCESSOR_CPU_LOAD_INFO` (`4P+6E`), `HOST_VM_INFO64` / `R9`, `IOAccelerator`, `IOBlockStorageDriver` | `1170.11 us` full host sample + `1761 B` JSON snapshot (`wired_limit_MiB=18186`, `live_reclaimable_MiB=7076`) | Rootless (`0` `sudo`), `0` steady-state heap blocks (`malloc_blocks_delta=0`) |
| `C21` Native embeddings & cross-encoder reranking | `prefillEmbeddings` (`lm_head` elided, Matryoshka `dim=64`, L2 norm) & `rerankSequences` (`4` docs) | `1050.77 us` embedding (`l2_norm=1.0000`); `2395.79 us` 4-doc cross-encoder rerank (`sorted_ok=yes`) | Reuses `G15` wide prefill while eliding `lm_head`; `0` swapouts |
| `R10` Opt-in `DiskPrefixCache` NVMe persistence | 16 KiB-aligned `F_NOCACHE` atomic spill & cold-restart reload (`256 KiB` `retired_q8` artifact) | `2059.17 us` spill (`0.13 GB/s`); `1823.08 us` cold reload (`0.14 GB/s`, `token_verified=yes`, `collision_rejected=yes`) | Full token-slice verification rejects hash collisions (`0` collision risk) |
| `C17` Plain-C `libaminfer.dylib` & Python `ctypes` SDK | Dynamic `dlopen` create/destroy lifecycle (`4` cycles) + real `Qwen3-0.6B` BF16 zero-swap check | `102.520 ms` mean create/destroy cycle (`leaked_wired_pages=0`); `Qwen3-0.6B` BF16 `48.21 tok/s` (`20.741 ms/tok`, `1213 MiB` wired) | `0` Python / third-party dylibs linked; `0` leaked wired pages; `swapouts_delta=0` |

### Raw verification outputs

`otool -L zig-out/lib/libaminfer.dylib`:

```text
zig-out/lib/libaminfer.dylib:
	@rpath/libaminfer.dylib (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate (compatibility version 1.0.0, current version 4.0.0)
	/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics (compatibility version 64.0.0, current version 2050.0.0)
	/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/CoreMedia.framework/Versions/A/CoreMedia (compatibility version 1.0.0, current version 3350.77.5)
	/System/Library/Frameworks/CoreVideo.framework/Versions/A/CoreVideo (compatibility version 1.2.0, current version 758.27.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 5027.0.69)
	/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit (compatibility version 1.0.0, current version 275.0.0)
	/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/ImageIO.framework/Versions/A/ImageIO (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0, current version 382.5.3)
	/System/Library/Frameworks/VideoToolbox.framework/Versions/A/VideoToolbox (compatibility version 1.0.0, current version 3350.77.5)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1359.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 5027.0.69)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
```

`TMPDIR=/Volumes/Akari/.tmp/m75-engine ./zig-out/bin/aminfer bench-m75`:

```text
bench.m75.device                 Apple M4 (m4 base, bus=120.0 GB/s)
bench.m75.w18.single_file_amf    inspect_us=127.96 inspect_wired_delta_B=0 auto_load_ms=9.778 parity=yes (file_B=786432, wired_weight_B=720896, meta_blob_B=32768)
bench.m75.w19.multi_lora         rank=8 pool_KiB=512 adapter_KiB=128 base_us=304.82 zero_adapter_us=273.47 active_lora_us=1684.67 delta_us=1379.85 zero_bypass_match=yes unload_match=yes malloc_blocks_delta=96
bench.m75.c22.grammar_bitmask    vocab=4096 compile_us=43.54 mask_eval_us=6.01 schema_match=yes malloc_blocks_delta=0 malloc_bytes_delta=0
bench.m75.c19.monitor_json       sample_us=1170.11 cores=4P+6E wired_limit_MiB=18186 live_reclaimable_MiB=7076 json_B=1761 malloc_blocks_delta=0
bench.m75.c21.embed_rerank       embed_dim=64 embed_us=1050.77 l2_norm=1.0000 rerank_docs=4 rerank_us=2395.79 sorted_ok=yes malloc_blocks_delta=128
bench.m75.r10.disk_prefix_cache  artifact_KiB=256 spill_us=2059.17 spill_GBs=0.13 cold_reload_us=1823.08 reload_GBs=0.14 token_verified=yes collision_rejected=yes
bench.m75.c17.c_abi_lifecycle    dylib_dlopen=yes cycles=4 mean_cycle_ms=102.520 leaked_wired_pages=0
bench.m75.real.qwen3_0_6b        status=verified_zero_swap wired_MiB=1213 prefill_ms=66.616 decode_ms_per_tok=20.741 decode_tok_s=48.21
bench.m75.real.qwen3_14b_4bit    status=skipped_live_ram_headroom wired_MiB=7924 prefill_ms=0.000 decode_ms_per_tok=0.000 decode_tok_s=0.00
bench.m75.swapouts_delta         0
```

`./zig-out/bin/aminfer monitor --json`:

```json
{"theme":"trans-pride","chip":{"generation":"m4","tier":"base","device_name":"Apple M4","gpu_apple_family":9,"p_cores":4,"e_cores":6,"has_sme2":true,"has_nax":false,"spec_bus_gbs":120.0},"cpu":{"core_count":10,"p_core_count":4,"e_core_count":6,"total_pct":24.00,"p_core_pct":17.50,"e_core_pct":28.33,"loadavg":[3.46,3.84,3.77],"per_core_pct":[50.00,25.00,25.00,25.00,25.00,20.00,20.00,25.00,25.00,0.00]},"memory":{"total_bytes":25769803776,"page_size":16384,"wired_bytes":4283826176,"wired_budget_bytes":19069665280,"binding_limit":"MTLDevice.recommendedMaxWorkingSetSize","user_wire_limit_bytes":19585050869,"gpu_wired_limit_bytes":19069665280,"reclaimable_zero_swap_bytes":6968803328,"free_bytes":833650688,"speculative_bytes":784367616,"purgeable_bytes":74940416,"external_bytes":5275844608,"active_bytes":6030622720,"inactive_bytes":5397741568,"compressor_bytes":7610253312,"swap_used_bytes":4498653184,"swap_total_bytes":5368709120,"swapouts_total":47790386,"swapouts_delta":0},"gpu":{"available":true,"accelerator_class":"AGXAcceleratorG16G","thermal_state":"Nominal","device_utilization_pct":73,"renderer_utilization_pct":72,"tiler_utilization_pct":73,"in_use_system_memory_bytes":1425555456,"alloc_system_memory_bytes":5296898048},"nvme":{"available":true,"driver_count":7,"bytes_read_total":21915024297472,"bytes_written_total":15913135154176,"ops_read_total":1785264173,"ops_written_total":499343470,"read_mb_s":209.30,"write_mb_s":0.00,"read_iops":3307.77,"write_iops":0.00},"server":{"attached":false,"online":false,"address":"","model":"","wired_mib":0,"active_requests":0,"active_batch_size":0,"prefill_slots":0,"decode_slots":0,"mean_batch_size":0.00,"tok_per_s":0.00,"decode_share":0.00,"prefix_hit_tokens":0,"retired_q8_kv_chunks":0,"cluster_peers_online":0}}
```

`./zig-out/bin/aminfer ls /Volumes/Akari/models/amf`:

```text
ls.dir                           /Volumes/Akari/models/amf
ls.models_found                  12
  model_id / file                   arch                     bpw   file_mib   wired_mib  meta        min_tier      
  qwen3-0.6b-bf16.amf               qwen3_dense            16.00     1138.6      1370.5  -/-/-       M4 16GB       
  qwen3-0.6b-gsq-rco-sme2.amf       qwen3_dense             4.72      336.9       568.9  -/-/-       M4 16GB       
  qwen3-0.6b-gsq-rco.amf            qwen3_dense             4.35      310.7       542.6  -/-/-       M4 16GB       
  qwen3-0.6b-lut16-sme2.amf         qwen3_dense             4.72      336.9       568.9  -/-/-       M4 16GB       
  qwen3-14b-4bit.amf                qwen3_dense             4.50     7924.7      8284.6  -/-/-       M4 16GB       
  qwen3-30b-a3b-3bit.amf            qwen3_moe               3.50    12744.3     12949.9  -/-/-       M4 16GB       
  qwen3-30b-a3b-eagle3.amf          qwen3_dense            16.00      995.8      1059.8  -/-/-       M4 16GB       
  qwen3-30b-a3b-gsq-rco.amf         qwen3_moe               2.70     9831.6     10037.3  -/-/-       M4 16GB       
  dspark.amf                        deepseek41             16.00        0.9        64.8  -/-/-       M4 16GB       
  engram-shard-00-of-02.amf         deepseek41              4.50        0.2        64.0  -/-/-       M4 16GB       
  engram-shard-01-of-02.amf         deepseek41              4.50        0.2        64.0  -/-/-       M4 16GB       
  model.amf                         deepseek41              5.68        5.8        69.8  -/-/-       M4 16GB       
```

Single-file `.amf` (`--embed-metadata`, zero sidecars) `inspect`, `decode`, and `chat --lora` via `stdin` pipe:

```text
inspect.path                     /Volumes/Akari/.tmp/m75-engine/cli_check/model.amf
inspect.model_id                 qwen3-tiny-dense-4bit
inspect.file_bytes               704512 (0.67 MiB)
inspect.header_crc32             0x0d6ebc66
inspect.directory_crc32          0xb504dc13
inspect.header_flags             0x00000001 (embedded_metadata=yes)
inspect.entries                  total=27 weights=25 metadata=2
inspect.wired_weight_bytes       638976 (0.61 MiB)
inspect.cold_weight_bytes        0
inspect.sparse_table_bytes       0
inspect.metadata_blob_bytes      32768
inspect.estimated_kv_scratch     67108864
inspect.estimated_total_wired    67747840 (64.61 MiB)
inspect.effective_bpw            4.52
inspect.config_source            embedded (embedded=yes, crc_valid=yes)
inspect.tokenizer_meta           embedded=yes (bytes=7305, crc_valid=yes)
inspect.adapter_config_meta      embedded=no (bytes=0, crc_valid=no)
inspect.arch                     qwen3 (attn=gqa, ffn=dense, kv=rgba16f_q8, sparse=none)
inspect.dims                     layers=2 hidden=128 heads=4/2 head_dim=64 vocab=512
inspect.tier_fit                 min_tier="M4 16GB" (m4_16gb=yes, m4_24gb=yes, pro_48gb=yes, max_128gb=yes)
inspect.swapouts_delta           0

 v000301 v000301 v000301 v000301 v000301 v000301
[6 tok | 923.4 tok/s | TTFT 5.4 ms | PrefixCache hit 0/82 tok]
```

## M7.8: Native Kernel Barrier Fusion, MMA FlashAttention, and SIMD Accelerators

Recorded on Apple M4 (`Mac16,12`, `Apple9` GPU, `24 GiB` unified memory,
`120.0 GB/s` bus, `has_nax=no`) via `aminfer bench-m78` (`-Doptimize=ReleaseFast`,
`swapouts_delta = 0`). `Apple10+` `NAX` FlashAttention-2 hardware benchmarking
remains gated on physical `Apple10+` silicon in Spike `S8`
([ADR-0023](../decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md)).

### Summary

| Track | Scope / Condition | Measured Result (M4, ReleaseFast) | Notes / Guard |
| --- | --- | --- | --- |
| **G18 Canary** | Combined `G16` + `G18` single-frame startup canary | `warm_us = 122.79 us` (`< 350 us`), `mixed_gsq_guard = yes` | `16 KiB` single-page reuse; zero extra submit/wait roundtrip |
| **G18 Synth A/B** | 4L `GraphModel` (`hidden=1024, inter=3072`, `prompt64`) | Prefill64 `9.216 -> 4.342 ms` (`2.12x`), decode `2244.54 -> 2017.60 us` (`1.11x`), `parity = yes` | Shape gates calibrated for M4 (`qkv_min_m=64`, `swiglu_min_m=32`, residual epilogue `M==1 \|\| M>=64`) |
| **G18 Real `Qwen3-0.6B` BF16** | `1,213 MiB` wired, `prompt64`, zero-swap | Prefill `48.767 -> 44.933 ms` (`1.09x`), decode `45.90 -> 47.77 tok/s` (`1.04x`), `parity = yes` | `Qwen3-14B-4bit` (`7,924 MiB`) & `Qwen3-30B-A3B-3bit` (`12,744 MiB`) present and cleanly `skipped_live_ram_headroom` on 24 GiB host |
| **G19 MMA FlashAttn-2** | Causal GQA prefill (`KV_len = 512`, `Apple9` `simdgroup_half8x8`) | `Q=1`: `scalar_vector` (`197.02 vs 201.06 us`); `Q=16`: `scalar_vector`; `Q=32`: `2.36x`; `Q=64`: `3.72x`; `Q=512`: `5.50x` (`4825.79 -> 877.30 us`) | Strictly gated `Q_len >= 32 && KV_len >= 64`; `max_diff <= 0.00049`; `Apple10+` NAX HW pending Spike `S8` |
| **G20 Indirect MoE** | GPU `MTLDispatchThreadgroupsIndirectArguments` dispatch | `guard_ok = yes`, `M=1` direct `44.66 us`, `M=16` direct `168.53 us` / indirect `163.33 us`, `parity = yes` | `shouldUseIndirectDispatch`: `M > 1 \|\| num_scheme_buckets > 1`; direct path kept for `M=1` homogeneous MoE |
| **G21 248K GPU Sampler + C22** | `vocab = 248,077`, `C22` JSON schema speculative tree | `greedy = 14.31 us`, `constrained_radix = 111.89 us`, `sync_roundtrip = 784.96 us`; constrained accept `9.00` vs unconstrained `2.25` | Unconstrained greedy `argmax` fast-path bypass + full tree bitmask propagation |
| **C23 ARM64 NEON SIMD** | `16 KiB` JSON structural scanner + BPE `PreTokenIterator` | JSON `1.46 -> 7.58 GB/s` (`5.18x`); BPE `1.88 -> 3.81 GB/s` (`2.02x`); `parity = yes` | Exact bit parity including `< 16 B` tails |
| **C24 SME2 `luti2` + Multi-LoRA** | `m=1/16, n=512, k=512` `q2_affine` / `LUT4`, `<= 4 MiB` L2 cap | `luti2_m1 = 3.08 us`, `luti2_m16 = 14.83 us` vs scalar `1299.23 us` (`87.64x`, `565.82 GOPS`), `hoisted_lora = 7.69 us`, `bus_gate_ok = yes` | `0` streaming `[sp` spills across `35` `_ami_sme2_*` symbols (`otool -tvV`) |
| **G22 ViT Tower** | 3-layer synthetic ViT + `PatchMerger` + AnyRes + aliased scratch | `vit_encode = 0.603 ms`, `aliased_saved = 128 KiB`, `owned_wired = 0 B` during decode, `text_fastpath_ok = yes`, `zero_copy = yes` | `WiredArena.allocAliasedPair` reuses prefill scratch (`0 B` steady-state text decode overhead) |

### Verbatim `aminfer bench-m78` output

`TMPDIR=/Volumes/Akari/.tmp/m78-engine ./zig-out/bin/aminfer bench-m78`:

```text
bench.m78.device                 Apple M4 (m4 base, bus=120.0 GB/s, has_nax=no)
bench.m78.g18.canary             warm_us=122.79 (<350us) passed=yes mixed_gsq_guard=yes
bench.m78.g18.synth_dense_ab     prefill64_unfused_ms=9.216 prefill64_fused_ms=4.342 decode_unfused_us=2244.54 decode_fused_us=2017.60 parity=yes
bench.m78.g18.real.qwen3_06b     status=verified_zero_swap wired_MiB=1213 prefill_unfused_ms=48.767 prefill_fused_ms=44.933 decode_unfused_tok_s=45.90 decode_fused_tok_s=47.77 parity=yes
bench.m78.g18.real.qwen3_14b     status=skipped_live_ram_headroom wired_MiB=7924 prefill_unfused_ms=0.000 prefill_fused_ms=0.000 decode_unfused_tok_s=0.00 decode_fused_tok_s=0.00 parity=no
bench.m78.g18.real.qwen3_30b     status=skipped_live_ram_headroom wired_MiB=12744 prefill_unfused_ms=0.000 prefill_fused_ms=0.000 decode_unfused_tok_s=0.00 decode_fused_tok_s=0.00 parity=no
bench.m78.g19.q1_kv512         route=scalar_vector vector_us=197.02 mma_us=201.06 speedup=0.98x max_diff=0.00001
bench.m78.g19.q16_kv512         route=scalar_vector vector_us=307.66 mma_us=201.13 speedup=1.53x max_diff=0.00002
bench.m78.g19.q32_kv512         route=mma_apple9 vector_us=534.84 mma_us=226.42 speedup=2.36x max_diff=0.00002
bench.m78.g19.q64_kv512         route=mma_apple9 vector_us=962.91 mma_us=259.11 speedup=3.72x max_diff=0.00002
bench.m78.g19.q512_kv512         route=mma_apple9 vector_us=4825.79 mma_us=877.30 speedup=5.50x max_diff=0.00049
bench.m78.g19.summary            guard_ok=yes parity_ok=yes nax_hw=gated_on_spike_s8_apple9_m4_host
bench.m78.g20.indirect_moe       guard_ok=yes m1_direct_us=44.66 m1_indirect_us=40.69 m16_direct_us=168.53 m16_indirect_us=163.33 parity=yes
bench.m78.g21.sampler_248k       vocab=248077 greedy_us=14.31 constrained_radix_us=111.89 sync_roundtrip_us=784.96
bench.m78.g21.c22_spec_accept    unconstrained_accept=2.25 constrained_accept=9.00 all_nodes_valid=yes
bench.m78.c23.neon_simd          json_scalar_GBs=1.46 json_neon_GBs=7.58 json_speedup=5.18x bpe_scalar_GBs=1.88 bpe_neon_GBs=3.81 bpe_speedup=2.02x parity=yes
bench.m78.c24.sme2_luti2         sme2=yes luti2_m1_us=3.08 luti2_m16_us=14.83 scalar_m16_us=1299.23 speedup=87.64x m16_GOPS=565.82 hoisted_lora_us=7.69 l2_cap_KiB=4096 bus_gate_ok=yes
bench.m78.g22.vit_tower          text_fastpath_ok=yes layers=3 vis_tokens=16 aliased_saved_KiB=128 owned_wired_B=0 vit_encode_ms=0.603 zero_copy=yes
bench.m78.swapouts_delta         0
```

## M8.1: Multi-Format Quantization Parity, ChatTemplate, C18 Hardening, and W24 KL Quality Gate

Recorded on Apple M4 (`Mac16,12`, `Apple9` GPU, `24 GiB` unified memory,
`120.0 GB/s` bus) via `aminfer bench-m81` (`-Doptimize=ReleaseFast`,
`swapouts_delta = 0`, [ADR-0024](../decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md)).

### Summary

| Track | Scope / Condition | Measured Result (M4, ReleaseFast) | Notes / Guard |
| --- | --- | --- | --- |
| **G23 M=1 GEMV** | `4096 x 4096` rotating `>= 64 MiB` working set across 14 `W21`/`W22` dtypes vs legacy `q4_g64` (`76.73 GB/s`, `63.9%` bus) | `q4_affine_g64`: `90.96 GB/s` (`75.8%` bus, `103.93 us`); `q4_affine_g32`: `87.34 GB/s` (`72.8%`); `gguf_q4_0`: `87.22 GB/s` (`72.7%`, `108.39 us`); `q8_affine_g64`: `86.73 GB/s` (`72.3%`); `q6_affine_g64`: `84.74 GB/s`; `gguf_q8_0`: `82.60 GB/s`; `q5_affine_g64`: `80.54 GB/s`; `nvfp4_g16`: `72.81 GB/s` (`60.7%`); `mxfp8_g32`: `66.32 GB/s`; `fp8_e4m3_block`: `65.22 GB/s`; `gguf_q4_k`: `60.62 GB/s`; `gguf_q2_0`: `54.94 GB/s`; `gguf_iq2_xxs`: `35.51 GB/s`; `gguf_tq1_0`: `21.16 GB/s` | `quant_formats.metal` compiled with `-std=metal4.0 -Wall -Werror -Wframe-larger-than=1` (`0` warnings, `0` register spills) |
| **G23 M=64 Prefill** | `M = 64, N = 2048, K = 2048` tiled `fp16` dequant + `G15` GEMM | `gguf_q4_k`: `0.604 ms` (`0.89 TFLOPS`); `nvfp4_g16`: `0.617 ms` (`0.87 TFLOPS`); `fp8_block`: `0.668 ms` (`0.80 TFLOPS`); `mxfp8_g32`: `0.693 ms` (`0.77 TFLOPS`) | Reuses preallocated `16 KiB`-aligned tile buffer (`0` hot-path allocs) |
| **G24 Act-Quant** | Opt-in `--act-quant` (`K = 4096`, `M = 1` & `M = 16`) | `none`: `0` dispatches (`0.00 us`); `fp8_g128`: `m1=13.57 us, m16=17.25 us, mae=0.028059`; `mxfp8_g32`: `m1=16.56 us, m16=28.64 us, mae=0.034028`; `nvfp4_g16`: `m1=25.23 us, m16=25.81 us, mae=0.118255`; `fouroversix_mae`: `m1=35.41 us, m16=36.20 us, mae=0.109436` (`99.9%` win rate, `7.46%` MAE reduction vs `nvfp4_g16`) | Default `none` incurs `0` dispatch overhead |
| **C25 SME2 `.s`** | Bare-metal ARMv9.2 SME2 `E2M1`/`Q2_0`/`TQ`/`IQ2` microkernels (`M = 4` & `M = 16`) | `tq1_0`: `159.91 / 298.93 GOPS`; `nvfp4_g16`: `50.92 / 229.37 GOPS`; `tq2_0`: `87.61 / 218.42 GOPS`; `q2_0`: `94.59 / 202.57 GOPS`; `iq2_xxs`: `98.93 / 185.45 GOPS` (`max_err=1.22e-4`); `mxfp4_g32`: `99.40 / 130.48 GOPS`; `c13_bus_gate = pass` | `otool -tvV` audited `67` `_ami_sme2_*` symbols (`61` public `.globl` entrypoints + `6` local `.s` helpers; `0` streaming `[sp` spills, `0` leaked ops, `0` missing `d8–d15` saves) |
| **W21 / W22 E2E** | Dense `nvfp4_g16` + `q8_affine` embed, hetero asymmetric MoE (`nvfp4_g16`+`gguf_q2_0`/`gguf_q4_k` + `fp8_e4m3_block` attn + `gguf_q8_0` embed), quantized `DraftHead`, fork guard | `dense_e2e_parity = pass`, `moe_hetero_parity = pass`, `draft_quant_parity = pass`, `fork_stq1_reject = pass` | `M = 16` prefill + `M = 1` decode + `M = 4` `verifyTree` all finite; colliding `STQ1_0` payload on `type 43` rejected |
| **W24 + C18** | Live `Inkling-2-Mini` (`o200k` + `ignore_merges`) & `Hy4-flash` tokenizers, zero-alloc `ChatTemplate`, live loopback HTTP/SSE | `Inkling-2`: `46.18 MB/s` (`live_metadata`); `Hy4`: `27.57 MB/s` (`live_metadata`); `ChatTemplate` render: `chatml=0.38 us`, `inkling2=0.49 us`, `hy4=0.25 us`; `keepalive=pass`, `hmac_sig_400=pass`, `resp_completed_output=pass` | Tested against `/Volumes/Akari/models/metadata/{thinkingmachines__Inkling-2-Mini,tencent__Hy4-flash}/tokenizer.json` |
| **W24 KL Quality Gate** | 11-format synthetic `BF16` vs quantized sweep + real on-disk `Qwen3-0.6B` `BF16` vs `GSQ-RCO` multi-prompt math + code eval (`8` greedy steps = `4` math + `4` code, `R9` live-RAM gated) | Synthetic: `gguf_q8_0` `mean_kl=0.000356` (`100.0%` top-1, `8/8` greedy), `q8_affine_g64` `0.000387`, `q6_affine_g64` `0.005392`, `fp8_e4m3_block` `0.013708`, `mxfp8_g32` `0.017925` (`100.0%` top-1, `8/8` greedy), `q5_affine_g64` `0.019284`, `gguf_q4_k` `0.079079`, `q4_affine_g32` `0.080456`, `nvfp4_g16` `0.252315`, `nvfp4_g16_aq46` `0.380999`, `gguf_q2_0` `2.302979`; Real `qwen3-0.6b-bf16_vs_qwen3-0.6b-gsq-rco`: `status=ok`, `mean_kl=0.401752`, `max_kl=1.712455`, `top1=62.5% (5/8)`, `greedy=3/8`, `math=3/4`, `code=0/4`, `tok=resolved`, `finite=yes` | Standalone `aminfer quality-gate` matches identically (and runs cleanly with `tok=fallback` when `-k` is omitted on non-embedded `.amf`); `swapouts_delta = 0` |

### Verbatim `aminfer bench-m81` output

`TMPDIR=/Volumes/Akari/.tmp/m81-engine ./zig-out/bin/aminfer bench-m81`:

```text
bench.m81.chip                   Apple M4 (m4, spec_bus=120.0 GB/s)
bench.m81.swapouts_delta         0
bench.m81.g23.baseline.q4_g64    76.73 GB/s (63.9% bus)
bench.m81.g23.gemv.q4_affine_g64    90.96 GB/s (75.8% bus, 103.93 us)
bench.m81.g23.gemv.q4_affine_g32    87.34 GB/s (72.8% bus, 120.24 us)
bench.m81.g23.gemv.q5_affine_g64    80.54 GB/s (67.1% bus, 143.42 us)
bench.m81.g23.gemv.q6_affine_g64    84.74 GB/s (70.6% bus, 161.06 us)
bench.m81.g23.gemv.q8_affine_g64    86.73 GB/s (72.3% bus, 205.72 us)
bench.m81.g23.gemv.fp8_e4m3_block   65.22 GB/s (54.3% bus, 257.57 us)
bench.m81.g23.gemv.mxfp8_g32        66.32 GB/s (55.3% bus, 261.14 us)
bench.m81.g23.gemv.nvfp4_g16        72.81 GB/s (60.7% bus, 129.84 us)
bench.m81.g23.gemv.gguf_q8_0        82.60 GB/s (68.8% bus, 216.02 us)
bench.m81.g23.gemv.gguf_q4_0        87.22 GB/s (72.7% bus, 108.39 us)
bench.m81.g23.gemv.gguf_q4_k        60.62 GB/s (50.5% bus, 155.94 us)
bench.m81.g23.gemv.gguf_q2_0        54.94 GB/s (45.8% bus, 86.19 us)
bench.m81.g23.gemv.gguf_iq2_xxs     35.51 GB/s (29.6% bus, 122.26 us)
bench.m81.g23.gemv.gguf_tq1_0       21.16 GB/s (17.6% bus, 168.05 us)
bench.m81.g23.prefill64.nvfp4_g16   0.617 ms (0.87 TFLOPS)
bench.m81.g23.prefill64.mxfp8_g32   0.693 ms (0.77 TFLOPS)
bench.m81.g23.prefill64.fp8_block   0.668 ms (0.80 TFLOPS)
bench.m81.g23.prefill64.gguf_q4_k   0.604 ms (0.89 TFLOPS)
bench.m81.g24.none_dispatches    0
bench.m81.g24.act.none              m1=0.00 us m16=0.00 us mae=0.000000
bench.m81.g24.act.mxfp8_g32         m1=16.56 us m16=28.64 us mae=0.034028
bench.m81.g24.act.fp8_g128          m1=13.57 us m16=17.25 us mae=0.028059
bench.m81.g24.act.nvfp4_g16         m1=25.23 us m16=25.81 us mae=0.118255
bench.m81.g24.act.fouroversix_mae   m1=35.41 us m16=36.20 us mae=0.109436
bench.m81.g24.fouroversix        win_rate=99.9% mae_reduction=7.46%
bench.m81.c25.sme2_hw            active (c13_bus_gate=pass)
bench.m81.c25.sme2.nvfp4_g16        m4=50.92 GOPS m16=229.37 GOPS max_err=0.00e0 parity=pass
bench.m81.c25.sme2.mxfp4_g32        m4=99.40 GOPS m16=130.48 GOPS max_err=0.00e0 parity=pass
bench.m81.c25.sme2.q2_0             m4=94.59 GOPS m16=202.57 GOPS max_err=0.00e0 parity=pass
bench.m81.c25.sme2.tq1_0            m4=159.91 GOPS m16=298.93 GOPS max_err=0.00e0 parity=pass
bench.m81.c25.sme2.tq2_0            m4=87.61 GOPS m16=218.42 GOPS max_err=0.00e0 parity=pass
bench.m81.c25.sme2.iq2_xxs          m4=98.93 GOPS m16=185.45 GOPS max_err=1.22e-4 parity=pass
bench.m81.w21.dense_e2e_parity   pass
bench.m81.w22.moe_hetero_parity  pass
bench.m81.w22.draft_quant_parity pass
bench.m81.w22.fork_stq1_reject   pass
bench.m81.w24.tok.inkling2       46.18 MB/s (source=live_metadata)
bench.m81.w24.tok.hy4            27.57 MB/s (source=live_metadata)
bench.m81.w24.tpl_render_us      chatml=0.38 inkling2=0.49 hy4=0.25
bench.m81.c18.http_hardening     keepalive=pass hmac_sig_400=pass resp_completed_output=pass
quality_gate.swapouts_delta      0
quality_gate.synth.fp8_e4m3_block       mean_kl=0.013708 max_kl=0.018387 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.mxfp8_g32            mean_kl=0.017925 max_kl=0.031343 top1=100.0% (8/8) greedy=8/8 act_quant=none finite=yes
quality_gate.synth.q8_affine_g64        mean_kl=0.000387 max_kl=0.000628 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q6_affine_g64        mean_kl=0.005392 max_kl=0.008049 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q5_affine_g64        mean_kl=0.019284 max_kl=0.024888 top1=75.0% (6/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q4_affine_g32        mean_kl=0.080456 max_kl=0.118350 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.nvfp4_g16            mean_kl=0.252315 max_kl=0.338868 top1=50.0% (4/8) greedy=1/8 act_quant=none finite=yes
quality_gate.synth.nvfp4_g16_aq46       mean_kl=0.380999 max_kl=0.447204 top1=37.5% (3/8) greedy=0/8 act_quant=fouroversix_mae finite=yes
quality_gate.synth.gguf_q8_0            mean_kl=0.000356 max_kl=0.000722 top1=100.0% (8/8) greedy=8/8 act_quant=none finite=yes
quality_gate.synth.gguf_q4_k            mean_kl=0.079079 max_kl=0.134074 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.gguf_q2_0            mean_kl=2.302979 max_kl=4.410398 top1=0.0% (0/8) greedy=0/8 act_quant=none finite=yes
quality_gate.real.qwen3-0.6b-bf16_vs_qwen3-0.6b-gsq-rco status=ok mean_kl=0.401752 max_kl=1.712455 top1=62.5% (5/8) greedy=3/8 math=3/4 code=0/4 tok=resolved act_quant=none finite=yes
```

## M8.15: Sampling Penalty Parity, GSQ-RCO v2 Fidelity Hardening, and Router-Trace Loop Quality Gate

Recorded on Apple M4 (`Mac16,12`, `Apple9` GPU, `24 GiB` unified memory,
`120.0 GB/s` bus) during Milestone M8.15
([ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md#m815-gsq-rco-v2-fidelity-hardening-w8-w14-w17-w24-c18-c20),
[ADR-0025](../decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md)).

### 1. Root-cause diagnosis: `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`) vs uniform `3-bit` baseline (`qwen3-30b-a3b-3bit.amf`)

Controlled CLI and tensor-level comparison on `2026-11-24` isolated why
`qwen3-30b-a3b-gsq-rco.amf` (`2.700b`) exhibited open-`<think>` repetition loops
and factual collapse on long reasoning prompts while passing short M6 sanity
prompts (`Paris is the capital of France.`):

- **Tensor distribution & dequantization audit (`18,867` tensors, `10,309,206,016 B` file / `10,306,764,800 B` wired)**:
  - `5,178 / 6,144` routed experts (`84.3%` overall, `52.3%`–`92.2%` per layer
    — `67 / 128` at layer `0` up to `118 / 128` at layer `29`;
    `15,534` expert projection tensors) were double-quantized from the `3-bit`
    source (`mlx-community/Qwen3-30B-A3B-3bit`) down to `q2_affine` (`g64`);
  - All `3,140` `q3_affine` tensors (`2,898` hot expert tensors across `966`
    experts + `192` attention projections + `48` routers + `embed_tokens` +
    `lm_head`) are byte-for-byte identical (`same = 3140, diff = 0`) to
    `qwen3-30b-a3b-3bit.amf`;
  - Dequantizing the `q2_affine` expert projections against
    `qwen3-30b-a3b-3bit.amf` yields cosine similarity `cos = 0.9523–0.9543`
    (confirming bit-exact `(scale, bias)` and `dequant8_q2` kernel layout) but a
    per-matrix relative RMSE of `rel_rmse = 0.2988–0.3052` (`~30%` relative
    error compounded across 48 top-8 MoE layers), aggravated by shallow
    `embed_router_sample` (`48` static embedding tokens) heat selection.
- **Controlled CLI matrix (same engine binary)**:

| Model | Command & flags | Prompt | Measured behavior |
| --- | --- | --- | --- |
| `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`) | `chat --max-seq-len 4096 -n 512 -t 0.6 -V -C` (4 turns via `/retry`) | `hi` | Stochastic `<think>` loop without penalties: Turn 1 (`88 tok`), Retry 2 (`75 tok`), and Retry 3 (`74 tok`) terminate normally; Retry 1 (`512 tok`, `PrefixCache hit 19/20`) misses `</think>` at paragraph boundary and repeats the opening `<think>` paragraph 11+ times under unpenalized full-vocab sampling |
| `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`) | `chat --max-seq-len 4096 -n 300 -t 0 -V -C` & `decode --no-verify-crc -n 200 -t 0` | `告诉我黎曼猜想是什么` | Knowledge collapse + deterministic repetition loop even at greedy `T = 0` and `-N` (`--no-think`, `12.0–15.53 tok/s`, `swapouts_delta = 0`), hallucinating names/dates and looping template clauses |
| `qwen3-0.6b-gsq-rco.amf` (`4.348b`, `0` `q2_affine`) | `decode --no-verify-crc -n 128 -t 0` | `告诉我黎曼猜想是什么` | Coherent (`128 tok`, `83.10 tok/s`): accurately explains the Riemann zeta function $\zeta(s)$ without loops |
| `qwen3-30b-a3b-3bit.amf` (`3.50b` uniform, `0` `q2_affine`) | `chat --max-seq-len 512 -n 200 -t 0 -V -C` | `告诉我黎曼猜想是什么` | Accurate, zero repetition loops (`200 tok`, `12.8 tok/s`, `TTFT 653.2 ms`), deriving $\zeta(s) = \sum_{n=1}^{\infty} n^{-s}$, the $s = 1$ pole, and $\operatorname{Re}(s) = 1/2$ |

### 2. M8.15 verification summary and 3-prompt `W24` quality-gate telemetry (`7bcab5a`, `066bbb2`, `019bad7`, `b8c4824`)

| Track | Scope / Condition | Measured Result (M4) | Notes / Guard |
| --- | --- | --- | --- |
| **Zig Engine & CLI (`C18`, `C20`, `W8`, `W24`)** | `zig build test` (Debug & `-Doptimize=ReleaseSafe`) | **`480/480` passed** (`459` lib + `15` CLI + `6` C-ABI SDK across `10/10` steps; `0` `catch {}`, `zig fmt --check` clean) | Covers `BatchServeRunner` request/prompt-keyed `pending_jobs` concurrent slot binding + per-slot `DefaultPenaltyTracker` (`slot_penalties`) + constrained `temperatures = 0.0` prefill/decode gating, `aminfer chat` `--top-k`/`--top-p`/`--min-p`/`--repetition-penalty`/`--presence-penalty`/`--frequency-penalty` (`T = 0` bit-exact greedy bypass), `aminfer decode --dump-router-trace` single-frame 64 B/layer scratch redirect matching `stepDecodeCaptureLayerRouting` bit-for-bit, and `W24` 4-gram/consecutive repetition loop detector |
| **Python `GSQ-RCO v2` & SDK (`W8`, `W14`, `W17`, `C17`)** | `uv run --directory tools python -m unittest discover -s tests -v` & `python/tests` | **`115/115` `tools/` passed** (`14/14` in `test_m5_gsq_rco.py`) and **`6/6` `python/tests` SDK passed** | Verified activation-weighted OLS RCO (2D + 3D stacked MoE `col_weights` & per-expert `.counts` from `parse_gguf_imatrix`), `_BACKBONE_PROTECTED_ROLES` non-`q2_affine` floor (`target_bits=2.1` -> `ValueError`), `max_q2_share`, auto `4.15` Dense (`0` `q2_affine`) / `3.50` MoE `target_bits` (incl. `qwen4_exp` & HF 3D stacked `mlp.experts.*`), CLI low-bit `3-bit -> q2_affine` double-quantization guard (`--allow-retranscode-lowbit`), `--fine-rungs` (`g32` + `Q5`/`Q6`/`Q8_AFFINE` capped `<= 5` schemes/layer, uniform per-layer MoE `TensorLayout` under `--lut16-sme2`) round-trip via `aminfer verify-amf` & `aminfer decode`, and `decode --dump-router-trace` -> `expert-heat --calibration-jsonl` (`calibration_mode = "router_trace"`, no stderr warning) |
| **3-Prompt `W24` Quality Gate (`--steps 32`)** | `qwen3-0.6b-bf16.amf` vs `qwen3-0.6b-gsq-rco.amf` across math (`11` steps), code (`11` steps), and conceptual reasoning (`Riemann zeta`, `10` steps) (`ReleaseFast`) | `status=ok`, `mean_kl=0.313250`, `max_kl=1.712455`, `top1=62.5% (20/32)`, `greedy=4/32` (`math=4/11`, `code=0/11`, `reason=0/10`), `max_4gram=1`, `loop=no`, `tok=resolved`, `act_quant=none`, `finite=yes` | `quality_gate.swapouts_delta = 0` (external `vm_stat` swapouts `55156714 -> 55156714`, `delta = 0`); all `11/11` synthetic format pairs `finite=yes` |

### Verbatim 32-step 3-prompt `aminfer quality-gate` output (`ReleaseFast`, commits `7bcab5a` / `019bad7`)

`./zig-out/bin/aminfer quality-gate --ref /Volumes/Akari/models/amf/qwen3-0.6b-bf16.amf --cand /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf --config /Volumes/Akari/models/Qwen3-0.6B/config.json --tokenizer /Volumes/Akari/models/Qwen3-0.6B/tokenizer.json --steps 32`:

```text
quality_gate.swapouts_delta      0
quality_gate.synth.fp8_e4m3_block       mean_kl=0.013708 max_kl=0.018387 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.mxfp8_g32            mean_kl=0.017925 max_kl=0.031343 top1=100.0% (8/8) greedy=8/8 act_quant=none finite=yes
quality_gate.synth.q8_affine_g64        mean_kl=0.000387 max_kl=0.000628 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q6_affine_g64        mean_kl=0.005392 max_kl=0.008049 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q5_affine_g64        mean_kl=0.019284 max_kl=0.024888 top1=75.0% (6/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.q4_affine_g32        mean_kl=0.080456 max_kl=0.118350 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.nvfp4_g16            mean_kl=0.252315 max_kl=0.338868 top1=50.0% (4/8) greedy=1/8 act_quant=none finite=yes
quality_gate.synth.nvfp4_g16_aq46       mean_kl=0.380999 max_kl=0.447204 top1=37.5% (3/8) greedy=0/8 act_quant=fouroversix_mae finite=yes
quality_gate.synth.gguf_q8_0            mean_kl=0.000356 max_kl=0.000722 top1=100.0% (8/8) greedy=8/8 act_quant=none finite=yes
quality_gate.synth.gguf_q4_k            mean_kl=0.079079 max_kl=0.134074 top1=87.5% (7/8) greedy=4/8 act_quant=none finite=yes
quality_gate.synth.gguf_q2_0            mean_kl=2.302979 max_kl=4.410398 top1=0.0% (0/8) greedy=0/8 act_quant=none finite=yes
quality_gate.real.qwen3-0.6b-bf16_vs_qwen3-0.6b-gsq-rco status=ok mean_kl=0.313250 max_kl=1.712455 top1=62.5% (20/32) greedy=4/32 math=4/11 code=0/11 reason=0/10 max_4gram=1 loop=no tok=resolved act_quant=none finite=yes
```



