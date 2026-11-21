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

Measured on 2026-11-21 on the Apple M4 24 GiB baseline (`aminfer bench-m5`,
`verify-amf`, `bench-decode`, `decode`, `spec-decode`, and live `aminfer serve`,
ReleaseFast at commits `b78fc8e`–`d5069a2`, `swapouts_delta = 0` across every
archived run).

> **Host contention note:** During the M5 benchmark window an unrelated user
> GPU workload (`eval/benchmark_suite.py` on `Qwen3.5-0.8B`) was running
> concurrently on the shared Apple M4 GPU and unified memory bus. All A/B
> comparisons below (`BF16` vs `GSQ-RCO` vs `GSQ-MinMax`, `L1` vs `L1+C6`,
> `PEARL off` vs `PEARL on/auto`, and live `aminfer serve`) were therefore
> executed back-to-back under the same host conditions, and every archived run
> passed the strict `vmstat.checkNoSwap` (`swapouts_delta = 0`) gate.

### G13 `q2_affine` Metal 4 calibration & scheme-bucketed heterogeneous MoE (`aminfer bench-m5`)

| Kernel / Profile | Shape / Geometry | Bytes/disp | vs `q4` bytes | Buckets | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemv_q2` | `1×2048×2048` | 1,318,912 | — | — | 56.70 | 23.26 | 19.4% |
| `gemv_q2` | `1×6144×2048` | 3,948,544 | — | — | 82.63 | 47.79 | 39.8% |
| `gemv_q2` | `1×2048×768` | 497,152 | — | — | 46.03 | 10.80 | 9.0% |
| `gemv_q2` | `1×4096×4096` | 5,259,264 | — | — | 192.75 | 27.29 | 22.7% |
| `gemm_q2` | `16×2048×2048` | 1,441,792 | — | — | 302.20 | 4.77 | 4.0% |
| `gemm_q2` | `16×4096×4096` | 5,505,024 | — | — | 669.16 | 8.23 | 6.9% |
| `gemm_q2` | `64×2048×2048` | 1,835,008 | — | — | 514.61 | 3.57 | 3.0% |
| `gemm_q2` | `64×4096×4096` | 6,291,456 | — | — | 1,594.52 | 3.95 | 3.3% |
| MoE `uniform_q4` | `M=1, E=128, k=8, U=8` | 21,242,368 | `0.0%` | 1 | 449.06 | 47.30 | 39.4% |
| MoE `uniform_q3` | `M=1, E=128, k=8, U=8` | 16,523,776 | `-22.2%` | 1 | 450.23 | 36.70 | 30.6% |
| MoE `uniform_q2` | `M=1, E=128, k=8, U=8` | 11,805,184 | `-44.4%` | 1 | 663.26 | 17.80 | 14.8% |
| MoE `hetero_q2_q3_q4` | `M=1, E=128, k=8, U=8` | 15,344,128 | **`-27.8%`** | 3 | 530.87 | 28.90 | 24.1% |
| MoE `uniform_q4` | `M=8, E=128, k=8, U=24` | 63,770,624 | `0.0%` | 1 | 2,208.36 | 28.88 | 24.1% |
| MoE `uniform_q3` | `M=8, E=128, k=8, U=24` | 49,614,848 | `-22.2%` | 1 | 2,102.16 | 23.60 | 19.7% |
| MoE `uniform_q2` | `M=8, E=128, k=8, U=24` | 35,459,072 | `-44.4%` | 1 | 2,137.79 | 16.59 | 13.8% |
| MoE `hetero_q2_q3_q4` | `M=8, E=128, k=8, U=24` | 46,075,904 | **`-27.7%`** | 3 | 2,326.66 | 19.80 | 16.5% |

Notes:
- Zero-sync, zero-allocation 3-bucket CSR dispatch (`ami_moe_route_topk_bucketed`
  + per-scheme specialized Metal 4 pipelines) cuts MoE weight traffic by
  **`-27.7%` to `-27.8%`** vs uniform `q4` with `0` hot-path heap allocations
  and `0` CPU-GPU syncs; `SpecEngine` automatically wires `Qwen3Model`'s
  per-expert static byte & per-scheme bandwidth tables into `D7`
  `PruneCostModel` ([ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md)).

### W14 + G13 real `Qwen3-0.6B` GSQ + RCO vs naive min-max vs uniform BF16 (`ADR-0017`)

Side-by-side comparison on real `Qwen/Qwen3-0.6B` (28 layers, tied embeddings,
310 tensors; all files verified with `aminfer verify-amf`, `swapouts_delta = 0`):

| Profile (`target-bits`) | Tensor mix | File size (MB) | `wired_mib` (`seq=512` / `decode`) | `bytes/tok` (`seq=512`) | Mean quant MSE (RCO vs MinMax) | `bench-decode` (`seq=512`) | EN / ZH `decode` (`tok/s`) | Output coherence |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Uniform `bf16` baseline | `bf16=310` | 1,193.89 | 1,251 / 1,195 | 1,250,936,832 | `0` (ref) | 37.88 tok/s (`26.40 ms`) | 36.99 / 40.68 | Coherent EN & ZH |
| **`GSQ + RCO` (`4.15` eff-b, archived)** | `bf16=113, q3=84, q4=113` | **310.95** (`-74.0%`) | **409 (`-67.3%`) / 353 (`-70.5%`)** | **367,993,408 (`-70.6%`)** | **`1.3506e-5`** (`-17.95%` vs MinMax, `1.22×` lower) | **89.49 tok/s (`2.36×`)** | **101.32 (`2.74×`) / 92.19 (`2.27×`)** | Coherent EN & ZH |
| `GSQ MinMax` (`4.15` eff-b, `--no-rco`) | `bf16=113, q3=83, q4=114` | 310.95 | 409 / 353 | 367,993,408 | `1.6461e-5` (baseline) | 73.86 tok/s (`1.95×`) | 95.16 / 105.08 | Coherent EN & ZH |
| `GSQ + RCO` (`3.25` eff-b, stress sweep) | `bf16=113, q2=149, q3=24, q4=24` | 243.84 (`-79.6%`) | 345 (`-72.4%`) / 289 (`-75.8%`) | 300,884,544 (`-75.9%`) | **`6.7565e-5`** (`-46.86%` vs MinMax, `1.88×` lower) | 59.34 tok/s (`1.76×` vs `33.71`) | 69.72 / 96.64 | Degraded (58.4% 2-bit on 0.6B dense) |
| `GSQ MinMax` (`3.25` eff-b, `--no-rco`) | `bf16=113, q2=146, q3=31, q4=20` | 243.84 | 345 / 289 | 300,884,544 | `1.2715e-4` (baseline) | 69.00 tok/s (`2.05×` vs `33.71`) | 96.55 / 98.61 | Collapsed (`tr tr`, `Aتم...`) |

Key takeaways (directly validating [ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md)):
- Offline RCO alternating ridge-OLS scale/bias refinement cuts reconstruction
  MSE by **`-17.95%` (`1.22×` lower)** at `4.15` eff-bits (`-18.03%` on
  matched-scheme tensors) and **`-46.86%` (`1.88×` lower)** at `3.25` eff-bits
  (`-47.49%` on matched-scheme tensors) with zero runtime overhead.
- At `4.15` eff-bits (`/Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf`),
  `Qwen3-0.6B` drops wired RAM by **`-67.3%` to `-70.5%`**, cuts `bytes/tok` by
  **`-70.6%` to `-73.5%`**, accelerates decode **`2.27×`–`2.74×`**, and stays
  coherent in English and Chinese with `swapouts_delta = 0`.
- Pushing a tiny `0.6B` dense model down to `3.25` eff-bits forces `149/197`
  matrices (`58.4%` weights) into `q2_affine` (because `q4` tied embeddings
  alone consume 26% of parameters at `4.50` eff-bits), degrading output quality
  and confirming why `q2_affine` belongs on low-sensitivity / cold MoE experts
  and why `ADR-0017` mandates uniform-baseline quality comparison.

### C6 repo-scale `F_NOCACHE` NVMe `.ang` N-gram table (`aminfer-repo.ang`)

Built `/Volumes/Akari/models/amf/aminfer-repo.ang` from
`/Volumes/Akari/workflow/aminfer/{src,docs,tools}` using
`aminfer-tools build-ngram` + `Qwen3-0.6B/tokenizer.json`:
**`134,234,112 B` (`128.02 MiB`)**, `8,192` × 16 KiB bucket pages, `879,949`
`2..4`-gram entries (`0` dropped), `88` files, `1,194,964` tokens,
`table_crc32 = 0xfe3a0eb2`, `header_crc32 = 0x4523e5c0`, `512 KiB` wired LRU
slot cache (`32` × 16 KiB pages, `F_NOCACHE`).

- **`bench-m5` `F_NOCACHE` lookup latency** (`256` buckets / `4.02 MiB` `.ang`):
  - Cold `F_NOCACHE` read p50: **`2.79 us`** (p99: **`16.04 us`**)
  - Wired LRU cache hit p50: **`0.08 us` (`80 ns`)**
  - Async `kqueue_aio` `prefetch()` + `query()` hit p50: **`0.13 us`**
- **Real-checkpoint `T = 0` lossless `spec-decode` with vs without `--ngram-nvme aminfer-repo.ang`**:

| Model & Prompt | Mode | Bit-exact vs `decode` | `tau` (mean accept) | Rounds | `ngram` queries / hits / seed hits | Mean `.ang` query (us) | `bytes/acc tok` | Decode `tok/s` | `swapouts_delta` |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig) | `decode` baseline | ref (`32/32`) | `1.000` | 31 | — | — | 1,197.8 MB | 41.48 | 0 |
| `Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig) | `spec-decode l1` | `YES (32/32)` | `2.818` | 11 | — | — | 424.2 MB | 106.33 (`2.56×`) | 0 |
| **`Qwen3-0.6B` BF16 (`APPLE_PAGE_SIZE` Zig)** | **`spec-decode l1 + C6 .ang`** | **`YES (32/32)`** | **`5.167`** | **6** | **`3 / 3 / 2`** | **`396.10`** | **231.5 MB (`-45.4%` vs L1)** | **216.28 (`2.03×` vs L1, `5.21×` vs base)** | **0** |
| `Qwen3-0.6B` BF16 (`alignUp` Zig) | `decode` baseline | ref (`64/64`) | `1.000` | 63 | — | — | 1,204.1 MB | 36.70 | 0 |
| `Qwen3-0.6B` BF16 (`alignUp` Zig) | `spec-decode l1` | `YES (64/64)` | `1.500` | 42 | — | — | 800.6 MB | 55.82 (`1.52×`) | 0 |
| **`Qwen3-0.6B` BF16 (`alignUp` Zig)** | **`spec-decode l1 + C6 .ang`** | **`YES (64/64)`** | **`1.615`** | **39** | **`34 / 18 / 1`** | **`913.44`** | **743.4 MB (`-7.1%` vs L1)** | **56.13 (`1.53×` vs base)** | **0** |
| `Qwen3-14B-4bit` (`page.zig` Zig test) | `decode` baseline | ref (`48/48`) | `1.000` | 47 | — | — | 7,887.6 MB | 5.95 | 0 |
| `Qwen3-14B-4bit` (`page.zig` Zig test) | `spec-decode l1` | `YES (48/48)` | `1.469` | 32 | — | — | 5,367.4 MB | 5.98 | 0 |
| **`Qwen3-14B-4bit` (`page.zig` Zig test)** | **`spec-decode l1 + C6 .ang`** | **`YES (48/48)`** | **`1.621`** | **29** | **`27 / 16 / 3`** | **`1,207.78`** | **4,864.4 MB (`-9.4%` vs L1)** | 3.72 (larger tree `4.86` vs `3.16` nodes under host GPU load) | **0** |

### D8 target-latency-gated PEARL async pipelined small draft (`ADR-0008`)

- **Gate verification (`--pearl auto`, default `pearl_min_target_us = 40 ms`)**:
  - `Qwen3-14B-4bit + Qwen3-0.6B` (`~95–177 ms/step`): automatically gates **`enabled = yes` (`own_queue = yes`)**.
  - `Qwen3-30B-A3B-3bit + EAGLE-3` (48L MoE, `--cold-experts-per-layer 64` under host load): automatically stays **`enabled = no`** (`0` PEARL rounds, `swapouts_delta = 0`), obeying [ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md).
- **`bench-m5` synthetic PEARL check**: `own_compute_queue = yes`,
  `bit_exact_off = yes`, `bit_exact_on = yes`, `pearl_rounds = 6`,
  **`pearl_hits = 6/6` (`0` rollbacks)**, `pearl_saved_ms = 7.001 ms`,
  `tau = 3.167`, `swapouts_delta = 0`.
- **Real `Qwen3-14B-4bit + Qwen3-0.6B` (`T = 0`, 64 tokens, 100% bit-exact `token_ids`, `swapouts_delta = 0`)**:

| Run | Draft model & tree | PEARL mode | `wired_mib` | Bit-exact vs `decode` | `tau` | `pearl_hits` / `prefix` / `rollbacks` | `pearl_saved_ms` | `draft_ms` / `verify_ms` | `bytes/acc tok` | Decode `tok/s` |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |
| `D8-1` baseline | None | — | 8,006 | ref (`64/64`) | `1.000` | — | — | `—` / `177.1 ms` | 7,885.6 MB | 5.65 |
| `D8-2` `0.6B-GSQ-RCO` | `4.15b` (`d=4, w=1, k=1`) | `off` | **8,432** | **`YES (64/64)`** | `2.333` | `0 / 0 / 0` | `0.0` | `28.0 / 334.5 ms` | 3,377.3 MB (`-57.2%`) | **6.13 (`1.08×`)** |
| `D8-4` `0.6B-BF16` | `bf16` (`d=4, w=1, k=1`) | `off` | 9,274 | **`YES (64/64)`** | `2.625` | `0 / 0 / 0` | `0.0` | `54.8 / 342.8 ms` | 3,002.4 MB (`-61.9%`) | **6.18 (`1.10×`)** |
| **`D8-5` `0.6B-BF16`** | `bf16` (`d=4, w=1, k=1`) | **`on` (`own_queue=yes`)** | 9,274 | **`YES (64/64)`** | **`2.739`** | **`7 / 8 / 16` (`65.2%` hit+prefix)** | **`2,728.4`** | **`48.3` (`-11.8%`) / `459.7 ms`** | **2,877.3 MB (`-63.5%`)** | 4.84 (single-GPU contention) |
| **`D8-3w` `0.6B-BF16` (wide)** | `bf16` (`cap=16, d=4, w=2`) | **`auto -> ON`** | 9,347 | **`YES (64/64)`** | **`3.150`** | **`5 / 10 / 15` (`75.0%` hit+prefix)** | **`2,175.8`** | `51.0 / 665.9 ms` | **2,502.6 MB (`-68.3%`)** | 4.20 (single-GPU contention) |

Architectural finding (`ADR-0008`): PEARL's asynchronous lock-free worker
correctly overlaps round `r + 1` pre-drafting with `target.verifyTree(T_r)`,
achieving `65.2%`–`75.0%` hit+prefix reuse, hiding `2.18–2.73 s` of sequential
draft latency (`draft_ms` down `-11.8%`), raising `tau` (`2.625 -> 2.739`), and
maintaining **64/64 (100%) bit-exact `T = 0` parity** with $O(1)$ KV rollback.
On single-GPU 10-core Apple M4 (further loaded by the concurrent host GPU
process), running `small_draft` on a second Metal GPU queue simultaneously with
`14B` `verifyTree` contends for the same GPU compute cores and 120 GB/s bus
(`verify_ms` `+116.9 ms/round`), offsetting the `2.73 s` overlapped draft
savings. This empirically validates keeping `--pearl` configurable (`auto|on|off`)
and motivates running the small draft off-GPU (ANE / SME2) or on higher-SM
Max/Ultra chips to turn overlapped draft hiding into net wall-clock gain.

### C7 + C8 live `aminfer serve` HTTP/1.1 + SSE & concurrency load test

- **`bench-m5` zero-allocation `kqueue` loopback microbenchmark**:
  - `GET /health` p50: **`100.46 us`**
  - Single-client `POST /v1/chat/completions` (`stream: true`) TTFT p50: **`131.21 us`**, E2E p50: **`232.54 us`**
  - `8` concurrent SSE clients: TTFT p50 **`896.13 us`**, p99 **`1,130.75 us`**, aggregate **`84,495.4 tok/s`**
  - Steady-state `malloc_zone_statistics` delta: **`0` blocks / `0` B** (`all_ok = yes`, `swapouts_delta = 0`)
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

### Verbatim `aminfer bench-m5`, `build-ngram`, `gsq-convert`, and `verify-amf` raw stdout (commit `b78fc8e`)

```text
=== M5 Serving, Pipelining, C6 NVMe N-Gram & ADR-0017 GSQ+RCO Hybrid Quant Suite ===

--- G13 Metal 4 q2_affine Kernel Bandwidth Calibration ---
kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
gemv_q2   1x2048x2048           1318912      64         56.70        23.26     19.4
gemv_q2   1x6144x2048           3948544      21         82.63        47.79     39.8
gemv_q2   1x2048x768             497152     170         46.03        10.80      9.0
gemv_q2   1x4096x4096           5259264      16        192.75        27.29     22.7
gemm_q2   16x2048x2048          1441792      64        302.20         4.77      4.0
gemm_q2   16x4096x4096          5505024      16        669.16         8.23      6.9
gemm_q2   64x2048x2048          1835008      64        514.61         3.57      3.0
gemm_q2   64x4096x4096          6291456      16       1594.52         3.95      3.3

--- G13 Scheme-Bucketed Heterogeneous MoE ({q2, q3, q4}) vs Uniform CSR Gather ---
profile          m    E    k   U    bkts    bytes/disp   vs_q4(%)   latency(us)   beta(GB/s)   bus(%)
uniform_q4       1    128  8   8    1         21242368        0.0        449.06        47.30     39.4
uniform_q3       1    128  8   8    1         16523776      -22.2        450.23        36.70     30.6
uniform_q2       1    128  8   8    1         11805184      -44.4        663.26        17.80     14.8
hetero_q2_q3_q4  1    128  8   8    3         15344128      -27.8        530.87        28.90     24.1
uniform_q4       8    128  8   24   1         63770624        0.0       2208.36        28.88     24.1
uniform_q3       8    128  8   24   1         49614848      -22.2       2102.16        23.60     19.7
uniform_q2       8    128  8   24   1         35459072      -44.4       2137.79        16.59     13.8
hetero_q2_q3_q4  8    128  8   24   3         46075904      -27.7       2326.66        19.80     16.5

--- C6 F_NOCACHE NVMe .ang N-Gram Lookup Latency ---
ngram.file_bytes                 4210688
ngram.buckets                    256
ngram.total_entries              1024
ngram.wired_cache_kib            512
ngram.cold_nvme_p50_us           2.79
ngram.cold_nvme_p99_us           16.04
ngram.cache_hit_p50_us           0.08
ngram.prefetch_hit_p50_us        0.13
ngram.queries                    345
ngram.hits                       345
ngram.seed_hits                  345
ngram.nvme_reads                 87

--- D8 PEARL Async Pipelined Small-Draft Synthetic Parity & Overlap ---
pearl_synth.own_compute_queue    yes
pearl_synth.bit_exact_off        yes
pearl_synth.bit_exact_on         yes
pearl_synth.base_tok_per_s       628.30
pearl_synth.off_tok_per_s        613.35
pearl_synth.on_tok_per_s         524.60
pearl_synth.mean_accept_len      3.167
pearl_synth.pearl_rounds         6
pearl_synth.pearl_hits           6
pearl_synth.pearl_rollbacks      0
pearl_synth.pearl_saved_ms       7.001

--- C7 + C8 kqueue HTTP/1.1 + SSE Server Loopback Microbenchmark ---
server.health_p50_us             100.46
server.single_ttft_p50_us        131.21
server.single_e2e_p50_us         232.54
server.concurrent_clients        8
server.concurrent_ttft_p50_us    896.13
server.concurrent_ttft_p99_us    1130.75
server.concurrent_tok_per_s      84495.4
server.steady_malloc_blocks      0
server.steady_malloc_bytes       0
server.all_ok                    yes

M5 Swap Check
m5.swap.status                   ok
m5.swap.swapouts_delta           0
m5.swap.pageouts_delta           23

=== BUILD REPO-SCALE C6 AMINFER-REPO.ANG ===
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

=== BUILD QWEN3-0.6B-GSQ-RCO.AMF (4.15 eff-bits, RCO vs MinMax) ===
gsq-converted 310 tensors into /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf (310951936 bytes, target=4.15b, effective=4.149b, rco=on) in 460.9 s [bf16=113, q3_affine=84, q4_affine=113]
gsq-converted 310 tensors into /Volumes/Akari/.tmp/qwen3-0.6b-gsq-minmax-415.amf (310951936 bytes, target=4.15b, effective=4.149b, rco=off) in 121.2 s [bf16=113, q3_affine=83, q4_affine=114]
RCO   : target=4.15 eff=4.1494 alg_bytes=309157888 mean_quant_mse=1.350598e-05 counts={'q3_affine': 84, 'q4_affine': 113} share={'q3_affine': 0.35184, 'q4_affine': 0.64805}
MinMax: target=4.15 eff=4.1494 alg_bytes=309157888 mean_quant_mse=1.646058e-05 counts={'q3_affine': 83, 'q4_affine': 114} share={'q3_affine': 0.35184, 'q4_affine': 0.64805}
Overall RCO MSE reduction: -17.95% (1.22x lower MSE)
Matched-scheme RCO MSE reduction: -18.03% (RCO=1.186088e-05 vs MinMax=1.446955e-05)
amf.status       ok
amf.model_id     qwen3-0.6b-gsq-rco
amf.tensors      310
amf.file_bytes   310951936
amf.wired_bytes  310886400
amf.cold_bytes   0

=== 3.25 EFF-BITS STRESS SWEEP (RCO vs MinMax) ===
packed 310 tensors into /Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf (243843072 bytes) in 655.8 s [bf16=113, q2_affine=149, q3_affine=24, q4_affine=24]
gsq-converted 310 tensors into /Volumes/Akari/.tmp/qwen3-0.6b-gsq-minmax.amf (243843072 bytes, target=3.25b, effective=3.249b, rco=off) in 131.5 s [bf16=113, q2_affine=146, q3_affine=31, q4_affine=20]
RCO vs MinMax MSE reduction: -46.86% (1.88x lower MSE; matched-scheme -47.49%)
```

### Verbatim real-checkpoint `W14/G13`, `C6`, `D8` PEARL, and live `aminfer serve` raw stdout (commits `b78fc8e`–`d5069a2`)

```text
--- W14/G13 bench-decode BF16 vs GSQ-RCO-4.15 (seq=512, 32 steps) ---
BF16:         wired_mib=1251  measured_ms_per_tok=26.401  measured_tok_per_s=37.88  measured_bytes_tok=1250936832  swapouts_delta=0
GSQ-RCO-4.15: wired_mib=409   measured_ms_per_tok=11.174  measured_tok_per_s=89.49  measured_bytes_tok=367993408   swapouts_delta=0
GSQ-MM-4.15:  wired_mib=409   measured_ms_per_tok=13.540  measured_tok_per_s=73.86  measured_bytes_tok=367993408   swapouts_delta=0

--- W14/G13 EN & ZH decode GSQ-RCO-4.15 ---
EN: wired_mib=353 decode_ms_per_tok=9.870  decode_tok_per_s=101.32 bytes_per_tok=317301312 swapouts_delta=0
    text: Explain Apple Silicon Unified Memory in Two Clear Sentences:\n\n1. **Unified Memory (US)** is a type of memory that is used in a single chip...
ZH: wired_mib=353 decode_ms_per_tok=10.847 decode_tok_per_s=92.19  bytes_per_tok=314892864 swapouts_delta=0
    text: 请用“统一内存架构”和“内存统一化”的优势介绍 Apple Silicon 架构。<|im_end|>

--- C6 repo-scale aminfer-repo.ang on Qwen3-0.6B BF16 (T=0, 32/32 bit-exact) ---
base:    decode_ms_per_tok=24.107 decode_tok_per_s=41.48  bytes_per_tok=1197836288 swapouts_delta=0
l1_only: rounds=11 mean_accept_len=2.818 decode_tok_per_s=106.33 bytes_per_accepted_tok=424181958 swapouts_delta=0
l1+c6:   rounds=6  mean_accept_len=5.167 decode_tok_per_s=216.28 bytes_per_accepted_tok=231527127 swapouts_delta=0
         ngram_nvme=on(buckets=8192, entries=879949, wired_kib=512) queries=3 hits=3 seed_hits=2 cache_hits=2 nvme_reads=4 mean_query_us=396.10
         token_ids=[284, 220, 16, 15, 17, 19, 353, 220, 16, 15, 17, 19, 353, 220, 16, 15, 17, 19, 353, 220, 16, 15, 17, 19, 353, 220, 16, 15, 17, 19, 353, 220]

--- D8 PEARL A/B on real Qwen3-14B-4bit + Qwen3-0.6B (T=0, 64/64 bit-exact) ---
D8-1 baseline:           wired_mib=8006 decode_ms_per_tok=177.077 decode_tok_per_s=5.65 bytes_per_tok=7885611840 swapouts_delta=0
D8-2 0.6B-GSQ-RCO off:   wired_mib=8432 rounds=27 mean_accept_len=2.333 draft_ms=28.015 verify_ms=334.532 decode_tok_per_s=6.13 bytes_per_acc=3377305783 pearl_mode=off swapouts_delta=0
D8-4 0.6B-BF16 off:      wired_mib=9274 rounds=24 mean_accept_len=2.625 draft_ms=54.792 verify_ms=342.828 decode_tok_per_s=6.18 bytes_per_acc=3002390344 pearl_mode=off swapouts_delta=0
D8-5 0.6B-BF16 on:       wired_mib=9274 rounds=23 mean_accept_len=2.739 draft_ms=48.308 verify_ms=459.710 decode_tok_per_s=4.84 bytes_per_acc=2877284135 pearl_mode=on(enabled=yes,own_queue=yes) pearl_rounds=23 pearl_hits=7 pearl_partial_prefix_hits=8 pearl_rollbacks=16 pearl_saved_ms=2728.414 swapouts_delta=0
D8-3w 0.6B-BF16 auto->ON: wired_mib=9347 rounds=20 mean_accept_len=3.150 draft_ms=50.953 verify_ms=665.902 decode_tok_per_s=4.20 bytes_per_acc=2502579718 pearl_mode=auto(enabled=yes,own_queue=yes) pearl_rounds=20 pearl_hits=5 pearl_partial_prefix_hits=10 pearl_rollbacks=15 pearl_saved_ms=2175.841 swapouts_delta=0
30B+EAGLE-3 --pearl auto: wired_mib=8084 (64 cold/layer) mode=eagle3 pearl=OFF swapouts_delta=0

--- C7 + C8 live aminfer serve (Qwen3-0.6B BF16, loopback ephemeral port) ---
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
serve.status                     stopped
serve.total_requests             20
serve.rejected_503               0
serve.cancelled                  0
serve.swapouts_delta             0
```
