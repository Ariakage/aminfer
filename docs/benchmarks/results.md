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
