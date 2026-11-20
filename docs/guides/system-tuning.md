# System tuning: wired memory and monitoring

> These commands change kernel tunables with `sudo`. The values reset on
> reboot. Read the risks before using the benchmark profile.

## Two independent limits

| Limit | Governs | Baseline default (M4, 24 GB) |
| --- | --- | --- |
| `vm.user_wire_limit` (per process), `vm.global_user_wire_limit` (system) | `mlock` | 19,585,050,869 B (≈ 18.24 GiB) |
| `iogpu.wired_limit_mb` (0 = system default) | GPU-resident working set | `recommendedMaxWorkingSetSize` = 18,186 MiB |

aminfer computes its wired budget at startup from both limits and a headroom
reserved for macOS (default 3 GiB), and refuses to load a plan that does not
fit ([ADR-0012](../decisions/0012-wired-memory-budget-headroom.md)).

Inspect the current values:

```sh
sysctl iogpu.wired_limit_mb vm.user_wire_limit vm.global_user_wire_limit hw.memsize
```

## Profiles for the 24 GB baseline

### Default — no tuning

The 3-bit primary target with a 32K context is estimated to fit the default
limits ([../architecture/memory.md](../architecture/memory.md#budget-estimates)).

### Everyday raised limits — 21 GiB, keeps 3 GiB for macOS

```sh
sudo sysctl -w iogpu.wired_limit_mb=21504
sudo sysctl -w vm.global_user_wire_limit=22548578304
sudo sysctl -w vm.user_wire_limit=22548578304
```

### Dedicated benchmark session — 22.5 GiB ceiling

```sh
sudo sysctl -w iogpu.wired_limit_mb=23040
sudo sysctl -w vm.global_user_wire_limit=24159191040
sudo sysctl -w vm.user_wire_limit=24159191040
```

Risks: this leaves about 1.5 GiB to macOS and every other process. Close other
applications first; expect stalls if the system needs more memory. Use it only
for measured benchmark runs.

Writing the `vm.*` wire limits has not yet been verified on the baseline (it
requires `sudo`); record the outcome in [../process.md](../process.md) the
first time it is done.

### Reset

```sh
sudo sysctl -w iogpu.wired_limit_mb=0
sudo sysctl -w vm.global_user_wire_limit=19585050869
sudo sysctl -w vm.user_wire_limit=19585050869
```

Use the defaults recorded on your own machine; a reboot also restores them.

## Monitoring

```sh
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
vm_stat 1            # the "Swapouts" counter must not grow during a run
sysctl vm.swapusage
memory_pressure
```
