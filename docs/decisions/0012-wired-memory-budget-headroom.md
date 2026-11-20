# ADR-0012: Wired memory budget with headroom

- Status: Accepted
- Date: 2026-11-20
- Source: proposal discussed and accepted by the user (init session)

## Context

The charter's baseline setting `sudo sysctl -w iogpu.wired_limit_mb=23040`
(22.5 GiB) leaves about 1.5 GiB to macOS on a 24 GB machine. It also covers
only the GPU working set: `mlock` is governed separately by
`vm.user_wire_limit` and `vm.global_user_wire_limit`, 18.24 GiB by default on
the baseline.

## Decision

1. The wired budget is computed at startup as the minimum of: the configured
   cap; physical memory minus the headroom (default 3 GiB); `vm.user_wire_limit`;
   `vm.global_user_wire_limit`; Metal's `recommendedMaxWorkingSetSize`.
2. A model plan that exceeds the budget is rejected with an error that names
   the binding limit — never silently degraded to pageable memory.
3. `iogpu.wired_limit_mb=23040` is documented as the ceiling for dedicated
   benchmark sessions on the 24 GB baseline, together with the `vm.*` wire
   limits it needs ([../guides/system-tuning.md](../guides/system-tuning.md)).

## Consequences

- The default configuration runs without `sudo` whenever the plan fits the
  default limits.
- Larger plans need raised limits or cold experts on NVMe.

## Evidence

- Baseline defaults: `vm.user_wire_limit` = `vm.global_user_wire_limit` =
  19,585,050,869 B; `iogpu.wired_limit_mb` = 0 with
  `recommendedMaxWorkingSetSize` = 18,186 MiB.
- Budget estimates: Qwen3-30B-A3B at 3 bits with a 32K fp16 KV cache needs
  about 16.5 GiB; at 4 bits about 20.1 GiB
  ([../architecture/memory.md](../architecture/memory.md#budget-estimates)).

## Open items

- None.
