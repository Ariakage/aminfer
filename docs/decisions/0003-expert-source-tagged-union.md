# ADR-0003: `ExpertSource` tagged union and the reserved Thunderbolt peer

- Status: Accepted
- Date: 2026-11-20
- Source: project charter (user); "Mac interconnect inference is long-term,
  only reserve the interface" (user, init session)

## Context

Experts can be resident in wired DRAM, cold on the internal NVMe SSD, or — in
a future cluster — in another Mac's memory over Thunderbolt. Adding the third
placement later must not force a rewrite of the scheduler or the Metal
kernels.

## Decision

1. `ExpertSource` is a Zig tagged union with exactly three variants:
   `local_wired_ram`, `local_nvme_ssd` and `thunderbolt_peer`.
2. Kernels and the scheduler consume only resident bytes — a wired region or a
   filled slot. Requesting bytes from a non-resident source returns an error;
   loading is an explicit, asynchronous step.
3. `local_nvme_ssd` locations are validated at construction: offset and length
   are 16 KiB multiples and the length is non-zero.
4. `thunderbolt_peer` is reserved: it carries a placeholder location and
   reports that its transport is unavailable. No transport is implemented.

## Consequences

- Switches over `ExpertSource` are exhaustive, so a new variant is a compile
  error wherever it is not handled.
- Remote experts will reuse the cold-expert machinery with their own latency
  and bandwidth model.

## Transport security and authentication

- Baseline cluster execution assumes a dedicated, trusted point-to-point
  Thunderbolt Bridge / RDMA link between peers rather than an untrusted shared
  LAN (`L5`).
- Every framed cluster RPC (`FrameChannel` in
  [`src/cluster/transport.zig`](../../src/cluster/transport.zig)) supports
  optional pre-shared key (`psk`) HMAC-SHA256 authentication across all frame
  headers and payloads (`PeerWorkerConfig.psk` and `CoordinatorConfig.psk`,
  landed in `0fe3d1e`). When `psk` is configured via `--cluster-psk` or
  `AMINFER_CLUSTER_PSK` (`d52873b`), unauthenticated or mismatched peers are
  rejected with `error.AuthenticationFailed`.

## Evidence

- The baseline machine is Thunderbolt 4 class (40 Gb/s): IP over Thunderbolt
  Bridge only. RDMA over Thunderbolt requires Thunderbolt 5 and macOS 26.2+
  (Apple TN3205).

## Open items

- Transport choice (TCP versus RDMA) and expert-parallel protocol — deferred to
  the scale-out work (status rows X1, X2).
