# Experts and storage

> Design reference; status lives in [../status.md](../status.md).

## ExpertSource

Accepted in [ADR-0003](../decisions/0003-expert-source-tagged-union.md):
where an expert's weights live is a Zig tagged union, so new placements never
change scheduler or kernel signatures.

```zig
pub const ExpertSource = union(enum) {
    local_wired_ram: WiredExpert, // view into a wired region; already GPU-visible
    local_nvme_ssd: NvmeExpert,   // 16 KiB-aligned blob in an .amf file; needs a slot first
    thunderbolt_peer: PeerExpert, // reserved for remote expert memory (future)
};
```

- Kernels and the scheduler only ever see resident bytes (a wired region or a
  filled slot). Asking a non-resident source for bytes is an error, not an
  implicit load.
- `thunderbolt_peer` is reserved: it reports that its transport is unavailable
  until the scale-out work starts.
- Payload fields grow with M2 and M4; the variant set is fixed.

## Residency and caching

- **Hot set**: chosen offline from expert heat statistics (tools) and refined
  online; pinned in the wired arena.
- **Cold set**: stays in `.amf` files on NVMe and is read into a ring of
  preallocated 16 KiB-aligned slots, each wrapped once as an `MTLBuffer`.
- **Eviction** (Draft): least-frequently-used with recency decay; experts
  referenced by an in-flight verification pass are never evicted.
- **Sizing**: the wired budget
  ([ADR-0012](../decisions/0012-wired-memory-budget-headroom.md)) bounds the
  hot set plus the slot ring.

## AMF format

aminfer Model Format, accepted in
[ADR-0006](../decisions/0006-amf-cold-experts-direct-io.md). Standard
safetensors and GGUF headers and tensor layouts are not 16 KiB aligned, so
they cannot be read straight into wired slots. Draft v0 layout:

- Little-endian. Every section offset and length is a multiple of 16 KiB.
- **Page 0** — header: magic, format version, model identifier, quantization
  scheme, tensor count, directory offset and length, checksum algorithm.
- **Directory** — fixed-size records: tensor kind, layer, expert index,
  quantization bits, group size, shape, blob offset, blob length, checksum.
- **Blobs** — each starts on a 16 KiB boundary and is zero-padded to a
  16 KiB multiple. An expert blob holds its gate, up and down projections
  (packed weights, scales, biases) contiguously, in exactly the layout the
  Metal kernels read, so a slot is usable as soon as the read completes.
- Written only by `tools/`; the Zig reader validates the header, directory
  bounds and alignment at startup, never on the hot path.
- Conversion source (Draft, status row W4): MLX community quantized
  safetensors (affine, group size 64), which already exist for the target
  models.
- Open: one container for all weights (dense, hot and cold, residency decided
  at load time) versus cold-expert packs only; checksum algorithm (CRC-32 IEEE,
  available in both Zig `std` and Python's `zlib`, versus BLAKE2b); no
  compression in v0.

## Cold-expert I/O path

Accepted in [ADR-0006](../decisions/0006-amf-cold-experts-direct-io.md):

1. Open the `.amf` file read-only and set `fcntl(fd, F_NOCACHE, 1)`, so reads
   bypass the unified buffer cache and do not squeeze the wired pool.
2. For each predicted cold expert, reserve a slot and submit `aio_read` for its
   blob with `SIGEV_KEVENT` targeting the I/O thread's kqueue.
3. On `EVFILT_AIO`, read the result from `kevent64`: `ext[0]` is the error,
   `ext[1]` the byte count (measured; a later `aio_return` fails with `EINVAL`).
4. Publish the filled slot to the scheduler through a lock-free ring; the GPU
   uses the slot's existing `MTLBuffer`.

Limits and fallback: kernel AIO is served by `kern.aiothreads` = 4 workers on
the baseline. If that caps throughput, the fallback is a few E-core QoS
threads issuing blocking `pread` on the same `F_NOCACHE` descriptor (status row
C4). Spike S3 decides, measuring queue depths 1–16, p50 / p99 latency, CPU
cost and the effect of QoS and I/O policy
([../process.md](../process.md#spikes)).

## Expert prediction and prefetch

- **ANE expert predictor** (project charter): predicts per-layer expert sets
  for upcoming tokens and draft-tree nodes ahead of verification — early enough
  to hide NVMe latency.
- **Consumers**: the prefetch queue, the pruning cost model
  ([ADR-0009](../decisions/0009-expert-byte-budget-tree-pruning.md)) and the
  gather plan of the verification kernels.
- **Undecided**: the predictor's architecture — a small learned model over
  hidden states or embeddings, or the target's own gates applied ahead of time.
  A GPU-side cross-layer gate lookahead serves as the baseline it must beat.
- **Metrics**: recall of predicted experts, prefetch lead time versus NVMe
  latency, bytes read but unused.

## N-gram tables on NVMe

Draft (status row C6): N-gram / suffix tables larger than the context — for
example built from a code repository — stored on NVMe and queried by L1 with
direct I/O. Format, build tooling and lookup path are designed in M5.

## Thunderbolt peer

Future; only the `thunderbolt_peer` variant is reserved.

- Thunderbolt 4 machines, including the baseline, can only use IP over
  Thunderbolt Bridge. Thunderbolt 5 Macs on macOS 26.2+ can use RDMA over
  Thunderbolt (Apple TN3205), the transport behind MLX's JACCL backend.
- Intended semantics: experts are partitioned across machines; hidden states
  travel to the owning machine and results come back. The scheduler treats a
  remote expert like a cold expert with its own latency and bandwidth model.
