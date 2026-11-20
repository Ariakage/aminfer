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
they cannot be read straight into wired slots. v0 layout (`src/format/amf.zig`):

- Little-endian, unified container (dense weights, hot experts, cold experts,
  embeddings, norms, routers, LM head; residency decided by `TensorKind` at
  load time). Every section offset and length is a multiple of 16 KiB; no
  compression in v0.
- **Page 0** — 128-byte `Header` zero-padded to 16 KiB: magic
  `"AMF0\x00\x00\x00\x00"`, version `1`, page size `16384`, checksum algorithm
  `1` (CRC-32 IEEE / ISO-HDLC), tensor count, directory offset (`16384`),
  directory length (16 KiB multiple), directory CRC-32, header CRC-32,
  64-byte `model_id`.
- **Directory** — `tensor_count` fixed 128-byte `TensorEntry` records,
  zero-padded to a 16 KiB multiple: 48-byte name, `TensorKind`, `DType`
  (implemented in M1: `f16`, `f32`, `q4_affine`, `q3_affine`; confirmed for M2.5:
  `bf16 = 4` per [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)),
  bit-width, rank, group size (`64` for affine quantized tensors, `0` for
  unquantized tensors), layer and expert indices, blob CRC-32, 4-element shape,
  16 KiB-aligned `blob_offset` and `blob_bytes`, `data_bytes`, 64 B-aligned
  `params_offset`, and `params_bytes`.
- **Blobs** — each starts on a 16 KiB boundary and is zero-padded to a
  16 KiB multiple. For quantized 2D matrices (`[N, K]`, `K % 64 == 0`), bytes
  `[0..data_bytes)` hold MLX-compatible LSB-first packed weights (4-bit: 2
  values/byte, 32 B/group; 3-bit: 8 values/3 bytes, 24 B/group), followed at
  `params_offset` (64 B aligned) by interleaved `half2(scale, bias)` per group
  (`w = q * scale + bias`). Unquantized `f16`, `bf16` (M2.5), and `f32` tensors
  store contiguous little-endian elements in `[0..data_bytes)` with
  `params_bytes = 0`. GPU kernels read these blob layouts directly without
  repacking.
- Written offline by `tools/`; the Zig reader (`AmfFile.open`, `F_NOCACHE`)
  validates header, directory CRC, alignment, bounds, non-overlapping blobs
  and blob CRCs at startup, never on the hot path.
- Conversion sources:
  - Implemented in M1 (status row W4): single-file MLX community quantized
    safetensors (affine, group size 64; `bf16` scales/biases converted to `f16`
    on export).
  - Confirmed for M2.5 (status row W12, [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)):
    streaming multi-shard Hugging Face / MLX checkpoints
    (`model.safetensors.index.json` across shard files with bounded host RAM)
    plus native bit-exact `BF16` / `FP16` unquantized and mixed-precision tensor
    export.

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
