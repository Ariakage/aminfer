# ADR-0006: `.amf` cold-expert format and `F_NOCACHE` direct I/O

- Status: Accepted
- Date: 2026-11-20
- Source: user requirement (init session, "new requirements" 3)

## Context

Standard safetensors and GGUF files do not align headers and tensor data to
16 KiB pages, so they cannot be read straight into wired, GPU-visible slots.
A plain `pread` also fills the macOS unified buffer cache, and that kernel file
cache competes with — and squeezes — the wired memory pool.

## Decision

1. Offline tools in `tools/` export a dedicated, 16 KiB page-aligned **`.amf`
   (aminfer Model Format)** cold-expert pack: aligned header, directory and
   blobs, with each expert blob in the exact layout the Metal kernels read
   ([../architecture/experts-and-storage.md](../architecture/experts-and-storage.md#amf-format)).
2. At runtime, cold-expert file descriptors are opened with
   `fcntl(fd, F_NOCACHE, 1)` to bypass the page cache.
3. Reads are issued with `aio_read` and completed through `kqueue`
   (`SIGEV_KEVENT` → `EVFILT_AIO`), landing by DMA directly in preallocated,
   16 KiB-aligned slots of a lock-free ring.
4. Prefetch threads run at an E-core QoS class.

## Consequences

- A conversion step (`tools/` → `.amf`) is required before serving a model.
- The Zig reader validates layout and alignment once at startup.
- Completion results come from `kevent64` (`ext[0]` error, `ext[1]` bytes),
  not from `aio_return`.

## Evidence

Measured on the baseline:

- `fcntl(fd, F_NOCACHE, 1)` succeeds.
- `aio_read` with `SIGEV_KEVENT` delivers `EVFILT_AIO` on macOS 27.2; the
  1 MiB test read matched; `aio_return` afterwards returns `EINVAL`.
- Kernel AIO concurrency: `kern.aiothreads` = 4, `kern.aioprocmax` = 256.

## Open items

- Spike S3: whether 4 kernel AIO workers cap NVMe throughput; the fallback is
  E-core `pread` workers on the same `F_NOCACHE` descriptor (status row C4).
- Whether `IOPOL_IMPORTANT` on the submitting thread reaches requests served
  by kernel AIO workers.
- `.amf` details: single container for all weights versus cold packs only;
  checksum algorithm.
