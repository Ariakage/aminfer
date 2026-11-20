# Process

The single record of project progress: milestones, spikes, the current task
list, the verification log and the session handoff snapshot. Capability status
lives in [status.md](status.md); decisions live in
[decisions/](decisions/README.md).

## Current focus

M3 — speculative decoding completed (2026-11-21); next: **M4 — Neural Engine and storage** (hybrid wavefront prefill `A1`/`A3`/`G6`/`S6`, ANE cross-layer MoE expert prediction `A2`, CoreML `.mlpackage` export `W7`, offline expert heat statistics `W8`, cold-expert direct I/O & cache policy `C3`–`C5`/`S3`, same-family small draft model `D4`).

## Milestones

Status-row IDs refer to [status.md](status.md).

| Milestone | Goal | Status rows | Exit criteria | State |
| --- | --- | --- | --- | --- |
| M0 | Project initialization | F1–F9 | `zig build` and `zig build test` pass in Debug and ReleaseSafe; docs baseline committed; local ikat index built | Completed |
| M1 | GPU and weight foundation | R1, R2, F6 (hot path), W1–W4, G1, G2, C9 | Wired budget enforced at startup; `.amf` round-trip (tools → Zig) with alignment validation; fused dequant GEMV / GEMM match a reference within tolerance; effective bandwidth measured per kernel | Completed |
| M2 | Non-speculative decode | W5, W6, C1, C2, R3, R4, G3, G4 | Greedy end-to-end decode of the primary target from `.amf`; baseline tokens/s and bytes per token recorded | Completed |
| M2.5 | Standard FP16 / BF16 and multi-shard import | W12, W13, G12 | Multi-shard `aminfer-tools pack-amf` converts local `Qwen3-30B-A3B-3bit`, `Qwen3-14B-4bit`, and `Qwen3-0.6B` (`BF16`) with bounded RAM; native `FP16` and `BF16` GEMV / GEMM / MoE SwiGLU match CPU fp32 reference; real-checkpoint greedy decode verified end-to-end (incl. `Qwen3-0.6B` `BF16`) with bandwidth archived in [benchmarks/results.md](benchmarks/results.md) | Completed |
| M3 | Speculative decoding | D1–D3, D5–D7, D9, G5 | Lossless output versus M2/M2.5; speedup over the non-speculative baseline, acceptance-length and experts-per-pass statistics recorded | Completed |
| M4 | Neural Engine and storage | A1–A3, G6, W7, W8, C3–C5, D4 | Hybrid prefill throughput versus GPU-only prefill; expert-prediction recall and prefetch lead time; cold-expert decode without swap | Not started |
| M5 | Serving and pipelining | C6–C8, D8, W14, G13 | Streaming server under load; pipelined-drafting gain measured; optional GSQ+RCO per-block `QuantScheme` pipeline verified | Not started |
| M6 | Frontier hybrid sparse models (`Qwen3.8-Flash-Next`) | R6, W9, W10, G7–G9, G10 (GR), C10 (PLE) | Composable model operator graph; `Qwen3.8-Flash-Next` (`qwen4_exp`: Gated DeltaNet + QSA + 512-expert MoE + 51B NVMe PLE + 4B MTP) verified end-to-end (128 GB+ full wired residency; 24 GB M4 hot/cold + PLE async prefetch without swap) | Not started |
| M7 | Ultra & cluster scale-out (`DeepSeek-V4.1-Flash-1017`) | R7, W11, G10 (mHC), G11, D10, C10 (Engram), X1–X3 | Causal Encoder-Decoder scheduler, MLA/CSA2 + FP4 KV, 196B Engram lookup, DSpark drafting and Thunderbolt 5 RDMA expert parallelism verified on 512 GB Ultra or multi-Mac TB5 clusters | Future |

## Spikes

Short measurements that must run before the related design moves from Draft to
Confirmed, or before an accepted design is tuned.

| ID | Question | Feeds | Milestone | State |
| --- | --- | --- | --- | --- |
| S1 | KV tile read path: R16Float vs RGBA16Float texture vs aliasing `MTLBuffer`; token- vs channel-major rows; decode-attention bandwidth | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), R4 | M2 | Closed |
| S2 | Metal 4 steady-state allocations and encode overhead per decode round (Instruments Allocations, signposts) | [ADR-0011](decisions/0011-metal4-zero-allocation-hot-path.md) | M1 | Closed |
| S3 | Cold-expert reads: `aio_read` + `kqueue` vs E-core `pread` workers; queue depth 1–16; 2–3 MiB reads with `F_NOCACHE`; QoS and I/O-policy effects; p50 / p99 latency | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), C4 | M4 | Open |
| S4 | ANE handoff latency (GPU → CPU → ANE → CPU → GPU) and bandwidth contention of concurrent ANE weight streaming on GPU GEMV | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md) | M3 | Closed |
| S5 | Unique experts per layer versus draft-tree size for Qwen3-30B-A3B; effective bandwidth of deduplicated vs per-slot expert kernels | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md) | M3 | Closed |
| S6 | ANE attention block: CoreML compilation, static shapes for a growing KV length, fp16 numerics, throughput vs GPU | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), A3 | M4 | Open |

## M0 task list

- [x] Environment probe and design review with the user; decisions recorded
      as ADR-0001 to ADR-0014
- [x] Install the Metal Toolchain component (27A266a)
- [x] `AGENTS.md` (commit `1ad0da8`)
- [x] Dual MIT / Apache-2.0 license (commit `6d5053d`)
- [x] Documentation baseline: README, architecture, decisions, guides, status,
      process
- [x] `build.zig` graph: Zig core + ObjC bridge (translate-c) + embedded
      `.metallib` + tests, with `aminfer probe` for Metal and ANE facts (F1, F3)
- [x] 16 KiB wired regions and sysctl probe, test first (F2, F9)
- [x] `ExpertSource` tagged union, test first (F4)
- [x] Bridge: NoCopy buffer and Metal 4 smoke dispatch, test first (F5, F6)
- [x] `tools/` uv scaffold (F7)
- [x] `justfile` shortcuts (F8)
- [x] Verify Debug / ReleaseSafe / ReleaseFast; build the local ikat index

## M1 task list

- [x] Startup wired-memory budget enforcement and `WiredArena` slicer (R1)
- [x] Compile-time quantization spec (`G2` host) and `.amf` v0 reader/validator (W1, W3)
- [x] Offline `.amf` v0 packer and MLX safetensors importer in `tools/` + round-trip validation (W2, W4)
- [x] Metal 4 multi-frame hot path, command batching, residency set and specialized pipelines (F6, R2, G2 GPU)
- [x] Fused 3-bit and 4-bit dequantization GEMV / GEMM Metal kernels verified against CPU reference (G1, G2)
- [x] Benchmark harness with swap checks (`C9`), per-kernel bandwidth calibration and Spike S2 measurement

## M1 calibration and Spike S2 results

Canonical methodology, hardware/software environment, full summary tables,
`sysctl` verification, `powermetrics` telemetry and verbatim raw run outputs
live in [benchmarks/results.md](benchmarks/results.md).

Measured on 2026-11-20 on the Apple M4 24 GiB baseline (`aminfer bench`,
ReleaseFast at commit `16322279`, single 81 MiB `WiredArena` validated against
the 18,186 MiB host budget, ≥ 64 MiB rotating weight pool streaming from
unified DRAM, `swapouts_delta = 0`, `pageouts_delta = 0`):

| Kernel | Shape (`M×N×K`) | Bytes/dispatch | Latency (us) | Effective `beta` (GB/s) | % of 120 GB/s bus |
| --- | --- | ---: | ---: | ---: | ---: |
| `gemv_q4` | `1×2048×2048` | 2,367,488 | 29.38 | 80.58 | 67.2% |
| `gemv_q4` | `1×6144×2048` | 7,094,272 | 90.86 | 78.08 | 65.1% |
| `gemv_q4` | `1×2048×768` | 890,368 | 14.31 | 62.21 | 51.8% |
| `gemv_q4` | `1×4096×4096` | 9,453,568 | 95.74 | 98.74 | 82.3% |
| `gemv_q3` | `1×2048×2048` | 1,843,200 | 24.41 | 75.51 | 62.9% |
| `gemv_q3` | `1×6144×2048` | 5,521,408 | 63.22 | 87.34 | 72.8% |
| `gemv_q3` | `1×2048×768` | 693,760 | 11.04 | 62.83 | 52.4% |
| `gemv_q3` | `1×4096×4096` | 7,356,416 | 86.89 | 84.67 | 70.6% |
| `gemm_q4` | `16×2048×2048` | 2,490,368 | 105.82 | 23.54 | 19.6% |
| `gemm_q4` | `16×4096×4096` | 9,699,328 | 401.20 | 24.18 | 20.1% |
| `gemm_q4` | `64×2048×2048` | 2,883,584 | 210.40 | 13.71 | 11.4% |
| `gemm_q4` | `64×4096×4096` | 10,485,760 | 812.83 | 12.90 | 10.8% |
| `gemm_q3` | `16×2048×2048` | 1,966,080 | 103.55 | 18.99 | 15.8% |
| `gemm_q3` | `16×4096×4096` | 7,602,176 | 403.27 | 18.85 | 15.7% |
| `gemm_q3` | `64×2048×2048` | 2,359,296 | 209.73 | 11.25 | 9.4% |
| `gemm_q3` | `64×4096×4096` | 8,388,608 | 828.26 | 10.13 | 8.4% |

Spike S2 (1,000 steady-state rounds × 32 dispatches/round on a 2-frame Metal 4
`Compute` ring with `insert_barrier_before` and uniform streaming, after
2,048-round driver-pool warmup):

| Metric | Value |
| --- | --- |
| CPU encode time | 7.78 us/round (243.3 ns/dispatch) |
| Full pipelined round time | 98.23 us/round |
| `malloc_zone_statistics` `blocks_in_use` delta | 0 |
| `malloc_zone_statistics` `size_in_use` delta | 0 B |
| `malloc_zone_statistics` `size_allocated` delta | 0 B |
| System `swapouts` / `pageouts` delta | 0 / 0 |

## M2 task list

- [x] Deterministic synthetic Qwen3-MoE (3-bit) & Qwen3-Dense (4-bit) `.amf` / `config.json` / `tokenizer.json` generator in `tools/` (`synth-qwen3`)
- [x] Preallocated lock-free SPSC / MPSC rings (`C1`), macOS QoS / disk I/O steering (`C2`), and native Zig Qwen byte-level BPE tokenizer (`W5`)
- [x] IOSurface `OneComponent16Half` KV tiles with zero-copy `RGBA16Float` / `R16Float` views (`R3`), Spike S1 microbenchmark (`S1`, `R4`), RMSNorm, QK-Norm/NeoX RoPE/KV-Write and FlashDecoding split-KV GQA decode attention (`G4`)
- [x] MoE router, deduplicated active-expert fused 3/4-bit SwiGLU FFN (`G3`), dense SwiGLU utilities and single-dispatch GPU greedy argmax
- [x] End-to-end Qwen3-MoE & Qwen3-Dense greedy decode engine (`W6`) with multi-shard `WiredArena` support, zero hot-path heap allocation, and step-for-step CPU fp32 reference verification
- [x] CLI `aminfer decode` + M2 ReleaseFast benchmark suite (`aminfer bench` / `aminfer bench-m2`) archived in [benchmarks/results.md](benchmarks/results.md)

## M2 non-speculative decode and Spike S1 results

Canonical methodology, full tables, and verbatim raw run outputs live in
[benchmarks/results.md](benchmarks/results.md). Key M2 baseline measurements on
Apple M4 24 GiB (`aminfer bench`, ReleaseFast at commit `517ba0da`, default
18,186 MiB wired budget, `swapouts_delta = 0`, hot-path `malloc` delta `0`):

- **Spike S1 (`R4` settled)**: token-major `RGBA16Float` `80.23 GB/s` and
  `R16Float` `91.75 GB/s` vs aliasing `MTLBuffer` `82.14 GB/s` (zero
  texture-view penalty; exact 16 KiB-aligned `kIOSurfaceAllocSize = 524,288 B`;
  `mlock` supported). Settled on **512-token chunks, token-major `RGBA16Float`**.
- **`G4` GQA decode attention**: `24.55 us` (`42.72 GB/s`) at `seq_len=512`,
  `106.98 us` (`78.42 GB/s`) at `seq_len=4096`, and **`391.96 us` (`85.61 GB/s`,
  71.3% bus)** at `seq_len=16384`.
- **`G3` MoE deduplicated FFN (3-bit, `E=128, k=8`)**: `M=1` `228.34 us`
  (`72.33 GB/s`); `M=8` saves `62.5%` weight bytes at `U=24/64` (`49.5 MB` vs
  `132.1 MB` naive) and `37.5%` at `U=40/64`.
- **End-to-end greedy decode from `.amf` (`seq_len = 512`, `vocab = 151,936`)**:
  - **Qwen3-30B-A3B (3-bit MoE)**: measured 4-layer `.amf` `3.508 ms/tok`
    (`285.05 tok/s`, `241.6 MB/tok`, `68.86 GB/s`, `0.477 ms/L + 1.598 ms head`);
    linear 48-layer scaling **`24.515 ms/tok` (`40.79 tok/s`, `1,401,238,400 B/tok`,
    `57.16 GB/s`)**.
  - **Qwen3-14B (4-bit Dense)**: measured 4-layer `.amf` `14.926 ms/tok`
    (`67.00 tok/s`, `1,189.3 MB/tok`, `79.68 GB/s`, `2.436 ms/L + 5.182 ms head`);
    linear 40-layer scaling **`102.622 ms/tok` (`9.74 tok/s`, `7,954,260,800 B/tok`,
    `77.51 GB/s`)**.

## M2.5 task list

- [x] `.amf` `DType.bf16 = 4` in `src/format/amf.zig` and `tools/src/aminfer_tools/amf.py` (`--dtype auto|f16|bf16`), keeping 16 KiB alignment and CRC-32 validation intact (`W12`)
- [x] Streaming multi-shard `model.safetensors.index.json` importer in `aminfer-tools pack-amf` (header-first directory layout + bounded-RAM shard streaming across `Qwen3-30B-A3B-3bit`, `Qwen3-14B-4bit`, and `Qwen3-0.6B` `BF16`, including MLX 3D `switch_mlp` expert slicing and quantized `embed_tokens` / `mlp.gate` handling) (`W12`)
- [x] Specialized Metal 4 `FP16` (`half4` / `simdgroup_half8x8`) and `BF16` (`bfloat4` / `simdgroup_bfloat8x8`) GEMV, GEMM, MoE SwiGLU, embedding, RMSNorm, and `lm_head` kernels with FP32 accumulation (`G12`)
- [x] `Qwen3Model` unquantized / mixed-precision weight path (`quantization_config` optional, per-tensor `amf.DType` dispatch, zero-duplicate `tie_word_embeddings`, `RGBA16Float` `KvTile` boundary) verified against CPU fp32 reference (`W13`)
- [x] End-to-end real-checkpoint conversion & greedy decode verification (`Qwen/Qwen3-0.6B` `BF16` + multi-shard MLX checkpoints in `/Volumes/Akari/models/`) and `FP16` / `BF16` kernel bandwidth benchmarks archived in [benchmarks/results.md](benchmarks/results.md)

## M2.5 FP16 / BF16 and real-checkpoint results

Canonical methodology, conversion statistics, full 16-shape `FP16`/`BF16`
calibration tables, real-checkpoint decode metrics and verbatim raw run outputs
live in [benchmarks/results.md](benchmarks/results.md). Key M2.5 highlights on
Apple M4 24 GiB (commits `2a48c66..1893ab6`, ReleaseFast, default 18,186 MiB
wired budget, `swapouts_delta = 0` across all runs):

- **Bounded-RAM multi-shard `.amf` conversion (`W12`)**:
  - `Qwen3-0.6B` (BF16, 1 shard): 310 tensors (`lm_head` deduped), `1.19 GB`, packed in `4.7 s` (`35 MB` peak RSS), verified in `4.6 s` (`21 MB` RSS).
  - `Qwen3-14B-4bit` (2 shards): 443 tensors (`q4_affine` + `bf16` norms), `8.31 GB`, packed in `50.4 s` (`116 MB` peak RSS), verified in `28.3 s` (`21 MB` RSS).
  - `Qwen3-30B-A3B-3bit` (3 shards): 18,867 tensors (`q3_affine` experts/routers/embed/head + `bf16` norms), `13.36 GB`, packed in `111.6 s` (`127 MB` peak RSS), verified in `50.9 s` (`25 MB` RSS).
- **Native `FP16` / `BF16` Metal 4 kernels (`G12`, `aminfer bench-m25`)**:
  - Decode GEMV (`1×4096×4096`): **`88.25 GB/s` (73.5% bus)** for `gemv_bf16` and **`86.76 GB/s` (72.3% bus)** for `gemv_f16`.
  - Batched `simdgroup_8x8` GEMM (`M = 64`): **`2.690 TFLOP/s`** (`gemm_f16`) and **`2.509 TFLOP/s`** (`gemm_bf16`).
- **Real full-layer greedy decode (`W13`, `seq_len = 512` `bench-decode` & bilingual EN/ZH `decode`)**:
  - **`Qwen3-0.6B` (BF16, 28L, tied, 1,251 MiB wired)**: **`47.88 tok/s`** (`20.884 ms/tok`, `59.90 GB/s`, `0` malloc delta) at `seq=512`; **`62.98–66.17 tok/s`** (`75.5–79.3 GB/s`) on 32-token EN/ZH decode.
  - **`Qwen3-14B-4bit` (40L, 8,086 MiB wired)**: **`9.52 tok/s`** (`104.997 ms/tok`, `75.76 GB/s`, `-2.3%` vs M2's 4L linear projection `9.74 tok/s`; `10.56–10.67 tok/s` / `83.1–84.0 GB/s` on EN/ZH decode).
  - **`Qwen3-30B-A3B-3bit` (48L, 128 experts, 12,839 MiB wired)**: **`51.86 tok/s`** (`19.282 ms/tok`, `1,381,577,600 B/tok`, `71.65 GB/s`, **+27.1%** faster than M2's 4L projection `40.79 tok/s` thanks to quantized `q3` routers, 48L command-buffer amortization, and real expert locality).

## M3 task list

- [x] Preallocated zero-allocation L1 suffix automaton (`D1`), compile-time dual-source prefix trie (`D6`), submodular expert-byte-budget pruner (`D7`), and lossless greedy (`T = 0`) / recursive rejection-sampling (`T > 0`) tree verification (`D9`) in `src/spec/{sam,tree,accept}.zig`
- [x] Metal 4 tree-attention verification with 64-bit ancestor masks, batched `M`-node QK-Norm/RoPE and accepted-path IOSurface KV commit (`G5`), batched top-k logit selection, weight-stationary bit-exact batched GEMV/MoE kernels, CoreML ANE zero-copy IOSurface bridge (`D5`), and Spikes `S4` & `S5` microbenchmarks
- [x] `.amf` `DType.u32 = 5` and offline EAGLE-3 / EAGLE-2 / MTP draft head packers & synthetic draft generators in `tools/` (`aminfer-tools pack-amf` / `synth-draft`), plus real `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` conversion (`qwen3-30b-a3b-eagle3.amf`, 996 MiB)
- [x] End-to-end `DraftHead` (`eagle3`, `eagle2`, `mtp`, `.auto`/`.gpu`/`.ane` `D5` startup placement gate) and `SpecEngine` (`D1`–`D3`, `D5`–`D7`, `D9`, `G5`) integrated with `Qwen3Model.verifyTree` / `commitTreeKv` / `prefillChunk` (`src/spec/{draft,engine}.zig`, `src/model/qwen3.zig`)
- [x] CLI `aminfer spec-decode` and `aminfer bench-m3` (wired into `aminfer bench`), verified bit-for-bit lossless on synthetic MoE/Dense + real `Qwen3-0.6B` BF16, `Qwen3-14B-4bit`, and `Qwen3-30B-A3B-3bit` (+ real `EAGLE-3`) checkpoints and archived in [benchmarks/results.md](benchmarks/results.md)

## M3 speculative decoding and Spikes S4 & S5 results

Canonical methodology, full tables, and verbatim raw run outputs live in
[benchmarks/results.md](benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5).
Key M3 highlights on Apple M4 24 GiB (commits `cb6ad86..034492f`, ReleaseFast,
default 18,186 MiB wired budget, `swapouts_delta = 0` across all runs):

- **Spike S4 (`D5` settled on Apple M4)**: Zero-copy IOSurface
  `GPU → CPU → ANE → CPU → GPU` round-trip handoff measures **`574.64 us`**
  (`uses_ane = yes`, `zero_copy_backing = yes`). On a `2L × 2048×2048 fp16`
  draft workload, ANE + handoff takes `941.57 us` (`M = 1`) and `937.84 us`
  (`batch = 32`) vs **`201.87 us` / `277.36 us`** in-CB GPU (`3.38–4.66×`
  faster) → `D5` gate selects `.gpu`. Concurrent 40 MiB ANE streaming
  (`50.00 GB/s`) cuts GPU `gemv_q3` by `-30.8%` and `gemv_q4` by `-32.5%`.
- **Spike S5 (`D7` calibrated)**: On `Qwen3-30B-A3B` (`q3, E=128, top_k=8`),
  CSR-deduplicated MoE crosses over per-slot at `M = 2` under correlated
  routing (**`1.36×`** at `M = 4`, **`1.78×`** / `36.78 GB/s` at `M = 8`,
  **`1.87×`** at `M = 16`, saving `65.0–91.0%` expert weight reads) and at
  `M = 16` under uniform routing (`1.08×` at `M = 16`, `1.35×` at `M = 32`).
- **L1 suffix automaton (`D1`, 32,768 tokens)**: `5.14 MiB` arena,
  **`0.041 us` (`41 ns`)** active-state query (`1,220×` below the `< 50 us`
  charter target), `0` hot-path heap allocations.
- **Real-checkpoint lossless speculative decode (`T = 0`, 100% bit-for-bit
  identical `token_ids` verified against `aminfer decode`)**:
  - **`Qwen3-0.6B` BF16 (28L)**: `L1` lifts decode `69.20 →` **`327.49 tok/s`
    (`4.73×`)**, `tau = 5.600`, `-82.2%` DRAM bytes/accepted token, prefill
    `11.0×` faster; stochastic `T = 0.7` (`D9`) verified coherent at `69.28 tok/s`.
  - **`Qwen3-14B-4bit` (40L)**: `L1` lifts decode `10.23 →` **`17.45 tok/s`
    (`1.71×`)**, `tau = 4.357`, `-77.1%` DRAM bytes/accepted token, prefill
    `3.13×` faster.
  - **`Qwen3-30B-A3B-3bit` (48L MoE) + real `RedHatAI` `EAGLE-3`**:
    - *C++20 reasoning (`<think>`)*: unpruned `cap = 16` EAGLE-3 slows decode
      `46.00 → 24.93 tok/s` (`0.54×`, `31.28` unique experts/layer), whereas
      **`EAGLE-3 + L1 + D7`** cuts verified nodes `13.44 → 2.93`, halves active
      experts to `15.83/layer` (`-49.4%`) in `0.390 ms`, cuts DRAM traffic
      `-32.4%`, and achieves **`48.38 tok/s` (`1.94×` vs unpruned, `1.05×` vs
      baseline)** — validating [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md).
    - *TypeScript code edit (`/no_think`)*: **`L1 + D7`** lifts decode
      `50.15 →` **`59.24 tok/s` (`1.18×`)** (`15.82` experts/layer, `-37.0%`
      bytes); `EAGLE-3 + L1 D6 seed + D7` lifts `tau` further (`2.500 → 3.412`,
      `14/17` L1 seed rounds, `-45.3%` bytes/accepted token) while exposing the
      `996 MiB` unquantized BF16 draft-head sync cost (`8.31 ms/round`,
      `33.82 tok/s`) as an M4/M5 quantization target.

## Verification log

Only checks that actually ran are listed.

| Date | Check | Result |
| --- | --- | --- |
| 2026-11-20 | Throwaway probe (not in repo): Zig 0.17.1 `addTranslateC` + ObjC `-fobjc-arc -Werror` + Metal link + `zig build test` | Pass |
| 2026-11-20 | `@cImport` under Zig 0.17.1 | `error: invalid builtin function: '@cImport'` |
| 2026-11-20 | `aio_read` with `SIGEV_KEVENT` on macOS 27.2 | Pass: `EVFILT_AIO` delivered, bytes in `kevent64.ext[1]`, data matches; `aio_return` afterwards fails with `EINVAL` |
| 2026-11-20 | IOSurface fp16 tile → R16Float and RGBA16Float textures, with `MTL_DEBUG_LAYER=1` | Pass |
| 2026-11-20 | `xcrun metal -std=metal4.0 -mmacosx-version-min=26.0` | Pass (after installing the Metal Toolchain) |
| 2026-11-20 | F1: `zig build`; `zig build test` (Debug, ReleaseSafe): 3/3 tests; `aminfer probe` matches the throwaway probe; binary `minos 26.0` | Pass |
| 2026-11-20 | F2, F9: `zig build test` (Debug, ReleaseSafe): 10/10 tests; `aminfer probe` sysctl output matches [hardware.md](architecture/hardware.md) | Pass |
| 2026-11-20 | F4: `zig build test` (Debug, ReleaseSafe): 16/16 tests | Pass |
| 2026-11-20 | F5, F6: `zig build`; `zig build test` (Debug, ReleaseSafe): 20/20 tests; direct test binary with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: 20/20 tests | Pass |
| 2026-11-20 | F7: `cd tools && uv sync`; `uv run python -m unittest discover -s tests`: 4/4 tests; `uv run aminfer-tools --help` | Pass |
| 2026-11-20 | F8: `just --list`, `just fmt-check`, `just sysctl-show`, `just swap-check`, `just tools-test`, `just probe` (sudo recipes not run) | Pass |
| 2026-11-20 | M0 final gate: `zig fmt --check`, zero `catch {}`, `zig build` + `zig build test` (Debug, ReleaseSafe, ReleaseFast: 20/20 tests), `minos 26.0`, `tools` unittest (4/4), 28 markdown files link-checked (0 broken), `ikat init` | Pass |
| 2026-11-20 | R1: `zig build`, `aminfer probe` (`budget.wired_limit_mib 18186`, binding `MTLDevice.recommendedMaxWorkingSetSize`), `zig build test` (Debug, ReleaseSafe): 27/27 tests | Pass |
| 2026-11-20 | W1, W3, G2 (host): `zig build`, `zig build test` (Debug, ReleaseSafe): 36/36 tests (incl. `F_NOCACHE` disk `.amf` load, CRC-32 & overlap rejection) | Pass |
| 2026-11-20 | W2, W4: `cd tools && uv run python -m unittest discover -s tests`: 7/7 tests (incl. MLX safetensors → `pack-amf` → `zig-out/bin/aminfer verify-amf` cross-language round-trip) | Pass |
| 2026-11-20 | F6 (hot path), R2, G2 (GPU): `zig build`, `zig build test` (Debug, ReleaseSafe): 37/37 tests; `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: 37/37 tests | Pass |
| 2026-11-20 | G1: `zig build`, `zig build test` (Debug, ReleaseSafe): 42/42 tests (3/4-bit GEMV & GEMM vs CPU fp32 reference + `.amf` → `WiredArena` → GPU round-trip); `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: 42/42 tests | Pass |
| 2026-11-20 | C9, Spike S2, M1 final gate: `zig fmt --check`, zero `catch {}`, `zig build`, `zig build test` (Debug, ReleaseSafe): 50/50 tests; direct test binary with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: 50/50 tests; `just bench` (`ReleaseFast` `aminfer bench`): 16 C9 shapes + 1,000-round S2 (0 steady-state malloc delta, 0 swapouts); `tools` unittest (7/7) | Pass |
| 2026-11-20 | Privileged `sysctl` & `powermetrics` telemetry: `iogpu.wired_limit_mb=21504`, `vm.global_user_wire_limit=22548578304`, `vm.user_wire_limit=22548578304` (all rc=0) raised non-root `aminfer probe` `recommendedMaxWorkingSetSize` and `budget.wired_limit_mib` from `18186` to `21504 MiB` (`physical_headroom`); reverse-order restore verified back to `0` / `19585050869` / `18186 MiB`; `powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 6` during `aminfer bench` captured peak GPU `1470 MHz` (`1459 MHz` active avg, `70.98%` active, `1513 mW`), ANE `0 mW`, `0` swapouts; archived in [benchmarks/results.md](benchmarks/results.md) | Pass |
| 2026-11-20 | M2 `tools/synth.py` + `synth-qwen3`: `cd tools && uv run python -m unittest discover -s tests -v`: 14/14 tests (incl. cross-language `aminfer verify-amf` on tiny + real 1-layer Qwen3-30B-A3B-3bit & Qwen3-14B-4bit `.amf` bundles) | Pass |
| 2026-11-20 | M2 final gate (`W5`, `W6`, `C1`, `C2`, `R3`, `R4`, `G3`, `G4`, `S1`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build`, `zig build test` (Debug, ReleaseSafe, ReleaseFast): 96/96 tests (95 lib + 1 cli); direct Debug test binary under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (API + GPU Validation enabled): 95/95 tests; ReleaseFast `./zig-out/bin/aminfer bench` (C9 + S2 + S1 + G3 + G4 + 4-layer `.amf` Qwen3-30B-A3B-3bit & Qwen3-14B-4bit decode, `0` hot-path malloc delta, `0` swapouts) + end-to-end `synth-qwen3` → `aminfer decode` (`tiny-moe`, `tiny-dense`, 4-layer `qwen3-30b-a3b-3bit`, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md) | Pass |
| 2026-11-20 | Throwaway probe (not in repo): `xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -Wall -Wextra -Werror` with `bfloat4`, `simdgroup_bfloat8x8`, and `simdgroup_half8x8` accumulating into `simdgroup_float8x8` | Pass |
| 2026-11-20 | M2.5 `W12` (`2a48c66`): `DType.bf16`, `O(1)` `Parsed.index`, streaming `verifyAllBlobs`, multi-shard `pack-amf` (`--dtype auto\|f16\|bf16`, 3D `switch_mlp` slicing, `tie_word_embeddings` dedupe): `zig build` + `zig build test` (Debug/ReleaseSafe: 100/100), `tools` unittest (23/23), real conversion & ReleaseFast `verify-amf` of `qwen3-0.6b-bf16.amf` (310 tensors, 35 MB RSS), `qwen3-14b-4bit.amf` (443 tensors, 116 MB RSS), and `qwen3-30b-a3b-3bit.amf` (18,867 tensors, 127 MB RSS) | Pass |
| 2026-11-20 | M2.5 `G12` (`32dc321`): native `FP16`/`BF16` GEMV/GEMM, `BF16` RMSNorm/QK-Norm, `BF16`/`q3`/`q4` MoE router, `FP16`/`BF16` deduplicated MoE SwiGLU, GPU embedding lookup: `zig fmt --check`, `zig build`, `zig build test` (Debug/ReleaseSafe/ReleaseFast: 112/112), `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (111/111, 0 validation errors) | Pass |
| 2026-11-21 | M2.5 `W13` + `bench-m25` + real-checkpoint gate (`1893ab6`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build`, `zig build test` (Debug/ReleaseSafe/ReleaseFast: 121/121), direct Debug test binary under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (120/120, API + GPU Validation enabled, 0 errors), `tools` unittest (23/23), ReleaseFast `aminfer bench-m25` (16 `FP16`/`BF16` GEMV/GEMM shapes up to `88.25 GB/s` / `2.690 TFLOP/s` + synthetic `BF16` decode), and real full-layer `verify-amf` + bilingual EN/ZH `decode` + `bench-decode` (`seq=512`, `swapouts_delta = 0`) on `Qwen3-0.6B` BF16 (`47.88 tok/s`, 32-tok `62.98–66.17 tok/s`), `Qwen3-14B-4bit` (`9.52 tok/s`), and `Qwen3-30B-A3B-3bit` (`51.86 tok/s`) archived in [benchmarks/results.md](benchmarks/results.md) | Pass |
| 2026-11-21 | M3 Phase 1 (`cb6ad86`, `c8414eb`, `d240329`): L1 SAM (`D1`), dual-source trie (`D6`), D7 submodular pruner, D9 acceptance; Metal 4 tree-attention & KV commit (`G5`), CoreML ANE zero-copy bridge (`D5`), Spikes `S4` & `S5`; `.amf` `DType.u32 = 5` & EAGLE-3/EAGLE-2/MTP packers (`tools` unittest: 28/28, real `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` packed & verified with `aminfer verify-amf`: 15 tensors, 996 MiB) | Pass |
| 2026-11-21 | M3 final gate (`034492f`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 158/158 tests), direct Debug test binary under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (156/156, API + GPU Validation enabled, 0 errors), `tools` unittest (28/28), ReleaseFast `aminfer bench-m3` (`S4` `d5_gate=gpu`, `S5` `beta_dedup=36.78 GB/s` / crossover `M=2`, `L1 SAM` `0.041 us`, `swapouts_delta = 0`), and real-checkpoint `T = 0` lossless `decode` vs `spec-decode` (100% bit-for-bit `token_ids` match, `swapouts_delta = 0`) on `Qwen3-0.6B` BF16 (`4.73×`, plus `T = 0.7` `D9`), `Qwen3-14B-4bit` (`1.71×`), and `Qwen3-30B-A3B-3bit` + real `EAGLE-3` (reasoning `48.38 tok/s` = `1.94×` vs unpruned / `1.05×` vs base; code-edit `L1 + D7` `59.24 tok/s` = `1.18×`) archived in [benchmarks/results.md](benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5) | Pass |

The measured platform facts are collected in
[architecture/hardware.md](architecture/hardware.md).

## Decision log

Decisions are recorded as ADRs; the index is
[decisions/README.md](decisions/README.md). In the 2026-11-20 init session the
user accepted ADR-0008 (benchmark-gated draft placement), ADR-0009
(expert-byte-budget pruning), ADR-0010 (hybrid wavefront prefill), ADR-0011
(Metal 4 hot path, macOS 26.0 floor), ADR-0012 (wired budget headroom) and
ADR-0013 (EAGLE-2 tree with EAGLE-3 / EAGLE-2 heads). ADR-0001 to ADR-0007 and
ADR-0014 record the project charter and the four new requirements of the same
session. After M2 completed on 2026-11-20, the user added the tiered frontier
model roadmap ([ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md)):
Tier 2 `Qwen3.8-Flash-Next` (`qwen4_exp`, M6) and Tier 3
`DeepSeek-V4.1-Flash-1017` (`deepseek41`, M7 / 512 GB Ultra & Thunderbolt 5 RDMA
clusters), and scheduled **M2.5** right before M3 for native `FP16` / `BF16`
execution and multi-shard safetensors streaming import
([ADR-0016](decisions/0016-native-fp16-bf16-and-multishard-import.md)).
During M3 close-out on 2026-11-21, the optional offline **GSQ + RCO**
per-tensor / per-expert hybrid quantization pipeline (`W14`, `G13`, targeting
M5 with strict per-block granularity and zero runtime Python or intra-tensor
bit mixing) was accepted as
[ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md).

## Session handoff snapshot

Update this before the context window runs out or when handing off.

- Last updated: 2026-11-21
- Commits so far:
  - M0: `1ad0da8` AGENTS.md, `6d5053d` license, `3a573a5` docs baseline,
    `0518ea8` build graph and probe, `06e8656` wired regions, `e6c9221`
    ExpertSource, `1b49162` Metal 4 bridge, `f1cd47f` tools scaffold, `a89b83e`
    justfile, `0430fe1` M0 close-out
  - M1: `936f16c` wired budget & `WiredArena` (R1), `36eb191` comptime quant
    specs & `.amf` v0 reader/validator (W1, W3, G2 host), `3a90c56` `.amf`
    packer & MLX importer (W2, W4), `04d0a62` Metal 4 multi-frame batching &
    specialized pipelines (F6 hot path, R2, G2 GPU), `65a1a62` fused 3/4-bit
    dequant GEMV/GEMM kernels (G1), `6bfe351` VM/heap probe, C9 bandwidth
    calibration & Spike S2 harness (C9, S2), `1632227` M1 close-out, `9ab393b`
    unified benchmark log & sysctl/powermetrics verification
  - M2: `dfb2906` synthetic Qwen3 MoE/dense `.amf` & tokenizer generator,
    `bd05973` lock-free rings, QoS steering & native Qwen BPE tokenizer (C1, C2,
    W5), `e29033f` IOSurface KV tiles, Spike S1 & GQA decode attention (R3, R4,
    S1, G4), `3a90b9f` deduplicated MoE gather/SwiGLU & greedy argmax (G3),
    `c2b5856` M2 build/root wiring, `3197b12` Qwen3-MoE & Qwen3-Dense greedy
    decode engine (W6), `517ba0d` M2 benchmarks & `aminfer decode` CLI,
    `e1ea8a7` M2 close-out & benchmark archive update
  - Roadmap & M2.5 planning: `188dc75` tiered frontier model roadmap (`ADR-0015`),
    `287f0e9` M2.5 plan (`ADR-0016`), `05d6de7` real-checkpoint layout notes
  - M2.5: `2a48c66` `.amf` `bf16` dtype & streaming multi-shard packer (`W12`),
    `32dc321` Metal 4 native `FP16`/`BF16` GEMV, GEMM, MoE, norm & embedding
    kernels (`G12`), `1893ab6` unquantized/mixed-precision `Qwen3Model` &
    `bench-m25` (`W13`), `9200b25` M2.5 benchmark archive & close-out docs
  - M3: `cb6ad86` L1 SAM, dual-source prefix trie, D7 pruner & D9 lossless
    acceptance (`D1`, `D6`, `D7`, `D9`), `c8414eb` tree-attention verification,
    KV commit, CoreML ANE zero-copy bridge & Spikes `S4`/`S5` (`G5`, `D5`, `S4`,
    `S5`), `d240329` `.amf` `u32` dtype & EAGLE-3/EAGLE-2/MTP draft head packers,
    `034492f` end-to-end `DraftHead`/`SpecEngine`/`Qwen3Model.verifyTree`,
    `aminfer spec-decode` & `bench-m3` (`D2`, `D3`, `D5`–`D7`, `D9`, `G5`),
    `e46b7f5` M3 benchmark archive & close-out docs
  - Roadmap docs: `880d487` optional offline GSQ+RCO hybrid quantization
    ([ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md)), plus
    `W14`/`G13` status & process registration
- In flight: none (M3 completed; ready to begin M4 Neural Engine and storage)
- Awaiting the user: nothing
- Key evidence: [architecture/hardware.md](architecture/hardware.md),
  [benchmarks/results.md](benchmarks/results.md),
  [decisions/0004-kv-cache-iosurface-texture-sharing.md](decisions/0004-kv-cache-iosurface-texture-sharing.md),
  [decisions/0008-benchmark-gated-draft-placement.md](decisions/0008-benchmark-gated-draft-placement.md),
  [decisions/0009-expert-byte-budget-tree-pruning.md](decisions/0009-expert-byte-budget-tree-pruning.md),
  [decisions/0011-metal4-zero-allocation-hot-path.md](decisions/0011-metal4-zero-allocation-hot-path.md),
  [decisions/0013-eagle-tree-and-head-formats.md](decisions/0013-eagle-tree-and-head-formats.md),
  [decisions/0016-native-fp16-bf16-and-multishard-import.md](decisions/0016-native-fp16-bf16-and-multishard-import.md),
  [decisions/0017-optional-gsq-rco-hybrid-quantization.md](decisions/0017-optional-gsq-rco-hybrid-quantization.md)
