# Process

The single record of project progress: milestones, spikes, the current task
list, the verification log and the session handoff snapshot. Capability status
lives in [status.md](status.md); decisions live in
[decisions/](decisions/README.md).

## Current focus

M4.5 — bare-metal ARMv9 SME2 pre-packed & `ZT0`-LUT engine completed (2026-11-21); next: **M5 — serving and pipelining** (`C6`–`C8`, `D8`, `W14`, `G13`), followed by **M6 — frontier hybrid sparse models (`Qwen3.8-Flash-Next`)**.

## Milestones

Status-row IDs refer to [status.md](status.md).

| Milestone | Goal | Status rows | Exit criteria | State |
| --- | --- | --- | --- | --- |
| M0 | Project initialization | F1–F9 | `zig build` and `zig build test` pass in Debug and ReleaseSafe; docs baseline committed; local ikat index built | Completed |
| M1 | GPU and weight foundation | R1, R2, F6 (hot path), W1–W4, G1, G2, C9 | Wired budget enforced at startup; `.amf` round-trip (tools → Zig) with alignment validation; fused dequant GEMV / GEMM match a reference within tolerance; effective bandwidth measured per kernel | Completed |
| M2 | Non-speculative decode | W5, W6, C1, C2, R3, R4, G3, G4 | Greedy end-to-end decode of the primary target from `.amf`; baseline tokens/s and bytes per token recorded | Completed |
| M2.5 | Standard FP16 / BF16 and multi-shard import | W12, W13, G12 | Multi-shard `aminfer-tools pack-amf` converts local `Qwen3-30B-A3B-3bit`, `Qwen3-14B-4bit`, and `Qwen3-0.6B` (`BF16`) with bounded RAM; native `FP16` and `BF16` GEMV / GEMM / MoE SwiGLU match CPU fp32 reference; real-checkpoint greedy decode verified end-to-end (incl. `Qwen3-0.6B` `BF16`) with bandwidth archived in [benchmarks/results.md](benchmarks/results.md) | Completed |
| M3 | Speculative decoding | D1–D3, D5–D7, D9, G5 | Lossless output versus M2/M2.5; speedup over the non-speculative baseline, acceptance-length and experts-per-pass statistics recorded | Completed |
| M4 | Neural Engine and storage | A1–A3, G6, W7, W8, C3–C5, D4 | Hybrid prefill throughput versus GPU-only prefill; expert-prediction recall and prefetch lead time; cold-expert decode without swap | Completed |
| M4.5 | Bare-metal ARMv9 SME2 pre-packed & `ZT0`-LUT engine | W15, D11, C11–C13 | Pre-`smstart` `d8–d15` canary & `fp32` parity tests pass; `sme2_panel64` single-owner layout preserves `.amf` v0 binary compatibility; Spike `S7` benchmarks 3-way `.s` kernels vs `Accelerate`/`BNNS`, P-cluster thread saturation, and GPU co-execution (`> 1.05×` gate, `0` swap); Secure Enclave TRNG `D11` seeding verified | Completed |
| M5 | Serving and pipelining | C6–C8, D8, W14, G13 | Streaming server under load; pipelined-drafting gain measured; optional GSQ+RCO per-block `QuantScheme` pipeline verified | Planned |
| M6 | Frontier hybrid sparse models (`Qwen3.8-Flash-Next`) | R6, W9, W10, G7–G9, G10 (GR), C10 (PLE) | Composable model operator graph; `Qwen3.8-Flash-Next` (`qwen4_exp`: Gated DeltaNet + QSA + 512-expert MoE + 51B NVMe PLE + 4B MTP) verified end-to-end (128 GB+ full wired residency; 24 GB M4 hot/cold + PLE async prefetch without swap) | Planned |
| M7 | Ultra & cluster scale-out (`DeepSeek-V4.1-Flash-1017`) | R7, W11, G10 (mHC), G11, D10, C10 (Engram), X1–X3 | Causal Encoder-Decoder scheduler, MLA/CSA2 + FP4 KV, 196B Engram lookup, DSpark drafting and Thunderbolt 5 RDMA expert parallelism verified on 512 GB Ultra or multi-Mac TB5 clusters | Future |

## Spikes

Short measurements that must run before the related design moves from Draft to
Confirmed, or before an accepted design is tuned.

| ID | Question | Feeds | Milestone | State |
| --- | --- | --- | --- | --- |
| S1 | KV tile read path: R16Float vs RGBA16Float texture vs aliasing `MTLBuffer`; token- vs channel-major rows; decode-attention bandwidth | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), R4 | M2 | Closed |
| S2 | Metal 4 steady-state allocations and encode overhead per decode round (Instruments Allocations, signposts) | [ADR-0011](decisions/0011-metal4-zero-allocation-hot-path.md) | M1 | Closed |
| S3 | Cold-expert reads: `aio_read` + `kqueue` vs E-core `pread` workers; queue depth 1–16; 2–3 MiB reads with `F_NOCACHE`; QoS and I/O-policy effects; p50 / p99 latency | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), C4 | M4 | Closed |
| S4 | ANE handoff latency (GPU → CPU → ANE → CPU → GPU) and bandwidth contention of concurrent ANE weight streaming on GPU GEMV | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md) | M3 | Closed |
| S5 | Unique experts per layer versus draft-tree size for Qwen3-30B-A3B; effective bandwidth of deduplicated vs per-slot expert kernels | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md) | M3 | Closed |
| S6 | ANE attention block: CoreML compilation, static shapes for a growing KV length, fp16 numerics, throughput vs GPU | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), A3 | M4 | Closed |
| S7 | Bare-metal M4+ SME2 throughput (`M=1..4` `ZT0-LUT+Dot`, `M=5..16` `1×4` `ZA`, `M>=32` `2×2` `ZA`) vs `Accelerate`/`BNNS`, P-cluster threads-per-cluster saturation, `otool` zero-spill check, and unified-memory bus contention gate vs Metal GPU | [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), C13 | M4.5 | Closed |

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

## M4 task list

- [x] `F_NOCACHE` direct NVMe async I/O (`KqueueAioEngine` `EVFILT_AIO` auto-reap `C3`, `EcorePreadPool` E-core worker fallback `C4`, `AsyncExpertReader`, Spike `S3`) and preallocated wired `ExpertSlotPool` + LFU-decay `ExpertCache` with pass pinning & prefetch coalescing (`C5`) in `src/storage/{aio,cache}.zig`
- [x] Offline W8 expert heat profiler (`aminfer-tools expert-heat`, `schema_version = 1`, `pack-amf --heat-profile`) and pure-stdlib CoreML `.mlpackage` exporter (`W7`, `aminfer-tools export-coreml` for expert predictor, ANE attention QKV, and draft FFN) in `tools/` (40/40 unittests; real `qwen3-30b-a3b-heat.json` and `.mlpackage` artifacts verified)
- [x] Hybrid 2-chunk ANE/GPU wavefront prefill scheduler (`WavefrontPrefill`, `A1`), static power-of-two KV-bucket ANE attention (`A3`, Spike `S6`), and zero-copy IOSurface ANE cross-layer MoE expert predictor (`AneExpertPredictor`, `A2`) in `src/ane/{wavefront,predictor}.zig`
- [x] Engine integration in `src/model/qwen3.zig`, `src/spec/{draft,engine}.zig`, `src/bench/runner.zig`, and `src/main.zig`: `runLayerFfnChunk` (`G6`) + `prefillHybridWavefront` (`A1`), cold-expert NVMe streaming with overlapped ANE lookahead prefetch & `D7` cold-penalty residency bitmasks (`C3`–`C5`, `A2`, `W8`), `D4` standalone `SmallDraftModel` (`--draft-mode small`), and `aminfer bench-m4` / `aminfer bench-a2`
- [x] Real-checkpoint M4 ReleaseFast benchmarks archived in [benchmarks/results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft): Spikes `S3` & `S6`, real 48L `Qwen3-30B-A3B-3bit` `A2` predictor recall & lead time, real 48L `Qwen3-30B-A3B-3bit` cold-expert NVMe decode (`6,869–7,751 MiB` < `8,192 MiB` cap, 100% bit-exact, `0` swapouts), and real `Qwen3-14B-4bit` + `Qwen3-0.6B` BF16 `D4` small draft (`1.27–1.40×`, 100% bit-exact, `0` swapouts)

## M4 Neural Engine, storage and D4 small draft results

Canonical methodology, full tables, and verbatim raw run outputs live in
[benchmarks/results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft).
Key M4 highlights on Apple M4 24 GiB (commits `176740c..8701fdb`, ReleaseFast,
`swapouts_delta = 0` across all runs):

- **Spike S3 (`C3` + `C4` settled)**: On `/Volumes/Akari` (external SSD capped
  at `~0.64 GB/s`), `kqueue_aio` (`SIGEV_KEVENT` → `EVFILT_AIO`) peaks at
  **`0.638 GB/s`** (Q3 `2,064,384 B`, QD=16, `IOPOL_IMPORTANT`) vs 4-worker
  `ecore_pread` **`0.626 GB/s`** (tighter QD=16 p99 tail `53.7 ms` vs `84.2 ms`),
  with `0` swapouts. Both remain first-class via `--io-backend {auto,aio,pread}`.
- **Spike S6 (`A3` settled) & `A1`/`G6` hybrid wavefront prefill**:
  - `bucketed_full_ane` (`B ∈ {512, 1024, 2048, 4096}`) maps 100% of attention
    ops onto ANE (`25/25`–`27/27`, `zero_copy = yes`), stays within `0.0024`
    max abs error vs CPU FP64 at `kv_len = 1024`, and beats GPU-only attention
    by **`1.66–2.22×`** across every bucket.
  - 2-chunk ANE/GPU wavefront overlap achieves **`1.189×`** (`1,767` vs `1,485 tok/s`,
    `1.198×` overlap efficiency) on 6-layer `Qwen3-30B-A3B` geometry and
    **`1.87×` / `1.76×`** (`68,761` / `62,402 tok/s`) on end-to-end synthetic
    `1024`/`2048`-token `Qwen3Model` prefill.
- **Real 48-layer `Qwen3-30B-A3B-3bit` `A2` cross-layer expert predictor**:
  - In-memory `post_attention_layernorm`-folded ANE router bank executes in
    **`156.44 us`** (`uses_ne = yes`, `zero_copy = yes`, **`17.85×` faster**
    than GPU 3-layer lookahead `2,792.99 us`), matching exact GPU lookahead
    recall (`R@8/12/16 = 0.861/0.958/0.982` at `delta = 1`, `0.809/0.927/0.960`
    at `delta = 2`, `0.774/0.891/0.933` at `delta = 3`) with **`646–1,048 us`**
    of async NVMe lead time (`+15.6 pp` `R@8` over unfolded offline `.mlpackage`).
- **Real 48-layer `Qwen3-30B-A3B-3bit` cold-expert NVMe decode (`C3`–`C5` + `A2` + `W8`)**:
  - Keeping `64` cold experts/layer (`3,072` experts = `6.05 GiB`) on disk cuts
    wired RAM from `12,791 MiB` down to **`6,869 MiB` (`-46.3%`, 64 slots)** or
    **`7,751 MiB` (`-39.4%`, 512 slots)** — cleanly passing `--wired-cap-mib 8192`
    with **32/32 (100%) bit-for-bit identical `token_ids`** and `swapouts_delta = 0`.
  - With `512` slots (`aio` + ANE predictor), total cold hit rate reaches
    **`85.2%`** (`+17.0 pp` over LFU-only `68.2%`), synchronous misses drop
    **`-53.4%`** (`1,879 → 876`), and decode reaches **`3.02 tok/s`** even on
    the `0.64 GB/s` external SSD (vs `0.94 tok/s` at 64 slots).
- **Real `Qwen3-14B-4bit` + `Qwen3-0.6B` BF16 `D4` same-family small draft**:
  - 100% bit-for-bit lossless (`64/64` `token_ids` on all 3 prompts, `0` swapouts,
    `D5` gate → `.gpu`): **`15.13 tok/s` (`1.40×`**, `tau = 3.938`, `-74.6%`
    bytes) on Python LRU cache, **`13.81 tok/s` (`1.27×`**, `tau = 3.500`,
    `12.48 tok/s` default tree) on B-tree + binary search, and `tau = 1.909–2.100`
    on zero-L1-seed Rust prose (`l1_seed_rounds = 0`).

## M4.5 task list

- [x] Runtime `FEAT_SME2` / `SME_B16F32` / `SME_F16F32` / `pcore_clusters` / `pcore_l2_kib` platform probe (`C11`, `src/platform/sysinfo.zig`, `aminfer probe`) and scoped `-march=armv9.2-a+sme2+sme-f16f16+sme-b16b16` build rule restricted exclusively to `src/cpu/sme2_kernels.s` (verified `0` SME/SVE leaks on `-Dcpu=apple_m1`)
- [x] Hand-written ARMv9.2 `.s` SME2 3-way microkernels (`M=1..4` `ZT0-LUT + Dot`, `M=5..16` `1×4` `ZA`, `M>=32` `2×2` `ZA`) across `bf16`/`f16`/`q4`/`q3`/`q2`, pre-`smstart` `x19–x30`/`d8–d15` ABI canary, `ld1b`/`ldnt1b` cache steering, fused `ZA` epilogue (`add_residual`, `silu_mul_gate`), and P-cluster lock-free dispatcher (`C11`, `C12`, `src/cpu/{sme2_kernels.s,sme.zig}`)
- [x] Binary-compatible `.amf` v0 `TensorEntry.reserved0` `sme2_panel64` layout tag (`W15`, 128-byte struct preserved, `reserved0 == 0` stays `row_major`), Python `aminfer-tools repack-sme2` & `pack-amf --sme2-targets` pre-packer (`tools/`), and `D11` two-tier Apple Silicon Secure Enclave TRNG (`getentropy`) → slot-local `Xoshiro256` `RequestRng` (`src/runtime/rng.zig`)
- [x] Engine & CLI integration (`src/model/qwen3.zig`, `src/spec/engine.zig`, `src/bench/runner.zig`, `src/main.zig`): single-owner `sme2_panel64` loading (zero RAM duplication), L2-resident SME2 speculative tree router prediction (`C13`, `--sme-router auto|sme2|gpu`), `evalSwiGluExpertSme2` (`ldnt1b`), `D11` TRNG seeding in `decode`/`spec-decode` with `seed_used`/`seed_source`, and `aminfer bench-m45` (`Spike S7` + `C13` concurrent bus gate + `ldnt1b` L2 preservation)
- [x] Real-hardware ReleaseFast audits & real-checkpoint M4.5 benchmarks archived in [benchmarks/results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng): `otool` zero-spill check, 4-checkpoint `verify-amf` compatibility, real `Qwen3-0.6B` `repack-sme2` + `D11` TRNG/replay, and real `Qwen3-30B-A3B-3bit` + `EAGLE-3` GPU-router vs SME2 L2-router A/B (`0` swapouts)

## M4.5 bare-metal SME2, C13 and D11 results

Canonical methodology, full tables, and verbatim raw run outputs live in
[benchmarks/results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng).
Key M4.5 highlights on Apple M4 24 GiB (commits `7de016d..b23f787`, ReleaseFast,
`swapouts_delta = 0` across all archived runs):

- **ABI safety & zero-spill binary audit (`C11`, `C12`)**:
  - `-Dcpu=apple_m1` build contains **`0`** SME/SVE instructions outside the 34
    `_ami_sme2_*` symbols (M1–M3 fallback safe without `SIGILL`).
  - Native ReleaseFast `otool -tvV` audit confirms `minos 26.0`, all 30 entry
    kernels save `x19–x30` & `d8–d15` before `smstart` and restore after
    `smstop` (`pre_saves = 10`, `post_restores = 10`), and **`0`** stack-pointer
    spills inside streaming regions (`total_streaming_sp_accesses = 0`).
- **Spike S7 (`C12`, `aminfer bench-m45` vs Apple `Accelerate` `cblas_sgemm`)**:
  - L2-hot `bf16` (`1024×2048`): peaks at **`1.553 TFLOP/s`** (`M = 128`),
    beating `Accelerate` by **`4.00×`** at `M = 8` (`49.58 us`) and **`3.80×`**
    at `M = 16` (`46.13 us`, `1.455 TFLOP/s`); `f16` peaks at **`1.584 TFLOP/s`**
    (`M = 512`, `1.06×`).
  - Quantized `ZT0-LUT` (`q4`/`q3`/`q2`): beats `dequant + Accelerate` by
    **`14.06–20.43×`** in L2 (`M = 1`) and **`17.81–26.56×`** on DRAM SLC-bust
    `4096×4096` (`bf16 ldnt` **`70.89 GB/s`**, `2.19×` `Accelerate`; `q4`/`q2`
    `4.60–4.77×` `Accelerate`), sustaining **`1.474–1.568 TFLOP/s`** at `M = 128`.
  - P-cluster thread sweep (`1/2/4` threads on M4's 1 P-cluster): 1 thread
    saturates `95–100%` of L2-hot SME2 compute (`1.00–1.05×` at 2–4T, peaking at
    `1.717 TFLOP/s` at 2T, dipping to `0.91×` at 4T), validating 1 worker per
    P-cluster for compute-bound workloads; DRAM-bound `q4 M = 1` scales
    `1.00× → 1.17× → 1.24×` (`54.49 → 67.74 GB/s`).
- **`C13` concurrent Metal 4 GPU + CPU SME2 bus gate (`> 1.05×`) & `ldnt1b` L2 preservation**:
  - **L2-resident SME2 router + GPU `gemv_q3`**: GPU `78.04 → 79.62 GB/s`
    (`+2.0%`, zero bus penalty), aggregate **`155.57 GB/s`**, **`1.80×`** net
    speedup (**PASS**).
  - **DRAM SME2 `q4` + GPU `gemv_q3`**: `ld1b` trims GPU `-12.5%` (`90.65 GB/s`,
    `1.33×` **PASS**); `ldnt1b` cuts GPU interference by a third to **`-8.2%`**
    (`71.64 GB/s`, **`93.49 GB/s`** aggregate, **`1.33×` PASS**).
  - **`ldnt1b` cold-expert L2 preservation**: sweeping `16.88 MiB` (`6` cold `q4`
    SwiGLU experts) with `ld1b` slows the hot `4 MiB` router by **`+40.9%`**
    (`44.17 → 62.25 us`), whereas **`ldnt1b`** keeps the hot router at
    **`44.08 us` (`-0.2%`, zero L2 eviction)** and speeds cold experts by `-5.6%`
    (`86.31 → 81.51 us/expert`).
- **Real `Qwen3-30B-A3B-3bit` + `EAGLE-3` (`C13` L2-resident SME2 tree router, `T = 0`)**:
  - Startup calibration on the `4 MiB` (`8×128×2048`) router bank selects `sme2`
    at **`78.0 us` vs `975.5 us` GPU (`12.51×` faster)** (`17.71×`/`11.48×`/`7.15×`
    across `M = 1/8/16` in `bench-m45`).
  - End-to-end 48L `EAGLE-3 + L1 + D7` speculative decode improves
    `prune_ms_per_round` **`0.396 → 0.217 ms` (`1.82×` faster, `-45.2%`)**,
    cuts Metal CB submission `malloc_blocks_delta` **`320 → 194` (`-39.4%`)**,
    and lifts decode throughput **`49.73` (base) → `52.49` (GPU router) →
    `52.97 tok/s` (SME2 router, `1.065×` vs base)** with **64/64 (100%)
    bit-exact `token_ids`** and `swapouts_delta = 0`.
- **Real `Qwen3-0.6B` BF16 `D11` Secure Enclave TRNG + bit-exact replay & `W15` compatibility**:
  - All 4 existing `.amf` checkpoints pass `aminfer verify-amf` unchanged (`100%`
    v0 binary backward compatibility). `repack-sme2 --targets all-linear` repacks
    196 projections in-place (`1,195 MiB` wired, zero RAM duplication, `0`
    swapouts); full-model `M = 1` CPU SME2 (`21.94 tok/s`) vs GPU (`68.67 tok/s`)
    re-confirms keeping full dense decode on GPU while steering L2 routers and
    cold-expert digestion to SME2.
  - Both `decode` and `spec-decode` at `--temperature 0.7` draw distinct 64-bit
    Secure Enclave `hardware_trng` seeds with diverse outputs when `--seed` is
    omitted, and reproduce **32/32 (100%) bit-exact `token_ids`** when replayed
    with `--seed <seed_used>` (`seed_source = explicit`, `swapouts_delta = 0`).

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
| 2026-11-21 | M4 Phase 1 (`176740c`, `cf498b3`, `02d22a0`): `F_NOCACHE` `kqueue_aio`/`ecore_pread`/Spike `S3` & wired `ExpertSlotPool`/`ExpertCache` (`C3`–`C5`); W8 `expert-heat` + W7 pure-stdlib `export-coreml` (`tools` unittest: 40/40; real `qwen3-30b-a3b-heat.json` & `.mlpackage` exports validated); `WavefrontPrefill`, static KV-bucket ANE attention / Spike `S6` (`A1`, `A3`), and `AneExpertPredictor` (`A2`) | Pass |
| 2026-11-21 | M4 Phase 2 & real-checkpoint gate (`6b6fb09`, `8701fdb`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 188/188 tests), `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` on M4/D4 tests (0 validation errors), ReleaseFast `aminfer bench-m4` (Spikes `S3` & `S6`, synthetic `A2`, `A1`/`G6` `1.76–1.87×`, synthetic cold-expert parity, synthetic `D4` lossless, `0` swapouts), real 48L `Qwen3-30B-A3B-3bit` `bench-a2` (`156.44 us` ANE `R@8/12/16 = 0.861/0.958/0.982` at `delta = 1`, `0` swapouts), real 48L `Qwen3-30B-A3B-3bit` cold-expert NVMe `decode` (`6,869–7,751 MiB` < `8,192 MiB` cap, `85.2%` hit rate / `3.02 tok/s` at 512 slots, 32/32 bit-exact, `0` swapouts), and real `Qwen3-14B-4bit` + `Qwen3-0.6B` BF16 `D4` `spec-decode` (`13.81–15.13 tok/s` = `1.27–1.40×`, `tau = 3.50–3.94`, 64/64 bit-exact across 3 prompts, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) | Pass |
| 2026-11-21 | M4.5 Phase 1 (`7de016d`, `945cb4d`): ARMv9.2 `.s` SME2 3-way microkernels (`bf16`/`f16`/`q4`/`q3`/`q2`, `ld1b`/`ldnt1b`, fused `ZA` epilogue, pre-`smstart` `d8–d15` canary, P-cluster dispatcher) (`C11`, `C12`), `W15` `sme2_panel64` `.amf` v0 layout & `aminfer-tools repack-sme2`, and `D11` Secure Enclave TRNG `RequestRng` | Pass |
| 2026-11-21 | M4.5 Phase 2 & real-checkpoint gate (`09de8e5`, `bd17c20`, `b23f787`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 209/209 tests), direct Debug test binary under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (206/206, API + GPU Validation enabled, 0 errors), `tools` unittest (51/51, incl. Python `repack-sme2` → Zig `decode`/`spec-decode` parity & `D11` TRNG CLI), `-Dcpu=apple_m1` `otool` audit (`0` SME/SVE leaks outside 34 `_ami_sme2_*` symbols), native `otool -tvV` spill audit (`minos 26.0`, `0` streaming `sp` accesses), ReleaseFast `aminfer probe` + `aminfer bench-m45` (`Spike S7` up to `1.717 TFLOP/s` / `4.00×` Accelerate / `26.56×` dequant+Accelerate, `C13` concurrent bus `1.80×`/`1.33×` PASS, `ldnt1b` `-0.2%` vs `ld1b` `+40.9%` hot L2 penalty, `0` swapouts), `verify-amf` on all 4 real `.amf` checkpoints, real `Qwen3-0.6B` `repack-sme2` & `D11` TRNG + bit-exact replay (`decode` & `spec-decode`), and real `Qwen3-30B-A3B-3bit` + `EAGLE-3` GPU-router vs SME2 L2-router A/B (`prune 0.396 → 0.217 ms`, `52.49 → 52.97 tok/s`, 64/64 bit-exact, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) | Pass |

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
During M4 close-out on 2026-11-21, the user formally confirmed inserting
Milestone **M4.5** for the bare-metal ARMv9 SME2 pre-packed & `ZT0`-LUT engine
(`C11`–`C13`, `W15`, Spike `S7`,
[ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md)), adopting
two-tier Apple Silicon Secure Enclave TRNG (`getentropy`) → slot-local
`Xoshiro256` per-request sampling RNG (`D11`), and resolving `R5`, `C8`, `D8`,
`ADR-0005`, `ADR-0016` residual alignment, and `ADR-0017` `G13` initial scheme
selection ([architecture/open-decisions-research.md](architecture/open-decisions-research.md)).

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
    `W14`/`G13` status & process registration; `98c43c6` external research &
    open-decision recommendations (`docs/architecture/open-decisions-research.md`)
  - M4: `176740c` `F_NOCACHE` kqueue AIO, E-core pread pool, Spike `S3` & LFU
    expert cache (`C3`–`C5`, `S3`), `cf498b3` offline expert-heat profiler &
    pure-stdlib CoreML `.mlpackage` exporter (`W7`, `W8`), `02d22a0` hybrid
    wavefront prefill, Spike `S6` static-shape buckets & ANE expert predictor
    (`A1`–`A3`, `S6`), `6b6fb09` M4 engine/CLI/benchmark integration (`G6`,
    `D4`, cold-expert NVMe streaming, `bench-m4`, `bench-a2`), `8701fdb`
    two-phase `bench-a2` weight release for zero-swap real-checkpoint eval,
    `bd93b09` M4 benchmark archive & close-out docs
  - M4.5 / confirmed decisions docs: `f91fff9` `ADR-0018` bare-metal SME2
    engine, `D11` TRNG seeding & confirmed open-decision codification, plus
    M4.5 status/process roadmap sync
  - M4.5: `7de016d` ARMv9.2 `.s` SME2 3-way microkernels, `ZT0-LUT` panels &
    P-cluster dispatcher (`C11`, `C12`), `945cb4d` `W15` `sme2_panel64` `.amf`
    v0 layout tooling & `D11` Secure Enclave TRNG `RequestRng`, `09de8e5` root
    module exports & test deduplication, `bd17c20` single-owner `sme2_panel64`
    loading, L2-resident SME2 router prediction (`C13`) & `ldnt1b` cold-expert
    eval in `Qwen3Model`/`SpecEngine`, `b23f787` `aminfer bench-m45` (`Spike S7`
    + `C13` concurrent bus gate), `probe` SME2 facts, `D11` TRNG CLI &
    Python-Zig numerical parity, plus M4.5 benchmark archive & close-out docs
- In flight: none (M4.5 completed; ready to begin M5 serving and pipelining)
- Awaiting the user: nothing
- Key evidence: [architecture/hardware.md](architecture/hardware.md),
  [benchmarks/results.md](benchmarks/results.md),
  [decisions/0004-kv-cache-iosurface-texture-sharing.md](decisions/0004-kv-cache-iosurface-texture-sharing.md),
  [decisions/0006-amf-cold-experts-direct-io.md](decisions/0006-amf-cold-experts-direct-io.md),
  [decisions/0008-benchmark-gated-draft-placement.md](decisions/0008-benchmark-gated-draft-placement.md),
  [decisions/0009-expert-byte-budget-tree-pruning.md](decisions/0009-expert-byte-budget-tree-pruning.md),
  [decisions/0010-hybrid-wavefront-prefill.md](decisions/0010-hybrid-wavefront-prefill.md),
  [decisions/0011-metal4-zero-allocation-hot-path.md](decisions/0011-metal4-zero-allocation-hot-path.md),
  [decisions/0013-eagle-tree-and-head-formats.md](decisions/0013-eagle-tree-and-head-formats.md),
  [decisions/0016-native-fp16-bf16-and-multishard-import.md](decisions/0016-native-fp16-bf16-and-multishard-import.md),
  [decisions/0017-optional-gsq-rco-hybrid-quantization.md](decisions/0017-optional-gsq-rco-hybrid-quantization.md),
  [decisions/0018-bare-metal-armv9-sme2-engine.md](decisions/0018-bare-metal-armv9-sme2-engine.md)
