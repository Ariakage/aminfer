# Process

The single record of project progress: milestones, spikes, the current task
list, the verification log and the session handoff snapshot. Capability status
lives in [status.md](status.md); decisions live in
[decisions/](decisions/README.md).

## Current focus

M6.5 — continuous batching, long-context KV tiering, shared prefix caching, and zero-copy multimodal ingress (`R5`, `R8`, `G14`, `C14`) completed (2026-11-22); next: **M7 — Ultra and cluster scale-out (`DeepSeek-V4.1-Flash-1017`, `R7`, `W11`, `G10` mHC, `G11`, `D10`, `C10` Engram, `X1`–`X2`)**.

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
| M5 | Serving and pipelining | C6–C8, D8, W14, G13 | Streaming server under load; pipelined-drafting gain measured; optional GSQ+RCO per-block `QuantScheme` pipeline verified | Completed |
| M6 | Frontier hybrid sparse models (`Qwen3.8-Flash-Next`), recurrent depth & heterogeneous MoE scaling | R6, W9, W10, W16, W17, G7–G9, G10 (GR), D12, D13, C10 (PLE) | Composable operator graph + `LoopSchedule` (`GDN S_tmp` isolation, `SelfSpecRecurrent`, L2 SME2 `Step-LoRA`); synthetic + scale-calibrated `qwen4_exp` (GDN + QSA + 512-expert MoE + 51B NVMe PLE + 4B MTP) verified end-to-end without swap; `Qwen3-30B-A3B` & 512-expert GSQ+RCO + `ZT0` non-uniform LUTs and `Placement.cpu_sme2` draft heads benchmarked | Completed |
| M6.5 | Continuous batching, long-context KV tiering & zero-copy multimodal ingress | R5, R8, G14, C14 | Server iteration-level continuous batching (`M > 1` slots per GPU/SME2 pass), shared prefix `KvTile`/`GDN` cache, 32K+ GPU `q8` retired KV tier, domain `.ang` packs, and `VideoToolbox → IOSurface → MTLTexture` zero-copy vision/video ingress verified without swap | Completed |
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
      bytes); `EAGLE-3 + L1 D6 seed + D7` lifts `tau` (`2.500 → 3.412`, `14/17`
      L1 seed rounds, `-45.3%` bytes/accepted token) but runs slower at
      `33.82 tok/s` (`0.67×` vs baseline) due to the `996 MiB` unquantized BF16
      draft-head `syncAccepted` cost (`8.31 ms/round`).

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
    `85.2%` (`+17.0 pp` over LFU-only `68.2%`), synchronous misses drop `-53.4%`
    (`1,879 → 876`), and decode runs at `3.02 tok/s` (`0.94 tok/s` at 64 slots,
    vs `48.47 tok/s` all-hot; bounded by the `0.64 GB/s` external SSD link).
- **Real `Qwen3-14B-4bit` + `Qwen3-0.6B` BF16 `D4` same-family small draft**:
  - Bit-for-bit lossless (`64/64` `token_ids` on all 3 prompts, `0` swapouts,
    `D5` gate → `.gpu`): `15.13 tok/s` (`1.40×`, `tau = 3.938`, `-74.6%` bytes)
    on Python LRU cache, `13.81 tok/s` (`1.27×`, `tau = 3.500`, `12.48 tok/s`
    default tree) on B-tree + binary search, and `9.55–10.55 tok/s` (`0.88–0.97×`
    vs `10.83 tok/s` baseline, `tau = 1.909–2.100`) on zero-L1-seed Rust prose
    (`l1_seed_rounds = 0`) where BF16 `0.6B` draft time exceeds saved verify steps.

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

## M5 task list

- [x] Zero-allocation `kqueue` HTTP/1.1 + SSE server (`C7`, `C8`, `src/server/http_sse.zig`) with preallocated connection slots, `SO_NOSIGPIPE`, `EVFILT_READ`/`EVFILT_WRITE`/`EVFILT_USER`, incremental UTF-8-safe `<think>...</think>` → `delta.reasoning_content` splitting, `D11` `seed_used` echo & replay, `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (`stream: true|false`), and thin stateless `POST /v1/responses` & `POST /v1/messages` adapters
- [x] `C6` 16 KiB-aligned `.ang` v1 NVMe N-gram table offline builder (`tools/src/aminfer_tools/ngram.py`, `aminfer-tools build-ngram`) and Zig `F_NOCACHE` reader with wired LRU slot cache & async `kqueue_aio` prefetch (`src/spec/ngram_nvme.zig`) integrated as L1 fallback into `SpecEngine`
- [x] `D8` target-latency-gated & contention-gated multi-prefix PEARL async pipelined small-draft engine (`src/spec/{draft,engine,tree}.zig`, `src/cpu/sme.zig`, `src/model/qwen3.zig`, `--pearl auto|on|off`, off-GPU pure-CPU ARMv9 SME2 `ldnt1b` `stepDraftTreeBatch` + independent Metal 4 queue fallback, lock-free worker overlapping round `r + 1` pre-drafting with `target.verifyTree(T_r)`, zero-rewind multi-prefix speculation, early-stop on verify completion, and post-L1-merge `hard_cap` truncation)
- [x] Optional offline Python GSQ + RCO sensitivity & residual-compensation `.amf` exporter (`W14`, `tools/src/aminfer_tools/gsq_rco.py`, `aminfer-tools gsq-convert` & `pack-amf --quant-profile gsq-rco`, scale-invariant LN peakiness proxy + boundary-layer protection) and Metal 4 `q2_affine` kernels + zero-sync 3-bucket `{q2, q3, q4}` scheme-bucketed MoE gather (`G13`, `src/kernels/{dequant_gemm,moe_ffn}.metal`, `src/gpu/{dequant,moe,attention}.zig`, `src/model/qwen3.zig`) wired into `SpecEngine` `D7` `PruneCostModel` ([ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md))
- [x] CLI & benchmark integration (`src/root.zig`, `src/main.zig`, `src/bench/runner.zig`): `aminfer serve`, `--ngram-nvme` & `--pearl` on `aminfer spec-decode`, and `aminfer bench-m5` (wired into `aminfer bench`)
- [x] Real-hardware, real-checkpoint & live-server ReleaseFast benchmarks archived in [benchmarks/results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization): idle-GPU `bench-m5`, real `Qwen3-0.6B` GSQ+RCO `4.35b` vs MinMax vs BF16 (`qwen3-0.6b-gsq-rco.amf`, fluent EN/ZH), repo-scale `aminfer-repo.ang` `spec-decode`, real `Qwen3-14B-4bit + Qwen3-0.6B` off-GPU SME2 (`qwen3-0.6b-gsq-rco-sme2.amf`) & same-GPU PEARL A/B + `Qwen3-30B-A3B-3bit` `--pearl auto` gate check, and live `aminfer serve` HTTP/SSE + `1/4/8`-client load test (`swapouts_delta = 0`)

## M5 serving, pipelining, C6 NVMe N-gram and GSQ+RCO results

Canonical methodology, full tables, and verbatim raw run outputs live in
[benchmarks/results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization).
Key M5 highlights on Apple M4 24 GiB (commits `ca7e237..f9d8c11`, ReleaseFast,
idle-GPU M5 measurements, `swapouts_delta = 0` across every archived run):

- **`W14` + `G13` optional GSQ + RCO hybrid quantization (`ADR-0017`)**:
  - In `aminfer bench-m5`, Metal 4 `gemv_q2` reaches up to `72.15 GB/s`
    (`46.63–72.15 GB/s` across runs) and 3-bucket `{q2, q3, q4}` scheme-bucketed
    MoE gather cuts active expert weight bytes by `27.7%–27.8%` at `1.00–1.11×`
    the speed of 1-bucket uniform `q4`.
  - On real `Qwen3-0.6B` (`qwen3-0.6b-gsq-rco.amf`, `4.35` eff-bits,
    `[bf16=113, q3=29, q4=168]`): offline RCO reduces quantization MSE by
    `16.82%` (`1.20×` lower) vs min-max (`46.86%` / `1.88×` lower on the `3.25b`
    sweep, where `58.4%` `q2` degrades `0.6B` dense output), reduces file size by
    `72.7%` and wired RAM by `66.2%–69.3%` (`1,251 → 423 MiB` / `1,195 → 367 MiB`),
    and runs at `171.05–177.45 tok/s` (`2.80–2.90×` vs BF16) at `seq=512` and
    `195.80–205.52 tok/s` (`2.90–3.17×`) on EN/ZH chat decode (`0` swapouts).
- **`C6` repo-scale `F_NOCACHE` NVMe `.ang` N-gram table (`aminfer-repo.ang`)**:
  - Packed `/Volumes/Akari/models/amf/aminfer-repo.ang` (`128.02 MiB`, `8,192`
    buckets, `879,949` `2..4`-gram entries, `0` dropped, `512 KiB` wired LRU
    cache). In `bench-m5`, wired LRU hits measure `0.04 us` and async prefetch
    hits `0.13 us` (`1.13–1.54 us` on a freshly written `4.02 MiB` temp file still
    in drive DRAM cache), whereas uncached `F_NOCACHE` reads on the `128.02 MiB`
    external-SSD `aminfer-repo.ang` file average `396.10–1,207.78 us`.
  - All `L1 + C6` runs match `decode` bit-for-bit (`T = 0`). On a repetitive
    numeric continuation (`APPLE_PAGE_SIZE`, GPU-loaded window), `.ang` seed hits
    lifted `tau` `2.818 → 5.167` (`106.33 → 216.28 tok/s` vs `41.48` base); on
    non-repetitive code prompts (`alignUp`, `page.zig`), `396–1,208 us` external
    SSD queries yielded `0.95×`–`1.01×` of in-RAM L1 SAM on `0.6B` (`162.35` vs
    `170.58 tok/s` idle; `56.13` vs `55.82 tok/s` 64-tok) and `3.72` vs `5.98
    tok/s` on `14B` under host GPU load.
- **`D8` off-GPU SME2 + multi-prefix PEARL async pipelined small draft (`ADR-0008`, `ADR-0018`)**:
  - Running `SmallDraftModel` off-GPU on ARMv9 SME2 (`qwen3-0.6b-gsq-rco-sme2.amf`,
    `ldnt1b`, `0` draft Metal CBs) avoids the `+6.0` to `+44.8 ms` `verify_ms`
    contention seen when sharing M4's 10-core GPU (`verify_ms` `247.3 ms` off vs
    `225.2–225.7 ms` on/auto). With tuned tree bounds (`cap=8, d=5, w=1, k=2`,
    64 tokens, 64/64 bit-exact, `0` swapouts), SME2 PEARL lifts throughput from
    `10.67 tok/s` (`--pearl off`) to `11.81 tok/s` (`--pearl on`, `+10.7%`) and
    `12.10 tok/s` (`--pearl auto`, `+13.4%`, vs `10.79 tok/s` `14B` greedy base;
    default `cap=16` 32-token SME2 run measures `10.22 tok/s`).
  - On single-GPU 10-core M4, sequential same-GPU `0.6B-GSQ-RCO` (`--pearl off`,
    `13.55 tok/s`, `1.26×` vs base) remains faster overall than off-GPU SME2
    PEARL (`12.10 tok/s`) because GPU `q4`/`q3` drafting (`15.0 ms`) outruns
    full-model CPU SME2 drafting (`29.4 ms` + `19.4 ms` sync). `--pearl auto`
    disables same-GPU overlap when its median contention probe detects negative
    net gain (`13.06` vs `11.83 tok/s` forced on; `13.33` vs `11.07 tok/s` on
    BF16), and keeps `Qwen3-30B-A3B-3bit + EAGLE-3` `OFF` from startup.
- **`C7` + `C8` live `aminfer serve` HTTP/SSE & concurrency load test**:
  - `bench-m5` synthetic loopback framing benchmark (in-memory token stub, no
    model inference) measures `/health` `72.33–77.33 us` p50, single SSE TTFT
    `106.21–109.08 us` p50, 8-client SSE framing rate `100,277.5–101,245.8 tok/s`
    (`791.21–806.08 us` TTFT p50), and `0` steady-state `malloc` blocks/bytes.
  - Live `aminfer serve` backed by real `Qwen3-0.6B` BF16 (`1,251 MiB` wired)
    passes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
    (`stream=true` `<think>` → `delta.reasoning_content` at `36.74 tok/s`;
    `stream=false` at `36.16 tok/s`), `D11` unseeded TRNG + explicit `"seed"`
    bit-exact SSE replay, `POST /v1/responses`, `POST /v1/messages`, and
    `1/4/8`-client serialized SSE load (`20/20` HTTP `200`, `19.93 / 22.80 /
    23.30 agg tok/s`, `0` 503s, `0` cancels, `serve.swapouts_delta = 0`).

## M6 task list

- [x] `C10` sharded `F_NOCACHE` NVMe PLE store (`src/storage/ple.zig`) with `LoopPleScratch` enforcing `0` NVMe reads on `r >= 1`, `W16` L2-resident SME2 `SmeStepLoraAdapter` (`alpha_0 == 0` bit-exact identity), `W17` `.amf` `reserved0 = 0x0101` `lut16_codebook` auto-dispatch, and `D13` `SmeDraftProjection` + `calibrateSmeVsGpuFn` (`1c75816`)
- [x] Metal 4 hybrid-sparse kernels (`src/kernels/hybrid_ops.metal`, `src/gpu/hybrid.zig`, `src/gpu/moe.zig`): `G8` GDN prefill/decode/`S_tmp` loop-isolation commit (`R6`), `G9` QSA `[11, 11, 10]` MRoPE + YaRN + `512`/`128` `HybridKvRing` + `r >= 1` index reuse, `G10` 4-branch Gated Residual (`GR`), and `G7` `>= 512`-expert `top-10+` MoE with unconditional `shared_expert` accumulation (`be75050`)
- [x] Offline Python M6 packers and generators (`tools/`): `qwen4_exp` vision-tower strip, sharded PLE & MTP packer (`W10`), `synth-qwen4` (`tiny-qwen4` / `scale-qwen4-512e`), `synth-loop` (`tiny-looplm` / `tiny-steplora`) (`W16`), heat-guided large-MoE GSQ + RCO (`qwen3-30b-a3b-gsq-rco.amf`, `2.700b`, `9.599 GiB` wired), and `0x0101` `ZT0` `LUT16` SME2 packing (`qwen3-0.6b-lut16-sme2.amf`, `196` `LUT16` tensors) (`W17`, `5baa49e`)
- [x] `W9` / `W10` / `W16` composable model graph and `LoopSchedule` executor (`src/model/graph.zig`, `src/root.zig`) enforcing all four `ADR-0019` loop isolations (`GDN S_tmp`, `PLE r >= 1`, `QSA` index reuse, `MoE` CSR dedup & per-token `r0_selected` pinned `r >= 1` routing) plus `r = 0` shared KV/GDN state across recurrent passes and aligned `ami_gdn_decode` `out_norm` staging for bit-exact tree verification (`c93a5fa`, `7651cad`, `8a4bc68`, `750b56c`)
- [x] `D12` `SelfSpecRecurrent` (`DraftMode.self_recurrent`) and `D13` `Placement.cpu_sme2` (`sme2`) in `src/spec/{draft,engine}.zig` with `GraphModel` `SpecEngine` integration (`7651cad`, `f6b4705`)
- [x] CLI routing across `verify-amf`, `decode`, `spec-decode`, `bench-decode`, `serve`, and `aminfer bench-m6` (`src/main.zig`, `src/bench/runner.zig`) plus full ReleaseFast synthetic and real-checkpoint hardware verification archived in [benchmarks/results.md](benchmarks/results.md#m6-frontier-hybrid-sparse-models-recurrent-depth-large-moe-gsqrco-and-off-gpu-sme2-draft-heads) (`1af0f8d`, `8a4bc68`)

## M6 frontier hybrid sparse models, recurrent depth and heterogeneous MoE results

Canonical methodology, full tables, trade-off analysis, and unedited verbatim
CLI outputs live in
[benchmarks/results.md](benchmarks/results.md#m6-frontier-hybrid-sparse-models-recurrent-depth-large-moe-gsqrco-and-off-gpu-sme2-draft-heads).
Key M6 measurements on Apple M4 24 GiB (commits `1c75816..f6b4705`, ReleaseFast,
`18,186 MiB` wired budget, `swapouts_delta = 0` across all timed runs):

- **Hybrid-sparse operators (`G8`, `G9`, `G10`, `G7`) & `C10` PLE**:
  - `G8` GDN (`16K/32V x 128`): prefill `seq = 256` at `3,269.5 us` (`78,300.2 tok/s`), decode `M = 1` at `30.1 us` (`140.30 GB/s`, SLC-hot), loop-isolated `S_tmp` eval `76.5 us` + commit `27.2 us` (`stmp_isolated = true`).
  - `G9` QSA (`rot = 64` `[11, 11, 10]` MRoPE + YaRN `16.8 us`): `r >= 1` index reuse achieves `1.55x` speedup on `full512` (`38.7 -> 25.0 us`) and `3.04x` on `swa128` (`77.7 -> 25.6 us`).
  - `G10` 4-branch GR (`H = 2048`): `31.5 us` (`4.56 GB/s`) at `M = 1` vs `23.1 us` (`67.44 GB/s`) at `M = 16`.
  - `G7` 512-expert MoE (`top-10 + shared`, `217 MiB` wired): `239.1 us` (`10` unique experts) at `M = 1` and `850.7 us` (`24` unique experts, `6.67x` CSR deduplication) at `M = 16`.
  - `C10` 4-shard NVMe PLE (`144 KiB` wired): `0.17 us` wired LRU hit vs `1.54 us` p50 / `2.04 us` p99 on fresh SSD file, and `0` NVMe reads across `32` `r >= 1` loop reuses.
  - End-to-end `scale-qwen4-512e` (`98 MiB` wired, `3:1` GDN:QSA, 4-branch GR, `512E+shared`, `32` cold/layer, 4-shard PLE, MTP `Placement.cpu_sme2`): `100%` `T = 0` bit-exact (`ple_r_ge1_nvme_reads = 0`).
- **`D12` `SelfSpecRecurrent` (`43.88 MiB q4 Core` > L2/SLC)**:
  - Serial `M = 1` slows down across `R = 1..4` (`1.449 -> 1.900 -> 2.644 -> 3.322 ms/tok`, `1.00x -> 0.44x`), while batched `verifyTree` reaches raw break-even at `M = 4` (`1.138 ms/node <= 1.449 ms` at `R = 4`, `0.794 ms/node` at `M = 16`) and remains `100%` `T = 0` bit-exact via `r = 0` shared KV. End-to-end `SelfSpecRecurrent` on untrained random weights drops from `tau = 2.50` (`R = 1`) to `tau = 1.00` (`R = 4`), so end-to-end speedup requires trained `LoopLM` weights with higher shallow-to-deep acceptance.
- **`D13` `Placement.cpu_sme2` vs `.gpu`**:
  - `cpu_sme2` wins `11/12` L2-resident (`4-7.5 MiB`) `EAGLE-3` `fc` / `MTP` `eh_proj` shapes by `1.10x-10.81x` (`0` Metal CBs), while `.gpu` wins `mtp_eh_proj_q4_2048x4096` at `M = 4` (`386.1` vs `401.4 us`). On real `qwen3-30b-a3b-eagle3.amf`, `fc` is `24 MiB` BF16 (`> 16 MiB` L2 cap), so `D13` retains `.gpu`.
- **Real `W17` hardware verification**:
  - `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`, `9.599 GiB` wired; `bf16 = 193`, `q3_affine = 3,140`, `q2_affine = 15,534`, `0` `q4_affine`, `0` cold NVMe experts): reduces wired footprint by `-22.9%` (`-2,912 MiB` at `seq = 512`: `9,927` vs `12,839 MiB`) and traffic by `-13.8%` (`1,190.7` vs `1,381.6 MB/tok`), runs `49.59 tok/s` at `seq = 512` (`-4.4%` vs uniform 1-bucket `3-bit` `51.86 tok/s` due to scheme-bucketed `{q2, q3}` MoE dispatch with `ami_moe_route_topk_bucketed` and 2 active buckets) and `48.93-53.06 tok/s` on coherent bilingual EN/ZH decode, and matches `qwen3-30b-a3b-eagle3.amf` `spec-decode` `38/38` bit-exact (`10,968 MiB` wired, `tau = 1.682`, `38.66 tok/s`).
  - `qwen3-0.6b-lut16-sme2.amf` (`0x0101`, `196` `LUT16` tensors, `336.9 MiB`): CPU SME2 decode runs `26.01-26.09 tok/s` (`0` malloc delta, `0` swapouts) vs `26.38-26.62 tok/s` affine `gsq-rco-sme2` (with candid row-variance MSE trade-off `1.325x` on `0.6B` vs `-44%..-62.5%` on bimodal weights), and as an off-GPU `D8` PEARL draft for `Qwen3-14B-4bit` (`8,545 MiB` wired) improves throughput from `11.89 tok/s` (`--pearl off`) to `14.12 tok/s` (`--pearl on`, `tau = 3.917`, `48/48` bit-exact, `1.32x` vs `14B` serial `10.67 tok/s`).

## M6.5 task list

- [x] `G14` zero-copy `VideoToolbox` / `ImageIO` `CVPixelBuffer` → `16 KiB` `IOSurface` → `MTLTexture` (`BGRA8`, bi-planar `NV12`) bridge (`src/bridge/{apple_bridge.h,apple_bridge.m,metal.zig}`), Metal 4 patchify + `2x2` spatial-temporal merge + linear embed kernels (`src/kernels/vision_ingress.metal`), `VisionIngress` wired workspace, and 3D MRoPE builder (`src/gpu/vision.zig`) (`9f1efa7`, `c004372`)
- [x] `C14` iteration-level `BatchScheduler` (`src/runtime/batch_scheduler.zig`), `Server.startBatched` / `runBatchedWorkerLoop` (`src/server/http_sse.zig`), and hot-swappable `DomainNgramRegistry` (`src/spec/ngram_nvme.zig`) with `GET /v1/ngram-packs`, `POST /v1/ngram-packs/select`, and per-request `"ngram_domain"` routing (`1364da5`, `c004372`)
- [x] `R5` 32K+ GPU group-64 `q8` retired KV tier (`RetiredKvQ8Pool`, `ami_kv_retire_tile_to_q8`, `ami_gqa_tiered_decode_attn_split`, `ami_gqa_tiered_tree_verify_attn_split`, QSA tiered `q8` blocks), `R8` wired `PrefixCache` + GPU CoW tail (`ami_kv_tile_cow_copy`) + `GdnSnapshotPool` (`ami_gdn_clone_state`), and `C14` batched slot decode kernels (`ami_batch_qknorm_rope_kv_write`, `ami_batch_gqa_decode_attn_split`, `ami_gdn_batch_decode`) (`bb6dfcf`, `c004372`)
- [x] Real-model continuous batching, `PrefixCache`, `R5` tiered `q8` KV retirement, and `G14` multimodal ingress in `Qwen3Model`/`Qwen3BatchRuntime` (`src/model/qwen3.zig`, `abb552c`) and `GraphModel`/`GraphBatchRuntime` (`src/model/graph.zig`, `f0640ae`)
- [x] `aminfer bench-m65` (`src/bench/runner.zig`, `6a5869a`, `8332ee8`, `990d2a4`), `SpecEngine` `DomainNgramRegistry` routing (`src/spec/engine.zig`), filtered `test-vision`/`test-hybrid`/`test-sme` build targets (`build.zig`, `c4c319f`), and CLI/server integration (`--batch-mode`, `--prefix-cache-chunks`, `--kv-retire-q8-after-chunks`, repeatable `--ngram-pack`, `--image`, `--vt-test-video` in `src/main.zig`, `60dc7ec`, `37ecb74`)
- [x] Offline second domain `.ang` corpus pack (`/Volumes/Akari/models/amf/aminfer-docs.ang`, `32.02 MiB`, `198,113` entries) and ReleaseFast hardware verification (`bench-m65`, live `aminfer serve` serial vs continuous on `Qwen3-0.6B` BF16 and GSQ+RCO, `R8` prefix cache, multi-pack `.ang` HTTP endpoints, real `R5` >1,024-tok decode, and `G14` `qwen4_exp` HW HEVC multimodal decode) archived in [benchmarks/results.md](benchmarks/results.md#m65-continuous-batching-long-context-kv-tiering-shared-prefix-cache-and-zero-copy-multimodal-ingress)

## M6.5 continuous batching, long-context KV tiering and multimodal ingress results

Canonical tables, trade-off analysis, and unedited multi-line verbatim CLI/server
outputs live in
[benchmarks/results.md](benchmarks/results.md#m65-continuous-batching-long-context-kv-tiering-shared-prefix-cache-and-zero-copy-multimodal-ingress).
Key M6.5 measurements on Apple M4 24 GiB (commits `9f1efa7..990d2a4`, ReleaseFast,
`18,186 MiB` wired budget, `swapouts_delta = 0` across all runs):

- **`G14` hardware `VideoToolbox` → zero-copy `IOSurface` → Metal 4 vision + 3D MRoPE**:
  - `bench-m65` (`448x448`, `256` merged visual tokens, `hw_vt = yes`, `zero_copy = yes`, `page16k = yes`): `100.5–137.4 fps` end-to-end (`461.3–883.8 us` HW decode + `6,513.3–9,355.3 us` GPU patchify/merge/embed/MRoPE across H.264/HEVC `NV12`/`BGRA8`).
  - End-to-end `qwen4_exp` (`tiny-qwen4`, `--vt-test-video hevc`): 2 `128x128` NV12 frames (`32` merged visual tokens, `grid = 2x4x4`, `vision_next_mrope_pos = 25`, `vision_encode_us = 2,400.38 us`), decoding at `865.82 tok/s` (`0` swapouts).
- **`R5` 32K+ retired group-64 `q8` KV tier**:
  - `bench-m65` (`32,768` tokens, `Qwen3-30B-A3B` 1-layer geometry, `60` retired `q8` + `4` active `fp16` tiles): reduces wired KV from `64.00` to `35.88 MiB/layer` (`-28.13 MiB`, `-43.95%`), retires at `22.04 us/chunk`, runs decode attention in `1,076.40 us` vs `1,143.17 us` pure `fp16` (`max_abs_err = 0.000006`, `34.95` vs `58.70 GB/s`), and `M = 16` tree verify in `11,421.67 us` vs `13,403.57 us`.
  - Real `Qwen3-0.6B` BF16 at `1,358` prompt tokens (`--kv-retire-q8-after-chunks 1`): lowers total wired RAM from `1,307` to `1,255 MiB` (`active=56.00`, `retired_q8=59.50` vs `168.00 MiB` pure fp16), increases decode throughput from `62.74` to `65.12 tok/s` (`0` malloc delta, `0` swapouts), and matches `25/25` greedy `token_ids` bit-for-bit.
- **`R8` shared `PrefixCache`, GPU CoW partial tail, and `GDN` snapshot clone**:
  - `bench-m65`: `1,024`-tok exact prefix hit (`cow = 0`) speeds prefill `7.62x` (`26.660 -> 3.500 ms`); `1,200`-tok partial prefix hit (`cow = 1`, `cow_tail_176tok = 101.80 us`, `bit_exact = yes`) speeds prefill `31.76x` (`44.669 -> 1.407 ms`); 36-layer `73.69 MiB` `GDN` snapshot clone runs in `2,100.15 us` (`73.58 GB/s`, `bit_exact = yes`).
  - Live `aminfer serve` (`qwen3-0.6b-gsq-rco.amf`, `~728`-tok shared system prompt): warm 4-client concurrent TTFT p50 drops from `566.75 ms` (cold miss) to `63.93 ms` (`8.87x` lower TTFT, `2,849` prefix hit tokens across `4` clients, `297.49 agg tok/s`, `0` swapouts).
- **`C14` continuous batching & `DomainNgramRegistry` multi-pack `.ang` routing**:
  - `bench-m65` scaling (`B = 1 -> 8`): `4.99x` on GQA+MoE `q4` (`1,413.88 -> 7,052.57 agg tok/s`) and `6.74x` on hybrid `GDN + QSA + MoE q4` (`1,297.61 -> 8,752.01 agg tok/s`).
  - Live `aminfer serve` (`C = 1 / 4 / 8` concurrent SSE clients, `D11` explicit `seed = 12345` at `T = 0.7` and `T = 0` greedy bit-exact across serial and continuous, `0` swapouts):
    - `Qwen3-0.6B` BF16 (`24` tok/req): `C = 8` aggregate throughput increases `7.60x` (`40.07 -> 304.35 tok/s`) and TTFT p50/p99 drops from `2,632.84 / 4,460.87 ms` to `131.30 / 196.59 ms` (`20.1x–22.7x` lower), while per-slot TPOT rises from `14.36` to `22.64 ms`.
    - `Qwen3-0.6B` GSQ+RCO (`32` tok/req): `C = 8` aggregate throughput increases `3.94x` (`131.68 -> 518.75 tok/s`) and TTFT p50/p99 drops from `1,035.62 / 1,786.04 ms` to `83.01 / 130.08 ms` (`12.5x–13.7x` lower), while per-slot TPOT rises from `4.95` to `13.72 ms`.
  - `DomainNgramRegistry` with real `aminfer-repo.ang` (`879,949` entries) and `aminfer-docs.ang` (`198,113` entries): `528 KiB` wired cache, `F_NOCACHE` lookup (`verify_table_crc = false`, `F_NOCACHE` key probe bypassing XNU UBC) `257.88 us` p50 / `418.92 us` p99 (external-SSD controller DRAM may still warm from the preceding key probe vs `396–1,208 us` cold NAND flash reads in M5), cache hit `0.05 us`, `selectDefaultDomain` `3.6 ns`, atomic `swapDomainPack` `305.75 us`, `steady_malloc_blocks = 0`; live `GET /v1/ngram-packs`, `POST /v1/ngram-packs/select`, and per-request `"ngram_domain"` (`200` / `400`) verified.

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
| 2026-11-21 | M5 Phase 1 (`ca7e237`, `c697ef1`, `f2efddc`): zero-alloc `kqueue` HTTP/SSE server (`C7`, `C8`), `C6` `.ang` NVMe N-gram table + `D8` PEARL async pipelined small-draft engine, and `ADR-0017` `W14` GSQ+RCO offline exporter + `G13` `q2_affine` & 3-bucket scheme-bucketed MoE Metal 4 kernels | Pass |
| 2026-11-21 | M5 Phase 2, audit remediation & idle-GPU gate (`1f0db2a`, `b78fc8e`, `d5069a2`, `beded1b`, `19b7372`, `f9d8c11`, `552b556`, `137239e`, `a599b6f`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast, `minos 26.0`), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 241/241 tests = 237 lib + 4 CLI, plus 10× consecutive ReleaseSafe & 5× ReleaseFast stress runs = 3,555/3,555 test executions with zero flakes), direct Debug lib test binary under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (237/237, API + GPU Validation enabled, 0 errors), `tools` unittest (64/64), live `aminfer serve` strict `json.loads` validation across `/health`, `/v1/models`, `400`/`404`/`405` errors, and streaming/non-streaming `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, ReleaseFast `aminfer bench-m5` (`gemv_q2` `46.63–72.15 GB/s`, 3-bucket MoE `1.00×`–`1.11×` vs `q4` & `-27.8%` bytes, `C6` cache `0.04 us` / temp-file `1.13–1.54 us`, synthetic SSE framing stub `0` steady-state malloc & `100.3K–101.2K tok/s`, `0` swapouts), real `Qwen3-0.6B` GSQ+RCO `4.35b` (`qwen3-0.6b-gsq-rco.amf`, `verify-amf` OK, `-16.82%` MSE vs MinMax, `-66.2%` to `-69.3%` wired RAM, `171.05–177.45 tok/s` `seq=512` & `195.80–205.52 tok/s` EN/ZH decode), real `aminfer-repo.ang` (`128.02 MiB`, `879,949` entries, `396–1,208 us` external-SSD query, bit-exact), real `Qwen3-14B-4bit + Qwen3-0.6B` (`13.55 tok/s` same-GPU GSQ-RCO `--pearl off`; `11.81–12.10 tok/s` off-GPU SME2 PEARL vs `10.67` off / `10.79` base, 64/64 bit-exact) + `Qwen3-30B-A3B-3bit + EAGLE-3` `--pearl auto → OFF`, and live `aminfer serve` (`20/20` OK, `19.93–23.30 agg tok/s`, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) | Pass |
| 2026-11-22 | M6 Phase 1 (`1c75816`, `be75050`, `5baa49e`): `C10` sharded `F_NOCACHE` NVMe PLE store & `LoopPleScratch`, `W16` L2 SME2 `SmeStepLoraAdapter`, `W17` `0x0101` `LUT16` dispatch, `D13` `SmeDraftProjection`; Metal 4 `G8` GDN, `G9` QSA (`[11,11,10]` MRoPE + YaRN + `512`/`128` `HybridKvRing`), `G10` 4-branch GR, and `G7` 512-expert `top-10 + shared_expert` MoE kernels; Python `qwen4_exp` / `synth-qwen4` / `synth-loop` / large-MoE GSQ+RCO & `LUT16` packers (`tools` unittest: 72/72; built `qwen3-30b-a3b-gsq-rco.amf` `9.599 GiB` & `qwen3-0.6b-lut16-sme2.amf` `336.9 MiB`) | Pass |
| 2026-11-22 | M6 Phase 2 & final gate (`c93a5fa`, `7651cad`, `1af0f8d`, `8a4bc68`): `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 267/267 tests = 262 lib + 5 CLI), direct Debug lib + CLI test binaries under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (267/267 = 262 lib + 5 CLI, API + GPU Validation enabled, 0 errors), `tools` unittest (72/72), Python `synth-qwen4`/`synth-loop` -> Zig `verify-amf`/`decode`/`spec-decode`/`bench-decode` end-to-end parity (100% bit-exact), ReleaseFast `aminfer bench-m6` (`G8` `30.1 us` / `140.30 GB/s`, `G9` `1.55x`/`3.04x` reuse, `G10` `67.44 GB/s`, `G7` `6.67x` CSR dedup, `C10` `0` `r>=1` NVMe reads, `scale-qwen4-512e+MTP` bit-exact, `D12` `R=1..4` bit-exact / `M=4` raw break-even, `D13` `11/12` L2 wins up to `10.81x`, `0` swapouts), and real `W17` hardware verification (`qwen3-30b-a3b-gsq-rco.amf` `9,927 MiB` / `49.59 tok/s` `seq=512` / `38/38` `EAGLE-3` bit-exact; `qwen3-0.6b-lut16-sme2.amf` `26.01-26.09 tok/s` SME2 & `D8` `14B` `14.12 tok/s` PEARL `48/48` bit-exact, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md#m6-frontier-hybrid-sparse-models-recurrent-depth-large-moe-gsqrco-and-off-gpu-sme2-draft-heads) | Pass |
| 2026-11-22 | M6 audit remediation (`750b56c`, `f6b4705`): `r = 0` cross-token GDN state sharing (`ami_gdn_conv1d_q_silu` + `encodeGdnConv1dQSilu`), `ami_gdn_decode` `out_norm` `half`-staging alignment, per-token `r0_selected` cold-expert `r >= 1` masking, unaligned synthetic draft builder write fix, and 4-layer `qwen4_exp` (`R = 3`, `core = [0, 4)`) chunk-invariance + 16-token `self-recurrent` regression test; `zig fmt --check build.zig src`, zero `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug/ReleaseSafe/ReleaseFast: 268/268 = 263 lib + 5 CLI), direct Debug lib + CLI under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (268/268, 0 validation errors), ReleaseFast CLI bit-exact parity on both `synth-loop --preset tiny-steplora --base-arch qwen4_exp --max-loops 3` (`8/8`) and `synth-qwen4 --preset tiny-qwen4 --num-layers 4 --cold-experts-per-layer 2 --max-loops 3` (`24/24` across `decode`, `self-recurrent`, and `mtp`), and ReleaseFast `aminfer bench-m6` re-check (`0` swapouts) | Pass |
| 2026-11-22 | M6.5 Phase 1 (`9f1efa7`, `1364da5`, `bb6dfcf`, `c004372`): `G14` `VideoToolbox`/`ImageIO` zero-copy `IOSurface` bridge & `vision_ingress.metal` (`src/gpu/vision.zig`), `C14` `BatchScheduler` + `Server.startBatched` + hot-swappable `DomainNgramRegistry` (`/v1/ngram-packs*`), `R5` `RetiredKvQ8Pool` + tiered `q8` GQA/QSA kernels, `R8` wired `PrefixCache` + GPU CoW + `GdnSnapshotPool`, and `C14` batched slot decode kernels | Pass |
| 2026-11-22 | M6.5 Phase 2 & audit remediation (`abb552c`, `f0640ae`, `6a5869a`, `60dc7ec`, `8332ee8`, `37ecb74`, `ac5b4e3`, `c4c319f`, `990d2a4`): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug, ReleaseSafe x2 uncached, ReleaseFast: `300/300` = `294` lib + `6` CLI), filtered `test-vision` (`11/11`), `test-hybrid` (`13/13`), and `test-sme` (`10/10`) in Debug and ReleaseSafe, direct Debug `test-vision` (`11/11`), `test-hybrid` (`13/13`), `test-sme` (`10/10`), lib (`294/294`), and CLI (`6/6`) binaries under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (API + GPU Validation enabled, `0` errors), `tools` unittest (`72/72`), built `aminfer-docs.ang` (`32.02 MiB`, `198,113` entries), ReleaseFast `aminfer bench-m65` (with `F_NOCACHE` key probe and `verify_table_crc = false`: `ngram.cold_query_us p50=257.88 / p99=418.92`), live `aminfer serve` serial vs continuous (`C = 1/4/8`, `Qwen3-0.6B` BF16 `7.60x` & GSQ+RCO `3.94x`, `D11` bit-exact), live `R8` `PrefixCache` (`8.87x` TTFT p50), live multi-pack `.ang` HTTP API, real `Qwen3-0.6B` BF16 `R5` >1,024-tok decode (`25/25` bit-exact, `-52 MiB`, `+3.8%` tok/s), and `G14` `qwen4_exp` HW HEVC multimodal decode (`0` swapouts across all runs) archived in [benchmarks/results.md](benchmarks/results.md#m65-continuous-batching-long-context-kv-tiering-shared-prefix-cache-and-zero-copy-multimodal-ingress) | Pass |

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
After M5 close-out on 2026-11-22, the user confirmed
[ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md):
integrating composable recurrent depth (`LoopSchedule` under `W9`, `W16` native
LoopLM & L2-resident SME2 `Step-LoRA` adapters, `D12` `SelfSpecRecurrent` batch
tree verification, and `GDN S_tmp` / `PLE` / `QSA` loop isolations), large-MoE
GSQ+RCO + `QuantParamKind = 1` `ZT0` non-uniform LUTs (`W17`), and off-GPU SME2
draft-head placement (`D13`) into **M6** (validated synthetic-first with drop-in
real checkpoint support), plus scheduling **M6.5** (`R5`, `R8`, `G14`, `C14`)
for iteration-level continuous batching, shared prefix `KvTile`/`GDN` caching,
32K+ GPU `q8` retired KV compression, domain `.ang` packs, and zero-copy
`VideoToolbox → IOSurface → MTLTexture` multimodal ingress.

## Session handoff snapshot

Update this before the context window runs out or when handing off.

- Last updated: 2026-11-22
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
    `e46b7f5` M3 benchmark archive & close-out docs, `33ab2a0` M3 verification note
  - Roadmap docs: `880d487` optional offline GSQ+RCO hybrid quantization
    ([ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md)),
    `7a0005f` `W14`/`G13` status & process registration; `98c43c6` external
    research & open-decision recommendations
    ([architecture/open-decisions-research.md](architecture/open-decisions-research.md))
  - M4: `176740c` `F_NOCACHE` kqueue AIO, E-core pread pool, Spike `S3` & LFU
    expert cache (`C3`–`C5`, `S3`), `cf498b3` offline expert-heat profiler &
    pure-stdlib CoreML `.mlpackage` exporter (`W7`, `W8`), `02d22a0` hybrid
    wavefront prefill, Spike `S6` static-shape buckets & ANE expert predictor
    (`A1`–`A3`, `S6`), `6b6fb09` M4 engine/CLI/benchmark integration (`G6`,
    `D4`, cold-expert NVMe streaming, `bench-m4`, `bench-a2`), `8701fdb`
    two-phase `bench-a2` weight release for zero-swap real-checkpoint eval,
    `bd93b09` M4 benchmark archive & close-out docs
  - M4.5 / confirmed decisions docs: `f91fff9` `ADR-0018` bare-metal SME2
    engine, `D11` TRNG seeding & confirmed open-decision codification, `5e75378`
    M4.5 status/process roadmap sync
  - M4.5: `7de016d` ARMv9.2 `.s` SME2 3-way microkernels, `ZT0-LUT` panels &
    P-cluster dispatcher (`C11`, `C12`), `945cb4d` `W15` `sme2_panel64` `.amf`
    v0 layout tooling & `D11` Secure Enclave TRNG `RequestRng`, `09de8e5` root
    module exports & test deduplication, `bd17c20` single-owner `sme2_panel64`
    loading, L2-resident SME2 router prediction (`C13`) & `ldnt1b` cold-expert
    eval in `Qwen3Model`/`SpecEngine`, `b23f787` `aminfer bench-m45` (`Spike S7`
    + `C13` concurrent bus gate), `probe` SME2 facts, `D11` TRNG CLI &
    Python-Zig numerical parity, `ee4afd8` M4.5 benchmark archive & close-out docs
  - M5: `ca7e237` zero-allocation `kqueue` HTTP/SSE server (`C7`, `C8`),
    `c697ef1` `C6` `F_NOCACHE` NVMe `.ang` N-gram table & `D8` PEARL async
    pipelined small-draft engine, `f2efddc` `ADR-0017` `W14` GSQ+RCO offline
    exporter, `q2_affine` kernels & `G13` 3-bucket scheme-bucketed MoE,
    `1f0db2a` root module exports, `build-ngram` CLI & heterogeneous `D7`
    pruning wiring, `b78fc8e` `aminfer serve`, `C6`/`D8` `spec-decode` flags &
    `aminfer bench-m5`, `d5069a2` unpruned post-L1-merge `hard_cap` tree clamp,
    `beded1b` initial M5 archive, `19b7372` scale-invariant LN peakiness proxy
    & boundary protection in GSQ-RCO, `f9d8c11` off-GPU SME2 draft tree,
    multi-prefix PEARL, `SmeDispatcher` barrier fix & `ADR-0008` contention
    gate, `552b556` refreshed idle-GPU M5 docs & ADR-0007/path fixes,
    `137239e` HTTP JSON closing-brace fixes & strict JSON parser tests,
    `a599b6f` PEARL depth-1 root completion before early-stop & median
    contention probe, `106827b` Round-2 HTTP JSON & PEARL verification record
  - M6: `eb4f308` ADR-0019 roadmap codification, `1c75816` `C10` NVMe PLE store,
    `W16` SME2 `Step-LoRA`, `W17` `0x0101` `LUT16` dispatch & `D13`
    `SmeDraftProjection`, `be75050` Metal 4 `G8` GDN, `G9` QSA, `G10` 4-branch
    GR & `G7` 512-expert shared MoE kernels, `5baa49e` offline `qwen4_exp` /
    `LoopLM` packers, heat-guided large-MoE GSQ+RCO & `LUT16` SME2 tools,
    `c93a5fa` `W9`/`W10`/`W16` composable model graph & `LoopSchedule` executor
    with all 4 `ADR-0019` isolations, `7651cad` `D12` `SelfSpecRecurrent`, `D13`
    `Placement.cpu_sme2`, `r = 0` shared loop KV & `GraphModel` `SpecEngine`
    integration, `1af0f8d` CLI routing (`verify-amf`/`decode`/`spec-decode`/
    `bench-decode`/`serve`) & `aminfer bench-m6`, `8a4bc68` QSA partial MRoPE
    `rotary_dim` derivation & `bench-m6` formatting polish, `5f8acc3` initial M6
    archive & docs, `750b56c` `r = 0` GDN/cold-expert loop sharing, `ami_gdn_decode`
    `out_norm` alignment & 4-layer `qwen4_exp` `R = 3` regression test, `f6b4705`
    unaligned synthetic draft builder write fix
  - M6.5: `9f1efa7` `G14` zero-copy `VideoToolbox`/`IOSurface` vision ingress &
    3D MRoPE, `1364da5` `C14` continuous batching `BatchScheduler`/`Server` &
    hot-swappable `DomainNgramRegistry`, `bb6dfcf` `R5` 32K `q8` retired KV tier,
    `R8` CoW `PrefixCache`/`GdnSnapshotPool` & batched slot kernels, `c004372`
    root/build module exports, `abb552c` `Qwen3Model` `R5`/`R8`/`C14` runtime,
    `f0640ae` `GraphModel` `R5`/`G14`/`C14`/`R8` runtime, `6a5869a` `bench-m65`
    suite, `60dc7ec` CLI/server M6.5 integration & `SpecEngine` domain ngram
    routing, `8332ee8` `bench-m65` GPU signal wait fix, `37ecb74` non-positive
    steady-state malloc check tolerance for async driver frees, `ac5b4e3` M6.5
    benchmark archive & close-out docs, `c4c319f` filtered real module tests
    for `test-vision`/`test-hybrid`/`test-sme`, `990d2a4` `F_NOCACHE` key probe
    and skip pre-timing CRC scan in `bench-m65`
- Current test & build baseline (`M0`–`M6.5` complete, `0` `catch {}`):
  - Zig suite: `300/300` passing (`294` lib + `6` CLI) in `Debug`,
    `ReleaseSafe` (2x uncached), and `ReleaseFast`; filtered `test-vision`
    (`11/11`), `test-hybrid` (`13/13`), and `test-sme` (`10/10`) passing in
    `Debug` and `ReleaseSafe`; direct Debug binaries under
    `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (`test-vision` `11/11`,
    `test-hybrid` `13/13`, `test-sme` `10/10`, lib `294/294`, CLI `6/6`,
    `0` validation errors)
  - Offline Python `tools/` suite: `72/72` passing (`cd tools && uv run python -m unittest discover -s tests`)
- On-disk checkpoints & artifacts (strict rule: keep all `.amf`/`.ang`/`.mlpackage`
  and `TMPDIR` work on `/Volumes/Akari/models/` and `/Volumes/Akari/.tmp/`, never
  on `/` or `/tmp`):
  - Source checkpoints: `/Volumes/Akari/models/{Qwen3-0.6B,Qwen3-14B-4bit,Qwen3-30B-A3B-3bit,Qwen3-30B-A3B-Eagle3}`
  - Verified `.amf` / `.ang` / heat artifacts in `/Volumes/Akari/models/amf/`:
    `qwen3-0.6b-bf16.amf` (`1.11 GiB`), `qwen3-0.6b-gsq-rco.amf` (`311 MB`,
    `4.348b`), `qwen3-0.6b-gsq-rco-sme2.amf` (`337 MB`, off-GPU `sme2_panel64`
    small draft), `qwen3-0.6b-lut16-sme2.amf` (`336.9 MiB`, `196` `0x0101`
    `LUT16` tensors), `qwen3-14b-4bit.amf` (`7.74 GiB`),
    `qwen3-30b-a3b-3bit.amf` (`12.45 GiB`), `qwen3-30b-a3b-gsq-rco.amf`
    (`9.599 GiB` wired, `2.700b` heat-guided GSQ+RCO),
    `qwen3-30b-a3b-eagle3.amf` (`996 MiB`), `qwen3-30b-a3b-heat.json`,
    `aminfer-repo.ang` (`128.02 MiB`, `879,949` repo N-grams), and
    `aminfer-docs.ang` (`32.02 MiB`, `198,113` docs N-grams, `table_crc32=0xc1f2f6ed`)
  - Exported ANE `.mlpackage` bundles: `/Volumes/Akari/models/coreml/`
- Session & `/swarm` operating conventions:
  - Swarm models restricted per user instruction to `gemini-4-carbon-*` and
    `gemini-4-neon-*` only (never `claude-*` or `gpt-*`); commits stay local
    (do not `git push` unless explicitly requested) and run `ikat sync` after
    every commit
- In flight: none (`M0`–`M6.5` completed; next milestone is **M7** covering
  Ultra and cluster scale-out: `DeepSeek-V4.1-Flash-1017`, `R7`, `W11`, `G10`
  mHC, `G11`, `D10`, `C10` Engram, and `X1`–`X2` per
  [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md))
- Awaiting the user: nothing
- Key evidence: [architecture/hardware.md](architecture/hardware.md),
  [architecture/model-roadmap.md](architecture/model-roadmap.md),
  [architecture/open-decisions-research.md](architecture/open-decisions-research.md),
  [benchmarks/results.md](benchmarks/results.md),
  [decisions/0004-kv-cache-iosurface-texture-sharing.md](decisions/0004-kv-cache-iosurface-texture-sharing.md),
  [decisions/0006-amf-cold-experts-direct-io.md](decisions/0006-amf-cold-experts-direct-io.md),
  [decisions/0008-benchmark-gated-draft-placement.md](decisions/0008-benchmark-gated-draft-placement.md),
  [decisions/0009-expert-byte-budget-tree-pruning.md](decisions/0009-expert-byte-budget-tree-pruning.md),
  [decisions/0010-hybrid-wavefront-prefill.md](decisions/0010-hybrid-wavefront-prefill.md),
  [decisions/0011-metal4-zero-allocation-hot-path.md](decisions/0011-metal4-zero-allocation-hot-path.md),
  [decisions/0013-eagle-tree-and-head-formats.md](decisions/0013-eagle-tree-and-head-formats.md),
  [decisions/0015-tiered-frontier-model-roadmap.md](decisions/0015-tiered-frontier-model-roadmap.md),
  [decisions/0016-native-fp16-bf16-and-multishard-import.md](decisions/0016-native-fp16-bf16-and-multishard-import.md),
  [decisions/0017-optional-gsq-rco-hybrid-quantization.md](decisions/0017-optional-gsq-rco-hybrid-quantization.md),
  [decisions/0018-bare-metal-armv9-sme2-engine.md](decisions/0018-bare-metal-armv9-sme2-engine.md),
  [decisions/0019-recurrent-depth-and-post-m5-accelerators.md](decisions/0019-recurrent-depth-and-post-m5-accelerators.md)
