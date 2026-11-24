# Process

The single record of project progress: milestones, spikes, the current task
list, the verification log and the session handoff snapshot. Capability status
lives in [status.md](status.md); decisions live in
[decisions/](decisions/README.md).

## Current focus

**M8.15 (`W8`, `W14`, `W17`, `W24`, `C18`, `C20`; [ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md#m815-gsq-rco-v2-fidelity-hardening-w8-w14-w17-w24-c18-c20))**: **Completed** on M4 ([benchmarks/results.md](benchmarks/results.md#m815-sampling-penalty-parity-gsq-rco-v2-fidelity-hardening-and-router-trace-loop-quality-gate)). **M8.1 (`W21`, `W22`, `W24`, `G23`, `G24`, `C25`, plus `C17`/`C18`/`X1` boundary hardening; [ADR-0024](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md))**: **Completed** on M4 ([benchmarks/results.md](benchmarks/results.md#m81-multi-format-quantization-parity-chattemplate-c18-hardening-and-w24-kl-quality-gate)). **M8.2** (`W23`, `G25`, `D14`, `C26`; [ADR-0024](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md), [ADR-0025](decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md): deep `Inkling 2 Mini & Small` co-design + `Hy4` operators, `DFlare`/`DFly2` block-parallel draft heads, `SME2` `gate_rank=64` controllers, `40/48` `HybridKvRing` SWA, NVMe cold-expert/Engram placement, and end-to-end `Inkling-2-Mini` + `hy4-preview-tiny-random` runs on `24 GB` M4) is **Planned next**, followed sequentially by **M9.1** (`MiMo-V2.6` family: `W25`, `G26`, `D15`), **M9.2** (`GLM-5` family: `W26`, `G27`, `D15`), and **M9.3** (`Ling-3.1-flash-1019` & `C27` dynamic MoE tiering: `W27`, `G28`, `C27`). **M7.8 (`W20`, `G18`–`G22`, `C23`–`C24`; [ADR-0023](decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md))**: **Completed** on M4 ([benchmarks/results.md](benchmarks/results.md#m78-native-kernel-barrier-fusion-mma-flashattention-and-simd-accelerators)); `Apple10+` `NAX` FlashAttention-2 hardware benchmarking remains gated on physical `Apple10+` silicon in Spike `S8`. **M7.5** (`R10`, `W18`, `W19`, `C17`–`C22`; [ADR-0022](decisions/0022-native-c-python-sdk-and-full-protocol-parity.md)) is **Completed** and verified on M4 ([benchmarks/results.md](benchmarks/results.md#m75-native-cpython-sdk-full-httpsse-protocol-parity-rootless-tui-monitor-and-local-ergonomics)). **M7** (`DeepSeek-V4.1-Flash-1017`, `R7`, `W11`, `G10` mHC, `G11`, `D10`, `C10` Engram, `X1`–`X3`; [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0003](decisions/0003-expert-source-tagged-union.md)) engine, kernels, tools, CLI, live HF metadata dry-run validation (`cd72273`), and 2/4-rank loopback TCP cluster runs are verified on M4 ([benchmarks/results.md](benchmarks/results.md#m7-ultra-and-cluster-scale-out-deepseek41-ced-mlacsa2-fp4-mega-mhc-engram-dspark-and-loopback-tcp-cluster)); physical `Apple10+` Spike `S8` acceptance (`M6.9`, `G19` NAX HW), physical multi-Mac TB5 RDMA HCA transport (`X1`–`X3`), and full-scale `552B + 196B` `deepseek41` execution (`W11`) remain gated on hardware.

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
| M6.8 | Wide-chunk GPU prefill, `prompt_tail` pruning, serving QoS & live-RAM / `.metallib` guards | R9, G15, G16, C15 | Default-on `--gpu-prefill-mode wide\|tree64` (`512+` FlashAttention + `BM=64, BN=64` GEMM, direct `KvTile` write) & lossless `prompt_tail` pruning verified bit-consistent vs `tree64`; single-request wide-prefill/`verifyTree` scratch aliasing; default-on `--decode-share`, `--adaptive-prefill-narrow`, and `--message-boundary-cache` `PrefixCache` checkpoints benchmarked under concurrent load; `HOST_VM_INFO64` non-anonymous live RAM soft cache scaling (`+ --strict-live-ram`) & startup `.metallib` numerical canary (`~0.19–0.24 ms` warm in-buffer / `~1.3–3.8 ms` cold first-submit) + `Apple10` `NAX` hook verified without swap | Completed |
| M6.9 | Multi-generation Apple Silicon (`Apple9` + `Apple10+` GPU `NAX`) single-binary adaptation | G17, C16 | `MetalPerformancePrimitives` (`mpp::tensor_ops::matmul2d`) `f16`/`bf16` & staged group-64 `q2`/`q3`/`q4` kernels compiled into unified `aminfer.metallib` with lazy `MTL4Compiler` `Apple10`/`Apple11` selection (`--prefill-matmul-backend auto\|simdgroup\|nax`), `G16` canary auto-fallback to `simdgroup_apple9` (`+ --strict-nax`), chip-parameterized `spec_bus_gbs` (`--bus-bandwidth-gbs`), and runtime prefill/draft crossover calibration verified with zero M4 regression (`Apple10+` HW throughput gated on Spike `S8`) | In progress (code complete; awaiting Apple10+ HW acceptance & Spike S8 bench) |
| M7 | Ultra & cluster scale-out (`DeepSeek-V4.1-Flash-1017`) | R7, W11, G10 (mHC), G11, D10, C10 (Engram), X1–X3 | Causal Encoder-Decoder scheduler, MLA/CSA2 + FP4 KV (`890 B/tok`), Single-Pass Mega-mHC, 196B Engram lookup, DSpark drafting and Thunderbolt 5 RDMA / TB4 TCP expert parallelism verified on M4 loopback cluster (`b2034c2..69d30af`); physical multi-Mac TB5 RDMA HCA transport and full 552B + 196B execution gated on 512 GB Ultra or multi-Mac TB5 RDMA hardware | In progress (code and M4 loopback cluster verified; 512 GB Ultra / TB5 RDMA HW pending) |
| M7.5 | Native C/Python SDK, full HTTP/SSE protocol parity, rootless TUI monitor & local ergonomics | R10, W18, W19, C17–C22 | `libaminfer.dylib` + `include/aminfer.h` + zero-dependency Python `ctypes` SDK (`python/aminfer/`); full `/v1/chat/completions`, `/v1/responses`, `/v1/messages` (+ `count_tokens`, `--api-key`, `--unix-socket`); rootless `aminfer monitor` TUI (`trans-pride` + `--theme`); single-file `.amf` metadata (`ls`/`inspect`); scrolling `aminfer chat` REPL + `stdin` pipe; `/v1/embeddings` & `/v1/rerank`; zero-alloc `json_schema`/GBNF bitmask engine; runtime multi-LoRA routing; opt-in `--disk-prefix-cache` NVMe persistence verified with zero hot-path alloc and zero swap | Completed |
| M7.8 | Native Metal 4 sublayer barrier fusion, MMA FlashAttention-2, GPU indirect MoE dispatch, full ViT vision encoder tower & NEON/SME2 accelerators | W20, G18–G22, C23–C24 | Fused `o_proj`/`down_proj` `+= residual` epilogues, `ami_fused_residual_rmsnorm`, and homogeneous-`DType` `ami_fused_qkv` + `ami_dense_gate_up_swiglu` (`0` stack spill); `simdgroup_half8x8` (`Apple9`) + `NAX` (`Apple10+`) FlashAttention-2 prefill (`Q_len >= 32`, `KV_len >= 64` gated); `MTLDispatchThreadgroupsIndirectArguments` MoE dispatch (`M > 1` & multi-scheme `G13`); fused `248K`-vocab GPU radix constrained sampler + 16/64-byte NEON JSON/BPE scanning; ARMv9.2 SME2 `luti2` (`2-bit`) `.s` microkernels (`0` `[sp` spills); full multi-layer bidirectional 2D-RoPE `ViT` tower + 2-layer `PatchMerger`/`Aligner` MLP + AnyRes tiling (`W20`, `G22`) benchmarked in `aminfer bench-m78` with zero M4 decode/prefill regression and zero swap | Completed (M4 verified; Apple10+ NAX HW pending Spike S8) |
| M8.1 | Zero-loss multi-format quantization substrate (`FP8`/`MXFP8`/`NVFP4`/`GGUF`/`MLX`), tokenizers, declarative `ChatTemplate`, `C17`/`C18`/`W24`/`X1` boundary hardening, and `KL` quality gate | W21, W22, W24, G23, G24, C25 (plus C17/C18/X1 hardening) | Native `.amf` dtypes & register-dequant Metal/`SME2` kernels (`fp8_e4m3_block`/`_row`/`_tensor`, `mxfp8_g32`, `nvfp4_g16`, MLX `5/6/8-bit` `g32`/`g128`, GGUF `Q*`/`K`/`IQ*`/`TQ*`/`MXFP4`/`Q2_0`, `M = 1` bus-% gated vs `q4_affine`); bounded-disk stream packing & lossless pure-stdlib `GGUF` -> `.amf` repacker rejecting colliding fork type IDs (`STQ1_0` vs `F8_E4M3`); opt-in `--act-quant emulate`; `o200k`/`ignore_merges` + `Hy4` tokenizers; offline-compiled declarative `.amf` JSON `chat_template_spec` + zero-alloc Zig state-machine `ChatTemplate` replacing hardcoded ChatML; full `reasoning_effort` mapping + 12-item `http_sse.zig` reasoning/stateful/keep-alive/`TTFT` hardening + C/Python SDK sampling/token bounds (`engine.py`, `c_api.zig`) & defensive `amf.zig`/`transport.zig` guards; `W24` `KL`/top-1/code-math quality gate | Completed |
| M8.15 | Sampling/penalty parity (`aminfer chat` & `BatchServeRunner`), `GSQ-RCO v2` fidelity hardening, single-frame GPU `--dump-router-trace`, and `W24` repetition-loop quality gate | W8, W14, W17, W24, C18, C20 | `aminfer chat` (`C20`) `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty` (`1.05` at `T > 0`, `1.0` bit-exact greedy at `T = 0`), `--presence-penalty`, `--frequency-penalty` via per-turn `DefaultPenaltyTracker`; `BatchServeRunner` (`C18`) per-slot `slot_penalties` wired into GPU `sampleSync` & CPU fallback constrained sampling; `aminfer decode --dump-router-trace` zero-copy single-frame GPU router top-k JSONL export + `expert-heat` shallow-fallback stderr warning (`W8`); `GSQ-RCO v2` (`W14`, `W17`: activation-weighted OLS RCO using `--imatrix` per-channel $E[x_c^2]$, hard non-`q2_affine` backbone role floor, `--max-q2-share`, auto `4.15` Dense / `3.50` MoE `target_bits`, CLI low-bit `q2_affine` double-quantization guard with `--allow-retranscode-lowbit`, and `--fine-rungs` `g32` + `Q5`/`Q6`/`Q8_AFFINE` rungs capped at `<= 5` schemes/layer); `aminfer quality-gate` (`W24`) 3rd conceptual reasoning prompt (`Riemann zeta`) + 4-gram/consecutive repetition loop detector `fail_repetition_loop` | Completed |
| M8.2 | Deep `Inkling 2` (`Mini` & `Small`) co-design, `Hy4` family operators, `DFlare`/`DFly2` block-parallel draft heads, and `Inkling-2-Mini` + `hy4-preview-tiny-random` on `24 GB` M4 | W23, G25, D14, C26 | SIMD register-fused `Inkling 2` sublayer kernels (`RMSNorm + k=4 sconv`, single-pass `q/rel_l2_clip + log_scaling + d_rel=16 rel_attn + attn_sink_proj + gate_attn + softcap`, MoE `bias + route_scale + softcaps + 6-layer attn_res`, `0` stack spill), L2 `SME2` `sme2_panel64` `gate_rank=64` loop controllers, `40/48` `1024`-token `HybridKvRing` SWA (`128K–256K` M4 context), `W20` `23L` ViT sidecar split + `Small` `GSQ-RCO` profiles; `Hy4` (`IndexCache` period-4, `IndexPool`, `LatentMoE`, sigmoid + correction-bias routing, `linear_gate`, `learnable_sink_param`, `swiglu_limit=10`, `sandwich_norm`, per-row `F8_E4M3` Engram NVMe streaming, `40L`/`27L` AnyRes ViT); `3L`/`5L` `DFlare` (`block_size=8`, `M=8` batch `Core×3` verify) & `DFly2` (`block_size=7`) (`D14`); end-to-end `Inkling-2-Mini` (`<= 17.76 GiB` wired, `0` swap) & `hy4-preview-tiny-random` verified on `24 GB` M4 (`Small`/`Hy4`/`Hy4-flash` full runs storage/HW-gated) | Planned |
| M9.1 | `MiMo-V2.6` family deep optimization (`Flash` `309B`, `Pro` `1.02T`, `Distill-Qwen-9B` `9.4B`) | W25, G26, D15 | `mimo_v2` & dense `qwen3_5` `.amf` converters + `GSQ-RCO` tier-down profiles (`Flash` `128 GB` Max, `Pro` single `512 GB` Ultra); asymmetric SWA (`swa_head_dim=192`, `swa_v_head_dim=128`, `128+sink` `HybridKvRing` across `40/48` & `60/70` layers); chained `3L` native `MTP` + `DFlash` `8–16`-token block-diffusion drafting + `D7` cold-expert pruning (`Distill-9B` & synthetic `mimo_v2` verified locally on `24 GB` M4 with `0` swap; full `Flash`/`Pro` storage/HW-gated) | Planned |
| M9.2 | `GLM-5` family deep optimization (`GLM-5.2` & `5.3` `753B`, `GLM-5.3-Flash` `320B`) | W26, G27, D15 | `glm_moe_dsa` & multimodal `glm5_next` `.amf` converters (`W20` `24L` ViT sidecar split + `GSQ-RCO` single-`512 GB` Ultra / `128 GB` Max profiles); `index_topk_freq=4` + `index_share_for_mtp_iteration=true` zero-reindex `MLA`/`DSA`, single-kernel register-fused 20-step `hc_mult=4` `mHC` Sinkhorn, `34`-layer `fused_qkvbfg_a_proj` linear attention + `11`-layer `NoPE MLA` (`qk_rope=0`, `index_kpool=4`), and `1L` MTP / `DFlash2` verified on synthetic & live metadata fixtures (`0` swap; full `320B`/`753B` storage/HW-gated) | Planned |
| M9.3 | `Ling-3.1-flash-1019` (`572B`) deep optimization & cross-family dynamic MoE tiering | W27, G28, C27 | `bailing_hybrid` `.amf` converter + `RCO-Q3` dense-`KDA` compression (`24.51B` always-active shrunk to `~9.2 GiB`, `256 GB` Studio full-wired `GSQ-RCO`); `63`-layer `KDA` (`kda_lower_bound=-5.0`, `kda_safe_gate`, `k=4` short conv) + `9`-layer Gated `MLA` (`g_proj`) + L2 `SME2` `eh_proj` `1L` MTP; lossless session-local dynamic hot-expert promotion in `WiredArena` + opt-in `--cold-expert-min-weight` (`0.0` default lossless, `W24` quality-gated) verified on `24 GB` M4 (`0` swap) | Planned |

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
| S8 | `Apple10+` (M5+) `NAX` `mpp::tensor_ops::matmul2d` (`f16`/`bf16`, staged group-64 `q2`/`q3`/`q4_affine`, `FP4` `e2m1`) TFLOP/s, numerical parity, `G16` canary margins vs offline `metalfe`, and GPU-only vs ANE wavefront prefill crossover on physical `Apple10+` silicon | [ADR-0021](decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md), G17, C16 | M6.9 | Open (requires Apple10+ host) |

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

## M6.8 task list

- [x] `R9` live `HOST_VM_INFO64` non-anonymous RAM snapshot (`src/bench/vmstat.zig`), soft wired-cache auto-sizing + opt-in `--strict-live-ram` admission (`src/mem/budget.zig`), `WiredArena.allocAliasedPair` single-request wide-prefill / `verifyTree` scratch aliasing (`src/mem/arena.zig`), and `C15` continuous-batching `decode-share` wall-time QoS ledger, depth-scaled chunk narrowing (`512 -> 256 -> 128`), evict-first `PrefixCache` message-boundary checkpoints, and zero-alloc ChatML boundary extraction (`src/runtime/{batch_scheduler,prefix_cache}.zig`, `src/server/http_sse.zig`) (`ec1a3b7`)
- [x] `G15` wide-chunk (`512–1024`) `BM=64, BN=64` GEMM (`src/kernels/dequant_gemm.metal`, `src/gpu/dequant.zig`), direct bindless `KvTile` QK-Norm/RoPE/KV-write + causal FlashAttention (`src/kernels/attention.metal`, `src/gpu/attention.zig`), wide hybrid `GDN` / `QSA` / `128`-token circular SWA (`src/kernels/hybrid_ops.metal`, `src/gpu/hybrid.zig`), and `G16` startup `.metallib` numerical canary (`f16`/`bf16`/`q4`) + `Apple10` (`NAX`) capability probe & dispatch hook reservation (`src/bridge/{apple_bridge.h,apple_bridge.m,metal.zig}`) (`356f654`)
- [x] Real-model `Qwen3Model`/`Qwen3BatchRuntime`, `GraphModel`/`GraphBatchRuntime`, and `SpecEngine` integration (`src/root.zig`, `src/model/{qwen3,graph}.zig`, `src/spec/engine.zig`): default-on `--gpu-prefill-mode wide|tree64`, lossless `prompt_tail` final-layer compute pruning, single-request scratch aliasing (keeping `lm_head` 100% wired), startup canary gating, live-RAM soft cache sizing, and ChatML message-boundary `PrefixCache` checkpoints (`f70becf`)
- [x] CLI, `aminfer probe`, `aminfer serve`, and `aminfer bench-m68` integration (`src/main.zig`, `src/bench/runner.zig`), plus Metal 4 `@autoreleasepool` destroy draining (`src/bridge/apple_bridge.m`), explicit `WiredRegion` `munlock` (`src/mem/page.zig`), and audit remediation hardening (`R9` live-RAM `bench-m68` gating + `cold_rt` wide prefill, `ADR-0019` QSA/PLE `R = 3` wide-prefill isolations, cold + warm in-buffer canary profiling, `serve` default-slot IOSurface auto-clamping & `continuous_batch_only` minimal base KV tiles) (`ad3ee7f`, `2063bc9`, `2b9312d`)
- [x] ReleaseFast hardware verification (`aminfer probe`, 3/3 consecutive `aminfer bench-m68` zero-swap runs, standalone `aminfer bench-decode` `tree64` vs `wide` A/B on real `Qwen3-0.6B` GSQ+RCO, `Qwen3-14B-4bit`, and `Qwen3-30B-A3B` GSQ+RCO, and live `aminfer serve` with default `--max-active-requests` at `--max-seq-len 10240 --prefix-cache-chunks 16`) archived in [benchmarks/results.md](benchmarks/results.md#m68-wide-chunk-gpu-prefill-prompt_tail-pruning-serving-qos-live-ram-admission-and-metallib-canary)

## M6.8 wide-chunk GPU prefill, prompt_tail pruning, serving QoS and live-RAM results

Canonical tables, honest trade-off notes, and unedited multi-line verbatim CLI
and server outputs live in
[benchmarks/results.md](benchmarks/results.md#m68-wide-chunk-gpu-prefill-prompt_tail-pruning-serving-qos-live-ram-admission-and-metallib-canary).
Key M6.8 measurements on Apple M4 24 GiB (commits `ec1a3b7..2b9312d`,
ReleaseFast, `18,186 MiB` static wired budget, `swapouts_delta = 0` across all
runs including 3/3 consecutive `bench-m68` executions):

- **`G16` startup `.metallib` numerical canary & `Apple10` (`NAX`) probe**:
  - `aminfer probe` and `bench-m68` report `gpu.supports_apple10 = no` and `gpu.prefill_matmul_backend = simdgroup_apple9` on M4 (`Apple9`, `Apple10` NAX hook reserved cleanly).
  - Startup canary completes in `187.0–237.3 us` (`~0.19–0.24 ms`) warm in-buffer over the resident `16 KiB` scratch page (`2,132.5–2,223.5 us` cold first-submit including initial Metal 4 queue/residency/AGX wake; `1,325.2–3,780.0 us` cold in `bench-decode`) with `max_abs_err`: `f16 = 0.000023`, `bf16 = 0.001953`, `q4 = 0.000028`; corrupted-slice negative check passes (`corrupted_slice_rejected = yes`).
- **`G15` wide-chunk GPU prefill + `prompt_tail` pruning & `R9` scratch aliasing**:
  - Single-load `bench-m68` A/B (`tree64` vs `wide` + pruning, `100%` `greedy = yes` across all 7 rows): `qwen3-0.6b-bf16` `1.91x` (`512`, `2,405.1 tok/s`) / `1.72x` (`2,048`, `1,874.2 tok/s`); `qwen3-0.6b-gsq-rco` `2.69x` (`512`, `2,475.2 tok/s`) / `2.33x` (`2,048`, `1,920.2 tok/s`); `qwen3-14b-4bit` `2.29x` (`512`, `6,331.50 ms`, `80.9 tok/s`, `alias = 78.02 MiB`); `qwen3-30b-a3b-gsq-rco` `3.91x` under `R9` cold-expert streaming (`512`, `4,603.94 ms`, `8,809 MiB` wired, `alias = 53.34 MiB`) and `1.14x–1.28x` full-hot (`9,979 MiB`); synthetic `qwen4_exp-512e-R2` `2.16x` (`512`, `15,116.2 tok/s`, `alias = 2.06 MiB`).
  - Real `EAGLE-3` wide prefill + aliased `verifyTree` transition (`qwen3-30b-a3b-gsq-rco`, `prompt = 480`, `gen = 16`, `R9` live-RAM gated): dynamically sized `cold_exp = 66 / 41 / 96` (`wired_mib = 6,770 / 8,387 / 4,702`) across 3/3 consecutive runs with `alias_mib = 19.63`, `greedy_match = yes`, `malloc_blk = 0`, and `infer_swapouts_delta = total_swapouts_delta = 0`.
  - Fresh-process standalone `bench-decode` (`512` tokens, `100%` hot weights, `tree64` vs `wide`, `swapouts_delta = 0`): `qwen3-0.6b-gsq-rco` `541.84 -> 222.49 ms` (`2.44x`, `2,301.26 tok/s`, `-38 MiB` wired); `qwen3-14b-4bit` `12,537.84 -> 5,604.32 ms` (`2.24x`, `91.36 tok/s`, `-37 MiB` wired); `qwen3-30b-a3b-gsq-rco` `3,469.70 -> 2,729.46 ms` (`1.27x`, `187.58 tok/s`, `-39 MiB` wired), while keeping `lm_head` 100% wired.
- **`C15` serving QoS, adaptive prefill narrowing & ChatML message-boundary `PrefixCache`**:
  - `bench-m68` QoS (`qwen3-0.6b-gsq-rco`, `10,240`-tok background prefill + periodic 2-slot foreground bursts): `decode_tpot_p50` drops `103.7x` (`562.94 -> 5.43 ms`) and deep-context `p99` across `576` samples drops `1.47x` (`987.28 -> 673.20 ms`, `qos_steps = d:270/p:6`, `narrowed_chunks = 3` at `>= 8,192` tokens) at `+16.3%` background TTFT (`12,432.5 -> 14,455.8 ms`).
  - `bench-m68` ChatML boundary cache (`boundary_pos = 359` below first `512`-tok tile): `0/383 -> 359/383` prefix hits, TTFT `183.05 -> 28.74 ms` (`6.37x` faster, `greedy_match = yes`), `evict_first = 1`, `canonical_512_survived = yes`.
  - Live `aminfer serve` HTTP/SSE (`qwen3-0.6b-gsq-rco.amf`, default `--max-active-requests` omitted, `--max-seq-len 10240 --prefix-cache-chunks 16 --decode-share 0.50 --adaptive-prefill-narrow on --message-boundary-cache on`): auto-clamps `max_batch_size = 2` (`3,534 MiB` wired RAM, `max_active_requests = 16`, `0` `TileCreationFailed`); multi-turn ChatML Turn 1 -> Turn 2 TTFT drops `337.05 -> 30.48 ms` (`11.06x`, `boundary_checkpoints = 1`, `boundary_hits = 1`, `evict_first_evictions = 1`); concurrent `~9.4K` background prefill + 8 foreground streams achieve `fg_tpot_p50 = 4.83 ms`, `fg_tpot_p99 = 413.62 ms`, `qos_steps = decode=210 prefill=4 narrowed_chunks=1`, `swapouts_delta = 0`.
- **`R9` live non-anonymous RAM admission & soft wired-cache auto-sizing**:
  - `probe` / `bench-m68` live reclaimable RAM: `10,053 MiB` / `9,624 MiB` (`free + speculative + purgeable + external`, excluding `inactive`, `internal`, `compressor`, `wired`); soft cache auto-sizing scales `64 -> 16` chunks under tight headroom (`cache_scaled_down = yes`), `--strict-live-ram` refuses oversubscribed admissions (`strict_live_ram_rejected = yes`), and `infer_swapouts_delta = total_swapouts_delta = 0` across 3/3 consecutive `bench-m68` runs.

## M6.9 task list

- [x] `C16` multi-generation Apple Silicon chip profile (`src/platform/sysinfo.zig` `cpuBrand`/`hwModel`, `src/platform/chip_profile.zig` `ChipProfile.resolve`/`spec_bus_gbs`/`selectPrefillRoute`/`passesC13CoexecGate`), `spec_bus_gbs` parameterization in `src/gpu/attention.zig`, `wavefront.shouldUseWavefront` in `src/ane/wavefront.zig`, and `SmeCaps.draftL2BudgetBytes()` / `D5` placement gating in `src/cpu/sme.zig` and `src/spec/draft.zig` (`3bf7cc4`)
- [x] `G17` + `G16` unified-`.metallib` Metal 4 `NAX` TensorOps (`src/kernels/nax_tensorops.metal`: `ami_dense_gemm_nax`, `ami_dequant_gemm_nax`, `ami_fp4_e2m1_gemm_nax` compiled via `build.zig` under `-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`), `Apple7..11` GPU family probe & `Library.hasFunction` (`src/bridge/{apple_bridge.h,apple_bridge.m,metal.zig}`), and lazy `MTL4Compiler` `*_nax` PSOs + canary-guarded `nax -> simdgroup_apple9` atomic demotion & `--strict-nax` enforcement (`src/gpu/dequant.zig`) (`76e08c7`)
- [x] Engine root and real-model integration (`src/root.zig`, `src/model/{qwen3,graph}.zig`): `chip_profile` export & tests, `LoadOptions` (`prefill_backend_mode`, `strict_nax`, `bus_bandwidth_gbs_override`), resolved `ChipProfile` stored on `Qwen3Model`/`GraphModel`, `DequantPipelines.initWithOptions` + `runStartupNumericalCanaryInBufferMut` startup canary demotion wiring, and `wavefront.shouldUseWavefront` routing with zero M4 behavior change (`4fcfc0c`)
- [x] CLI, `aminfer probe`, `spec_bus_gbs` benchmark telemetry, and `aminfer bench-m69` harness (`src/main.zig`, `src/bench/runner.zig`): `--prefill-matmul-backend auto|simdgroup|nax`, `--strict-nax`, and `--bus-bandwidth-gbs <f64>` across `decode`, `spec-decode`, `bench-decode`, `serve`, `probe`, and `bench-m69`; enhanced `probe` chip/GPU/metallib/NAX reporting; `runM69WithConfig` Section 1 gates (`.metallib` symbols, `ChipProfile`, fallback & simulated canary demotion) and Section 2 `Spike S8` hardware parity/TFLOP/s sweep (honestly reporting `spike_s8.nax_hw_status=skipped_requires_apple10` on `Apple9` M4) (`320791e`)
- [x] M4 (`Apple9`) build, cross-generation `-Dcpu=apple_m1`, `Debug`/`ReleaseSafe`/`ReleaseFast` full & filtered unit tests, Metal API/Shader validation (`0` errors), CLI `probe`/`--help` smoke checks, offline `tools/` tests, and markdown verification (`340/340` Zig tests, `72/72` Python tests)
- [ ] Physical `Apple10+` (M5+) hardware acceptance and `Spike S8` benchmark run (`aminfer bench-m69` TFLOP/s, `fp32` parity, `G16` canary margins vs offline `metalfe`, and GPU-vs-ANE wavefront prefill crossover) archived in [benchmarks/results.md](benchmarks/results.md)

## M7 task list

- [x] `X1`–`X3` Thunderbolt-peer cluster transport and runtime (`src/cluster/transport.zig`, `src/cluster/expert_parallel.zig`, `src/mem/expert.zig`): runtime `probeTb5Rdma()` (`auto -> tb4_tcp` on M4 TB4, strict `--strict-tb5-rdma` rejection), 16 KiB wired-slab `kqueue` `FrameChannel` (`AMI7` CRC-32), `ExpertPartitionPlan` (`contiguous` / `round-robin`), `PeerWorkerServer`, and overlapped `ExpertParallelCoordinator` (`b2034c2`)
- [x] `C10` 196B Engram conditional memory store, `D10` `DSparkProposer`, and offline `deepseek41` toolchain (`src/storage/ple.zig`, `src/spec/dspark.zig`, `tools/src/aminfer_tools/deepseek41.py`, `tools/src/aminfer_tools/cli.py`): multi-shard `F_NOCACHE` Engram lookup with `r = 0` single-lookup loop reuse, semi-autoregressive `DSpark` block proposer, and `synth-deepseek41` / `pack-deepseek41` / `partition-experts` CLI with ADR-0015 Tier 3 budget guard (`2abc0fe`, `69d30af`)
- [x] `R7` / `G11` MLA + CSA2 (`Full` / `Reindex` / `Reuse`) + FP4 (`E2M1` + `E4M3` scale, `890 B/tok`) KV + SWA bounded replay and `G10` Single-Pass Mega-mHC (`mhc_single_pass` / `mhc_post_mix`, Sinkhorn-Knopp Birkhoff) Metal 4 kernels and host arenas (`src/kernels/hybrid_ops.metal`, `src/gpu/hybrid.zig`) (`b17bb9d`, `69d30af`)
- [x] `W11` / `D10` / `X2` / `X3` engine, speculative, and root/build integration (`src/root.zig`, `build.zig`, `src/model/graph.zig`, `src/spec/{draft,engine}.zig`): `deepseek41` `BackboneKind.causal_enc_dec` two-phase scheduler (`0` encoder dispatches during decode), `AttentionKind.mla_csa2`, `ResidualKind.mhc_single_pass`, `MemoryLookupKind.engram`, `GraphBatchRuntime` (`C14`), `DraftMode.dspark`, `ExpertParallelCoordinator` MoE wiring, and `test-cluster` build step (`7edf3c3`, `0a84cef`)
- [x] M7 CLI and benchmark harness (`src/main.zig`, `src/bench/runner.zig`): `aminfer peer-worker`, `--peer` / `--rank` / `--world-size` / `--partition` / `--cluster-transport` / `--strict-tb5-rdma` / `--shutdown-peers` and `--draft-mode dspark` across `decode`, `spec-decode`, and `bench-decode`, plus `aminfer bench-m7` covering `R7`/`G11`, `G10` mHC, `C10` Engram, `W11`/`D10` synthetic `deepseek41` CED + `DSpark`, and `X1`–`X3` 2-rank and 4-rank loopback TCP cluster (`b9f8b9d`, `69d30af`)
- [x] M4 (`Apple9`, 24 GiB, TB4) verification gate, ReleaseFast `aminfer bench-m7`, persistent compact synthetic `deepseek41` bundle (`/Volumes/Akari/models/amf/synth-deepseek41/`, `7.0 MiB`), and live 2-OS-process `peer-worker` + `decode --peer` + `spec-decode --draft-mode dspark` parity check archived in [benchmarks/results.md](benchmarks/results.md#m7-ultra-and-cluster-scale-out-deepseek41-ced-mlacsa2-fp4-mega-mhc-engram-dspark-and-loopback-tcp-cluster)
- [ ] Physical multi-Mac Thunderbolt 5 RDMA HCA transport acceptance (`X1`–`X3`) and full-scale `552B + 196B` `DeepSeek-V4.1-Flash-1017` (`W11`) execution on a 512 GB Apple Silicon Ultra node or multi-Mac TB5 RDMA cluster

## M7.8 task list

- [x] `C23` & `C24` explicit 16/64-byte ARM64 NEON JSON/HTTP + BPE pre-tokenizer boundary scanners (`http_sse.zig`, `tokenizer.zig`) and bare-metal ARMv9.2 SME2 `.s` `luti2` (`2-bit` `q2_affine` / `LUT4`) microkernels (`0` streaming `[sp` spills across `35` `_ami_sme2_*` symbols) + `<= 4 MiB` L2 / `C13`-gated hoisted multi-LoRA dispatch (`9f13f37`)
- [x] `G22` & `W20` Metal 4 multi-layer bidirectional 2D-RoPE `ViT` tower + 2-layer `PatchMerger`/`Aligner` + AnyRes tiling (`src/gpu/vision.zig`) and Python `pack-vision` / `export-coreml --component vision-tower` offline toolchain (`32bfe1a`)
- [x] `G18` & `G19` Metal 4 sublayer/residual-epilogue fusion (`has_residual_epilogue`, `ami_fused_residual_rmsnorm`, homogeneous-`DType` `ami_fused_qkv_*` & `ami_dense_gate_up_swiglu`, combined single-frame `G16`+`G18` startup canary) and `Apple9` `simdgroup_half8x8` + `NAX` MMA FlashAttention-2 prefill (`G19`, `Q_len >= 32 && KV_len >= 64`) (`d56fff0`)
- [x] `G20` & `G21` GPU indirect MoE threadgroup dispatch (`MTLDispatchThreadgroupsIndirectArguments`) and 248K-vocab GPU radix/bucket constrained sampler with unconstrained greedy `argmax` bypass (`src/gpu/sampler.zig`) (`a9846a2`)
- [x] M7.8 Phase 2 integration (`root.zig`, `Qwen3Model`, `GraphModel` aliased ViT scratch via `WiredArena.allocAliasedPair` + `initTextOnly`, `lora.zig`, `C22` speculative tree bitmask propagation in `draft.zig`/`accept.zig`/`dspark.zig`/`engine.zig`, `c_api.zig`), M4 zero-regression `G18` shape calibration (`qkv_min_m=64`, `swiglu_min_m=32`, residual `M==1 || M>=64`), and ReleaseFast `aminfer bench-m78` archived in [benchmarks/results.md](benchmarks/results.md#m78-native-kernel-barrier-fusion-mma-flashattention-and-simd-accelerators) (`0de204b`, `5345566`)
- [ ] Physical `Apple10+` `NAX` FlashAttention-2 hardware throughput bench (gated on `Apple10+` silicon in Spike `S8`)

## M8.1 task list

- [x] `W21` native `.amf` dtypes (`fp8_e4m3_block`/`_row`/`_tensor`, `mxfp8_g32`, `nvfp4_g16`, MLX `q5`/`q6`/`q8` + `g32`/`g128`), bounded-disk / HTTP Range stream packing, and imatrix-aware GSQ+RCO source protection (`de23a63`, `e459c23`)
- [x] `W22` pure-stdlib lossless offline `GGUF` -> `.amf` streaming repacker (`tools/src/aminfer_tools/gguf.py`) with offset-derived byte-length fork guard (rejecting colliding `STQ1_0` vs `F8_E4M3 = 43`, `VQ-GGUF`, `EXL3`), `mmproj` -> `W20`, and `arch=dflash` draft extraction (`de23a63`, `e459c23`)
- [x] `G23`, `G24` & `C25` bit-exact Metal 4 register-dequant GEMV/MoE + tiled `fp16` prefill kernels, opt-in `--act-quant emulate`, and ARMv9.2 SME2 `.s` `E2M1`/`Q2_0`/`TQ`/`IQ2` microkernels (`0` `[sp` spills, `M = 1` bus-% gated) (`ce828bd`, `c578d8a`, `e459c23`)
- [x] `W24` `o200k`/`ignore_merges` + `Hy4` `<｜hy_*｜>` tokenizers, offline-compiled declarative `.amf` JSON `chat_template_spec` + deterministic zero-alloc Zig state-machine `ChatTemplate` renderer across CLI/server/SDK/bench, and `KL`/top-1/code-math quality-gate harness (`de23a63`, `c184772`, `e459c23`)
- [x] `W24` `reasoning_effort` 4-part remediation: (1) expand domain to `{none, no_think, minimal, low, medium, high, xhigh, max}` + `[0.0, 0.99]` float (`Inkling 2` numeric map vs `Hy4` `high`/`no_think` normalization); (2) parse `/v1/chat/completions` top-level `reasoning_effort` & `chat_template_kwargs.{reasoning_effort, preserved_thinking}` in `http_sse.zig`; (3) emit template-specific prompt effort directives instead of collapsing positive levels to boolean `enable_thinking`; (4) expose `--reasoning-effort` in CLI (`chat_repl.zig`, `main.zig`) and `reasoning_effort` in C/Python SDK (`include/aminfer.h`, `c_api.zig`, `python/aminfer/engine.py`) (`c578d8a`, `c184772`, `e459c23`)
- [x] `C18` / `W24` 12-item `http_sse.zig` reasoning, stateful-response, timing & connection hardening: (1) deterministic thinking-field precedence & conflict `400`; (2) active `thinking_budget_tokens` decode cutoff; (3) zero-LLM-overhead `reasoning_text` + `reasoning.summary` (`auto`/`detailed` zero-copy pass-through, `none` -> `[]`, `concise` `< 1 us` CPU head+tail extraction); (4) post-`ThinkSplitter` channel-attributed `reasoning_tokens` counting; (5) full `output` + `created_at` + `previous_response_id` in streaming `response.completed`; (6) `HMAC-SHA256` Anthropic `thinking` `signature` across streaming `signature_delta`, non-streaming `thinking.signature`, and multi-turn inbound verification; (7) wider preallocated buffers (`--max-response-bytes`) + explicit `status: "incomplete"` instead of silent tail/transcript truncation; (8) never evict active `.in_progress` `StatefulResponseSlot`s (`stateful_response_slots >= max_connections`, `503` when full); (9) roll back `StatefulResponseSlot` on `job_ring.tryPush` `503` failure; (10) monotonic `--http-read-timeout-ms` (`408`) across incomplete headers/`Content-Length`/`chunked`; (11) zero-alloc HTTP/1.1 `keep-alive` socket reuse; (12) separate `prefill_done_ns` / `first_generated_token_ns` / `first_emitted_ns` so zero-output or buffered/special tokens never skew `ttft_ms` / `tpot_ms` (`c184772`, `e459c23`)
- [x] `C17`, `W21` & `X1` SDK, `.amf` & cluster boundary hardening: (1) reject negative/overflowing `max_tokens` & `top_k` (fixing `ctypes` `mod 2^32` wraparound and `out_ids[:gen_count]` out-of-bounds read), non-finite or out-of-range sampling/penalty params, and out-of-range/boolean `token_id` sequences in both `python/aminfer/engine.py` (`GenerationConfig.to_c`, `_prompt_to_ids`, `generate`, `step_decode`, `detokenize`) and `src/sdk/c_api.zig` (`generateTokensImpl` -> `AMI_SDK_ERR_INVALID_ARGUMENT`); (2) add defensive `std.math.mul`/`cast` narrowing guards in `src/format/amf.zig` `parseBytes()`; (3) enforce `u32` / `MAX_FRAME_PAYLOAD_BYTES` & `tx_buf.len` bounds in `src/cluster/transport.zig` `preparePayload()` and `commitFrame()` (`c578d8a`, `de23a63`)

## M8.15 task list

- [x] `C20` & `C18` sampling and penalty parity: wire `--top-k` (`40`), `--top-p` (`0.95`), `--min-p` (`0.05`), `--repetition-penalty` (`1.05` when `T > 0`; `1.0` when `T == 0` for bit-exact greedy parity), `--presence-penalty` (`0.0`), and `--frequency-penalty` (`0.0`) into `ChatConfig` / `ChatEngineRunner` (`src/cli/chat_repl.zig`, `src/main.zig`); wire request/prompt-keyed `pending_jobs` concurrent slot binding and per-slot `slot_penalties: []DefaultPenaltyTracker` in `BatchServeRunner` across prompt seeding, GPU `gs.sampleSync` (`seen_frequencies`), CPU fallback `sampleSlotConstrained`, and forced thinking-close tokens with zeroed `temperatures` for constrained slots during `prefillSlotCallback` and `stepDecodeBatchCallback` (`7bcab5a`, `019bad7`)
- [x] `W8` zero-copy single-frame GPU router top-k trace export (`aminfer decode --dump-router-trace <path.jsonl>` via `LoadOptions.enable_router_trace` carving `num_layers * 64 B` in `scratch_arena` across single-token `runStep`/`runStepWithLora`/`runStepHeterogeneous`/`runStepCold` + `lastStepRouterTopK`) and `aminfer-tools expert-heat` stderr warning when `calibration_mode != "router_trace"` (`7bcab5a`, `066bbb2`)
- [x] `W14` & `W17` `GSQ-RCO v2` fidelity hardening (`tools/src/aminfer_tools/gsq_rco.py`, `amf.py`, `cli.py`, `qwen4_exp.py`): activation-weighted OLS RCO minimizing $\sum_i w_i (\hat{x}_i - x_i)^2$ from 2D & 3D stacked MoE `--imatrix` per-channel $E[x_c^2]$ (`col_weights`); hard non-`q2_affine` backbone role floor (`_BACKBONE_PROTECTED_ROLES`, `enforce_backbone_floor = True`); `--max-q2-share` (`0.0` Dense / `0.55` MoE CLI default) & `--allow-unsafe-q2`; architecture-aware auto `target_bits` (`4.15` Dense / `3.50` MoE); CLI low-bit `q2_affine` double-quantization guard (`--allow-retranscode-lowbit`); and `--fine-rungs` (`(Q2/Q3/Q4_AFFINE, g32)` + `(Q5/Q6/Q8_AFFINE, g64)`) capped at `<= 5` schemes/layer (`moe.NUM_SCHEME_BUCKETS = 5`, uniform per-layer MoE `TensorLayout`) (`066bbb2`, `b8c4824`)
- [x] `W24` `aminfer quality-gate` 3rd conceptual reasoning prompt (`Riemann zeta`) and 4-gram / consecutive-token repetition loop detector (`computeMaxNgramRepeatCount`, `hasDegenerateRepetitionLoop`, `cand_max_4gram_repeats`, `repetition_loop_detected` -> `"fail_repetition_loop"` in `src/bench/m81.zig`) (`7bcab5a`)

## M8.2 task list

- [ ] `W23` & `G25` `Inkling 2` (`inkling2`, `inkling2_loop`: SIMD register-fused `RMSNorm + k=4 sconv`, single-pass `q/rel_l2_clip + log_scaling + d_rel=16 rel_attn + attn_sink_proj + gate_attn + softcap`, MoE `bias + route_scale + softcaps + 6-layer attn_res`, `40/48` `1024`-token `HybridKvRing` SWA, `W20` `23L` ViT sidecar split, and `Small` `GSQ-RCO` profiles) and `Hy4` (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`: `IndexCache`/`IndexPool`, `LatentMoE`, `learnable_sink_param`, `linear_gate`, `swiglu_limit=10`, `sandwich_norm`, `40L`/`27L` `ViT` wiring) `.amf` converters and Metal 4 operator suite
- [ ] `D14` block-parallel speculative draft heads (`Inkling 2` `3L`/`5L` `DFlare` `block_size=8` `rnn_hidden` Markov + standalone `dflash` GGUF paired with `M=8` batch `Core×3` loop verification, and `Hy4` `DFly2` `block_size=7`) wired into `DraftSource` + `G5`/`D9`
- [ ] `C26` L2 `SME2` `sme2_panel64` `gate_rank=64` loop controller offload + cold-expert NVMe `F_NOCACHE` placement for `Inkling-2-Mini` (`24 GB` M4 `<= 17.76 GiB` wired fit, `128K+` SWA KV headroom, `0` swap) and `Hy4-flash` per-row `F8_E4M3` Engram NVMe streaming, plus end-to-end local verification on `Inkling-2-Mini` (once disk space allows) and `yujiepan/hy4-preview-tiny-random` / scale-calibrated fixtures (`Small`/`Hy4`/`Hy4-flash` full runs storage/HW-gated)

## M9.1–M9.3 staged task lists

- [ ] **M9.1 (`MiMo-V2.6` family: `W25`, `G26`, `D15`)**: `mimo_v2` (`Flash` `309B`, `Pro` `1.02T`) & dense `qwen3_5` (`Distill-Qwen-9B`) `.amf` converters + `GSQ-RCO` profiles; asymmetric `192/128` head-dim `128+sink` `HybridKvRing` SWA (`40/48` & `60/70` layers); chained `3L` native `MTP` + `DFlash` block-diffusion drafting + `D7` pruning; local `Distill-9B` & synthetic `mimo_v2` verification on `24 GB` M4
- [ ] **M9.2 (`GLM-5` family: `W26`, `G27`, `D15`)**: `glm_moe_dsa` (`GLM-5.2`/`5.3` `753B`) & `glm5_next` (`GLM-5.3-Flash` `320B` + `W20` `24L` ViT sidecar split) `.amf` converters + `GSQ-RCO` profiles; `index_topk_freq=4` + `index_share_for_mtp_iteration=true` zero-reindex `MLA`/`DSA`, single-kernel register-fused 20-step `hc_mult=4` `mHC` Sinkhorn, `34L` `fused_qkvbfg_a_proj` linear attention + `11L` `NoPE MLA` (`qk_rope=0`, `index_kpool=4`), and `1L` MTP / `DFlash2`
- [ ] **M9.3 (`Ling-3.1-flash-1019` & `C27` dynamic MoE tiering: `W27`, `G28`, `C27`)**: `bailing_hybrid` (`572B`) `.amf` converter + `RCO-Q3` dense-`KDA` compression; `63L` `KDA` (`kda_lower_bound=-5.0`, `kda_safe_gate`, `k=4` short conv) + `9L` Gated `MLA` (`g_proj`) + L2 `SME2` `eh_proj` `1L` MTP; lossless session-local dynamic hot-expert working-set promotion in `WiredArena` + opt-in `--cold-expert-min-weight` (`0.0` default lossless, `W24` quality-gated)

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
| 2026-11-22 | M6.8 Phase 1 (`ec1a3b7`, `356f654`): `R9` live `HOST_VM_INFO64` non-anonymous RAM admission (`vmstat`, `budget`) & `WiredArena.allocAliasedPair`; `C15` `decode-share` QoS ledger, depth-scaled chunk narrowing, evict-first `PrefixCache` boundary checkpoints & ChatML boundary parser; `G15` wide-chunk `BM=64, BN=64` GEMM, direct bindless `KvTile` FlashAttention & wide hybrid `GDN`/`QSA`/SWA kernels; `G16` startup `.metallib` numerical canary (`f16`/`bf16`/`q4`) & `Apple10` (`NAX`) capability probe | Pass |
| 2026-11-22 | M6.8 Phase 2, audit remediation & final gate (`f70becf`, `ad3ee7f`, `2063bc9`, `2b9312d`): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug/ReleaseFast), `zig build test` (Debug, ReleaseSafe, ReleaseFast: `327/327` = `320` lib + `7` CLI, incl. `R = 3` QSA sparse `tail_m1_last` + PLE Isolation #2/#3 regression test and `autoClampServeBatchSlots` tests), filtered `test-vision` (`11/11`), `test-hybrid` (`16/16`), and `test-sme` (`10/10`) in Debug and ReleaseSafe, direct Debug `test-vision` (`11/11`), `test-hybrid` (`16/16`), `test-sme` (`10/10`), lib (`320/320`), and CLI (`7/7`) binaries under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (Metal API + GPU Validation enabled, `0` errors), `tools` unittest (`72/72`), ReleaseFast `aminfer probe` (`supports_apple10=no`, `simdgroup_apple9`, `live_zero_swap_reclaimable_mib=10053`, `canary.cold_elapsed_us=2132.5`, `canary.warm_inbuf_elapsed_us=219.0`), 3/3 consecutive ReleaseFast `aminfer bench-m68` runs (`1.72x–3.91x` wide+pruning across 6 real + 1 synth rows, `100%` greedy match, `R9`-gated `EAGLE-3` `cold_exp=66/41/96` with `greedy_match=yes` & `malloc_blk=0`, QoS `p50` `103.7x` / `p99` `1.47x` with `narrowed_chunks=3`, ChatML boundary `6.37x` TTFT, soft cache `64 -> 16`, `--strict-live-ram` rejection, `infer_swapouts_delta=0`, `total_swapouts_delta=0` on 3/3 runs), standalone `aminfer bench-decode` `tree64` vs `wide` A/B (`0.6B` `2.44x`, `14B` `2.24x`, `30B-A3B` `1.27x`, `-37..-39 MiB` wired, `0` swapouts), and live `aminfer serve` HTTP/SSE with default `--max-active-requests` at `--max-seq-len 10240 --prefix-cache-chunks 16` (`max_batch_size=2`, `3,534 MiB` wired, `Turn 1 -> 2` `11.06x` TTFT, `boundary_hits=1`, `evict_first_evictions=1`, `qos_steps=decode=210 prefill=4 narrowed_chunks=1`, `0` swapouts) archived in [benchmarks/results.md](benchmarks/results.md#m68-wide-chunk-gpu-prefill-prompt_tail-pruning-serving-qos-live-ram-admission-and-metallib-canary) | Pass |
| 2026-11-22 | M6.9 Phase 1 (`3bf7cc4`, `76e08c7`): `C16` `ChipProfile` (`sysinfo`, `chip_profile`, `attention`, `wavefront`, `sme`, `draft`) and `G17` Metal 4 `NAX` TensorOps (`nax_tensorops.metal` compiled into unified `aminfer.metallib` with `-std=metal4.0 -Wall -Werror`, `Apple7..11` probe + `Library.hasFunction`, lazy `*_nax` PSOs & canary auto-demotion in `dequant.zig`) | Pass |
| 2026-11-22 | M6.9 Phase 2 code & M4 (`Apple9`) verification gate (`4fcfc0c`, `320791e`): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug) & `zig build -Dcpu=apple_m1`, `zig build test` (Debug, ReleaseSafe, ReleaseFast: `340/340` = `332` lib + `8` CLI), filtered `test-vision` (`12/12`), `test-hybrid` (`17/17`), and `test-sme` (`11/11`) in Debug and ReleaseSafe, direct Debug binaries (`lib` `332/332`, `CLI` `8/8`, `test-vision` `12/12`, `test-hybrid` `17/17`, `test-sme` `11/11`) under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (Metal API + GPU Validation enabled, `0` errors), `./zig-out/bin/aminfer --help` & `./zig-out/bin/aminfer probe` (`chip.generation=m4`, `chip.spec_bus_gbs=120.00`, `gpu.apple_family=9`, `gpu.supports_nax=no`, `metallib.has_*_nax=yes`, `gpu.nax_fallback_reason=unsupported_gpu_family`, plus `--bus-bandwidth-gbs 546 --prefill-matmul-backend simdgroup` override check), `tools` unittest (`72/72`), and markdown link/anchor/hygiene audit across all 38 `.md` files (`Spike S8` physical `Apple10+` hardware bench deferred) | Pass (M4 code/unit gate; Apple10+ Spike S8 pending) |
| 2026-11-23 | M7 Phase 1 (`b2034c2`, `2abc0fe`, `b17bb9d`, `7edf3c3`): TB4 TCP / runtime TB5 RDMA-probed `kqueue` cluster transport & `ExpertParallelCoordinator`/`PeerWorkerServer` (`X1`–`X3`), `C10` 196B Engram store & `D10` `DSparkProposer` + `tools` `deepseek41` converter/partitioner (`W11`), Metal 4 `R7`/`G11` MLA+CSA2 FP4 (`890 B/tok`) KV + SWA replay & `G10` Single-Pass Mega-mHC kernels, and `src/root.zig` / `build.zig` `test-cluster` exports | Pass |
| 2026-11-23 | M7 Phase 2 engine, CLI, loopback cluster & M4 verification gate (`0a84cef`, `b9f8b9d`, `69d30af`): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug) & `zig build -Dcpu=apple_m1`, `zig build test` (Debug, ReleaseSafe, ReleaseFast: `365/365` = `356` lib + `9` CLI), filtered Debug `test-vision` (`12/12`), `test-hybrid` (`20/20`), `test-sme` (`11/11`), and `test-cluster` (`21/21`), direct Debug binaries (`lib` `356/356`, `CLI` `9/9`, `test-vision` `12/12`, `test-hybrid` `20/20`, `test-sme` `11/11`, `test-cluster` `21/21`) under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` (`0` Metal validation errors), `tools` unittest (`81/81`), generated compact synthetic `deepseek41` bundle (`/Volumes/Akari/models/amf/synth-deepseek41/`, `7.0 MiB`, `verify-amf` OK on backbone, 2 Engram shards, and `dspark.amf`), ReleaseFast `aminfer bench-m7` (`890 B/tok` `3.51x`/`81.1x`, `66.7%` 3-layer CSA2 KV savings, Mega-mHC `31.85 GB/s` at `M=64`, Engram `7.47 -> 0.82 -> 0.49 us` / `0` `r>=1` SSD reads, synthetic CED `341.26 tok/s` & `DSpark` `greedy_match=yes` with `0` encoder decode dispatches, `tb5_rdma_hw=unavailable` -> `tb4_tcp` & `strict_tb5_reject=pass`, 2-rank CED `greedy_match=yes` at `118.73 us` RPC, 4-rank MoE `0.000000` max error at `121.20 us` RPC, `swapouts_delta=0`), and live 2-OS-process `peer-worker` (Rank 1) + `decode --peer` (Rank 0) + `spec-decode --draft-mode dspark` (`15/15` bit-exact greedy token parity across all three, `0` encoder decode dispatches, `swapouts_delta=0`, `--strict-tb5-rdma` returning `Tb5RdmaUnavailable`) archived in [benchmarks/results.md](benchmarks/results.md#m7-ultra-and-cluster-scale-out-deepseek41-ced-mlacsa2-fp4-mega-mhc-engram-dspark-and-loopback-tcp-cluster); physical multi-Mac TB5 RDMA and full `552B + 196B` execution deferred pending hardware | Pass (M4 code, bench-m7 & 2-process loopback gate; 512 GB Ultra / TB5 RDMA HW pending) |
| 2026-11-24 | M7.8 Phase 1 (`9f13f37`, `32bfe1a`, `d56fff0`, `a9846a2`) & Phase 2 (`0de204b`, `5345566`) integration, M4 zero-regression `G18` calibration, `bench-m78`, and verification gate (incl. post-audit fixes): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug) & `zig build -Dcpu=apple_m1`, `zig build test` (Debug & ReleaseSafe: `433/433` = `419` lib + `10` CLI + `4` C-ABI SDK), `test-vision` (`19/19`), `test-sme` (`16/16`), `test-hybrid` (`23/23`), `tools` unittest (`94/94`), `python/tests` SDK unittest (`5/5`), `otool -tvV` SME2 streaming `[sp` audit (`0` streaming spills across `35` `_ami_sme2_*` symbols), and `ReleaseFast` `aminfer bench-m78` (`G18` single-frame canary `122.79 us < 350 us`, synth A/B `9.216 -> 4.342 ms` prefill64 & `2244.54 -> 2017.60 us` decode, real `Qwen3-0.6B` BF16 `48.767 -> 44.933 ms` & `45.90 -> 47.77 tok/s`, `14B`/`30B` cleanly `skipped_live_ram_headroom`, `G19` `2.36x`–`5.50x` for `Q=32..512`, `G20` indirect guard+parity, `G21` 248K `14.31 us` greedy / `111.89 us` radix + `C22` accept `9.00`, `C23` JSON `5.18x` / BPE `2.02x`, `C24` `luti2` `87.64x` / `565.82 GOPS` & `7.69 us` hoisted multi-LoRA, `G22` 3L ViT `0.603 ms` / `128 KiB` aliased / `0 B` decode overhead, `swapouts_delta=0`) archived in [benchmarks/results.md](benchmarks/results.md#m78-native-kernel-barrier-fusion-mma-flashattention-and-simd-accelerators); `Apple10+` NAX FlashAttention-2 HW bench gated on Spike `S8` | Pass (M4 code & bench-m78 gate; Apple10+ NAX HW pending Spike S8) |
| 2026-11-24 | M8.1 Phase 1 (`c578d8a`, `ce828bd`, `de23a63`, `c184772`), Phase 2 (`e459c23`), and Phase 3 verifier remediation (`f091cc5`) integration, CLI, `W24` `KL` quality gate, `bench-m81`, and full verification gate: `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug & ReleaseFast) & `zig build -Dcpu=apple_m1`, `xcrun -sdk macosx metal -std=metal4.0 -Wall -Werror -Wframe-larger-than=1` on `quant_formats.metal` (`0` warnings, `0` spills), `otool -tvV` SME2 audit (`0` streaming `[sp` spills, `0` leaked ops, `0` missing `d8–d15` saves across `67` `_ami_sme2_*` symbols: `61` public `.globl` entrypoints + `6` local `.s` helpers in Debug and ReleaseFast), `zig build test` (Debug & ReleaseSafe: `474/474` = `456` lib + `12` CLI + `6` C-ABI SDK), `test-quant-formats` (`21/21`), `test-sme` (`22/22`), `test-vision` (`22/22`), `test-hybrid` (`26/26`), `test-cluster` (`27/27`), `tools` unittest (`107/107`), `python/tests` SDK unittest (`6/6`), standalone `aminfer quality-gate` (both `-k` omitted and explicit `-k`) and `ReleaseFast` `aminfer bench-m81` (`G23` `M=1` GEMV up to `90.96 GB/s` / `75.8%` bus on `q4_affine_g64`, `87.22 GB/s` on `gguf_q4_0`, and `72.81 GB/s` on `nvfp4_g16`, `M=64` prefill `0.604–0.693 ms`, `G24` `none` `0` dispatches & `fouroversix_mae` `99.9%` win rate / `7.46%` MAE reduction, `C25` SME2 up to `298.93 GOPS` & `c13_bus_gate=pass`, `W21`/`W22` E2E dense/hetero-MoE/draft/fork-reject pass, `W24` live `Inkling-2` `46.18 MB/s` & `Hy4` `27.57 MB/s` tokenizers, `ChatTemplate` render `0.25–0.49 us`, `C18` `keepalive`/`hmac_sig_400`/`resp_completed_output` pass, real `qwen3-0.6b-bf16_vs_qwen3-0.6b-gsq-rco` `mean_kl=0.401752` / `top1=62.5% (5/8)` / `greedy=3/8` / `math=3/4` / `code=0/4` / `tok=resolved`, `swapouts_delta=0`) archived in [benchmarks/results.md](benchmarks/results.md#m81-multi-format-quantization-parity-chattemplate-c18-hardening-and-w24-kl-quality-gate) | Pass |
| 2026-11-24 | M8.15 (`7bcab5a`, `066bbb2`, `019bad7`, `b8c4824`): `C20` `aminfer chat` & `C18` `BatchServeRunner` full sampling/penalty parity (`DefaultPenaltyTracker` per-turn/per-slot seeding, request/prompt-keyed `pending_jobs` concurrent slot binding, GPU `sampleSync` + CPU fallback constrained sampling, `T = 0` bit-exact greedy bypass), `W8` zero-copy single-frame GPU router top-k trace export (`aminfer decode --dump-router-trace`) + `expert-heat` shallow-fallback stderr warning, `W14`/`W17` `GSQ-RCO v2` (activation-weighted OLS RCO using 2D & 3D stacked MoE `--imatrix` per-channel $E[x_c^2]$, hard non-`q2_affine` backbone role floor, `--max-q2-share`, auto `4.15` Dense / `3.50` MoE `target_bits`, CLI low-bit `q2_affine` double-quantization guard with `--allow-retranscode-lowbit`, and `--fine-rungs` `g32` + `Q5`/`Q6`/`Q8_AFFINE` rungs capped at `<= 5` schemes/layer with uniform per-layer MoE `TensorLayout`), and `W24` 3-prompt + 4-gram/consecutive repetition loop quality gate (`fail_repetition_loop`): `zig fmt --check build.zig src`, `0` `catch {}`, `zig build` (Debug & ReleaseFast), `zig build test` (Debug & ReleaseSafe: `480/480` = `459` lib + `15` CLI + `6` C-ABI SDK across `10/10` steps), `tools` unittest (`115/115`, incl. `14/14` in `test_m5_gsq_rco.py` covering 2D & 3D MoE weighted OLS RCO, backbone floor, `--max-q2-share`, low-bit guard, `--fine-rungs` + `--lut16-sme2` `verify-amf`/`decode`, and `decode --dump-router-trace` -> `expert-heat` `calibration_mode="router_trace"`), `python/tests` SDK unittest (`6/6`), and `ReleaseFast` 32-step 3-prompt `aminfer quality-gate` on `qwen3-0.6b-bf16.amf` vs `qwen3-0.6b-gsq-rco.amf` (`status=ok`, `mean_kl=0.313250`, `max_kl=1.712455`, `top1=62.5% (20/32)`, `greedy=4/32`, `math=4/11`, `code=0/11`, `reason=0/10`, `max_4gram=1`, `loop=no`, `tok=resolved`, `act_quant=none`, `finite=yes`, `swapouts_delta=0`, `vm_stat` `55156714 -> 55156714`) archived in [benchmarks/results.md](benchmarks/results.md#m815-sampling-penalty-parity-gsq-rco-v2-fidelity-hardening-and-router-trace-loop-quality-gate) | Pass |

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
After M6.5 close-out on 2026-11-22, the user confirmed
[ADR-0020](decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md),
inserting Milestone **M6.8** (`R9`, `G15`, `G16`, `C15`) before M7:
configurable wide-chunk (`512–1024`) FlashAttention / `BM=64, BN=64` GEMM GPU
prefill (`P0-A`, `--gpu-prefill-mode wide|tree64`, default `wide`), lossless
`prompt_tail` final-layer compute pruning (`P0-B1`), single-request
wide-prefill / `verifyTree` activation scratch aliasing without unloading
`lm_head` weights (`P0-B2`), default-on continuous-batching `decode-share`
wall-time QoS, depth-scaled chunk narrowing, and ChatML message-boundary
evict-first `PrefixCache` checkpoints (`P1-A`), live `HOST_VM_INFO64`
non-anonymous RAM soft cache auto-sizing plus `--strict-live-ram` (`P1-B`), and
a startup `.metallib` numerical canary (`~0.19–0.24 ms` warm in-buffer / `~1.3–3.8 ms`
cold first-submit) plus `Apple10` (M5 `NAX`) capability/dispatch hook reservation (`P2`).
After M6.8 close-out on 2026-11-22, the user confirmed
[ADR-0021](decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md),
scheduling Milestone **M6.9** (`G17`, `C16`, Spike `S8`) before M7 for
single-binary / single-`.metallib` multi-generation Apple Silicon adaptation:
co-packaging `Apple9` (`simdgroup_matrix`) and `Apple10+`
(`MetalPerformancePrimitives` `mpp::tensor_ops::matmul2d`) wide prefill /
batched GEMM kernels with lazy `MTL4Compiler` `Apple10`/`Apple11` selection,
`G16` canary-guarded automatic fallback to `simdgroup_apple9`, chip-parameterized
`spec_bus_gbs` telemetry, and runtime ANE/GPU/SME2 crossover calibration
(with `Apple9` AIR/dispatch/fallback verified on M4 and physical `Apple10+`
hardware measurements gated on Spike `S8`).
Following M7 M4-loopback close-out and live HF metadata validation on
2026-11-23, the user confirmed
[ADR-0022](decisions/0022-native-c-python-sdk-and-full-protocol-parity.md),
scheduling Milestone **M7.5** (`R10`, `W18`, `W19`, `C17`–`C22`) for an outward
plain-C shared library (`libaminfer.dylib` + `include/aminfer.h`) and
zero-dependency Python `ctypes` SDK (`python/aminfer/`), full three-endpoint
HTTP/SSE protocol parity across `/v1/chat/completions`, `/v1/responses`, and
`/v1/messages` (plus `POST /v1/messages/count_tokens`, tool calling, bounded
stateful `previous_response_id`, shared sampling, stop holdback, constant-time
`--api-key` auth, and `AF_UNIX` `--unix-socket` listener), the rootless
`aminfer monitor` terminal TUI (default `trans-pride` TrueColor palette +
`btop` `.theme` support, `--once`/`--json`, and `--server` attach),
self-contained single-file `.amf` metadata (`W18`, `aminfer ls`/`inspect`),
scrolling `aminfer chat` CLI REPL + `stdin` pipe (`C20`, explicitly not a
full-screen TUI), native `/v1/embeddings` and `/v1/rerank` (`C21`), a
preallocated `json_schema`/GBNF token-bitmask engine compatible with `G5`/`D9`
speculative tree verification (`C22`), runtime multi-LoRA hot-loading and
`"<base>:<adapter>"` routing (`W19`), and opt-in `--disk-prefix-cache` NVMe
KV/GDN persistence (`R10`), explicitly excluding audio/speech ingress.
On the same date, the user confirmed
[ADR-0023](decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md),
scheduling Milestone **M7.8** (`W20`, `G18`–`G22`, `C23`–`C24`) for native Metal 4
sublayer fusion and `MTLStageDispatch` barrier elimination (`G18`), hardware
`simdgroup_half8x8` (`Apple9` MMA) and `Apple10+` `NAX` FlashAttention-2
prefill (`G19`), GPU-driven indirect MoE threadgroup dispatch (`G20`), fused
GPU `248K`-vocab radix/bucket constrained sampling (`G21`) paired with explicit
ARM64 NEON JSON/BPE vector scanning (`C23`), ARMv9.2 SME2 `luti2` (`2-bit`)
assembly microkernels (`C24`), and full multi-layer Vision Transformer (`ViT`)
encoder tower + 2-layer `PatchMerger`/`Aligner` MLP packing and Metal 4 / CoreML
ANE execution (`W20`, `G22`), alongside strict zero-overhead fast paths and
occupancy/crossover non-regression gates across `M7.5` and `M7.8`.

On the same date, Milestone **M7.5** (`R10`, `W18`, `W19`, `C17`–`C22`,
`996782f..ddd90a4`) landed and was verified end-to-end on M4
([benchmarks/results.md](benchmarks/results.md#m75-native-cpython-sdk-full-httpsse-protocol-parity-rootless-tui-monitor-and-local-ergonomics)):
rootless `aminfer monitor` TUI (`C19`, `trans-pride` default palette + `btop`
`.theme` loader, `--once`, `--json`, `--server`), plain scrolling `aminfer chat`
REPL + `stdin` pipe (`C20`, `/clear`, `/retry`, `/edit` `C15` boundary
checkpoints, `/image` & `/video` `G14`), plain-C `libaminfer.dylib` +
`include/aminfer.h` and zero-dependency Python `ctypes` SDK (`C17`, `0`
third-party dylibs via `otool -L`, `0` leaked wired pages across create/destroy
cycles), single-file `.amf` embedded metadata + `aminfer ls`/`inspect` (`W18`,
`127.96 us` inspect with `0 B` wired delta), runtime multi-LoRA hot-loading and
`"<base>:<adapter>"` routing (`W19`, `<= 4 MiB` L2 wired pool with `0`-overhead
`active_adapters == 0` bypass), native `lm_head`-elided `/v1/embeddings` and
`/v1/rerank` (`C21`), opt-in `--disk-prefix-cache` 16 KiB-aligned `F_NOCACHE`
spill/reload (`R10`, full token-slice collision rejection), preallocated
`json_schema`/GBNF token-bitmask PDA (`C22`, `43.54 us` compile, `6.01 us`
per-step bitmask eval, `0` per-step heap alloc), and full three-endpoint
HTTP/SSE protocol parity (`C18`, `/v1/chat/completions`, `/v1/responses`,
`/v1/messages` + `count_tokens`, `--api-key`, `--unix-socket`).

Following `M7.5` and Hugging Face checkpoint/header verification (`Inkling 2` and
`Hy4` safetensors, tokenizers, chat templates, `FP8`/`MXFP8`/`NVFP4` scale
granularities, and GGUF per-tensor type scans), the user confirmed
[ADR-0024](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md),
scheduling Milestone **M8** (`W21`–`W24`, `G23`–`G25`, `D14`, `C25`–`C26`)
after `M7.8`: zero-loss native `.amf` dtypes and Metal/SME2 register
dequantization (`fp8_e4m3_block` `128×128` `F32`/`BF16`, `fp8_e4m3_row`,
`mxfp8_g32`, `nvfp4_g16`, MLX `5/6/8-bit` `g32`/`g128`), a pure-stdlib lossless
offline `GGUF` -> `.amf` repacker rejecting colliding fork type IDs (such as
`AngelSlim` `STQ1_0` overloading upstream `GGML_TYPE_F8_E4M3 = 43`), opt-in
`--act-quant emulate` (`G24`), imatrix-aware sub-4-bit protection paired with a
mandatory `KL`/top-1 capability gate (`W24`), `o200k`/`ignore_merges` + `Hy4`
tokenizers and pluggable chat templates replacing hardcoded ChatML, end-to-end
`Inkling-2-Mini` (`24L` recurrent + NVMe cold experts, plus `Q8_0`/`BF16`
`dflash`) on the `24 GB` M4 baseline, and `Inkling 2` / `Hy4` operators +
`DFlare`/`DFly2` block-parallel drafting (`D14`), with full-scale
`Inkling-2-Small`, `Hy4-preview`, `Hy4`, and `Hy4-flash` runs hardware-gated
on `128–512 GB` / `TB5` clusters.

On `2026-11-24`, following completion of `M7.8` (`8929b81`), the user refined
[ADR-0024](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md)
execution with four pre-implementation confirmations: (1) splitting `M8` into
sequential **M8.1** (`W21`, `W22`, `W24`, `G23`, `G24`, `C25` substrate) ->
**M8.2** (`W23`, `G25`, `D14`, `C26` `Inkling 2` / `Hy4` operators, draft heads,
and end-to-end runs); (2) compiling Hugging Face `chat_template.jinja` offline
in `tools/` into a declarative `.amf` JSON `chat_template_spec` rendered by a
deterministic zero-allocation Zig state machine (`W24`, zero runtime Jinja2 AST
interpreter); (3) fixing all four confirmed `reasoning_effort` mapping gaps in
`M8.1` (`W24`: full `none`..`xhigh`/`max` + `[0.0, 0.99]` domain with `Inkling 2`
numeric and `Hy4` `high`/`no_think` normalization, `/v1/chat/completions`
top-level + `chat_template_kwargs` parsing in `http_sse.zig`, template-faithful
prompt directive emission instead of boolean collapse, and CLI `--reasoning-effort`
+ C/Python SDK surface parity); (4) adding the 11 verified `http_sse.zig`
(`C18`/`W24`) reasoning, stateful-response, and connection hardening items to
`M8.1` (deterministic thinking-field precedence, active `thinking_budget_tokens`
cutoff, zero-LLM-overhead `reasoning_text` + `reasoning.summary` rule-based
routing, channel-accurate `reasoning_tokens` accounting, full `output` in
streaming `response.completed`, `HMAC-SHA256` Anthropic `thinking` signatures in
stream/non-stream + inbound verification, wider non-silent buffers, in-flight
`StatefulResponseSlot` protection + `503` leak rollback, `408` read timeout,
zero-alloc HTTP/1.1 `keep-alive` reuse, and separated `prefill_done_ns` /
`first_generated_token_ns` / `first_emitted_ns` `TTFT`/`TPOT` accounting);
(5) adding C/Python SDK (`engine.py` + `c_api.zig` sampling/`max_tokens`/`token_id`
validation), `.amf` (`parseBytes` defensive narrowing), and cluster (`transport.zig`
`u32` payload bounds) boundary hardening to `M8.1`; and (6) pausing heavy `M8`
checkpoint downloads until disk space on `/Volumes/Akari` (`12 GiB` free / `95%`
full) is cleaned up.

Later on `2026-11-24`, after `M8.1` landed (`5da4f9e`) and live Hugging Face
`config.json` / quantized-tree byte audits were completed across `Inkling 2`,
`MiMo-V2.6`, `GLM-5`, and `Ling-3.1`, the user confirmed
[ADR-0025](decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md):
(1) folding deep architecture-specific `Inkling-2-Mini` and `Inkling-2-Small`
co-design directly into **`M8.2`** (SIMD register-fused `sconv`/`d_rel=16`/`clip`/`sink`/`softcap`/`attn_res`
sublayers, L2 `SME2` `sme2_panel64` `gate_rank=64` loop controllers, `40/48`
`1024`-token `HybridKvRing` SWA, `W20` ViT sidecar stripping, `3L`/`5L` `DFlare`
`M=8` batch loop verification, and `Small` `GSQ-RCO` tier-down profiles); and
(2) scheduling three sequential post-`M8.2` deep-optimization milestones —
**`M9.1`** (`MiMo-V2.6-Flash` `309B`, `Pro` `1.02T`, `Distill-Qwen-9B`: `W25`,
`G26`, `D15`), **`M9.2`** (`GLM-5.2` & `5.3` `753B`, `GLM-5.3-Flash` `320B`:
`W26`, `G27`, `D15`), and **`M9.3`** (`Ling-3.1-flash-1019` `572B` + cross-family
session-local dynamic hot-expert promotion & opt-in `--cold-expert-min-weight`:
`W27`, `G28`, `C27`).

Following a root-cause investigation of repetition loops and knowledge collapse
on `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`) on `2026-11-24`, the user confirmed
inserting Milestone **`M8.15`** ([ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md#m815-gsq-rco-v2-fidelity-hardening-w8-w14-w17-w24-c18-c20))
immediately before `M8.2` to remedy all three compounding causes: (1) `GSQ-RCO v2`
(`W14`, `W17`) activation-weighted OLS RCO using `--imatrix` per-channel $E[x_c^2]$,
hard non-`q2_affine` backbone role floor, `--max-q2-share`, auto `4.15` Dense /
`3.50` MoE `target_bits`, CLI low-bit `3-bit -> q2_affine` double-quantization
guard (`--allow-retranscode-lowbit`), and `--fine-rungs` (`g32` +
`Q5`/`Q6`/`Q8_AFFINE`) capped at `<= 5` schemes/layer; (2) single-frame zero-copy
GPU router top-k trace export (`aminfer decode --dump-router-trace`, `W8`) plus
`expert-heat` shallow-fallback stderr warnings; and (3) full constrained
sampling and penalty parity across `aminfer chat` (`C20`) and `BatchServeRunner`
(`C18`), paired with a 3rd conceptual reasoning prompt and a 4-gram /
consecutive repetition loop detector (`fail_repetition_loop`) in `aminfer
quality-gate` (`W24`).

## Session handoff snapshot

Update this before the context window runs out or when handing off.

- Last updated: 2026-11-24
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
    and skip pre-timing CRC scan in `bench-m65`, `8d31ba2` M6.5 refreshed docs
  - M6.8: `ca784d9` `ADR-0020` M6.8 roadmap codification, `ec1a3b7` `R9` live
    non-anonymous RAM admission & `WiredArena.allocAliasedPair` + `C15` QoS
    ledger, depth-scaled chunk narrowing & evict-first ChatML boundary
    `PrefixCache`, `356f654` `G15` wide-chunk `BM=64, BN=64` GEMM, direct
    bindless `KvTile` FlashAttention & wide hybrid kernels + `G16` startup
    `.metallib` numerical canary & `Apple10` probe, `f70becf` `Qwen3Model` /
    `GraphModel` / `SpecEngine` wide prefill, `prompt_tail` pruning, scratch
    aliasing, canary & live-RAM wiring, `ad3ee7f` CLI/server M6.8 flags, `probe`
    canary/live-RAM & `bench-m68` suite, `2063bc9` Metal 4 `@autoreleasepool`
    destroy drain, explicit `WiredRegion` `munlock` & single-load `bench-m68`
    polish, `9ee57f0` initial M6.8 docs archive, `2b9312d` Round-1 audit
    remediation (`R9`-gated `bench-m68` + `cold_rt` wide prefill, `ADR-0019`
    QSA/PLE `R = 3` wide-prefill isolations, cold + warm in-buffer canary
    profiling, `serve` default-slot IOSurface auto-clamping & `narrowed_chunks`
    `bench-m68` coverage), `365cccd` M6.8 refreshed docs
  - M6.9 (code complete; `Spike S8` awaiting `Apple10+` HW): `7c22f0d`
    `ADR-0021` M6.9 roadmap codification, `3bf7cc4` `C16` multi-generation
    `ChipProfile` & parameterized bandwidth/crossover gates, `76e08c7` `G17`
    Metal 4 `NAX` TensorOps (`nax_tensorops.metal`) compiled into unified
    `aminfer.metallib` with `Apple7..11` probe & canary auto-fallback,
    `4fcfc0c` `root` / `Qwen3Model` / `GraphModel` `ChipProfile`, `NAX`
    backend options, canary demotion & wavefront routing, `320791e` CLI flags,
    enhanced `probe` telemetry, `spec_bus_gbs` benchmark wiring & `bench-m69`
    harness
  - M7 (engine, kernels, tools, CLI & loopback TCP cluster verified on M4;
    physical multi-Mac TB5 RDMA & full 552B + 196B pending hardware):
    `b2034c2` TB4 TCP / runtime TB5 RDMA-probed transport & multi-rank
    `ExpertParallelCoordinator`/`PeerWorkerServer` (`X1`–`X3`), `2abc0fe` 196B
    Engram store, `DSparkProposer` & `deepseek41` offline toolchain (`C10`,
    `D10`, `W11`), `b17bb9d` Metal 4 MLA+CSA2 FP4 (`890 B/tok`) KV, SWA replay
    & Single-Pass Mega-mHC kernels (`R7`, `G10`, `G11`), `7edf3c3` root module
    exports & `test-cluster` build step, `0a84cef` `GraphModel` `deepseek41`
    CED two-phase scheduler, MLA+CSA2, Mega-mHC, Engram, `DSpark` & cluster
    coordinator integration, `b9f8b9d` `peer-worker` CLI, cluster flags,
    `--draft-mode dspark`, MLA loop-commit fix & `bench-m7` suite, `69d30af`
    tree SWA ring FP4 fallback alignment, Engram layer regex & chunked
    `bench-m7` KV fill
  - Live HF metadata & tokenizer compatibility: `cd72273` native DeepSeek
    3-stage pre-tokenizer auto-detection in `tokenizer.zig`, live
    `Qwen/Qwen3.8-Flash-Next` and `deepseek-ai/DeepSeek-V4.1-Flash-1017`
    tokenizer & `weight_map` key-alias fixes, and `aminfer-tools inspect-metadata`
    dry-run validator
  - M7.5: `996782f` `C19` rootless `btop`-style TUI `aminfer monitor`
    (`trans-pride` default + `.theme`) & `C20` scrolling `aminfer chat` REPL +
    `stdin` pipe, `198fe6b` `C17` `libaminfer.dylib` (`include/aminfer.h`) &
    zero-dependency Python `ctypes` SDK (`python/aminfer/`), `2531d69` `W18`
    single-file `.amf` metadata (`ls`/`inspect`), `W19` `LoraRegistry`
    (`.amlora`), `C21` `prefillEmbeddings`/`rerankSequences` & `R10`
    `DiskPrefixCache`, `4933d1d` `C22` preallocated `json_schema`/GBNF bitmask
    engine, `sampling.zig`/`StopHoldback` & `C18` full 3-endpoint HTTP/SSE
    protocol parity (`count_tokens`, `--api-key`, `--unix-socket`), `ddd90a4`
    CLI (`ls`, `inspect`, `monitor`, `chat`, `serve`), `W18` C-SDK auto-metadata
    & `aminfer bench-m75` integration
  - M7.8 (complete on M4; `Apple10+` `NAX` FlashAttention-2 HW bench pending Spike
    `S8`): `9f13f37` `C23` ARM64 NEON JSON/BPE scanners & `C24` SME2 `.s` `luti2`
    microkernels + `<= 4 MiB` L2 / `C13`-gated multi-LoRA dispatch, `32bfe1a`
    `W20`/`G22` Metal 4 bidirectional 2D-RoPE `ViT` tower + `PatchMerger`/AnyRes
    & `pack-vision`/`export-coreml --component vision-tower` tools, `d56fff0`
    `G18` Metal 4 sublayer/residual-epilogue fusion & `G19` `Apple9`/`NAX` MMA
    FlashAttention-2 prefill, `a9846a2` `G20` indirect MoE dispatch & `G21`
    248K GPU radix sampler, `0de204b` Phase-2 engine/speculative/SDK integration
    (`root.zig`, `qwen3.zig`, `graph.zig`, `lora.zig`, `spec/*`, `c_api.zig`,
    single-frame `G16`+`G18` canary, M4 `G18` shape calibration), and `5345566`
    `aminfer bench-m78` CLI & benchmark gates, `83a3763` M7.8 docs archive,
    `76530ff` verifier remediation (`c_api.zig` preallocated `GpuSampler` +
    penalty dedup, `graph.zig` ANE ViT bucket/merge alignment), and `8929b81`
    M7.8 doc audit fixes (`229c616`, `378d582`, `3fc29a3` `ADR-0024` M8 codification)
  - M8.1 (complete on M4): `c578d8a` `C25` SME2 multi-format kernels (`NVFP4`,
    `MXFP4`, `Q2_0`, `TQ1_0`, `TQ2_0`, `IQ2_XXS`), `C17`/`W24` SDK reasoning
    effort & parameter bounds hardening, and `X1` `transport.zig` `u32` bounds;
    `ce828bd` `G23`/`G24` Metal 4 `quant_formats.metal` + `quant_formats.zig`;
    `de23a63` `W21`/`W22`/`W24` `amf.zig` `DType 0..37`, `src/quant/formats.zig`,
    and Python `amf`/`gguf`/`chat_template`/`gsq_rco`/`cli` tools; `c184772`
    `W24` `Inkling 2`/`Hy4` tokenizers, zero-alloc `ChatTemplate`, `chat_repl.zig`
    and 12-item `C18` `http_sse.zig` hardening; `e459c23` Phase-2 engine, graph,
    draft, SDK, CLI (`--act-quant`, `--reasoning-effort`, `--max-response-bytes`,
    `--http-read-timeout-ms`, `quality-gate`, `bench-m81`), and `W24` `KL`
    quality-gate integration; `f091cc5` Phase-3 verifier remediation (Inkling 2 /
    Hy4 structural control-tag preservation and dialect EOS registration across
    `ServeRunner`, `BatchServeRunner`, `ChatEngineRunner`, and `SdkEngine`,
    Inkling 2 recipient-header tool-call parsing, active `thinking_budget_tokens`
    KV-state cutoff injection, zero-output `ttft_ms = 0.0` accounting,
    `model_id`-bound Anthropic `thinking` HMAC-SHA256, and dynamic
    multi-prompt math + code `quality-gate`), `5da4f9e` M8.1 docs & ADR-0025
    roadmap codification
  - M8.15 (complete on M4): `7bcab5a` `C20` `aminfer chat` & `C18`
    `BatchServeRunner` full constrained sampling/penalty parity, `W8`
    single-frame GPU `aminfer decode --dump-router-trace`, and `W24` 3-prompt +
    4-gram/consecutive repetition loop quality gate (`fail_repetition_loop`);
    `066bbb2` `W14`/`W17` `GSQ-RCO v2` activation-weighted OLS RCO, hard
    non-`q2_affine` backbone floor, `--max-q2-share`, auto `4.15` Dense / `3.50`
    MoE `target_bits`, CLI low-bit `q2_affine` double-quantization guard,
    `--fine-rungs` (`<= 5` schemes/layer), and `W8` `expert-heat` shallow-mode
    stderr warning; `019bad7` `C18` `BatchServeRunner` request/prompt-keyed
    `pending_jobs` concurrent slot binding & orphan guard; `b8c4824` `W14`/`W17`
    3D stacked MoE `--imatrix` channel slicing, uniform per-layer MoE
    `TensorLayout` guard under `--fine-rungs --lut16-sme2`, `qwen4_exp` auto
    `target_bits` parity, and non-finite `target_bits` validation
- Current test & build baseline (`M0`–`M6.8`, `M7.5`, `M7.8`, `M8.1`, and `M8.15`
  complete on M4, `M6.9` code complete, `M7` code & M4 loopback cluster verified,
  live HF frontier metadata & tokenizer compatibility verified, `0` `catch {}`):
  - Zig suite: `480/480` passing (`459` lib + `15` CLI + `6` C-ABI SDK across
    `10/10` steps) in `Debug` and `ReleaseSafe`, plus `test-quant-formats`
    (`21/21`), `test-sme` (`22/22`), `test-vision` (`22/22`), `test-hybrid`
    (`26/26`), and `test-cluster` (`27/27`); `zig fmt --check build.zig src`
    clean; `quant_formats.metal` `-std=metal4.0 -Wall -Werror
    -Wframe-larger-than=1` clean (`0` spills); `0` streaming `[sp` spills across
    `67` `_ami_sme2_*` symbols (`61` public `.globl` entrypoints + `6` local
    `.s` helpers, `otool -tvV`); `ReleaseFast` `aminfer bench-m75`, `bench-m78`,
    and `bench-m81` + standalone 32-step 3-prompt `aminfer quality-gate` all
    pass with `swapouts_delta = 0` (`loop=no`, `max_4gram=1`)
  - Offline Python `tools/` suite: `115/115` passing (`uv run --directory tools python -m unittest discover -s tests -v`, incl. `14/14` in `test_m5_gsq_rco.py`)
  - Zero-dependency Python `ctypes` SDK suite: `6/6` passing (`PYTHONPATH=python python3 -m unittest discover -s python/tests -v`)
    and `aminfer-tools inspect-metadata` dry-run validation against live
    Hugging Face JSON metadata (no weight shards on disk):
    - `Qwen/Qwen3.8-Flash-Next` (`HTTP 200`, `13 MiB` in `/Volumes/Akari/models/metadata/Qwen__Qwen3.8-Flash-Next/`: `config.json` `4,745 B`, `tokenizer.json` `12.22 MiB`, `tokenizer_config.json` `9,984 B`, `model.safetensors.index.json` `170,726 B`): `48` layers (`36` GDN + `12` QSA), `hidden=2560`, `512` routed experts (top-10 + shared), `hc=4`, `128`-shard PLE at layer `2`, `1` MTP layer, `24,576 B/tok` FP16 KV + `108.0 MiB` FP32 GDN state, `1,658` tensors across `131` shards (`335.28 GiB` raw BF16, est `q3_affine` backbone `52.56 GiB`; `300` long `hyper_connection` names `> 47 B` flagged for canonical aliasing)
    - `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`HTTP 200`, `15 MiB` in `/Volumes/Akari/models/metadata/deepseek-ai__DeepSeek-V4.1-Flash-1017/`: `config.json` `3,311 B`, `tokenizer.json` `6.07 MiB`, `tokenizer_config.json` `801 B`, `model.safetensors.index.json` `7.12 MiB`): `40` layers (`20` enc + `20` dec from `compress_ratios`), `hidden=5120`, `384` routed experts (top-6 + `1` shared, `fp8` + `fp4` `ue8m0`), `mHC=4`, Engram at layers `[1, 14]` (`768,022,850` rows), `3`-layer `128`-expert `mtp.*` DSpark head, `4,608 B/tok` BF16 / `1,256 B/tok` FP4 KV across 4 KV-source layers, `96,085` tensors across `48` shards (`475.24 GiB` raw, est `q3_affine` backbone `224.79 GiB`; `0` backbone names `> 47 B`; fits `512gb-ultra`, `tb5-cluster-4x128gb`, `tb5-cluster-2x256gb`)
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
    `aminfer-repo.ang` (`128.02 MiB`, `879,949` repo N-grams),
    `aminfer-docs.ang` (`32.02 MiB`, `198,113` docs N-grams, `table_crc32=0xc1f2f6ed`),
    and `/Volumes/Akari/models/amf/synth-deepseek41/` (`7.0 MiB` compact
    synthetic `deepseek41` CED bundle: `model.amf`, 2 Engram shards,
    `dspark.amf`, `config.json`, `dspark_config.json`, `tokenizer.json`)
  - Exported ANE `.mlpackage` bundles: `/Volumes/Akari/models/coreml/`
- Session & `/swarm` operating conventions:
  - 3-tier swarm model hierarchy per user instruction (`0` `claude-*` / `gpt-*`):
    - **Top tier (`gemini-4-carbon-high`, effort `high`)**: `coordinator`
      (architecture, cross-module arbitration, merge orchestration) and final
      milestone independent `verifier-*`
    - **Middle tier (`gemini-4-neon-xhigh`, effort `xhigh`)**: subsystem /
      Phase-2 engine integration (`*-engine`), code review, and `monitor`
    - **Bottom tier (`gemini-4-hydrogen-xhigh`, effort `xhigh`)**: parallel
      module / kernel / tool implementation workers (`workspace: "share"`)
  - Commits stay local (do not `git push` unless explicitly requested) and run
    `ikat sync` after every commit
- In flight: **M8.2** (`W23`, `G25`, `D14`, `C26`: `Inkling 2` and `Hy4` family
  operators, `DFlare`/`DFly2` block-parallel draft heads, NVMe cold-expert and
  Engram placement, and end-to-end `Inkling-2-Mini` + `hy4-preview-tiny-random`
  on `24 GB` M4 per
  [ADR-0024](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md)
  and
  [ADR-0025](decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md))
  is planned next. **M8.15** (`W8`, `W14`, `W17`, `W24`, `C18`, `C20` per
  [ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md#m815-gsq-rco-v2-fidelity-hardening-w8-w14-w17-w24-c18-c20)),
  **M8.1** (`W21`, `W22`, `W24`, `G23`, `G24`, `C25`, plus
  `C17`/`C18`/`X1` boundary hardening), **M7.8** (`W20`, `G18`–`G22`,
  `C23`–`C24` per
  [ADR-0023](decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md)),
  and **M7.5** (`R10`, `W18`, `W19`, `C17`–`C22`
  per [ADR-0022](decisions/0022-native-c-python-sdk-and-full-protocol-parity.md))
  are complete on M4. **M6.9** (`G17`, `C16`, Spike `S8` per
  [ADR-0021](decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md))
  and **M7** (`R7`, `W11`, `G10` mHC, `G11`, `D10`, `C10` Engram, `X1`–`X3` per
  [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md) and
  [ADR-0003](decisions/0003-expert-source-tagged-union.md)) engine, kernel,
  tool, CLI, live HF metadata dry-run (`cd72273`), and M4 loopback verification
  gates are complete; remaining hardware-gated acceptances are physical
  `Apple10+` (M5+) `Spike S8` (`aminfer bench-m69` and `G19` `NAX`
  FlashAttention-2 HW bench), physical multi-Mac TB5 RDMA HCA transport
  (`X1`–`X3`), and full-scale `552B + 196B` `DeepSeek-V4.1-Flash-1017` (`W11`)
  execution on a 512 GB Ultra node or multi-Mac TB5 RDMA cluster
- Awaiting the user: none for M8.1 / M8.15 (completed on M4); confirm go-ahead
  for M8.2 (`Inkling 2` / `Hy4` operators + checkpoint downloads on
  `/Volumes/Akari`)
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
  [decisions/0019-recurrent-depth-and-post-m5-accelerators.md](decisions/0019-recurrent-depth-and-post-m5-accelerators.md),
  [decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md](decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md),
  [decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md](decisions/0021-multi-generation-apple-silicon-and-nax-tensorops.md),
  [decisions/0022-native-c-python-sdk-and-full-protocol-parity.md](decisions/0022-native-c-python-sdk-and-full-protocol-parity.md),
  [decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md](decisions/0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md),
  [decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md](decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md),
  [decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md](decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md)
