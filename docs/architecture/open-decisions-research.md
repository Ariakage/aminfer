# Open-decision research and recommendations

> Technical survey and engineering rationale behind open-question resolutions.
> Items confirmed by the user on **2026-11-21** have been codified into their
> governing [ADRs](../decisions/README.md); authoritative capability status
> always lives in [../status.md](../status.md) and milestone/spike state in
> [../process.md](../process.md#spikes).

## Summary matrix

| ID | Question | Resolution state | Settled / recommended direction | Governing ADR / reference |
| --- | --- | --- | --- | --- |
| `R5` | `int8` KV tiles (`OneComponent8`) | **Confirmed (2026-11-21)** | Decline `OneComponent8` for ANE↔GPU; keep `OneComponent16Half` / `RGBA16Float` (incl. M6 `512`+`128` hybrid ring pools); optional GPU-side `q8` `MTLBuffer` compression for retired 32K+ chunks | [ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md), KIVI ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) |
| `C8` | kqueue HTTP/SSE server API shape | **Confirmed (2026-11-21)** | Stateless `POST /v1/chat/completions` (`delta.reasoning_content`, `seed`/`seed_used`) + `GET /v1/models` + `GET /health`; optional stateless `/v1/responses` & `/v1/messages` | [runtime-pipeline.md](runtime-pipeline.md#request-lifecycle), `vLLM`, `llama-server` |
| `A2` | ANE MoE expert predictor architecture | Empirical in M4 | `lookahead = 2..3` cross-layer `mlp.gate` weights SRAM-resident (`< 2 MiB`); add bottleneck MLP only if recall falls short | [experts-and-storage.md](experts-and-storage.md#expert-prediction-and-prefetch), *Fate* ([arXiv:2502.12224](https://arxiv.org/abs/2502.12224)) |
| `A3` | Static CoreML ANE shapes for growing KV | Empirical in M4 (Spike `S6`) | Benchmark static `[512, hidden]` ANE projections + GPU split-KV vs static KV buckets (`512..4096`) in Spike `S6` | [ADR-0010](../decisions/0010-hybrid-wavefront-prefill.md), [../process.md](../process.md#spikes) |
| `ADR-0016` | Optional `bf16` inter-layer residual buffer | **Confirmed (2026-11-21)** | Keep `f16` default for QK-Norm models (`Qwen3`); add opt-in `bf16` trunk residual on GPU/SME2 for non-QK-Norm backbones, narrowing `fp32 → fp16` strictly at `RGBA16Float` IOSurface boundaries | [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md), [arXiv:2402.17762](https://arxiv.org/abs/2402.17762) |
| `ADR-0017` | Initial `G13` GSQ+RCO affine bit-widths | **Confirmed (2026-11-21)** | Lock initial M5 `G13` set to `{q2_affine, q3_affine, q4_affine, bf16, f16}` (`<= 3` CSR buckets; `q6`/`q8` deferred); reuse `TensorEntry.reserved0` for binary compatibility | [ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md) |
| `D8` | PEARL-style pipelined small-draft execution | **Confirmed (2026-11-21)** | Enable selectively on high-latency targets (`Qwen3-14B` at `~95 ms/step`), leaving fast MoE + in-CB EAGLE-3 (`~19 ms/step`) un-pipelined | [ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md), *PEARL* ([arXiv:2408.11850](https://arxiv.org/abs/2408.11850)) |
| `ADR-0005` | Adaptive vs fixed 4-token L1 seed threshold | **Confirmed (2026-11-21)** | Keep fixed `match_len >= 4` through M4/M4.5; revisit adaptive thresholds after M5 NVMe N-gram store (`C6`) | [ADR-0005](../decisions/0005-dual-source-tree-merging.md) |
| `ADR-0018` | Bare-metal M4+ ARMv9 SME2 matrix & `ZT0`-LUT engine | **Confirmed (2026-11-21)** | Insert Milestone **M4.5** (`C11`–`C13`, `W15`, Spike `S7`): 90% pure Zig (`src/cpu/sme.zig`) + ~10% 3-way `.s` microkernels (`src/cpu/sme2_kernels.s`), single-owner `sme2_panel64`, `d8–d15` canary & M1–M3 fallback | [ADR-0018](../decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0001](../decisions/0001-language-and-toolchain-boundaries.md) |
| `D11` | Hardware entropy seeding for `T > 0` sampling | **Confirmed (2026-11-21)** | Per-request 64-bit initial seed from Apple Silicon Secure Enclave TRNG (`getentropy` when `seed == null`) feeding slot-local `Xoshiro256` + `seed_used` echo | [speculative-decoding.md](speculative-decoding.md#sampling-rng-and-hardware-entropy-seeding) |
| `M6.5` (opt.) | Zero-copy multimodal vision ingest | Future / Optional (post-M5) | `VideoToolbox` → `IOSurface` → `MTLTexture` → vision patch projector without CPU `memcpy` | [ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md), [ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md) |

## Detailed findings

### `R5`: `int8` KV tiles (`OneComponent8`)

- **Context**: [ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md) originally tracked `OneComponent8` KV tiles as an open question (`status.md` `R5`).
- **Apple CoreML / CoreVideo / Metal constraints**:
  - Apple's documentation for [`MLMultiArray.init(pixelBuffer:shape:)`](https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:)) explicitly states that float16 zero-copy IOSurface-backed arrays require `kCVPixelFormatType_OneComponent16Half` (`MLMultiArrayDataType.float16`).
  - `kCVPixelFormatType_OneComponent8` stores unsigned 8-bit integer texels (`0..255`), mapped in Metal to `MTLPixelFormatR8Unorm` (`[0.0, 1.0]`) or `MTLPixelFormatR8Uint` / `MTLPixelFormatR8Sint`. Without per-group scales stored alongside the surface, raw 8-bit normalized/integer pixel buffers cannot represent heterogeneous channel outliers accurately on the ANE or GPU.
- **KV quantization quality evidence**:
  - **KIVI** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) established that LLM Keys concentrate extreme magnitude outliers along fixed *channel* dimensions while Values concentrate outliers per *token*, requiring separate scaling metadata.
  - **`mlx-lm`** ([ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)) and **`llama.cpp`** ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) achieve near-lossless 8-bit KV compression (`mlx-lm` `--kv-bits 8 --kv-group-size 64`: mean KL `~0.0002 nats`, `wikitext-2-raw` `ΔPPL ~ +0.005`; `llama.cpp` `q8_0`: `ΔPPL < +0.004`) specifically because both formats store explicit per-block `scale` (and optional `zero`/`bias`) metadata alongside 8-bit payloads rather than bare `OneComponent8` pixels.
- **Confirmed resolution (2026-11-21)**: Decline `OneComponent8` for ANE↔GPU sharing and keep zero-copy `KvTile`s strictly on `OneComponent16Half` / `RGBA16Float` (measured at `85.61 GB/s` in [results.md](../benchmarks/results.md), including M6 `512`-token full + `128`-token sliding-window circular ring pools). If future 32K+ context workloads on 24 GB Macs require 50% KV compression for retired chunks, compress retired chunks GPU-side into a group-64 `q8` `MTLBuffer` layout rather than `OneComponent8` IOSurfaces, reserving radical KV compression (`890 B/tok` FP4) for Tier 3 `R7` ([ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)).

### `C8`: kqueue HTTP / SSE server API shape

- **Ecosystem survey**:
  - [`vLLM` OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and [`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) expose `/v1/chat/completions`, `/v1/responses`, `/v1/messages` (Anthropic Messages compatibility), and `/v1/models`.
  - For reasoning models such as `Qwen3` (`<think>...</think>`) and `DeepSeek-R1`, both `vLLM` (`--reasoning-parser`) and `llama-server` (`--reasoning-format`) separate thinking tokens from final assistant text into `message.reasoning_content` (non-streaming) and `choices[0].delta.reasoning_content` (SSE streaming) on `/v1/chat/completions`, and into typed `reasoning` output items on `/v1/responses`.
  - Supporting stateful `previous_response_id` on `/v1/responses` requires server-side conversation persistence, whereas stateless `/v1/chat/completions` passes full message history per turn—directly maximizing prefix hits in aminfer's L1 suffix automaton (`D1`, `0.04 us` lookup).
- **Confirmed resolution (2026-11-21)**: Make stateless **`POST /v1/chat/completions`** (SSE `chat.completion.chunk`, tool calls, `<think>` extraction into `delta.reasoning_content`, and optional `"seed"` with `seed_used` echo), **`GET /v1/models`**, and **`GET /health`** the M5 core contract ([runtime-pipeline.md](runtime-pipeline.md#request-lifecycle)). Optionally expose **`POST /v1/responses`** (stateless request translation without a persistent session database, matching `llama-server`) and **`POST /v1/messages`** as thin parsers over the exact same `kqueue` SSE token emitter.

### `A2`: ANE cross-layer MoE expert predictor architecture

- **External systems & local hardware boundary**:
  - **Fate** ([arXiv:2502.12224](https://arxiv.org/abs/2502.12224)) demonstrates on edge MoE offloading that reusing gate inputs and hidden states from adjacent layers predicts upcoming routed experts accurately without heavy auxiliary networks or extra GPU overhead, reaching up to ~99% cache hit rates when paired with shallow-favoring caching.
  - **Pre-gated MoE** ([arXiv:2308.12066](https://arxiv.org/abs/2308.12066)) and **ProMoE** ([arXiv:2410.22134](https://arxiv.org/abs/2410.22134)) show that hiding slow storage transfers requires `>= 1–2` layers of lookahead lead time.
  - **Spike S4** ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md), [results.md](../benchmarks/results.md)) measured `574.64 us` for a synchronous `GPU ↔ IOSurface ↔ ANE` round trip and showed that streaming `40 MiB` (`50.00 GB/s`) on ANE reduces concurrent GPU `gemv_q3` bandwidth by `30.8%` on the shared 120 GB/s M4 bus.
- **Recommendation**: Keep the ANE predictor strictly inside ANE on-chip SRAM (`< 2–4 MiB`, per [hardware.md](hardware.md)) by projecting current hidden states through concatenated **target `mlp.gate` router weights for `lookahead = 2..3` layers** (`3 × 128 × 2048 × 2 B = 1.5 MiB` for `Qwen3-30B-A3B`) or a batched lookahead schedule every `2–3` layers, amortized against `> 1 ms` NVMe cold-expert reads (`C3`/`C4`). Escalate to a small trained 2-layer bottleneck MLP only if `lookahead >= 2` Recall@16 proves insufficient in M4 measurements.

### `A3`: CoreML static shapes for growing KV length

- **CoreML ANE findings**:
  - Apple's [`coremltools` Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) guide documents `ct.EnumeratedShapes` (capped at `128` explicit shapes so CoreML/ANE pre-optimizes fixed graphs) vs unbounded ranges that risk CPU/GPU fallback; [`coremltools` Multifunction Models](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html) shares deduplicated weights across separate `mlprogram` entrypoints (macOS 15+) for distinct tasks/adapters, not per-step KV cache growth.
  - Recompilation (`ANECCompile`) on unseen dynamic shapes incurs hundreds of milliseconds to seconds, which is unacceptable during inference.
- **Recommendation**: In Spike S6, prefer splitting hybrid prefill (`ADR-0010`) so ANE compiles **a single static `[512, hidden]` graph** for QKV + O projections while GPU Split-KV Attention (`G4`, already measured at `85.61 GB/s` over `RGBA16Float` IOSurface `KvTile`s) handles growing historical KV lengths, comparing it head-to-head against power-of-two `EnumeratedShapes` KV buckets (`512, 1024, 2048, 4096`).

### `ADR-0016` Open item: `bf16` vs `f16` inter-layer residual stream

- **Numerical evidence**:
  - *Massive Activations in Large Language Models* ([arXiv:2402.17762](https://arxiv.org/abs/2402.17762)) proved that sparse fixed feature channels emerge in FFN/GLU layers with magnitudes up to `100,000×` typical activations and persist through the residual stream across layers.
  - Because IEEE `FP16` maxes out at `±65,504` (5 exponent bits), architectures lacking per-head QK-Norm or employing high RMSNorm gains (such as `Gemma 2/3` and certain `Llama` / `Qwen2.5` configurations) can overflow to `Inf`/`NaN` when their residual stream runs in `FP16`, whereas `BF16` preserves `FP32`'s `~3.4e38` range (8 exponent bits).
- **Confirmed resolution (2026-11-21)**: Keep `f16` activations as the default for QK-Norm architectures (`Qwen3`, verified end-to-end across M2–M3), and support an opt-in native `bf16` trunk residual buffer mode (`bfloat4` on Metal 4 per [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md) and `bfmopa` on CPU SME2 per [ADR-0018](../decisions/0018-bare-metal-armv9-sme2-engine.md)) for architectures without QK-Norm, narrowing `fp32 → fp16` zero-copy strictly at `RGBA16Float` IOSurface `KvTile` and CoreML ANE boundaries.

### Remaining M3–M5 open tuning items (`ADR-0017`, `D8`, `ADR-0005`)

- **`ADR-0017` (`G13` initial scheme set, confirmed 2026-11-21)**: Lock initial M5 implementation to `{q2_affine, q3_affine, q4_affine, bf16, f16}` (≤ 3 active MoE CSR buckets per layer; `q6`/`q8` deferred unless benchmark-justified), storing single-owner `TensorLayout` and `QuantParamKind` in `TensorEntry.reserved0` so existing `.amf` v0 files stay 100% binary compatible.
- **`D8` (PEARL pipelined small drafting, confirmed 2026-11-21)**: *PEARL* ([arXiv:2408.11850](https://arxiv.org/abs/2408.11850)) and *P-EAGLE* ([arXiv:2602.01469](https://arxiv.org/abs/2602.01469)) reduce draft-target mutual waiting, while M3 Spike S4 measured `~575 us` ANE round-trip handoff and `30.8%` GPU bandwidth contention; therefore apply PEARL-style overlapped small drafting (`D4`) selectively to slower high-latency targets (`Qwen3-14B` at `~95 ms/step`), leaving fast MoE + in-CB EAGLE-3 (`~19 ms/step`) un-pipelined ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)).
- **ADR-0005 (L1 seed threshold, confirmed 2026-11-21)**: Keep
  `SEED_MIN_MATCH_LEN = 4` fixed until M5 NVMe N-gram storage (`C6`) provides
  broader repository context for ablation.
- **ADR-0018 & `D11` (confirmed 2026-11-21)**:
  - Insert Milestone **M4.5** ([ADR-0018](../decisions/0018-bare-metal-armv9-sme2-engine.md))
    implementing 90% pure Zig (`src/cpu/sme.zig`) + ~10% 3-way ARMv9.2 `.s`
    microkernels (`src/cpu/sme2_kernels.s`: `M=1..4` multi-vector `ZT0-LUT + Dot`,
    `M=5..16` `1×4` `ZA`, `M>=32` `2×2` `ZA`) with `ldnt1b` cold-expert streaming,
    single-owner `sme2_panel64` layout in `TensorEntry.reserved0`, pre-`smstart`
    `d8–d15` canary verification, and Spike `S7` placement gating.
  - Adopt Apple Silicon Secure Enclave TRNG (`getentropy`) initial 64-bit
    per-request seeding (`seed: ?u64 = null`) feeding slot-local `Xoshiro256`
    PRNGs with `seed_used` echo (`D11`,
    [speculative-decoding.md](speculative-decoding.md#sampling-rng-and-hardware-entropy-seeding)).
- **Post-M5 optional multimodal blueprint (`M6.5`, future / non-binding)**:
  extend the zero-copy `IOSurface` substrate
  ([ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md)) from KV
  tiles to `VideoToolbox` hardware decode → `CVPixelBuffer` (`IOSurface`) →
  `MTLTexture` → Metal/ANE vision patch projection when onboarding multimodal
  checkpoints in the `Qwen3.8` family ([ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)).

## Sources

- Apple Developer Documentation, *MLMultiArray.init(pixelBuffer:shape:)* (`kCVPixelFormatType_OneComponent16Half`): <https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:)>
- Apple CoreML Tools, *Flexible Input Shapes (`EnumeratedShapes`)*: <https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html>
- Apple CoreML Tools, *Multifunction Models*: <https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html>
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (ICML 2024): <https://arxiv.org/abs/2402.02750>
- Sun et al., *Massive Activations in Large Language Models* (COLM 2024): <https://arxiv.org/abs/2402.17762>
- Liu et al., *PEARL: Parallel Speculative Decoding with Adaptive Draft Length* (ICLR 2025): <https://arxiv.org/abs/2408.11850>
- Hui et al., *P-EAGLE: Parallel-Drafting EAGLE with Scalable Training* (2026): <https://arxiv.org/abs/2602.01469>
- *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference*: <https://arxiv.org/abs/2308.12066>
- *ProMoE: Fast MoE-based LLM Serving using Proactive Caching*: <https://arxiv.org/abs/2410.22134>
- *Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate* (WWW 2025): <https://arxiv.org/abs/2502.12224>
- vLLM Documentation, *OpenAI-Compatible Server* (`/v1/responses`, `/v1/chat/completions`, `--reasoning-parser`): <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- `llama.cpp`, *`llama-server` README* (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `reasoning_content`, `-ctk q8_0`): <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- `ml-explore/mlx-lm` (`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`): <https://github.com/ml-explore/mlx-lm>
