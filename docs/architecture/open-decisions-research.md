# Open-decision research and recommendations

> Non-binding technical survey and engineering recommendations for capabilities
> still marked **Undecided**, **Draft**, or **Open** in [../status.md](../status.md)
> and the [ADRs](../decisions/README.md). No item listed here is binding until
> confirmed by the user or closed by its corresponding spike in
> [../process.md](../process.md#spikes).

## Summary matrix

| ID | Open question | Current status | Recommended direction | Key external & local basis |
| --- | --- | --- | --- | --- |
| `R5` | `int8` KV tiles (`OneComponent8`) | Undecided | Decline `OneComponent8` for ANE↔GPU; keep `OneComponent16Half`; optionally compress older GPU-resident KV in `MTLBuffer` (`q8` group-64 / per-head) | Apple `init(pixelBuffer:shape:)` requires `OneComponent16Half`; KIVI ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)); `mlx-lm` `8-bit g=64` |
| `C8` | kqueue HTTP/SSE server API shape | Undecided | Required stateless `POST /v1/chat/completions` (with `delta.reasoning_content`) + `GET /v1/models` + `GET /health`; optional stateless wrappers for `/v1/responses` & `/v1/messages` | `vLLM` OpenAI server & `llama-server` (`reasoning_content`, stateless `/v1/responses`, `/v1/messages`) |
| `A2` | ANE MoE expert predictor architecture | Undecided | Start with `lookahead = 2..3` cross-layer `mlp.gate` weights SRAM-resident (`< 2 MiB`); add tiny bottleneck MLP only if recall falls short | *Fate* ([arXiv:2502.12224](https://arxiv.org/abs/2502.12224)), *ProMoE* ([arXiv:2410.22134](https://arxiv.org/abs/2410.22134)), *Pre-gated MoE* ([arXiv:2308.12066](https://arxiv.org/abs/2308.12066)); Spike S4 (`-30.8%` DRAM contention) |
| `A3` | Static CoreML ANE shapes for growing KV | Undecided (Spike S6) | Run static `[512, hidden]` QKV + output projections on ANE; run split-KV attention over full growing history on GPU (benchmark vs `EnumeratedShapes` buckets in S6) | `coremltools` `EnumeratedShapes` (`<= 128` shapes) & `Multifunction` guides; G4 GPU `RGBA16Float` at `85.6 GB/s` |
| `ADR-0016` | Optional `bf16` inter-layer residual buffer | Open | Keep `f16` default for QK-Norm models (`Qwen3`); add opt-in `bf16` residual buffer before onboarding non-QK-Norm families (`Gemma`, `Llama`, `DeepSeek` variants) | *Massive Activations* ([arXiv:2402.17762](https://arxiv.org/abs/2402.17762)); `FP16` `±65,504` overflow; M2.5 `bfloat4` Metal 4 support |
| `ADR-0017` | Initial `G13` GSQ+RCO affine bit-widths | Open | Launch with `{q2_affine, q3_affine, q4_affine, bf16, f16}` (defer `q6`/`q8` unless benchmarks demand them) | Keep `comptime` kernel variants & CSR bucket dispatch overhead small (`<= 3` active buckets) |
| `D8` | PEARL-style pipelined small-draft execution | Draft | Enable only when pairing a small draft model (`D4`, `Qwen3-0.6B`) with slow dense targets (`Qwen3-14B` at `~95 ms/step`), not `Qwen3-30B-A3B` (`~19 ms/step`) | *PEARL* ([arXiv:2408.11850](https://arxiv.org/abs/2408.11850)), *P-EAGLE* ([arXiv:2602.01469](https://arxiv.org/abs/2602.01469)); Spike S4 (`574.6 us` ANE handoff) |
| `ADR-0005` | Adaptive vs fixed 4-token L1 seed threshold | Open | Retain fixed `match_len >= 4` through M4; revisit after M5 NVMe N-gram store (`C6`) lands | Avoids false-positive 1–3 token branches crowding `D7` tree budgets |

## Detailed findings

### `R5`: `int8` KV tiles (`OneComponent8`)

- **Context**: [ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md) tracks `OneComponent8` KV tiles as Undecided (`status.md` `R5`).
- **Apple CoreML / CoreVideo / Metal constraints**:
  - Apple's documentation for [`MLMultiArray.init(pixelBuffer:shape:)`](https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:)) explicitly states that float16 zero-copy IOSurface-backed arrays require `kCVPixelFormatType_OneComponent16Half` (`MLMultiArrayDataType.float16`).
  - `kCVPixelFormatType_OneComponent8` stores unsigned 8-bit integer texels (`0..255`), mapped in Metal to `MTLPixelFormatR8Unorm` (`[0.0, 1.0]`) or `MTLPixelFormatR8Uint` / `MTLPixelFormatR8Sint`. Without per-group scales stored alongside the surface, raw 8-bit normalized/integer pixel buffers cannot represent heterogeneous channel outliers accurately on the ANE or GPU.
- **KV quantization quality evidence**:
  - **KIVI** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) established that LLM Keys concentrate extreme magnitude outliers along fixed *channel* dimensions while Values concentrate outliers per *token*, requiring separate scaling metadata.
  - **`mlx-lm`** ([ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)) and **`llama.cpp`** ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) achieve near-lossless 8-bit KV compression (`mlx-lm` `--kv-bits 8 --kv-group-size 64`: mean KL `~0.0002 nats`, `wikitext-2-raw` `ΔPPL ~ +0.005`; `llama.cpp` `q8_0`: `ΔPPL < +0.004`) specifically because both formats store explicit per-block `scale` (and optional `zero`/`bias`) metadata alongside 8-bit payloads rather than bare `OneComponent8` pixels.
- **Recommendation**: Keep ANE↔GPU zero-copy `KvTile`s strictly on `OneComponent16Half` / `RGBA16Float` (already measured at `85.61 GB/s` in [results.md](../benchmarks/results.md)). If future 32K+ context workloads on 24 GB Macs require 50% KV compression for cold/older GPU-committed chunks, compress retired chunks GPU-side into a group-64 `q8` `MTLBuffer` layout rather than adding `OneComponent8` IOSurface views, reserving radical KV compression (`890 B/tok` FP4) for Tier 3 `R7` ([ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)).

### `C8`: kqueue HTTP / SSE server API shape

- **Ecosystem survey**:
  - [`vLLM` OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and [`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) expose `/v1/chat/completions`, `/v1/responses`, `/v1/messages` (Anthropic Messages compatibility), and `/v1/models`.
  - For reasoning models such as `Qwen3` (`<think>...</think>`) and `DeepSeek-R1`, both `vLLM` (`--reasoning-parser`) and `llama-server` (`--reasoning-format`) separate thinking tokens from final assistant text into `message.reasoning_content` (non-streaming) and `choices[0].delta.reasoning_content` (SSE streaming) on `/v1/chat/completions`, and into typed `reasoning` output items on `/v1/responses`.
  - Supporting stateful `previous_response_id` on `/v1/responses` requires server-side conversation persistence, whereas stateless `/v1/chat/completions` passes full message history per turn—directly maximizing prefix hits in aminfer's L1 suffix automaton (`D1`, `0.04 us` lookup).
- **Recommendation**: Make stateless **`POST /v1/chat/completions`** (SSE `chat.completion.chunk`, tool calls, and `<think>` extraction into `delta.reasoning_content`), **`GET /v1/models`**, and **`GET /health`** the M5 core contract. Optionally expose **`POST /v1/responses`** (stateless request translation without a persistent session database, matching `llama-server`) and **`POST /v1/messages`** as thin parsers over the exact same `kqueue` SSE token emitter.

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
- **Recommendation**: Keep `f16` activations as the default for QK-Norm architectures (`Qwen3`, already verified end-to-end across M2–M3), and add an opt-in `bf16` residual buffer mode (`bfloat4` supported natively on Metal 4 per [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)) before M6/Tier 2–3 onboarding of architectures without QK-Norm, casting post-norm/RoPE K/V safely to `RGBA16Float` `KvTile`s.

### Remaining M3–M5 open tuning items (`ADR-0017`, `D8`, `ADR-0005`)

- **`ADR-0017` (`G13` initial scheme set)**: Restrict initial M4/M5 implementation to `{q2_affine, q3_affine, q4_affine, bf16, f16}` (≤ 3 active MoE CSR buckets per layer), matching our existing M1–M2.5 Metal 4 specialization suite while capturing the `Qwen3-30B-A3B` / `Qwen3.8-Flash-Next` size-accuracy frontier.
- **`D8` (PEARL pipelined small drafting)**: *PEARL* ([arXiv:2408.11850](https://arxiv.org/abs/2408.11850)) and *P-EAGLE* ([arXiv:2602.01469](https://arxiv.org/abs/2602.01469)) reduce draft-target mutual waiting, but M3 Spike S4 measured `~575 us` ANE round-trip handoff and `30.8%` GPU bandwidth contention; therefore apply PEARL-style overlapped ANE small drafting (`D4`) selectively to slower large dense targets (`Qwen3-14B` at `~95 ms/step`), leaving fast MoE + in-CB EAGLE-3 (`~19 ms/step`) un-pipelined.
- **`ADR-0005` (L1 seed threshold)**: Keep `SEED_MIN_MATCH_LEN = 4` fixed until M5 NVMe N-gram storage (`C6`) provides broader repository context for ablation.

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
