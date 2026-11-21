# ADR-0016: Native FP16 / BF16 execution and multi-shard safetensors import (M2.5)

- Status: Accepted
- Date: 2026-11-20
- Source: user request and milestone confirmation (post-M2 session)

## Context

In M1 and M2, aminfer established its 16 KiB-aligned `.amf` v0 container,
compile-time-specialized 3-bit and 4-bit MLX-affine dequantization kernels
(`G1`, `G2`), deduplicated MoE SwiGLU (`G3`), IOSurface `RGBA16Float` KV tiles
(`R3`, `R4`, `G4`), and end-to-end greedy decode (`W6`) verified against
synthetic `Qwen3-30B-A3B` (3-bit) and `Qwen3-14B` (4-bit) bundles.

Two capabilities are needed before M3 (speculative decoding) to move from
synthetic single-file test bundles to real Hugging Face checkpoints and to make
aminfer a general-purpose precision substrate across Apple Silicon tiers:

1. **Multi-shard Hugging Face / MLX safetensors streaming import.** Production
   checkpoints (`mlx-community/Qwen3-30B-A3B-3bit` across 3 shards,
   `mlx-community/Qwen3-14B-4bit` across 2 shards, and multi-shard unquantized
   models) use `model.safetensors.index.json`. Currently,
   `tools/src/aminfer_tools/amf.py` accepts only a single `.safetensors` file
   and materializes packed blobs in host memory before writing.
2. **Standard unquantized and mixed-precision `FP16` / `BF16` support.**
   Upstream open-weight releases (including `Qwen/Qwen3-0.6B`, `Qwen3-8B`,
   `Qwen3-14B`, `Qwen3-30B-A3B`, `Qwen3.8`, and `DeepSeek` checkpoints, as well
   as EAGLE / MTP heads and small draft models in M3/M4) publish weights in
   `BF16` or `FP16`. Blindly casting unquantized `BF16` weights to IEEE 754
   `FP16` offline risks overflow beyond `±65,504` or subnormal underflow, while
   supporting only 3/4-bit affine quantization prevents losslessly evaluating
   reference checkpoints, draft models (`D2`–`D4`), or mixed-precision layers
   (such as sensitive norms, routers, embeddings, and shared experts).

At the same time, two hardware boundaries on Apple Silicon (`Apple9` M4, macOS
26.0+) constrain how `BF16` interacts with zero-copy KV sharing:

- Metal 4 (`-std=metal4.0 -mmacosx-version-min=26.0`) natively supports `bfloat`,
  `bfloat4`, `simdgroup_bfloat8x8`, `half`, `half4`, and `simdgroup_half8x8`,
  all accumulating into `simdgroup_float8x8` (`FP32`).
- Neither `MTLPixelFormat` nor CoreML's zero-copy `MLMultiArray(pixelBuffer:shape:)`
  ([ADR-0004](0004-kv-cache-iosurface-texture-sharing.md)) provides a native
  `BF16` IOSurface texture/pixel format (`OneComponent16Half` / `RGBA16Float`
  is `FP16`).

## Decision

Schedule **Milestone M2.5** immediately before M3 with the following design:

1. **Add `DType.bf16 = 4` to `.amf` v0 (`W12`).**
   - Extend `amf.DType` in `src/format/amf.zig` and `tools/src/aminfer_tools/amf.py`
     with `bf16 = 4` (`quant_bits = 0` matching unquantized `f16`/`f32`,
     `group_size = 0`, `params_bytes = 0`, 2 bytes per element),
     keeping the 128-byte `Header`, 128-byte `TensorEntry`, `FORMAT_VERSION = 1`,
     and 16 KiB blob alignment unchanged.
   - `aminfer-tools pack-amf` preserves unquantized `BF16` tensors bit-exact as
     `DType.bf16` by default (`--dtype auto|f16|bf16`), while continuing to
     pack MLX affine quantized `.weight` / `.scales` / `.biases` triples into
     `q3_affine` or `q4_affine` with `half2(scale, bias)` group parameters.
2. **Stream multi-shard `model.safetensors.index.json` checkpoints (`W12`).**
   - `aminfer-tools pack-amf` accepts either a single `.safetensors` file, an
     `index.json` file, or a checkpoint directory.
   - Pass 1 parses only the JSON headers of each shard (kilobytes) to plan the
     complete `.amf` tensor directory and deterministic 16 KiB-aligned offsets;
     Pass 2 streams raw tensor byte ranges directly from the owning shard file(s)
     into the `.amf` output with bounded host memory and incremental CRC-32
     calculation, never loading a multi-gigabyte model into Python RAM.
3. **Specialize native Metal 4 `FP16` and `BF16` kernels (`G12`).**
   - Extend the GPU kernel suite under [ADR-0014](0014-compile-time-specialization.md)
     (Zig `comptime` + Metal function constants) with native `FP16` and `BF16`
     weight paths:
     - **Decode GEMV (`M = 1`)**: 128-bit coalesced `half4`/`half8` and
       `bfloat4`/`bfloat8` weight loads with `FP32` FMA accumulation and
       simdgroup reductions.
     - **Batched GEMM (`M > 1`, prefill & tree verification) and MoE SwiGLU**:
       `simdgroup_half8x8` and `simdgroup_bfloat8x8` matrix multiply-accumulate
       into `simdgroup_float8x8` (`FP32`), reusing M2's active-expert CSR
       deduplication (`G3`).
     - **Embeddings, RMSNorm, and `lm_head`**: per-tensor `f16` / `bf16` support
       (including `tie_word_embeddings = true`, where the embedding table backs
       both token lookup and final `lm_head` GEMV/GEMM without duplicating wired
       pages).
4. **Generalize `ModelConfig` and preserve the `RGBA16Float` IOSurface KV boundary (`W13`).**
   - Replace the mandatory quantized assumption in `src/model/qwen3.zig` with a
     weight precision descriptor supporting unquantized Hugging Face
     `config.json` files (`quantization_config` absent; `torch_dtype` `"bfloat16"`
     or `"float16"`) as well as mixed-precision `.amf` containers where
     individual tensors carry their own `amf.DType`.
   - Keep `KvTile` in `IOSurface` `OneComponent16Half` viewed as token-major
     `RGBA16Float` ([ADR-0004](0004-kv-cache-iosurface-texture-sharing.md), `R3`,
     `R4`). Post-QK-Norm, post-RoPE K/V vectors have bounded unit-scale
     magnitude well within `FP16` range and cast from `FP32` registers to
     `half4` inside `ami_qknorm_rope_kv_write`. This preserves `85.6 GB/s`
     FlashDecoding (`G4`) and 100% zero-copy compatibility with M4's ANE hybrid
     wavefront prefill ([ADR-0010](0010-hybrid-wavefront-prefill.md)).
5. **Hardware-honest residency tiers for unquantized `FP16` / `BF16`.**
   - **24 GB M4 baseline**: runs unquantized `FP16`/`BF16` models up to the ~8B
     class fully wired in DRAM (`Qwen/Qwen3-0.6B` ~1.2 GiB weights; `4B` ~7.6 GiB;
     `8B` ~15.3 GiB within the 18.18–21.5 GiB wired budget). `Qwen/Qwen3-0.6B`
     serves both as M2.5's real-checkpoint `BF16` verification target and as the
     same-family small draft model candidate for `D4`
     ([ADR-0013](0013-eagle-tree-and-head-formats.md)).
   - **36 GB / 64 GB / 128 GB+ Macs**: run full-wired `Qwen3-14B` (`BF16`,
     ~27.6 GiB) and `Qwen3-30B-A3B` (`BF16`, ~56.9 GiB total, ~5.7 GiB/token
     active). On the 24 GB M4 baseline, `Budget.validate`
     ([ADR-0012](0012-wired-memory-budget-headroom.md)) cleanly rejects
     full-wired 27.6–56.9 GiB plans rather than swapping; 24 GB M4 serves
     `Qwen3-14B-4bit` (~7.75 GiB) and `Qwen3-30B-A3B-3bit` (~12.46 GiB) fully
     wired, plus NVMe cold-expert streaming in M4.

## Consequences

- Real multi-shard MLX quantized checkpoints (`Qwen3-30B-A3B-3bit`,
  `Qwen3-14B-4bit`) and standard Hugging Face `BF16`/`FP16` checkpoints
  (`Qwen3-0.6B`) convert into `.amf` and run through the exact same `aminfer`
  runtime before M3 speculative decoding begins.
- M3 (`D2`, `D3`) and M4 (`D4`) draft heads / small draft models can run in
  native `BF16` or `FP16` on the GPU without forced 3/4-bit quantization loss.
- No `.amf` header layout break: `DType.bf16 = 4` is an additive enum variant on
  the existing 1-byte `TensorEntry.dtype` field.

## Evidence

- Verified on the baseline machine (`Apple9` M4, macOS 27.2, Metal Toolchain
  `27A266a`): `xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -Wall -Wextra -Werror`
  compiles `bfloat`, `bfloat4`, `simdgroup_bfloat8x8`, and `simdgroup_half8x8`
  `simdgroup_multiply_accumulate` into `simdgroup_float8x8` cleanly with zero
  warnings.
- Local reference checkpoints downloaded and verified on external NVMe
  (`/Volumes/Akari/models/`), revealing three real-checkpoint layout details
  that M2.5's importer and loader will handle directly:
  - `Qwen/Qwen3-0.6B` (`1.40 GiB`, 1 shard, `311` tensors, all `BF16`,
    `tie_word_embeddings: true` with both `embed_tokens.weight` and
    `lm_head.weight` present, `qDim = 2048 != hidden_size = 1024`).
  - `mlx-community/Qwen3-14B-4bit` (`7.74 GiB`, 2 shards, `1007` tensors,
    `SHA-256` verified): MLX quantizes `embed_tokens` (`U32 [151936, 640]`,
    4-bit affine group-64) as well as linear projections, while norms are
    `BF16`.
  - `mlx-community/Qwen3-30B-A3B-3bit` (`12.44 GiB`, 3 shards, `1351` tensors,
    `SHA-256` verified): MLX quantizes `embed_tokens` and the MoE router
    `mlp.gate` (3-bit affine group-64) and stores routed experts as stacked 3D
    `mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` tensors
    (`[128, N, K_packed]`), which `pack-amf` slices/packs into per-expert `.amf`
    blobs while preserving quantized `embed_tokens` and `mlp.gate` routers
    natively.
- **M2.5 real-checkpoint verification (2026-11-20–2026-11-21 on Apple M4 24 GiB,
  commits `2a48c66..1893ab6`, full telemetry in
  [../benchmarks/results.md](../benchmarks/results.md)):**
  - Streaming multi-shard conversion (`aminfer-tools pack-amf --dtype auto`)
    converted all three checkpoints with bounded host RAM (**35–127 MB** peak
    Python RSS) and verified via `aminfer verify-amf` (**21–25 MB** RSS).
  - Native `FP16`/`BF16` Metal 4 kernels reached **88.25 GB/s** (`gemv_bf16`)
    and **86.76 GB/s** (`gemv_f16`) at `1×4096×4096`, and **2.51–2.69 TFLOP/s**
    on `M = 64` `simdgroup_bfloat8x8` / `simdgroup_half8x8` GEMM.
  - All three real checkpoints produced coherent bilingual (English & Chinese)
    greedy output with `swapouts_delta = 0` over the `f16` residual stream and
    `RGBA16Float` IOSurface `KvTile` boundary: `Qwen3-0.6B` BF16 28L (**47.88
    tok/s** at `seq=512`, **62.98–66.17 tok/s** on 32-token decode),
    `Qwen3-14B-4bit` 40L (**9.52 tok/s** at `seq=512`, **75.76–84.00 GB/s**),
    and `Qwen3-30B-A3B-3bit` 48L (**51.86 tok/s** at `seq=512`, **71.65 GB/s**).

## Open items

- **Confirmed inter-layer residual stream policy (resolved 2026-11-21):**
  - Keep `f16` inter-layer residual buffers as the default for QK-Norm
    architectures (`Qwen3` dense and MoE, verified lossless and coherent across
    M2.5–M3).
  - Support an **opt-in native `bf16` trunk residual stream mode** on Metal 4
    (`bfloat4`) and M4.5 CPU SME2 (`bfmopa`,
    [ADR-0018](0018-bare-metal-armv9-sme2-engine.md)) for architectures without
    QK-Norm or with high RMSNorm gains susceptible to massive activations
    exceeding `±65,504` (Sun et al., *Massive Activations in Large Language
    Models*, `arXiv:2402.17762`; see
    [open-decisions-research.md](../architecture/open-decisions-research.md#adr-0016-open-item-bf16-vs-f16-inter-layer-residual-stream)).
  - **Zero-copy IOSurface precision boundary
    ([ADR-0004](0004-kv-cache-iosurface-texture-sharing.md)):** even when trunk
    residuals execute in `bf16`, `fp32` accumulators narrow zero-copy to `fp16`
    (`half4`) strictly at `RGBA16Float` IOSurface `KvTile` and CoreML ANE
    boundaries, where post-norm / post-RoPE vectors are magnitude-bounded.
