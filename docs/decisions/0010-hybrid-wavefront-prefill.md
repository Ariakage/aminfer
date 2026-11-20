# ADR-0010: Hybrid ANE / GPU wavefront prefill

- Status: Accepted
- Date: 2026-11-20
- Source: proposal discussed and accepted by the user (init session); replaces
  whole-model ANE chunked prefill from the reference architecture

## Context

The reference architecture ran chunked prefill (512-token steps) of the whole
target model on the ANE. For the MoE target this does not work: the ANE needs
its own CoreML copy of the weights (Qwen3-30B-A3B at 3 bits is about
12.4 GiB, so two copies do not fit in 24 GB), and a static ANE graph has no
efficient dynamic per-token expert dispatch.

## Decision

1. Prompts longer than 512 tokens are prefilled in 512-token chunks. For each
   layer and chunk, the ANE runs the attention block (QKV projection,
   attention, output projection) and writes the chunk's K and V into IOSurface
   KV tiles ([ADR-0004](0004-kv-cache-iosurface-texture-sharing.md)); the GPU
   runs the FFN or MoE block.
2. Two chunks are in flight, the earlier one a layer ahead, so ANE attention
   and GPU FFN overlap
   ([../architecture/runtime-pipeline.md](../architecture/runtime-pipeline.md#hybrid-wavefront-prefill)).
3. The ANE holds only attention-block weights (about 0.9B parameters for
   Qwen3-30B-A3B, about 0.42 GiB at 4 bits palettized).
4. The GPU-only prefill remains for short prompts and as the fallback and
   baseline.

## Consequences

- Each layer of each chunk crosses ANE → GPU → ANE once; hand-off latency
  matters (spike S4).
- The CoreML attention graphs and their export tooling join `tools/`.

## Open items

- Spike S6: CoreML compilation of attention blocks, static shapes for a
  growing KV length (shape buckets or multifunction models), fp16 numerics,
  throughput versus the GPU (status row A3).
