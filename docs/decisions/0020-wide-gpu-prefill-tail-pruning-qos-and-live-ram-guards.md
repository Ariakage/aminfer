# ADR-0020: Configurable wide-chunk GPU prefill, lossless tail pruning, serving QoS, and live non-anonymous RAM admission

- Status: Accepted
- Date: 2026-11-22
- Source: user roadmap confirmation following post-M6.5 reference engine audit

## Context

With Milestones M0–M6.5 implemented on the 24 GiB Apple M4 baseline
([../benchmarks/results.md](../benchmarks/results.md)), five concrete execution
and operational gaps remain before scaling out to Tier 3 cluster targets in M7:

1. **Pure-GPU and `C14` slot prefill are still sliced through `verifyTree`'s
   `64`-token ancestor mask.** In `Qwen3Model.prefillChunk`,
   `GraphModel.prefillChunk`, `Qwen3BatchRuntime.prefillSlot`, and
   `GraphBatchRuntime.prefillSlot`, a `512`-token prefill chunk executes as 8
   sequential calls of `<= 64` rows through `verifyTree` (`u64` ancestor-mask
   cap `MAX_TREE_NODES = 64`) plus `commitTreeKv`, while `ami_dequant_gemm` uses
   `BM = 32, BN = 32, BK = 64` tiles. Across a `512`-token chunk, that repeats
   weight reads up to 8 times on the GPU path, evaluates masked upper-triangular
   KV positions inside each `64`-row block, and stages K/V through `tree_k` and
   `tree_v` prior to `KvTile` commit.
2. **Non-terminal prefill rows execute unused final-layer projections and
   full-vocabulary `lm_head`.** Intermediate prefill chunks (`chunk <
   final_chunk`) and non-final rows (`0 .. C - 2`) of the final chunk currently
   compute final layer `L - 1`'s attention output, `O_proj`, `MoE` / dense
   `SwiGLU` FFN, final `RMSNorm`, and `151,936+`-vocabulary `lm_head` logits
   even when only the next-token distribution after the last prompt token is
   needed.
3. **Unloading `lm_head` weights during prefill breaks `tie_word_embeddings`,
   `mlock`, and `C14` continuous batching.** Reference designs that temporarily
   unmap `lm_head` weights to fund wide prefill activation buffers fail when
   `embed_tokens` and `lm_head` share physical pages (`tie_word_embeddings`),
   introduce disk re-reads between prefill and decode (violating hot-path
   zero-I/O rules), and cannot run under `C14` continuous batching where active
   decode slots evaluate `lm_head` on every iteration alongside prefilling
   slots. Aliasing activation scratch regions rather than model weights captures
   the single-request memory reduction without violating any engine invariant.
4. **Long-context background prefill and multi-turn edits degrade serving tail
   latency.** Under `C14` continuous batching, an unthrottled `512`-token
   prefill chunk at `seq_pos >= 8,192` lengthens per-step GPU attention time and
   inflates streaming decode `TPOT p99` unless governed by a decode wall-time
   budget and depth-scaled chunk narrowing. Separately, `512`-token `KvTile`
   prefix boundaries (`R8`) require recomputing up to `511` intra-tile tokens
   when multi-turn ChatML clients edit, regenerate, or branch the final user
   turn.
5. **Static `sysctl` wire caps do not track live non-anonymous RAM availability
   or offline `.metallib` numerical regressions.** `ADR-0012` enforces
   deterministic static wire limits at startup, yet concurrent desktop processes
   may hold wired or anonymous pages; counting macOS `inactive_count` pages as
   freely available would force XNU to compress or swap anonymous memory when
   optional caches wire pages. In addition, offline `xcrun metal` / `metalfe` or
   OS driver updates can introduce silent `bf16` / `f16` / `q4` shader numerical
   corruption on machines other than the build host unless checked at startup.

The user confirmed inserting **Milestone M6.8** (`R9`, `G15`, `G16`, `C15` in
[../status.md](../status.md) and [../process.md](../process.md)) with explicit
feature toggles and defaults covering `P0-A`, `P0-B1`, `P0-B2`, `P1-A`, `P1-B`,
and `P2`.

## Decision

### 1. Configurable wide-chunk GPU prefill (`P0-A` -> `G15`, default ON)

- Decouple pure-GPU prefill (`Qwen3Model.prefillChunk`,
  `GraphModel.prefillChunk`) and `C14` batched slot prefill
  (`Qwen3BatchRuntime.prefillSlot`, `GraphBatchRuntime.prefillSlot`) from
  `verifyTree`'s `MAX_TREE_NODES = 64` (`u64` mask) limit and `tree_k`/`tree_v`
  staging.
- Add dedicated Metal 4 **FlashAttention prefill** kernels (`BQ = 32, BK = 32`
  online softmax with `15 KiB` threadgroup SRAM for `2x` occupancy and
  `MTL_SHADER_VALIDATION=1` compliance, `simdgroup` reductions, skipping strictly
  upper-triangular KV blocks where `kv_block_start > q_block_end` for causal GQA,
  sliding-window SWA, and `QSA` sparse attention) that write post-QK-Norm /
  post-RoPE K and V vectors directly into target `KvTile`s (`RGBA16Float`
  IOSurface textures, [ADR-0004](0004-kv-cache-iosurface-texture-sharing.md)).
- Pair FlashAttention prefill with wide-tile (`BM = 64, BN = 64`) prefill GEMM
  specialization (`f16`, `bf16`, `q2_affine`, `q3_affine`, `q4_affine`) and
  wide-batch (`512–1024` token) CSR-sorted MoE execution.
- Expose **`--gpu-prefill-mode wide|tree64`** (and matching Zig runtime config
  fields), **enabled by default (`wide`)**. Selecting `tree64` preserves the
  legacy `<= 64`-token `verifyTree` + `commitTreeKv` prefill path for A/B
  benchmarking and diagnostic isolation. Hybrid ANE/GPU wavefront prefill
  ([ADR-0010](0010-hybrid-wavefront-prefill.md)) remains unchanged when ANE
  attention bundles are explicitly configured.

### 2. Lossless `prompt_tail` final-layer compute pruning (`P0-B1` -> `G15`, always ON)

- Whenever full-prompt `logprobs` / prompt echo are not requested:
  - **Non-final prefill chunks (`chunk < final_chunk`)** terminate at final
    layer `L - 1` immediately after writing the chunk's `KvTile`, `GDN`
    recurrent state, and short-convolution state, skipping `L - 1` attention
    value output, `O_proj`, `MoE` / dense `SwiGLU` FFN, final `RMSNorm`, and
    full-vocabulary `lm_head`.
  - **Final prefill chunk (`chunk == final_chunk`, length `C`)** writes all `C`
    tokens' `KvTile` / `GDN` / `Conv` states at `L - 1`, then narrows the active
    query / hidden slice to `M = 1` (row `C - 1`, the final prompt token) so
    `L - 1` attention output, `O_proj`, `MoE` / `MLP`, final `RMSNorm`, and
    `lm_head` execute as `M = 1` GEMV rather than `M = C` GEMM.
- When prompt-wide logits are explicitly requested, the full `M = C` tail path
  remains available without altering kernel interfaces.

### 3. Single-request activation scratch aliasing (`P0-B2` -> `R9`)

- Never unload, unmap, or re-read `lm_head` weights between prefill and decode:
  `tie_word_embeddings`, `mlock` residency, and concurrent `C14` decode slots
  stay intact at all times.
- In single-request execution (`decode`, `spec-decode`, and single-stream
  benchmarks), alias the `P0-A` wide-prefill rolling activation scratch region
  with the speculative `verifyTree` scratch region inside `WiredArena`,
  reclaiming the smaller of the two workspaces (`~32–160 MiB` depending on model
  width and chunk size) with zero runtime allocation or page table churn.
- In `C14` multi-slot `serve` mode, keep per-pass GPU scratch serialized safely
  on the command stream without overlapping live per-slot state or weights.

### 4. Configurable serving QoS and ChatML boundary prefix caching (`P1-A` -> `C15`, defaults ON)

- **Wall-time `decode-share` QoS ledger (`--decode-share <0.0..1.0>`, default
  `0.50`, `0` = off)** and **depth-scaled background prefill chunk narrowing
  (`--adaptive-prefill-narrow on|off`, default `on`)**:
  - `BatchScheduler` records monotonic GPU wall time (`ns`) spent in prefill
    steps versus decode steps whenever both prefilling and decoding slots are
    active, scheduling decode steps first whenever the rolling decode wall-time
    share drops below `--decode-share`.
  - When `--adaptive-prefill-narrow` is `on` and at least one decode slot is
    active, background prefill chunks narrow from `512` tokens to `256` at
    `seq_pos >= 8,192` and `128` at `seq_pos >= 16,384` (falling back to full
    wide chunks whenever zero decode slots are active).
- **ChatML message-boundary `PrefixCache` checkpoints
  (`--message-boundary-cache on|off`, default `on`)**:
  - On multi-turn chat requests (`/v1/chat/completions`, `/v1/responses`,
    `/v1/messages`), `PrefixCache` (`R8`) records an additional copy-on-write
    tail `KvTile` + `GDN` snapshot checkpoint at the token offset immediately
    before the final user message, marked `evict_first` so canonical `512`-token
    tile-boundary entries retain higher cache priority when slots fill.

### 5. Live non-anonymous RAM admission and soft cache auto-sizing (`P1-B` -> `R9`)

- Extend `src/platform/vmstat.zig` and `src/mem/budget.zig`
  ([ADR-0012](0012-wired-memory-budget-headroom.md)) with a `HOST_VM_INFO64`
  live zero-swap pool query counting **only**:
  `free_count + speculative_count + purgeable_count + external_page_count`
  (explicitly excluding `inactive_count` / `internal_page_count` anonymous pages
  that require XNU compression or swap-out to reclaim).
- Keep required model weights, active KV/GDN arenas, and core workspaces
  governed deterministically by the static `ADR-0012` `Budget`. Automatically
  scale optional wired caches (`PrefixCache`, `PlePageCache`, `NgramCache`, and
  extra `ExpertSlotPool` slots above the single-pass minimum) down at startup so
  total new wiring stays within `live_zero_swap_bytes - live_headroom_bytes`,
  logging both static budget and live reclaimable headroom.
- Add opt-in **`--strict-live-ram`** to turn any remaining deficit against the
  live non-anonymous pool into an immediate pre-`mlock` admission error.

### 6. Startup `.metallib` numerical canary and `Apple10` (`NAX`) hook reservation (`P2` -> `G16`)

- Execute a sub-millisecond **startup GPU numerical canary** once during
  pipeline initialization over preallocated scratch (`16 × 16` deterministic
  `f16`, `bf16`, and `q4_affine` reference probes across `32x32` and `64x64`
  kernels checked for finiteness and tight `fp32` tolerance; measured `~0.19–0.24 ms`
  warm in-buffer GPU check and `~1.3–3.8 ms` cold first-submit including initial
  Metal 4 queue/residency wake on M4), aborting startup with an explicit
  toolchain/driver error if the compiled `.metallib` miscomputes
  ([ADR-0016](0016-native-fp16-bf16-and-multishard-import.md)).
- Expose `apple_gpu_family >= 10` (`Apple10`, M5 GPU Neural Accelerators / `NAX`
  `mpp::tensor_ops::matmul2d`) detection in `apple_bridge` / `aminfer probe` and
  reserve compile-time / function-constant dispatch hooks in `src/gpu/dequant.zig`
  ([ADR-0014](0014-compile-time-specialization.md)), without shipping unverified
  M5-only shader bodies on the M4 (`Apple9`) baseline
  ([../architecture/hardware.md](../architecture/hardware.md)).

## Consequences

- **Positive**:
  - `512+`-token GPU prefill reads projection and MoE weights once per chunk
    instead of 8 times, skips upper-triangular attention blocks, writes directly
    into `KvTile`s, and prunes unused final-layer / `lm_head` rows (`P0-A`,
    `P0-B1`), while `--gpu-prefill-mode tree64` keeps an instant runtime A/B
    baseline.
  - Single-request runs offset wide-prefill scratch via `WiredArena` workspace
    aliasing (`P0-B2`) without breaking `tie_word_embeddings` or `C14` serving.
  - Concurrent `serve` workloads protect active decode `TPOT p99` during
    long-context arrivals and reuse ChatML pre-last-user state on multi-turn
    edits/retries (`P1-A`, both configurable and ON by default).
  - Desktop co-tenancy avoids silent XNU anonymous-page compression/swap when
    sizing optional wired caches (`P1-B`), and corrupted `.metallib` builds fail
    at startup (`~0.19–0.24 ms` warm in-buffer GPU check; `~1.3–3.8 ms` cold
    first-submit including Metal 4 queue/residency wake, `P2`).
- **Trade-offs and invariants**:
  - Adding `BM = 64, BN = 64` prefill GEMM and FlashAttention prefill kernels
    increases `.metallib` pipeline count modestly at startup while keeping hot
    steady-state allocations at zero
    ([ADR-0011](0011-metal4-zero-allocation-hot-path.md)).
  - `Apple10` `NAX` kernels remain interface-reserved until an `Apple10` (M5+)
    host is available to run hardware parity and bandwidth gates.

## Evidence

- Post-M6.5 `Qwen3Model` / `GraphModel` prefill inspection (`<= 64`-row
  `verifyTree` loop and `BM = 32, BN = 32` GEMM tiles) and M6.5 `C14` / `R8`
  serving baseline in [../benchmarks/results.md](../benchmarks/results.md).
- Static wired budget invariants in
  [ADR-0012](0012-wired-memory-budget-headroom.md) and
  [../architecture/memory.md](../architecture/memory.md); prefill and serving
  schedules in
  [../architecture/runtime-pipeline.md](../architecture/runtime-pipeline.md).

## Open items

- None blocking Milestone M6.8 implementation. Measured A/B benchmarks (`wide`
  vs `tree64` TTFT, `prompt_tail` savings, `decode-share` + chunk-narrowing
  `TPOT p99`, ChatML boundary hit TTFT, live-RAM soft cache scaling, and startup
  canary latency) will be recorded verbatim in
  [../benchmarks/results.md](../benchmarks/results.md).
