"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DTYPE_MODES,
    QUANT_PROFILES,
    SME2_PACK_MODES,
    SME2_REPACK_TARGETS,
    QuantParamKind,
    TensorLayout,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    repack_amf_sme2,
    retag_amf_cold_experts,
)
from aminfer_tools.gsq_rco import gsq_convert
from aminfer_tools.qwen4_exp import pack_qwen4_checkpoint
from aminfer_tools import ngram as ngram_mod
from aminfer_tools.coreml import (
    COREML_KINDS,
    COREML_OPS,
    DRAFT_STAGES,
    compile_mlpackage,
    export_attention_qkv,
    export_draft_head,
    export_expert_predictor,
)
from aminfer_tools.heat import DEFAULT_SAMPLE_TOKENS, compute_expert_heat, write_heat_profile
from aminfer_tools.synth import (
    DRAFT_DTYPES,
    DRAFT_FORMATS,
    DRAFT_PRESETS,
    LOOP_PRESETS,
    PRESETS,
    QWEN4_PRESETS,
    generate_draft_bundle,
    generate_loop_bundle,
    generate_qwen3_bundle,
    generate_qwen4_bundle,
)


def align_up(length: int, alignment: int = APPLE_PAGE_SIZE) -> int:
    """Round a non-negative byte length up to a multiple of `alignment`."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a positive power of two")
    return (length + alignment - 1) & ~(alignment - 1)


def _parse_cold_experts(raw: str | None) -> set[tuple[int, int]]:
    if not raw:
        return set()
    result: set[tuple[int, int]] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        layer_str, sep, expert_str = item.partition(":")
        if not sep:
            raise ValueError(f"expected LAYER:EXPERT in --cold-experts, got {item!r}")
        result.add((int(layer_str), int(expert_str)))
    return result


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_int_csv(raw: str | None) -> tuple[int, ...] | None:
    items = _parse_csv(raw)
    if not items:
        return None
    return tuple(int(tok) for tok in items)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aminfer-tools",
        description=(
            "Offline preparation tools for aminfer (never used at inference runtime)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser(
        "pack-amf",
        help=(
            "Stream a safetensors checkpoint (single file, index.json or directory) "
            "into a 16 KiB-aligned .amf v0 file"
        ),
    )
    pack.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help=(
            "Input .safetensors file, model.safetensors.index.json, or checkpoint directory "
            "(auto-locates model.safetensors.index.json or model.safetensors)"
        ),
    )
    pack.add_argument("--output", required=True, type=pathlib.Path, help="Output .amf path")
    pack.add_argument("--model-id", default="aminfer-model", help="Model identifier stored in header")
    pack.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    pack.add_argument(
        "--dtype",
        choices=DTYPE_MODES,
        default="auto",
        help=(
            "Unquantized tensor precision: auto preserves F16/BF16/F32 bit-exact; "
            "f16 converts BF16 to F16; bf16 converts F16 to BF16 (default: auto)"
        ),
    )
    pack_tie = pack.add_mutually_exclusive_group()
    pack_tie.add_argument(
        "--tie-word-embeddings",
        dest="tie_word_embeddings",
        action="store_const",
        const=True,
        default=None,
        help="Skip a duplicate lm_head when embed_tokens is present (default: read config.json)",
    )
    pack_tie.add_argument(
        "--keep-tied-lm-head",
        dest="tie_word_embeddings",
        action="store_const",
        const=False,
        help="Always emit lm_head even if config.json sets tie_word_embeddings",
    )
    pack.add_argument(
        "--cold-experts",
        default=None,
        help="Comma-separated LAYER:EXPERT pairs to mark as cold_expert (e.g. 0:2,0:5)",
    )
    pack.add_argument(
        "--heat-profile",
        type=pathlib.Path,
        default=None,
        help="Schema-v1 JSON heat profile from `expert-heat` specifying cold experts",
    )
    pack.add_argument(
        "--cold-experts-per-layer",
        type=int,
        default=None,
        help="Override cold expert count per layer when using --heat-profile",
    )
    pack.add_argument(
        "--all-experts-cold",
        action="store_true",
        help="Mark every expert tensor as cold_expert",
    )
    pack.add_argument(
        "--exclude-tensors",
        default=None,
        help="Comma-separated tensor names to omit from the output .amf (e.g. embed_tokens.weight)",
    )
    pack.add_argument(
        "--sme2-pack",
        choices=SME2_PACK_MODES,
        default="none",
        help=(
            "Pre-pack eligible 2D weights into single-owner ARMv9 SME2 sme2_panel64 layout "
            "(ADR-0018, W15): none (default, row_major), routers, or all-linear"
        ),
    )
    pack.add_argument(
        "--lut16-sme2",
        action="store_true",
        help=(
            "Fit per-(panel, group) weighted Lloyd-Max ZT0 codebooks + per-column f16 RCO biases "
            "into QuantParamKind.LUT16_CODEBOOK (reserved0=0x0101) sme2_panel64 blobs (W17)"
        ),
    )
    pack.add_argument(
        "--quant-profile",
        choices=QUANT_PROFILES,
        default="uniform",
        help=(
            "Quantization profile: uniform (default, preserves source dtypes unchanged) or "
            "gsq-rco (ADR-0017 offline GSQ + RCO per-tensor / per-expert hybrid quantization)"
        ),
    )
    pack.add_argument(
        "--target-bits",
        type=float,
        default=None,
        help="Target average effective bits/weight when --quant-profile gsq-rco is selected (default: 3.25)",
    )
    pack.add_argument(
        "--no-rco",
        action="store_true",
        help="Disable RCO scale/bias refinement (use naive min-max baseline) under --quant-profile gsq-rco",
    )
    pack.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help="Optional JSON output path for the GSQ+RCO allocation report",
    )
    pack.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for offline GSQ+RCO / LUT16 quantization (default: 1)",
    )
    pack.add_argument(
        "--no-strip-vision",
        action="store_true",
        help="Keep visual.*/vision_tower.*/multi_modal_projector.* tensors instead of stripping them (W10)",
    )
    pack.add_argument(
        "--ple-output",
        type=pathlib.Path,
        default=None,
        help="Base .amf path for extracted SPARSE_TABLE PLE shard(s) (W10)",
    )
    pack.add_argument(
        "--ple-shards",
        type=int,
        default=1,
        help="Number of 16 KiB-aligned SPARSE_TABLE PLE .amf shards to emit (default: 1)",
    )
    pack.add_argument(
        "--mtp-output",
        type=pathlib.Path,
        default=None,
        help="Output .amf path for extracted 4B MTP draft head tensors (W10)",
    )
    pack.add_argument(
        "--loop-adapter-input",
        type=pathlib.Path,
        default=None,
        help="Optional safetensors path/dir with Step-LoRA Gated Loop Adapters (TensorKind.LOOP_ADAPTER=9, W16)",
    )
    pack.add_argument(
        "--config-output",
        type=pathlib.Path,
        default=None,
        help="Optional output config.json path with canonical operator_axes & loop_schedule metadata",
    )

    gsq = sub.add_parser(
        "gsq-convert",
        help=(
            "Convert a float/quantized safetensors checkpoint or existing .amf into a "
            "GSQ + RCO hybrid-quantized 16 KiB-aligned .amf v0 file (ADR-0017, W14)"
        ),
    )
    gsq.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Input .amf file, .safetensors file, model.safetensors.index.json, or checkpoint directory",
    )
    gsq.add_argument("--output", required=True, type=pathlib.Path, help="Output .amf path")
    gsq.add_argument(
        "--target-bits",
        required=True,
        type=float,
        help="Target average effective bits/weight across all model weights (e.g. 3.25)",
    )
    gsq.add_argument(
        "--heat-profile",
        type=pathlib.Path,
        default=None,
        help="Optional W8 schema-v1 heat.json profile guiding MoE expert sensitivity & cold tagging",
    )
    gsq.add_argument("--model-id", default=None, help="Override model identifier stored in header")
    gsq.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    gsq.add_argument(
        "--dtype",
        choices=DTYPE_MODES,
        default="auto",
        help="Float fallback dtype for unquantized tensors: auto (default), f16, or bf16",
    )
    gsq_tie = gsq.add_mutually_exclusive_group()
    gsq_tie.add_argument(
        "--tie-word-embeddings",
        dest="tie_word_embeddings",
        action="store_const",
        const=True,
        default=None,
        help="Skip a duplicate lm_head when embed_tokens is present (default: read config.json)",
    )
    gsq_tie.add_argument(
        "--keep-tied-lm-head",
        dest="tie_word_embeddings",
        action="store_const",
        const=False,
        help="Always emit lm_head even if config.json sets tie_word_embeddings",
    )
    gsq.add_argument("--cold-experts", default=None, help="Comma-separated LAYER:EXPERT cold pairs")
    gsq.add_argument(
        "--cold-experts-per-layer",
        type=int,
        default=None,
        help="Override cold expert count per layer when using --heat-profile",
    )
    gsq.add_argument("--all-experts-cold", action="store_true", help="Mark every expert tensor as cold_expert")
    gsq.add_argument("--exclude-tensors", default=None, help="Comma-separated tensor names to omit")
    gsq.add_argument(
        "--sme2-pack",
        choices=SME2_PACK_MODES,
        default="none",
        help="Pre-pack eligible 2D weights into sme2_panel64: none (default), routers, or all-linear",
    )
    gsq.add_argument(
        "--lut16-sme2",
        action="store_true",
        help="Pack quantized SME2 tensors with weighted Lloyd-Max ZT0 LUT16 codebooks (reserved0=0x0101, W17)",
    )
    gsq.add_argument("--no-rco", action="store_true", help="Use naive min-max instead of RCO refinement")
    gsq.add_argument("--rco-iters", type=int, default=3, help="RCO alternating OLS iterations per group (default: 3)")
    gsq.add_argument("--lloyd-iters", type=int, default=10, help="Lloyd-Max iterations per LUT16 group (default: 10)")
    gsq.add_argument("--sample-rows", type=int, default=32, help="Rows sampled per tensor for GSQ scoring (default: 32)")
    gsq.add_argument("--manifest", type=pathlib.Path, default=None, help="Optional JSON allocation manifest path")
    gsq.add_argument("--workers", type=int, default=1, help="Parallel worker processes (default: 1)")
    gsq.add_argument("--no-strip-vision", action="store_true", help="Keep vision tower tensors if present")

    repack = sub.add_parser(
        "repack-sme2",
        help=(
            "Convert selected 2D weights in an existing .amf into single-owner sme2_panel64 "
            "layout while preserving 16 KiB alignment and IEEE CRC-32 (W15)"
        ),
    )
    repack.add_argument("--input", required=True, type=pathlib.Path, help="Input .amf path")
    repack.add_argument("--output", required=True, type=pathlib.Path, help="Output .amf path")
    repack.add_argument(
        "--targets",
        choices=SME2_REPACK_TARGETS,
        default="routers",
        help="Which 2D weights to pack into sme2_panel64: routers (default) or all-linear",
    )
    repack.add_argument(
        "--lut16",
        action="store_true",
        help="Fit per-(panel, group) weighted Lloyd-Max ZT0 codebooks + RCO biases (QuantParamKind=1, 0x0101; W17)",
    )
    repack.add_argument("--lloyd-iters", type=int, default=10, help="Lloyd-Max iterations per LUT16 group (default: 10)")
    repack.add_argument("--workers", type=int, default=1, help="Parallel worker processes for --lut16 (default: 1)")

    retag = sub.add_parser(
        "retag-cold",
        help="Retag HOT_EXPERT / COLD_EXPERT directory entries in an existing .amf without rewriting blobs",
    )
    retag.add_argument("--input", required=True, type=pathlib.Path, help="Input .amf path")
    retag.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional output .amf path (default: update header and directory in place)",
    )
    retag.add_argument(
        "--cold-experts",
        default=None,
        help="Comma-separated LAYER:EXPERT pairs to mark as cold_expert",
    )
    retag.add_argument(
        "--heat-profile",
        type=pathlib.Path,
        default=None,
        help="Schema-v1 JSON heat profile specifying cold experts",
    )
    retag.add_argument(
        "--cold-experts-per-layer",
        type=int,
        default=None,
        help="Override cold expert count per layer when using --heat-profile",
    )
    retag.add_argument(
        "--all-experts-cold",
        action="store_true",
        help="Mark every expert tensor as cold_expert",
    )

    synth = sub.add_parser(
        "synth-qwen3",
        help="Generate deterministic synthetic Qwen3-MoE or Qwen3-Dense .amf, config.json and tokenizer.json",
    )
    synth.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS.keys()),
        help="Model geometry preset",
    )
    synth.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where model.amf, config.json and tokenizer.json are written",
    )
    synth.add_argument("--num-layers", type=int, default=None, help="Override number of transformer layers")
    synth.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    synth.add_argument("--model-id", default=None, help="Override model identifier stored in .amf header")
    synth.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")
    tie_group = synth.add_mutually_exclusive_group()
    tie_group.add_argument(
        "--tie-word-embeddings",
        action="store_true",
        help="Share embed_tokens with quantized lm_head",
    )
    tie_group.add_argument(
        "--untie-word-embeddings",
        action="store_true",
        help="Emit a distinct quantized lm_head tensor",
    )
    synth.add_argument(
        "--embed-f16",
        action="store_true",
        help="Store embed_tokens as F16 instead of quantized affine weights",
    )
    synth.add_argument(
        "--cold-experts",
        default=None,
        help="Comma-separated LAYER:EXPERT pairs to mark as cold_expert",
    )

    sq4 = sub.add_parser(
        "synth-qwen4",
        help=(
            "Generate deterministic synthetic qwen4_exp (Qwen3.8-Flash-Next) backbone .amf, "
            "sharded PLE .amf, MTP .amf, config.json and tokenizer.json (W10)"
        ),
    )
    sq4.add_argument(
        "--preset",
        required=True,
        choices=sorted(QWEN4_PRESETS.keys()),
        help="qwen4_exp geometry preset (tiny-qwen4 or scale-qwen4-512e)",
    )
    sq4.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where model.amf, ple*.amf, mtp.amf, config.json and tokenizer.json are written",
    )
    sq4.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    sq4.add_argument("--hidden-size", type=int, default=None, help="Override hidden_size (multiple of 64)")
    sq4.add_argument("--num-experts", type=int, default=None, help="Override routed expert count (up to 1024)")
    sq4.add_argument("--top-k", type=int, default=None, help="Override num_experts_per_tok (1..16)")
    sq4.add_argument("--vocab-size", type=int, default=None, help="Override vocab_size")
    sq4.add_argument("--ple-shards", type=int, default=None, help="Override number of SPARSE_TABLE PLE shards")
    sq4.add_argument("--ple-rows", type=int, default=None, help="Override total PLE table row count")
    sq4.add_argument("--no-mtp", action="store_true", help="Omit MTP draft head .amf")
    sq4.add_argument("--cold-experts", default=None, help="Comma-separated LAYER:EXPERT cold pairs")
    sq4.add_argument("--cold-experts-per-layer", type=int, default=0, help="Mark last N experts per layer as cold")
    sq4.add_argument("--model-id", default=None, help="Override model identifier stored in .amf header")
    sq4.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")

    sloop = sub.add_parser(
        "synth-loop",
        help=(
            "Generate deterministic synthetic native LoopLM (Prelude/Core/Coda) or "
            "Gated Loop Adapter (Step-LoRA, alpha_0=0.0) bundle (W16, ADR-0019)"
        ),
    )
    sloop.add_argument(
        "--preset",
        required=True,
        choices=sorted(LOOP_PRESETS.keys()),
        help="Loop preset (tiny-looplm or tiny-steplora)",
    )
    sloop.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where model.amf, config.json and tokenizer.json are written",
    )
    sloop.add_argument(
        "--base-arch",
        choices=("qwen3", "qwen4_exp"),
        default=None,
        help="Base backbone architecture (qwen3 or qwen4_exp)",
    )
    sloop.add_argument("--num-layers", type=int, default=None, help="Override total layer count")
    sloop.add_argument("--start-layer", type=int, default=None, help="Override loop core start_layer")
    sloop.add_argument("--end-layer", type=int, default=None, help="Override loop core end_layer (exclusive)")
    sloop.add_argument("--max-loops", type=int, default=None, help="Override loop iteration count R")
    sloop.add_argument("--rank", type=int, default=None, help="Override Step-LoRA rank (1..64)")
    sloop.add_argument("--model-id", default=None, help="Override model identifier stored in .amf header")
    sloop.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")

    draft = sub.add_parser(
        "synth-draft",
        help="Generate deterministic synthetic L2 draft head bundles (eagle3, eagle2, mtp; ADR-0013)",
    )
    draft_src = draft.add_mutually_exclusive_group(required=True)
    draft_src.add_argument(
        "--format",
        choices=DRAFT_FORMATS,
        default=None,
        help="L2 draft head layout (starts from matching tiny-* preset)",
    )
    draft_src.add_argument(
        "--preset",
        choices=sorted(DRAFT_PRESETS.keys()),
        default=None,
        help="Named synthetic draft head preset",
    )
    draft.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where model.amf and config.json are written",
    )
    draft.add_argument("--vocab-size", type=int, default=None, help="Override target vocabulary size")
    draft.add_argument("--draft-vocab-size", type=int, default=None, help="Override EAGLE-3 reduced draft vocabulary size")
    draft.add_argument("--hidden-size", type=int, default=None, help="Override hidden size")
    draft.add_argument("--num-heads", type=int, default=None, help="Override attention head count")
    draft.add_argument("--num-kv-heads", type=int, default=None, help="Override KV head count")
    draft.add_argument("--head-dim", type=int, default=None, help="Override attention head dimension")
    draft.add_argument("--intermediate-size", type=int, default=None, help="Override MLP intermediate dimension")
    draft.add_argument("--dtype", choices=DRAFT_DTYPES, default=None, help="Weight precision (default: preset)")
    draft.add_argument("--model-id", default=None, help="Override model identifier stored in .amf header")
    draft.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")
    draft.add_argument(
        "--aux-layer-ids",
        default=None,
        help="Comma-separated EAGLE-3 aux hidden state layer IDs (e.g. 2,24,45)",
    )
    draft_qk = draft.add_mutually_exclusive_group()
    draft_qk.add_argument("--qk-norm", dest="include_qk_norm", action="store_const", const=True, default=None)
    draft_qk.add_argument("--no-qk-norm", dest="include_qk_norm", action="store_const", const=False)
    draft_emb = draft.add_mutually_exclusive_group()
    draft_emb.add_argument("--include-embed-tokens", dest="include_embed_tokens", action="store_const", const=True, default=None)
    draft_emb.add_argument("--no-embed-tokens", dest="include_embed_tokens", action="store_const", const=False)
    draft_lm = draft.add_mutually_exclusive_group()
    draft_lm.add_argument("--include-lm-head", dest="include_lm_head", action="store_const", const=True, default=None)
    draft_lm.add_argument("--no-lm-head", dest="include_lm_head", action="store_const", const=False)

    heat = sub.add_parser(
        "expert-heat",
        help="Compute offline MoE expert activation heat statistics and hot/cold split JSON (W8)",
    )
    heat.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Input .amf file, .safetensors file, model.safetensors.index.json, or checkpoint directory",
    )
    heat.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output JSON path (default: print JSON to stdout)",
    )
    heat.add_argument(
        "--calibration-jsonl",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL calibration file (router trace {'layer','experts'} or token IDs {'token_ids'})",
    )
    heat.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of active experts selected per token (default: 8)",
    )
    heat_budget = heat.add_mutually_exclusive_group()
    heat_budget.add_argument(
        "--cold-experts-per-layer",
        type=int,
        default=None,
        help="Number of coldest experts per layer to designate as cold_expert",
    )
    heat_budget.add_argument(
        "--wired-budget-mib",
        type=float,
        default=None,
        help="Target wired model weight budget in MiB; experts exceeding the budget are marked cold",
    )
    heat.add_argument(
        "--slot-reserve-mib",
        type=float,
        default=0.0,
        help="Optional MiB reserved for cold-expert slots when computing --wired-budget-mib (default: 0)",
    )
    heat.add_argument(
        "--sample-tokens",
        type=int,
        default=DEFAULT_SAMPLE_TOKENS,
        help=f"Built-in representative token sample count when --calibration-jsonl is omitted (default: {DEFAULT_SAMPLE_TOKENS})",
    )
    heat.add_argument("--model-id", default=None, help="Override model_id recorded in the heat JSON")
    heat.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")

    coreml = sub.add_parser(
        "export-coreml",
        help="Export pure-stdlib CoreML .mlpackage bundles for the Apple Neural Engine (W7)",
    )
    coreml.add_argument(
        "--kind",
        required=True,
        choices=COREML_KINDS,
        help="CoreML graph kind: expert-predictor (A2), attention-qkv (A1), or draft-head (D4/D5)",
    )
    coreml.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output .mlpackage path (or directory for multi-window expert-predictor export)",
    )
    coreml.add_argument(
        "--amf",
        type=pathlib.Path,
        default=None,
        help="Optional .amf file to extract/dequantize real weights from (omit for synthetic weights)",
    )
    coreml.add_argument(
        "--op",
        choices=COREML_OPS,
        default="inner-product",
        help="Linear operator layout: inner-product ([M, C], default) or conv1x1 ([C, 1, M])",
    )
    coreml.add_argument("--model-id", default=None, help="Override model_id recorded in CoreML metadata")
    coreml.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed for synthetic weights")
    coreml.add_argument(
        "--layers",
        default=None,
        help="Layer range/list for expert-predictor (e.g. 0..47, 1..2, 0,2,4)",
    )
    coreml.add_argument(
        "--base-layer",
        type=int,
        default=None,
        help="Base layer index for single-window expert-predictor (predicts base+1..base+lookahead)",
    )
    coreml.add_argument(
        "--lookahead",
        type=int,
        default=2,
        help="Number of upcoming layers predicted by expert-predictor (default: 2)",
    )
    coreml.add_argument(
        "--pack-all-layers",
        action="store_true",
        help="Pack all selected --layers routers into one combined expert-predictor .mlpackage",
    )
    coreml.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Transformer layer index for attention-qkv or draft-head (default: 0)",
    )
    coreml.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Chunk token length for attention-qkv (default: 512, ADR-0010)",
    )
    coreml.add_argument(
        "--include-o-proj",
        action="store_true",
        help="Include output projection (attn_in -> o_out) in attention-qkv bundle",
    )
    coreml.add_argument(
        "--stage",
        choices=DRAFT_STAGES,
        default="ffn",
        help="Graph stage for draft-head: ffn (SwiGLU, default), fc (EAGLE-3), or linear",
    )
    coreml.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch/token row count M for expert-predictor and draft-head (default: 32)",
    )
    coreml.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden dimension for synthetic exports (default: 2048 for predictor/attn, 1024 for draft)",
    )
    coreml.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Expert count per layer for synthetic expert-predictor (default: 128)",
    )
    coreml.add_argument("--q-dim", type=int, default=4096, help="Q projection width for synthetic attention-qkv")
    coreml.add_argument("--kv-dim", type=int, default=512, help="K/V projection width for synthetic attention-qkv")
    coreml.add_argument(
        "--intermediate-size",
        type=int,
        default=3072,
        help="FFN intermediate width for synthetic draft-head (default: 3072)",
    )
    coreml.add_argument(
        "--out-dim",
        type=int,
        default=None,
        help="Output width for synthetic draft-head linear/fc stage",
    )
    coreml.add_argument(
        "--verify-compile",
        type=pathlib.Path,
        default=None,
        help="Optional directory to verify exported .mlpackage via `xcrun coremlcompiler compile`",
    )
    ngram_mod.add_subparser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-ngram":
        return ngram_mod.run_args(args)

    if args.command == "pack-amf":
        cold = _parse_cold_experts(args.cold_experts)
        excludes = _parse_csv(args.exclude_tensors)
        started = time.monotonic()
        use_qwen4_pipeline = (
            args.ple_output is not None
            or args.ple_shards != 1
            or args.mtp_output is not None
            or args.loop_adapter_input is not None
            or args.config_output is not None
        )
        stripped_vision = 0
        extra_notes: list[str] = []
        if use_qwen4_pipeline:
            rep_q4 = pack_qwen4_checkpoint(
                args.input,
                args.output,
                model_id=args.model_id,
                group_size=args.group_size,
                cold_experts=cold,
                all_experts_cold=args.all_experts_cold,
                dtype_mode=args.dtype,
                tie_word_embeddings=args.tie_word_embeddings,
                exclude_tensors=excludes,
                heat_profile=args.heat_profile,
                cold_experts_per_layer=args.cold_experts_per_layer,
                sme2_pack=args.sme2_pack,
                lut16_sme2=args.lut16_sme2,
                quant_profile=args.quant_profile,
                target_bits=args.target_bits,
                use_rco=not args.no_rco,
                manifest_path=args.manifest,
                workers=args.workers,
                strip_vision=not args.no_strip_vision,
                ple_output=args.ple_output,
                ple_shards=args.ple_shards,
                mtp_output=args.mtp_output,
                loop_adapter_input=args.loop_adapter_input,
                config_output=args.config_output,
            )
            metas = rep_q4.metas
            stripped_vision = rep_q4.stripped_vision_tensors
            if rep_q4.ple_shard_paths:
                extra_notes.append(f"ple_shards={len(rep_q4.ple_shard_paths)}")
            if rep_q4.mtp_amf_path is not None:
                extra_notes.append(f"mtp_tensors={len(rep_q4.mtp_metas)}")
            if rep_q4.loop_adapter_count > 0:
                extra_notes.append(f"loop_adapters={rep_q4.loop_adapter_count}")
        else:
            stats_map: dict[str, int] = {}
            metas = pack_amf_from_safetensors(
                args.input,
                args.output,
                model_id=args.model_id,
                group_size=args.group_size,
                cold_experts=cold,
                all_experts_cold=args.all_experts_cold,
                dtype_mode=args.dtype,
                tie_word_embeddings=args.tie_word_embeddings,
                exclude_tensors=excludes,
                heat_profile=args.heat_profile,
                cold_experts_per_layer=args.cold_experts_per_layer,
                sme2_pack=args.sme2_pack,
                lut16_sme2=args.lut16_sme2,
                quant_profile=args.quant_profile,
                target_bits=args.target_bits,
                use_rco=not args.no_rco,
                manifest_path=args.manifest,
                workers=args.workers,
                strip_vision=not args.no_strip_vision,
                stats_out=stats_map,
            )
            stripped_vision = stats_map.get("stripped_vision_tensors", 0)
        elapsed = time.monotonic() - started
        total_bytes = args.output.stat().st_size
        by_dtype: dict[str, int] = {}
        sme2_count = 0
        lut16_count = 0
        for m in metas:
            by_dtype[m.dtype.name.lower()] = by_dtype.get(m.dtype.name.lower(), 0) + 1
            if m.layout == TensorLayout.SME2_PANEL64:
                sme2_count += 1
            if m.quant_param_kind == QuantParamKind.LUT16_CODEBOOK:
                lut16_count += 1
        dtype_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items()))
        if sme2_count > 0:
            extra_notes.append(f"sme2_panel64={sme2_count}")
        if lut16_count > 0:
            extra_notes.append(f"lut16_codebook={lut16_count}")
        if stripped_vision > 0:
            extra_notes.append(f"stripped_vision_tensors={stripped_vision}")
        extra_str = (", " + ", ".join(extra_notes)) if extra_notes else ""
        print(
            f"packed {len(metas)} tensors into {args.output} ({total_bytes} bytes) "
            f"in {elapsed:.1f} s [{dtype_summary}{extra_str}]"
        )
        return 0

    if args.command == "gsq-convert":
        cold = _parse_cold_experts(args.cold_experts)
        excludes = _parse_csv(args.exclude_tensors)
        started = time.monotonic()
        rep = gsq_convert(
            args.input,
            args.output,
            target_bits=args.target_bits,
            model_id=args.model_id,
            group_size=args.group_size,
            dtype_mode=args.dtype,
            heat_profile=args.heat_profile,
            cold_experts=cold,
            cold_experts_per_layer=args.cold_experts_per_layer,
            all_experts_cold=args.all_experts_cold,
            tie_word_embeddings=args.tie_word_embeddings,
            exclude_tensors=excludes,
            sme2_pack=args.sme2_pack,
            lut16_sme2=args.lut16_sme2,
            use_rco=not args.no_rco,
            rco_iters=args.rco_iters,
            lloyd_iters=args.lloyd_iters,
            sample_rows=args.sample_rows,
            manifest_path=args.manifest,
            workers=args.workers,
            strip_vision=not args.no_strip_vision,
        )
        elapsed = time.monotonic() - started
        total_bytes = args.output.stat().st_size
        by_dtype: dict[str, int] = {}
        for m in rep.metas:
            by_dtype[m.dtype.name.lower()] = by_dtype.get(m.dtype.name.lower(), 0) + 1
        dtype_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items()))
        lut_note = f", lut16_codebook={rep.lut16_tensor_count}" if rep.lut16_tensor_count > 0 else ""
        print(
            f"gsq-converted {len(rep.metas)} tensors into {args.output} ({total_bytes} bytes, "
            f"target={rep.target_bits:.2f}b, effective={rep.effective_bits:.3f}b, "
            f"rco={'on' if rep.use_rco else 'off'}{lut_note}) in {elapsed:.1f} s [{dtype_summary}]"
        )
        return 0

    if args.command == "repack-sme2":
        started = time.monotonic()
        metas = repack_amf_sme2(
            args.input,
            args.output,
            targets=args.targets,
            lut16=args.lut16,
            lloyd_iters=args.lloyd_iters,
            workers=args.workers,
        )
        elapsed = time.monotonic() - started
        total_bytes = args.output.stat().st_size
        sme2_count = sum(1 for m in metas if m.layout == TensorLayout.SME2_PANEL64)
        lut16_count = sum(1 for m in metas if m.quant_param_kind == QuantParamKind.LUT16_CODEBOOK)
        row_major_count = len(metas) - sme2_count
        lut_note = f", lut16_codebook={lut16_count}" if lut16_count > 0 else ""
        print(
            f"repacked {args.output} ({total_bytes} bytes, targets={args.targets}): "
            f"sme2_panel64={sme2_count}{lut_note}, row_major={row_major_count} in {elapsed:.2f} s"
        )
        return 0

    if args.command == "retag-cold":
        cold = _parse_cold_experts(args.cold_experts)
        if args.heat_profile is not None:
            cold.update(
                load_cold_experts_from_heat_profile(
                    args.heat_profile,
                    cold_experts_per_layer=args.cold_experts_per_layer,
                )
            )
        elif args.cold_experts_per_layer is not None:
            parser.error("--cold-experts-per-layer requires --heat-profile")
        started = time.monotonic()
        hot_count, cold_count = retag_amf_cold_experts(
            args.input,
            cold,
            output_path=args.output,
            all_experts_cold=args.all_experts_cold,
        )
        elapsed = time.monotonic() - started
        dst = args.output if args.output is not None else args.input
        print(
            f"retagged {dst}: hot_expert_tensors={hot_count}, cold_expert_tensors={cold_count} "
            f"in {elapsed:.2f} s"
        )
        return 0

    if args.command == "expert-heat":
        started = time.monotonic()
        profile = compute_expert_heat(
            args.input,
            calibration_jsonl=args.calibration_jsonl,
            top_k=args.top_k,
            cold_experts_per_layer=args.cold_experts_per_layer,
            wired_budget_mib=args.wired_budget_mib,
            slot_reserve_mib=args.slot_reserve_mib,
            sample_tokens=args.sample_tokens,
            model_id=args.model_id,
            group_size=args.group_size,
        )
        elapsed = time.monotonic() - started
        if args.output is not None:
            write_heat_profile(profile, args.output)
            print(
                f"wrote heat profile for {profile.model_id} "
                f"({profile.num_layers} layers x {profile.num_experts} experts, top_k={profile.top_k}, "
                f"hot={profile.total_hot_experts}, cold={profile.total_cold_experts}, "
                f"mode={profile.calibration_mode}) to {args.output} in {elapsed:.2f} s"
            )
        else:
            print(json.dumps(profile.to_dict(), indent=2))
        return 0

    if args.command == "export-coreml":
        started = time.monotonic()
        if args.kind == "expert-predictor":
            res = export_expert_predictor(
                args.output,
                amf_path=args.amf,
                layers_spec=args.layers,
                base_layer=args.base_layer,
                lookahead=args.lookahead,
                batch_size=args.batch_size,
                op=args.op,
                pack_all_layers=args.pack_all_layers,
                hidden_size=args.hidden_size if args.hidden_size is not None else 2048,
                num_experts=args.num_experts,
                seed=args.seed,
                model_id=args.model_id,
            )
        elif args.kind == "attention-qkv":
            res = export_attention_qkv(
                args.output,
                amf_path=args.amf,
                layer=args.layer,
                seq_len=args.seq_len,
                op=args.op,
                include_o_proj=args.include_o_proj,
                hidden_size=args.hidden_size if args.hidden_size is not None else 2048,
                q_dim=args.q_dim,
                kv_dim=args.kv_dim,
                seed=args.seed,
                model_id=args.model_id,
            )
        elif args.kind == "draft-head":
            res = export_draft_head(
                args.output,
                amf_path=args.amf,
                layer=args.layer,
                stage=args.stage,
                batch_size=args.batch_size,
                op=args.op,
                hidden_size=args.hidden_size if args.hidden_size is not None else 1024,
                intermediate_size=args.intermediate_size,
                out_dim=args.out_dim,
                seed=args.seed,
                model_id=args.model_id,
            )
        else:
            parser.error(f"unsupported CoreML kind {args.kind!r}")

        if args.verify_compile is not None:
            for pkg in res.bundle_paths:
                compile_mlpackage(pkg, args.verify_compile)

        elapsed = time.monotonic() - started
        in_desc = ", ".join(f"{t.name}{list(t.shape)}" for t in res.inputs)
        out_desc = ", ".join(f"{t.name}{list(t.shape)}" for t in res.outputs)
        print(
            f"exported CoreML {res.kind} ({res.op}, {len(res.bundle_paths)} bundle(s), "
            f"{res.weight_bytes} weight bytes) -> {res.output_path} "
            f"[{in_desc} -> {out_desc}] in {elapsed:.2f} s"
        )
        return 0

    if args.command == "synth-draft":
        preset_key = args.preset if args.preset is not None else f"tiny-{args.format}"
        cfg = DRAFT_PRESETS[preset_key]
        overrides: dict[str, object] = {}
        if args.vocab_size is not None:
            overrides["vocab_size"] = args.vocab_size
            if cfg.draft_format != "eagle3" and args.draft_vocab_size is None:
                overrides["draft_vocab_size"] = args.vocab_size
        if args.draft_vocab_size is not None:
            overrides["draft_vocab_size"] = args.draft_vocab_size
        if args.hidden_size is not None:
            overrides["hidden_size"] = args.hidden_size
        if args.num_heads is not None:
            overrides["num_attention_heads"] = args.num_heads
        if args.num_kv_heads is not None:
            overrides["num_key_value_heads"] = args.num_kv_heads
        if args.head_dim is not None:
            overrides["head_dim"] = args.head_dim
        if args.intermediate_size is not None:
            overrides["intermediate_size"] = args.intermediate_size
        if args.dtype is not None:
            overrides["dtype"] = args.dtype
        if args.model_id is not None:
            overrides["model_id"] = args.model_id
        aux_ids = _parse_int_csv(args.aux_layer_ids)
        if aux_ids is not None:
            overrides["aux_layer_ids"] = aux_ids
        if args.include_qk_norm is not None:
            overrides["include_qk_norm"] = args.include_qk_norm
        if args.include_embed_tokens is not None:
            overrides["include_embed_tokens"] = args.include_embed_tokens
        if args.include_lm_head is not None:
            overrides["include_lm_head"] = args.include_lm_head
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        bundle = generate_draft_bundle(args.output_dir, cfg, seed=args.seed)
        print(
            f"synthesized {cfg.draft_format} draft ({len(bundle.metas)} tensors, "
            f"{bundle.total_bytes} bytes) into {bundle.amf_path.parent}"
        )
        return 0

    if args.command == "synth-qwen3":
        cfg = PRESETS[args.preset]
        overrides: dict[str, object] = {}
        if args.num_layers is not None:
            overrides["num_hidden_layers"] = args.num_layers
        if args.vocab_size is not None:
            overrides["vocab_size"] = args.vocab_size
        if args.model_id is not None:
            overrides["model_id"] = args.model_id
        if args.tie_word_embeddings:
            overrides["tie_word_embeddings"] = True
        elif args.untie_word_embeddings:
            overrides["tie_word_embeddings"] = False
        if args.embed_f16:
            overrides["embed_f16"] = True
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        cold = _parse_cold_experts(args.cold_experts)
        bundle = generate_qwen3_bundle(
            args.output_dir,
            cfg,
            seed=args.seed,
            cold_experts=cold,
        )
        print(
            f"synthesized {len(bundle.metas)} tensors ({bundle.total_bytes} bytes) "
            f"into {bundle.amf_path.parent}"
        )
        return 0

    if args.command == "synth-qwen4":
        cfg = QWEN4_PRESETS[args.preset]
        overrides: dict[str, object] = {}
        if args.num_layers is not None:
            overrides["num_hidden_layers"] = args.num_layers
        if args.hidden_size is not None:
            overrides["hidden_size"] = args.hidden_size
            overrides["ple_embed_dim"] = args.hidden_size
        if args.num_experts is not None:
            overrides["num_experts"] = args.num_experts
        if args.top_k is not None:
            overrides["num_experts_per_tok"] = args.top_k
        if args.vocab_size is not None:
            overrides["vocab_size"] = args.vocab_size
        if args.ple_shards is not None:
            overrides["ple_shards"] = args.ple_shards
        if args.ple_rows is not None:
            overrides["ple_num_rows"] = args.ple_rows
        if args.no_mtp:
            overrides["include_mtp"] = False
        if args.model_id is not None:
            overrides["model_id"] = args.model_id
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        cold = _parse_cold_experts(args.cold_experts)
        bundle = generate_qwen4_bundle(
            args.output_dir,
            cfg,
            seed=args.seed,
            cold_experts=cold,
            cold_experts_per_layer=args.cold_experts_per_layer,
        )
        print(
            f"synthesized qwen4_exp ({len(bundle.metas)} backbone tensors, "
            f"{len(bundle.ple_paths)} PLE shard(s), "
            f"{len(bundle.mtp_metas)} MTP tensors, {bundle.total_bytes} bytes) "
            f"into {bundle.amf_path.parent}"
        )
        return 0

    if args.command == "synth-loop":
        cfg = LOOP_PRESETS[args.preset]
        overrides = {}
        if args.base_arch is not None:
            overrides["base_arch"] = args.base_arch
        if args.num_layers is not None:
            overrides["num_hidden_layers"] = args.num_layers
        if args.start_layer is not None:
            overrides["start_layer"] = args.start_layer
        if args.end_layer is not None:
            overrides["end_layer"] = args.end_layer
        if args.max_loops is not None:
            overrides["max_loops"] = args.max_loops
        if args.rank is not None:
            overrides["adapter_rank"] = args.rank
        if args.model_id is not None:
            overrides["model_id"] = args.model_id
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        bundle = generate_loop_bundle(args.output_dir, cfg, seed=args.seed)
        print(
            f"synthesized {cfg.mode} loop bundle ({len(bundle.metas)} tensors, "
            f"core=[{cfg.start_layer},{cfg.end_layer})x{cfg.max_loops}, {bundle.total_bytes} bytes) "
            f"into {bundle.amf_path.parent}"
        )
        return 0

    print(
        f"aminfer-tools: '{args.command}' is not implemented yet; "
        "see docs/status.md for target milestones.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
