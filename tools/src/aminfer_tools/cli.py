"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AFFINE_GROUP_SIZES,
    DTYPE_MODES,
    NO_EXPERT,
    QUANT_PROFILES,
    SME2_PACK_MODES,
    SME2_REPACK_TARGETS,
    VISION_BLOCK_DTYPES,
    AmfReader,
    QuantParamKind,
    SyntheticVitConfig,
    TensorKind,
    TensorLayout,
    _collect_tensors,
    embed_metadata_into_amf,
    generate_synth_vit_amf,
    gsq_convert_with_vision,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    pack_vision_tower_amf,
    repack_amf_sme2,
    resolve_checkpoint,
    retag_amf_cold_experts,
)
from aminfer_tools.chat_template import compile_chat_template_bytes
from aminfer_tools.gguf import pack_amf_from_gguf
from aminfer_tools.lora import (
    LORA_DTYPES,
    LORA_LAYOUTS,
    generate_synthetic_lora,
    pack_lora_from_peft,
)
from aminfer_tools.gsq_rco import gsq_convert
from aminfer_tools.qwen4_exp import (
    pack_qwen4_checkpoint,
    validate_hf_metadata_dir as validate_qwen4_metadata_dir,
)
from aminfer_tools.deepseek41 import (
    DEEPSEEK41_PRESETS,
    VALID_DEEPSEEK41_DTYPES,
    VALID_PARTITION_STRATEGIES,
    VALID_TARGET_TIERS,
    HardwareTierError,
    convert_deepseek41_checkpoint,
    generate_synth_deepseek41_bundle,
    partition_deepseek41_amf_shards,
    validate_hf_metadata_dir as validate_deepseek41_metadata_dir,
)
from aminfer_tools import ngram as ngram_mod
from aminfer_tools.coreml import (
    COREML_KINDS,
    COREML_OPS,
    DEFAULT_VISION_TOKEN_BUCKETS,
    DRAFT_STAGES,
    VISION_MLP_ACTS,
    compile_mlpackage,
    export_attention_qkv,
    export_draft_head,
    export_expert_predictor,
    export_vision_tower,
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


def _add_gsq_safety_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--max-q2-share",
        type=float,
        default=None,
        help="Maximum fraction of quantizable weights permitted in q2_affine (default: 0.0 auto Dense, 0.55 MoE)",
    )
    subparser.add_argument(
        "--allow-unsafe-q2",
        action="store_true",
        help="Disable backbone non-Q2 floor and Q2 weight-share cap (enforce_backbone_floor=False, max_q2_share=1.0)",
    )
    subparser.add_argument(
        "--allow-retranscode-lowbit",
        action="store_true",
        help="Allow double-quantizing existing q2/q3/q4_affine source tensors down to q2_affine",
    )
    subparser.add_argument(
        "--fine-rungs",
        action="store_true",
        help="Enable intermediate g32 and q5/q6/q8_affine GSQ rungs (MoE layers capped <= 5 schemes)",
    )


def _is_moe_checkpoint_input(path: pathlib.Path) -> bool:
    """Return True when `path` is an MoE checkpoint (`num_experts > 1`) without loading tensor payloads."""
    try:
        if path.is_file() and path.suffix == ".amf":
            with AmfReader(path, verify_header_crc=False, verify_dir_crc=False) as rdr:
                experts = {
                    e.expert
                    for e in rdr.entries
                    if e.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT) and e.expert != NO_EXPERT
                }
                return len(experts) > 1
        resolved = resolve_checkpoint(path)
        tensors = _collect_tensors(resolved.shard_paths)
        expert_ids: set[int] = set()
        for name, t in tensors.items():
            if len(t.shape) == 3 and ".switch_mlp." in name and t.shape[0] > 1:
                return True
            m = re.search(r"\.experts\.(\d+)\.", name)
            if m is not None:
                expert_ids.add(int(m.group(1)))
                if len(expert_ids) > 1:
                    return True
        return False
    except (OSError, ValueError):
        return False


def _resolve_cli_gsq_safety_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_path: pathlib.Path,
) -> tuple[bool, float | None, bool, bool]:
    if args.max_q2_share is not None and not (0.0 <= args.max_q2_share <= 1.0):
        parser.error(f"--max-q2-share must be in [0.0, 1.0], got {args.max_q2_share}")
    enforce_backbone_floor = not args.allow_unsafe_q2
    if args.allow_unsafe_q2:
        max_q2_share = 1.0 if args.max_q2_share is None else args.max_q2_share
    elif args.max_q2_share is not None:
        max_q2_share = args.max_q2_share
    else:
        max_q2_share = 0.55 if _is_moe_checkpoint_input(input_path) else None
    return (
        enforce_backbone_floor,
        max_q2_share,
        bool(args.allow_retranscode_lowbit),
        bool(args.fine_rungs),
    )


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
    pack.add_argument(
        "--group-size",
        type=int,
        choices=AFFINE_GROUP_SIZES,
        default=None,
        help="Quantization group size (32, 64, or 128; default: auto from config.json or 64)",
    )
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
        help="Target average effective bits/weight when --quant-profile gsq-rco is selected (default: auto 4.15 Dense / 3.50 MoE)",
    )
    _add_gsq_safety_args(pack)
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
        "--include-vision",
        action="store_true",
        help="Include vision tower tensors (default when visual.blocks.*/vision.blocks.* are present, W20)",
    )
    pack.add_argument(
        "--strip-vision",
        action="store_true",
        help="Opt out of packing vision tower tensors and strip them from the output .amf (W20)",
    )
    pack.add_argument(
        "--no-strip-vision",
        action="store_true",
        help="Alias for --include-vision: keep visual.*/vision.*/aligner.* tensors (W10/W20)",
    )
    pack.add_argument(
        "--vision-sidecar",
        type=pathlib.Path,
        default=None,
        help="Extract vision tower tensors into a standalone 16 KiB-aligned vision.amf sidecar (W20)",
    )
    pack.add_argument(
        "--vision-block-dtype",
        choices=VISION_BLOCK_DTYPES,
        default="auto",
        help="Precision policy for ViT block projections: auto, f16, bf16, or q4_affine (W20)",
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
    pack.add_argument(
        "--embed-metadata",
        action="store_true",
        help=(
            "Embed config.json and tokenizer.json inside the single .amf file as 16 KiB-aligned "
            "cold __meta.* blobs with CRC-32 verification (W18, ADR-0022)"
        ),
    )
    pack.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Explicit config.json path to embed into __meta.config_json (implies --embed-metadata)",
    )
    pack.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        default=None,
        help="Explicit tokenizer.json path to embed into __meta.tokenizer_json (implies --embed-metadata)",
    )
    pack.add_argument(
        "--chat-template",
        type=pathlib.Path,
        default=None,
        help="Explicit chat_template.jinja path to compile into __meta.chat_template_json",
    )
    pack.add_argument(
        "--imatrix",
        type=pathlib.Path,
        default=None,
        help="Optional GGUF/.dat imatrix file weighting --quant-profile gsq-rco sensitivity",
    )
    pack.add_argument(
        "--no-strict-vocab",
        action="store_true",
        help="Skip cross-checking external tokenizer.json vocab against GGUF tokenizer.ggml.tokens",
    )

    pack_gguf = sub.add_parser(
        "pack-gguf",
        help="Convert a GGUF v2/v3 file (or split shards) directly into a 16 KiB-aligned .amf v0 file (W22)",
    )
    pack_gguf.add_argument("--input", required=True, type=pathlib.Path, help="Input .gguf path, split shard 00001-of-N, or directory")
    pack_gguf.add_argument("--output", required=True, type=pathlib.Path, help="Output .amf path")
    pack_gguf.add_argument("--model-id", default=None, help="Override model identifier stored in header")
    pack_gguf.add_argument("--dtype", choices=DTYPE_MODES, default="auto", help="Float fallback dtype: auto, f16, or bf16")
    pack_gguf.add_argument("--cold-experts", default=None, help="Comma-separated LAYER:EXPERT pairs marked cold_expert")
    pack_gguf.add_argument("--all-experts-cold", action="store_true", help="Mark all MoE expert tensors as cold_expert")
    pack_gguf.add_argument("--exclude-tensors", default="", help="Comma-separated raw or normalized tensor names to skip")
    pack_gguf.add_argument("--embed-metadata", action="store_true", help="Embed config, tokenizer, and compiled chat_template_spec")
    pack_gguf.add_argument("--config", type=pathlib.Path, default=None, help="Optional HF config.json override")
    pack_gguf.add_argument("--tokenizer", type=pathlib.Path, default=None, help="Preferred external HF tokenizer.json")
    pack_gguf.add_argument("--tokenizer-config", type=pathlib.Path, default=None, help="Optional HF tokenizer_config.json")
    pack_gguf.add_argument("--chat-template", type=pathlib.Path, default=None, help="Preferred external chat_template.jinja")
    pack_gguf.add_argument("--vision-block-dtype", choices=VISION_BLOCK_DTYPES, default="auto", help="W20 policy for clip mmproj blocks")
    pack_gguf.add_argument("--group-size", type=int, choices=AFFINE_GROUP_SIZES, default=64, help="Group size for clip W20 q4 blocks")
    pack_gguf.add_argument("--no-strict-vocab", action="store_true", help="Disable GGUF vs external HF vocab size check")
    pack_gguf.add_argument("--no-dedup-recurrent", action="store_true", help="Do not collapse unrolled recurrent layers")

    ct_cmd = sub.add_parser(
        "compile-chat-template",
        help="Compile HF chat_template.jinja / tokenizer_config.json into declarative chat_template_spec JSON (W24)",
    )
    ct_cmd.add_argument("--input", required=True, type=pathlib.Path, help="Checkpoint directory or chat_template.jinja path")
    ct_cmd.add_argument("--output", type=pathlib.Path, default=None, help="Optional output JSON path (prints to stdout if omitted)")
    ct_cmd.add_argument("--chat-template", type=pathlib.Path, default=None, help="Explicit chat_template.jinja path")
    ct_cmd.add_argument("--tokenizer-config", type=pathlib.Path, default=None, help="Explicit tokenizer_config.json path")
    ct_cmd.add_argument("--config", type=pathlib.Path, default=None, help="Explicit config.json path")

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
        default=None,
        type=float,
        help="Target average effective bits/weight across all model weights (default: auto 4.15 Dense / 3.50 MoE)",
    )
    _add_gsq_safety_args(gsq)
    gsq.add_argument(
        "--heat-profile",
        type=pathlib.Path,
        default=None,
        help="Optional W8 schema-v1 heat.json profile guiding MoE expert sensitivity & cold tagging",
    )
    gsq.add_argument("--model-id", default=None, help="Override model identifier stored in header")
    gsq.add_argument("--group-size", type=int, choices=AFFINE_GROUP_SIZES, default=64, help="Quantization group size (default: 64)")
    gsq.add_argument("--imatrix", type=pathlib.Path, default=None, help="Optional GGUF/.dat imatrix file weighting sensitivity")
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
    gsq.add_argument("--include-vision", action="store_true", help="Include vision tower tensors with W20 GSQ+RCO policy")
    gsq.add_argument("--strip-vision", action="store_true", help="Strip vision tower tensors")
    gsq.add_argument("--no-strip-vision", action="store_true", help="Alias for --include-vision")
    gsq.add_argument(
        "--vision-sidecar",
        type=pathlib.Path,
        default=None,
        help="Extract vision tower into a standalone 16 KiB-aligned vision.amf sidecar (W20)",
    )
    gsq.add_argument(
        "--vision-block-dtype",
        choices=VISION_BLOCK_DTYPES,
        default="auto",
        help="Precision for ViT block projections under GSQ+RCO: auto (q4_affine), f16, bf16, or q4_affine",
    )

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
    synth.add_argument(
        "--embed-metadata",
        action="store_true",
        help="Embed config.json and tokenizer.json inside model.amf as cold __meta.* blobs (W18)",
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

    sds41 = sub.add_parser(
        "synth-deepseek41",
        help=(
            "Generate deterministic synthetic deepseek41 (DeepSeek-V4.1-Flash-1017) CED backbone .amf, "
            "sharded Engram .amf, DSpark .amf, config.json and tokenizer.json (W11, ADR-0015)"
        ),
    )
    sds41.add_argument(
        "--preset",
        default="tiny-deepseek41",
        choices=sorted(DEEPSEEK41_PRESETS.keys()),
        help="deepseek41 geometry preset (tiny-deepseek41, scale-deepseek41-512e, or full-deepseek41-552b)",
    )
    sds41.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where model.amf, engram*.amf, dspark.amf, config.json and tokenizer.json are written",
    )
    sds41.add_argument("--num-encoder-layers", type=int, default=None, help="Override causal encoder layer count")
    sds41.add_argument("--num-decoder-layers", type=int, default=None, help="Override causal decoder layer count")
    sds41.add_argument("--hidden-size", type=int, default=None, help="Override hidden_size (multiple of 64)")
    sds41.add_argument("--num-experts", type=int, default=None, help="Override routed expert count (1..1024)")
    sds41.add_argument("--encoder-top-k", type=int, default=None, help="Override encoder MoE top-k")
    sds41.add_argument("--decoder-top-k", type=int, default=None, help="Override decoder MoE top-k")
    sds41.add_argument("--vocab-size", type=int, default=None, help="Override vocab_size")
    sds41.add_argument("--kv-lora-rank", type=int, default=None, help="Override MLA kv_lora_rank (multiple of 32)")
    sds41.add_argument("--engram-heads", type=int, default=None, help="Override Engram head count (2..8)")
    sds41.add_argument("--engram-shards", type=int, default=None, help="Override number of SPARSE_TABLE Engram shards")
    sds41.add_argument("--engram-rows", type=int, default=None, help="Override total Engram table row count")
    sds41.add_argument(
        "--dtype",
        choices=VALID_DEEPSEEK41_DTYPES,
        default=None,
        help="Override backbone weight dtype (q3_affine, q4_affine, bf16)",
    )
    sds41.add_argument("--no-dspark", action="store_true", help="Omit DSpark proposer head .amf")
    sds41.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="If > 1, also partition backbone experts across N peer rank shards (default: 1)",
    )
    sds41.add_argument(
        "--partition-strategy",
        choices=VALID_PARTITION_STRATEGIES,
        default="contiguous_by_rank",
        help="Expert partition strategy when --world-size > 1 (default: contiguous_by_rank)",
    )
    sds41.add_argument("--cold-experts", default=None, help="Comma-separated LAYER:EXPERT cold pairs")
    sds41.add_argument("--cold-experts-per-layer", type=int, default=0, help="Mark last N experts per layer as cold")
    sds41.add_argument("--model-id", default=None, help="Override model identifier stored in .amf header")
    sds41.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")

    pds41 = sub.add_parser(
        "pack-deepseek41",
        help=(
            "Stream a DeepSeek-V4.1-Flash-1017 multi-shard safetensors checkpoint into 16 KiB-aligned "
            ".amf v0 backbone, sharded Engram SPARSE_TABLE, and optional DSpark / multi-rank shards (W11)"
        ),
    )
    pds41.add_argument("--input", required=True, type=pathlib.Path, help="Input safetensors file, index.json, or directory")
    pds41.add_argument("--output", required=True, type=pathlib.Path, help="Output backbone .amf path")
    pds41.add_argument("--model-id", default="deepseek41", help="Model identifier stored in .amf header")
    pds41.add_argument(
        "--dtype",
        choices=("auto",) + VALID_DEEPSEEK41_DTYPES,
        default="auto",
        help="Backbone weight target dtype: auto, q3_affine, q4_affine, or bf16 (default: auto)",
    )
    pds41.add_argument(
        "--dtype-mode",
        choices=DTYPE_MODES,
        default="auto",
        help="Unquantized float precision conversion mode (auto, f16, bf16)",
    )
    pds41.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    pds41.add_argument(
        "--quant-profile",
        choices=QUANT_PROFILES,
        default="uniform",
        help="Quantization profile: uniform (default) or gsq-rco (hybrid mixed-bit GSQ+RCO)",
    )
    pds41.add_argument("--target-bits", type=float, default=None, help="Average target bits for --quant-profile gsq-rco")
    pds41.add_argument("--no-rco", action="store_true", help="Disable RCO refinement when using --quant-profile gsq-rco")
    pds41.add_argument("--manifest", type=pathlib.Path, default=None, help="Optional JSON manifest path for gsq-rco")
    pds41.add_argument("--cold-experts", default=None, help="Comma-separated LAYER:EXPERT cold pairs")
    pds41.add_argument("--heat-profile", type=pathlib.Path, default=None, help="JSON expert heat profile path")
    pds41.add_argument("--cold-experts-per-layer", type=int, default=None, help="Cold experts per layer from heat profile")
    pds41.add_argument("--all-experts-cold", action="store_true", help="Mark every routed expert as cold_expert")
    pds41.add_argument("--sme2-pack", choices=SME2_PACK_MODES, default="none", help="ARMv9 SME2 panel64 pre-packing mode")
    pds41.add_argument("--lut16-sme2", action="store_true", help="Pack eligible SME2 weights with non-uniform ZT0 LUT16")
    pds41.add_argument("--engram-output", type=pathlib.Path, default=None, help="Base output path for Engram SPARSE_TABLE .amf")
    pds41.add_argument("--engram-shards", type=int, default=1, help="Number of Engram .amf shards (default: 1)")
    pds41.add_argument("--dspark-output", type=pathlib.Path, default=None, help="Output path for extracted DSpark .amf head")
    pds41.add_argument("--config-output", type=pathlib.Path, default=None, help="Optional output path for normalized config.json")
    pds41.add_argument(
        "--target-tier",
        choices=VALID_TARGET_TIERS,
        default="auto",
        help="Target hardware tier for ADR-0015 wired memory validation (default: auto)",
    )
    pds41.add_argument("--host-ram-gib", type=float, default=None, help="Optional host RAM (GiB) override for tier check")
    pds41.add_argument("--world-size", type=int, default=1, help="Export multi-rank expert shards for N peer nodes (default: 1)")
    pds41.add_argument("--rank-dir", type=pathlib.Path, default=None, help="Directory for multi-rank shards when --world-size > 1")
    pds41.add_argument(
        "--partition-strategy",
        choices=VALID_PARTITION_STRATEGIES,
        default="contiguous_by_rank",
        help="Expert partition strategy when --world-size > 1",
    )
    pds41.add_argument("--allow-tier-override", action="store_true", help="Bypass ADR-0015 hardware-tier rejection (expert use)")
    pds41.add_argument("--workers", type=int, default=1, help="Worker threads for GSQ+RCO quantization")
    pds41.add_argument("--include-vision", action="store_true", help="Include vision.* + aligner.* tensors (default for full ViT towers, W20)")
    pds41.add_argument("--strip-vision", action="store_true", help="Strip vision.* + aligner.* tensors from output .amf")
    pds41.add_argument("--no-strip-vision", action="store_true", help="Alias for --include-vision")
    pds41.add_argument("--vision-sidecar", type=pathlib.Path, default=None, help="Extract vision.* + aligner.* into standalone vision.amf (W20)")
    pds41.add_argument("--vision-block-dtype", choices=VISION_BLOCK_DTYPES, default="auto", help="Precision for ViT block projections (auto, f16, bf16, q4_affine)")

    part = sub.add_parser(
        "partition-experts",
        help=(
            "Split routed MoE experts in a .amf file across world_size peer ranks and emit "
            "cluster_manifest.json with ExpertSource.thunderbolt_peer metadata (W11, X1-X3)"
        ),
    )
    part.add_argument("--input", required=True, type=pathlib.Path, help="Input .amf backbone file")
    part.add_argument("--world-size", required=True, type=int, help="Number of peer ranks (1..16)")
    part.add_argument("--output-dir", required=True, type=pathlib.Path, help="Output directory for rank shards and cluster_manifest.json")
    part.add_argument(
        "--strategy",
        choices=VALID_PARTITION_STRATEGIES,
        default="contiguous_by_rank",
        help="Expert partitioning strategy (contiguous_by_rank or round_robin)",
    )
    part.add_argument(
        "--replicate-backbone",
        action="store_true",
        help="Replicate non-expert backbone tensors onto every worker rank shard (default: rank 0 only)",
    )

    insp = sub.add_parser(
        "inspect-metadata",
        help=(
            "Offline metadata-only dry-run validator for HF qwen4_exp / deepseek_v41 checkpoints "
            "(parses config.json + model.safetensors.index.json without weight shards on disk)"
        ),
    )
    insp.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Metadata directory (containing config.json + model.safetensors.index.json) or config.json path",
    )
    insp.add_argument(
        "--arch",
        choices=("auto", "qwen4_exp", "deepseek41"),
        default="auto",
        help="Architecture validator to invoke (default: auto-detect from config.json)",
    )
    insp.add_argument(
        "--host-ram-gib",
        type=float,
        default=None,
        help="Optional host RAM (GiB) to include a custom tier-admission entry",
    )
    insp.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit full structured JSON report instead of human-readable summary",
    )

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
        help="Export pure-stdlib CoreML .mlpackage bundles for the Apple Neural Engine (W7, W20)",
    )
    coreml.add_argument(
        "--kind",
        default=None,
        choices=COREML_KINDS,
        help="CoreML graph kind: expert-predictor (A2), attention-qkv (A1), draft-head (D4/D5), or vision-tower (W20)",
    )
    coreml.add_argument(
        "--component",
        default=None,
        choices=COREML_KINDS,
        help="Alias for --kind (e.g. --component vision-tower for W20 ANE vision tower export)",
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
        "--token-buckets",
        default=None,
        help="Comma-separated static token buckets for vision-tower (default: 64,256,576)",
    )
    coreml.add_argument(
        "--num-vision-layers",
        type=int,
        default=4,
        help="Synthetic vision-tower layer count when --amf is omitted (default: 4)",
    )
    coreml.add_argument(
        "--vision-hidden-size",
        type=int,
        default=128,
        help="Synthetic vision-tower hidden width when --amf is omitted (default: 128)",
    )
    coreml.add_argument(
        "--vision-intermediate-size",
        type=int,
        default=256,
        help="Synthetic vision-tower MLP intermediate width when --amf is omitted (default: 256)",
    )
    coreml.add_argument(
        "--out-hidden-size",
        type=int,
        default=128,
        help="Synthetic PatchMerger/Aligner LM output width when --amf is omitted (default: 128)",
    )
    coreml.add_argument(
        "--spatial-merge-size",
        type=int,
        default=2,
        help="Spatial patch merge factor for vision-tower (default: 2)",
    )
    coreml.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Vision patch size recorded in CoreML metadata (default: 14)",
    )
    coreml.add_argument(
        "--t-patch",
        type=int,
        default=1,
        help="Temporal patch tube depth recorded in CoreML metadata (default: 1)",
    )
    coreml.add_argument(
        "--mlp-act",
        choices=VISION_MLP_ACTS,
        default="gelu",
        help="Vision-tower MLP activation for synthetic export: gelu (default) or swiglu",
    )
    coreml.add_argument(
        "--verify-compile",
        type=pathlib.Path,
        default=None,
        help="Optional directory to verify exported .mlpackage via `xcrun coremlcompiler compile`",
    )
    ngram_mod.add_subparser(sub)

    plora = sub.add_parser(
        "pack-lora",
        help=(
            "Convert a Hugging Face PEFT LoRA checkpoint or synthesize a test adapter into a "
            "16 KiB-aligned .amlora (.amf-based) bundle (W19, ADR-0022)"
        ),
    )
    plora.add_argument(
        "--input",
        type=pathlib.Path,
        default=None,
        help="Input PEFT directory (adapter_config.json + adapter_model.safetensors) or .safetensors file",
    )
    plora.add_argument("--output", required=True, type=pathlib.Path, help="Output .amlora path")
    plora.add_argument("--adapter-id", default=None, help="Adapter identifier stored in header (default: output stem)")
    plora.add_argument("--adapter-config", type=pathlib.Path, default=None, help="Explicit adapter_config.json override")
    plora.add_argument("--layout", choices=LORA_LAYOUTS, default="row_major", help="Weight layout (default: row_major)")
    plora.add_argument("--dtype", choices=LORA_DTYPES, default="f16", help="Adapter weight precision (default: f16)")
    plora.add_argument("--rank", type=int, default=None, help="LoRA rank r (override or synthetic rank, 1..64)")
    plora.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling numerator")
    plora.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a deterministic synthetic .amlora test adapter instead of reading --input",
    )
    plora.add_argument("--num-layers", type=int, default=2, help="Synthetic layer count (default: 2)")
    plora.add_argument("--hidden-size", type=int, default=64, help="Synthetic hidden_size (default: 64)")
    plora.add_argument("--q-dim", type=int, default=None, help="Synthetic q_proj output dim (default: hidden_size)")
    plora.add_argument("--kv-dim", type=int, default=None, help="Synthetic k/v_proj output dim (default: hidden_size)")
    plora.add_argument("--intermediate-size", type=int, default=128, help="Synthetic MLP intermediate_size (default: 128)")
    plora.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="Comma-separated synthetic target modules (default: q_proj,v_proj)",
    )
    plora.add_argument("--seed", type=int, default=42, help="Synthetic PRNG seed (default: 42)")
    plora.add_argument("--zero-init-b", action="store_true", help="Zero-initialize synthetic lora_b matrices")

    svit = sub.add_parser(
        "synth-vit",
        help="Synthesize a deterministic multi-layer ViT + PatchMerger/Aligner standalone vision.amf bundle (W20)",
    )
    svit.add_argument("--output", required=True, type=pathlib.Path, help="Output vision.amf path")
    svit.add_argument("--family", choices=("qwen_vl", "deepseek_v41"), default="qwen_vl", help="Vision tower family")
    svit.add_argument("--num-layers", type=int, default=4, help="Number of ViT encoder blocks (default: 4)")
    svit.add_argument("--vision-hidden-size", type=int, default=128, help="ViT hidden width (default: 128)")
    svit.add_argument("--vision-intermediate-size", type=int, default=256, help="ViT MLP width (default: 256)")
    svit.add_argument("--out-hidden-size", type=int, default=128, help="LM output hidden width (default: 128)")
    svit.add_argument("--patch-size", type=int, default=14, help="Patch size (default: 14)")
    svit.add_argument("--spatial-merge-size", type=int, default=2, help="Spatial merge size (default: 2)")
    svit.add_argument("--mlp-act", choices=VISION_MLP_ACTS, default="gelu", help="MLP activation (gelu or swiglu)")
    svit.add_argument("--dtype", choices=("bf16", "f16"), default="bf16", help="Base float dtype (default: bf16)")
    svit.add_argument("--quant-profile", choices=QUANT_PROFILES, default="uniform", help="uniform or gsq-rco")
    svit.add_argument("--vision-block-dtype", choices=VISION_BLOCK_DTYPES, default="auto", help="Vision block precision")
    svit.add_argument("--model-id", default="synth-vit-tower", help="Model identifier stored in .amf header")
    svit.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed (default: 42)")
    svit.add_argument("--embed-metadata", action="store_true", help="Embed vision_config JSON inside __meta.config_json")
    return parser


def _resolve_cli_vision_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[bool | None, bool | None]:
    inc = True if (getattr(args, "include_vision", False) or getattr(args, "no_strip_vision", False)) else None
    strip = True if getattr(args, "strip_vision", False) else None
    if inc is True and strip is True:
        parser.error("cannot specify both --include-vision/--no-strip-vision and --strip-vision")
    if strip is True and getattr(args, "vision_sidecar", None) is not None:
        parser.error("cannot specify both --strip-vision and --vision-sidecar")
    return inc, strip


def _is_gguf_input(path: pathlib.Path) -> bool:
    """Return True when `path` is a `.gguf` file/shard or a directory containing only `.gguf` shards."""
    if path.is_file():
        if path.suffix.lower() == ".gguf":
            return True
        with path.open("rb") as fh:
            return fh.read(4) == b"GGUF"
    if path.is_dir():
        has_gguf = any(path.glob("*.gguf"))
        has_sf = any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file()
        return has_gguf and not has_sf
    return path.suffix.lower() == ".gguf"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-ngram":
        return ngram_mod.run_args(args)

    if args.command == "compile-chat-template":
        compiled = compile_chat_template_bytes(
            args.input,
            tokenizer_config_path=args.tokenizer_config,
            config_path=args.config,
            chat_template_path=args.chat_template,
        )
        if compiled is None:
            raise FileNotFoundError(
                f"no chat_template.jinja, tokenizer_config.json, or config.json found for {args.input}"
            )
        pretty = json.dumps(json.loads(compiled.decode("utf-8")), ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(pretty, encoding="utf-8")
            print(f"wrote chat_template_spec JSON to {args.output} ({len(compiled)} compact bytes)")
        else:
            print(pretty, end="")
        return 0

    if args.command == "pack-gguf":
        cold = _parse_cold_experts(args.cold_experts)
        excludes = _parse_csv(args.exclude_tensors)
        started = time.monotonic()
        gguf_stats: dict[str, int] = {}
        metas = pack_amf_from_gguf(
            args.input,
            args.output,
            model_id=args.model_id,
            dtype_mode=args.dtype,
            cold_experts=cold,
            all_experts_cold=args.all_experts_cold,
            exclude_tensors=excludes,
            embed_metadata=args.embed_metadata
            or args.config is not None
            or args.tokenizer is not None
            or args.chat_template is not None,
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            tokenizer_config_path=args.tokenizer_config,
            chat_template_path=args.chat_template,
            vision_block_dtype=args.vision_block_dtype,
            group_size=args.group_size,
            strict_vocab_check=not args.no_strict_vocab,
            deduplicate_recurrent=not args.no_dedup_recurrent,
            stats_out=gguf_stats,
        )
        elapsed = time.monotonic() - started
        total_bytes = args.output.stat().st_size
        by_dtype: dict[str, int] = {}
        for m in metas:
            by_dtype[m.dtype.name.lower()] = by_dtype.get(m.dtype.name.lower(), 0) + 1
        dtype_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items()))
        notes: list[str] = []
        if gguf_stats.get("deduplicated_layers", 0) > 0:
            notes.append(f"dedup_layers={gguf_stats['deduplicated_layers']}")
        note_str = (", " + ", ".join(notes)) if notes else ""
        print(
            f"packed {len(metas)} tensors from GGUF into {args.output} ({total_bytes} bytes) "
            f"in {elapsed:.1f} s [{dtype_summary}{note_str}]"
        )
        return 0

    if args.command == "pack-amf":
        cold = _parse_cold_experts(args.cold_experts)
        excludes = _parse_csv(args.exclude_tensors)
        inc_vis, strip_vis = _resolve_cli_vision_flags(parser, args)
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
        if not use_qwen4_pipeline and _is_gguf_input(args.input):
            gguf_stats: dict[str, int] = {}
            metas = pack_amf_from_gguf(
                args.input,
                args.output,
                model_id=args.model_id,
                dtype_mode=args.dtype,
                cold_experts=cold,
                all_experts_cold=args.all_experts_cold,
                exclude_tensors=excludes,
                embed_metadata=args.embed_metadata
                or args.config is not None
                or args.tokenizer is not None
                or args.chat_template is not None,
                config_path=args.config,
                tokenizer_path=args.tokenizer,
                chat_template_path=args.chat_template,
                vision_block_dtype=args.vision_block_dtype,
                group_size=args.group_size or 64,
                strict_vocab_check=not args.no_strict_vocab,
                stats_out=gguf_stats,
            )
            if gguf_stats.get("deduplicated_layers", 0) > 0:
                extra_notes.append(f"dedup_layers={gguf_stats['deduplicated_layers']}")
        elif use_qwen4_pipeline:
            if (
                args.max_q2_share is not None
                or args.allow_unsafe_q2
                or args.allow_retranscode_lowbit
                or args.fine_rungs
            ):
                parser.error(
                    "--max-q2-share, --allow-unsafe-q2, --allow-retranscode-lowbit, and --fine-rungs "
                    "are not yet supported with qwen4 PLE/MTP/adapter pipeline flags; run `gsq-convert` separately"
                )
            rep_q4 = pack_qwen4_checkpoint(
                args.input,
                args.output,
                model_id=args.model_id,
                group_size=args.group_size or 64,
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
                strip_vision=strip_vis,
                include_vision=inc_vis,
                vision_sidecar=args.vision_sidecar,
                vision_block_dtype=args.vision_block_dtype,
                ple_output=args.ple_output,
                ple_shards=args.ple_shards,
                mtp_output=args.mtp_output,
                loop_adapter_input=args.loop_adapter_input,
                config_output=args.config_output,
            )
            metas = rep_q4.metas
            if args.embed_metadata or args.config is not None or args.tokenizer is not None:
                base_dir = args.input if args.input.is_dir() else args.input.parent
                cfg_p = args.config if args.config is not None else (args.config_output or (base_dir / "config.json"))
                tok_p = args.tokenizer if args.tokenizer is not None else (base_dir / "tokenizer.json")
                metas = embed_metadata_into_amf(
                    args.output,
                    config_bytes=cfg_p.read_bytes() if cfg_p is not None and cfg_p.is_file() else None,
                    tokenizer_bytes=tok_p.read_bytes() if tok_p is not None and tok_p.is_file() else None,
                    output_path=args.output,
                )
            stripped_vision = rep_q4.stripped_vision_tensors
            if rep_q4.vision_amf_path is not None:
                extra_notes.append(f"vision_sidecar={len(rep_q4.vision_metas)}")
            elif rep_q4.included_vision_tensors > 0:
                extra_notes.append(f"vision_tensors={rep_q4.included_vision_tensors}")
            if rep_q4.ple_shard_paths:
                extra_notes.append(f"ple_shards={len(rep_q4.ple_shard_paths)}")
            if rep_q4.mtp_amf_path is not None:
                extra_notes.append(f"mtp_tensors={len(rep_q4.mtp_metas)}")
            if rep_q4.loop_adapter_count > 0:
                extra_notes.append(f"loop_adapters={rep_q4.loop_adapter_count}")
        else:
            stats_map: dict[str, int] = {}
            if args.quant_profile == "gsq-rco":
                enforce_floor, max_q2, allow_lowbit, fine_rungs = _resolve_cli_gsq_safety_flags(
                    parser, args, args.input
                )
            else:
                enforce_floor, max_q2, allow_lowbit, fine_rungs = True, None, True, False
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
                strip_vision=strip_vis,
                include_vision=inc_vis,
                vision_sidecar=args.vision_sidecar,
                vision_block_dtype=args.vision_block_dtype,
                embed_metadata=args.embed_metadata,
                config_path=args.config,
                tokenizer_path=args.tokenizer,
                chat_template_path=args.chat_template,
                imatrix_path=args.imatrix,
                enforce_backbone_floor=enforce_floor,
                max_q2_share=max_q2,
                allow_lowbit_q2_retranscode=allow_lowbit,
                fine_rungs=fine_rungs,
                stats_out=stats_map,
            )
            stripped_vision = stats_map.get("stripped_vision_tensors", 0)
            if stats_map.get("sidecar_vision_tensors", 0) > 0:
                extra_notes.append(f"vision_sidecar={stats_map['sidecar_vision_tensors']}")
            elif stats_map.get("included_vision_tensors", 0) > 0:
                extra_notes.append(f"vision_tensors={stats_map['included_vision_tensors']}")
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
        inc_vis, strip_vis = _resolve_cli_vision_flags(parser, args)
        enforce_floor, max_q2, allow_lowbit, fine_rungs = _resolve_cli_gsq_safety_flags(
            parser, args, args.input
        )
        started = time.monotonic()
        use_vision_wrapper = (
            inc_vis is True
            or args.vision_sidecar is not None
            or args.vision_block_dtype != "auto"
            or (strip_vis is None)
        )
        if use_vision_wrapper:
            vis_stats: dict[str, int] = {}
            metas = gsq_convert_with_vision(
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
                strip_vision=strip_vis,
                include_vision=inc_vis,
                vision_sidecar=args.vision_sidecar,
                vision_block_dtype=args.vision_block_dtype,
                imatrix_path=args.imatrix,
                enforce_backbone_floor=enforce_floor,
                max_q2_share=max_q2,
                allow_lowbit_q2_retranscode=allow_lowbit,
                fine_rungs=fine_rungs,
                stats_out=vis_stats,
            )
            elapsed = time.monotonic() - started
            total_bytes = args.output.stat().st_size
            by_dtype: dict[str, int] = {}
            lut16_cnt = 0
            for m in metas:
                by_dtype[m.dtype.name.lower()] = by_dtype.get(m.dtype.name.lower(), 0) + 1
                if m.quant_param_kind == QuantParamKind.LUT16_CODEBOOK:
                    lut16_cnt += 1
            dtype_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items()))
            lut_note = f", lut16_codebook={lut16_cnt}" if lut16_cnt > 0 else ""
            vis_note = ""
            if vis_stats.get("sidecar_vision_tensors", 0) > 0:
                vis_note = f", vision_sidecar={vis_stats['sidecar_vision_tensors']}"
            elif vis_stats.get("included_vision_tensors", 0) > 0:
                vis_note = f", vision_tensors={vis_stats['included_vision_tensors']}"
            target_label = "auto" if args.target_bits is None else f"{args.target_bits:.2f}b"
            print(
                f"gsq-converted {len(metas)} tensors into {args.output} ({total_bytes} bytes, "
                f"target={target_label}, rco={'on' if not args.no_rco else 'off'}{lut_note}{vis_note}) "
                f"in {elapsed:.1f} s [{dtype_summary}]"
            )
            return 0
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
            strip_vision=True,
            imatrix_path=args.imatrix,
            enforce_backbone_floor=enforce_floor,
            max_q2_share=max_q2,
            allow_lowbit_q2_retranscode=allow_lowbit,
            fine_rungs=fine_rungs,
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
        if profile.calibration_mode != "router_trace":
            print(
                f"WARNING: expert-heat used shallow calibration_mode={profile.calibration_mode!r}; "
                "for accurate per-layer MoE routing heat, capture a trace with "
                "`aminfer decode --dump-router-trace <trace.jsonl>` and pass `--calibration-jsonl <trace.jsonl>`.",
                file=sys.stderr,
            )
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
        if args.kind is not None and args.component is not None and args.kind != args.component:
            parser.error(f"conflicting --kind {args.kind!r} and --component {args.component!r}")
        eff_kind = args.component or args.kind
        if eff_kind is None:
            parser.error("export-coreml requires --kind or --component")
        if eff_kind == "expert-predictor":
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
        elif eff_kind == "attention-qkv":
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
        elif eff_kind == "draft-head":
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
        elif eff_kind == "vision-tower":
            buckets = _parse_int_csv(args.token_buckets)
            if buckets is None:
                buckets = DEFAULT_VISION_TOKEN_BUCKETS
            res = export_vision_tower(
                args.output,
                amf_path=args.amf,
                token_buckets=buckets,
                op=args.op,
                num_layers=args.num_vision_layers,
                vision_hidden_size=args.vision_hidden_size,
                vision_intermediate_size=args.vision_intermediate_size,
                out_hidden_size=args.out_hidden_size,
                spatial_merge_size=args.spatial_merge_size,
                patch_size=args.patch_size,
                t_patch=args.t_patch,
                mlp_act=args.mlp_act,
                seed=args.seed,
                model_id=args.model_id,
            )
        else:
            parser.error(f"unsupported CoreML kind {eff_kind!r}")

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
            embed_metadata=args.embed_metadata,
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

    if args.command == "synth-deepseek41":
        cfg = DEEPSEEK41_PRESETS[args.preset]
        overrides = {}
        if args.num_encoder_layers is not None:
            overrides["num_encoder_layers"] = args.num_encoder_layers
            overrides["csa2_modes"] = ()
        if args.num_decoder_layers is not None:
            overrides["num_decoder_layers"] = args.num_decoder_layers
            overrides["csa2_modes"] = ()
        if args.hidden_size is not None:
            overrides["hidden_size"] = args.hidden_size
        if args.num_experts is not None:
            overrides["num_experts"] = args.num_experts
        if args.encoder_top_k is not None:
            overrides["encoder_moe_top_k"] = args.encoder_top_k
        if args.decoder_top_k is not None:
            overrides["decoder_moe_top_k"] = args.decoder_top_k
        if args.vocab_size is not None:
            overrides["vocab_size"] = args.vocab_size
        if args.kv_lora_rank is not None:
            overrides["kv_lora_rank"] = args.kv_lora_rank
        if args.engram_heads is not None:
            overrides["engram_num_heads"] = args.engram_heads
            base_orders = (2, 3, 4, 5, 2, 3, 4, 5)
            overrides["engram_orders"] = base_orders[: args.engram_heads]
        if args.engram_shards is not None:
            overrides["engram_shards"] = args.engram_shards
        if args.engram_rows is not None:
            overrides["engram_num_rows"] = args.engram_rows
        if args.dtype is not None:
            overrides["dtype"] = args.dtype
        if args.no_dspark:
            overrides["include_dspark"] = False
        if args.model_id is not None:
            overrides["model_id"] = args.model_id
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        cold = _parse_cold_experts(args.cold_experts)
        try:
            bundle = generate_synth_deepseek41_bundle(
                args.output_dir,
                preset=args.preset,
                config=cfg,
                seed=args.seed,
                cold_experts=cold,
                cold_experts_per_layer=args.cold_experts_per_layer,
                world_size=args.world_size,
                partition_strategy=args.partition_strategy,
            )
        except HardwareTierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        shard_note = (
            f", {len(bundle.rank_amf_paths)} peer rank shard(s)"
            if bundle.rank_amf_paths
            else ""
        )
        print(
            f"synthesized deepseek41 ({len(bundle.metas)} backbone tensors, "
            f"{len(bundle.engram_paths)} Engram shard(s), "
            f"{len(bundle.dspark_metas)} DSpark tensors{shard_note}, "
            f"{bundle.total_bytes} bytes) into {bundle.amf_path.parent}"
        )
        return 0

    if args.command == "pack-deepseek41":
        cold = _parse_cold_experts(args.cold_experts)
        inc_vis, strip_vis = _resolve_cli_vision_flags(parser, args)
        try:
            rep = convert_deepseek41_checkpoint(
                args.input,
                args.output,
                model_id=args.model_id,
                target_dtype=args.dtype,
                dtype_mode=args.dtype_mode,
                group_size=args.group_size,
                quant_profile=args.quant_profile,
                target_bits=args.target_bits,
                use_rco=not args.no_rco,
                manifest_path=args.manifest,
                cold_experts=cold,
                all_experts_cold=args.all_experts_cold,
                heat_profile=args.heat_profile,
                cold_experts_per_layer=args.cold_experts_per_layer,
                sme2_pack=args.sme2_pack,
                lut16_sme2=args.lut16_sme2,
                strip_vision=strip_vis,
                include_vision=inc_vis,
                vision_sidecar=args.vision_sidecar,
                vision_block_dtype=args.vision_block_dtype,
                engram_output=args.engram_output,
                engram_shards=args.engram_shards,
                dspark_output=args.dspark_output,
                config_output=args.config_output,
                target_tier=args.target_tier,
                host_ram_gib=args.host_ram_gib,
                world_size=args.world_size,
                rank_dir=args.rank_dir,
                partition_strategy=args.partition_strategy,
                allow_tier_override=args.allow_tier_override,
                workers=args.workers,
            )
        except HardwareTierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        vis_note = ""
        if rep.vision_amf is not None:
            vis_note = f", {len(rep.vision_metas)} vision sidecar tensors"
        elif rep.included_vision_tensors > 0:
            vis_note = f", {rep.included_vision_tensors} vision tensors"
        print(
            f"packed deepseek41 ({len(rep.backbone_metas)} backbone tensors, "
            f"{len(rep.engram_paths)} Engram shard(s), "
            f"{len(rep.dspark_metas)} DSpark tensors{vis_note}, "
            f"tier={rep.tier_check.target_tier} world_size={rep.tier_check.world_size}) "
            f"into {rep.backbone_amf}"
        )
        return 0

    if args.command == "synth-vit":
        started = time.monotonic()
        vit_cfg = SyntheticVitConfig(
            family=args.family,
            num_vision_layers=args.num_layers,
            vision_hidden_size=args.vision_hidden_size,
            vision_intermediate_size=args.vision_intermediate_size,
            out_hidden_size=args.out_hidden_size,
            patch_size=args.patch_size,
            spatial_merge_size=args.spatial_merge_size,
            mlp_act=args.mlp_act,
            dtype=args.dtype,
            model_id=args.model_id,
        )
        metas = generate_synth_vit_amf(
            args.output,
            vit_cfg,
            seed=args.seed,
            quant_profile=args.quant_profile,
            vision_block_dtype=args.vision_block_dtype,
            embed_metadata=args.embed_metadata,
        )
        elapsed = time.monotonic() - started
        print(
            f"synthesized {args.family} vision tower ({len(metas)} tensors, "
            f"{args.output.stat().st_size} bytes) -> {args.output} in {elapsed:.2f} s"
        )
        return 0

    if args.command == "partition-experts":
        rep = partition_deepseek41_amf_shards(
            args.input,
            args.world_size,
            args.output_dir,
            strategy=args.strategy,
            replicate_backbone=args.replicate_backbone,
        )
        print(
            f"partitioned {rep.num_experts} experts across {rep.world_size} rank(s) "
            f"({rep.strategy}) into {rep.output_dir} (manifest={rep.manifest_path.name})"
        )
        return 0

    if args.command == "inspect-metadata":
        base = args.input.parent if (args.input.is_file() and args.input.name == "config.json") else args.input
        cfg_file = base / "config.json"
        if not cfg_file.is_file():
            print(f"error: missing config.json in {base}", file=sys.stderr)
            return 2
        raw_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        arch = args.arch
        if arch == "auto":
            mt = str(raw_cfg.get("model_type", "")).lower()
            tc_mt = str(raw_cfg.get("text_config", {}).get("model_type", "")).lower() if isinstance(raw_cfg.get("text_config"), dict) else ""
            combined_mt = f"{mt} {tc_mt}"
            if "deepseek" in combined_mt:
                arch = "deepseek41"
            elif "qwen" in combined_mt:
                arch = "qwen4_exp"
            else:
                print(f"error: cannot auto-detect arch from model_type={mt!r}; pass --arch", file=sys.stderr)
                return 2
        try:
            rep = (
                validate_deepseek41_metadata_dir(base, host_ram_gib=args.host_ram_gib)
                if arch == "deepseek41"
                else validate_qwen4_metadata_dir(base, host_ram_gib=args.host_ram_gib)
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        data = rep.to_dict()
        if args.as_json:
            print(json.dumps(data, indent=2))
        elif arch == "qwen4_exp":
            gib = rep.index_total_size_bytes / (1024**3)
            q3_gib = rep.estimated_backbone_q3_bytes / (1024**3)
            print(
                f"qwen4_exp metadata OK ({rep.model_type}): "
                f"layers={rep.num_hidden_layers} (GDN={rep.gdn_layers}, QSA={rep.qsa_layers}), "
                f"hidden={rep.hidden_size}, experts={rep.num_experts} (top-{rep.num_experts_per_tok}, shared={rep.has_shared_expert}), "
                f"hc={rep.hc_count}, PLE shards={rep.ple_shards_detected}, MTP layers={rep.mtp_layers}, "
                f"KV={rep.kv_bytes_per_token_fp16} B/tok (FP16), GDN state={rep.gdn_state_bytes_fp32 / (1024**2):.1f} MiB"
            )
            print(
                f"  index: {rep.total_index_entries} tensors / {rep.shard_file_count} shards ({gib:.2f} GiB raw, "
                f"est q3 backbone={q3_gib:.2f} GiB); buckets: backbone={rep.backbone_tensor_count}, "
                f"vision={rep.vision_tensor_count}, ple={rep.ple_tensor_count}, mtp={rep.mtp_tensor_count}, "
                f"names>47B={len(rep.long_backbone_names)}"
            )
            for tier, info in rep.tier_admission.items():
                print(
                    f"  tier {tier}: budget={info['per_rank_budget_gib']} GiB/rank, "
                    f"q3_wired={info['per_rank_q3_wired_gib']} GiB/rank -> {info['mode']}"
                )
        else:
            gib = rep.index_total_size_bytes / (1024**3)
            q3_gib = rep.estimated_backbone_q3_bytes / (1024**3)
            print(
                f"deepseek41 metadata OK ({rep.model_type}): "
                f"layers={rep.num_hidden_layers} (enc={rep.num_encoder_layers}, dec={rep.num_decoder_layers}), "
                f"hidden={rep.hidden_size}, experts={rep.num_routed_experts} (top-{rep.num_experts_per_tok}, shared={rep.num_shared_experts}), "
                f"mHC={rep.mhc_num_streams}, Engram layers={list(rep.engram_layer_ids)} ({rep.engram_total_embeddings:,} rows), "
                f"DSpark/MTP={rep.dspark_layers}L/{rep.dspark_n_routed_experts}E, "
                f"KV={rep.kv_bytes_per_token_bf16} B/tok BF16 / {rep.kv_bytes_per_token_fp4} B/tok FP4"
            )
            print(
                f"  index: {rep.total_index_entries} tensors / {rep.shard_file_count} shards ({gib:.2f} GiB raw, "
                f"est q3 backbone={q3_gib:.2f} GiB); buckets: backbone={rep.backbone_tensor_count}, "
                f"vision={rep.vision_tensor_count}, engram={rep.engram_tensor_count}, dspark_mtp={rep.dspark_mtp_tensor_count}, "
                f"routed_exp={rep.routed_expert_tensor_count} ({rep.routed_expert_ids_detected} IDs), names>47B={len(rep.long_backbone_names)}"
            )
            for tier, info in rep.tier_admission.items():
                status = "FITS" if info["fits_all_wired_q3"] else "EXCEEDS"
                print(
                    f"  tier {tier} (ws={info['world_size']}): budget={info['per_rank_budget_gib']} GiB/rank, "
                    f"q3_wired={info['per_rank_q3_wired_gib']} GiB/rank -> {status}"
                )
        return 0

    if args.command == "pack-lora":
        started = time.monotonic()
        if args.synthetic:
            targets = _parse_csv(args.target_modules) or ["q_proj", "v_proj"]
            rank = args.rank if args.rank is not None else 4
            alpha = args.lora_alpha if args.lora_alpha is not None else float(rank * 2)
            rep = generate_synthetic_lora(
                args.output,
                adapter_id=args.adapter_id or args.output.stem,
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                q_dim=args.q_dim,
                kv_dim=args.kv_dim,
                intermediate_size=args.intermediate_size,
                rank=rank,
                lora_alpha=alpha,
                target_modules=targets,
                layout=args.layout,
                dtype=args.dtype,
                seed=args.seed,
                zero_init_b=args.zero_init_b,
            )
        else:
            if args.input is None:
                parser.error("pack-lora requires --input <peft_dir_or_safetensors> unless --synthetic is set")
            rep = pack_lora_from_peft(
                args.input,
                args.output,
                adapter_id=args.adapter_id,
                adapter_config_path=args.adapter_config,
                layout=args.layout,
                dtype=args.dtype,
                rank_override=args.rank,
                alpha_override=args.lora_alpha,
            )
        elapsed = time.monotonic() - started
        l2_tag = "fits_4MiB_L2" if rep.fits_l2_4mib else "EXCEEDS_4MiB_L2"
        print(
            f"packed LoRA {rep.adapter_id!r} ({rep.num_pairs} pair(s), rank={rep.rank}, "
            f"alpha={rep.lora_alpha:g}, scale={rep.scale:g}, layout={rep.layout}, dtype={rep.dtype}, "
            f"wired={rep.total_wired_bytes} B [{l2_tag}], file={rep.total_file_bytes} B) "
            f"-> {rep.output_path} in {elapsed:.2f} s"
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
