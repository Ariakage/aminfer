"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import DTYPE_MODES, pack_amf_from_safetensors
from aminfer_tools.synth import (
    DRAFT_DTYPES,
    DRAFT_FORMATS,
    DRAFT_PRESETS,
    PRESETS,
    generate_draft_bundle,
    generate_qwen3_bundle,
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
        "--all-experts-cold",
        action="store_true",
        help="Mark every expert tensor as cold_expert",
    )
    pack.add_argument(
        "--exclude-tensors",
        default=None,
        help="Comma-separated tensor names to omit from the output .amf (e.g. embed_tokens.weight)",
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

    sub.add_parser(
        "expert-heat",
        help="Compute offline MoE expert activation heat statistics (planned: M4)",
    )
    sub.add_parser(
        "export-coreml",
        help="Export CoreML .mlpackage graphs for the Apple Neural Engine (planned: M4)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "pack-amf":
        cold = _parse_cold_experts(args.cold_experts)
        excludes = _parse_csv(args.exclude_tensors)
        started = time.monotonic()
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
        )
        elapsed = time.monotonic() - started
        total_bytes = args.output.stat().st_size
        by_dtype: dict[str, int] = {}
        for m in metas:
            by_dtype[m.dtype.name.lower()] = by_dtype.get(m.dtype.name.lower(), 0) + 1
        dtype_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items()))
        print(
            f"packed {len(metas)} tensors into {args.output} ({total_bytes} bytes) "
            f"in {elapsed:.1f} s [{dtype_summary}]"
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

    print(
        f"aminfer-tools: '{args.command}' is not implemented yet; "
        "see docs/status.md for target milestones.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
