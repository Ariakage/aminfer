"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import pack_amf_from_safetensors
from aminfer_tools.synth import PRESETS, generate_qwen3_bundle


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
        help="Pack MLX affine quantized safetensors into a 16 KiB-aligned .amf v0 file",
    )
    pack.add_argument("--input", required=True, type=pathlib.Path, help="Input .safetensors path")
    pack.add_argument("--output", required=True, type=pathlib.Path, help="Output .amf path")
    pack.add_argument("--model-id", default="aminfer-model", help="Model identifier stored in header")
    pack.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
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
        metas = pack_amf_from_safetensors(
            args.input,
            args.output,
            model_id=args.model_id,
            group_size=args.group_size,
            cold_experts=cold,
            all_experts_cold=args.all_experts_cold,
        )
        total_bytes = args.output.stat().st_size
        print(f"packed {len(metas)} tensors into {args.output} ({total_bytes} bytes)")
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
