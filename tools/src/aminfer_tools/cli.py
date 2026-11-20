"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import pack_amf_from_safetensors


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

    print(
        f"aminfer-tools: '{args.command}' is not implemented yet; "
        "see docs/status.md for target milestones.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
