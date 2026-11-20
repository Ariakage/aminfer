"""Command-line entry point for the offline aminfer toolchain."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aminfer_tools import APPLE_PAGE_SIZE


def align_up(length: int, alignment: int = APPLE_PAGE_SIZE) -> int:
    """Round a non-negative byte length up to a multiple of `alignment`."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a positive power of two")
    return (length + alignment - 1) & ~(alignment - 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aminfer-tools",
        description=(
            "Offline preparation tools for aminfer (never used at inference runtime)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "pack-amf",
        help="Pack quantized weights into 16 KiB-aligned .amf files (planned: M1)",
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
    print(
        f"aminfer-tools: '{args.command}' is not implemented yet; "
        "see docs/status.md for target milestones.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
