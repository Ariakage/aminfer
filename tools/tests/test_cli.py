"""Tests for the offline toolchain scaffold."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.cli import align_up, main


class CliTests(unittest.TestCase):
    def test_page_size_constant(self) -> None:
        self.assertEqual(APPLE_PAGE_SIZE, 16384)

    def test_align_up(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 16384)
        self.assertEqual(align_up(16384), 16384)
        self.assertEqual(align_up(16385), 32768)
        with self.assertRaises(ValueError):
            align_up(-1)

    def test_help_exits_zero(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        self.assertIn("pack-amf", text)
        self.assertIn("synth-qwen3", text)
        self.assertIn("synth-draft", text)
        self.assertIn("expert-heat", text)
        self.assertIn("export-coreml", text)

    def test_pack_amf_help_lists_m25_options(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["pack-amf", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        for flag in ("--dtype", "--tie-word-embeddings", "--keep-tied-lm-head", "--exclude-tensors", "index.json"):
            self.assertIn(flag, text)
        for mode in ("auto", "f16", "bf16"):
            self.assertIn(mode, text)

    def test_synth_draft_help_lists_m3_formats(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["synth-draft", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        for token in ("--format", "--preset", "eagle3", "eagle2", "mtp", "--output-dir"):
            self.assertIn(token, text)

    def test_planned_subcommands_report_unimplemented(self) -> None:
        for cmd in ("expert-heat", "export-coreml"):
            err = io.StringIO()
            with redirect_stderr(err):
                code = main([cmd])
            self.assertEqual(code, 2)
            self.assertIn("docs/status.md", err.getvalue())


if __name__ == "__main__":
    unittest.main()
