"""Regression tests for F18 and F24.3 offline toolchain security and UTF-8 fixes."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import tempfile
import unittest

from aminfer_tools import amf, chat_template, gguf, hy4, lora


class TestReviewSecurityFixes(unittest.TestCase):
    tmp_dir: pathlib.Path

    def setUp(self) -> None:
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="aminfer_tools_sec_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_safetensors(
        self,
        path: pathlib.Path,
        header: dict[str, object],
        payload: bytes = b"",
    ) -> None:
        hdr_bytes = json.dumps(header).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + payload)

    def test_f18_resolve_checkpoint_rejects_traversal_abs_and_escaping_symlinks(self) -> None:
        outside_file = self.tmp_dir / "outside_secret.txt"
        outside_file.write_bytes(b"SECRET_OUTSIDE_DATA")

        ckpt = self.tmp_dir / "ckpt"
        ckpt.mkdir()
        valid_shard = ckpt / "model-00001-of-00001.safetensors"
        self._write_safetensors(
            valid_shard,
            {"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}},
            b"\x00\x00\x00\x00",
        )

        # 1. Parent directory `../` traversal (Appendix A.5).
        idx_rel = ckpt / "model.safetensors.index.json"
        idx_rel.write_text(
            json.dumps({"weight_map": {"w": "../outside_secret.txt"}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            amf.resolve_checkpoint(ckpt)

        # 2. Absolute path injection in weight_map (Appendix A.5).
        idx_rel.write_text(
            json.dumps({"weight_map": {"w": str(outside_file)}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            amf.resolve_checkpoint(idx_rel)

        # 3. Symlink inside checkpoint directory escaping root_resolved.
        link_shard = ckpt / "escaped.safetensors"
        link_shard.symlink_to(outside_file)
        idx_rel.write_text(
            json.dumps({"weight_map": {"w": "escaped.safetensors"}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            amf.resolve_checkpoint(idx_rel)

        # 4. Symlink config.json escaping root_resolved.
        (ckpt / "config.json").symlink_to(outside_file)
        idx_rel.write_text(
            json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            amf.resolve_checkpoint(ckpt)

        # 5. Legitimate nested relative path inside checkpoint stays accepted.
        (ckpt / "config.json").unlink()
        sub = ckpt / "shards"
        sub.mkdir()
        sub_shard = sub / "part0.safetensors"
        shutil.copy2(valid_shard, sub_shard)
        idx_rel.write_text(
            json.dumps({"weight_map": {"w": "shards/part0.safetensors"}}),
            encoding="utf-8",
        )
        resolved = amf.resolve_checkpoint(ckpt)
        self.assertEqual(resolved.shard_paths, (sub_shard.resolve(),))

    def test_f18_safetensors_negative_offsets_and_negative_shape_rejected(self) -> None:
        # Appendix A.5: data_offsets: [-16, 0] leaked JSON header bytes.
        bad_off = self.tmp_dir / "neg_offset.safetensors"
        self._write_safetensors(
            bad_off,
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [-16, 0]}},
        )
        with self.assertRaises(ValueError):
            amf._read_shard_header(bad_off)

        # Negative dimension in shape.
        bad_dim = self.tmp_dir / "neg_shape.safetensors"
        self._write_safetensors(
            bad_dim,
            {"w": {"dtype": "F32", "shape": [-1, 4], "data_offsets": [0, 0]}},
        )
        with self.assertRaises(ValueError):
            amf._read_shard_header(bad_dim)

    def test_f18_http_range_reader_rejects_non_http_schemes(self) -> None:
        for bad_url in (
            "file:///etc/passwd",
            "ftp://example.com/model.safetensors",
            "gopher://127.0.0.1/1",
            "http:///missing-host.safetensors",
            "relative/path.safetensors",
        ):
            with self.subTest(bad_url=bad_url), self.assertRaises(ValueError):
                amf.HttpRangeStreamReader(bad_url)

        ok_http = amf.HttpRangeStreamReader("http://127.0.0.1:8080/s.safetensors")
        ok_https = amf.HttpRangeStreamReader("HTTPS://example.com/s.safetensors")
        self.assertEqual(ok_http.url, "http://127.0.0.1:8080/s.safetensors")
        self.assertEqual(ok_https.url, "HTTPS://example.com/s.safetensors")

    def test_f18_lora_path_containment_and_negative_offsets(self) -> None:
        outside = self.tmp_dir / "outside.safetensors"
        self._write_safetensors(
            outside,
            {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [-8, 0]}},
        )
        lora_dir = self.tmp_dir / "lora_dir"
        lora_dir.mkdir()
        (lora_dir / "adapter_model.safetensors").symlink_to(outside)

        with self.assertRaises(ValueError):
            lora.pack_lora_from_peft(lora_dir, self.tmp_dir / "out.amlora")

        with self.assertRaises(ValueError):
            lora.pack_lora_from_peft(outside, self.tmp_dir / "out2.amlora")

    def test_f18_hy4_oversized_header_and_negative_offsets_rejected(self) -> None:
        hy_dir = self.tmp_dir / "hy4_bad"
        hy_dir.mkdir()
        (hy_dir / "config.json").write_text(
            json.dumps({"model_type": "hy_v4", "num_hidden_layers": 1, "hidden_size": 64}),
            encoding="utf-8",
        )
        sf_path = hy_dir / "model.safetensors"

        # 1. Oversized header (> file_size and > 256 MiB) rejected before fh.read(hlen).
        sf_path.write_bytes(struct.pack("<Q", 512 * 1024 * 1024) + b"{}")
        with self.assertRaises(ValueError):
            hy4.validate_hf_metadata_dir(hy_dir)

        # 2. Negative data_offsets rejected.
        self._write_safetensors(
            sf_path,
            {"model.norm.weight": {"dtype": "F16", "shape": [4], "data_offsets": [-8, 0]}},
        )
        with self.assertRaises(ValueError):
            hy4.validate_hf_metadata_dir(hy_dir)

        # 3. Negative shape dimension rejected.
        self._write_safetensors(
            sf_path,
            {"model.norm.weight": {"dtype": "F16", "shape": [-4], "data_offsets": [0, 8]}},
            b"\x00" * 8,
        )
        with self.assertRaises(ValueError):
            hy4.validate_hf_metadata_dir(hy_dir)

    def test_f18_gguf_nested_array_depth_and_element_budget_rejected(self) -> None:
        # 1. Deeply nested GGUF array (depth > 4) triggers bounded recursion guard.
        deep_buf = bytearray()
        deep_buf += gguf.GGUF_MAGIC
        deep_buf += struct.pack("<IQQ", 3, 0, 1)  # version=3, tensor_count=0, kv_count=1
        key = b"evil.deep_array"
        deep_buf += struct.pack("<Q", len(key)) + key
        deep_buf += struct.pack("<I", gguf._GGUF_TYPE_ARRAY)
        for _ in range(5):
            deep_buf += struct.pack("<IQ", gguf._GGUF_TYPE_ARRAY, 1)
        deep_buf += struct.pack("<IQI", 4, 1, 42)  # inner uint32[1] = [42]
        while len(deep_buf) < 64:
            deep_buf += b"\x00"

        deep_path = self.tmp_dir / "deep_bomb.gguf"
        deep_path.write_bytes(bytes(deep_buf))
        with self.assertRaisesRegex(ValueError, "depth"):
            gguf.read_gguf_header(deep_path)

        # 2. Cumulative nested array element budget > 10_000_000 rejected without OOM.
        budget_buf = bytearray()
        budget_buf += gguf.GGUF_MAGIC
        budget_buf += struct.pack("<IQQ", 3, 0, 1)
        key2 = b"evil.elem_bomb"
        budget_buf += struct.pack("<Q", len(key2)) + key2
        budget_buf += struct.pack("<I", gguf._GGUF_TYPE_ARRAY)
        budget_buf += struct.pack("<IQ", gguf._GGUF_TYPE_ARRAY, 3)  # 3 nested arrays
        budget_buf += struct.pack("<IQ", 4, 6_000_000)  # first claims 6M uint32s
        while len(budget_buf) < 64:
            budget_buf += b"\x00"

        budget_path = self.tmp_dir / "elem_bomb.gguf"
        budget_path.write_bytes(bytes(budget_buf))
        with self.assertRaises(ValueError):
            gguf.read_gguf_header(budget_path)

    def test_f24_3_chat_template_unescape_preserves_utf8_hy4_markers(self) -> None:
        # Appendix A.5 case 3: fullwidth vertical bar U+FF5C must not turn into Latin-1 mojibake.
        tpl = (
            "{%- set HYTK = ':opensource' -%}"
            "{%- set bos = '<｜hy_start{}｜>'.format(HYTK) -%}"
            "{%- set user_hdr = '<｜hy_User｜>\\n' -%}"
            "{%- set sys_hdr = '<｜hy_place▁holder▁no▁2｜>\\t\"quoted\"\\'\\\\' -%}"
        )
        vars_map = chat_template._resolve_jinja_string_vars(tpl)
        self.assertEqual(vars_map["bos"], "<\uff5chy_start:opensource\uff5c>")
        self.assertEqual(vars_map["user_hdr"], "<\uff5chy_User\uff5c>\n")
        self.assertEqual(vars_map["sys_hdr"], "<\uff5chy_place\u2581holder\u2581no\u25812\uff5c>\t\"quoted\"'\\")

        spec = chat_template.compile_chat_template_spec(
            jinja_text=tpl,
            tokenizer_config={"bos_token": "<｜hy_start:opensource｜>", "eos_token": "<｜hy_end｜>"},
            model_config={"model_type": "hy_v4"},
        )
        self.assertEqual(spec["dialect"], "hy4")
        self.assertEqual(spec["bos_token"], "<\uff5chy_start:opensource\uff5c>")


if __name__ == "__main__":
    unittest.main()
