"""Unit & integration tests for M9.5 W30 universal multi-architecture PEFT LoRA & DoRA `.amlora` packing."""

from __future__ import annotations

import json
import os
import pathlib
import struct
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    FLAG_EMBEDDED_METADATA,
    META_ADAPTER_CONFIG_JSON,
    DType,
    TensorKind,
    TensorLayout,
    load_amf_metadata,
    read_amf_metadata,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.lora import (
    LORA_DORA_TENSOR_SUFFIX,
    TARGET_MODULE_SUFFIXES,
    generate_synthetic_lora,
    pack_lora_from_peft,
)


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    cursor = 0
    chunks: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(raw)],
        }
        chunks.append(raw)
        cursor += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for ch in chunks:
            fh.write(ch)


def _f32_to_bf16_bytes(vals: list[float]) -> bytes:
    raw = struct.pack(f"<{len(vals)}f", *vals)
    out = bytearray(len(vals) * 2)
    for i in range(len(vals)):
        bits = struct.unpack_from("<I", raw, i * 4)[0]
        rounding = 0x7FFF + ((bits >> 16) & 1)
        struct.pack_into("<H", out, i * 2, ((bits + rounding) >> 16) & 0xFFFF)
    return bytes(out)


class TestM95UniversalLora(unittest.TestCase):
    def test_universal_prefixes_and_module_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            peft_dir = root / "universal_peft"
            peft_dir.mkdir()

            rank = 8
            k_dim = 64
            n_dim = 64
            a_f16 = struct.pack(f"<{rank * k_dim}e", *[0.05] * (rank * k_dim))
            b_f16 = struct.pack(f"<{n_dim * rank}e", *[0.02] * (n_dim * rank))

            cases = [
                ("base_model.model.model.llm.layers.0.attn.wq_du", "layers.0.self_attn.q_proj"),
                ("base_model.model.model.language_model.layers.0.self_attn.k_proj.linear", "layers.0.self_attn.k_proj"),
                ("base_model.model.model.language_model.layers.0.self_attn.o_proj.gate_proj", "layers.0.self_attn.gate_attn"),
                ("base_model.model.model.language_model.layers.0.self_attn.o_proj.linear", "layers.0.self_attn.o_proj"),
                ("base_model.model.model.layers.1.attn.wv_dv", "layers.1.self_attn.v_proj"),
                ("base_model.model.layers.1.attn.wr_du", "layers.1.self_attn.wr_du"),
                ("model.llm.layers.2.attn.wqkv", "layers.2.self_attn.qkv_proj"),
                ("model.layers.2.ffn.w1", "layers.2.mlp.gate_proj"),
                ("layers.2.ffn.w3", "layers.2.mlp.up_proj"),
                ("layers.2.ffn.w2", "layers.2.mlp.down_proj"),
                ("base_model.model.model.layers.3.ffn.w13", "layers.3.mlp.gate_up_proj"),
                ("base_model.model.model.layers.3.self_attn.q_a_proj", "layers.3.self_attn.q_a_proj"),
                ("base_model.model.model.layers.3.self_attn.q_b_proj", "layers.3.self_attn.q_b_proj"),
                ("base_model.model.model.layers.3.self_attn.kv_a_proj_with_mqa", "layers.3.self_attn.kv_a_proj"),
                ("base_model.model.model.layers.3.self_attn.kv_b_proj", "layers.3.self_attn.kv_b_proj"),
                ("base_model.model.model.layers.3.self_attn.eh_proj", "layers.3.self_attn.eh_proj"),
            ]

            tensors: dict[str, tuple[str, list[int], bytes]] = {}
            for raw_prefix, _ in cases:
                tensors[f"{raw_prefix}.lora_A.weight"] = ("F16", [rank, k_dim], a_f16)
                tensors[f"{raw_prefix}.lora_B.default.weight"] = ("F16", [n_dim, rank], b_f16)

            (peft_dir / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 16}),
                encoding="utf-8",
            )
            _write_safetensors(peft_dir / "adapter_model.safetensors", tensors)

            out_path = root / "universal.amlora"
            rep = pack_lora_from_peft(peft_dir, out_path, layout="row_major", dtype="f16")
            self.assertEqual(rep.num_layers, 4)
            self.assertEqual(rep.num_pairs, len(cases))
            self.assertEqual(set(rep.target_modules), set(TARGET_MODULE_SUFFIXES.keys()))
            self.assertFalse(rep.use_dora)
            self.assertEqual(out_path.stat().st_size % APPLE_PAGE_SIZE, 0)

            _, metas = load_amf_metadata(out_path)
            by_name = {m.name: m for m in metas}
            for _, canonical_stem in cases:
                self.assertIn(f"{canonical_stem}.lora_a", by_name)
                self.assertIn(f"{canonical_stem}.lora_b", by_name)

    def test_auto_layout_selection_and_rank64_support(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)

            # 1. rank=16, K=64 (% 64 == 0) + layout="auto" -> sme2_panel64
            p16 = root / "auto_r16.amlora"
            rep16 = generate_synthetic_lora(
                p16,
                num_layers=1,
                hidden_size=64,
                rank=16,
                lora_alpha=32.0,
                layout="auto",
            )
            self.assertEqual(rep16.layout, "sme2_panel64")

            # 2. rank=32 (> 16), K=64 + layout="auto" -> compact row_major
            p32 = root / "auto_r32.amlora"
            rep32 = generate_synthetic_lora(
                p32,
                num_layers=1,
                hidden_size=64,
                rank=32,
                lora_alpha=64.0,
                layout="auto",
            )
            self.assertEqual(rep32.layout, "row_major")

            # 3. rank=8, K=80 (% 64 != 0) + layout="auto" -> compact row_major
            pk80 = root / "auto_k80.amlora"
            rep80 = generate_synthetic_lora(
                pk80,
                num_layers=1,
                hidden_size=80,
                intermediate_size=160,
                rank=8,
                lora_alpha=16.0,
                layout="auto",
            )
            self.assertEqual(rep80.layout, "row_major")

            # 4. Explicit sme2_panel64 with K % 64 != 0 raises ValueError
            with self.assertRaises(ValueError):
                generate_synthetic_lora(
                    root / "bad_sme2.amlora",
                    num_layers=1,
                    hidden_size=80,
                    rank=8,
                    layout="sme2_panel64",
                )

            # 5. Explicit sme2_panel64 with rank=64 and K=64 succeeds
            p64 = root / "sme_r64.amlora"
            rep64 = generate_synthetic_lora(
                p64,
                num_layers=1,
                hidden_size=64,
                rank=64,
                lora_alpha=64.0,
                layout="sme2_panel64",
            )
            self.assertEqual(rep64.layout, "sme2_panel64")
            self.assertEqual(rep64.rank, 64)

    def test_dora_packing_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            peft_dir = root / "peft_dora"
            peft_dir.mkdir()

            rank = 4
            k_dim = 64
            n_dim = 48
            a_f32 = struct.pack(f"<{rank * k_dim}f", *[0.1] * (rank * k_dim))
            b_f32 = struct.pack(f"<{n_dim * rank}f", *[-0.05] * (n_dim * rank))
            mag_vals = [0.5 + 0.01 * i for i in range(n_dim)]
            mag_f32 = struct.pack(f"<{n_dim}f", *mag_vals)
            mag_f16 = struct.pack(f"<{n_dim}e", *mag_vals)
            mag_bf16 = _f32_to_bf16_bytes(mag_vals)

            (peft_dir / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": rank,
                        "lora_alpha": 8,
                        "use_dora": True,
                    }
                ),
                encoding="utf-8",
            )
            _write_safetensors(
                peft_dir / "adapter_model.safetensors",
                {
                    # 1) Bare .lora_magnitude_vector (F32, matches real Inkling2 abliteration/dora)
                    "base_model.model.model.language_model.layers.4.self_attn.o_proj.linear.lora_A.weight": (
                        "F32",
                        [rank, k_dim],
                        a_f32,
                    ),
                    "base_model.model.model.language_model.layers.4.self_attn.o_proj.linear.lora_B.weight": (
                        "F32",
                        [n_dim, rank],
                        b_f32,
                    ),
                    "base_model.model.model.language_model.layers.4.self_attn.o_proj.linear.lora_magnitude_vector": (
                        "F32",
                        [n_dim],
                        mag_f32,
                    ),
                    # 2) .lora_magnitude_vector.default.weight (F16) with 3-digit layer index (<= 47 B name check)
                    "base_model.model.model.layers.120.self_attn.o_proj.gate_proj.lora_A.default.weight": (
                        "F32",
                        [rank, k_dim],
                        a_f32,
                    ),
                    "base_model.model.model.layers.120.self_attn.o_proj.gate_proj.lora_B.default.weight": (
                        "F32",
                        [n_dim, rank],
                        b_f32,
                    ),
                    "base_model.model.model.layers.120.self_attn.o_proj.gate_proj.lora_magnitude_vector.default.weight": (
                        "F16",
                        [n_dim],
                        mag_f16,
                    ),
                    # 3) .weight_m_wdecomp.weight (BF16)
                    "model.llm.layers.121.attn.wr_du.lora_A.weight": ("F32", [rank, k_dim], a_f32),
                    "model.llm.layers.121.attn.wr_du.lora_B.weight": ("F32", [n_dim, rank], b_f32),
                    "model.llm.layers.121.attn.wr_du.weight_m_wdecomp.weight": ("BF16", [n_dim], mag_bf16),
                },
            )

            out_path = root / "dora_out.amlora"
            rep = pack_lora_from_peft(peft_dir, out_path, layout="auto", dtype="f16")
            self.assertTrue(rep.use_dora)
            self.assertEqual(rep.layout, "sme2_panel64")
            self.assertEqual(rep.num_pairs, 3)

            hdr, metas = load_amf_metadata(out_path)
            self.assertEqual(hdr.flags & FLAG_EMBEDDED_METADATA, FLAG_EMBEDDED_METADATA)
            cfg = json.loads(read_amf_metadata(out_path, META_ADAPTER_CONFIG_JSON).decode("utf-8"))
            self.assertTrue(cfg["use_dora"])

            raw_bytes = out_path.read_bytes()
            by_name = {m.name: m for m in metas}
            for canonical_stem in (
                "layers.4.self_attn.o_proj",
                "layers.120.self_attn.gate_attn",
                "layers.121.self_attn.wr_du",
            ):
                mag_name = f"{canonical_stem}.{LORA_DORA_TENSOR_SUFFIX}"
                self.assertLessEqual(len(mag_name.encode("utf-8")), 47)
                self.assertIn(mag_name, by_name)
                mm = by_name[mag_name]
                self.assertEqual(mm.kind, TensorKind.LOOP_ADAPTER)
                self.assertEqual(mm.dtype, DType.F32)
                self.assertEqual(mm.layout, TensorLayout.ROW_MAJOR)
                self.assertEqual(mm.shape, (n_dim,))
                unpacked = struct.unpack(f"<{n_dim}f", raw_bytes[mm.blob_offset : mm.blob_offset + mm.data_bytes])
                for got, exp in zip(unpacked, mag_vals, strict=True):
                    self.assertAlmostEqual(got, exp, places=2)

            # Negative tests: missing magnitude vector when use_dora=true
            bad_missing = root / "bad_missing"
            bad_missing.mkdir()
            (bad_missing / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 8, "use_dora": True}),
                encoding="utf-8",
            )
            _write_safetensors(
                bad_missing / "adapter_model.safetensors",
                {
                    "layers.0.self_attn.q_proj.lora_A.weight": ("F32", [rank, k_dim], a_f32),
                    "layers.0.self_attn.q_proj.lora_B.weight": ("F32", [n_dim, rank], b_f32),
                },
            )
            with self.assertRaisesRegex(ValueError, "missing DoRA magnitude"):
                pack_lora_from_peft(bad_missing, root / "bad_missing.amlora")

            # Negative test: shape mismatch (N - 1)
            bad_shape = root / "bad_shape"
            bad_shape.mkdir()
            (bad_shape / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 8, "use_dora": True}),
                encoding="utf-8",
            )
            _write_safetensors(
                bad_shape / "adapter_model.safetensors",
                {
                    "layers.0.self_attn.q_proj.lora_A.weight": ("F32", [rank, k_dim], a_f32),
                    "layers.0.self_attn.q_proj.lora_B.weight": ("F32", [n_dim, rank], b_f32),
                    "layers.0.self_attn.q_proj.lora_magnitude_vector": ("F32", [n_dim - 1], mag_f32[:-4]),
                },
            )
            with self.assertRaisesRegex(ValueError, "magnitude shape mismatch"):
                pack_lora_from_peft(bad_shape, root / "bad_shape.amlora")

    def test_synthetic_dora_and_cli_synth_lora(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            out1 = root / "pack_dora.amlora"
            rc1 = cli_main(
                [
                    "pack-lora",
                    "--synthetic",
                    "--dora",
                    "--output",
                    str(out1),
                    "--rank",
                    "32",
                    "--target-modules",
                    "q_proj,o_proj",
                    "--extra-modules",
                    "gate_attn,wr_du",
                    "--layout",
                    "auto",
                ]
            )
            self.assertEqual(rc1, 0)
            _, metas1 = load_amf_metadata(out1)
            cfg1 = json.loads(read_amf_metadata(out1, META_ADAPTER_CONFIG_JSON).decode("utf-8"))
            self.assertTrue(cfg1["use_dora"])
            self.assertEqual(cfg1["layout"], "row_major")
            self.assertEqual(cfg1["target_modules"], ["gate_attn", "o_proj", "q_proj", "wr_du"])
            mag_entries = [m for m in metas1 if m.name.endswith(f".{LORA_DORA_TENSOR_SUFFIX}")]
            self.assertEqual(len(mag_entries), 2 * 4)  # 2 layers * 4 targets
            self.assertTrue(all(m.dtype == DType.F32 and m.layout == TensorLayout.ROW_MAJOR for m in mag_entries))

            out2 = root / "synth_lora_alias.amlora"
            rc2 = cli_main(
                [
                    "synth-lora",
                    "--dora",
                    "--output",
                    str(out2),
                    "--num-layers",
                    "1",
                    "--rank",
                    "8",
                    "--target-modules",
                    "q_proj,gate_attn,eh_proj",
                ]
            )
            self.assertEqual(rc2, 0)
            _, metas2 = load_amf_metadata(out2)
            names2 = {m.name for m in metas2}
            self.assertIn(f"layers.0.self_attn.eh_proj.{LORA_DORA_TENSOR_SUFFIX}", names2)

    def test_optional_real_inkling2_peft_and_dora_models(self) -> None:
        models_dir_env = os.environ.get("AMINFER_MODELS_DIR")
        if not models_dir_env:
            self.skipTest("AMINFER_MODELS_DIR not set")
        inkling_root = pathlib.Path(models_dir_env) / "Inkling-2-Mini-Unbound-v2.1"
        r16_dir = inkling_root / "lora_v2.1" / "peft-r16"
        dora_dir = inkling_root / "abliteration" / "dora"
        if not r16_dir.is_dir():
            self.skipTest(f"{r16_dir} not found")

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            r16_out = root / "r16.amlora"
            rep16 = pack_lora_from_peft(r16_dir, r16_out, layout="row_major", dtype="f16")
            self.assertEqual(rep16.rank, 16)
            self.assertEqual(rep16.num_layers, 24)
            self.assertEqual(rep16.num_pairs, 120)
            self.assertEqual(rep16.target_modules, ("gate_attn", "k_proj", "o_proj", "q_proj", "v_proj"))
            self.assertFalse(rep16.use_dora)

            if dora_dir.is_dir() and (dora_dir / "adapter_model.safetensors").is_file():
                dora_out = root / "dora.amlora"
                rep_dora = pack_lora_from_peft(dora_dir, dora_out, layout="row_major", dtype="f16")
                self.assertEqual(rep_dora.rank, 1)
                self.assertEqual(rep_dora.num_pairs, 20)
                self.assertEqual(rep_dora.target_modules, ("o_proj",))
                self.assertTrue(rep_dora.use_dora)
                _, dora_metas = load_amf_metadata(dora_out)
                mags = [m for m in dora_metas if m.name.endswith(f".{LORA_DORA_TENSOR_SUFFIX}")]
                self.assertEqual(len(mags), 20)
                self.assertTrue(all(m.dtype == DType.F32 and m.shape == (2304,) for m in mags))


if __name__ == "__main__":
    unittest.main()
