"""Tests for Milestone M4 offline tools: W8 expert-heat, AMF retag/read, and W7 CoreML export."""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AmfReader,
    DType,
    TensorKind,
    bf16_bytes_to_f16_bytes,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    pack_mlx_affine_3bit,
    pack_mlx_affine_4bit,
    retag_amf_cold_experts,
    write_amf,
)
from aminfer_tools.cli import main
from aminfer_tools.coreml import (
    compile_mlpackage,
    export_attention_qkv,
    export_draft_head,
    export_expert_predictor,
    parse_layer_spec,
)
from aminfer_tools.heat import compute_expert_heat, write_heat_profile
from aminfer_tools.synth import DRAFT_PRESETS, PRESETS, generate_draft_bundle, generate_qwen3_bundle

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        chunks.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)


def _f32_to_bf16(values: list[float]) -> bytes:
    out = bytearray()
    for v in values:
        out.extend(struct.pack("<f", v)[2:4])
    return bytes(out)


def _coreml_metadata(pkg_path: pathlib.Path) -> dict[str, object]:
    proc = subprocess.run(
        ["xcrun", "coremlcompiler", "metadata", str(pkg_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"coremlcompiler metadata failed:\n{proc.stderr}\n{proc.stdout}")
    items = json.loads(proc.stdout)
    assert isinstance(items, list) and len(items) == 1
    return items[0]


def _make_tiny_moe_safetensors(st_path: pathlib.Path, *, num_layers: int = 2, num_experts: int = 4) -> None:
    hidden = 64
    inter = 64
    vocab = 32
    tensors: dict[str, tuple[str, list[int], bytes]] = {}

    # Distinct embeddings per token so router argmax varies cleanly across tokens.
    embed_vals: list[float] = []
    for t in range(vocab):
        for k in range(hidden):
            embed_vals.append(0.1 * (((t * 13 + k * 7) % 19) - 9))
    tensors["model.embed_tokens.weight"] = ("F16", [vocab, hidden], struct.pack(f"<{vocab * hidden}e", *embed_vals))

    for l in range(num_layers):
        # Router weights (Q3_AFFINE)
        q3_rows = []
        s_vals = []
        b_vals = []
        for e in range(num_experts):
            codes = [((l * 3 + e * 5 + c) % 8) for c in range(hidden)]
            q3_rows.append(pack_mlx_affine_3bit(codes))
            s_vals.append(0.05 + 0.02 * e)
            b_vals.append(-0.15 - 0.01 * l)
        tensors[f"model.layers.{l}.mlp.gate.weight"] = ("U32", [num_experts, hidden * 3 // 32], b"".join(q3_rows))
        tensors[f"model.layers.{l}.mlp.gate.scales"] = ("BF16", [num_experts, 1], _f32_to_bf16(s_vals))
        tensors[f"model.layers.{l}.mlp.gate.biases"] = ("BF16", [num_experts, 1], _f32_to_bf16(b_vals))

        # Stacked switch_mlp experts (Q4_AFFINE)
        for proj, rows, cols in (("gate_proj", inter, hidden), ("up_proj", inter, hidden), ("down_proj", hidden, inter)):
            w_all = bytearray()
            s_all = []
            b_all = []
            for e in range(num_experts):
                for r in range(rows):
                    w_all.extend(pack_mlx_affine_4bit([(l + e + r + c) % 16 for c in range(cols)]))
                    s_all.append(0.02)
                    b_all.append(-0.1)
            tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.weight"] = (
                "U32",
                [num_experts, rows, cols * 4 // 32],
                bytes(w_all),
            )
            tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.scales"] = (
                "F16",
                [num_experts, rows, cols // 64],
                struct.pack(f"<{len(s_all)}e", *s_all),
            )
            tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.biases"] = (
                "F16",
                [num_experts, rows, cols // 64],
                struct.pack(f"<{len(b_all)}e", *b_all),
            )

    _write_safetensors(st_path, tensors)


class AmfReaderAndRetagTests(unittest.TestCase):
    def test_dequantize_q3_q4_and_f16_reader_parity(self) -> None:
        k = 64
        q3_codes = [(i * 3 + 1) % 8 for i in range(k)]
        q4_codes = [(i * 5 + 2) % 16 for i in range(k)]
        q3_scale, q3_bias = 0.125, -0.5
        q4_scale, q4_bias = 0.0625, -0.25
        p3 = struct.pack("<2e", q3_scale, q3_bias)
        p4 = struct.pack("<2e", q4_scale, q4_bias)

        with tempfile.TemporaryDirectory() as td:
            amf_path = pathlib.Path(td) / "dequant.amf"
            write_amf(
                amf_path,
                "dequant-test",
                [
                    (
                        "layers.0.mlp.gate",
                        TensorKind.ROUTER,
                        DType.Q3_AFFINE,
                        3,
                        64,
                        0,
                        0xFFFF,
                        (2, k),
                        pack_mlx_affine_3bit(q3_codes) * 2,
                        p3 * 2,
                    ),
                    (
                        "layers.0.self_attn.q_proj",
                        TensorKind.DENSE,
                        DType.Q4_AFFINE,
                        4,
                        64,
                        0,
                        0xFFFF,
                        (2, k),
                        pack_mlx_affine_4bit(q4_codes) * 2,
                        p4 * 2,
                    ),
                ],
            )
            with AmfReader(amf_path) as reader:
                self.assertEqual(reader.header.model_id, "dequant-test")
                r_mat = reader.read_matrix_f32(reader.by_name["layers.0.mlp.gate"])
                self.assertEqual(len(r_mat), 2)
                for idx, code in enumerate(q3_codes):
                    expected = code * q3_scale + q3_bias
                    self.assertAlmostEqual(r_mat[0][idx], expected, places=5)
                    self.assertAlmostEqual(r_mat[1][idx], expected, places=5)

                q_row1 = reader.read_rows_f32(reader.by_name["layers.0.self_attn.q_proj"], [1])[0]
                for idx, code in enumerate(q4_codes):
                    self.assertAlmostEqual(q_row1[idx], code * q4_scale + q4_bias, places=5)

                f16_bytes = reader.read_tensor_f16_bytes(reader.by_name["layers.0.mlp.gate"])
                self.assertEqual(len(f16_bytes), 2 * k * 2)
                unpacked = struct.unpack(f"<{2 * k}e", f16_bytes)
                for idx, code in enumerate(q3_codes):
                    self.assertAlmostEqual(unpacked[idx], code * q3_scale + q3_bias, places=3)

    def test_retag_amf_cold_experts_inplace_and_copy_verified_by_zig(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bundle = generate_qwen3_bundle(tmp / "moe", cfg, seed=7)
            amf_path = bundle.amf_path

            # Copy-mode retagging: mark 2 experts per layer cold.
            cold_set = {(0, 1), (0, 3), (1, 0), (1, 2)}
            copied = tmp / "moe_cold.amf"
            hot_n, cold_n = retag_amf_cold_experts(amf_path, cold_set, output_path=copied)
            total_experts = cfg.num_hidden_layers * cfg.num_experts
            self.assertEqual(cold_n, len(cold_set) * 3)
            self.assertEqual(hot_n, (total_experts - len(cold_set)) * 3)

            proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(copied)], check=False, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertIn("amf.status       ok", proc.stdout)
            self.assertNotIn("amf.cold_bytes   0", proc.stdout)

            # In-place retagging via CLI using a heat profile.
            heat_path = tmp / "heat.json"
            profile = compute_expert_heat(amf_path, top_k=2, cold_experts_per_layer=2)
            write_heat_profile(profile, heat_path)

            rc = main(
                [
                    "retag-cold",
                    "--input",
                    str(amf_path),
                    "--heat-profile",
                    str(heat_path),
                    "--cold-experts-per-layer",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            proc2 = subprocess.run([str(AMINFER_BIN), "verify-amf", str(amf_path)], check=False, capture_output=True, text=True)
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr + proc2.stdout)
            self.assertIn("amf.status       ok", proc2.stdout)

            with AmfReader(amf_path) as reader:
                expected_cold = load_cold_experts_from_heat_profile(heat_path, cold_experts_per_layer=2)
                for entry in reader.entries:
                    if entry.expert != 0xFFFF:
                        expected_kind = (
                            TensorKind.COLD_EXPERT if (entry.layer, entry.expert) in expected_cold else TensorKind.HOT_EXPERT
                        )
                        self.assertEqual(entry.kind, expected_kind)

    def test_pack_amf_with_heat_profile_and_cold_experts_per_layer(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            st_path = tmp / "model.safetensors"
            _make_tiny_moe_safetensors(st_path, num_layers=2, num_experts=4)
            heat_path = tmp / "heat.json"
            profile = compute_expert_heat(st_path, top_k=2, cold_experts_per_layer=1)
            write_heat_profile(profile, heat_path)

            amf_out = tmp / "packed.amf"
            rc = main(
                [
                    "pack-amf",
                    "--input",
                    str(st_path),
                    "--output",
                    str(amf_out),
                    "--model-id",
                    "tiny-heat-pack",
                    "--heat-profile",
                    str(heat_path),
                    "--cold-experts-per-layer",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            expected_cold = load_cold_experts_from_heat_profile(heat_path, cold_experts_per_layer=2)
            self.assertEqual(len(expected_cold), 4)

            with AmfReader(amf_out) as reader:
                cold_pairs = {(e.layer, e.expert) for e in reader.entries if e.kind == TensorKind.COLD_EXPERT}
                self.assertEqual(cold_pairs, expected_cold)

            proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(amf_out)], check=False, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertIn("amf.status       ok", proc.stdout)


class ExpertHeatTests(unittest.TestCase):
    def test_builtin_sample_parity_between_safetensors_and_amf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            st_path = tmp / "model.safetensors"
            amf_path = tmp / "model.amf"
            _make_tiny_moe_safetensors(st_path, num_layers=2, num_experts=4)
            pack_amf_from_safetensors(st_path, amf_path, model_id="tiny-moe-st")

            prof_st = compute_expert_heat(
                st_path, top_k=2, cold_experts_per_layer=2, sample_tokens=16, model_id="tiny-moe-st"
            )
            prof_amf = compute_expert_heat(amf_path, top_k=2, cold_experts_per_layer=2, sample_tokens=16)

            self.assertEqual(prof_st.schema_version, 1)
            self.assertEqual(prof_amf.schema_version, 1)
            self.assertEqual(prof_st.num_layers, 2)
            self.assertEqual(prof_st.num_experts, 4)
            self.assertEqual(prof_st.top_k, 2)
            self.assertEqual(prof_st.total_cold_experts, 4)
            self.assertEqual(prof_st.total_hot_experts, 4)
            self.assertEqual(prof_st.cold_spec_arg, prof_amf.cold_spec_arg)
            for l_st, l_amf in zip(prof_st.layers, prof_amf.layers):
                self.assertEqual(l_st.counts, l_amf.counts)
                self.assertEqual(l_st.ranked_hot_to_cold, l_amf.ranked_hot_to_cold)
                self.assertEqual(l_st.cold_experts, l_amf.cold_experts)
                self.assertEqual(len(l_amf.ranked_hot_to_cold), 4)
                self.assertEqual(set(l_amf.ranked_hot_to_cold), {0, 1, 2, 3})
                self.assertAlmostEqual(sum(l_amf.frequencies), 1.0, places=4)

    def test_router_trace_and_token_jsonl_calibration_modes(self) -> None:
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bundle = generate_qwen3_bundle(tmp / "moe", cfg, seed=11)

            # Mode 1: recorded router trace JSONL
            trace_path = tmp / "trace.jsonl"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps({"layer": 0, "experts": [2, 3], "weight": 10}),
                        json.dumps({"layer": 0, "experts": [[2, 0], [2, 3]]}),
                        json.dumps({"layer": 1, "experts": [1, 3], "count": 5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prof_trace = compute_expert_heat(
                bundle.amf_path,
                calibration_jsonl=trace_path,
                top_k=2,
                cold_experts_per_layer=2,
            )
            self.assertEqual(prof_trace.calibration_mode, "router_trace")
            self.assertEqual(prof_trace.layers[0].counts, (1, 0, 12, 11, 0, 0, 0, 0))
            self.assertEqual(prof_trace.layers[0].ranked_hot_to_cold[:3], (2, 3, 0))
            self.assertEqual(len(prof_trace.layers[0].cold_experts), 2)
            self.assertNotIn(2, prof_trace.layers[0].cold_experts)
            self.assertNotIn(3, prof_trace.layers[0].cold_experts)
            self.assertEqual(prof_trace.layers[1].counts, (0, 5, 0, 5, 0, 0, 0, 0))
            self.assertNotIn(1, prof_trace.layers[1].cold_experts)
            self.assertNotIn(3, prof_trace.layers[1].cold_experts)
            self.assertEqual(len(prof_trace.cold_spec_arg.split(",")), 4)

            # Mode 2: token-ID JSONL scored by embed_tokens + routers
            toks_path = tmp / "tokens.jsonl"
            toks_path.write_text(
                "\n".join(
                    [
                        json.dumps({"token_ids": [0, 1, 2, 3, 4, 5]}),
                        json.dumps([1, 2, 10, 15]),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prof_toks = compute_expert_heat(
                bundle.amf_path,
                calibration_jsonl=toks_path,
                top_k=2,
                cold_experts_per_layer=1,
            )
            self.assertEqual(prof_toks.calibration_mode, "token_jsonl_embed_router")
            self.assertEqual(prof_toks.sample_tokens, 10)
            for lp in prof_toks.layers:
                self.assertEqual(lp.total_activations, 20)
                self.assertEqual(len(lp.cold_experts), 1)

    def test_wired_budget_mib_and_cli_json_output(self) -> None:
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bundle = generate_qwen3_bundle(tmp / "moe", cfg, seed=19)
            # Compute non-expert size + half of experts to force a split via --wired-budget-mib.
            with AmfReader(bundle.amf_path) as reader:
                non_exp = sum(e.blob_bytes for e in reader.entries if e.expert == 0xFFFF)
                exp_total = sum(e.blob_bytes for e in reader.entries if e.expert != 0xFFFF)
            target_bytes = non_exp + (exp_total // 2)
            budget_mib = target_bytes / (1024.0 * 1024.0)

            out_json = tmp / "budget_heat.json"
            rc = main(
                [
                    "expert-heat",
                    "--input",
                    str(bundle.amf_path),
                    "--output",
                    str(out_json),
                    "--top-k",
                    "2",
                    "--wired-budget-mib",
                    f"{budget_mib:.6f}",
                ]
            )
            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 1)
            self.assertLessEqual(data["estimated_wired_bytes"], target_bytes + APPLE_PAGE_SIZE)
            self.assertGreater(data["total_cold_experts"], 0)
            self.assertGreater(data["total_hot_experts"], 0)
            self.assertEqual(
                data["total_hot_experts"] + data["total_cold_experts"],
                cfg.num_hidden_layers * cfg.num_experts,
            )


class CoreMlExportTests(unittest.TestCase):
    def test_parse_layer_spec(self) -> None:
        self.assertEqual(parse_layer_spec("0..3"), [0, 1, 2, 3])
        self.assertEqual(parse_layer_spec("0,2..4,7-8"), [0, 2, 3, 4, 7, 8])
        with self.assertRaises(ValueError):
            parse_layer_spec("5..2")
        with self.assertRaises(ValueError):
            parse_layer_spec("")

    def test_export_and_compile_expert_predictor_amf_and_synthetic(self) -> None:
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bundle = generate_qwen3_bundle(tmp / "moe", cfg, seed=3)

            # 1) Single .mlpackage from AMF (padded automatically to 32-aligned IOSurface trailing dim)
            pkg1 = tmp / "pred_amf.mlpackage"
            res1 = export_expert_predictor(
                pkg1,
                amf_path=bundle.amf_path,
                layers_spec="0..1",
                lookahead=2,
                batch_size=32,
            )
            self.assertTrue((pkg1 / "Manifest.json").is_file())
            self.assertTrue((pkg1 / "Data" / "com.apple.CoreML" / "model.mlmodel").is_file())
            manifest = json.loads((pkg1 / "Manifest.json").read_text(encoding="utf-8"))
            self.assertIn(manifest["rootModelIdentifier"], manifest["itemInfoEntries"])
            self.assertEqual(res1.inputs[0].shape, (32, cfg.hidden_size))
            self.assertEqual(res1.outputs[0].shape, (32, 32))

            compiled1 = compile_mlpackage(pkg1, tmp / "c1")
            self.assertTrue((compiled1 / "model.espresso.net").is_file())
            meta1 = _coreml_metadata(pkg1)
            self.assertEqual(meta1["specificationVersion"], 7)
            self.assertEqual(meta1["storagePrecision"], "Float16")

            # 2) Directory multi-window mode + conv1x1 mode
            win_dir = tmp / "pred_windows"
            res_dir = export_expert_predictor(
                win_dir,
                amf_path=bundle.amf_path,
                layers_spec="0..1",
                lookahead=1,
                batch_size=32,
                op="conv1x1",
            )
            self.assertTrue((win_dir / "predictor_manifest.json").is_file())
            self.assertEqual(len(res_dir.bundle_paths), 1)
            compiled2 = compile_mlpackage(res_dir.bundle_paths[0], tmp / "c2")
            self.assertTrue(compiled2.is_dir())

            # 3) Synthetic Qwen3-30B-A3B geometry predictor (2048 -> 2*128 = 256)
            pkg3 = tmp / "pred_synth.mlpackage"
            res3 = export_expert_predictor(
                pkg3,
                lookahead=2,
                batch_size=32,
                hidden_size=128,
                num_experts=32,
                seed=99,
            )
            self.assertEqual(res3.outputs[0].shape, (32, 64))
            compiled3 = compile_mlpackage(pkg3, tmp / "c3")
            self.assertTrue(compiled3.is_dir())

    def test_export_and_compile_attention_qkv_amf_and_synthetic(self) -> None:
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bundle = generate_qwen3_bundle(tmp / "moe", cfg, seed=5)

            # From AMF with 512-token chunk + multi-output Q/K/V
            pkg_qkv = tmp / "attn_qkv.mlpackage"
            res = export_attention_qkv(
                pkg_qkv,
                amf_path=bundle.amf_path,
                layer=0,
                seq_len=512,
                include_o_proj=True,
            )
            self.assertEqual([t.name for t in res.inputs], ["hidden_in", "attn_in"])
            self.assertEqual([t.name for t in res.outputs], ["q_out", "k_out", "v_out", "o_out"])
            self.assertEqual(res.inputs[0].shape, (512, cfg.hidden_size))
            self.assertEqual(res.outputs[1].shape[0], 512)
            self.assertEqual(res.outputs[1].shape[1] % 32, 0)

            compiled = compile_mlpackage(pkg_qkv, tmp / "c_qkv")
            self.assertTrue((compiled / "model.espresso.net").is_file())
            meta = _coreml_metadata(pkg_qkv)
            self.assertEqual(meta["specificationVersion"], 7)
            self.assertEqual(len(meta["outputSchema"]), 4)

            # Synthetic conv1x1 attention-qkv (512 tokens)
            pkg_conv = tmp / "attn_conv.mlpackage"
            export_attention_qkv(
                pkg_conv,
                seq_len=512,
                op="conv1x1",
                hidden_size=128,
                q_dim=128,
                kv_dim=64,
            )
            self.assertTrue(compile_mlpackage(pkg_conv, tmp / "c_conv").is_dir())

    def test_export_and_compile_draft_head_all_stages(self) -> None:
        dcfg = DRAFT_PRESETS["tiny-eagle3"]
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            # Synthesize an EAGLE-3 draft bundle (has both fc.weight and SwiGLU FFN!)
            draft_bundle = generate_draft_bundle(tmp / "eagle3", dcfg, seed=13)

            # 1) SwiGLU FFN stage (verifies sigmoid + multiply op=1 in compiled espresso.net!)
            pkg_ffn = tmp / "draft_ffn.mlpackage"
            res_ffn = export_draft_head(pkg_ffn, amf_path=draft_bundle.amf_path, layer=0, stage="ffn", batch_size=32)
            self.assertEqual(res_ffn.layer_count, 6)
            compiled_ffn = compile_mlpackage(pkg_ffn, tmp / "c_ffn")
            net = json.loads((compiled_ffn / "model.espresso.net").read_text(encoding="utf-8"))
            elem_ops = [l.get("operation") for l in net["layers"] if l.get("type") == "elementwise"]
            self.assertEqual(elem_ops, [1, 1])  # both SiLU and SwiGLU multiplications must be op=1 (multiply)

            # 2) EAGLE-3 fc fusion projection stage
            pkg_fc = tmp / "draft_fc.mlpackage"
            res_fc = export_draft_head(pkg_fc, amf_path=draft_bundle.amf_path, stage="fc", batch_size=32)
            self.assertEqual(res_fc.inputs[0].shape, (32, 3 * dcfg.hidden_size))
            self.assertEqual(res_fc.outputs[0].shape, (32, dcfg.hidden_size))
            self.assertTrue(compile_mlpackage(pkg_fc, tmp / "c_fc").is_dir())

            # 3) Synthetic linear stage
            pkg_lin = tmp / "draft_lin.mlpackage"
            export_draft_head(pkg_lin, stage="linear", batch_size=32, hidden_size=128, out_dim=64)
            self.assertTrue(compile_mlpackage(pkg_lin, tmp / "c_lin").is_dir())

    def test_iosurface_alignment_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            with self.assertRaises(ValueError):
                export_attention_qkv(tmp / "bad.mlpackage", seq_len=512, hidden_size=48, q_dim=64, kv_dim=64)

    def test_cli_export_coreml_all_three_kinds_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            verify_dir = tmp / "compiled"
            cases = [
                [
                    "export-coreml",
                    "--kind",
                    "expert-predictor",
                    "--output",
                    str(tmp / "cli_pred.mlpackage"),
                    "--hidden-size",
                    "128",
                    "--num-experts",
                    "32",
                    "--lookahead",
                    "2",
                    "--verify-compile",
                    str(verify_dir),
                ],
                [
                    "export-coreml",
                    "--kind",
                    "attention-qkv",
                    "--output",
                    str(tmp / "cli_qkv.mlpackage"),
                    "--seq-len",
                    "512",
                    "--hidden-size",
                    "128",
                    "--q-dim",
                    "128",
                    "--kv-dim",
                    "64",
                    "--verify-compile",
                    str(verify_dir),
                ],
                [
                    "export-coreml",
                    "--kind",
                    "draft-head",
                    "--output",
                    str(tmp / "cli_draft.mlpackage"),
                    "--stage",
                    "ffn",
                    "--hidden-size",
                    "128",
                    "--intermediate-size",
                    "256",
                    "--verify-compile",
                    str(verify_dir),
                ],
            ]
            for argv in cases:
                self.assertEqual(main(argv), 0)
            self.assertTrue((verify_dir / "cli_pred.mlmodelc").is_dir())
            self.assertTrue((verify_dir / "cli_qkv.mlmodelc").is_dir())
            self.assertTrue((verify_dir / "cli_draft.mlmodelc").is_dir())


if __name__ == "__main__":
    unittest.main()
