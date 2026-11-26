from __future__ import annotations

import json
import pathlib
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aminfer_tools.amf import (  # noqa: E402
    AmfReader,
    DType,
    META_CHAT_TEMPLATE_JSON,
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    TensorKind,
    bf16_bytes_to_f32_list,
    dequantize_fp8_block_row_f32,
    dequantize_fp8_row_f32,
    dequantize_q4_row_f32,
    e4m3_byte_to_f32,
    f32_to_bf16_bytes as f32_values_to_bf16_bytes,
    f32_values_to_f16_bytes,
)
from aminfer_tools.cli import main as cli_main  # noqa: E402
from aminfer_tools.gguf import (  # noqa: E402
    dequantize_gguf_amf_row_f32,
    pack_amf_from_gguf,
)
from aminfer_tools.lora import pack_lora_from_peft  # noqa: E402
from aminfer_tools.mimo_v2 import (  # noqa: E402
    MIMO_TIER_PROFILES,
    MimoQkvSpec,
    MimoV2SynthConfig,
    Qwen35SynthConfig,
    convert_mimo_v2_checkpoint,
    convert_qwen35_checkpoint,
    dequantize_hf_mxfp4_row_f32,
    dequantize_mimo_qkv_row_f32,
    export_mimo_dflash,
    generate_synth_mimo_bundle,
    generate_synth_qwen35_bundle,
    map_mimo_contiguous_qkv_row,
    repack_hf_mxfp4_row_to_gguf,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"
GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32


def _verify_amf_with_zig(test: unittest.TestCase, path: pathlib.Path, expected_tensors: int) -> None:
    test.assertTrue(
        AMINFER_BIN.is_file(),
        f"missing compiled binary {AMINFER_BIN} required for verify-amf checks; run `zig build` first",
    )
    res = subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    test.assertEqual(res.returncode, 0, f"aminfer verify-amf failed on {path}:\n{res.stdout}\n{res.stderr}")
    test.assertIsNotNone(
        re.search(rf"^amf\.tensors\s+{expected_tensors}$", res.stdout, re.MULTILINE),
        f"expected amf.tensors={expected_tensors} in:\n{res.stdout}",
    )


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payloads.append(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_bytes)) % 8
    hdr_padded = hdr_bytes + b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_padded)))
        fh.write(hdr_padded)
        for p in payloads:
            fh.write(p)


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _write_synth_gguf(
    path: pathlib.Path,
    kv: dict[str, tuple[str, object]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    """Write a minimal GGUF v3 file (`tensors` uses AMF row-major shapes; dims reversed on write)."""
    buf = bytearray()
    buf.extend(GGUF_MAGIC)
    buf.extend(struct.pack("<IQQ", 3, len(tensors), len(kv)))
    type_code = {"u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "arr": 9}
    for k, (vt, val) in kv.items():
        buf.extend(_pack_gguf_str(k))
        buf.extend(struct.pack("<I", type_code[vt]))
        if vt == "u32":
            buf.extend(struct.pack("<I", int(val)))  # type: ignore[arg-type]
        elif vt == "i32":
            buf.extend(struct.pack("<i", int(val)))  # type: ignore[arg-type]
        elif vt == "f32":
            buf.extend(struct.pack("<f", float(val)))  # type: ignore[arg-type]
        elif vt == "bool":
            buf.extend(struct.pack("<B", 1 if val else 0))
        elif vt == "str":
            buf.extend(_pack_gguf_str(str(val)))
        elif vt == "arr":
            elem_vt, items = val  # type: ignore[misc]
            buf.extend(struct.pack("<IQ", type_code[elem_vt], len(items)))
            for it in items:
                if elem_vt == "i32":
                    buf.extend(struct.pack("<i", int(it)))
                elif elem_vt == "str":
                    buf.extend(_pack_gguf_str(str(it)))
                elif elem_vt == "bool":
                    buf.extend(struct.pack("<B", 1 if it else 0))
                else:
                    raise ValueError(f"unsupported arr elem {elem_vt}")

    cur_off = 0
    tensor_blobs: list[bytes] = []
    for name, amf_shape, ggml_type, payload in tensors:
        buf.extend(_pack_gguf_str(name))
        gguf_dims = list(reversed(amf_shape))
        buf.extend(struct.pack("<I", len(gguf_dims)))
        for d in gguf_dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<IQ", ggml_type, cur_off))
        pad_t = (-len(payload)) % GGUF_DEFAULT_ALIGNMENT
        tensor_blobs.append(payload + b"\x00" * pad_t)
        cur_off += len(payload) + pad_t

    pad_hdr = (-len(buf)) % GGUF_DEFAULT_ALIGNMENT
    buf.extend(b"\x00" * pad_hdr)
    for b in tensor_blobs:
        buf.extend(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def _det_f32(seed: int, n: int, *, std: float = 0.25) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, std) for _ in range(n)]


def _read_embedded_json(path: pathlib.Path, key: str = META_CONFIG_JSON) -> dict:
    with AmfReader(path) as rdr:
        entry = rdr.by_name[key]
        raw = rdr.read_blob(entry, verify_crc=True)[: entry.data_bytes]
        return json.loads(raw.decode("utf-8"))


class M91MimoV2AndQwen35Tests(unittest.TestCase):
    def test_mxfp4_hf_to_gguf_repack_numeric_parity(self) -> None:
        rng = random.Random(101)
        k = 256
        groups = k // 32
        w_row = bytes(rng.randrange(256) for _ in range(k // 2))
        # Cover zero-exp (0), sub-1 (125), 1.0 (127), >1 (129), and edge exponents
        s_row = bytes([0, 125, 127, 128, 129, 126, 130, 124][:groups])
        expected = dequantize_hf_mxfp4_row_f32(w_row, s_row, k)
        gguf_row = repack_hf_mxfp4_row_to_gguf(w_row, s_row, k)
        self.assertEqual(len(gguf_row), groups * 17)
        actual = dequantize_gguf_amf_row_f32(DType.GGUF_MXFP4, gguf_row, k)
        self.assertEqual(actual, expected)

    def test_mimo_qkv_tp4_and_tp8_deinterleave_map(self) -> None:
        # Flash-like TP=4 global: 64q, 4kv, hd=192, vd=128 -> 13568 rows, scale_inv rows = 108
        spec4 = MimoQkvSpec(tp_degree=4, q_heads=64, kv_heads=4, head_dim=192, v_head_dim=128, k_cols=4096)
        self.assertEqual(spec4.total_rows, 13568)
        self.assertEqual(spec4.expected_scale_rows, 108)
        seen_src = {map_mimo_contiguous_qkv_row(spec4, r)[0] for r in range(spec4.total_rows)}
        self.assertEqual(len(seen_src), spec4.total_rows)
        # First K row sits immediately after rank 0's Q block (q_tp = 16*192 = 3072)
        k0_src, k0_sc = map_mimo_contiguous_qkv_row(spec4, spec4.q_total)
        self.assertEqual(k0_src, 3072)
        self.assertEqual(k0_sc, 24)

        # Pro-like TP=8: 64q, 8kv, hd=192, vd=128 -> 14848 rows, scale_inv rows = 120
        spec8 = MimoQkvSpec(tp_degree=8, q_heads=64, kv_heads=8, head_dim=192, v_head_dim=128, k_cols=6144)
        self.assertEqual(spec8.total_rows, 14848)
        self.assertEqual(spec8.expected_scale_rows, 120)
        seen_src8 = {map_mimo_contiguous_qkv_row(spec8, r)[0] for r in range(spec8.total_rows)}
        self.assertEqual(len(seen_src8), spec8.total_rows)

    def test_convert_mimo_v2_tp4_safetensors_mxfp4_mtp_dflash_and_cold_tier(self) -> None:
        # Realistic asymmetric MiMo geometry: TP=4, q=4 heads, kv=4 heads, hd=192, vd=128, hidden=256
        # Per rank: Q=192 (2 scale blks), K=192 (2 blks), V=128 (1 blk) => 5 blks/rank => 20 total (!= contiguous 16)
        hidden = 256
        inter = 256
        moe_inter = 128
        vocab = 128
        qh = 4
        kvh = 4
        hd = 192
        vd = 128
        n_exp = 4
        n_layers = 3
        tp = 4
        spec = MimoQkvSpec(tp, qh, kvh, hd, vd, hidden)
        self.assertEqual(spec.expected_scale_rows, 20)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp) / "mimo_hf"
            ckpt.mkdir(parents=True)
            src_cfg = {
                "model_type": "mimo_v2",
                "hidden_size": hidden,
                "intermediate_size": inter,
                "moe_intermediate_size": moe_inter,
                "num_hidden_layers": n_layers,
                "vocab_size": vocab,
                "num_attention_heads": qh,
                "num_key_value_heads": kvh,
                "swa_num_attention_heads": qh,
                "swa_num_key_value_heads": kvh,
                "head_dim": hd,
                "v_head_dim": vd,
                "swa_head_dim": hd,
                "swa_v_head_dim": vd,
                "sliding_window_size": 128,
                "hybrid_layer_pattern": [0, 1, 1],
                "moe_layer_freq": [0, 1, 1],
                "n_routed_experts": n_exp,
                "num_experts_per_tok": 2,
                "n_shared_experts": 0,
                "rope_theta": 10000000.0,
                "swa_rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "attention_value_scale": 0.707,
                "add_swa_attention_sink_bias": True,
                "add_full_attention_sink_bias": False,
                "num_nextn_predict_layers": 3,
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "routed_scaling_factor": 2.446,
            }
            (ckpt / "config.json").write_text(json.dumps(src_cfg, indent=2), encoding="utf-8")
            (ckpt / "tokenizer.json").write_text(
                json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(vocab)}, "merges": []}}),
                encoding="utf-8",
            )

            rng = random.Random(2026)
            # Non-trivial FP8 bytes (safe finite E4M3 codes 0x20..0x4f) + distinct block scales per rank/block
            qkv_raw_l1 = bytes(rng.randrange(0x20, 0x4F) for _ in range(spec.total_rows * hidden))
            qkv_sc_l1 = [0.25 + 0.05 * i for i in range(spec.expected_scale_rows * (hidden // 128))]
            mxfp4_w = bytes(rng.randrange(256) for _ in range(moe_inter * (hidden // 2)))
            mxfp4_s = bytes(rng.randrange(122, 132) for _ in range(moe_inter * (hidden // 32)))
            mxfp4_dw = bytes(rng.randrange(256) for _ in range(hidden * (moe_inter // 2)))
            mxfp4_ds = bytes(rng.randrange(122, 132) for _ in range(hidden * (moe_inter // 32)))

            main_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "model.embed_tokens.weight": ("BF16", (vocab, hidden), f32_values_to_bf16_bytes(_det_f32(1, vocab * hidden))),
                "model.norm.weight": ("F32", (hidden,), struct.pack(f"<{hidden}f", *_det_f32(2, hidden))),
                "lm_head.weight": ("BF16", (vocab, hidden), f32_values_to_bf16_bytes(_det_f32(3, vocab * hidden))),
                # Long unused audio tensor (>47B) must be stripped automatically
                "audio_encoder.conv_layers.0.very_long_unused_audio_submodule_weight": (
                    "BF16",
                    (4, 4),
                    f32_values_to_bf16_bytes([0.1] * 16),
                ),
            }
            for l_idx in range(n_layers):
                qkv_raw = qkv_raw_l1 if l_idx == 1 else bytes(0x38 for _ in range(spec.total_rows * hidden))
                qkv_sc = qkv_sc_l1 if l_idx == 1 else [0.5] * len(qkv_sc_l1)
                main_tensors[f"model.layers.{l_idx}.input_layernorm.weight"] = (
                    "F32",
                    (hidden,),
                    struct.pack(f"<{hidden}f", *[1.0] * hidden),
                )
                main_tensors[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = (
                    "F32",
                    (hidden,),
                    struct.pack(f"<{hidden}f", *[1.0] * hidden),
                )
                main_tensors[f"model.layers.{l_idx}.self_attn.qkv_proj.weight"] = (
                    "F8_E4M3",
                    (spec.total_rows, hidden),
                    qkv_raw,
                )
                main_tensors[f"model.layers.{l_idx}.self_attn.qkv_proj.weight_scale_inv"] = (
                    "F32",
                    (spec.expected_scale_rows, hidden // 128),
                    struct.pack(f"<{len(qkv_sc)}f", *qkv_sc),
                )
                main_tensors[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = (
                    "BF16",
                    (hidden, qh * vd),
                    f32_values_to_bf16_bytes(_det_f32(10 + l_idx, hidden * qh * vd)),
                )
                if l_idx > 0:
                    main_tensors[f"model.layers.{l_idx}.self_attn.attention_sink_bias"] = (
                        "F32",
                        (qh,),
                        struct.pack(f"<{qh}f", *[-0.25 * (i + 1) for i in range(qh)]),
                    )
                if l_idx == 0:
                    for p_name, shp in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden)), ("down_proj", (hidden, inter))):
                        main_tensors[f"model.layers.0.mlp.{p_name}.weight"] = (
                            "BF16",
                            shp,
                            f32_values_to_bf16_bytes(_det_f32(20, shp[0] * shp[1])),
                        )
                else:
                    main_tensors[f"model.layers.{l_idx}.mlp.gate.weight"] = (
                        "F32",
                        (n_exp, hidden),
                        struct.pack(f"<{n_exp * hidden}f", *_det_f32(30 + l_idx, n_exp * hidden)),
                    )
                    main_tensors[f"model.layers.{l_idx}.mlp.gate.e_score_correction_bias"] = (
                        "F32",
                        (n_exp,),
                        struct.pack(f"<{n_exp}f", *[0.05 * e for e in range(n_exp)]),
                    )
                    for e in range(n_exp):
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.gate_proj.weight"] = ("U8", (moe_inter, hidden // 2), mxfp4_w)
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.gate_proj.weight_scale"] = ("U8", (moe_inter, hidden // 32), mxfp4_s)
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.up_proj.weight"] = ("U8", (moe_inter, hidden // 2), mxfp4_w)
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.up_proj.weight_scale"] = ("U8", (moe_inter, hidden // 32), mxfp4_s)
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.down_proj.weight"] = ("U8", (hidden, moe_inter // 2), mxfp4_dw)
                        main_tensors[f"model.layers.{l_idx}.mlp.experts.{e}.down_proj.weight_scale"] = ("U8", (hidden, moe_inter // 32), mxfp4_ds)
            _write_safetensors(ckpt / "model.safetensors", main_tensors)

            # Companion 3L native MTP shard with TP=4 qkv_proj + pre_mlp_layernorm + final_layernorm
            mtp_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            for k in range(3):
                mtp_tensors[f"mtp.layers.{k}.enorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                mtp_tensors[f"mtp.layers.{k}.hnorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                mtp_tensors[f"mtp.layers.{k}.eh_proj.weight"] = ("BF16", (hidden, 2 * hidden), f32_values_to_bf16_bytes(_det_f32(50 + k, hidden * 2 * hidden)))
                mtp_tensors[f"mtp.layers.{k}.input_layernorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                mtp_tensors[f"mtp.layers.{k}.pre_mlp_layernorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                mtp_tensors[f"mtp.layers.{k}.final_layernorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                mtp_tensors[f"mtp.layers.{k}.self_attn.qkv_proj.weight"] = ("F8_E4M3", (spec.total_rows, hidden), qkv_raw_l1 if k == 0 else bytes(0x38 for _ in range(spec.total_rows * hidden)))
                mtp_tensors[f"mtp.layers.{k}.self_attn.qkv_proj.weight_scale_inv"] = (
                    "F32",
                    (spec.expected_scale_rows, hidden // 128),
                    struct.pack(f"<{len(qkv_sc_l1)}f", *(qkv_sc_l1 if k == 0 else [0.5] * len(qkv_sc_l1))),
                )
                mtp_tensors[f"mtp.layers.{k}.self_attn.o_proj.weight"] = ("BF16", (hidden, qh * vd), f32_values_to_bf16_bytes(_det_f32(60 + k, hidden * qh * vd)))
                mtp_tensors[f"mtp.layers.{k}.self_attn.attention_sink_bias"] = ("F32", (qh,), struct.pack(f"<{qh}f", *[-0.1] * qh))
                mtp_tensors[f"mtp.layers.{k}.mlp.gate_proj.weight"] = ("BF16", (inter, hidden), f32_values_to_bf16_bytes(_det_f32(70 + k, inter * hidden)))
                mtp_tensors[f"mtp.layers.{k}.mlp.up_proj.weight"] = ("BF16", (inter, hidden), f32_values_to_bf16_bytes(_det_f32(80 + k, inter * hidden)))
                mtp_tensors[f"mtp.layers.{k}.mlp.down_proj.weight"] = ("BF16", (hidden, inter), f32_values_to_bf16_bytes(_det_f32(90 + k, hidden * inter)))
            _write_safetensors(ckpt / "model_mtp.safetensors", mtp_tensors)

            # Companion DFlash shard
            df_dir = ckpt / "dflash"
            df_dir.mkdir()
            (df_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "dflash",
                        "hidden_size": hidden,
                        "intermediate_size": inter,
                        "num_hidden_layers": 2,
                        "num_attention_heads": qh,
                        "num_key_value_heads": kvh,
                        "head_dim": 64,
                        "dflash_config": {"block_size": 8, "mask_token_id": 151675, "target_layer_ids": [0, 1, 2]},
                    }
                ),
                encoding="utf-8",
            )
            df_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "fc.weight": ("BF16", (hidden, 3 * hidden), f32_values_to_bf16_bytes(_det_f32(100, hidden * 3 * hidden))),
                "hidden_norm.weight": ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden)),
                "norm.weight": ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden)),
            }
            for dl in range(2):
                df_tensors[f"layers.{dl}.input_layernorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                df_tensors[f"layers.{dl}.post_attention_layernorm.weight"] = ("F32", (hidden,), struct.pack(f"<{hidden}f", *[1.0] * hidden))
                df_tensors[f"layers.{dl}.self_attn.q_proj.weight"] = ("BF16", (qh * 64, hidden), f32_values_to_bf16_bytes(_det_f32(110 + dl, qh * 64 * hidden)))
                df_tensors[f"layers.{dl}.self_attn.k_proj.weight"] = ("BF16", (kvh * 64, hidden), f32_values_to_bf16_bytes(_det_f32(120 + dl, kvh * 64 * hidden)))
                df_tensors[f"layers.{dl}.self_attn.v_proj.weight"] = ("BF16", (kvh * 64, hidden), f32_values_to_bf16_bytes(_det_f32(130 + dl, kvh * 64 * hidden)))
                df_tensors[f"layers.{dl}.self_attn.o_proj.weight"] = ("BF16", (hidden, qh * 64), f32_values_to_bf16_bytes(_det_f32(140 + dl, hidden * qh * 64)))
                df_tensors[f"layers.{dl}.self_attn.q_norm.weight"] = ("F32", (64,), struct.pack("<64f", *[1.0] * 64))
                df_tensors[f"layers.{dl}.self_attn.k_norm.weight"] = ("F32", (64,), struct.pack("<64f", *[1.0] * 64))
                df_tensors[f"layers.{dl}.self_attn.attention_sink_bias"] = ("F32", (qh,), struct.pack(f"<{qh}f", *[-0.2] * qh))
                df_tensors[f"layers.{dl}.mlp.gate_proj.weight"] = ("BF16", (inter, hidden), f32_values_to_bf16_bytes(_det_f32(150 + dl, inter * hidden)))
                df_tensors[f"layers.{dl}.mlp.up_proj.weight"] = ("BF16", (inter, hidden), f32_values_to_bf16_bytes(_det_f32(160 + dl, inter * hidden)))
                df_tensors[f"layers.{dl}.mlp.down_proj.weight"] = ("BF16", (hidden, inter), f32_values_to_bf16_bytes(_det_f32(170 + dl, hidden * inter)))
            _write_safetensors(df_dir / "dflash_draft_model.safetensors", df_tensors)

            out_dir = pathlib.Path(tmp) / "out_amf"
            rep = convert_mimo_v2_checkpoint(
                ckpt,
                out_dir / "model.amf",
                target_tier="24gb-m4",
                cold_experts_per_layer=2,
                mtp_output=out_dir / "mtp.amf",
                dflash_output=out_dir / "dflash.amf",
            )
            self.assertEqual(rep.qkv_tp_degree, 4)
            self.assertEqual(rep.stripped_audio_tensors, 1)
            self.assertEqual(rep.cold_expert_count, 2 * 2 * 3)  # 2 layers * 2 cold experts * 3 projs
            self.assertIsNotNone(rep.mtp_amf)
            self.assertIsNotNone(rep.dflash_amf)

            # Verify <= 47 B name invariant, Zig verify-amf, TP=4 numeric parity, MXFP4 parity, and metadata
            for p_amf, metas in (
                (rep.backbone_amf, rep.backbone_metas),
                (rep.mtp_amf, rep.mtp_metas),
                (rep.dflash_amf, rep.dflash_metas),
            ):
                assert p_amf is not None
                for m in metas:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47, m.name)
                _verify_amf_with_zig(self, p_amf, len(metas))

            bb_cfg = _read_embedded_json(rep.backbone_amf)
            self.assertEqual(bb_cfg["model_type"], "mimo_v2")
            self.assertEqual(bb_cfg["layer_types"], ["full_attention", "sliding_attention", "sliding_attention"])
            self.assertTrue(bb_cfg["add_swa_attention_sink_bias"])
            self.assertEqual(bb_cfg["attention_projection_layout"], "split")

            with AmfReader(rep.backbone_amf) as rdr:
                self.assertIn("layers.1.self_attn.sink_bias", rdr.by_name)
                self.assertNotIn("layers.0.self_attn.sink_bias", rdr.by_name)
                self.assertIn("layers.1.mlp.gate.bias", rdr.by_name)
                self.assertEqual(rdr.by_name["layers.1.mlp.experts.0.gate_proj.weight"].dtype, DType.GGUF_MXFP4)
                self.assertEqual(rdr.by_name["layers.1.mlp.experts.0.gate_proj.weight"].kind, TensorKind.HOT_EXPERT)
                self.assertEqual(rdr.by_name["layers.1.mlp.experts.2.gate_proj.weight"].kind, TensorKind.COLD_EXPERT)

                # Verify exact MXFP4 round-trip on layer 1 expert 0 gate_proj row 3
                e_mxfp4 = rdr.by_name["layers.1.mlp.experts.0.gate_proj.weight"]
                blob_m = rdr.read_blob(e_mxfp4, verify_crc=True)
                rb_m = (hidden // 32) * 17
                got_m = dequantize_gguf_amf_row_f32(DType.GGUF_MXFP4, blob_m[3 * rb_m : 4 * rb_m], hidden)
                exp_m = dequantize_hf_mxfp4_row_f32(
                    mxfp4_w[3 * (hidden // 2) : 4 * (hidden // 2)],
                    mxfp4_s[3 * (hidden // 32) : 4 * (hidden // 32)],
                    hidden,
                )
                self.assertEqual(got_m, exp_m)

                # Verify exact TP=4 de-interleaving & Q/K/V split on layer 1
                for proj, global_offset, rows_cnt in (
                    ("q_proj.weight", 0, spec.q_total),
                    ("k_proj.weight", spec.q_total, spec.k_total),
                    ("v_proj.weight", spec.q_total + spec.k_total, spec.v_total),
                ):
                    pe = rdr.by_name[f"layers.1.self_attn.{proj}"]
                    self.assertIn(pe.dtype, (DType.FP8_E4M3_ROW, DType.FP8_E4M3_BLOCK))
                    self.assertEqual(pe.shape, (rows_cnt, hidden))
                    pblob = rdr.read_blob(pe, verify_crc=True)
                    data_mv = memoryview(pblob)[: pe.data_bytes]
                    sc_cnt = pe.params_bytes // 4
                    sc_vals = struct.unpack(f"<{sc_cnt}f", pblob[pe.params_offset : pe.params_offset + pe.params_bytes])
                    for check_r in (0, 17, rows_cnt - 1):
                        if pe.dtype == DType.FP8_E4M3_ROW:
                            got_f32 = dequantize_fp8_row_f32(
                                data_mv[check_r * hidden : (check_r + 1) * hidden],
                                sc_vals[check_r],
                                hidden,
                            )
                        else:
                            got_f32 = dequantize_fp8_block_row_f32(
                                data_mv[check_r * hidden : (check_r + 1) * hidden],
                                check_r,
                                sc_vals,
                                rows_cnt,
                                hidden,
                            )
                        src_r, src_sc = map_mimo_contiguous_qkv_row(spec, global_offset + check_r)
                        exp_f32 = dequantize_mimo_qkv_row_f32(
                            qkv_raw_l1[src_r * hidden : (src_r + 1) * hidden],
                            src_sc,
                            qkv_sc_l1,
                            hidden,
                        )
                        for g, exp_v in zip(got_f32, exp_f32):
                            self.assertAlmostEqual(g, exp_v, delta=max(0.05, abs(exp_v) * 0.15))

            # Check MTP embedded metadata & TP=4 numeric parity
            assert rep.mtp_amf is not None
            mtp_cfg = _read_embedded_json(rep.mtp_amf)
            self.assertEqual(mtp_cfg["format"], "mtp")
            self.assertTrue(mtp_cfg["mtp_chained"])
            self.assertEqual(mtp_cfg["num_hidden_layers"], 3)
            with AmfReader(rep.mtp_amf) as mrdr:
                for k in range(3):
                    self.assertIn(f"layers.{k}.enorm.weight", mrdr.by_name)
                    self.assertIn(f"layers.{k}.hnorm.weight", mrdr.by_name)
                    self.assertIn(f"layers.{k}.eh_proj.weight", mrdr.by_name)
                    self.assertIn(f"layers.{k}.post_attention_layernorm.weight", mrdr.by_name)
                    self.assertIn(f"layers.{k}.final_layernorm.weight", mrdr.by_name)
                    self.assertIn(f"layers.{k}.self_attn.sink_bias", mrdr.by_name)

            # Check DFlash embedded metadata & tensor names
            assert rep.dflash_amf is not None
            df_cfg = _read_embedded_json(rep.dflash_amf)
            self.assertEqual(df_cfg["draft_kind"], "dflash")
            self.assertEqual(df_cfg["block_size"], 8)
            self.assertEqual(df_cfg["markov_rank"], 0)
            with AmfReader(rep.dflash_amf) as drdr:
                self.assertIn("fc.weight", drdr.by_name)
                self.assertIn("hidden_norm.weight", drdr.by_name)
                self.assertIn("norm.weight", drdr.by_name)
                self.assertIn("layers.0.self_attn.sink_bias", drdr.by_name)

    def test_convert_qwen35_mlx4bit_exact_slicing_and_vision_sidecar(self) -> None:
        # Compact MLX 4-bit qwen3_5 checkpoint (group_size=64, 4L: 3 GDN + 1 GQA, + ViT)
        h = 128
        inter = 128
        vocab = 128
        # GDN: 2 key heads * 64 = 128 Q, 128 K; 2 val heads * 64 = 128 V => conv_dim = 384
        l_kh, l_vh, l_kd, l_vd = 2, 2, 64, 64
        kq_dim = l_kh * l_kd
        z_dim = l_vh * l_vd
        conv_dim = 2 * kq_dim + z_dim
        # GQA: 2 heads * 64 head_dim = 128, doubled q_proj = 256 rows (head-interleaved)
        n_heads, kv_heads, hd = 2, 1, 64

        rng = random.Random(303)

        def _make_mlx4(rows: int, cols: int) -> tuple[bytes, bytes, bytes, list[float]]:
            gs = 64
            g_cnt = cols // gs
            u32_cols = cols // 8
            w_words = bytearray()
            sc_vals: list[float] = []
            bi_vals: list[float] = []
            flat_f32: list[float] = []
            for _r in range(rows):
                for _g in range(g_cnt):
                    sc = round(rng.uniform(0.02, 0.08), 4)
                    bi = round(rng.uniform(-0.3, 0.3), 4)
                    sc_vals.append(sc)
                    bi_vals.append(bi)
                    q = [rng.randrange(16) for _ in range(gs)]
                    for qv in q:
                        flat_f32.append(bi + qv * sc)
                    for wi in range(gs // 8):
                        word = 0
                        for ni in range(8):
                            word |= (q[wi * 8 + ni] & 0xF) << (4 * ni)
                        w_words.extend(struct.pack("<I", word))
            sc_raw = f32_values_to_bf16_bytes(sc_vals)
            bi_raw = f32_values_to_bf16_bytes(bi_vals)
            # Re-derive exact float values after BF16 scale/bias quantization so comparison is exact!
            sc_bf = bf16_bytes_to_f32_list(sc_raw)
            bi_bf = bf16_bytes_to_f32_list(bi_raw)
            re_f32: list[float] = []
            pos = 0
            for r in range(rows):
                for g in range(g_cnt):
                    s = sc_bf[r * g_cnt + g]
                    b = bi_bf[r * g_cnt + g]
                    for wi in range(gs // 8):
                        w = struct.unpack_from("<I", w_words, (r * u32_cols + g * (gs // 8) + wi) * 4)[0]
                        for ni in range(8):
                            re_f32.append(b + ((w >> (4 * ni)) & 0xF) * s)
                        pos += 8
            return bytes(w_words), sc_raw, bi_raw, re_f32

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp) / "qwen35_mlx"
            ckpt.mkdir(parents=True)
            ltypes = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            raw_cfg = {
                "model_type": "qwen3_5",
                "quantization": {"group_size": 64, "bits": 4},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": h,
                    "intermediate_size": inter,
                    "num_hidden_layers": 4,
                    "vocab_size": vocab,
                    "num_attention_heads": n_heads,
                    "num_key_value_heads": kv_heads,
                    "head_dim": hd,
                    "attn_output_gate": True,
                    "layer_types": ltypes,
                    "full_attention_interval": 4,
                    "linear_num_key_heads": l_kh,
                    "linear_num_value_heads": l_vh,
                    "linear_key_head_dim": l_kd,
                    "linear_value_head_dim": l_vd,
                    "linear_conv_kernel_dim": 4,
                    "rope_parameters": {
                        "mrope_interleaved": True,
                        "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000.0,
                    },
                    "tie_word_embeddings": False,
                },
                "vision_config": {"depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 2},
            }
            (ckpt / "config.json").write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")

            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "model.language_model.embed_tokens.weight": ("BF16", (vocab, h), f32_values_to_bf16_bytes(_det_f32(1, vocab * h))),
                "model.language_model.norm.weight": ("F32", (h,), struct.pack(f"<{h}f", *_det_f32(2, h))),
                "model.language_model.lm_head.weight": ("BF16", (vocab, h), f32_values_to_bf16_bytes(_det_f32(3, vocab * h))),
                "model.visual.patch_embed.proj.weight": ("BF16", (64, 48), f32_values_to_bf16_bytes(_det_f32(4, 64 * 48))),
                "model.visual.blocks.0.attn.qkv.weight": ("BF16", (192, 64), f32_values_to_bf16_bytes(_det_f32(5, 192 * 64))),
            }

            qkv_ref_l0: list[float] = []
            a_ref_l0: list[float] = []
            q2x_ref_l3: list[float] = []

            def _add_mlx(prefix: str, rows: int, cols: int) -> list[float]:
                w_b, s_b, b_b, vals = _make_mlx4(rows, cols)
                tensors[f"{prefix}.weight"] = ("U32", (rows, cols // 8), w_b)
                tensors[f"{prefix}.scales"] = ("BF16", (rows, cols // 64), s_b)
                tensors[f"{prefix}.biases"] = ("BF16", (rows, cols // 64), b_b)
                return vals

            for l_idx, lt in enumerate(ltypes):
                tensors[f"model.language_model.layers.{l_idx}.input_layernorm.weight"] = ("F32", (h,), struct.pack(f"<{h}f", *[0.1] * h))
                tensors[f"model.language_model.layers.{l_idx}.post_attention_layernorm.weight"] = ("F32", (h,), struct.pack(f"<{h}f", *[0.1] * h))
                _add_mlx(f"model.language_model.layers.{l_idx}.mlp.gate_proj", inter, h)
                _add_mlx(f"model.language_model.layers.{l_idx}.mlp.up_proj", inter, h)
                _add_mlx(f"model.language_model.layers.{l_idx}.mlp.down_proj", h, inter)
                if lt == "linear_attention":
                    vals_qkv = _add_mlx(f"model.language_model.layers.{l_idx}.linear_attn.in_proj_qkv", conv_dim, h)
                    _add_mlx(f"model.language_model.layers.{l_idx}.linear_attn.in_proj_z", z_dim, h)
                    _add_mlx(f"model.language_model.layers.{l_idx}.linear_attn.out_proj", h, z_dim)
                    vals_a = _add_mlx(f"model.language_model.layers.{l_idx}.linear_attn.in_proj_a", l_vh, h)
                    _add_mlx(f"model.language_model.layers.{l_idx}.linear_attn.in_proj_b", l_vh, h)
                    if l_idx == 0:
                        qkv_ref_l0 = vals_qkv
                        a_ref_l0 = vals_a
                    tensors[f"model.language_model.layers.{l_idx}.linear_attn.conv1d.weight"] = (
                        "BF16",
                        (conv_dim, 1, 4),
                        f32_values_to_bf16_bytes(_det_f32(50 + l_idx, conv_dim * 4)),
                    )
                    tensors[f"model.language_model.layers.{l_idx}.linear_attn.A_log"] = ("F32", (l_vh,), struct.pack(f"<{l_vh}f", *[-1.5] * l_vh))
                    tensors[f"model.language_model.layers.{l_idx}.linear_attn.dt_bias"] = ("F32", (l_vh,), struct.pack(f"<{l_vh}f", *[0.2] * l_vh))
                    tensors[f"model.language_model.layers.{l_idx}.linear_attn.norm.weight"] = ("F32", (l_vd,), struct.pack(f"<{l_vd}f", *[1.0] * l_vd))
                else:
                    vals_2xq = _add_mlx(f"model.language_model.layers.{l_idx}.self_attn.q_proj", 2 * n_heads * hd, h)
                    if l_idx == 3:
                        q2x_ref_l3 = vals_2xq
                    _add_mlx(f"model.language_model.layers.{l_idx}.self_attn.k_proj", kv_heads * hd, h)
                    _add_mlx(f"model.language_model.layers.{l_idx}.self_attn.v_proj", kv_heads * hd, h)
                    _add_mlx(f"model.language_model.layers.{l_idx}.self_attn.o_proj", h, n_heads * hd)
                    tensors[f"model.language_model.layers.{l_idx}.self_attn.q_norm.weight"] = ("F32", (hd,), struct.pack(f"<{hd}f", *[0.05] * hd))
                    tensors[f"model.language_model.layers.{l_idx}.self_attn.k_norm.weight"] = ("F32", (hd,), struct.pack(f"<{hd}f", *[0.05] * hd))

            _write_safetensors(ckpt / "model.safetensors", tensors)
            out_dir = pathlib.Path(tmp) / "out_qwen35"
            rep = convert_qwen35_checkpoint(
                ckpt,
                out_dir / "qwen35.amf",
                vision_sidecar=out_dir / "vision.amf",
            )
            self.assertIsNotNone(rep.vision_amf)
            assert rep.vision_amf is not None
            for m in rep.backbone_metas:
                self.assertLessEqual(len(m.name.encode("utf-8")), 47, m.name)
                self.assertFalse(m.name.startswith("visual."))
            _verify_amf_with_zig(self, rep.backbone_amf, len(rep.backbone_metas))
            _verify_amf_with_zig(self, rep.vision_amf, len(rep.vision_metas))

            cfg = _read_embedded_json(rep.backbone_amf)
            self.assertEqual(cfg["model_type"], "qwen3_5")
            self.assertTrue(cfg["rmsnorm_unit_offset"])
            self.assertEqual(cfg["linear_attn_norm_offset"], 0.0)
            self.assertTrue(cfg["attn_output_gate"])
            self.assertAlmostEqual(cfg["partial_rotary_factor"], 0.25)
            self.assertEqual(cfg["mrope_section"], [11, 11, 10])

            with AmfReader(rep.backbone_amf) as rdr:
                # 1. `conv1d.weight` squeezed from [conv_dim, 1, 4] to 2D [conv_dim, 4]
                conv_e = rdr.by_name["layers.0.linear_attn.conv1d.weight"]
                self.assertEqual(conv_e.shape, (conv_dim, 4))

                # 2. `in_proj_a` dequantized from MLX 4-bit into BF16 `linear_attn.a_proj.weight`
                a_e = rdr.by_name["layers.0.linear_attn.a_proj.weight"]
                self.assertEqual(a_e.dtype, DType.BF16)
                a_blob = rdr.read_blob(a_e, verify_crc=True)[: a_e.data_bytes]
                got_a = bf16_bytes_to_f32_list(a_blob)
                for g, e in zip(got_a, a_ref_l0):
                    self.assertAlmostEqual(g, e, delta=0.02)

                # 3. `in_proj_qkv` losslessly sliced into Q4_AFFINE `linear_attn.{q,k,v}_proj.weight`
                for sub_p, r_off, r_cnt in (
                    ("q_proj.weight", 0, kq_dim),
                    ("k_proj.weight", kq_dim, kq_dim),
                    ("v_proj.weight", 2 * kq_dim, z_dim),
                ):
                    se = rdr.by_name[f"layers.0.linear_attn.{sub_p}"]
                    self.assertEqual(se.dtype, DType.Q4_AFFINE)
                    self.assertEqual(se.shape, (r_cnt, h))
                    sblob = rdr.read_blob(se, verify_crc=True)
                    rb = h // 2
                    pb = (h // 64) * 4
                    for row_i in (0, r_cnt // 2, r_cnt - 1):
                        got_r = dequantize_q4_row_f32(
                            sblob[row_i * rb : (row_i + 1) * rb],
                            sblob[se.params_offset + row_i * pb : se.params_offset + (row_i + 1) * pb],
                            h,
                            group_size=64,
                        )
                        exp_start = (r_off + row_i) * h
                        for g, e in zip(got_r, qkv_ref_l0[exp_start : exp_start + h]):
                            self.assertAlmostEqual(g, e, delta=1e-2)

                # 4. Head-interleaved GQA 2x `q_proj` -> `q_proj` + `q_gate_proj` exact per-head parity!
                qe = rdr.by_name["layers.3.self_attn.q_proj.weight"]
                qge = rdr.by_name["layers.3.self_attn.q_gate_proj.weight"]
                self.assertEqual(qe.dtype, DType.Q4_AFFINE)
                self.assertEqual(qge.dtype, DType.Q4_AFFINE)
                self.assertEqual(qe.shape, (n_heads * hd, h))
                self.assertEqual(qge.shape, (n_heads * hd, h))
                qblob = rdr.read_blob(qe, verify_crc=True)
                qgblob = rdr.read_blob(qge, verify_crc=True)
                rb = h // 2
                pb = (h // 64) * 4
                for head_i in range(n_heads):
                    for local_r in (0, hd - 1):
                        out_r = head_i * hd + local_r
                        got_q = dequantize_q4_row_f32(
                            qblob[out_r * rb : (out_r + 1) * rb],
                            qblob[qe.params_offset + out_r * pb : qe.params_offset + (out_r + 1) * pb],
                            h,
                            group_size=64,
                        )
                        src_q_start = (head_i * 2 * hd + local_r) * h
                        for g, e in zip(got_q, q2x_ref_l3[src_q_start : src_q_start + h]):
                            self.assertAlmostEqual(g, e, delta=1e-2)

                        got_g = dequantize_q4_row_f32(
                            qgblob[out_r * rb : (out_r + 1) * rb],
                            qgblob[qge.params_offset + out_r * pb : qge.params_offset + (out_r + 1) * pb],
                            h,
                            group_size=64,
                        )
                        src_g_start = (head_i * 2 * hd + hd + local_r) * h
                        for g, e in zip(got_g, q2x_ref_l3[src_g_start : src_g_start + h]):
                            self.assertAlmostEqual(g, e, delta=1e-2)

            # Verify --strip-vision path too
            stripped_amf = out_dir / "stripped.amf"
            rep_strip = convert_qwen35_checkpoint(ckpt, stripped_amf, strip_vision=True)
            self.assertIsNone(rep_strip.vision_amf)
            with AmfReader(stripped_amf) as srdr:
                self.assertFalse(any(e.name.startswith("visual.") for e in srdr.entries))

    def test_synthetic_gguf_mimo2_mtp_dflash_and_qwen35_conversions(self) -> None:
        h = 128
        with tempfile.TemporaryDirectory() as tmp:
            troot = pathlib.Path(tmp)

            # 1. `dflash` GGUF (`arch = dflash`, 1-indexed `target_layers=[1,12,24,36,48]`, `attn_sinks.weight`)
            df_gguf = troot / "dflash.gguf"
            df_kv: dict[str, tuple[str, object]] = {
                "general.architecture": ("str", "dflash"),
                "dflash.block_count": ("u32", 1),
                "dflash.embedding_length": ("u32", h),
                "dflash.feed_forward_length": ("u32", 128),
                "dflash.attention.head_count": ("u32", 4),
                "dflash.attention.head_count_kv": ("u32", 2),
                "dflash.attention.key_length": ("u32", 32),
                "dflash.attention.value_length": ("u32", 32),
                "dflash.attention.value_scale": ("f32", 0.612),
                "dflash.attention.causal": ("bool", False),
                "dflash.block_size": ("u32", 16),
                "dflash.target_layers": ("arr", ("i32", [1, 12, 24, 36, 48])),
                "tokenizer.ggml.mask_token_id": ("u32", 151675),
            }
            df_tensors = [
                ("fc.weight", (h, 5 * h), 30, f32_values_to_bf16_bytes(_det_f32(1, h * 5 * h))),
                ("hidden_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("output_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.0.attn_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.0.ffn_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.0.attn_q.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(2, 128 * h))),
                ("blk.0.attn_k.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(3, 64 * h))),
                ("blk.0.attn_v.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(4, 64 * h))),
                ("blk.0.attn_output.weight", (h, 128), 30, f32_values_to_bf16_bytes(_det_f32(5, h * 128))),
                ("blk.0.attn_q_norm.weight", (32,), 0, struct.pack("<32f", *[1.0] * 32)),
                ("blk.0.attn_k_norm.weight", (32,), 0, struct.pack("<32f", *[1.0] * 32)),
                ("blk.0.attn_sinks.weight", (4,), 0, struct.pack("<4f", *[-0.3] * 4)),
                ("blk.0.ffn_gate.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(6, 128 * h))),
                ("blk.0.ffn_up.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(7, 128 * h))),
                ("blk.0.ffn_down.weight", (h, 128), 30, f32_values_to_bf16_bytes(_det_f32(8, h * 128))),
            ]
            _write_synth_gguf(df_gguf, df_kv, df_tensors)
            df_amf = troot / "dflash_from_gguf.amf"
            df_metas = export_mimo_dflash(df_gguf, df_amf)
            _verify_amf_with_zig(self, df_amf, len(df_metas))
            df_cfg = _read_embedded_json(df_amf)
            self.assertEqual(df_cfg["target_layer_ids"], [0, 11, 23, 35, 47])
            self.assertEqual(df_cfg["block_size"], 16)
            self.assertEqual(df_cfg["mask_token_id"], 151675)
            with AmfReader(df_amf) as drdr:
                self.assertIn("fc.weight", drdr.by_name)
                self.assertIn("hidden_norm.weight", drdr.by_name)
                self.assertIn("norm.weight", drdr.by_name)
                self.assertIn("layers.0.self_attn.sink_bias", drdr.by_name)

            # 2. `mtp` GGUF (`arch = mimo2`, `blk.48..50` remapped to `layers.0..2.*`, `attn_qkv` split!)
            mtp_gguf = troot / "mtp-MiMo-V2.6-Flash-RL-BF16.gguf"
            mtp_kv: dict[str, tuple[str, object]] = {
                "general.architecture": ("str", "mimo2"),
                "mimo2.block_count": ("u32", 51),
                "mimo2.nextn_predict_layers": ("u32", 3),
                "mimo2.embedding_length": ("u32", h),
                "mimo2.feed_forward_length": ("u32", 128),
                "mimo2.attention.head_count": ("u32", 4),
                "mimo2.attention.head_count_kv": ("arr", ("i32", [1] * 48 + [2, 2, 2])),
                "mimo2.attention.key_length": ("u32", 64),
                "mimo2.attention.value_length": ("u32", 32),
                "mimo2.attention.sliding_window": ("u32", 128),
                "mimo2.attention.sliding_window_pattern": ("arr", ("i32", [0] * 48 + [1, 1, 1])),
            }
            # q=4*64=256, k=2*64=128, v=2*32=64 => 448 rows in attn_qkv
            mtp_g_tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
            for idx, b in enumerate((48, 49, 50)):
                mtp_g_tensors.extend([
                    (f"blk.{b}.nextn.enorm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                    (f"blk.{b}.nextn.hnorm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                    (f"blk.{b}.nextn.eh_proj.weight", (h, 2 * h), 30, f32_values_to_bf16_bytes(_det_f32(20 + idx, h * 2 * h))),
                    (f"blk.{b}.attn_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                    (f"blk.{b}.attn_qkv.weight", (448, h), 30, f32_values_to_bf16_bytes(_det_f32(30 + idx, 448 * h))),
                    (f"blk.{b}.attn_output.weight", (h, 4 * 32), 30, f32_values_to_bf16_bytes(_det_f32(40 + idx, h * 128))),
                    (f"blk.{b}.attn_sinks.weight", (4,), 0, struct.pack("<4f", *[-0.1] * 4)),
                    (f"blk.{b}.post_attention_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                    (f"blk.{b}.ffn_post_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                    (f"blk.{b}.ffn_gate.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(50 + idx, 128 * h))),
                    (f"blk.{b}.ffn_up.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(60 + idx, 128 * h))),
                    (f"blk.{b}.ffn_down.weight", (h, 128), 30, f32_values_to_bf16_bytes(_det_f32(70 + idx, h * 128))),
                ])
            _write_synth_gguf(mtp_gguf, mtp_kv, mtp_g_tensors)
            mtp_amf = troot / "mtp_from_gguf.amf"
            mtp_metas = pack_amf_from_gguf(mtp_gguf, mtp_amf, mimo_mtp_mode=True)
            _verify_amf_with_zig(self, mtp_amf, len(mtp_metas))
            m_cfg = _read_embedded_json(mtp_amf)
            self.assertEqual(m_cfg["format"], "mtp")
            self.assertEqual(m_cfg["num_hidden_layers"], 3)
            with AmfReader(mtp_amf) as mrdr:
                self.assertIn("layers.0.enorm.weight", mrdr.by_name)
                self.assertIn("layers.2.final_layernorm.weight", mrdr.by_name)
                self.assertIn("layers.1.self_attn.q_proj.weight", mrdr.by_name)
                self.assertIn("layers.1.self_attn.k_proj.weight", mrdr.by_name)
                self.assertIn("layers.1.self_attn.v_proj.weight", mrdr.by_name)
                self.assertEqual(mrdr.by_name["layers.1.self_attn.v_proj.weight"].shape, (64, h))
                self.assertIn("layers.2.self_attn.sink_bias", mrdr.by_name)

            # 3. `qwen35` GGUF (`arch = qwen35`, GDN `attn_qkv` + `ssm_*` + head-interleaved 2x `attn_q`)
            q_gguf = troot / "qwen35.gguf"
            q_kv: dict[str, tuple[str, object]] = {
                "general.architecture": ("str", "qwen35"),
                "qwen35.block_count": ("u32", 2),
                "qwen35.embedding_length": ("u32", h),
                "qwen35.feed_forward_length": ("u32", 128),
                "qwen35.attention.head_count": ("u32", 2),
                "qwen35.attention.head_count_kv": ("u32", 1),
                "qwen35.attention.key_length": ("u32", 64),
                "qwen35.ssm.group_count": ("u32", 2),
                "qwen35.ssm.time_step_rank": ("u32", 2),
                "qwen35.ssm.state_size": ("u32", 64),
                "qwen35.ssm.inner_size": ("u32", 128),
            }
            q_g_tensors = [
                ("token_embd.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(1, 64 * h))),
                ("output_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("output.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(2, 64 * h))),
                # L0: GDN
                ("blk.0.attn_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.0.post_attention_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.0.attn_qkv.weight", (384, h), 30, f32_values_to_bf16_bytes(_det_f32(3, 384 * h))),
                ("blk.0.attn_gate.weight", (128, h), 30, f32_values_to_bf16_bytes(_det_f32(4, 128 * h))),
                ("blk.0.ssm_out.weight", (h, 128), 30, f32_values_to_bf16_bytes(_det_f32(5, h * 128))),
                ("blk.0.ssm_alpha.weight", (2, h), 30, f32_values_to_bf16_bytes(_det_f32(6, 2 * h))),
                ("blk.0.ssm_beta.weight", (2, h), 30, f32_values_to_bf16_bytes(_det_f32(7, 2 * h))),
                ("blk.0.ssm_conv1d.weight", (384, 4), 30, f32_values_to_bf16_bytes(_det_f32(8, 384 * 4))),
                ("blk.0.ssm_a", (2,), 0, struct.pack("<2f", -1.5, -1.5)),
                ("blk.0.ssm_dt.bias", (2,), 0, struct.pack("<2f", 0.2, 0.2)),
                ("blk.0.ssm_norm.weight", (64,), 0, struct.pack("<64f", *[1.0] * 64)),
                # L1: GQA 2x q_proj (256 rows = 2 * 2 * 64)
                ("blk.1.attn_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.1.post_attention_norm.weight", (h,), 0, struct.pack(f"<{h}f", *[1.0] * h)),
                ("blk.1.attn_q.weight", (256, h), 30, f32_values_to_bf16_bytes(_det_f32(9, 256 * h))),
                ("blk.1.attn_k.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(10, 64 * h))),
                ("blk.1.attn_v.weight", (64, h), 30, f32_values_to_bf16_bytes(_det_f32(11, 64 * h))),
                ("blk.1.attn_output.weight", (h, 128), 30, f32_values_to_bf16_bytes(_det_f32(12, h * 128))),
            ]
            _write_synth_gguf(q_gguf, q_kv, q_g_tensors)
            q_amf = troot / "qwen35_from_gguf.amf"
            rep_qg = convert_qwen35_checkpoint(q_gguf, q_amf)
            _verify_amf_with_zig(self, q_amf, len(rep_qg.backbone_metas))
            with AmfReader(q_amf) as qrdr:
                self.assertIn("layers.0.linear_attn.q_proj.weight", qrdr.by_name)
                self.assertIn("layers.0.linear_attn.g_proj.weight", qrdr.by_name)
                self.assertIn("layers.0.linear_attn.A_log", qrdr.by_name)
                self.assertIn("layers.1.self_attn.q_proj.weight", qrdr.by_name)
                self.assertIn("layers.1.self_attn.q_gate_proj.weight", qrdr.by_name)
                self.assertEqual(qrdr.by_name["layers.1.self_attn.q_gate_proj.weight"].shape, (128, h))

    def test_gsq_rco_protects_mimo_and_qwen35_sensitive_tensors_and_lora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            troot = pathlib.Path(tmp)
            # Generate synth qwen3_5 & re-quantize via GSQ-RCO v2 target_bits=3.0 -> A_log/dt_bias/conv1d/norms stay float!
            s_q35 = generate_synth_qwen35_bundle(troot / "synth_q35", config=Qwen35SynthConfig(vision_mode="strip"))
            gsq_q35 = troot / "q35_gsq.amf"
            rep_q = convert_qwen35_checkpoint(
                s_q35.backbone_amf,
                gsq_q35,
                quant_profile="gsq-rco",
                target_bits=3.5,
                strip_vision=True,
            )
            _verify_amf_with_zig(self, gsq_q35, len(rep_q.backbone_metas))
            with AmfReader(gsq_q35) as qrdr:
                for p_name in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
                    dt = qrdr.by_name[f"layers.0.linear_attn.{p_name}"].dtype
                    self.assertIn(dt, (DType.F32, DType.F16, DType.BF16), p_name)

            # Generate synth mimo (BF16 experts so GSQ-RCO quantizes experts) + protect sink_bias / gate.bias / norms
            s_mimo = generate_synth_mimo_bundle(
                troot / "synth_mimo_fp",
                config=MimoV2SynthConfig(expert_dtype="bf16", head_dim=64, v_head_dim=64, include_mtp=False, include_dflash=False),
            )
            gsq_mimo = troot / "mimo_gsq.amf"
            rep_m = convert_mimo_v2_checkpoint(
                s_mimo.backbone_amf,
                gsq_mimo,
                quant_profile="gsq-rco",
                target_bits=3.5,
            )
            _verify_amf_with_zig(self, gsq_mimo, len(rep_m.backbone_metas))
            with AmfReader(gsq_mimo) as mrdr:
                self.assertIn(mrdr.by_name["layers.1.self_attn.sink_bias"].dtype, (DType.F32, DType.F16, DType.BF16))
                self.assertIn(mrdr.by_name["layers.1.mlp.gate.bias"].dtype, (DType.F32, DType.F16, DType.BF16))

            # Verify tier profile constants match charter numbers
            self.assertTrue(MIMO_TIER_PROFILES["24gb-m4"].allow_nvme_cold_experts)
            self.assertFalse(MIMO_TIER_PROFILES["128gb-max"].allow_nvme_cold_experts)
            self.assertFalse(MIMO_TIER_PROFILES["512gb-ultra"].allow_nvme_cold_experts)

            # Verify LoRA PEFT mapping for Qwen3.5 GDN + GQA `q_gate_proj`
            peft_dir = troot / "peft_qwen35"
            peft_dir.mkdir()
            (peft_dir / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8")
            lora_tensors = {
                "base_model.model.layers.0.linear_attn.in_proj_z.lora_A.weight": ("F16", (8, 64), f32_values_to_f16_bytes([0.1] * 512)),
                "base_model.model.layers.0.linear_attn.in_proj_z.lora_B.weight": ("F16", (64, 8), f32_values_to_f16_bytes([0.1] * 512)),
                "base_model.model.layers.0.linear_attn.out_proj.lora_A.weight": ("F16", (8, 64), f32_values_to_f16_bytes([0.1] * 512)),
                "base_model.model.layers.0.linear_attn.out_proj.lora_B.weight": ("F16", (64, 8), f32_values_to_f16_bytes([0.1] * 512)),
                "base_model.model.layers.3.self_attn.q_gate_proj.lora_A.weight": ("F16", (8, 64), f32_values_to_f16_bytes([0.1] * 512)),
                "base_model.model.layers.3.self_attn.q_gate_proj.lora_B.weight": ("F16", (64, 8), f32_values_to_f16_bytes([0.1] * 512)),
            }
            _write_safetensors(peft_dir / "adapter_model.safetensors", lora_tensors)
            lora_amf = troot / "lora_q35.amf"
            lrep = pack_lora_from_peft(peft_dir, lora_amf)
            self.assertEqual(lrep.num_pairs, 3)
            _verify_amf_with_zig(self, lora_amf, 7)

    def test_synth_cli_subcommands_mimo_and_qwen35(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            troot = pathlib.Path(tmp)
            m_dir = troot / "cli_mimo_b16"
            q_dir = troot / "cli_qwen35"
            rc1 = cli_main(["synth-mimo", "--output-dir", str(m_dir), "--preset", "tiny-mimo-flash-b16"])
            self.assertEqual(rc1, 0)
            rc2 = cli_main(["synth-qwen35", "--output-dir", str(q_dir), "--preset", "tiny-qwen35"])
            self.assertEqual(rc2, 0)

            for fname in ("model.amf", "mtp.amf", "dflash.amf"):
                with AmfReader(m_dir / fname) as rdr:
                    n = len(rdr.entries)
                _verify_amf_with_zig(self, m_dir / fname, n)
            df_cfg = _read_embedded_json(m_dir / "dflash.amf")
            self.assertEqual(df_cfg["block_size"], 16)

            for fname in ("model.amf", "vision.amf"):
                with AmfReader(q_dir / fname) as rdr:
                    n = len(rdr.entries)
                _verify_amf_with_zig(self, q_dir / fname, n)

            # `cli inspect-metadata` round-trip check
            self.assertEqual(cli_main(["inspect-metadata", "--input", str(m_dir)]), 0)
            self.assertEqual(cli_main(["inspect-metadata", "--input", str(q_dir)]), 0)


if __name__ == "__main__":
    unittest.main()
