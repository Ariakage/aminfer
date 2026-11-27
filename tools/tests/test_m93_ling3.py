"""Milestone M9.3 Ling-3.1-flash-1019 (`bailing_hybrid`) W28/W27/RCO-Q3/MTP tests."""

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
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    TensorKind,
    TensorLayout,
    bf16_bytes_to_f32_list,
    f16_bytes_to_f32_list,
    f32_bytes_to_f32_list,
    f32_to_bf16_bytes,
    pack_amf_from_safetensors,
)
from aminfer_tools.cli import main as cli_main  # noqa: E402
from aminfer_tools.gguf import pack_amf_from_gguf  # noqa: E402
from aminfer_tools.gsq_rco import gsq_convert, quantize_rows_affine  # noqa: E402
from aminfer_tools.ling3 import (  # noqa: E402
    LING3_TIER_PROFILES,
    Ling3SynthConfig,
    convert_ling3_checkpoint,
    export_ling3_mtp,
    generate_synth_ling3_bundle,
    normalize_ling3_tensor_name,
    resolve_ling3_layer_types,
)
from aminfer_tools.lora import pack_lora_from_peft  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _verify_amf_with_zig(test: unittest.TestCase, path: pathlib.Path, expected_tensors: int | None = None) -> None:
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
    if expected_tensors is not None:
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
    hdr_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_raw)) & 7
    hdr_bytes = hdr_raw + (b" " * pad)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for chunk in payloads:
            fh.write(chunk)


def _pack_u32_from_mlx_qbytes(q_bytes: bytes, n_rows: int, k_cols: int, bits: int) -> bytes:
    """Repack byte-padded `q_bytes` into raw little-endian U32 words `[n_rows, (k_cols * bits) // 32]`."""
    row_b = (k_cols * bits) // 8
    words = bytearray(n_rows * row_b)
    words[:] = q_bytes
    return bytes(words)


def _f32_to_e4m3_bytes(vals: list[float]) -> bytes:
    from aminfer_tools.amf import f32_to_e4m3fn_byte

    return bytes(f32_to_e4m3fn_byte(v) for v in vals)


def _quantize_fp8_block(vals: list[float], rows: int, cols: int, block: int = 128) -> tuple[bytes, bytes, tuple[int, int]]:
    from aminfer_tools.amf import e4m3_byte_to_f32, f32_to_e4m3fn_byte

    rb = (rows + block - 1) // block
    cb = (cols + block - 1) // block
    scales = [1.0] * (rb * cb)
    q_bytes = bytearray(rows * cols)
    for br in range(rb):
        r0, r1 = br * block, min(rows, (br + 1) * block)
        for bc in range(cb):
            c0, c1 = bc * block, min(cols, (bc + 1) * block)
            max_abs = 0.0
            for r in range(r0, r1):
                base = r * cols
                for c in range(c0, c1):
                    max_abs = max(max_abs, abs(vals[base + c]))
            sc = (max_abs / 448.0) if max_abs > 1e-12 else 1.0
            scales[br * cb + bc] = sc
            inv = 1.0 / sc
            for r in range(r0, r1):
                base = r * cols
                for c in range(c0, c1):
                    q_bytes[base + c] = f32_to_e4m3fn_byte(vals[base + c] * inv)
    _ = e4m3_byte_to_f32(q_bytes[0])
    return bytes(q_bytes), struct.pack(f"<{len(scales)}f", *scales), (rb, cb)


def _pack_gguf_kv(key: str, vtype: int, val: object) -> bytes:
    kb = key.encode("utf-8")
    out = bytearray(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vtype))
    if vtype == 4:  # UINT32
        out += struct.pack("<I", int(val))  # type: ignore[arg-type]
    elif vtype == 6:  # FLOAT32
        out += struct.pack("<f", float(val))  # type: ignore[arg-type]
    elif vtype == 7:  # BOOL
        out += struct.pack("<B", 1 if val else 0)
    elif vtype == 8:  # STRING
        sb = str(val).encode("utf-8")
        out += struct.pack("<Q", len(sb)) + sb
    elif vtype == 9:  # ARRAY
        et, seq = val  # type: ignore[misc]
        seq_l = list(seq)
        out += struct.pack("<IQ", et, len(seq_l))
        for item in seq_l:
            if et == 4:
                out += struct.pack("<I", int(item))
            elif et == 8:
                sb = str(item).encode("utf-8")
                out += struct.pack("<Q", len(sb)) + sb
            else:
                raise ValueError(f"unsupported test array elem type {et}")
    else:
        raise ValueError(f"unsupported GGUF KV type {vtype}")
    return bytes(out)


def _write_gguf_v3(
    path: pathlib.Path,
    kv_entries: list[tuple[str, int, object]],
    tensors: list[tuple[str, int, tuple[int, ...], bytes]],
) -> None:
    """Write a minimal GGUF v3 file (`dims` in AMF shape order -> reversed to GGUF ne)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), len(kv_entries))
    for k, vt, val in kv_entries:
        header += _pack_gguf_kv(k, vt, val)
    rel_off = 0
    aligned_chunks: list[bytes] = []
    for name, ggml_type, amf_shape, raw in tensors:
        nb = name.encode("utf-8")
        gguf_dims = tuple(reversed(amf_shape))
        header += struct.pack("<Q", len(nb)) + nb
        header += struct.pack(f"<I{len(gguf_dims)}QIQ", len(gguf_dims), *gguf_dims, ggml_type, rel_off)
        pad = (-len(raw)) & 31
        aligned_chunks.append(raw + (b"\x00" * pad))
        rel_off += len(raw) + pad
    pad_hdr = (-len(header)) & 31
    header += b"\x00" * pad_hdr
    with path.open("wb") as fh:
        fh.write(header)
        for ch in aligned_chunks:
            fh.write(ch)


def _ling3_base_config(
    *,
    num_layers: int = 8,
    layer_group_size: int = 4,
    first_k_dense: int = 1,
    hidden: int = 128,
    inter: int = 256,
    moe_inter: int = 64,
    n_experts: int = 4,
    vocab: int = 192,
) -> dict[str, object]:
    return {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingMoeV3ForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "moe_shared_expert_intermediate_size": moe_inter,
        "num_hidden_layers": num_layers,
        "layer_group_size": layer_group_size,
        "first_k_dense_replace": first_k_dense,
        "vocab_size": vocab,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "qk_head_dim": 96,
        "v_head_dim": 32,
        "num_experts": n_experts,
        "num_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
        "linear_silu": True,
        "short_conv_kernel_size": 4,
        "gated_attention_proj_granularity_type": "head_wise",
        "rope_interleave": True,
        "rope_theta": 6000000.0,
        "rms_norm_eps": 1e-6,
        "num_nextn_predict_layers": 1,
        "mtp_use_kda": False,
        "tie_word_embeddings": False,
        "expert_swiglu_limit_list": [0.0] * 5 + [10.0] * (num_layers + 1 - 5 if num_layers >= 5 else 2),
        "share_expert_swiglu_limit_list": [0.0] * 5 + [10.0] * (num_layers + 1 - 5 if num_layers >= 5 else 2),
    }


def _det(rng: random.Random, n: int, scale: float = 0.04) -> list[float]:
    return [rng.gauss(0.0, scale) for _ in range(n)]


def _build_hf_ling3_safetensors(
    model_dir: pathlib.Path,
    cfg: dict[str, object],
    *,
    quant_mode: str = "bf16",
    seed: int = 7,
) -> pathlib.Path:
    """Create a realistic HuggingFace `bailing_hybrid` checkpoint (`bf16`, `fp8`, or `nvfp4`) + `1L` MTP."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    vocab = int(cfg["vocab_size"])  # type: ignore[arg-type]
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {f"<t{i}>": i for i in range(vocab)}}}),
        encoding="utf-8",
    )
    rng = random.Random(seed)
    n_layers = int(cfg["num_hidden_layers"])  # type: ignore[arg-type]
    first_k = int(cfg["first_k_dense_replace"])  # type: ignore[arg-type]
    h = int(cfg["hidden_size"])  # type: ignore[arg-type]
    inter = int(cfg["intermediate_size"])  # type: ignore[arg-type]
    moe_inter = int(cfg["moe_intermediate_size"])  # type: ignore[arg-type]
    n_exp = int(cfg["num_experts"])  # type: ignore[arg-type]
    n_heads = int(cfg["num_attention_heads"])  # type: ignore[arg-type]
    head_dim = int(cfg["head_dim"])  # type: ignore[arg-type]
    qkv_dim = n_heads * head_dim
    q_lora = int(cfg["q_lora_rank"])  # type: ignore[arg-type]
    kv_lora = int(cfg["kv_lora_rank"])  # type: ignore[arg-type]
    qk_nope = int(cfg["qk_nope_head_dim"])  # type: ignore[arg-type]
    qk_rope = int(cfg["qk_rope_head_dim"])  # type: ignore[arg-type]
    v_hd = int(cfg["v_head_dim"])  # type: ignore[arg-type]
    ltypes = resolve_ling3_layer_types(cfg)

    tmap: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def add_bf16(nm: str, shp: tuple[int, ...], mean: float = 0.0) -> None:
        numel = 1
        for d in shp:
            numel *= d
        vals = [mean + x for x in _det(rng, numel)]
        tmap[nm] = ("BF16", shp, f32_to_bf16_bytes(vals))

    def add_f32(nm: str, shp: tuple[int, ...], mean: float = 0.0) -> None:
        numel = 1
        for d in shp:
            numel *= d
        vals = [mean + x for x in _det(rng, numel)]
        tmap[nm] = ("F32", shp, struct.pack(f"<{numel}f", *vals))

    def add_linear(base: str, rows: int, cols: int, *, pin_bf16: bool = False) -> None:
        if pin_bf16 or quant_mode == "bf16":
            add_bf16(f"{base}.weight", (rows, cols))
            return
        vals = _det(rng, rows * cols)
        if quant_mode == "fp8":
            qb, sb, s_shp = _quantize_fp8_block(vals, rows, cols, 128)
            tmap[f"{base}.weight"] = ("F8_E4M3", (rows, cols), qb)
            if s_shp != (1, 1) or rows >= 128:
                tmap[f"{base}.weight_scale_inv"] = ("F32", s_shp, sb)
            else:
                tmap[f"{base}.weight_scale_inv"] = ("F32", (1, 1), struct.pack("<f", 0.01))
        elif quant_mode == "nvfp4":
            # Deterministic valid ModelOpt NVFP4 packed U8 `[rows, cols//2]` + `F8_E4M3` micro-scales.
            tmap[f"{base}.weight"] = ("U8", (rows, cols // 2), bytes((i * 17) & 0xFF for i in range(rows * (cols // 2))))
            tmap[f"{base}.weight_scale"] = ("F8_E4M3", (rows, cols // 16), bytes([0x38] * (rows * (cols // 16))))
            tmap[f"{base}.weight_scale_2"] = ("F32", (1,), struct.pack("<f", 0.05))
        else:
            raise ValueError(quant_mode)

    add_bf16("model.word_embeddings.weight", (vocab, h))
    add_bf16("model.norm.weight", (h,), mean=1.0)
    add_bf16("lm_head.weight", (vocab, h))

    def emit_layer(prefix: str, is_kda: bool, is_moe: bool) -> None:
        add_bf16(f"{prefix}.input_layernorm.weight", (h,), mean=1.0)
        add_bf16(f"{prefix}.post_attention_layernorm.weight", (h,), mean=1.0)
        if is_kda:
            for p in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                add_linear(f"{prefix}.attention.{p}", qkv_dim, h)
            add_linear(f"{prefix}.attention.o_proj", h, qkv_dim)
            add_bf16(f"{prefix}.attention.b_proj.weight", (n_heads, h))
            add_bf16(f"{prefix}.attention.o_norm.weight", (head_dim,), mean=1.0)
            for cn in ("q_conv1d", "k_conv1d", "v_conv1d"):
                add_bf16(f"{prefix}.attention.{cn}.weight", (qkv_dim, 1, 4))
            add_f32(f"{prefix}.attention.A_log", (n_heads,), mean=-2.0)
            add_bf16(f"{prefix}.attention.dt_bias", (qkv_dim,))
        else:
            add_linear(f"{prefix}.attention.q_a_proj", q_lora, h)
            add_bf16(f"{prefix}.attention.q_a_layernorm.weight", (q_lora,), mean=1.0)
            add_linear(f"{prefix}.attention.q_b_proj", n_heads * (qk_nope + qk_rope), q_lora)
            add_linear(f"{prefix}.attention.kv_a_proj_with_mqa", kv_lora + qk_rope, h)
            add_bf16(f"{prefix}.attention.kv_a_layernorm.weight", (kv_lora,), mean=1.0)
            add_linear(f"{prefix}.attention.kv_b_proj", n_heads * (qk_nope + v_hd), kv_lora)
            add_bf16(f"{prefix}.attention.g_proj.weight", (n_heads, h))
            add_linear(f"{prefix}.attention.dense", h, n_heads * v_hd)

        if not is_moe:
            add_linear(f"{prefix}.mlp.gate_proj", inter, h)
            add_linear(f"{prefix}.mlp.up_proj", inter, h)
            add_linear(f"{prefix}.mlp.down_proj", h, inter)
        else:
            add_bf16(f"{prefix}.mlp.gate.weight", (n_exp, h))
            add_f32(f"{prefix}.mlp.gate.expert_bias", (n_exp,))
            for sp, r_d, c_d in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                add_linear(f"{prefix}.mlp.shared_experts.{sp}", r_d, c_d)
            for e in range(n_exp):
                for sp, r_d, c_d in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                    add_linear(f"{prefix}.mlp.experts.{e}.{sp}", r_d, c_d)

    for li in range(n_layers):
        emit_layer(f"model.layers.{li}", ltypes[li] == "linear_attention", li >= first_k)

    # 1L MTP (`model.layers.{n_layers}.*`) uses Gated MLA + MoE + final_layernorm.
    mtp_pref = f"model.layers.{n_layers}"
    add_bf16(f"{mtp_pref}.enorm.weight", (h,), mean=1.0)
    add_bf16(f"{mtp_pref}.hnorm.weight", (h,), mean=1.0)
    add_bf16(f"{mtp_pref}.eh_proj.weight", (h, 2 * h))
    add_bf16(f"{mtp_pref}.final_layernorm.weight", (h,), mean=1.0)
    emit_layer(mtp_pref, False, True)

    sf_p = model_dir / "model.safetensors"
    _write_safetensors(sf_p, tmap)
    return sf_p


class TestM93Ling3Toolchain(unittest.TestCase):
    def test_ling3_tensor_normalization_and_47b_budget(self) -> None:
        ltypes = resolve_ling3_layer_types(_ling3_base_config(num_layers=72, layer_group_size=8))
        self.assertEqual(ltypes.count("linear_attention"), 63)
        self.assertEqual(ltypes.count("mla_attention"), 9)

        cases = {
            "model.word_embeddings.weight": "embed_tokens.weight",
            "model.layers.66.attention.f_proj.weight_packed": "layers.66.linear_attn.f_proj.weight",
            "model.layers.66.attention.g_proj.weight": "layers.66.linear_attn.g_proj.weight",
            "model.layers.66.attention.q_conv1d.weight": "layers.66.linear_attn.q_conv1d.weight",
            "model.layers.71.attention.dense.weight": "layers.71.self_attn.o_proj.weight",
            "model.layers.71.attention.g_proj.weight": "layers.71.self_attn.g_proj.weight",
            "model.layers.71.mlp.shared_experts.down_proj.weight": "layers.71.mlp.shared_experts.down_proj.weight",
            "model.layers.71.mlp.gate.expert_bias": "layers.71.mlp.gate.bias",
            "model.layers.71.mlp.gate.gate_proj.weight": "layers.71.mlp.gate.weight",
        }
        for raw, expected in cases.items():
            norm = normalize_ling3_tensor_name(raw, layer_types=ltypes)
            self.assertEqual(norm, expected)
            self.assertLessEqual(len(norm.encode("utf-8")), 47)

        self.assertEqual(
            normalize_ling3_tensor_name("model.layers.72.eh_proj.weight", mtp_source_layer=72, mtp_standalone=True),
            "layers.0.eh_proj.weight",
        )
        self.assertEqual(
            normalize_ling3_tensor_name("mtp.0.attention.dense.weight", mtp_standalone=True),
            "layers.0.self_attn.o_proj.weight",
        )
        self.assertEqual(normalize_ling3_tensor_name("eh_proj.weight", mtp_standalone=True), "eh_proj.weight")
        self.assertEqual(normalize_ling3_tensor_name("norm.weight", mtp_standalone=True), "norm.weight")

    def test_synth_ling3_rco_q3_and_verify_amf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "synth"
            rc = cli_main(
                [
                    "synth-ling3",
                    "--preset",
                    "tiny-ling3-rco-q3",
                    "--output-dir",
                    str(out_dir),
                    "--repack-sme2",
                    "--mtp-eh-proj-bits",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            bb_amf = out_dir / "model.amf"
            rco_amf = out_dir / "model-rco-q3.amf"
            mtp_amf = out_dir / "mtp.amf"
            self.assertTrue(bb_amf.is_file() and rco_amf.is_file() and mtp_amf.is_file())
            _verify_amf_with_zig(self, bb_amf)
            _verify_amf_with_zig(self, rco_amf)
            _verify_amf_with_zig(self, mtp_amf)

            with AmfReader(rco_amf) as rdr:
                cfg_meta = json.loads(rdr.read_metadata_bytes(META_CONFIG_JSON).decode("utf-8"))
                self.assertEqual(cfg_meta["model_type"], "bailing_hybrid")
                self.assertTrue(cfg_meta["dense_kda_rco_q3"])
                self.assertTrue(cfg_meta["no_kda_lora"])
                self.assertTrue(cfg_meta["kda_safe_gate"])
                self.assertAlmostEqual(cfg_meta["kda_lower_bound"], -5.0)
                self.assertEqual(cfg_meta["mla_gate_act"], "sigmoid")
                _ = json.loads(rdr.read_metadata_bytes(META_TOKENIZER_JSON).decode("utf-8"))
                kda_q3_cnt = 0
                mla_proj_cnt = 0
                for m in rdr.entries:
                    if m.kind == TensorKind.METADATA:
                        continue
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)
                    stem = m.name[:-7] if m.name.endswith(".weight") else m.name
                    if ".linear_attn." in stem and stem.endswith(
                        (".q_proj", ".k_proj", ".v_proj", ".f_proj", ".g_proj", ".o_proj")
                    ):
                        self.assertEqual(m.dtype, DType.Q3_AFFINE, f"expected RCO-Q3 on KDA dense {m.name}")
                        kda_q3_cnt += 1
                    elif stem.endswith((".linear_attn.b_proj", ".self_attn.g_proj")):
                        self.assertIn(m.dtype, (DType.BF16, DType.F16), f"gate {m.name} must stay float")
                    elif stem.endswith(
                        (".linear_attn.A_log", ".linear_attn.dt_bias", ".linear_attn.q_conv1d", ".linear_attn.k_conv1d", ".linear_attn.v_conv1d")
                    ):
                        self.assertEqual(m.dtype, DType.F32, f"{m.name} must be F32")
                    elif ".self_attn." in stem and (
                        stem.endswith(("_proj", ".kv_a_proj_with_mqa")) and not stem.endswith(".g_proj")
                    ):
                        self.assertNotIn(m.dtype, (DType.Q2_AFFINE, DType.Q3_AFFINE), f"MLA {m.name} fell below Q4")
                        mla_proj_cnt += 1
                    if m.kind == TensorKind.DENSE or m.kind == TensorKind.SHARED_EXPERT:
                        self.assertNotEqual(m.dtype, DType.Q2_AFFINE, f"backbone floor violated on {m.name}")
                self.assertEqual(kda_q3_cnt, 36)
                self.assertEqual(mla_proj_cnt, 10)

            with AmfReader(mtp_amf) as mrdr:
                eh = mrdr.by_name["layers.0.eh_proj.weight"]
                self.assertEqual(eh.dtype, DType.Q3_AFFINE)
                self.assertEqual(eh.layout, TensorLayout.SME2_PANEL64)
                mcfg = json.loads(mrdr.read_metadata_bytes(META_CONFIG_JSON).decode("utf-8"))
                self.assertEqual(mcfg["model_type"], "bailing_hybrid_mtp")
                self.assertEqual(mcfg["layer_types"], ["mla_attention"])

            # Verify tiny-ling3-512e geometry matches 512E config + cold tagging.
            dir_512 = pathlib.Path(tmp) / "s512"
            b512 = generate_synth_ling3_bundle(
                dir_512,
                config=Ling3SynthConfig(
                    preset="tiny-ling3-512e-test",
                    model_id="tiny-ling3-512e",
                    num_hidden_layers=4,
                    layer_group_size=4,
                    first_k_dense_replace=1,
                    n_routed_experts=16,
                    num_experts_per_tok=4,
                    n_group=4,
                    topk_group=2,
                    cold_experts_per_layer=8,
                ),
            )
            _verify_amf_with_zig(self, b512.backbone_amf)

    def test_convert_ling3_bf16_fp8_nvfp4_and_tier_profiles(self) -> None:
        self.assertIn("24gb-m4-rco-q3", LING3_TIER_PROFILES)
        self.assertTrue(LING3_TIER_PROFILES["24gb-m4-rco-q3"].dense_kda_rco_q3)
        self.assertFalse(LING3_TIER_PROFILES["256gb-studio"].dense_kda_rco_q3)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = _ling3_base_config()

            for mode in ("bf16", "fp8", "nvfp4"):
                src_dir = root / f"hf_{mode}"
                _build_hf_ling3_safetensors(src_dir, cfg, quant_mode=mode)
                out_amf = root / f"{mode}.amf"
                mtp_amf = root / f"{mode}_mtp.amf"
                mtp_cfg_out = root / f"{mode}_mtp_config.json"
                rep = convert_ling3_checkpoint(
                    src_dir,
                    out_amf,
                    mtp_output=mtp_amf,
                    mtp_config_output=mtp_cfg_out,
                    mtp_repack_sme2=True,
                    mtp_eh_proj_bits=3,
                    cold_experts_per_layer=2,
                )
                self.assertEqual(rep.model_type, "bailing_hybrid")
                with AmfReader(mtp_amf) as chk_mrdr:
                    self.assertEqual(len(rep.mtp_metas), len(chk_mrdr.entries))
                _verify_amf_with_zig(self, out_amf, len(rep.backbone_metas))
                _verify_amf_with_zig(self, mtp_amf, len(rep.mtp_metas))

                with AmfReader(out_amf) as rdr:
                    names = {m.name for m in rdr.entries}
                    sfx = ".weight" if mode == "bf16" else ""
                    self.assertIn(f"layers.3.self_attn.o_proj{sfx}", names)
                    self.assertIn(f"layers.0.linear_attn.f_proj{sfx}", names)
                    self.assertIn(f"layers.0.linear_attn.g_proj{sfx}", names)
                    self.assertNotIn("layers.8.eh_proj.weight", names)
                    self.assertNotIn("layers.8.eh_proj", names)
                    self.assertEqual(rdr.by_name["layers.0.linear_attn.q_conv1d.weight"].dtype, DType.F32)
                    self.assertEqual(rdr.by_name["layers.0.linear_attn.q_conv1d.weight"].shape, (128, 4))
                    self.assertEqual(rdr.by_name["layers.1.mlp.gate.bias"].dtype, DType.F32)
                    self.assertEqual(rdr.by_name["layers.0.linear_attn.dt_bias"].dtype, DType.F32)

                with AmfReader(mtp_amf) as mrdr:
                    eh = mrdr.by_name["layers.0.eh_proj.weight"]
                    self.assertEqual(eh.dtype, DType.Q3_AFFINE)
                    self.assertEqual(eh.layout, TensorLayout.SME2_PANEL64)
                mtp_cfg = json.loads(mtp_cfg_out.read_text(encoding="utf-8"))
                self.assertEqual(mtp_cfg["model_type"], "bailing_hybrid_mtp")
                self.assertEqual(mtp_cfg["layer_types"], ["mla_attention"])

            # Verify `24gb-m4-rco-q3` tier vs explicit `dense_kda_rco_q3=False` from BF16 source.
            bf16_dir = root / "hf_bf16"
            q3_out = root / "tier_rco_q3.amf"
            rep_q3 = convert_ling3_checkpoint(
                bf16_dir,
                q3_out,
                target_tier="24gb-m4-rco-q3",
                target_bits=3.50,
            )
            self.assertTrue(rep_q3.dense_kda_rco_q3)
            _verify_amf_with_zig(self, q3_out, len(rep_q3.backbone_metas))
            with AmfReader(q3_out) as rdr:
                self.assertEqual(rdr.by_name["layers.0.linear_attn.f_proj"].dtype, DType.Q3_AFFINE)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.g_proj"].dtype, DType.Q3_AFFINE)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.b_proj.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["layers.3.self_attn.g_proj.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["layers.3.self_attn.q_a_proj"].dtype, DType.Q4_AFFINE)

            no_q3_out = root / "no_rco_q3.amf"
            gsq_rep_no_q3 = gsq_convert(
                bf16_dir,
                no_q3_out,
                target_bits=3.50,
                dense_kda_rco_q3=False,
            )
            _verify_amf_with_zig(self, no_q3_out, len(gsq_rep_no_q3.metas))
            with AmfReader(no_q3_out) as rdr:
                f_entry = rdr.by_name.get("layers.0.linear_attn.f_proj") or rdr.by_name["layers.0.linear_attn.f_proj.weight"]
                self.assertNotIn(
                    f_entry.dtype,
                    (DType.Q2_AFFINE, DType.Q3_AFFINE),
                )

    def test_convert_ling3_mlx_embed_q_unembed_out_and_mtp(self) -> None:
        """Verify MLX `oQ4e-mtp` (`embed_q`+`unembed_out`->`kv_b_proj`, `switch_mlp`, `gate.gate_proj`, `mtp.0`)."""
        with tempfile.TemporaryDirectory() as tmp:
            mlx_dir = pathlib.Path(tmp) / "mlx_ling3"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            cfg = _ling3_base_config(num_layers=4, layer_group_size=4, first_k_dense=1, n_experts=4)
            (mlx_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            rng = random.Random(11)
            h, inter, moe_inter, n_exp, vocab = 128, 256, 64, 4, 192
            n_heads, head_dim, qkv_dim = 4, 32, 128
            q_lora, kv_lora, qk_nope, qk_rope, v_hd = 64, 64, 64, 32, 64

            tmap: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

            def add_mlx_q(prefix: str, rows: int, cols: int, bits: int = 4) -> list[list[float]]:
                mat = [_det(rng, cols, 0.1) for _ in range(rows)]
                qb, pb, _, _ = quantize_rows_affine(mat, bits=bits, group_size=64, use_rco=True)
                groups = cols // 64
                scales = bytearray(rows * groups * 2)
                biases = bytearray(rows * groups * 2)
                for r in range(rows):
                    for g in range(groups):
                        off = (r * groups + g) * 4
                        dst = (r * groups + g) * 2
                        scales[dst : dst + 2] = pb[off : off + 2]
                        biases[dst : dst + 2] = pb[off + 2 : off + 4]
                packed_cols = (cols * bits) // 32
                tmap[f"{prefix}.weight"] = ("U32", (rows, packed_cols), _pack_u32_from_mlx_qbytes(qb, rows, cols, bits))
                tmap[f"{prefix}.scales"] = ("F16", (rows, groups), bytes(scales))
                tmap[f"{prefix}.biases"] = ("F16", (rows, groups), bytes(biases))
                return mat

            def add_bf16(nm: str, shp: tuple[int, ...], mean: float = 0.0) -> None:
                n_el = 1
                for d in shp:
                    n_el *= d
                tmap[nm] = ("BF16", shp, f32_to_bf16_bytes([mean + x for x in _det(rng, n_el)]))

            def add_switch(layer_pref: str, proj: str, rows: int, cols: int) -> None:
                all_q: list[bytes] = []
                all_s: list[bytes] = []
                all_b: list[bytes] = []
                groups = cols // 64
                for _e in range(n_exp):
                    mat = [_det(rng, cols, 0.08) for _ in range(rows)]
                    qb, pb, _, _ = quantize_rows_affine(mat, bits=4, group_size=64, use_rco=True)
                    all_q.append(qb)
                    s_e = bytearray(rows * groups * 2)
                    b_e = bytearray(rows * groups * 2)
                    for i in range(rows * groups):
                        s_e[i * 2 : i * 2 + 2] = pb[i * 4 : i * 4 + 2]
                        b_e[i * 2 : i * 2 + 2] = pb[i * 4 + 2 : i * 4 + 4]
                    all_s.append(bytes(s_e))
                    all_b.append(bytes(b_e))
                tmap[f"{layer_pref}.mlp.switch_mlp.{proj}.weight"] = ("U32", (n_exp, rows, (cols * 4) // 32), b"".join(all_q))
                tmap[f"{layer_pref}.mlp.switch_mlp.{proj}.scales"] = ("F16", (n_exp, rows, groups), b"".join(all_s))
                tmap[f"{layer_pref}.mlp.switch_mlp.{proj}.biases"] = ("F16", (n_exp, rows, groups), b"".join(all_b))

            def add_mlx_kv_pair(layer_pref: str) -> tuple[list[list[float]], list[list[float]]]:
                eq_rows_truth: list[list[float]] = []
                uo_rows_truth: list[list[float]] = []
                eq_w, eq_s, eq_b = [], [], []
                uo_w, uo_s, uo_b = [], [], []
                eq_g = qk_nope // 64
                uo_g = kv_lora // 64
                for _h in range(n_heads):
                    m_eq = [_det(rng, qk_nope, 0.2) for _ in range(kv_lora)]
                    eq_rows_truth.extend(m_eq)
                    qb, pb, _, _ = quantize_rows_affine(m_eq, bits=8, group_size=64, use_rco=True)
                    eq_w.append(qb)
                    s_h, b_h = bytearray(kv_lora * eq_g * 2), bytearray(kv_lora * eq_g * 2)
                    for i in range(kv_lora * eq_g):
                        s_h[i * 2 : i * 2 + 2] = pb[i * 4 : i * 4 + 2]
                        b_h[i * 2 : i * 2 + 2] = pb[i * 4 + 2 : i * 4 + 4]
                    eq_s.append(bytes(s_h))
                    eq_b.append(bytes(b_h))

                    m_uo = [_det(rng, kv_lora, 0.2) for _ in range(v_hd)]
                    uo_rows_truth.extend(m_uo)
                    qb2, pb2, _, _ = quantize_rows_affine(m_uo, bits=8, group_size=64, use_rco=True)
                    uo_w.append(qb2)
                    s_h2, b_h2 = bytearray(v_hd * uo_g * 2), bytearray(v_hd * uo_g * 2)
                    for i in range(v_hd * uo_g):
                        s_h2[i * 2 : i * 2 + 2] = pb2[i * 4 : i * 4 + 2]
                        b_h2[i * 2 : i * 2 + 2] = pb2[i * 4 + 2 : i * 4 + 4]
                    uo_s.append(bytes(s_h2))
                    uo_b.append(bytes(b_h2))

                tmap[f"{layer_pref}.attention.embed_q.weight"] = ("U32", (n_heads, kv_lora, (qk_nope * 8) // 32), b"".join(eq_w))
                tmap[f"{layer_pref}.attention.embed_q.scales"] = ("F16", (n_heads, kv_lora, eq_g), b"".join(eq_s))
                tmap[f"{layer_pref}.attention.embed_q.biases"] = ("F16", (n_heads, kv_lora, eq_g), b"".join(eq_b))
                tmap[f"{layer_pref}.attention.unembed_out.weight"] = ("U32", (n_heads, v_hd, (kv_lora * 8) // 32), b"".join(uo_w))
                tmap[f"{layer_pref}.attention.unembed_out.scales"] = ("F16", (n_heads, v_hd, uo_g), b"".join(uo_s))
                tmap[f"{layer_pref}.attention.unembed_out.biases"] = ("F16", (n_heads, v_hd, uo_g), b"".join(uo_b))
                return eq_rows_truth, uo_rows_truth

            add_bf16("model.word_embeddings.weight", (vocab, h))
            add_bf16("model.norm.weight", (h,), mean=1.0)
            add_mlx_q("lm_head", vocab, h, 4)
            # Layer 0 KDA dense
            add_bf16("model.layers.0.input_layernorm.weight", (h,), mean=1.0)
            add_bf16("model.layers.0.post_attention_layernorm.weight", (h,), mean=1.0)
            for p in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                add_mlx_q(f"model.layers.0.attention.{p}", qkv_dim, h, 4)
            add_mlx_q("model.layers.0.attention.o_proj", h, qkv_dim, 4)
            add_mlx_q("model.layers.0.attention.b_proj", n_heads, h, 4)
            add_bf16("model.layers.0.attention.o_norm.weight", (head_dim,), mean=1.0)
            for cn in ("q_conv1d", "k_conv1d", "v_conv1d"):
                add_bf16(f"model.layers.0.attention.{cn}.weight", (qkv_dim, 4, 1))
            add_bf16("model.layers.0.attention.A_log", (n_heads,), mean=-2.0)
            add_bf16("model.layers.0.attention.dt_bias", (qkv_dim,))
            for mp, r_d, c_d in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
                add_mlx_q(f"model.layers.0.mlp.{mp}", r_d, c_d, 4)

            # Layer 1, 2 KDA MoE minimal -> keep 4 layers fast: reuse 1..2
            for li in (1, 2):
                add_bf16(f"model.layers.{li}.input_layernorm.weight", (h,), mean=1.0)
                add_bf16(f"model.layers.{li}.post_attention_layernorm.weight", (h,), mean=1.0)
                for p in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                    add_mlx_q(f"model.layers.{li}.attention.{p}", qkv_dim, h, 4)
                add_mlx_q(f"model.layers.{li}.attention.o_proj", h, qkv_dim, 4)
                add_bf16(f"model.layers.{li}.attention.b_proj.weight", (n_heads, h))
                add_bf16(f"model.layers.{li}.attention.o_norm.weight", (head_dim,), mean=1.0)
                for cn in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    add_bf16(f"model.layers.{li}.attention.{cn}.weight", (qkv_dim, 1, 4))
                add_bf16(f"model.layers.{li}.attention.A_log", (n_heads,), mean=-2.0)
                add_bf16(f"model.layers.{li}.attention.dt_bias", (qkv_dim,))
                add_mlx_q(f"model.layers.{li}.mlp.gate.gate_proj", n_exp, h, 8)
                tmap[f"model.layers.{li}.mlp.gate.expert_bias"] = ("F32", (n_exp,), struct.pack(f"<{n_exp}f", *[0.0] * n_exp))
                for sp, r_d, c_d in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                    add_mlx_q(f"model.layers.{li}.mlp.shared_experts.{sp}", r_d, c_d, 4)
                    add_switch(f"model.layers.{li}", sp, r_d, c_d)

            # Layer 3 Gated MLA with embed_q + unembed_out
            add_bf16("model.layers.3.input_layernorm.weight", (h,), mean=1.0)
            add_bf16("model.layers.3.post_attention_layernorm.weight", (h,), mean=1.0)
            add_mlx_q("model.layers.3.attention.q_a_proj", q_lora, h, 4)
            add_bf16("model.layers.3.attention.q_a_layernorm.weight", (q_lora,), mean=1.0)
            add_mlx_q("model.layers.3.attention.q_b_proj", n_heads * (qk_nope + qk_rope), q_lora, 4)
            add_mlx_q("model.layers.3.attention.kv_a_proj_with_mqa", kv_lora + qk_rope, h, 4)
            add_bf16("model.layers.3.attention.kv_a_layernorm.weight", (kv_lora,), mean=1.0)
            eq_truth, uo_truth = add_mlx_kv_pair("model.layers.3")
            add_mlx_q("model.layers.3.attention.g_proj", n_heads, h, 4)
            add_mlx_q("model.layers.3.attention.dense", h, n_heads * v_hd, 4)
            add_mlx_q("model.layers.3.mlp.gate.gate_proj", n_exp, h, 8)
            tmap["model.layers.3.mlp.gate.expert_bias"] = ("F32", (n_exp,), struct.pack(f"<{n_exp}f", *[0.0] * n_exp))
            for sp, r_d, c_d in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                add_mlx_q(f"model.layers.3.mlp.shared_experts.{sp}", r_d, c_d, 4)
                add_switch("model.layers.3", sp, r_d, c_d)

            # MLX MTP `mtp.0.*`
            add_bf16("mtp.0.enorm.weight", (h,), mean=1.0)
            add_bf16("mtp.0.hnorm.weight", (h,), mean=1.0)
            add_bf16("mtp.0.eh_proj.weight", (h, 2 * h))
            add_bf16("mtp.0.final_layernorm.weight", (h,), mean=1.0)
            add_bf16("mtp.0.input_layernorm.weight", (h,), mean=1.0)
            add_bf16("mtp.0.post_attention_layernorm.weight", (h,), mean=1.0)
            add_mlx_q("mtp.0.attention.q_a_proj", q_lora, h, 4)
            add_bf16("mtp.0.attention.q_a_layernorm.weight", (q_lora,), mean=1.0)
            add_mlx_q("mtp.0.attention.q_b_proj", n_heads * (qk_nope + qk_rope), q_lora, 4)
            add_mlx_q("mtp.0.attention.kv_a_proj_with_mqa", kv_lora + qk_rope, h, 4)
            add_bf16("mtp.0.attention.kv_a_layernorm.weight", (kv_lora,), mean=1.0)
            add_mlx_kv_pair("mtp.0")
            add_bf16("mtp.0.attention.g_proj.weight", (n_heads, h))
            add_mlx_q("mtp.0.attention.dense", h, n_heads * v_hd, 4)
            add_mlx_q("mtp.0.mlp.gate.gate_proj", n_exp, h, 8)
            tmap["mtp.0.mlp.gate.expert_bias"] = ("F32", (n_exp,), struct.pack(f"<{n_exp}f", *[0.0] * n_exp))
            for sp, r_d, c_d in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                add_mlx_q(f"mtp.0.mlp.shared_experts.{sp}", r_d, c_d, 4)
                add_switch("mtp.0", sp, r_d, c_d)

            _write_safetensors(mlx_dir / "model.safetensors", tmap)
            bb_out = pathlib.Path(tmp) / "mlx_bb.amf"
            mtp_out = pathlib.Path(tmp) / "mlx_mtp.amf"
            rep = convert_ling3_checkpoint(
                mlx_dir,
                bb_out,
                mtp_output=mtp_out,
                mtp_repack_sme2=True,
                mtp_eh_proj_bits=3,
                embed_metadata=True,
            )
            _verify_amf_with_zig(self, bb_out, len(rep.backbone_metas))
            _verify_amf_with_zig(self, mtp_out, len(rep.mtp_metas))

            with AmfReader(bb_out) as rdr:
                self.assertEqual(rdr.by_name["layers.0.linear_attn.b_proj.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["layers.3.self_attn.g_proj.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["layers.3.mlp.gate.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.A_log"].dtype, DType.F32)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.q_conv1d.weight"].shape, (qkv_dim, 4))
                kv_m = rdr.by_name["layers.3.self_attn.kv_b_proj.weight"]
                self.assertEqual(kv_m.shape, (n_heads * (qk_nope + v_hd), kv_lora))
                self.assertEqual(kv_m.dtype, DType.BF16)
                rows_got = rdr.read_rows_f32(kv_m, [0, qk_nope])
                # Row 0 = transpose col 0 of embed_q[0]; row qk_nope = row 0 of unembed_out[0].
                for kl in range(min(16, kv_lora)):
                    self.assertAlmostEqual(rows_got[0][kl], eq_truth[kl][0], delta=0.03)
                    self.assertAlmostEqual(rows_got[1][kl], uo_truth[0][kl], delta=0.03)

            with AmfReader(mtp_out) as mrdr:
                names = {m.name for m in mrdr.entries}
                self.assertIn("layers.0.self_attn.kv_b_proj.weight", names)
                self.assertIn("layers.0.mlp.experts.3.down_proj", names)
                self.assertEqual(mrdr.by_name["layers.0.eh_proj.weight"].dtype, DType.Q3_AFFINE)
                self.assertEqual(mrdr.by_name["layers.0.eh_proj.weight"].layout, TensorLayout.SME2_PANEL64)

            # Verify direct RCO-Q3 conversion on the MLX directory (uses _UnifiedSource decode_mlx_affine_3d_head_f32!).
            q3_mlx = pathlib.Path(tmp) / "mlx_q3.amf"
            gsq_rep = gsq_convert(mlx_dir, q3_mlx, target_bits=3.50, dense_kda_rco_q3=True)
            _verify_amf_with_zig(self, q3_mlx, len(gsq_rep.metas))
            with AmfReader(q3_mlx) as rdr:
                self.assertEqual(rdr.by_name["layers.0.linear_attn.f_proj"].dtype, DType.Q3_AFFINE)
                self.assertEqual(rdr.by_name["layers.3.self_attn.kv_b_proj"].dtype, DType.Q4_AFFINE)

    def test_convert_ling3_gguf_bailingmoe3_and_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gguf_p = pathlib.Path(tmp) / "ling3.gguf"
            rng = random.Random(19)
            h, inter, moe_inter, n_exp, vocab = 64, 128, 64, 4, 96
            n_heads, head_dim, qkv_dim = 2, 32, 64
            q_lora, kv_lora, qk_nope, qk_rope, v_hd = 64, 64, 32, 32, 32

            kv: list[tuple[str, int, object]] = [
                ("general.architecture", 8, "bailingmoe3"),
                ("general.name", 8, "tiny-ling3-gguf"),
                ("bailingmoe3.block_count", 4, 3),  # 2 backbone + 1 MTP
                ("bailingmoe3.nextn_predict_layers", 4, 1),
                ("bailingmoe3.leading_dense_block_count", 4, 1),
                ("bailingmoe3.embedding_length", 4, h),
                ("bailingmoe3.feed_forward_length", 4, inter),
                ("bailingmoe3.expert_feed_forward_length", 4, moe_inter),
                ("bailingmoe3.expert_shared_feed_forward_length", 4, moe_inter),
                ("bailingmoe3.expert_count", 4, n_exp),
                ("bailingmoe3.expert_shared_count", 4, 1),
                ("bailingmoe3.expert_used_count", 4, 2),
                ("bailingmoe3.expert_group_count", 4, 2),
                ("bailingmoe3.expert_group_used_count", 4, 1),
                ("bailingmoe3.expert_weights_scale", 6, 2.5),
                ("bailingmoe3.expert_weights_norm", 7, True),
                ("bailingmoe3.attention.head_count", 4, n_heads),
                ("bailingmoe3.attention.head_count_kv", 9, (4, [0, n_heads, n_heads])),
                ("bailingmoe3.attention.q_lora_rank", 4, q_lora),
                ("bailingmoe3.attention.kv_lora_rank", 4, kv_lora),
                ("bailingmoe3.attention.key_length_mla", 4, qk_nope + qk_rope),
                ("bailingmoe3.attention.value_length_mla", 4, v_hd),
                ("bailingmoe3.kda.head_dim", 4, head_dim),
                ("bailingmoe3.kda.safe_gate", 7, True),
                ("bailingmoe3.kda.gate_lower_bound", 6, -5.0),
                ("bailingmoe3.ssm.conv_kernel", 4, 4),
                ("bailingmoe3.rope.dimension_count", 4, qk_rope),
                ("bailingmoe3.rope.freq_base", 6, 6000000.0),
                ("bailingmoe3.vocab_size", 4, vocab),
            ]

            tensors: list[tuple[str, int, tuple[int, ...], bytes]] = []

            def add_bf16(nm: str, shp: tuple[int, ...], mean: float = 0.0) -> list[float]:
                n_el = 1
                for d in shp:
                    n_el *= d
                vals = [mean + x for x in _det(rng, n_el)]
                tensors.append((nm, 30, shp, f32_to_bf16_bytes(vals)))
                return vals

            def add_f32(nm: str, shp: tuple[int, ...], mean: float = 0.0) -> None:
                n_el = 1
                for d in shp:
                    n_el *= d
                vals = [mean + x for x in _det(rng, n_el)]
                tensors.append((nm, 0, shp, struct.pack(f"<{n_el}f", *vals)))

            add_bf16("token_embd.weight", (vocab, h))
            add_bf16("output_norm.weight", (h,), mean=1.0)
            add_bf16("output.weight", (vocab, h))

            # blk.0 KDA + dense MLP
            add_bf16("blk.0.attn_norm.weight", (h,), mean=1.0)
            add_bf16("blk.0.ffn_norm.weight", (h,), mean=1.0)
            for nm in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_f_a.weight", "ssm_g_a.weight"):
                add_bf16(f"blk.0.{nm}", (qkv_dim, h))
            add_bf16("blk.0.attn_output.weight", (h, qkv_dim))
            add_bf16("blk.0.ssm_beta.weight", (n_heads, h))
            add_bf16("blk.0.ssm_a", (n_heads, 1), mean=-2.0)
            add_bf16("blk.0.ssm_dt.bias", (qkv_dim,))
            for cn in ("ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"):
                add_bf16(f"blk.0.{cn}", (qkv_dim, 1, 4))
            add_bf16("blk.0.ssm_norm.weight", (head_dim,), mean=1.0)
            add_bf16("blk.0.ffn_gate.weight", (inter, h))
            add_bf16("blk.0.ffn_up.weight", (inter, h))
            add_bf16("blk.0.ffn_down.weight", (h, inter))

            # blk.1 & blk.2 MLA + MoE (blk.2 = MTP!)
            kb_truth: list[float] = []
            vb_truth: list[float] = []
            for bi in (1, 2):
                if bi == 2:
                    add_bf16("blk.2.nextn.enorm.weight", (h,), mean=1.0)
                    add_bf16("blk.2.nextn.hnorm.weight", (h,), mean=1.0)
                    add_bf16("blk.2.nextn.eh_proj.weight", (h, 2 * h))
                    add_bf16("blk.2.layer_output_norm.weight", (h,), mean=1.0)
                add_bf16(f"blk.{bi}.attn_norm.weight", (h,), mean=1.0)
                add_bf16(f"blk.{bi}.ffn_norm.weight", (h,), mean=1.0)
                add_bf16(f"blk.{bi}.attn_q_a.weight", (q_lora, h))
                add_bf16(f"blk.{bi}.attn_q_a_norm.weight", (q_lora,), mean=1.0)
                add_bf16(f"blk.{bi}.attn_q_b.weight", (n_heads * (qk_nope + qk_rope), q_lora))
                add_bf16(f"blk.{bi}.attn_kv_a_mqa.weight", (kv_lora + qk_rope, h))
                add_bf16(f"blk.{bi}.attn_kv_a_norm.weight", (kv_lora,), mean=1.0)
                kb = add_bf16(f"blk.{bi}.attn_k_b.weight", (n_heads, kv_lora, qk_nope))
                vb = add_bf16(f"blk.{bi}.attn_v_b.weight", (n_heads, v_hd, kv_lora))
                if bi == 1:
                    kb_truth, vb_truth = kb, vb
                add_bf16(f"blk.{bi}.attn_gate.weight", (n_heads, h))
                add_bf16(f"blk.{bi}.attn_output.weight", (h, n_heads * v_hd))
                add_bf16(f"blk.{bi}.ffn_gate_inp.weight", (n_exp, h))
                add_f32(f"blk.{bi}.exp_probs_b.bias", (n_exp,))
                add_bf16(f"blk.{bi}.ffn_gate_shexp.weight", (moe_inter, h))
                add_bf16(f"blk.{bi}.ffn_up_shexp.weight", (moe_inter, h))
                add_bf16(f"blk.{bi}.ffn_down_shexp.weight", (h, moe_inter))
                add_bf16(f"blk.{bi}.ffn_gate_exps.weight", (n_exp, moe_inter, h))
                add_bf16(f"blk.{bi}.ffn_up_exps.weight", (n_exp, moe_inter, h))
                add_bf16(f"blk.{bi}.ffn_down_exps.weight", (n_exp, h, moe_inter))

            _write_gguf_v3(gguf_p, kv, tensors)
            bb_out = pathlib.Path(tmp) / "gguf_bb.amf"
            mtp_out = pathlib.Path(tmp) / "gguf_mtp.amf"
            rep = convert_ling3_checkpoint(
                gguf_p,
                bb_out,
                mtp_output=mtp_out,
                mtp_repack_sme2=True,
                mtp_eh_proj_bits=3,
                strict_vocab_check=False,
            )
            _verify_amf_with_zig(self, bb_out, len(rep.backbone_metas))
            _verify_amf_with_zig(self, mtp_out, len(rep.mtp_metas))

            with AmfReader(bb_out) as rdr:
                names = {m.name for m in rdr.entries}
                self.assertIn("layers.0.linear_attn.f_proj.weight", names)
                self.assertIn("layers.0.linear_attn.g_proj.weight", names)
                self.assertIn("layers.1.self_attn.g_proj.weight", names)
                self.assertNotIn("layers.2.eh_proj.weight", names)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.A_log"].shape, (n_heads,))
                self.assertEqual(rdr.by_name["layers.0.linear_attn.A_log"].dtype, DType.F32)
                self.assertEqual(rdr.by_name["layers.0.linear_attn.q_conv1d.weight"].shape, (qkv_dim, 4))
                self.assertEqual(rdr.by_name["layers.0.linear_attn.q_conv1d.weight"].dtype, DType.F32)
                kv_m = rdr.by_name["layers.1.self_attn.kv_b_proj.weight"]
                self.assertEqual(kv_m.shape, (n_heads * (qk_nope + v_hd), kv_lora))
                rows_kv = rdr.read_rows_f32(kv_m, [0, qk_nope])
                for kl in range(8):
                    self.assertAlmostEqual(rows_kv[0][kl], kb_truth[kl * qk_nope], delta=0.02)
                    self.assertAlmostEqual(rows_kv[1][kl], vb_truth[kl], delta=0.02)

            with AmfReader(mtp_out) as mrdr:
                names = {m.name for m in mrdr.entries}
                self.assertIn("layers.0.enorm.weight", names)
                self.assertIn("layers.0.hnorm.weight", names)
                self.assertIn("layers.0.final_layernorm.weight", names)
                self.assertIn("layers.0.self_attn.kv_b_proj.weight", names)
                self.assertEqual(mrdr.by_name["layers.0.eh_proj.weight"].dtype, DType.Q3_AFFINE)
                self.assertEqual(mrdr.by_name["layers.0.eh_proj.weight"].layout, TensorLayout.SME2_PANEL64)

    def test_ling3_lora_aliases_and_root_mtp_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            peft_dir = root / "peft"
            peft_dir.mkdir(parents=True, exist_ok=True)
            (peft_dir / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8.0}), encoding="utf-8")
            h = 64
            t_lora: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            for key in (
                "base_model.model.layers.0.linear_attn.f_proj",
                "base_model.model.layers.0.linear_attn.g_proj",
                "base_model.model.layers.7.self_attn.g_proj",
                "base_model.model.layers.7.attention.dense",
                "base_model.model.layers.0.mtp.eh_proj",
            ):
                t_lora[f"{key}.lora_A.weight"] = ("BF16", (4, h), f32_to_bf16_bytes([0.01] * (4 * h)))
                t_lora[f"{key}.lora_B.weight"] = ("BF16", (h, 4), f32_to_bf16_bytes([0.01] * (h * 4)))
            _write_safetensors(peft_dir / "adapter_model.safetensors", t_lora)
            lora_out = root / "ling3.amlora"
            pack_lora_from_peft(peft_dir, lora_out, layout="row_major")
            _verify_amf_with_zig(self, lora_out)
            with AmfReader(lora_out) as lrdr:
                lnames = {m.name for m in lrdr.entries}
                self.assertIn("layers.0.linear_attn.f_proj.lora_a", lnames)
                self.assertIn("layers.0.linear_attn.g_proj.lora_a", lnames)
                self.assertIn("layers.7.self_attn.g_proj.lora_a", lnames)
                self.assertIn("layers.7.self_attn.o_proj.lora_a", lnames)
                self.assertIn("layers.0.self_attn.eh_proj.lora_a", lnames)

            # Root-alias MTP preservation test + L2 <= 16 MiB auto q3_affine SME2 test.
            mtp_src = root / "mtp_root"
            cfg = _ling3_base_config(num_layers=2, layer_group_size=2, first_k_dense=0)
            rng = random.Random(23)
            h2 = 128
            _write_safetensors(
                mtp_src / "model.safetensors",
                {
                    "enorm.weight": ("BF16", (h2,), f32_to_bf16_bytes([1.0] * h2)),
                    "hnorm.weight": ("BF16", (h2,), f32_to_bf16_bytes([1.0] * h2)),
                    "eh_proj.weight": ("BF16", (h2, 2 * h2), f32_to_bf16_bytes(_det(rng, h2 * 2 * h2))),
                    "norm.weight": ("BF16", (h2,), f32_to_bf16_bytes([1.0] * h2)),
                    "layers.0.input_layernorm.weight": ("BF16", (h2,), f32_to_bf16_bytes([1.0] * h2)),
                    "layers.0.post_attention_layernorm.weight": ("BF16", (h2,), f32_to_bf16_bytes([1.0] * h2)),
                    "layers.0.attention.q_a_proj.weight": ("BF16", (64, h2), f32_to_bf16_bytes(_det(rng, 64 * h2))),
                    "layers.0.attention.q_a_layernorm.weight": ("BF16", (64,), f32_to_bf16_bytes([1.0] * 64)),
                    "layers.0.attention.q_b_proj.weight": ("BF16", (384, 64), f32_to_bf16_bytes(_det(rng, 384 * 64))),
                    "layers.0.attention.kv_a_proj_with_mqa.weight": ("BF16", (96, h2), f32_to_bf16_bytes(_det(rng, 96 * h2))),
                    "layers.0.attention.kv_a_layernorm.weight": ("BF16", (64,), f32_to_bf16_bytes([1.0] * 64)),
                    "layers.0.attention.kv_b_proj.weight": ("BF16", (384, 64), f32_to_bf16_bytes(_det(rng, 384 * 64))),
                    "layers.0.attention.g_proj.weight": ("BF16", (4, h2), f32_to_bf16_bytes(_det(rng, 4 * h2))),
                    "layers.0.attention.dense.weight": ("BF16", (h2, 128), f32_to_bf16_bytes(_det(rng, h2 * 128))),
                },
            )
            (mtp_src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            root_mtp_amf = root / "root_mtp.amf"
            metas_r = export_ling3_mtp(mtp_src, root_mtp_amf, repack_sme2=True)
            _verify_amf_with_zig(self, root_mtp_amf, len(metas_r))
            with AmfReader(root_mtp_amf) as rrdr:
                rnames = {m.name for m in rrdr.entries}
                self.assertIn("enorm.weight", rnames)
                self.assertIn("hnorm.weight", rnames)
                self.assertIn("eh_proj.weight", rnames)
                self.assertIn("norm.weight", rnames)
                self.assertEqual(rrdr.by_name["eh_proj.weight"].layout, TensorLayout.SME2_PANEL64)


if __name__ == "__main__":
    unittest.main()
