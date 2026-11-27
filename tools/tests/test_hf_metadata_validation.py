"""Tests for offline HF metadata-only dry-run validators (`inspect-metadata`) and real key aliases."""

from __future__ import annotations

import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

from aminfer_tools.amf import _classify_tensor, _strip_model_prefix, is_vision_tensor, TensorKind
from aminfer_tools.cli import main as cli_main
from aminfer_tools.deepseek41 import (
    is_dspark_tensor,
    is_engram_tensor,
    validate_hf_metadata_dir as validate_deepseek41_metadata_dir,
)
from aminfer_tools.qwen4_exp import (
    infer_operator_axes_and_loop_schedule,
    normalize_ple_name,
    validate_hf_metadata_dir as validate_qwen4_metadata_dir,
)


class HfMetadataValidatorTests(unittest.TestCase):
    def test_real_hf_key_aliases_and_vision_stripping(self) -> None:
        self.assertEqual(
            _strip_model_prefix("model.language_model.layers.0.linear_attn.A_log"),
            "layers.0.linear_attn.A_log",
        )
        self.assertTrue(is_vision_tensor("model.visual.blocks.0.attn.qkv.weight"))
        self.assertTrue(is_vision_tensor("vision.blocks.0.attn.wqkv.weight"))
        self.assertTrue(is_vision_tensor("aligner.w1.weight"))
        self.assertTrue(is_vision_tensor("image_start"))
        self.assertTrue(is_vision_tensor("image_end"))
        self.assertTrue(is_vision_tensor("image_newline"))
        self.assertFalse(is_vision_tensor("embed.weight"))

        self.assertEqual(_classify_tensor("head.weight", set(), False)[0], TensorKind.LM_HEAD)
        self.assertEqual(_classify_tensor("layers.0.ffn.gate.weight", set(), False)[0], TensorKind.ROUTER)
        self.assertEqual(_classify_tensor("layers.0.ffn.gate.bias_vl", set(), False)[0], TensorKind.ROUTER)
        self.assertTrue(is_dspark_tensor("mtp.0.confidence_head.proj.weight"))
        self.assertTrue(is_engram_tensor("layers.14.engram.embed.weight"))
        self.assertEqual(
            normalize_ple_name("model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_7.weight"),
            "ple.layers.2.shard.7.weight",
        )

    def test_qwen4_exp_metadata_dry_run_nested_text_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            cfg = {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 4,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "hidden_size": 2560,
                    "vocab_size": 248320,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "partial_rotary_factor": 0.25,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "indexer_n_heads": 4,
                    "num_experts": 512,
                    "num_experts_per_tok": 10,
                    "moe_intermediate_size": 640,
                    "shared_expert_intermediate_size": 640,
                    "hc_count": 4,
                    "ple_layer_ids": [2],
                    "split_ngram_parts": 4,
                    "mtp_num_hidden_layers": 1,
                },
            }
            wm = {
                "model.language_model.embed_tokens.weight": "model-00001.safetensors",
                "lm_head.weight": "model-00001.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "model-00001.safetensors",
                "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": "model-00001.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "model-00001.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "model-00001.safetensors",
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": "model-00002.safetensors",
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_1.weight": "model-00002.safetensors",
                "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight": "model-00001.safetensors",
                "model.visual.blocks.0.attn.qkv.weight": "model-00001.safetensors",
                "mtp.fc_embedding.weight": "model-00001.safetensors",
            }
            (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 10_000_000_000}, "weight_map": wm}),
                encoding="utf-8",
            )

            rep = validate_qwen4_metadata_dir(d)
            self.assertEqual(rep.gdn_layers, 3)
            self.assertEqual(rep.qsa_layers, 1)
            self.assertEqual(rep.operator_axes.attention_kind, "hybrid_gdn_qsa")
            self.assertEqual(rep.operator_axes.residual_kind, "gated_4branch")
            self.assertEqual(rep.operator_axes.memory_lookup_kind, "ple_ngram")
            self.assertEqual(rep.vision_tensor_count, 1)
            self.assertEqual(rep.ple_tensor_count, 2)
            self.assertEqual(rep.ple_shards_detected, 2)
            self.assertEqual(rep.mtp_tensor_count, 1)
            self.assertEqual(rep.stacked_expert_tensor_count, 1)
            self.assertEqual(len(rep.long_backbone_names), 1)
            self.assertEqual(rep.kv_bytes_per_token_fp16, 1 * 2 * 2 * 256 * 2)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["inspect-metadata", "--input", str(d), "--json"])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertEqual(parsed["arch"], "qwen4_exp")

    def test_deepseek41_metadata_dry_run_hf_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            cfg = {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "vocab_size": 129280,
                    "hidden_size": 5120,
                    "moe_intermediate_size": 2304,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 1,
                    "head_dim": 512,
                    "qk_rope_head_dim": 64,
                    "q_lora_rank": 1280,
                    "o_lora_rank": 1024,
                    "o_groups": 8,
                    "n_routed_experts": 384,
                    "n_shared_experts": 1,
                    "num_experts_per_tok": 6,
                    "compress_ratios": [0, 2, 1, 1, 0],
                    "kv_source_layer_ids": [0, 2],
                    "hc_mult": 4,
                    "engram_layer_ids": [1],
                    "engram_num_embeddings": [1_000_000],
                    "num_nextn_predict_layers": 1,
                    "dspark_block_size": 5,
                    "dspark_n_routed_experts": 128,
                },
            }
            wm = {
                "embed.weight": "s0.safetensors",
                "head.weight": "s0.safetensors",
                "norm.weight": "s0.safetensors",
                "layers.0.attn.wkv.weight": "s0.safetensors",
                "layers.0.attn.indexer.wk.weight": "s0.safetensors",
                "layers.0.hc_attn_base": "s0.safetensors",
                "layers.0.ffn.experts.0.w1.weight": "s0.safetensors",
                "layers.0.ffn.experts.0.w1.scale": "s0.safetensors",
                "layers.0.ffn.shared_experts.w1.weight": "s0.safetensors",
                "layers.1.engram.embed.weight": "s1.safetensors",
                "mtp.0.confidence_head.proj.weight": "s2.safetensors",
                "vision.blocks.0.attn.wqkv.weight": "s3.safetensors",
                "aligner.w1.weight": "s3.safetensors",
                "image_start": "s3.safetensors",
            }
            (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 50_000_000_000}, "weight_map": wm}),
                encoding="utf-8",
            )

            rep = validate_deepseek41_metadata_dir(d)
            self.assertEqual(rep.num_encoder_layers, 2)
            self.assertEqual(rep.num_decoder_layers, 2)
            self.assertEqual(rep.vision_tensor_count, 3)
            self.assertEqual(rep.engram_tensor_count, 1)
            self.assertEqual(rep.dspark_mtp_tensor_count, 1)
            self.assertEqual(rep.routed_expert_tensor_count, 2)
            self.assertEqual(rep.routed_expert_ids_detected, 1)
            self.assertEqual(rep.kv_bytes_per_token_bf16, 2 * (512 + 64) * 2)
            self.assertEqual(rep.kv_bytes_per_token_fp4, 2 * (256 + 32 + 26))
            self.assertEqual(len(rep.long_backbone_names), 0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["inspect-metadata", "--input", str(d)])
            self.assertEqual(rc, 0)
            self.assertIn("deepseek41 metadata OK", buf.getvalue())

    def test_opportunistic_live_hf_metadata_directories(self) -> None:
        meta_env = os.environ.get("AMINFER_METADATA_DIR")
        models_env = os.environ.get("AMINFER_MODELS_DIR")
        meta_root = (
            pathlib.Path(meta_env)
            if meta_env
            else (pathlib.Path(models_env) / "metadata" if models_env else None)
        )
        if meta_root is None or not meta_root.is_dir():
            self.skipTest("AMINFER_METADATA_DIR / AMINFER_MODELS_DIR metadata dir not present")

        qwen_dir = meta_root / "Qwen__Qwen3.8-Flash-Next"
        if (qwen_dir / "config.json").is_file() and (qwen_dir / "model.safetensors.index.json").is_file():
            q_rep = validate_qwen4_metadata_dir(qwen_dir)
            self.assertEqual(q_rep.num_hidden_layers, 48)
            self.assertEqual(q_rep.gdn_layers, 36)
            self.assertEqual(q_rep.qsa_layers, 12)
            self.assertEqual(q_rep.num_experts, 512)
            self.assertEqual(q_rep.ple_shards_detected, 128)
            self.assertEqual(q_rep.operator_axes.attention_kind, "hybrid_gdn_qsa")
            self.assertEqual(q_rep.operator_axes.residual_kind, "gated_4branch")
            self.assertEqual(q_rep.operator_axes.memory_lookup_kind, "ple_ngram")
            self.assertEqual(q_rep.vision_tensor_count, 333)

        ds_dir = meta_root / "deepseek-ai__DeepSeek-V4.1-Flash-1017"
        if (ds_dir / "config.json").is_file() and (ds_dir / "model.safetensors.index.json").is_file():
            d_rep = validate_deepseek41_metadata_dir(ds_dir)
            self.assertEqual(d_rep.num_hidden_layers, 40)
            self.assertEqual(d_rep.num_encoder_layers, 20)
            self.assertEqual(d_rep.num_decoder_layers, 20)
            self.assertEqual(d_rep.num_routed_experts, 384)
            self.assertEqual(d_rep.routed_expert_ids_detected, 384)
            self.assertEqual(d_rep.dspark_layers, 3)
            self.assertEqual(d_rep.engram_layer_ids, (1, 14))
            self.assertEqual(len(d_rep.long_backbone_names), 0)
            self.assertFalse(d_rep.tier_admission["24gb-m4"]["fits_all_wired_q3"])


def _write_mimo_metadata_dir(
    root: pathlib.Path,
    *,
    num_layers: int = 4,
    tp_degree: int = 4,
    global_kv_heads: int = 4,
    swa_kv_heads: int = 8,
    include_mtp: bool = True,
    include_dflash: bool = True,
) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    hidden = 4096
    q_heads = 64
    hd = 192
    vhd = 128
    moe_inter = 2048
    n_experts = 256 if tp_degree == 4 else 384
    layer_pattern = [0 if (i % 5 == 0) else 1 for i in range(num_layers)]
    layer_types = ["full_attention" if x == 0 else "sliding_attention" for x in layer_pattern]
    cfg = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": 16384,
        "moe_intermediate_size": moe_inter,
        "vocab_size": 152576,
        "num_hidden_layers": num_layers,
        "hybrid_layer_pattern": layer_pattern,
        "layer_types": layer_types,
        "num_attention_heads": q_heads,
        "num_key_value_heads": global_kv_heads,
        "swa_num_attention_heads": q_heads,
        "swa_num_key_value_heads": swa_kv_heads,
        "head_dim": hd,
        "v_head_dim": vhd,
        "swa_head_dim": hd,
        "swa_v_head_dim": vhd,
        "sliding_window_size": 128,
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "partial_rotary_factor": 0.334,
        "attention_value_scale": 0.707 if tp_degree == 4 else 0.612,
        "add_swa_attention_sink_bias": True,
        "n_routed_experts": n_experts,
        "num_experts_per_tok": 8,
        "moe_layer_freq": [0 if i == 0 else 1 for i in range(num_layers)],
        "num_nextn_predict_layers": 3 if include_mtp else 0,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.446,
    }
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tensors: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [152576, hidden]},
        "model.norm.weight": {"dtype": "F32", "shape": [hidden]},
        "lm_head.weight": {"dtype": "BF16", "shape": [152576, hidden]},
        "audio_encoder.conv_layers.0.very_long_unused_audio_submodule_weight": {"dtype": "BF16", "shape": [8, 8]},
    }
    for l_idx, ltype in enumerate(layer_types):
        kvh = global_kv_heads if ltype == "full_attention" else swa_kv_heads
        qkv_rows = q_heads * hd + kvh * hd + kvh * vhd
        sb_rows = (
            ((q_heads // tp_degree) * hd + 127) // 128
            + ((kvh // tp_degree) * hd + 127) // 128
            + ((kvh // tp_degree) * vhd + 127) // 128
        )
        sb_total = sb_rows * tp_degree
        tensors[f"model.layers.{l_idx}.input_layernorm.weight"] = {"dtype": "F32", "shape": [hidden]}
        tensors[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = {"dtype": "F32", "shape": [hidden]}
        tensors[f"model.layers.{l_idx}.self_attn.qkv_proj.weight"] = {"dtype": "F8_E4M3", "shape": [qkv_rows, hidden]}
        tensors[f"model.layers.{l_idx}.self_attn.qkv_proj.weight_scale_inv"] = {"dtype": "F32", "shape": [sb_total, hidden // 128]}
        tensors[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = {"dtype": "F8_E4M3", "shape": [hidden, q_heads * vhd]}
        if ltype == "sliding_attention":
            tensors[f"model.layers.{l_idx}.self_attn.attention_sink_bias"] = {"dtype": "F32", "shape": [q_heads]}
        if l_idx == 0:
            for p in ("gate_proj", "up_proj"):
                tensors[f"model.layers.0.mlp.{p}.weight"] = {"dtype": "F8_E4M3", "shape": [16384, hidden]}
            tensors["model.layers.0.mlp.down_proj.weight"] = {"dtype": "F8_E4M3", "shape": [hidden, 16384]}
        else:
            tensors[f"model.layers.{l_idx}.mlp.gate.weight"] = {"dtype": "F32", "shape": [n_experts, hidden]}
            tensors[f"model.layers.{l_idx}.mlp.gate.e_score_correction_bias"] = {"dtype": "F32", "shape": [n_experts]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.gate_proj.weight"] = {"dtype": "U8", "shape": [moe_inter, hidden // 2]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.gate_proj.weight_scale"] = {"dtype": "U8", "shape": [moe_inter, hidden // 32]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.up_proj.weight"] = {"dtype": "U8", "shape": [moe_inter, hidden // 2]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.up_proj.weight_scale"] = {"dtype": "U8", "shape": [moe_inter, hidden // 32]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.down_proj.weight"] = {"dtype": "U8", "shape": [hidden, moe_inter // 2]}
            tensors[f"model.layers.{l_idx}.mlp.experts.0.down_proj.weight_scale"] = {"dtype": "U8", "shape": [hidden, moe_inter // 32]}

    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2), encoding="utf-8")

    mtp_tensors: dict[str, dict[str, object]] = {}
    if include_mtp:
        swa_qkv = q_heads * hd + swa_kv_heads * hd + swa_kv_heads * vhd
        mtp_sb = ((q_heads // tp_degree) * 2 + (swa_kv_heads // tp_degree) * 2) * tp_degree
        for k in range(3):
            mtp_tensors[f"mtp.layers.{k}.enorm.weight"] = {"dtype": "F32", "shape": [hidden]}
            mtp_tensors[f"mtp.layers.{k}.hnorm.weight"] = {"dtype": "F32", "shape": [hidden]}
            mtp_tensors[f"mtp.layers.{k}.eh_proj.weight"] = {"dtype": "BF16", "shape": [hidden, 2 * hidden]}
            mtp_tensors[f"mtp.layers.{k}.input_layernorm.weight"] = {"dtype": "F32", "shape": [hidden]}
            mtp_tensors[f"mtp.layers.{k}.pre_mlp_layernorm.weight"] = {"dtype": "F32", "shape": [hidden]}
            mtp_tensors[f"mtp.layers.{k}.final_layernorm.weight"] = {"dtype": "F32", "shape": [hidden]}
            mtp_tensors[f"mtp.layers.{k}.self_attn.qkv_proj.weight"] = {"dtype": "F8_E4M3", "shape": [swa_qkv, hidden]}
            mtp_tensors[f"mtp.layers.{k}.self_attn.qkv_proj.weight_scale_inv"] = {"dtype": "F32", "shape": [mtp_sb, hidden // 128]}
            mtp_tensors[f"mtp.layers.{k}.self_attn.o_proj.weight"] = {"dtype": "F8_E4M3", "shape": [hidden, q_heads * vhd]}
            mtp_tensors[f"mtp.layers.{k}.self_attn.attention_sink_bias"] = {"dtype": "F32", "shape": [q_heads]}
            mtp_tensors[f"mtp.layers.{k}.mlp.gate_proj.weight"] = {"dtype": "F8_E4M3", "shape": [16384, hidden]}
            mtp_tensors[f"mtp.layers.{k}.mlp.up_proj.weight"] = {"dtype": "F8_E4M3", "shape": [16384, hidden]}
            mtp_tensors[f"mtp.layers.{k}.mlp.down_proj.weight"] = {"dtype": "F8_E4M3", "shape": [hidden, 16384]}

    dflash_tensors: dict[str, dict[str, object]] = {}
    if include_dflash:
        (root / "dflash").mkdir(exist_ok=True)
        (root / "dflash" / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "dflash",
                    "num_hidden_layers": 5,
                    "hidden_size": hidden,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "block_size": 8,
                    "mask_token_id": 151675,
                    "target_layer_ids": [0, 1, 2, 3, 4],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        dflash_tensors = {
            "fc.weight": {"dtype": "BF16", "shape": [hidden, 5 * hidden]},
            "hidden_norm.weight": {"dtype": "F32", "shape": [hidden]},
            "norm.weight": {"dtype": "F32", "shape": [hidden]},
            "layers.0.input_layernorm.weight": {"dtype": "F32", "shape": [hidden]},
            "layers.0.post_attention_layernorm.weight": {"dtype": "F32", "shape": [hidden]},
            "layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8192, hidden]},
            "layers.0.self_attn.k_proj.weight": {"dtype": "BF16", "shape": [1024, hidden]},
            "layers.0.self_attn.v_proj.weight": {"dtype": "BF16", "shape": [1024, hidden]},
            "layers.0.self_attn.o_proj.weight": {"dtype": "BF16", "shape": [hidden, 8192]},
            "layers.0.self_attn.q_norm.weight": {"dtype": "F32", "shape": [128]},
            "layers.0.self_attn.k_norm.weight": {"dtype": "F32", "shape": [128]},
            "layers.0.self_attn.attention_sink_bias": {"dtype": "F32", "shape": [64]},
        }

    shards_summary: dict[str, object] = {"model-00001-of-00001.safetensors": {"tensors": tensors}}
    if mtp_tensors:
        shards_summary["model_mtp.safetensors"] = {"tensors": mtp_tensors}
    if dflash_tensors:
        shards_summary["dflash/dflash_draft_model.safetensors"] = {"tensors": dflash_tensors}
    (root / "safetensors_headers_summary.json").write_text(
        json.dumps({"shards": shards_summary}, indent=2),
        encoding="utf-8",
    )
    return root


def _write_qwen35_metadata_dir(root: pathlib.Path, *, mlx_4bit: bool = True) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    layer_types = ["linear_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(4)]
    cfg = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 248320,
            "num_hidden_layers": 4,
            "layer_types": layer_types,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
        "vision_config": {"depth": 27, "hidden_size": 1152, "intermediate_size": 4304},
    }
    if mlx_4bit:
        cfg["quantization"] = {"bits": 4, "group_size": 64}
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tensors: dict[str, dict[str, object]] = {
        "model.language_model.embed_tokens.weight": {"dtype": "U32" if mlx_4bit else "BF16", "shape": [248320, 512 if mlx_4bit else 4096]},
        "model.language_model.norm.weight": {"dtype": "F32", "shape": [4096]},
        "model.language_model.lm_head.weight": {"dtype": "U32" if mlx_4bit else "BF16", "shape": [248320, 512 if mlx_4bit else 4096]},
        "model.visual.patch_embed.proj.weight": {"dtype": "BF16", "shape": [1152, 3, 2, 16, 16]},
        "model.visual.blocks.0.attn.qkv.weight": {"dtype": "BF16", "shape": [3456, 1152]},
    }
    for l_idx, ltype in enumerate(layer_types):
        tensors[f"model.language_model.layers.{l_idx}.input_layernorm.weight"] = {"dtype": "F32", "shape": [4096]}
        tensors[f"model.language_model.layers.{l_idx}.post_attention_layernorm.weight"] = {"dtype": "F32", "shape": [4096]}
        tensors[f"model.language_model.layers.{l_idx}.mlp.gate_proj.weight"] = {"dtype": "BF16", "shape": [12288, 4096]}
        tensors[f"model.language_model.layers.{l_idx}.mlp.up_proj.weight"] = {"dtype": "BF16", "shape": [12288, 4096]}
        tensors[f"model.language_model.layers.{l_idx}.mlp.down_proj.weight"] = {"dtype": "BF16", "shape": [4096, 12288]}
        if ltype == "linear_attention":
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.in_proj_qkv.weight"] = {"dtype": "BF16", "shape": [8192, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.in_proj_z.weight"] = {"dtype": "BF16", "shape": [4096, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.out_proj.weight"] = {"dtype": "BF16", "shape": [4096, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.in_proj_a.weight"] = {"dtype": "U32" if mlx_4bit else "BF16", "shape": [32, 512 if mlx_4bit else 4096]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.in_proj_b.weight"] = {"dtype": "U32" if mlx_4bit else "BF16", "shape": [32, 512 if mlx_4bit else 4096]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.conv1d.weight"] = {"dtype": "BF16", "shape": [8192, 1, 4]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.A_log"] = {"dtype": "F32", "shape": [32]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.dt_bias"] = {"dtype": "F32", "shape": [32]}
            tensors[f"model.language_model.layers.{l_idx}.linear_attn.norm.weight"] = {"dtype": "F32", "shape": [128]}
        else:
            tensors[f"model.language_model.layers.{l_idx}.self_attn.q_proj.weight"] = {"dtype": "BF16", "shape": [8192, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.self_attn.k_proj.weight"] = {"dtype": "BF16", "shape": [1024, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.self_attn.v_proj.weight"] = {"dtype": "BF16", "shape": [1024, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.self_attn.o_proj.weight"] = {"dtype": "BF16", "shape": [4096, 4096]}
            tensors[f"model.language_model.layers.{l_idx}.self_attn.q_norm.weight"] = {"dtype": "F32", "shape": [256]}
            tensors[f"model.language_model.layers.{l_idx}.self_attn.k_norm.weight"] = {"dtype": "F32", "shape": [256]}

    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2), encoding="utf-8")
    (root / "safetensors_headers_summary.json").write_text(
        json.dumps({"shards": {"model-00001-of-00001.safetensors": {"tensors": tensors}}}, indent=2),
        encoding="utf-8",
    )
    return root


class MimoAndQwen35MetadataValidationTests(unittest.TestCase):
    def test_synthetic_mimo_v2_flash_and_pro_metadata_validation(self) -> None:
        from aminfer_tools.mimo_v2 import validate_mimo_v2_metadata_dir

        with tempfile.TemporaryDirectory() as tmp:
            flash_dir = _write_mimo_metadata_dir(pathlib.Path(tmp) / "flash", num_layers=6, tp_degree=4)
            flash_rep = validate_mimo_v2_metadata_dir(flash_dir)
            self.assertEqual(flash_rep.errors, ())
            self.assertEqual(flash_rep.long_backbone_names, 0)
            self.assertEqual(flash_rep.qkv_tp_degree_inferred, 4)
            self.assertEqual(flash_rep.mtp_layers, 3)
            self.assertEqual(flash_rep.dflash_block_size, 8)
            self.assertGreater(flash_rep.swa_kv_bytes_per_token_fp16, flash_rep.global_kv_bytes_per_token_fp16)

            pro_dir = _write_mimo_metadata_dir(pathlib.Path(tmp) / "pro", num_layers=6, tp_degree=8, global_kv_heads=8, swa_kv_heads=8)
            pro_rep = validate_mimo_v2_metadata_dir(pro_dir)
            self.assertEqual(pro_rep.errors, ())
            self.assertEqual(pro_rep.long_backbone_names, 0)
            self.assertEqual(pro_rep.qkv_tp_degree_inferred, 8)

    def test_synthetic_qwen35_metadata_validation(self) -> None:
        from aminfer_tools.mimo_v2 import validate_qwen35_metadata_dir

        with tempfile.TemporaryDirectory() as tmp:
            q_dir = _write_qwen35_metadata_dir(pathlib.Path(tmp) / "qwen35", mlx_4bit=True)
            rep = validate_qwen35_metadata_dir(q_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.long_backbone_names, 0)
            self.assertEqual(rep.linear_attention_layers, 3)
            self.assertEqual(rep.full_attention_layers, 1)
            self.assertTrue(rep.rmsnorm_unit_offset)
            self.assertTrue(rep.attn_output_gate)
            self.assertAlmostEqual(rep.partial_rotary_factor, 0.25)
            self.assertEqual(rep.quantization_mode, "mlx_4bit_g64")

    def test_opportunistic_live_staged_mimo_v2_and_qwen35_metadata(self) -> None:
        from aminfer_tools.mimo_v2 import validate_mimo_v2_metadata_dir, validate_qwen35_metadata_dir

        meta_env = os.environ.get("AMINFER_METADATA_DIR")
        if not meta_env:
            return
        meta_root = pathlib.Path(meta_env)

        flash_dir = meta_root / "XiaomiMiMo__MiMo-V2.6-Flash-RL"
        if flash_dir.is_dir():
            rep = validate_mimo_v2_metadata_dir(flash_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.num_hidden_layers, 48)
            self.assertEqual(rep.swa_layers, 39)
            self.assertEqual(rep.global_layers, 9)
            self.assertEqual(rep.qkv_tp_degree_inferred, 4)
            self.assertEqual(rep.long_backbone_names, 0)
            self.assertEqual(rep.mtp_layers, 3)

        pro_dir = meta_root / "XiaomiMiMo__MiMo-V2.6-Pro-RL"
        if pro_dir.is_dir():
            rep = validate_mimo_v2_metadata_dir(pro_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.num_hidden_layers, 70)
            self.assertEqual(rep.swa_layers, 60)
            self.assertEqual(rep.global_layers, 10)
            self.assertEqual(rep.qkv_tp_degree_inferred, 8)
            self.assertEqual(rep.long_backbone_names, 0)

        for dname in ("XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B", "prithivMLmods__MiMo-V2.6-Distill-Qwen-9B-MLX-4bit"):
            qdir = meta_root / dname
            if qdir.is_dir():
                rep = validate_qwen35_metadata_dir(qdir)
                self.assertEqual(rep.errors, ())
                self.assertEqual(rep.num_hidden_layers, 32)
                self.assertEqual(rep.linear_attention_layers, 24)
                self.assertEqual(rep.full_attention_layers, 8)
                self.assertTrue(rep.rmsnorm_unit_offset)
                self.assertTrue(rep.attn_output_gate)
                self.assertEqual(rep.long_backbone_names, 0)

    def test_synthetic_glm5_metadata_validation(self) -> None:
        from aminfer_tools.glm5 import (
            validate_glm5_dflash2_metadata_dir,
            validate_glm5_next_metadata_dir,
            validate_glm_moe_dsa_metadata_dir,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dsa_dir = root / "glm_dsa"
            dsa_dir.mkdir()
            (dsa_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm_moe_dsa",
                        "architectures": ["GlmMoeDsaForCausalLM"],
                        "hidden_size": 6144,
                        "intermediate_size": 12288,
                        "moe_intermediate_size": 2048,
                        "num_hidden_layers": 78,
                        "first_k_dense_replace": 3,
                        "vocab_size": 154880,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 64,
                        "q_lora_rank": 2048,
                        "kv_lora_rank": 512,
                        "qk_nope_head_dim": 192,
                        "qk_rope_head_dim": 64,
                        "v_head_dim": 256,
                        "index_n_heads": 32,
                        "index_head_dim": 128,
                        "index_topk": 2048,
                        "index_topk_freq": 4,
                        "index_skip_topk_offset": 3,
                        "index_share_for_mtp_iteration": True,
                        "n_routed_experts": 256,
                        "n_shared_experts": 1,
                        "num_experts_per_tok": 8,
                        "num_nextn_predict_layers": 1,
                    }
                ),
                encoding="utf-8",
            )
            wm_dsa: dict[str, str] = {
                "model.embed_tokens.weight": "s.safetensors",
                "lm_head.weight": "s.safetensors",
                "model.norm.weight": "s.safetensors",
                "model.layers.78.eh_proj.weight": "s.safetensors",
                "model.layers.78.shared_head.norm.weight": "s.safetensors",
            }
            for li in range(78):
                wm_dsa[f"model.layers.{li}.self_attn.q_a_proj.weight"] = "s.safetensors"
                if li < 3 or (li - 2) % 4 == 0:
                    wm_dsa[f"model.layers.{li}.self_attn.indexer.weights_proj.weight"] = "s.safetensors"
                    wm_dsa[f"model.layers.{li}.self_attn.indexer.wk.weight"] = "s.safetensors"
                if li >= 3:
                    wm_dsa[f"model.layers.{li}.mlp.gate.e_score_correction_bias"] = "s.safetensors"
                    wm_dsa[f"model.layers.{li}.mlp.experts.0.gate_proj.weight"] = "s.safetensors"
            (dsa_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm_dsa}), encoding="utf-8")
            dsa_rep = validate_glm_moe_dsa_metadata_dir(dsa_dir)
            self.assertEqual(dsa_rep.errors, ())
            self.assertEqual(dsa_rep.full_indexer_layers, 21)
            self.assertEqual(dsa_rep.shared_indexer_layers, 57)
            self.assertEqual(dsa_rep.dense_layers, 3)
            self.assertEqual(dsa_rep.moe_layers, 75)
            self.assertEqual(dsa_rep.long_backbone_names, 0)
            self.assertTrue(dsa_rep.tier_admission["512gb-ultra"]["fits_all_wired"])
            self.assertFalse(dsa_rep.tier_admission["128gb-max"]["fits_all_wired"])

            next_dir = root / "glm5_next"
            next_dir.mkdir()
            (next_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm5_next",
                        "architectures": ["Glm5NextForConditionalGeneration"],
                        "text_config": {
                            "model_type": "glm5_next_text",
                            "hidden_size": 4096,
                            "intermediate_size": 12288,
                            "moe_intermediate_size": 2048,
                            "num_hidden_layers": 45,
                            "first_k_dense_replace": 3,
                            "vocab_size": 154880,
                            "hc_mult": 4,
                            "hc_sinkhorn_iters": 20,
                            "hc_eps": 1e-6,
                            "linear_num_heads": 64,
                            "linear_head_dim": 128,
                            "linear_conv_kernel_dim": 4,
                            "linear_lower_bound": -5.0,
                            "num_attention_heads": 64,
                            "q_lora_rank": 1536,
                            "kv_lora_rank": 512,
                            "qk_nope_head_dim": 256,
                            "qk_rope_head_dim": 0,
                            "v_head_dim": 256,
                            "index_n_heads": 32,
                            "index_head_dim": 128,
                            "index_topk": 2048,
                            "index_kpool": 4,
                            "n_routed_experts": 288,
                            "n_shared_experts": 1,
                            "num_experts_per_tok": 8,
                            "swiglu_limit": 10.0,
                            "num_nextn_predict_layers": 1,
                        },
                        "vision_config": {"depth": 24, "hidden_size": 1024},
                    }
                ),
                encoding="utf-8",
            )
            wm_next: dict[str, str] = {
                "model.language_model.embed_tokens.weight": "s.safetensors",
                "lm_head.weight": "s.safetensors",
                "model.language_model.layers.45.eh_proj.weight": "s.safetensors",
                "model.visual.blocks.23.attn.qkv.weight": "s.safetensors",
            }
            for li in range(45):
                wm_next[f"model.language_model.layers.{li}.hc_attn_fn"] = "s.safetensors"
                if li % 4 == 3:
                    wm_next[f"model.language_model.layers.{li}.self_attn.kv_a_proj_with_mqa.weight"] = "s.safetensors"
                    wm_next[f"model.language_model.layers.{li}.self_attn.indexer.index_kpool_compress_ape"] = "s.safetensors"
                else:
                    wm_next[f"model.language_model.layers.{li}.self_attn.f_a_proj.weight"] = "s.safetensors"
                    wm_next[f"model.language_model.layers.{li}.self_attn.A_log"] = "s.safetensors"
            (next_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm_next}), encoding="utf-8")
            next_rep = validate_glm5_next_metadata_dir(next_dir)
            self.assertEqual(next_rep.errors, ())
            self.assertEqual(next_rep.linear_attention_layers, 34)
            self.assertEqual(next_rep.nope_mla_layers, 11)
            self.assertEqual(next_rep.hc_mult, 4)
            self.assertEqual(next_rep.hc_sinkhorn_iters, 20)
            self.assertEqual(next_rep.vision_layers, 24)
            self.assertEqual(next_rep.long_backbone_names, 0)
            self.assertTrue(next_rep.tier_admission["128gb-max"]["fits_all_wired"])

            df2_dir = root / "dflash2"
            df2_dir.mkdir()
            (df2_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "hidden_size": 4096,
                        "intermediate_size": 12288,
                        "num_hidden_layers": 5,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 154880,
                        "dflash_config": {
                            "block_size": 8,
                            "mask_token_id": 154856,
                            "conv_kernel_size": 2,
                            "conv_group_size": 16,
                            "selector_rank": 256,
                            "selector_top_k": 16,
                            "target_layer_ids": [5, 14, 24, 33, 42],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (df2_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "fc.weight": "m.safetensors",
                            "layers.4.attention_conv.kernel_projection.weight": "m.safetensors",
                            "candidate_selector.hidden_projection.weight": "m.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            df2_rep = validate_glm5_dflash2_metadata_dir(df2_dir)
            self.assertEqual(df2_rep.errors, ())
            self.assertEqual(df2_rep.long_names, 0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["inspect-metadata", "--input", str(next_dir), "--json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(buf.getvalue())["linear_attention_layers"], 34)

    def test_opportunistic_live_staged_glm5_metadata(self) -> None:
        from aminfer_tools.glm5 import (
            validate_glm5_dflash2_metadata_dir,
            validate_glm5_next_metadata_dir,
            validate_glm_moe_dsa_metadata_dir,
        )

        meta_env = os.environ.get("AMINFER_METADATA_DIR")
        if not meta_env:
            return
        meta_root = pathlib.Path(meta_env)

        for dname in ("zai-org__GLM-5.2", "zai-org__GLM-5.3", "RedHatAI__GLM-5.3-NVFP4"):
            dpath = meta_root / dname
            if dpath.is_dir():
                rep = validate_glm_moe_dsa_metadata_dir(dpath)
                self.assertEqual(rep.errors, (), f"{dname}: {rep.errors}")
                self.assertEqual(rep.num_hidden_layers, 78)
                self.assertEqual(rep.full_indexer_layers, 21)
                self.assertEqual(rep.shared_indexer_layers, 57)
                self.assertEqual(rep.long_backbone_names, 0)
                self.assertTrue(rep.index_share_for_mtp_iteration)

        yujie_dir = meta_root / "yujiepan__glm-moe-dsa-tiny-random"
        if yujie_dir.is_dir():
            rep = validate_glm_moe_dsa_metadata_dir(yujie_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.long_backbone_names, 0)

        for dname in (
            "zai-org__GLM-5.3-Flash",
            "nvidia__GLM-5.3-Flash-NVFP4",
            "TensorFold__GLM-5.3-Flash-MLX-4bit-MTP",
        ):
            dpath = meta_root / dname
            if dpath.is_dir():
                rep = validate_glm5_next_metadata_dir(dpath)
                self.assertEqual(rep.errors, (), f"{dname}: {rep.errors}")
                self.assertEqual(rep.num_hidden_layers, 45)
                self.assertEqual(rep.linear_attention_layers, 34)
                self.assertEqual(rep.nope_mla_layers, 11)
                self.assertEqual(rep.hc_mult, 4)
                self.assertEqual(rep.hc_sinkhorn_iters, 20)
                self.assertEqual(rep.vision_layers, 24)
                self.assertEqual(rep.long_backbone_names, 0)

        for dname in (
            "hf-tiny-v2__tiny-random-Glm5NextForConditionalGeneration",
            "trl-internal-testing__tiny-Glm5NextForConditionalGeneration",
        ):
            dpath = meta_root / dname
            if dpath.is_dir():
                rep = validate_glm5_next_metadata_dir(dpath)
                self.assertEqual(rep.errors, (), f"{dname}: {rep.errors}")
                self.assertEqual(rep.long_backbone_names, 0)

        df2_dir = meta_root / "lablup__GLM-5.3-Flash-DFlash2-Agentic"
        if df2_dir.is_dir():
            rep = validate_glm5_dflash2_metadata_dir(df2_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.long_names, 0)
            self.assertEqual(rep.conv_kernel_size, 2)
            self.assertEqual(rep.conv_group_size, 16)
            self.assertEqual(rep.selector_top_k, 16)

    def test_synthetic_ling3_w28_metadata_and_cli(self) -> None:
        from aminfer_tools.cli import main as cli_main
        from aminfer_tools.ling3 import validate_ling3_metadata_dir

        with tempfile.TemporaryDirectory() as tmp:
            meta_dir = pathlib.Path(tmp) / "ling3"
            meta_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "model_type": "bailing_hybrid",
                "architectures": ["BailingMoeV3ForCausalLM"],
                "hidden_size": 4096,
                "intermediate_size": 10240,
                "moe_intermediate_size": 1280,
                "moe_shared_expert_intermediate_size": 1280,
                "num_hidden_layers": 72,
                "layer_group_size": 8,
                "first_k_dense_replace": 4,
                "vocab_size": 157184,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "head_dim": 128,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "qk_head_dim": 192,
                "v_head_dim": 128,
                "num_experts": 512,
                "num_shared_experts": 1,
                "num_experts_per_tok": 8,
                "n_group": 8,
                "topk_group": 4,
                "routed_scaling_factor": 2.5,
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
                "expert_swiglu_limit_list": [0.0] * 65 + [10.0] * 8,
                "share_expert_swiglu_limit_list": [0.0] * 65 + [10.0] * 8,
            }
            (meta_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            wmap = {
                "model.word_embeddings.weight": "s0.safetensors",
                "model.norm.weight": "s1.safetensors",
                "lm_head.weight": "s1.safetensors",
                "model.layers.0.attention.q_proj.weight": "s0.safetensors",
                "model.layers.0.attention.k_proj.weight": "s0.safetensors",
                "model.layers.0.attention.v_proj.weight": "s0.safetensors",
                "model.layers.0.attention.f_proj.weight": "s0.safetensors",
                "model.layers.0.attention.g_proj.weight": "s0.safetensors",
                "model.layers.0.attention.b_proj.weight": "s0.safetensors",
                "model.layers.0.attention.o_proj.weight": "s0.safetensors",
                "model.layers.0.attention.q_conv1d.weight": "s0.safetensors",
                "model.layers.0.attention.k_conv1d.weight": "s0.safetensors",
                "model.layers.0.attention.v_conv1d.weight": "s0.safetensors",
                "model.layers.0.attention.A_log": "s0.safetensors",
                "model.layers.0.attention.dt_bias": "s0.safetensors",
                "model.layers.0.attention.o_norm.weight": "s0.safetensors",
                "model.layers.7.attention.q_a_proj.weight": "s0.safetensors",
                "model.layers.7.attention.q_b_proj.weight": "s0.safetensors",
                "model.layers.7.attention.kv_a_proj_with_mqa.weight": "s0.safetensors",
                "model.layers.7.attention.kv_b_proj.weight": "s0.safetensors",
                "model.layers.7.attention.g_proj.weight": "s0.safetensors",
                "model.layers.7.attention.dense.weight": "s0.safetensors",
                "model.layers.71.mlp.shared_experts.down_proj.weight": "s0.safetensors",
                "model.layers.71.mlp.gate.expert_bias": "s0.safetensors",
                "model.layers.71.mlp.experts.511.down_proj.weight": "s0.safetensors",
                "model.layers.72.enorm.weight": "s1.safetensors",
                "model.layers.72.hnorm.weight": "s1.safetensors",
                "model.layers.72.eh_proj.weight": "s1.safetensors",
                "model.layers.72.attention.dense.weight": "s1.safetensors",
                "model.layers.72.final_layernorm.weight": "s1.safetensors",
            }
            (meta_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}), encoding="utf-8")
            (meta_dir / "safetensors_headers_summary.json").write_text(
                json.dumps(
                    {
                        "sampled_headers": {
                            "s0.safetensors": {
                                "model.layers.0.attention.f_proj.weight": {"dtype": "F8_E4M3", "shape": [8192, 4096]},
                                "model.layers.0.attention.g_proj.weight": {"dtype": "F8_E4M3", "shape": [8192, 4096]},
                                "model.layers.0.attention.b_proj.weight": {"dtype": "BF16", "shape": [64, 4096]},
                                "model.layers.0.attention.o_norm.weight": {"dtype": "BF16", "shape": [128]},
                                "model.layers.7.attention.kv_a_proj_with_mqa.weight": {"dtype": "F8_E4M3", "shape": [576, 4096]},
                                "model.layers.7.attention.g_proj.weight": {"dtype": "BF16", "shape": [64, 4096]},
                                "model.layers.7.attention.dense.weight": {"dtype": "F8_E4M3", "shape": [4096, 8192]},
                                "model.layers.71.mlp.experts.511.down_proj.weight": {"dtype": "F8_E4M3", "shape": [4096, 1280]},
                            },
                            "s1.safetensors": {
                                "model.layers.72.eh_proj.weight": {"dtype": "BF16", "shape": [4096, 8192]},
                                "model.layers.72.final_layernorm.weight": {"dtype": "BF16", "shape": [4096]},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            rep = validate_ling3_metadata_dir(meta_dir)
            self.assertEqual(rep.errors, ())
            self.assertEqual(rep.long_backbone_names, 0)
            self.assertEqual(rep.long_mtp_names, 0)
            self.assertEqual(rep.linear_attention_layers, 63)
            self.assertEqual(rep.mla_layers, 9)
            self.assertEqual(rep.dense_layers, 4)
            self.assertEqual(rep.moe_layers, 68)
            self.assertEqual(rep.mtp_layers, 1)
            self.assertGreater(rep.always_active_plus_mtp_params, 24_000_000_000)
            self.assertLess(rep.always_active_plus_mtp_params, 25_000_000_000)
            self.assertGreater(rep.routed_per_tok_params, 8_000_000_000)
            self.assertLess(rep.rco_q3_always_active_plus_mtp_gib, rep.q4_always_active_plus_mtp_gib - 2.0)
            self.assertGreater(rep.m4_24gb_hot_cache_rco_q3_gib, rep.m4_24gb_hot_cache_q4_gib + 2.0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["inspect-metadata", "--input", str(meta_dir), "--json"])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertEqual(parsed["linear_attention_layers"], 63)
            self.assertEqual(parsed["mla_layers"], 9)

    def test_opportunistic_live_staged_ling3_metadata(self) -> None:
        from aminfer_tools.ling3 import validate_ling3_metadata_dir

        meta_env = os.environ.get("AMINFER_METADATA_DIR")
        if not meta_env:
            return
        meta_root = pathlib.Path(meta_env)

        for dname in (
            "inclusionAI__Ling-3.1-flash-1019",
            "inclusionAI__Ling-3.1-flash",
            "mlx-community__Ling-3.1-flash-1019-oQ4e-mtp",
            "mlx-community__Ling-3.1-flash-1019-oQ2e-mtp",
        ):
            dpath = meta_root / dname
            if dpath.is_dir():
                rep = validate_ling3_metadata_dir(dpath)
                self.assertEqual(rep.errors, (), f"{dname}: {rep.errors}")
                self.assertEqual(rep.num_hidden_layers, 72)
                self.assertEqual(rep.linear_attention_layers, 63)
                self.assertEqual(rep.mla_layers, 9)
                self.assertEqual(rep.dense_layers, 4)
                self.assertEqual(rep.moe_layers, 68)
                self.assertEqual(rep.n_routed_experts, 512)
                self.assertEqual(rep.mtp_layers, 1)
                self.assertTrue(rep.no_kda_lora)
                self.assertTrue(rep.kda_safe_gate)
                self.assertAlmostEqual(rep.kda_lower_bound, -5.0)
                self.assertEqual(rep.long_backbone_names, 0)
                self.assertEqual(rep.long_mtp_names, 0)
                self.assertLessEqual(rep.max_backbone_name_bytes, 47)
                if dname == "inclusionAI__Ling-3.1-flash-1019":
                    self.assertGreater(rep.header_summary_shards, 50)
                    self.assertGreater(rep.routed_expert_ids_detected, 500)

        tiny30 = meta_root / "inclusionAI__Ling-3.0-tiny"
        if tiny30.is_dir():
            rep_t = validate_ling3_metadata_dir(tiny30)
            self.assertEqual(rep_t.errors, ())
            self.assertEqual(rep_t.num_hidden_layers, 24)
            self.assertEqual(rep_t.linear_attention_layers, 18)
            self.assertEqual(rep_t.mla_layers, 6)
            self.assertEqual(rep_t.n_routed_experts, 128)
            self.assertEqual(rep_t.long_backbone_names, 0)


if __name__ == "__main__":
    unittest.main()
