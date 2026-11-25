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


if __name__ == "__main__":
    unittest.main()
