"""Deterministic synthetic Qwen3-MoE and Qwen3-Dense model bundle generator.

Produces matched, 16 KiB-aligned `.amf` v0 weight files, HuggingFace-compatible
`config.json`, and Byte-Level BPE `tokenizer.json` using Python stdlib only.
Strictly offline (ADR-0001).
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, replace

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AMF_MAGIC,
    AMF_VERSION,
    CHECKSUM_CRC32_IEEE,
    ENTRY_STRUCT,
    HEADER_STRUCT,
    NO_EXPERT,
    NO_LAYER,
    DType,
    PackedTensorMeta,
    TensorKind,
    f16_bytes_to_bf16_bytes,
)

_STREAM_CHUNK_BYTES = 16 * 1024 * 1024
_ZERO_PAGE = b"\x00" * APPLE_PAGE_SIZE


def _align_up(length: int, alignment: int) -> int:
    return (length + alignment - 1) & ~(alignment - 1)


def _tensor_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen2 / Qwen3 bijective 256-byte to printable unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@dataclass(frozen=True)
class Qwen3SynthConfig:
    model_id: str
    model_type: str  # "qwen3_moe" or "qwen3"
    quant_bits: int  # 3 or 4
    group_size: int
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    embed_f16: bool = False

    def with_overrides(self, **kwargs: object) -> Qwen3SynthConfig:
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_moe"

    @property
    def quant_dtype(self) -> DType:
        return DType.Q3_AFFINE if self.quant_bits == 3 else DType.Q4_AFFINE

    def special_token_ids(self) -> dict[str, int]:
        if self.vocab_size >= 151669:
            return {
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644,
                "<|im_end|>": 151645,
                "<|think|>": 151667,
                "</think>": 151668,
            }
        base = self.vocab_size - 5
        return {
            "<|endoftext|>": base,
            "<|im_start|>": base + 1,
            "<|im_end|>": base + 2,
            "<|think|>": base + 3,
            "</think>": base + 4,
        }

    def validate(self) -> None:
        if len(self.model_id.encode("utf-8")) == 0 or len(self.model_id.encode("utf-8")) > 63:
            raise ValueError(f"model_id must be 1..63 UTF-8 bytes, got {self.model_id!r}")
        if self.model_type not in ("qwen3_moe", "qwen3"):
            raise ValueError(f"unsupported model_type {self.model_type!r}")
        if self.quant_bits not in (3, 4):
            raise ValueError(f"quant_bits must be 3 or 4, got {self.quant_bits}")
        if self.group_size != 64:
            raise ValueError(f"aminfer v0 supports group_size=64 only, got {self.group_size}")
        if self.vocab_size < 300:
            raise ValueError(f"vocab_size must be >= 300, got {self.vocab_size}")
        if self.num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}")
        if self.hidden_size <= 0 or self.hidden_size % self.group_size != 0:
            raise ValueError(f"hidden_size must be a positive multiple of {self.group_size}")
        if self.head_dim <= 0 or self.head_dim % 32 != 0:
            raise ValueError(f"head_dim must be a positive multiple of 32, got {self.head_dim}")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.num_attention_heads * self.head_dim) % self.group_size != 0:
            raise ValueError("q_dim must be a multiple of group_size")
        if (self.num_key_value_heads * self.head_dim) % self.group_size != 0:
            raise ValueError("kv_dim must be a multiple of group_size")
        if self.tie_word_embeddings and self.embed_f16:
            raise ValueError("tied quantized lm_head requires quantized embed_tokens (embed_f16=False)")
        if self.is_moe:
            if self.num_experts <= 0 or self.num_experts > 256:
                raise ValueError(f"num_experts must be in 1..256, got {self.num_experts}")
            if not (1 <= self.num_experts_per_tok <= self.num_experts):
                raise ValueError("num_experts_per_tok must be in 1..num_experts")
            if self.moe_intermediate_size <= 0 or self.moe_intermediate_size % self.group_size != 0:
                raise ValueError("moe_intermediate_size must be a positive multiple of group_size")
        else:
            if self.num_experts != 0 or self.num_experts_per_tok != 0:
                raise ValueError("dense model must have num_experts=0 and num_experts_per_tok=0")
            if self.intermediate_size <= 0 or self.intermediate_size % self.group_size != 0:
                raise ValueError("intermediate_size must be a positive multiple of group_size")

    def to_hf_config_dict(self) -> dict[str, object]:
        specials = self.special_token_ids()
        arch = "Qwen3MoeForCausalLM" if self.is_moe else "Qwen3ForCausalLM"
        return {
            "model_type": self.model_type,
            "architectures": [arch],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "norm_topk_prob": self.norm_topk_prob,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": specials["<|endoftext|>"],
            "eos_token_id": specials["<|im_end|>"],
            "quantization": {
                "bits": self.quant_bits,
                "group_size": self.group_size,
            },
        }


PRESETS: dict[str, Qwen3SynthConfig] = {
    "tiny-moe": Qwen3SynthConfig(
        model_id="qwen3-tiny-moe-3bit",
        model_type="qwen3_moe",
        quant_bits=3,
        group_size=64,
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=2048,
    ),
    "tiny-dense": Qwen3SynthConfig(
        model_id="qwen3-tiny-dense-4bit",
        model_type="qwen3",
        quant_bits=4,
        group_size=64,
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        moe_intermediate_size=0,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=2048,
    ),
    "qwen3-30b-a3b-3bit": Qwen3SynthConfig(
        model_id="qwen3-30b-a3b-3bit-synth",
        model_type="qwen3_moe",
        quant_bits=3,
        group_size=64,
        vocab_size=151936,
        hidden_size=2048,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        max_position_embeddings=32768,
    ),
    "qwen3-14b-4bit": Qwen3SynthConfig(
        model_id="qwen3-14b-4bit-synth",
        model_type="qwen3",
        quant_bits=4,
        group_size=64,
        vocab_size=151936,
        hidden_size=5120,
        num_hidden_layers=4,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=17408,
        moe_intermediate_size=0,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=32768,
    ),
}


@dataclass(frozen=True)
class SynthBundle:
    amf_path: pathlib.Path
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    metas: list[PackedTensorMeta]
    total_bytes: int
    config: Qwen3SynthConfig


def build_qwen3_tokenizer_dict(config: Qwen3SynthConfig) -> dict[str, object]:
    """Build a valid HuggingFace Byte-Level BPE tokenizer.json matching `config.vocab_size`."""
    b2u = bytes_to_unicode()
    vocab: dict[str, int] = {}
    merges: list[str] = []

    next_id = 0
    for b in range(256):
        vocab[b2u[b]] = next_id
        next_id += 1

    def add_merge(left: str, right: str) -> None:
        nonlocal next_id
        merged = left + right
        if merged in vocab:
            return
        if left not in vocab or right not in vocab:
            raise ValueError(f"merge components missing from vocab: {left!r}, {right!r}")
        vocab[merged] = next_id
        next_id += 1
        merges.append(f"{left} {right}")

    sp = b2u[ord(" ")]
    seed_pairs: list[tuple[str, str]] = [
        (sp, sp),
        (sp + sp, sp + sp),
        ("t", "h"),
        ("h", "e"),
        ("th", "e"),
        (sp, "t"),
        (sp + "t", "he"),
        ("l", "l"),
        ("he", "ll"),
        ("hell", "o"),
        ("w", "o"),
        ("r", "l"),
        ("wo", "rl"),
        ("worl", "d"),
        (sp, "world"),
        ("a", "m"),
        ("i", "n"),
        ("f", "e"),
        ("fe", "r"),
        ("am", "in"),
        ("amin", "fer"),
        ("Q", "w"),
        ("e", "n"),
        ("Qw", "en"),
        ("Qwen", "3"),
        ("d", "e"),
        ("de", "f"),
        ("r", "e"),
        ("t", "u"),
        ("r", "n"),
        ("tu", "rn"),
        ("re", "turn"),
    ]
    for left, right in seed_pairs:
        add_merge(left, right)

    # Multi-byte UTF-8 merges for common Chinese characters so CJK prompts exercise BPE merges.
    for word in ("你", "好", "测", "试"):
        encoded_bytes = word.encode("utf-8")
        cur = b2u[encoded_bytes[0]]
        for byte_val in encoded_bytes[1:]:
            nxt = b2u[byte_val]
            add_merge(cur, nxt)
            cur = cur + nxt

    specials = config.special_token_ids()
    special_id_set = set(specials.values())
    if next_id > min(special_id_set):
        raise ValueError("base BPE vocab collides with special token IDs")

    for tok_id in range(next_id, config.vocab_size):
        if tok_id in special_id_set:
            continue
        vocab[f"{sp}v{tok_id:06d}"] = tok_id

    added_tokens = [
        {
            "id": tok_id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, tok_id in sorted(specials.items(), key=lambda kv: kv[1])
    ]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


@dataclass(frozen=True)
class _TensorPlan:
    name: str
    kind: TensorKind
    dtype: DType
    quant_bits: int
    group_size: int
    layer: int
    expert: int
    shape: tuple[int, ...]
    scale_hint: float


def _plan_qwen3_tensors(
    config: Qwen3SynthConfig,
    cold_experts: set[tuple[int, int]],
) -> list[_TensorPlan]:
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    h = config.hidden_size
    q_dtype = config.quant_dtype
    bits = config.quant_bits
    gs = config.group_size

    plans: list[_TensorPlan] = []
    if config.embed_f16:
        plans.append(
            _TensorPlan(
                "embed_tokens",
                TensorKind.EMBEDDING,
                DType.F16,
                0,
                0,
                NO_LAYER,
                NO_EXPERT,
                (config.vocab_size, h),
                0.20,
            )
        )
    else:
        plans.append(
            _TensorPlan(
                "embed_tokens",
                TensorKind.EMBEDDING,
                q_dtype,
                bits,
                gs,
                NO_LAYER,
                NO_EXPERT,
                (config.vocab_size, h),
                0.06 if bits == 3 else 0.03,
            )
        )

    for layer in range(config.num_hidden_layers):
        plans.append(
            _TensorPlan(
                f"layers.{layer}.input_layernorm.weight",
                TensorKind.NORM,
                DType.F16,
                0,
                0,
                layer,
                NO_EXPERT,
                (h,),
                1.0,
            )
        )
        attn_scale = (0.35 if bits == 3 else 0.17) / math.sqrt(h)
        o_scale = (0.35 if bits == 3 else 0.17) / math.sqrt(q_dim)
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.q_proj",
                TensorKind.DENSE,
                q_dtype,
                bits,
                gs,
                layer,
                NO_EXPERT,
                (q_dim, h),
                attn_scale,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.k_proj",
                TensorKind.DENSE,
                q_dtype,
                bits,
                gs,
                layer,
                NO_EXPERT,
                (kv_dim, h),
                attn_scale,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.v_proj",
                TensorKind.DENSE,
                q_dtype,
                bits,
                gs,
                layer,
                NO_EXPERT,
                (kv_dim, h),
                attn_scale,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.o_proj",
                TensorKind.DENSE,
                q_dtype,
                bits,
                gs,
                layer,
                NO_EXPERT,
                (h, q_dim),
                o_scale,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.q_norm.weight",
                TensorKind.NORM,
                DType.F16,
                0,
                0,
                layer,
                NO_EXPERT,
                (config.head_dim,),
                1.0,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.self_attn.k_norm.weight",
                TensorKind.NORM,
                DType.F16,
                0,
                0,
                layer,
                NO_EXPERT,
                (config.head_dim,),
                1.0,
            )
        )
        plans.append(
            _TensorPlan(
                f"layers.{layer}.post_attention_layernorm.weight",
                TensorKind.NORM,
                DType.F16,
                0,
                0,
                layer,
                NO_EXPERT,
                (h,),
                1.0,
            )
        )

        if config.is_moe:
            plans.append(
                _TensorPlan(
                    f"layers.{layer}.mlp.gate.weight",
                    TensorKind.ROUTER,
                    DType.F16,
                    0,
                    0,
                    layer,
                    NO_EXPERT,
                    (config.num_experts, h),
                    0.5 / math.sqrt(h),
                )
            )
            e_in_scale = (0.30 if bits == 3 else 0.15) / math.sqrt(h)
            e_out_scale = (0.30 if bits == 3 else 0.15) / math.sqrt(config.moe_intermediate_size)
            for expert in range(config.num_experts):
                kind = TensorKind.COLD_EXPERT if (layer, expert) in cold_experts else TensorKind.HOT_EXPERT
                plans.append(
                    _TensorPlan(
                        f"layers.{layer}.mlp.experts.{expert}.gate_proj",
                        kind,
                        q_dtype,
                        bits,
                        gs,
                        layer,
                        expert,
                        (config.moe_intermediate_size, h),
                        e_in_scale,
                    )
                )
                plans.append(
                    _TensorPlan(
                        f"layers.{layer}.mlp.experts.{expert}.up_proj",
                        kind,
                        q_dtype,
                        bits,
                        gs,
                        layer,
                        expert,
                        (config.moe_intermediate_size, h),
                        e_in_scale,
                    )
                )
                plans.append(
                    _TensorPlan(
                        f"layers.{layer}.mlp.experts.{expert}.down_proj",
                        kind,
                        q_dtype,
                        bits,
                        gs,
                        layer,
                        expert,
                        (h, config.moe_intermediate_size),
                        e_out_scale,
                    )
                )
        else:
            d_in_scale = (0.30 if bits == 3 else 0.15) / math.sqrt(h)
            d_out_scale = (0.30 if bits == 3 else 0.15) / math.sqrt(config.intermediate_size)
            plans.append(
                _TensorPlan(
                    f"layers.{layer}.mlp.gate_proj",
                    TensorKind.DENSE,
                    q_dtype,
                    bits,
                    gs,
                    layer,
                    NO_EXPERT,
                    (config.intermediate_size, h),
                    d_in_scale,
                )
            )
            plans.append(
                _TensorPlan(
                    f"layers.{layer}.mlp.up_proj",
                    TensorKind.DENSE,
                    q_dtype,
                    bits,
                    gs,
                    layer,
                    NO_EXPERT,
                    (config.intermediate_size, h),
                    d_in_scale,
                )
            )
            plans.append(
                _TensorPlan(
                    f"layers.{layer}.mlp.down_proj",
                    TensorKind.DENSE,
                    q_dtype,
                    bits,
                    gs,
                    layer,
                    NO_EXPERT,
                    (h, config.intermediate_size),
                    d_out_scale,
                )
            )

    plans.append(
        _TensorPlan(
            "norm.weight",
            TensorKind.NORM,
            DType.F16,
            0,
            0,
            NO_LAYER,
            NO_EXPERT,
            (h,),
            1.0,
        )
    )
    if not config.tie_word_embeddings:
        lm_scale = (0.45 if bits == 3 else 0.22) / math.sqrt(h)
        plans.append(
            _TensorPlan(
                "lm_head",
                TensorKind.LM_HEAD,
                q_dtype,
                bits,
                gs,
                NO_LAYER,
                NO_EXPERT,
                (config.vocab_size, h),
                lm_scale,
            )
        )
    return plans


def _stream_zeros(fh, count: int, crc: int) -> int:
    remaining = count
    while remaining > 0:
        n = min(remaining, len(_ZERO_PAGE))
        chunk = _ZERO_PAGE[:n]
        fh.write(chunk)
        crc = zlib.crc32(chunk, crc)
        remaining -= n
    return crc


def _stream_tiled(fh, tile: bytes, total_bytes: int, crc: int) -> int:
    if total_bytes == 0:
        return crc
    # Expand tile up to ~1 MiB so write/crc loops run in C with minimal overhead.
    block = tile * max(1, (1 << 20) // len(tile))
    remaining = total_bytes
    while remaining > 0:
        n = min(remaining, len(block))
        chunk = block[:n] if n < len(block) else block
        fh.write(chunk)
        crc = zlib.crc32(chunk, crc)
        remaining -= n
    return crc


def _build_scale_bias_tile(rng: random.Random, bits: int, base_scale: float) -> bytes:
    """Build a prime-length tile of varied finite half2(scale, bias) entries."""
    mid = 3.5 if bits == 3 else 7.5
    table_entries = 257  # prime to prevent row/group periodicity
    buf = bytearray(table_entries * 4)
    for i in range(table_entries):
        s = base_scale * (0.75 + 0.50 * rng.random())
        # Keep scale safely above fp16 subnormal range.
        s = max(s, 1e-4)
        b = -mid * s + (rng.random() - 0.5) * 0.25 * s
        struct.pack_into("<ee", buf, 4 * i, s, b)
    return bytes(buf)


def _build_f16_tile(rng: random.Random, amp: float, centered_one: bool, entries: int) -> bytes:
    buf = bytearray(entries * 2)
    for i in range(entries):
        if centered_one:
            v = 0.95 + 0.10 * rng.random()
        else:
            v = (rng.random() * 2.0 - 1.0) * amp
        struct.pack_into("<e", buf, 2 * i, v)
    return bytes(buf)


def _stream_tensor_blob(
    fh,
    plan: _TensorPlan,
    seed: int,
    blob_offset: int,
    *,
    u32_payload: bytes | None = None,
) -> PackedTensorMeta:
    rng = random.Random(_tensor_seed(seed, plan.name))
    crc = 0

    if plan.dtype in (DType.Q3_AFFINE, DType.Q4_AFFINE):
        n, k = plan.shape
        data_bytes = n * ((k * plan.quant_bits) // 8)
        params_offset = _align_up(data_bytes, 64)
        params_bytes = n * (k // plan.group_size) * 4
        content_len = params_offset + params_bytes
        blob_bytes = _align_up(content_len, APPLE_PAGE_SIZE)

        remaining = data_bytes
        while remaining > 0:
            take = min(remaining, _STREAM_CHUNK_BYTES)
            chunk = rng.randbytes(take)
            fh.write(chunk)
            crc = zlib.crc32(chunk, crc)
            remaining -= take

        if params_offset > data_bytes:
            crc = _stream_zeros(fh, params_offset - data_bytes, crc)

        sb_tile = _build_scale_bias_tile(rng, plan.quant_bits, plan.scale_hint)
        crc = _stream_tiled(fh, sb_tile, params_bytes, crc)

        if blob_bytes > content_len:
            crc = _stream_zeros(fh, blob_bytes - content_len, crc)
    elif plan.dtype in (DType.F16, DType.BF16):
        elems = 1
        for d in plan.shape:
            elems *= d
        data_bytes = elems * 2
        params_offset = 0
        params_bytes = 0
        blob_bytes = _align_up(data_bytes, APPLE_PAGE_SIZE)

        if plan.kind == TensorKind.NORM:
            tile = _build_f16_tile(rng, 0.05, centered_one=True, entries=elems)
        elif plan.kind == TensorKind.ROUTER:
            tile = _build_f16_tile(rng, plan.scale_hint, centered_one=False, entries=257)
        else:
            tile = _build_f16_tile(rng, plan.scale_hint, centered_one=False, entries=4099)
        if plan.dtype == DType.BF16:
            tile = f16_bytes_to_bf16_bytes(tile)
        crc = _stream_tiled(fh, tile, data_bytes, crc)

        if blob_bytes > data_bytes:
            crc = _stream_zeros(fh, blob_bytes - data_bytes, crc)
    elif plan.dtype == DType.U32:
        elems = 1
        for d in plan.shape:
            elems *= d
        data_bytes = elems * 4
        params_offset = 0
        params_bytes = 0
        blob_bytes = _align_up(data_bytes, APPLE_PAGE_SIZE)
        if u32_payload is None or len(u32_payload) != data_bytes:
            raise ValueError(f"missing or mismatched u32 payload for {plan.name!r}")
        fh.write(u32_payload)
        crc = zlib.crc32(u32_payload, crc)
        if blob_bytes > data_bytes:
            crc = _stream_zeros(fh, blob_bytes - data_bytes, crc)
    else:
        raise ValueError(f"unsupported synthetic dtype {plan.dtype}")

    return PackedTensorMeta(
        name=plan.name,
        kind=plan.kind,
        dtype=plan.dtype,
        quant_bits=plan.quant_bits,
        group_size=plan.group_size,
        layer=plan.layer,
        expert=plan.expert,
        shape=plan.shape,
        blob_offset=blob_offset,
        blob_bytes=blob_bytes,
        data_bytes=data_bytes,
        params_offset=params_offset,
        params_bytes=params_bytes,
        blob_crc32=crc & 0xFFFFFFFF,
    )


def _write_planned_amf(
    amf_path: pathlib.Path,
    model_id: str,
    plans: list[_TensorPlan],
    seed: int,
    *,
    u32_payloads: dict[str, bytes] | None = None,
) -> list[PackedTensorMeta]:
    tensor_count = len(plans)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)

    metas: list[PackedTensorMeta] = []
    with amf_path.open("wb") as fh:
        # Reserve header page + directory pages up front; fill after streaming all blobs.
        _stream_zeros(fh, dir_offset + dir_bytes, 0)
        cursor = dir_offset + dir_bytes
        for plan in plans:
            payload = u32_payloads.get(plan.name) if u32_payloads else None
            meta = _stream_tensor_blob(fh, plan, seed, cursor, u32_payload=payload)
            metas.append(meta)
            cursor += meta.blob_bytes

        dir_raw = bytearray(dir_raw_len)
        for i, m in enumerate(metas):
            padded_shape = [m.shape[d] if d < len(m.shape) else 0 for d in range(4)]
            ENTRY_STRUCT.pack_into(
                dir_raw,
                i * ENTRY_STRUCT.size,
                m.name.encode("utf-8"),
                int(m.kind),
                int(m.dtype),
                m.quant_bits,
                len(m.shape),
                m.group_size,
                m.layer,
                m.expert,
                0,
                m.blob_crc32,
                padded_shape[0],
                padded_shape[1],
                padded_shape[2],
                padded_shape[3],
                m.blob_offset,
                m.blob_bytes,
                m.data_bytes,
                m.params_offset,
                m.params_bytes,
                0,
            )
        dir_crc = zlib.crc32(dir_raw) & 0xFFFFFFFF

        model_id_bytes = model_id.encode("utf-8")
        header_zero_crc = HEADER_STRUCT.pack(
            AMF_MAGIC,
            AMF_VERSION,
            0,
            APPLE_PAGE_SIZE,
            CHECKSUM_CRC32_IEEE,
            tensor_count,
            0,
            dir_offset,
            dir_bytes,
            dir_crc,
            0,
            model_id_bytes,
            0,
        )
        header_crc = zlib.crc32(header_zero_crc) & 0xFFFFFFFF
        header_packed = HEADER_STRUCT.pack(
            AMF_MAGIC,
            AMF_VERSION,
            0,
            APPLE_PAGE_SIZE,
            CHECKSUM_CRC32_IEEE,
            tensor_count,
            0,
            dir_offset,
            dir_bytes,
            dir_crc,
            header_crc,
            model_id_bytes,
            0,
        )

        page0 = bytearray(APPLE_PAGE_SIZE)
        page0[0 : len(header_packed)] = header_packed
        fh.seek(0)
        fh.write(page0)

        if dir_bytes > 0:
            dir_page = bytearray(dir_bytes)
            dir_page[0 : len(dir_raw)] = dir_raw
            fh.seek(dir_offset)
            fh.write(dir_page)
    return metas


def generate_qwen3_bundle(
    output_dir: pathlib.Path,
    config: Qwen3SynthConfig,
    *,
    seed: int = 42,
    cold_experts: Iterable[tuple[int, int]] = (),
) -> SynthBundle:
    """Generate deterministic `model.amf`, `config.json`, and `tokenizer.json` in `output_dir`."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    amf_path = output_dir / "model.amf"
    config_path = output_dir / "config.json"
    tokenizer_path = output_dir / "tokenizer.json"

    plans = _plan_qwen3_tensors(config, set(cold_experts))
    metas = _write_planned_amf(amf_path, config.model_id, plans, seed)

    config_path.write_text(
        json.dumps(config.to_hf_config_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer_path.write_text(
        json.dumps(build_qwen3_tokenizer_dict(config), separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return SynthBundle(
        amf_path=amf_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        metas=metas,
        total_bytes=amf_path.stat().st_size,
        config=config,
    )


# ---------------------------------------------------------------------------
# Deterministic synthetic L2 draft head bundles (ADR-0013: EAGLE-3, EAGLE-2, MTP)
# ---------------------------------------------------------------------------

DRAFT_FORMATS = ("eagle3", "eagle2", "mtp")
DRAFT_DTYPES = ("bf16", "f16")


@dataclass(frozen=True)
class DraftSynthConfig:
    model_id: str
    draft_format: str  # "eagle3", "eagle2", or "mtp"
    vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int = 1
    dtype: str = "bf16"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 2048
    norm_before_residual: bool = True
    aux_layer_ids: tuple[int, ...] | None = None
    target_hidden_size: int | None = None
    include_embed_tokens: bool = True
    include_lm_head: bool = True
    include_qk_norm: bool = False

    def with_overrides(self, **kwargs: object) -> DraftSynthConfig:
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @property
    def float_dtype(self) -> DType:
        return DType.BF16 if self.dtype == "bf16" else DType.F16

    @property
    def num_aux_layers(self) -> int:
        return len(self.aux_layer_ids) if self.aux_layer_ids else 3

    @property
    def effective_target_hidden_size(self) -> int:
        return self.target_hidden_size if self.target_hidden_size is not None else self.hidden_size

    def validate(self) -> None:
        if len(self.model_id.encode("utf-8")) == 0 or len(self.model_id.encode("utf-8")) > 63:
            raise ValueError(f"model_id must be 1..63 UTF-8 bytes, got {self.model_id!r}")
        if self.draft_format not in DRAFT_FORMATS:
            raise ValueError(f"draft_format must be one of {DRAFT_FORMATS}, got {self.draft_format!r}")
        if self.dtype not in DRAFT_DTYPES:
            raise ValueError(f"dtype must be one of {DRAFT_DTYPES}, got {self.dtype!r}")
        if self.vocab_size < 16:
            raise ValueError(f"vocab_size must be >= 16, got {self.vocab_size}")
        if self.draft_format == "eagle3":
            if not (1 <= self.draft_vocab_size <= self.vocab_size):
                raise ValueError(
                    f"draft_vocab_size must be in 1..vocab_size ({self.vocab_size}), got {self.draft_vocab_size}"
                )
        elif self.draft_vocab_size != self.vocab_size:
            raise ValueError(
                f"{self.draft_format} requires draft_vocab_size == vocab_size "
                f"({self.draft_vocab_size} != {self.vocab_size})"
            )
        if self.num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}")
        if self.hidden_size <= 0 or self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 64, got {self.hidden_size}")
        if self.effective_target_hidden_size <= 0 or self.effective_target_hidden_size % 64 != 0:
            raise ValueError(
                f"target_hidden_size must be a positive multiple of 64, got {self.effective_target_hidden_size}"
            )
        if self.head_dim <= 0 or self.head_dim % 32 != 0:
            raise ValueError(f"head_dim must be a positive multiple of 32, got {self.head_dim}")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.intermediate_size <= 0 or self.intermediate_size % 64 != 0:
            raise ValueError(f"intermediate_size must be a positive multiple of 64, got {self.intermediate_size}")
        if self.aux_layer_ids is not None:
            if len(self.aux_layer_ids) == 0 or min(self.aux_layer_ids) < 0:
                raise ValueError(f"aux_layer_ids must be non-empty non-negative ints, got {self.aux_layer_ids!r}")
            if len(set(self.aux_layer_ids)) != len(self.aux_layer_ids):
                raise ValueError(f"aux_layer_ids must be distinct, got {self.aux_layer_ids!r}")

    def to_config_dict(self) -> dict[str, object]:
        if self.draft_format == "eagle3":
            return {
                "architectures": ["Eagle3DraftModel"],
                "speculators_model_type": "eagle3",
                "draft_vocab_size": self.draft_vocab_size,
                "eagle_aux_hidden_state_layer_ids": (
                    list(self.aux_layer_ids) if self.aux_layer_ids is not None else None
                ),
                "norm_before_residual": self.norm_before_residual,
                "target_hidden_size": self.target_hidden_size,
                "transformer_layer_config": {
                    "attention_bias": False,
                    "head_dim": self.head_dim,
                    "hidden_act": "silu",
                    "hidden_size": self.hidden_size,
                    "intermediate_size": self.intermediate_size,
                    "max_position_embeddings": self.max_position_embeddings,
                    "mlp_bias": False,
                    "model_type": "llama",
                    "num_attention_heads": self.num_attention_heads,
                    "num_hidden_layers": self.num_hidden_layers,
                    "num_key_value_heads": self.num_key_value_heads,
                    "rms_norm_eps": self.rms_norm_eps,
                    "rope_theta": self.rope_theta,
                    "vocab_size": self.vocab_size,
                },
            }
        if self.draft_format == "eagle2":
            return {
                "architectures": ["EagleLlamaForCausalLM"],
                "model_type": "eagle",
                "draft_format": "eagle2",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size,
                "rms_norm_eps": self.rms_norm_eps,
                "rope_theta": self.rope_theta,
                "max_position_embeddings": self.max_position_embeddings,
                "has_qk_norm": self.include_qk_norm,
                "share_embed_tokens": not self.include_embed_tokens,
            }
        return {
            "architectures": ["MtpDraftModel"],
            "model_type": "mtp",
            "draft_format": "mtp",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_nextn_predict_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "has_qk_norm": self.include_qk_norm,
            "share_lm_head": not self.include_lm_head,
            "share_embed_tokens": not self.include_embed_tokens,
        }


DRAFT_PRESETS: dict[str, DraftSynthConfig] = {
    "tiny-eagle3": DraftSynthConfig(
        model_id="tiny-eagle3-draft",
        draft_format="eagle3",
        vocab_size=512,
        draft_vocab_size=256,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=1,
        dtype="bf16",
        rope_theta=10_000.0,
        max_position_embeddings=2048,
        norm_before_residual=True,
        include_embed_tokens=True,
        include_lm_head=True,
        include_qk_norm=False,
    ),
    "tiny-eagle2": DraftSynthConfig(
        model_id="tiny-eagle2-draft",
        draft_format="eagle2",
        vocab_size=512,
        draft_vocab_size=512,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=1,
        dtype="bf16",
        rope_theta=1_000_000.0,
        max_position_embeddings=2048,
        include_embed_tokens=True,
        include_lm_head=True,
        include_qk_norm=False,
    ),
    "tiny-mtp": DraftSynthConfig(
        model_id="tiny-mtp-draft",
        draft_format="mtp",
        vocab_size=512,
        draft_vocab_size=512,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=1,
        dtype="bf16",
        rope_theta=1_000_000.0,
        max_position_embeddings=2048,
        include_embed_tokens=False,
        include_lm_head=False,
        include_qk_norm=True,
    ),
    "qwen3-30b-a3b-eagle3": DraftSynthConfig(
        model_id="qwen3-30b-a3b-eagle3-synth",
        draft_format="eagle3",
        vocab_size=151936,
        draft_vocab_size=64000,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=6144,
        num_hidden_layers=1,
        dtype="bf16",
        rope_theta=10_000.0,
        max_position_embeddings=40960,
        norm_before_residual=True,
        include_embed_tokens=True,
        include_lm_head=True,
        include_qk_norm=False,
    ),
}


@dataclass(frozen=True)
class DraftBundle:
    amf_path: pathlib.Path
    config_path: pathlib.Path
    metas: list[PackedTensorMeta]
    total_bytes: int
    config: DraftSynthConfig


def _build_synthetic_d2t_bytes(vocab_size: int, draft_vocab_size: int, seed: int) -> bytes:
    """Build deterministic, strictly increasing u32 target token IDs in `[0, vocab_size)`."""
    if draft_vocab_size == vocab_size:
        ids = list(range(vocab_size))
    else:
        rng = random.Random(_tensor_seed(seed, "d2t"))
        prefix = min(draft_vocab_size // 2, 256)
        rest_count = draft_vocab_size - prefix
        rest = rng.sample(range(prefix, vocab_size), rest_count)
        rest.sort()
        ids = list(range(prefix)) + rest
    return struct.pack(f"<{len(ids)}I", *ids)


def _plan_draft_tensors(config: DraftSynthConfig) -> list[_TensorPlan]:
    fdt = config.float_dtype
    h = config.hidden_size
    h_tgt = config.effective_target_hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    inv_sqrt_h = 1.0 / math.sqrt(h)
    inv_sqrt_q = 1.0 / math.sqrt(q_dim)
    inv_sqrt_inter = 1.0 / math.sqrt(inter)

    plans: list[_TensorPlan] = []

    if config.draft_format == "eagle3":
        plans.append(
            _TensorPlan("d2t", TensorKind.DENSE, DType.U32, 0, 0, NO_LAYER, NO_EXPERT, (config.draft_vocab_size,), 0.0)
        )
        if config.include_embed_tokens:
            plans.append(
                _TensorPlan(
                    "embed_tokens.weight",
                    TensorKind.EMBEDDING,
                    fdt,
                    0,
                    0,
                    NO_LAYER,
                    NO_EXPERT,
                    (config.vocab_size, h),
                    0.20,
                )
            )
        fc_in = config.num_aux_layers * h_tgt
        plans.append(
            _TensorPlan("fc.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h, fc_in), 0.5 / math.sqrt(fc_in))
        )
        for layer in range(config.num_hidden_layers):
            qkv_in = 2 * h if layer == 0 else h
            qkv_scale = 0.5 / math.sqrt(qkv_in)
            plans.append(
                _TensorPlan(f"layers.{layer}.input_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
            )
            if layer == 0:
                plans.append(
                    _TensorPlan(f"layers.{layer}.hidden_norm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
                )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (q_dim, qkv_in), qkv_scale)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, qkv_in), qkv_scale)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, qkv_in), qkv_scale)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, q_dim), 0.5 * inv_sqrt_q)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.post_attention_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.up_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.down_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, inter), 0.5 * inv_sqrt_inter)
            )
        plans.append(_TensorPlan("norm.weight", TensorKind.NORM, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
        if config.include_lm_head:
            plans.append(
                _TensorPlan(
                    "lm_head.weight",
                    TensorKind.LM_HEAD,
                    fdt,
                    0,
                    0,
                    NO_LAYER,
                    NO_EXPERT,
                    (config.draft_vocab_size, h),
                    0.5 * inv_sqrt_h,
                )
            )
        return plans

    if config.draft_format == "eagle2":
        if config.include_embed_tokens:
            plans.append(
                _TensorPlan(
                    "embed_tokens.weight",
                    TensorKind.EMBEDDING,
                    fdt,
                    0,
                    0,
                    NO_LAYER,
                    NO_EXPERT,
                    (config.vocab_size, h),
                    0.20,
                )
            )
        plans.append(
            _TensorPlan("fc.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h, 2 * h), 0.5 / math.sqrt(2 * h))
        )
        for layer in range(config.num_hidden_layers):
            plans.append(
                _TensorPlan(f"layers.{layer}.input_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (q_dim, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, q_dim), 0.5 * inv_sqrt_q)
            )
            if config.include_qk_norm:
                plans.append(
                    _TensorPlan(f"layers.{layer}.self_attn.q_norm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (config.head_dim,), 1.0)
                )
                plans.append(
                    _TensorPlan(f"layers.{layer}.self_attn.k_norm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (config.head_dim,), 1.0)
                )
            plans.append(
                _TensorPlan(f"layers.{layer}.post_attention_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.up_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.mlp.down_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, inter), 0.5 * inv_sqrt_inter)
            )
        plans.append(_TensorPlan("norm.weight", TensorKind.NORM, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
        if config.include_lm_head:
            plans.append(
                _TensorPlan("lm_head.weight", TensorKind.LM_HEAD, fdt, 0, 0, NO_LAYER, NO_EXPERT, (config.vocab_size, h), 0.5 * inv_sqrt_h)
            )
        return plans

    # MTP (D3 Multi-Token Prediction module)
    if config.include_embed_tokens:
        plans.append(
            _TensorPlan(
                "embed_tokens.weight",
                TensorKind.EMBEDDING,
                fdt,
                0,
                0,
                NO_LAYER,
                NO_EXPERT,
                (config.vocab_size, h),
                0.20,
            )
        )
    plans.append(_TensorPlan("enorm.weight", TensorKind.NORM, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
    plans.append(_TensorPlan("hnorm.weight", TensorKind.NORM, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
    plans.append(
        _TensorPlan("eh_proj.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h, 2 * h), 0.5 / math.sqrt(2 * h))
    )
    for layer in range(config.num_hidden_layers):
        plans.append(
            _TensorPlan(f"layers.{layer}.input_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (q_dim, h), 0.5 * inv_sqrt_h)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, h), 0.5 * inv_sqrt_h)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (kv_dim, h), 0.5 * inv_sqrt_h)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, q_dim), 0.5 * inv_sqrt_q)
        )
        if config.include_qk_norm:
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.q_norm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (config.head_dim,), 1.0)
            )
            plans.append(
                _TensorPlan(f"layers.{layer}.self_attn.k_norm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (config.head_dim,), 1.0)
            )
        plans.append(
            _TensorPlan(f"layers.{layer}.post_attention_layernorm.weight", TensorKind.NORM, fdt, 0, 0, layer, NO_EXPERT, (h,), 1.0)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.mlp.up_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (inter, h), 0.5 * inv_sqrt_h)
        )
        plans.append(
            _TensorPlan(f"layers.{layer}.mlp.down_proj.weight", TensorKind.DENSE, fdt, 0, 0, layer, NO_EXPERT, (h, inter), 0.5 * inv_sqrt_inter)
        )
    plans.append(_TensorPlan("shared_head.norm.weight", TensorKind.NORM, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
    if config.include_lm_head:
        plans.append(
            _TensorPlan("lm_head.weight", TensorKind.LM_HEAD, fdt, 0, 0, NO_LAYER, NO_EXPERT, (config.vocab_size, h), 0.5 * inv_sqrt_h)
        )
    return plans


def generate_draft_bundle(
    output_dir: pathlib.Path,
    config: DraftSynthConfig,
    *,
    seed: int = 42,
) -> DraftBundle:
    """Generate deterministic L2 draft head `model.amf` and `config.json` in `output_dir`.

    Supports all three ADR-0013 draft head layouts: `eagle3`, `eagle2`, and `mtp`.
    """
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    amf_path = output_dir / "model.amf"
    config_path = output_dir / "config.json"

    plans = _plan_draft_tensors(config)
    u32_payloads: dict[str, bytes] = {}
    if config.draft_format == "eagle3":
        u32_payloads["d2t"] = _build_synthetic_d2t_bytes(config.vocab_size, config.draft_vocab_size, seed)

    metas = _write_planned_amf(amf_path, config.model_id, plans, seed, u32_payloads=u32_payloads)
    config_path.write_text(
        json.dumps(config.to_config_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    return DraftBundle(
        amf_path=amf_path,
        config_path=config_path,
        metas=metas,
        total_bytes=amf_path.stat().st_size,
        config=config,
    )
