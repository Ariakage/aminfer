"""Pure-stdlib CoreML `.mlpackage` exporter for the Apple Neural Engine (W7, M4).

Emits genuine macOS CoreML Package v1.0.0 directory bundles (`Manifest.json`,
`Data/com.apple.CoreML/model.mlmodel`, `Metadata.json`, and
`FeatureDescriptions.json`) with zero third-party dependencies (`coremltools`,
`torch`, and `numpy` are not required — ADR-0001).

All exported graphs use `specificationVersion = 7` with `FLOAT16` (`65552`)
inputs and outputs whose trailing dimension is a multiple of 32 elements (64 B
row stride), matching `CVPixelBuffer` (`OneComponent16Half`) IOSurface tiles
bound zero-copy by `src/bridge/apple_bridge.m` (`ADR-0004`, `ADR-0010`).

Supported export kinds:
- `expert-predictor` (`A2`): cross-layer MoE router predictor dequantized from
  `.amf` `layers.{l}.mlp.gate` weights (or synthetic config).
- `attention-qkv` (`A1`): 512-token chunked layer Q/K/V (and optional O)
  projections from `.amf` `layers.{l}.self_attn.*` weights (or synthetic).
- `draft-head` (`D4`/`D5`): draft SwiGLU FFN (`gate_proj`/`up_proj`/`down_proj`),
  EAGLE-3 `fc` fusion projection, or linear projection from `.amf` or synthetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import shutil
import struct
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from aminfer_tools.amf import (
    NO_LAYER,
    AmfReader,
    PackedTensorMeta,
    TensorKind,
    _align_up,
)

COREML_SPEC_VERSION_FP16 = 7
COREML_ARRAY_FLOAT16 = 65552
COREML_ARRAY_FLOAT32 = 65568
COREML_EXACT_ARRAY_MAPPING = 1
IOSURFACE_F16_ROW_ALIGN = 32  # 32 fp16 elements = 64 bytes (Apple IOSurface row stride)

COREML_KINDS = ("expert-predictor", "attention-qkv", "draft-head")
COREML_OPS = ("inner-product", "conv1x1")
DRAFT_STAGES = ("ffn", "fc", "linear")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype_code: int = COREML_ARRAY_FLOAT16
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": "FLOAT16" if self.dtype_code == COREML_ARRAY_FLOAT16 else "FLOAT32",
            "description": self.description,
        }


@dataclass(frozen=True)
class CoreMlExportResult:
    output_path: pathlib.Path
    kind: str
    op: str
    model_id: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    layer_count: int
    weight_bytes: int
    model_bytes: int
    metadata: dict[str, str]
    bundle_paths: tuple[pathlib.Path, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "kind": self.kind,
            "op": self.op,
            "model_id": self.model_id,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "layer_count": self.layer_count,
            "weight_bytes": self.weight_bytes,
            "model_bytes": self.model_bytes,
            "metadata": dict(self.metadata),
            "bundle_paths": [str(p) for p in self.bundle_paths],
        }


# ---------------------------------------------------------------------------
# Minimal Protobuf wire encoder (CoreML Model / NeuralNetwork v7)
# Field numbers empirically verified against `xcrun coremlcompiler` on macOS.
# ---------------------------------------------------------------------------


def _encode_varint(val: int) -> bytes:
    if val < 0:
        raise ValueError(f"protobuf varint must be non-negative, got {val}")
    out = bytearray()
    v = val
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def _pb_varint(field: int, val: int) -> bytes:
    return _encode_varint((field << 3) | 0) + _encode_varint(val)


def _pb_bytes(field: int, payload: bytes | bytearray | memoryview) -> bytes:
    raw = bytes(payload) if not isinstance(payload, bytes) else payload
    return _encode_varint((field << 3) | 2) + _encode_varint(len(raw)) + raw


def _pb_str(field: int, text: str) -> bytes:
    return _pb_bytes(field, text.encode("utf-8"))


def _pb_feature_desc(spec: TensorSpec) -> bytes:
    # ArrayFeatureType: repeated int64 shape (1), ArrayDataType dataType (2)
    arr_parts = [_pb_varint(1, d) for d in spec.shape]
    arr_parts.append(_pb_varint(2, spec.dtype_code))
    arr_type = b"".join(arr_parts)
    # FeatureType: ArrayFeatureType multiArrayType (5)
    feat_type = _pb_bytes(5, arr_type)
    # FeatureDescription: name (1), shortDescription (2), type (3)
    parts = [_pb_str(1, spec.name)]
    if spec.description:
        parts.append(_pb_str(2, spec.description))
    parts.append(_pb_bytes(3, feat_type))
    return b"".join(parts)


def _pb_metadata(short_description: str, author: str, user_defined: dict[str, str]) -> bytes:
    # Metadata: shortDescription (1), author (3), map<string, string> userDefined (100)
    parts: list[bytes] = []
    if short_description:
        parts.append(_pb_str(1, short_description))
    if author:
        parts.append(_pb_str(3, author))
    for k, v in sorted(user_defined.items()):
        entry = _pb_str(1, str(k)) + _pb_str(2, str(v))
        parts.append(_pb_bytes(100, entry))
    return b"".join(parts)


def _pb_weight_params_f16(weights_f16: bytes) -> bytes:
    # WeightParams: bytes float16Value (2)
    return _pb_bytes(2, weights_f16)


def pb_inner_product_layer(
    name: str,
    input_blob: str,
    output_blob: str,
    c_in: int,
    c_out: int,
    weights_f16: bytes,
    bias_f16: bytes | None = None,
) -> bytes:
    """Encode a CoreML `NeuralNetworkLayer` with `InnerProductLayerParams` (field 140)."""
    expected = c_out * c_in * 2
    if len(weights_f16) != expected:
        raise ValueError(
            f"layer {name!r}: expected {expected} fp16 weight bytes for [{c_out}, {c_in}], got {len(weights_f16)}"
        )
    parts = [
        _pb_varint(1, c_in),
        _pb_varint(2, c_out),
        _pb_varint(10, 1 if bias_f16 is not None else 0),
        _pb_bytes(20, _pb_weight_params_f16(weights_f16)),
    ]
    if bias_f16 is not None:
        if len(bias_f16) != c_out * 2:
            raise ValueError(f"layer {name!r}: expected {c_out * 2} bias bytes, got {len(bias_f16)}")
        parts.append(_pb_bytes(30, _pb_weight_params_f16(bias_f16)))
    return (
        _pb_str(1, name)
        + _pb_str(2, input_blob)
        + _pb_str(3, output_blob)
        + _pb_bytes(140, b"".join(parts))
    )


def pb_conv1x1_layer(
    name: str,
    input_blob: str,
    output_blob: str,
    c_in: int,
    c_out: int,
    weights_f16: bytes,
    bias_f16: bytes | None = None,
) -> bytes:
    """Encode a CoreML `NeuralNetworkLayer` with 1x1 `ConvolutionLayerParams` (field 100)."""
    expected = c_out * c_in * 2
    if len(weights_f16) != expected:
        raise ValueError(
            f"layer {name!r}: expected {expected} fp16 weight bytes for [{c_out}, {c_in}], got {len(weights_f16)}"
        )
    parts = [
        _pb_varint(1, c_out),
        _pb_varint(2, c_in),
        _pb_varint(10, 1),  # nGroups
        _pb_varint(20, 1),  # kernelSize H
        _pb_varint(20, 1),  # kernelSize W
        _pb_varint(30, 1),  # stride H
        _pb_varint(30, 1),  # stride W
        _pb_bytes(50, b""),  # ValidPadding
        _pb_varint(70, 1 if bias_f16 is not None else 0),
        _pb_bytes(90, _pb_weight_params_f16(weights_f16)),
    ]
    if bias_f16 is not None:
        if len(bias_f16) != c_out * 2:
            raise ValueError(f"layer {name!r}: expected {c_out * 2} bias bytes, got {len(bias_f16)}")
        parts.append(_pb_bytes(95, _pb_weight_params_f16(bias_f16)))
    return (
        _pb_str(1, name)
        + _pb_str(2, input_blob)
        + _pb_str(3, output_blob)
        + _pb_bytes(100, b"".join(parts))
    )


def pb_linear_layer(
    op: str,
    name: str,
    input_blob: str,
    output_blob: str,
    c_in: int,
    c_out: int,
    weights_f16: bytes,
    bias_f16: bytes | None = None,
) -> bytes:
    if op == "inner-product":
        return pb_inner_product_layer(name, input_blob, output_blob, c_in, c_out, weights_f16, bias_f16)
    if op == "conv1x1":
        return pb_conv1x1_layer(name, input_blob, output_blob, c_in, c_out, weights_f16, bias_f16)
    raise ValueError(f"unsupported CoreML linear op {op!r} (expected one of {COREML_OPS})")


def pb_sigmoid_layer(name: str, input_blob: str, output_blob: str) -> bytes:
    """Encode `ActivationParams` with `ActivationSigmoid` (layer field 130, sigmoid field 40)."""
    return (
        _pb_str(1, name)
        + _pb_str(2, input_blob)
        + _pb_str(3, output_blob)
        + _pb_bytes(130, _pb_bytes(40, b""))
    )


def pb_add_layer(name: str, input_blobs: Sequence[str], output_blob: str) -> bytes:
    """Encode `AddLayerParams` (layer field 230)."""
    parts = [_pb_str(1, name)]
    for inp in input_blobs:
        parts.append(_pb_str(2, inp))
    parts.append(_pb_str(3, output_blob))
    parts.append(_pb_bytes(230, b""))
    return b"".join(parts)


def pb_multiply_layer(name: str, input_blobs: Sequence[str], output_blob: str) -> bytes:
    """Encode `MultiplyLayerParams` (layer field 231)."""
    parts = [_pb_str(1, name)]
    for inp in input_blobs:
        parts.append(_pb_str(2, inp))
    parts.append(_pb_str(3, output_blob))
    parts.append(_pb_bytes(231, b""))
    return b"".join(parts)


def build_neural_network_model(
    inputs: Sequence[TensorSpec],
    outputs: Sequence[TensorSpec],
    layer_messages: Sequence[bytes],
    *,
    use_exact_mapping: bool,
    specification_version: int = COREML_SPEC_VERSION_FP16,
    short_description: str = "",
    author: str = "aminfer-tools",
    user_defined: dict[str, str] | None = None,
) -> bytes:
    """Assemble a complete CoreML `Model` protobuf (`.mlmodel`) payload."""
    if not inputs or not outputs or not layer_messages:
        raise ValueError("CoreML model requires non-empty inputs, outputs, and layers")

    desc_parts = [_pb_bytes(1, _pb_feature_desc(t)) for t in inputs]
    desc_parts.extend(_pb_bytes(10, _pb_feature_desc(t)) for t in outputs)
    if short_description or author or user_defined:
        desc_parts.append(_pb_bytes(100, _pb_metadata(short_description, author, user_defined or {})))
    model_desc = b"".join(desc_parts)

    nn_parts: list[bytes] = []
    if use_exact_mapping:
        nn_parts.append(_pb_varint(5, COREML_EXACT_ARRAY_MAPPING))
    for lm in layer_messages:
        nn_parts.append(_pb_bytes(1, lm))
    nn_payload = b"".join(nn_parts)

    return (
        _pb_varint(1, specification_version)
        + _pb_bytes(2, model_desc)
        + _pb_bytes(500, nn_payload)
    )


def _validate_iosurface_shapes(specs: Sequence[TensorSpec]) -> None:
    for spec in specs:
        if len(spec.shape) < 2:
            raise ValueError(
                f"CoreML IOSurface tensor {spec.name!r} must have rank >= 2, got shape {spec.shape}"
            )
        for dim in spec.shape:
            if dim <= 0:
                raise ValueError(f"CoreML tensor {spec.name!r} has non-positive dim in {spec.shape}")
        if spec.dtype_code == COREML_ARRAY_FLOAT16 and spec.shape[-1] % IOSURFACE_F16_ROW_ALIGN != 0:
            raise ValueError(
                f"CoreML FLOAT16 tensor {spec.name!r} trailing dim {spec.shape[-1]} must be a multiple of "
                f"{IOSURFACE_F16_ROW_ALIGN} (64 B CVPixelBuffer IOSurface row stride)"
            )


def write_mlpackage(
    output_path: pathlib.Path,
    model_bytes: bytes,
    inputs: Sequence[TensorSpec],
    outputs: Sequence[TensorSpec],
    *,
    short_description: str = "",
    author: str = "aminfer-tools",
    user_defined: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write a validated macOS CoreML Package v1.0.0 `.mlpackage` directory atomically."""
    _validate_iosurface_shapes(inputs)
    _validate_iosurface_shapes(outputs)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f".{out.name}.partial.{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)

    try:
        data_dir = staging / "Data" / "com.apple.CoreML"
        data_dir.mkdir(parents=True, exist_ok=True)

        (data_dir / "model.mlmodel").write_bytes(model_bytes)

        meta_json: dict[str, object] = {}
        if short_description:
            meta_json["MLModelDescriptionKey"] = short_description
        if author:
            meta_json["MLModelAuthorKey"] = author
        if user_defined:
            meta_json["MLModelCreatorDefinedKey"] = {str(k): str(v) for k, v in sorted(user_defined.items())}
        (data_dir / "Metadata.json").write_text(json.dumps(meta_json, indent=2) + "\n", encoding="utf-8")

        feat_desc = {
            "Inputs": {t.name: {"MLFeatureShortDescription": t.description} for t in inputs},
            "Outputs": {t.name: {"MLFeatureShortDescription": t.description} for t in outputs},
            "TrainingInputs": {},
            "States": {},
        }
        (data_dir / "FeatureDescriptions.json").write_text(
            json.dumps(feat_desc, indent=2) + "\n",
            encoding="utf-8",
        )

        # Deterministic UUIDv5 identifiers for byte-reproducible .mlpackage manifests.
        ns = uuid.NAMESPACE_URL
        key_prefix = f"https://aminfer.local/coreml/{out.name}"
        root_id = str(uuid.uuid5(ns, f"{key_prefix}/model.mlmodel")).upper()
        meta_id = str(uuid.uuid5(ns, f"{key_prefix}/Metadata.json")).upper()
        feat_id = str(uuid.uuid5(ns, f"{key_prefix}/FeatureDescriptions.json")).upper()

        manifest = {
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": {
                root_id: {
                    "author": "com.apple.CoreML",
                    "description": "CoreML Model Specification",
                    "name": "model.mlmodel",
                    "path": "com.apple.CoreML/model.mlmodel",
                },
                meta_id: {
                    "author": "com.apple.CoreML",
                    "description": "External Metadata Overlay",
                    "name": "Metadata.json",
                    "path": "com.apple.CoreML/Metadata.json",
                },
                feat_id: {
                    "author": "com.apple.CoreML",
                    "description": "External FeatureDescription Overlay",
                    "name": "FeatureDescriptions.json",
                    "path": "com.apple.CoreML/FeatureDescriptions.json",
                },
            },
            "rootModelIdentifier": root_id,
        }
        (staging / "Manifest.json").write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")

        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out


# ---------------------------------------------------------------------------
# Helpers: layer specs, tensor lookup, synthetic weights, compiler verification
# ---------------------------------------------------------------------------


def parse_layer_spec(spec: str) -> list[int]:
    """Parse layer selection specs like `'0..47'`, `'0-3,8'`, or `'5'` into sorted unique IDs."""
    text = spec.strip()
    if not text:
        raise ValueError("layer spec must be non-empty")
    layers: set[int] = set()
    for part in text.split(","):
        tok = part.strip()
        if not tok:
            continue
        if ".." in tok:
            lo_s, sep, hi_s = tok.partition("..")
            if not sep or ".." in hi_s:
                raise ValueError(f"invalid layer range {tok!r}")
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid layer range {tok!r}")
            layers.update(range(lo, hi + 1))
        elif "-" in tok:
            lo_s, sep, hi_s = tok.partition("-")
            if not sep or "-" in hi_s:
                raise ValueError(f"invalid layer range {tok!r}")
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid layer range {tok!r}")
            layers.update(range(lo, hi + 1))
        else:
            val = int(tok)
            if val < 0:
                raise ValueError(f"layer index must be non-negative, got {val}")
            layers.add(val)
    if not layers:
        raise ValueError(f"layer spec {spec!r} matched no layers")
    return sorted(layers)


def _synth_weights_f16(rows: int, cols: int, seed: int, scale: float = 0.02) -> bytes:
    """Fast deterministic fp16 weight generator for synthetic CoreML graphs."""
    rng = random.Random(seed)
    # Build a small deterministic tile of 257 fp16 values and stride per row for speed + variety.
    palette_len = 257
    palette = [round((rng.random() * 2.0 - 1.0) * scale, 5) for _ in range(palette_len)]
    row_packer = struct.Struct(f"<{cols}e")
    out = bytearray(rows * cols * 2)
    row_bytes = cols * 2
    for r in range(rows):
        base = (r * 31 + seed) % palette_len
        vals = [palette[(base + c * 7) % palette_len] for c in range(cols)]
        row_packer.pack_into(out, r * row_bytes, *vals)
    return bytes(out)


def _find_amf_tensor(reader: AmfReader, candidates: Sequence[str]) -> PackedTensorMeta:
    for name in candidates:
        meta = reader.by_name.get(name)
        if meta is not None:
            return meta
    raise KeyError(f"none of {list(candidates)!r} found in {reader.path}")


def _pad_rows_f16(weights_f16: bytes, rows: int, cols: int, padded_rows: int) -> bytes:
    if padded_rows < rows:
        raise ValueError(f"padded_rows {padded_rows} < rows {rows}")
    if padded_rows == rows:
        return weights_f16
    return weights_f16 + bytes((padded_rows - rows) * cols * 2)


def _tensor_shape_for_op(op: str, batch_or_seq: int, channels: int) -> tuple[int, ...]:
    if op == "inner-product":
        return (batch_or_seq, channels)
    if op == "conv1x1":
        return (channels, 1, batch_or_seq)
    raise ValueError(f"unsupported CoreML op {op!r}")


def compile_mlpackage(pkg_path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Compile an `.mlpackage` with `xcrun coremlcompiler compile` and return the `.mlmodelc` path."""
    pkg = pathlib.Path(pkg_path)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg), str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"coremlcompiler failed (rc={proc.returncode}) on {pkg}:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    compiled = out / f"{pkg.stem}.mlmodelc"
    if not compiled.is_dir():
        raise RuntimeError(f"coremlcompiler succeeded but {compiled} was not created")
    return compiled


# ---------------------------------------------------------------------------
# Kind 1: expert-predictor (A2)
# ---------------------------------------------------------------------------


def export_expert_predictor(
    output_path: pathlib.Path,
    *,
    amf_path: pathlib.Path | None = None,
    layers_spec: str | None = None,
    base_layer: int | None = None,
    lookahead: int = 2,
    batch_size: int = 32,
    op: str = "inner-product",
    pack_all_layers: bool = False,
    hidden_size: int = 2048,
    num_experts: int = 128,
    seed: int = 42,
    model_id: str | None = None,
) -> CoreMlExportResult:
    """Export a cross-layer MoE router predictor `.mlpackage` (or directory of window bundles).

    - When `output_path` ends with `.mlpackage`, writes one CoreML bundle:
      - If `pack_all_layers=True`, stacks every router in `layers_spec` into
        `[M, len(selected_layers) * num_experts]`.
      - Else if `base_layer` is set, stacks target routers
        `[base_layer + 1 .. base_layer + lookahead]`.
      - Else if `len(selected_layers) <= lookahead`, stacks `selected_layers`.
      - Else stacks the first `lookahead` upcoming layers after
        `selected_layers[0]` (`selected_layers[1 : 1 + lookahead]`), matching
        `[M, lookahead * num_experts]`.
    - When `output_path` does NOT end with `.mlpackage`, treats it as a
      directory and emits one `.mlpackage` per base layer across `layers_spec`,
      plus `predictor_manifest.json`.
    """
    if op not in COREML_OPS:
        raise ValueError(f"op must be one of {COREML_OPS}, got {op!r}")
    if lookahead <= 0:
        raise ValueError(f"lookahead must be positive, got {lookahead}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if op == "conv1x1" and batch_size % IOSURFACE_F16_ROW_ALIGN != 0:
        raise ValueError(
            f"conv1x1 requires batch_size divisible by {IOSURFACE_F16_ROW_ALIGN}, got {batch_size}"
        )

    out = pathlib.Path(output_path)

    def _emit_single(
        pkg_path: pathlib.Path,
        target_layers: list[int],
        c_in: int,
        n_exp: int,
        stacked_f16: bytes,
        eff_model_id: str,
        eff_base_layer: int | None,
    ) -> CoreMlExportResult:
        raw_out = len(target_layers) * n_exp
        c_out = _align_up(raw_out, IOSURFACE_F16_ROW_ALIGN) if op == "inner-product" else raw_out
        if op == "inner-product" and c_in % IOSURFACE_F16_ROW_ALIGN != 0:
            raise ValueError(
                f"inner-product requires hidden_size divisible by {IOSURFACE_F16_ROW_ALIGN}, got {c_in}"
            )
        padded_w = _pad_rows_f16(stacked_f16, raw_out, c_in, c_out)

        in_spec = TensorSpec(
            name="hidden_in",
            shape=_tensor_shape_for_op(op, batch_size, c_in),
            description="Normalized hidden states [M, hidden_size] (fp16)",
        )
        out_spec = TensorSpec(
            name="router_logits",
            shape=_tensor_shape_for_op(op, batch_size, c_out),
            description=f"Predicted router logits for layers {target_layers} (fp16)",
        )
        layer_msg = pb_linear_layer(op, "router_predictor", "hidden_in", "router_logits", c_in, c_out, padded_w)
        meta = {
            "aminfer_kind": "expert-predictor",
            "aminfer_op": op,
            "model_id": eff_model_id,
            "target_layers": ",".join(str(l) for l in target_layers),
            "lookahead": str(len(target_layers)),
            "num_experts": str(n_exp),
            "hidden_size": str(c_in),
            "batch_size": str(batch_size),
            "unpadded_out_dim": str(raw_out),
        }
        if eff_base_layer is not None:
            meta["base_layer"] = str(eff_base_layer)
        short_desc = (
            f"aminfer A2 MoE expert predictor ({eff_model_id}, target layers {target_layers}, "
            f"lookahead={len(target_layers)}, experts={n_exp})"
        )
        model_bytes = build_neural_network_model(
            [in_spec],
            [out_spec],
            [layer_msg],
            use_exact_mapping=(op == "inner-product"),
            short_description=short_desc,
            user_defined=meta,
        )
        write_mlpackage(
            pkg_path,
            model_bytes,
            [in_spec],
            [out_spec],
            short_description=short_desc,
            user_defined=meta,
        )
        return CoreMlExportResult(
            output_path=pkg_path,
            kind="expert-predictor",
            op=op,
            model_id=eff_model_id,
            inputs=(in_spec,),
            outputs=(out_spec,),
            layer_count=1,
            weight_bytes=len(padded_w),
            model_bytes=len(model_bytes),
            metadata=meta,
            bundle_paths=(pkg_path,),
        )

    if amf_path is not None:
        with AmfReader(amf_path) as reader:
            eff_model_id = model_id or reader.header.model_id or pathlib.Path(amf_path).stem
            routers: dict[int, PackedTensorMeta] = {}
            for entry in reader.entries:
                if entry.kind == TensorKind.ROUTER and entry.layer != NO_LAYER:
                    routers[entry.layer] = entry
            if not routers:
                raise ValueError(f"no MoE router tensors found in {amf_path}")
            available = sorted(routers.keys())
            first_shape = routers[available[0]].shape
            if len(first_shape) != 2:
                raise ValueError(f"expected 2D router weight, got {first_shape}")
            n_exp, c_in = first_shape
            selected = parse_layer_spec(layers_spec) if layers_spec else available
            for lid in selected:
                if lid not in routers:
                    raise ValueError(f"requested router layer {lid} not present in {amf_path}")

            def _load_stacked(layer_ids: Sequence[int]) -> bytes:
                chunks: list[bytes] = []
                for lid in layer_ids:
                    r_meta = routers[lid]
                    if r_meta.shape != (n_exp, c_in):
                        raise ValueError(f"router shape mismatch at layer {lid}: {r_meta.shape} != {(n_exp, c_in)}")
                    chunks.append(reader.read_tensor_f16_bytes(r_meta))
                return b"".join(chunks)

            if out.suffix == ".mlpackage":
                if pack_all_layers:
                    targets = list(selected)
                    eff_base = None
                elif base_layer is not None:
                    targets = [l for l in available if base_layer < l <= base_layer + lookahead]
                    if not targets:
                        raise ValueError(
                            f"base_layer={base_layer} with lookahead={lookahead} has no target layers in {available[0]}..{available[-1]}"
                        )
                    eff_base = base_layer
                elif len(selected) <= lookahead:
                    targets = list(selected)
                    eff_base = max(0, selected[0] - 1) if selected[0] > 0 else None
                else:
                    targets = selected[1 : 1 + lookahead]
                    eff_base = selected[0]
                return _emit_single(out, targets, c_in, n_exp, _load_stacked(targets), eff_model_id, eff_base)

            # Directory mode: emit one .mlpackage per base layer with available lookahead targets.
            out.mkdir(parents=True, exist_ok=True)
            manifest_entries: list[dict[str, object]] = []
            bundle_paths: list[pathlib.Path] = []
            total_w = 0
            total_m = 0
            last_res: CoreMlExportResult | None = None
            selected_set = set(selected)
            for b_layer in selected:
                targets = [l for l in range(b_layer + 1, b_layer + 1 + lookahead) if l in selected_set]
                if not targets:
                    continue
                pkg_path = out / f"predictor_l{b_layer:02d}_la{len(targets)}.mlpackage"
                res = _emit_single(pkg_path, targets, c_in, n_exp, _load_stacked(targets), eff_model_id, b_layer)
                last_res = res
                bundle_paths.append(pkg_path)
                total_w += res.weight_bytes
                total_m += res.model_bytes
                manifest_entries.append(
                    {
                        "base_layer": b_layer,
                        "target_layers": targets,
                        "package": pkg_path.name,
                        "inputs": [t.to_dict() for t in res.inputs],
                        "outputs": [t.to_dict() for t in res.outputs],
                    }
                )
            if last_res is None:
                raise ValueError(f"layer selection {selected} produced no lookahead windows")
            (out / "predictor_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "expert-predictor",
                        "model_id": eff_model_id,
                        "lookahead": lookahead,
                        "num_experts": n_exp,
                        "hidden_size": c_in,
                        "batch_size": batch_size,
                        "op": op,
                        "windows": manifest_entries,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return CoreMlExportResult(
                output_path=out,
                kind="expert-predictor",
                op=op,
                model_id=eff_model_id,
                inputs=last_res.inputs,
                outputs=last_res.outputs,
                layer_count=len(bundle_paths),
                weight_bytes=total_w,
                model_bytes=total_m,
                metadata={
                    "aminfer_kind": "expert-predictor",
                    "model_id": eff_model_id,
                    "windows": str(len(bundle_paths)),
                    "lookahead": str(lookahead),
                },
                bundle_paths=tuple(bundle_paths),
            )

    # Synthetic expert-predictor mode (no --amf).
    eff_model_id = model_id or "synthetic-moe-predictor"
    selected = parse_layer_spec(layers_spec) if layers_spec else list(range(1, lookahead + 1))
    if out.suffix != ".mlpackage":
        out = out / "expert_predictor.mlpackage"
    if pack_all_layers or len(selected) <= lookahead:
        targets = list(selected)
        eff_base = base_layer
    else:
        targets = selected[1 : 1 + lookahead]
        eff_base = selected[0] if base_layer is None else base_layer
    stacked = b"".join(_synth_weights_f16(num_experts, hidden_size, seed + l * 997) for l in targets)
    return _emit_single(out, targets, hidden_size, num_experts, stacked, eff_model_id, eff_base)


# ---------------------------------------------------------------------------
# Kind 2: attention-qkv (A1)
# ---------------------------------------------------------------------------


def export_attention_qkv(
    output_path: pathlib.Path,
    *,
    amf_path: pathlib.Path | None = None,
    layer: int = 0,
    seq_len: int = 512,
    op: str = "inner-product",
    include_o_proj: bool = False,
    hidden_size: int = 2048,
    q_dim: int = 4096,
    kv_dim: int = 512,
    seed: int = 42,
    model_id: str | None = None,
) -> CoreMlExportResult:
    """Export a chunked-prefill attention QKV (and optional O) projection `.mlpackage` (`A1`)."""
    if op not in COREML_OPS:
        raise ValueError(f"op must be one of {COREML_OPS}, got {op!r}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if op == "conv1x1" and seq_len % IOSURFACE_F16_ROW_ALIGN != 0:
        raise ValueError(f"conv1x1 requires seq_len divisible by {IOSURFACE_F16_ROW_ALIGN}, got {seq_len}")

    out = pathlib.Path(output_path)
    if out.suffix != ".mlpackage":
        out = out / f"attention_qkv_l{layer:02d}.mlpackage"

    if amf_path is not None:
        with AmfReader(amf_path) as reader:
            eff_model_id = model_id or reader.header.model_id or pathlib.Path(amf_path).stem
            q_meta = _find_amf_tensor(
                reader,
                (f"layers.{layer}.self_attn.q_proj", f"layers.{layer}.self_attn.q_proj.weight"),
            )
            k_meta = _find_amf_tensor(
                reader,
                (f"layers.{layer}.self_attn.k_proj", f"layers.{layer}.self_attn.k_proj.weight"),
            )
            v_meta = _find_amf_tensor(
                reader,
                (f"layers.{layer}.self_attn.v_proj", f"layers.{layer}.self_attn.v_proj.weight"),
            )
            if len(q_meta.shape) != 2 or len(k_meta.shape) != 2 or len(v_meta.shape) != 2:
                raise ValueError(f"Q/K/V weights must be 2D at layer {layer}")
            eff_q_dim, eff_hidden = q_meta.shape
            eff_kv_dim, k_in = k_meta.shape
            v_out_dim, v_in = v_meta.shape
            if k_in != eff_hidden or v_in != eff_hidden or v_out_dim != eff_kv_dim:
                raise ValueError(
                    f"inconsistent QKV shapes at layer {layer}: Q={q_meta.shape}, K={k_meta.shape}, V={v_meta.shape}"
                )
            w_q = reader.read_tensor_f16_bytes(q_meta)
            w_k = reader.read_tensor_f16_bytes(k_meta)
            w_v = reader.read_tensor_f16_bytes(v_meta)
            w_o: bytes | None = None
            eff_o_out = eff_hidden
            if include_o_proj:
                o_meta = _find_amf_tensor(
                    reader,
                    (f"layers.{layer}.self_attn.o_proj", f"layers.{layer}.self_attn.o_proj.weight"),
                )
                if len(o_meta.shape) != 2 or o_meta.shape[1] != eff_q_dim:
                    raise ValueError(f"o_proj shape {o_meta.shape} does not match q_dim {eff_q_dim}")
                eff_o_out = o_meta.shape[0]
                w_o = reader.read_tensor_f16_bytes(o_meta)
    else:
        eff_model_id = model_id or "synthetic-attention-qkv"
        eff_hidden = hidden_size
        eff_q_dim = q_dim
        eff_kv_dim = kv_dim
        eff_o_out = hidden_size
        w_q = _synth_weights_f16(eff_q_dim, eff_hidden, seed + 1)
        w_k = _synth_weights_f16(eff_kv_dim, eff_hidden, seed + 2)
        w_v = _synth_weights_f16(eff_kv_dim, eff_hidden, seed + 3)
        w_o = _synth_weights_f16(eff_o_out, eff_q_dim, seed + 4) if include_o_proj else None

    inputs = [
        TensorSpec(
            name="hidden_in",
            shape=_tensor_shape_for_op(op, seq_len, eff_hidden),
            description="Post-layernorm chunk activations [seq_len, hidden_size] (fp16)",
        )
    ]
    outputs = [
        TensorSpec(
            name="q_out",
            shape=_tensor_shape_for_op(op, seq_len, eff_q_dim),
            description="Projected query chunk (fp16)",
        ),
        TensorSpec(
            name="k_out",
            shape=_tensor_shape_for_op(op, seq_len, eff_kv_dim),
            description="Projected key chunk for IOSurface KvTile (fp16)",
        ),
        TensorSpec(
            name="v_out",
            shape=_tensor_shape_for_op(op, seq_len, eff_kv_dim),
            description="Projected value chunk for IOSurface KvTile (fp16)",
        ),
    ]
    layers = [
        pb_linear_layer(op, f"l{layer}_q_proj", "hidden_in", "q_out", eff_hidden, eff_q_dim, w_q),
        pb_linear_layer(op, f"l{layer}_k_proj", "hidden_in", "k_out", eff_hidden, eff_kv_dim, w_k),
        pb_linear_layer(op, f"l{layer}_v_proj", "hidden_in", "v_out", eff_hidden, eff_kv_dim, w_v),
    ]
    weight_bytes = len(w_q) + len(w_k) + len(w_v)

    if include_o_proj and w_o is not None:
        inputs.append(
            TensorSpec(
                name="attn_in",
                shape=_tensor_shape_for_op(op, seq_len, eff_q_dim),
                description="Attention context input for output projection (fp16)",
            )
        )
        outputs.append(
            TensorSpec(
                name="o_out",
                shape=_tensor_shape_for_op(op, seq_len, eff_o_out),
                description="Attention output projection result (fp16)",
            )
        )
        layers.append(pb_linear_layer(op, f"l{layer}_o_proj", "attn_in", "o_out", eff_q_dim, eff_o_out, w_o))
        weight_bytes += len(w_o)

    meta = {
        "aminfer_kind": "attention-qkv",
        "aminfer_op": op,
        "model_id": eff_model_id,
        "layer": str(layer),
        "seq_len": str(seq_len),
        "hidden_size": str(eff_hidden),
        "q_dim": str(eff_q_dim),
        "kv_dim": str(eff_kv_dim),
        "include_o_proj": "true" if include_o_proj else "false",
    }
    short_desc = (
        f"aminfer A1 chunked attention QKV ({eff_model_id}, layer={layer}, "
        f"seq={seq_len}, hidden={eff_hidden}, q={eff_q_dim}, kv={eff_kv_dim})"
    )
    model_bytes = build_neural_network_model(
        inputs,
        outputs,
        layers,
        use_exact_mapping=(op == "inner-product"),
        short_description=short_desc,
        user_defined=meta,
    )
    write_mlpackage(
        out,
        model_bytes,
        inputs,
        outputs,
        short_description=short_desc,
        user_defined=meta,
    )
    return CoreMlExportResult(
        output_path=out,
        kind="attention-qkv",
        op=op,
        model_id=eff_model_id,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        layer_count=len(layers),
        weight_bytes=weight_bytes,
        model_bytes=len(model_bytes),
        metadata=meta,
        bundle_paths=(out,),
    )


# ---------------------------------------------------------------------------
# Kind 3: draft-head (D4 / D5)
# ---------------------------------------------------------------------------


def export_draft_head(
    output_path: pathlib.Path,
    *,
    amf_path: pathlib.Path | None = None,
    layer: int = 0,
    stage: str = "ffn",
    batch_size: int = 32,
    op: str = "inner-product",
    hidden_size: int = 1024,
    intermediate_size: int = 3072,
    out_dim: int | None = None,
    seed: int = 42,
    model_id: str | None = None,
) -> CoreMlExportResult:
    """Export a draft SwiGLU FFN or projection `.mlpackage` (`D4`/`D5`)."""
    if op not in COREML_OPS:
        raise ValueError(f"op must be one of {COREML_OPS}, got {op!r}")
    if stage not in DRAFT_STAGES:
        raise ValueError(f"stage must be one of {DRAFT_STAGES}, got {stage!r}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if op == "conv1x1" and batch_size % IOSURFACE_F16_ROW_ALIGN != 0:
        raise ValueError(f"conv1x1 requires batch_size divisible by {IOSURFACE_F16_ROW_ALIGN}, got {batch_size}")

    out = pathlib.Path(output_path)
    if out.suffix != ".mlpackage":
        out = out / f"draft_head_{stage}_l{layer:02d}.mlpackage"

    if stage == "ffn":
        if amf_path is not None:
            with AmfReader(amf_path) as reader:
                eff_model_id = model_id or reader.header.model_id or pathlib.Path(amf_path).stem
                g_meta = _find_amf_tensor(
                    reader,
                    (f"layers.{layer}.mlp.gate_proj", f"layers.{layer}.mlp.gate_proj.weight"),
                )
                u_meta = _find_amf_tensor(
                    reader,
                    (f"layers.{layer}.mlp.up_proj", f"layers.{layer}.mlp.up_proj.weight"),
                )
                d_meta = _find_amf_tensor(
                    reader,
                    (f"layers.{layer}.mlp.down_proj", f"layers.{layer}.mlp.down_proj.weight"),
                )
                eff_inter, eff_hidden = g_meta.shape
                if u_meta.shape != (eff_inter, eff_hidden) or d_meta.shape != (eff_hidden, eff_inter):
                    raise ValueError(
                        f"inconsistent FFN shapes at layer {layer}: gate={g_meta.shape}, "
                        f"up={u_meta.shape}, down={d_meta.shape}"
                    )
                w_gate = reader.read_tensor_f16_bytes(g_meta)
                w_up = reader.read_tensor_f16_bytes(u_meta)
                w_down = reader.read_tensor_f16_bytes(d_meta)
        else:
            eff_model_id = model_id or "synthetic-draft-ffn"
            eff_hidden = hidden_size
            eff_inter = intermediate_size
            w_gate = _synth_weights_f16(eff_inter, eff_hidden, seed + 11)
            w_up = _synth_weights_f16(eff_inter, eff_hidden, seed + 12)
            w_down = _synth_weights_f16(eff_hidden, eff_inter, seed + 13)

        in_spec = TensorSpec(
            name="hidden_in",
            shape=_tensor_shape_for_op(op, batch_size, eff_hidden),
            description="Draft FFN normalized input [M, hidden_size] (fp16)",
        )
        out_spec = TensorSpec(
            name="ffn_out",
            shape=_tensor_shape_for_op(op, batch_size, eff_hidden),
            description="Draft SwiGLU FFN output [M, hidden_size] (fp16)",
        )
        layers = [
            pb_linear_layer(op, "gate_proj", "hidden_in", "gate_pre", eff_hidden, eff_inter, w_gate),
            pb_linear_layer(op, "up_proj", "hidden_in", "up_pre", eff_hidden, eff_inter, w_up),
            pb_sigmoid_layer("gate_sig", "gate_pre", "gate_sig"),
            pb_multiply_layer("gate_silu", ("gate_pre", "gate_sig"), "gate_act"),
            pb_multiply_layer("swiglu_mul", ("gate_act", "up_pre"), "ffn_mid"),
            pb_linear_layer(op, "down_proj", "ffn_mid", "ffn_out", eff_inter, eff_hidden, w_down),
        ]
        weight_bytes = len(w_gate) + len(w_up) + len(w_down)
        meta = {
            "aminfer_kind": "draft-head",
            "aminfer_stage": "ffn",
            "aminfer_op": op,
            "model_id": eff_model_id,
            "layer": str(layer),
            "batch_size": str(batch_size),
            "hidden_size": str(eff_hidden),
            "intermediate_size": str(eff_inter),
        }
        short_desc = (
            f"aminfer D4/D5 draft SwiGLU FFN ({eff_model_id}, layer={layer}, "
            f"M={batch_size}, hidden={eff_hidden}, inter={eff_inter})"
        )
        model_bytes = build_neural_network_model(
            [in_spec],
            [out_spec],
            layers,
            use_exact_mapping=(op == "inner-product"),
            short_description=short_desc,
            user_defined=meta,
        )
        write_mlpackage(
            out,
            model_bytes,
            [in_spec],
            [out_spec],
            short_description=short_desc,
            user_defined=meta,
        )
        return CoreMlExportResult(
            output_path=out,
            kind="draft-head",
            op=op,
            model_id=eff_model_id,
            inputs=(in_spec,),
            outputs=(out_spec,),
            layer_count=len(layers),
            weight_bytes=weight_bytes,
            model_bytes=len(model_bytes),
            metadata=meta,
            bundle_paths=(out,),
        )

    # Single-projection stages: EAGLE-3 `fc` fusion or generic `linear` draft head.
    if amf_path is not None:
        with AmfReader(amf_path) as reader:
            eff_model_id = model_id or reader.header.model_id or pathlib.Path(amf_path).stem
            if stage == "fc":
                t_meta = _find_amf_tensor(reader, ("fc.weight", "fc"))
            else:
                t_meta = _find_amf_tensor(
                    reader,
                    (
                        "fc.weight",
                        "fc",
                        f"layers.{layer}.mlp.down_proj.weight",
                        f"layers.{layer}.mlp.down_proj",
                        "lm_head.weight",
                        "lm_head",
                    ),
                )
            if len(t_meta.shape) != 2:
                raise ValueError(f"draft projection tensor {t_meta.name!r} must be 2D, got {t_meta.shape}")
            c_out, c_in = t_meta.shape
            w_proj = reader.read_tensor_f16_bytes(t_meta)
    else:
        eff_model_id = model_id or f"synthetic-draft-{stage}"
        if stage == "fc":
            c_in = hidden_size * 3
            c_out = out_dim if out_dim is not None else hidden_size
        else:
            c_in = hidden_size
            c_out = out_dim if out_dim is not None else hidden_size
        w_proj = _synth_weights_f16(c_out, c_in, seed + 21)

    in_spec = TensorSpec(
        name="hidden_in",
        shape=_tensor_shape_for_op(op, batch_size, c_in),
        description=f"Draft {stage} input [M, {c_in}] (fp16)",
    )
    out_spec = TensorSpec(
        name="proj_out",
        shape=_tensor_shape_for_op(op, batch_size, c_out),
        description=f"Draft {stage} output [M, {c_out}] (fp16)",
    )
    layers = [pb_linear_layer(op, f"draft_{stage}_proj", "hidden_in", "proj_out", c_in, c_out, w_proj)]
    meta = {
        "aminfer_kind": "draft-head",
        "aminfer_stage": stage,
        "aminfer_op": op,
        "model_id": eff_model_id,
        "batch_size": str(batch_size),
        "in_dim": str(c_in),
        "out_dim": str(c_out),
    }
    short_desc = f"aminfer D4/D5 draft {stage} projection ({eff_model_id}, M={batch_size}, {c_in}->{c_out})"
    model_bytes = build_neural_network_model(
        [in_spec],
        [out_spec],
        layers,
        use_exact_mapping=(op == "inner-product"),
        short_description=short_desc,
        user_defined=meta,
    )
    write_mlpackage(
        out,
        model_bytes,
        [in_spec],
        [out_spec],
        short_description=short_desc,
        user_defined=meta,
    )
    return CoreMlExportResult(
        output_path=out,
        kind="draft-head",
        op=op,
        model_id=eff_model_id,
        inputs=(in_spec,),
        outputs=(out_spec,),
        layer_count=1,
        weight_bytes=len(w_proj),
        model_bytes=len(model_bytes),
        metadata=meta,
        bundle_paths=(out,),
    )
