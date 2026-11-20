# aminfer-tools

Offline Python toolchain for [aminfer](../README.md), managed by `uv`.

Python never runs inside the inference runtime
([ADR-0001](../docs/decisions/0001-language-and-toolchain-boundaries.md)).
This package exists strictly for offline preparation:

- packing 16 KiB page-aligned `.amf` weight and cold-expert files
  ([ADR-0006](../docs/decisions/0006-amf-cold-experts-direct-io.md));
- collecting offline MoE expert heat statistics;
- exporting CoreML `.mlpackage` graphs for the Apple Neural Engine
  ([ADR-0010](../docs/decisions/0010-hybrid-wavefront-prefill.md)).

Capability status for each tool is tracked in
[../docs/status.md](../docs/status.md).

## Usage

```sh
cd tools
uv sync
uv run aminfer-tools --help
uv run aminfer-tools pack-amf --input model.safetensors --output model.amf --model-id qwen3
uv run python -m unittest discover -s tests
```
