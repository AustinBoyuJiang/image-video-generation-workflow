#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r requirements-app.txt

if [ ! -d h3-comfy/.git ]; then
  mkdir -p h3-comfy
  git -C h3-comfy init
  git -C h3-comfy remote add origin https://github.com/comfyanonymous/ComfyUI.git
fi
COMFY_REVISION="$(.venv/bin/python -c 'import json; print(json.load(open("model-lock.json"))["comfyui"]["revision"])')"
if [ "$(git -C h3-comfy rev-parse HEAD 2>/dev/null || true)" != "$COMFY_REVISION" ]; then
  git -C h3-comfy fetch --depth=1 origin "$COMFY_REVISION"
  git -C h3-comfy checkout --detach FETCH_HEAD
fi
install -m 0644 custom_nodes/studio_ram_bridge.py h3-comfy/custom_nodes/studio_ram_bridge.py

"$PYTHON_BIN" -m venv h3-venv
h3-venv/bin/python -m pip install --upgrade pip
h3-venv/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "$TORCH_INDEX_URL"
h3-venv/bin/python -m pip install -r h3-comfy/requirements.txt -r requirements-h3.txt

printf 'Dependencies ready. Download models with: .venv/bin/python download_models.py\n'
