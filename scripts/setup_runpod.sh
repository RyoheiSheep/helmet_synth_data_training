#!/usr/bin/env bash
# Setup script for RunPod pod (1x A40 48GB, container disk >= 50GB).
# Run once after cloning the repo onto the pod.
#
# Usage:
#   git clone <repo-url> ~/helmet && cd ~/helmet
#   bash scripts/setup_runpod.sh
#
# Assumes: RunPod PyTorch template (CUDA 12.4, Python 3.11).
# Pinned versions: torch 2.6.0+cu124 required for diffusers 0.37.1
# (diffusers 0.37 uses `X | None` type annotations in FA3 custom ops
#  which torch < 2.5 cannot parse in infer_schema).
set -euo pipefail

echo "=== [1/4] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/4] Installing project (lightweight deps) ==="
uv sync --dev

echo "=== [3/4] Installing GPU dependencies ==="
# Install torch first with the correct CUDA index, then everything else.
# --no-cache-dir: avoid filling the container disk with wheel cache.
python3 -m pip install --no-cache-dir \
    "torch==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install --no-cache-dir \
    "diffusers==0.37.1" \
    transformers \
    accelerate \
    sentencepiece \
    peft \
    datasets \
    vllm \
    Pillow \
    pyyaml

echo "=== [4/4] Verifying GPU ==="
python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
python3 -c "from diffusers import Flux2KleinPipeline; print('diffusers: Flux2KleinPipeline OK')"

echo ""
echo "Setup complete. Next steps:"
echo "  export HF_TOKEN=<your-token>"
echo "  bash scripts/run_loop.sh 1"
