#!/usr/bin/env bash
# Setup script for RunPod pod (1x A40 48GB).
# Run once after cloning the repo onto the pod.
#
# Usage:
#   git clone <repo-url> ~/helmet && cd ~/helmet
#   bash scripts/setup_runpod.sh
#
# Assumes: RunPod PyTorch template (CUDA 12.x + Python 3.11+ pre-installed).
set -euo pipefail

echo "=== [1/4] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/4] Installing project (lightweight deps) ==="
uv sync --dev

echo "=== [3/4] Installing GPU dependencies ==="
# Single pip install for all steps (A, B, D, predict).
# Running directly on the pod — no Docker-in-Docker.
# --no-cache-dir: avoid filling the container disk with wheel cache
pip install --no-cache-dir \
    torch \
    "git+https://github.com/huggingface/diffusers.git" \
    transformers \
    accelerate \
    sentencepiece \
    peft \
    datasets \
    vllm \
    Pillow \
    pyyaml

echo "=== [4/4] Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Setup complete. Next steps:"
echo "  export HF_TOKEN=<your-token>"
echo "  bash scripts/run_loop.sh 1"
