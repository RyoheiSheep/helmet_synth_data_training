#!/usr/bin/env bash
#
# RunPod setup: install uv, project deps, GPU stack, then optionally fetch
# the helmet dataset from Cloudflare R2.
#
# To fetch data automatically during setup, export these BEFORE running:
#   CLOUDFLARE_R2_ACCOUNT_ID
#   CLOUDFLARE_R2_ACCESS_KEY_ID
#   CLOUDFLARE_R2_SECRET_ACCESS_KEY
#   CLOUDFLARE_R2_BUCKET
# If any are unset, step [5/5] is skipped and a manual command is printed.
#
# Manual fetch later:
#   uv run python scripts/fetch_data.py --split both
#
set -euo pipefail

echo "=== [1/5] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Persist for future shells (the curl installer only edits the profile on some
# shells; make it explicit so `uv` is on PATH after this script exits).
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

echo "=== [2/5] Installing project (lightweight deps + data fetcher) ==="
uv sync --dev --extra data

echo "=== [3/5] Installing GPU dependencies ==="
# Single pip install so the resolver picks a coherent torch+vllm+transformers
# combination. Splitting this across multiple `pip install` calls lets a later
# package (vllm) silently upgrade torch to a cu130 wheel that does not match
# the RunPod host driver (driver 12.8 -> needs cu128 or older).
python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1" \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    sentencepiece \
    peft \
    datasets \
    vllm \
    Pillow \
    pyyaml

echo "=== [4/5] Verifying GPU ==="
python3 - << 'EOF'
import torch
from diffusers import DiffusionPipeline  # noqa: F401  -- smoke-test import
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("diffusers import OK")
EOF

echo "=== [5/5] Fetching dataset from Cloudflare R2 ==="
if [[ -n "${CLOUDFLARE_R2_ACCOUNT_ID:-}" \
   && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" \
   && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" \
   && -n "${CLOUDFLARE_R2_BUCKET:-}" ]]; then
    uv run python scripts/fetch_data.py --split both
else
    echo "Skipped: R2 credentials not set in environment."
    echo "To fetch later, export the four CLOUDFLARE_R2_* vars (see top of this script) and run:"
    echo "  uv run python scripts/fetch_data.py --split both"
fi

echo ""
echo "Setup complete."
