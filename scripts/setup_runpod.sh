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

echo "=== [2/5] Installing project (lightweight deps + data fetcher) ==="
uv sync --dev --extra data

echo "=== [3/5] Installing GPU dependencies ==="

# 念のため壊れたcudnn削除（重要）
rm -rf /usr/local/lib/python3.11/dist-packages/~vidia* || true

# torch (CUDA 12.4での最大安定)
python3 -m pip install --no-cache-dir \
    "torch==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# diffusersは安定版に下げる
python3 -m pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    sentencepiece \
    peft \
    datasets \
    vllm \
    Pillow \
    pyyaml
python3 -m pip install --upgrade transformers
echo "=== [4/5] Verifying GPU ==="
python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ここ重要：pipeline import確認
python3 - << 'EOF'
from diffusers import DiffusionPipeline
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