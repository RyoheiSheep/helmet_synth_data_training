#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/4] Installing project (lightweight deps) ==="
uv sync --dev

echo "=== [3/4] Installing GPU dependencies ==="

# 念のため壊れたcudnn削除（重要）
rm -rf /usr/local/lib/python3.11/dist-packages/~vidia* || true

# torch (CUDA 12.4での最大安定)
python3 -m pip install --no-cache-dir \
    "torch==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# diffusersは安定版に下げる
python3 -m pip install --no-cache-dir \
    "diffusers==0.35.1" \
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

# ここ重要：pipeline import確認
python3 - << 'EOF'
from diffusers import DiffusionPipeline
print("diffusers import OK")
EOF

echo ""
echo "Setup complete."