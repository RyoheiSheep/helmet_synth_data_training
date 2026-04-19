#!/usr/bin/env bash
# Run a full pipeline loop (A -> B -> C -> D -> Predict -> Evaluate).
# Designed for RunPod where all deps are installed directly (no Docker).
#
# Usage:
#   export HF_TOKEN=<your-token>
#   bash scripts/run_loop.sh 1          # Loop 1
#   bash scripts/run_loop.sh 2          # Loop 2 (compares with Loop 1)
set -euo pipefail

LOOP=${1:?Usage: run_loop.sh <loop_number>}
PREV_LOOP=$((LOOP - 1))
BASE_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('config/step_d.yaml')); print(c['base_model'])")
STUDENT_BATCH_SIZE=${STUDENT_BATCH_SIZE:-4}

echo "============================================"
echo "  Loop ${LOOP} — full pipeline"
echo "============================================"

# ── Step A: Image Generation ──────────────────────────────────────────
echo ""
echo "=== Step A: Image Generation (loop ${LOOP}) ==="
python3 docker/step_a_imagegen/generate.py \
    --config config/step_a.yaml \
    --seeds-dir seeds \
    --output-dir "generated/loop_${LOOP}"

# ── Step B: Screening ────────────────────────────────────────────────
echo ""
echo "=== Step B: Screening (loop ${LOOP}) ==="
if [ "${LOOP}" -eq 1 ]; then
    echo "(loop 1: passthrough — inheriting seed labels, no VLM filter)"
    python3 docker/step_b_screening/screen_and_label.py \
        --loop "${LOOP}" \
        --provider passthrough \
        --generated-dir generated \
        --output-dir screened
elif [ -d "models/loop_${PREV_LOOP}/lora_weights" ]; then
    echo "(using Student model from loop ${PREV_LOOP} as filter)"
    python3 docker/step_b_screening/screen_and_label.py \
        --loop "${LOOP}" \
        --provider student \
        --model-dir "models/loop_${PREV_LOOP}" \
        --base-model "${BASE_MODEL}" \
        --batch-size "${STUDENT_BATCH_SIZE}" \
        --generated-dir generated \
        --output-dir screened
else
    echo "(using vLLM Teacher model — Student from previous loop not found)"
    python3 docker/step_b_screening/screen_and_label.py \
        --loop "${LOOP}" \
        --provider vllm \
        --config config/step_b.yaml \
        --generated-dir generated \
        --output-dir screened
fi

# ── Step C: Dataset Build (accumulate all prior loops) ────────────────
echo ""
echo "=== Step C: Dataset Build (loop ${LOOP}, accumulate) ==="
python3 scripts/build_dataset.py \
    --loop "${LOOP}" \
    --accumulate \
    --screened-dir screened \
    --generated-dir generated \
    --output-dir dataset

# ── Step D: Fine-tuning ──────────────────────────────────────────────
echo ""
echo "=== Step D: Fine-tuning (loop ${LOOP}) ==="
python3 docker/step_d_finetune/finetune.py \
    --loop "${LOOP}" \
    --config config/step_d.yaml \
    --dataset-dir dataset \
    --output-dir models

# ── Predict on test_real ──────────────────────────────────────────────
echo ""
echo "=== Predict: test_real (loop ${LOOP}) ==="
mkdir -p "eval_out/loop_${LOOP}"
python3 scripts/predict.py \
    --eval-dir eval/test_real \
    --output "eval_out/loop_${LOOP}/test_real.jsonl" \
    --provider vllm \
    --model-dir "models/loop_${LOOP}" \
    --base-model "${BASE_MODEL}"

# ── Predict on edge_cases ─────────────────────────────────────────────
echo ""
echo "=== Predict: edge_cases (loop ${LOOP}) ==="
python3 scripts/predict.py \
    --eval-dir eval/edge_cases \
    --output "eval_out/loop_${LOOP}/edge_cases.jsonl" \
    --provider vllm \
    --model-dir "models/loop_${LOOP}" \
    --base-model "${BASE_MODEL}"

# ── Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "=== Evaluate: test_real (loop ${LOOP}) ==="
EVAL_ARGS=(
    --predictions "eval_out/loop_${LOOP}/test_real.jsonl"
    --output "eval_out/loop_${LOOP}/metrics.json"
)
if [ "${PREV_LOOP}" -ge 1 ] && [ -f "eval_out/loop_${PREV_LOOP}/test_real.jsonl" ]; then
    EVAL_ARGS+=(--prev-predictions "eval_out/loop_${PREV_LOOP}/test_real.jsonl")
    echo "(comparing with loop ${PREV_LOOP})"
fi
python3 scripts/evaluate.py "${EVAL_ARGS[@]}"

echo ""
echo "=== Evaluate: edge_cases (loop ${LOOP}) ==="
python3 scripts/evaluate.py \
    --predictions "eval_out/loop_${LOOP}/edge_cases.jsonl" \
    --output "eval_out/loop_${LOOP}/edge_metrics.json"

echo ""
echo "============================================"
echo "  Loop ${LOOP} complete!"
echo "  Results: eval_out/loop_${LOOP}/metrics.json"
echo "============================================"
