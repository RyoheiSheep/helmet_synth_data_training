"""Inference script for helmet chinstrap VLM evaluation.

Loads a fine-tuned LoRA adapter (or uses a dummy provider) and produces
predictions JSONL consumable by scripts/evaluate.py.

Output schema (one JSON object per line):
    {"image_id": "...", "label": "tight"|"loose", "ground_truth": "tight"|"loose"}
"""

import argparse
import csv
import json
import random
from pathlib import Path


QUESTION_WITH_RATIONALE = (
    'Is the helmet chinstrap properly fastened?\n'
    'Answer with JSON: {"label": "tight"|"loose", "rationale": "<reason>"}'
)


def load_eval_set(eval_dir: Path) -> list[dict]:
    """Load evaluation images and ground-truth labels.

    Expects:
        eval_dir/labels.csv   — image_id,label
        eval_dir/images/      — {image_id}.png

    Returns:
        List of {"image_id", "label", "image_path"}.
    """
    labels_path = eval_dir / "labels.csv"
    entries = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            label = row["label"]
            if label not in ("tight", "loose"):
                raise ValueError(
                    f"Invalid ground-truth label for {image_id}: {label}"
                )
            image_path = eval_dir / "images" / f"{image_id}.png"
            entries.append({
                "image_id": image_id,
                "label": label,
                "image_path": image_path,
            })
    return entries


def predict_dummy(
    eval_entries: list[dict],
    accuracy: float = 0.8,
    seed: int = 42,
) -> list[dict]:
    """Dummy provider: simulate predictions by flipping ground truth at a configurable error rate.

    Args:
        eval_entries: Output of load_eval_set().
        accuracy: Fraction of correct predictions.
        seed: Random seed for reproducibility.

    Returns:
        Predictions list: [{"image_id", "label", "ground_truth"}, ...].
    """
    rng = random.Random(seed)
    predictions = []
    for entry in eval_entries:
        gt = entry["label"]
        if rng.random() < accuracy:
            pred = gt
        else:
            pred = "loose" if gt == "tight" else "tight"
        predictions.append({
            "image_id": entry["image_id"],
            "label": pred,
            "ground_truth": gt,
        })
    return predictions


def predict_vllm(
    eval_entries: list[dict],
    model_dir: Path,
    base_model: str = "Qwen/Qwen3.5-8B",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> list[dict]:
    """Run inference using vLLM with a LoRA adapter.

    Heavy imports are deferred so tests don't need GPU/vllm.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    label_schema = {
        "type": "object",
        "required": ["label"],
        "properties": {
            "label": {"type": "string", "enum": ["tight", "loose"]},
        },
        "additionalProperties": False,
    }

    lora_dir = model_dir / "lora_weights"
    llm = LLM(
        model=base_model,
        enable_lora=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("helmet_lora", 1, str(lora_dir))

    structured_outputs = StructuredOutputsParams(json=label_schema)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=structured_outputs,
    )

    conversations = []
    for entry in eval_entries:
        messages = [
            {"role": "system", "content": "You are a safety inspection assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"file://{entry['image_path']}"}},
                {"type": "text", "text": QUESTION_WITH_RATIONALE},
            ]},
        ]
        conversations.append(messages)

    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    predictions = []
    for entry, output in zip(eval_entries, outputs):
        text = output.outputs[0].text.strip()
        try:
            parsed = json.loads(text)
            pred_label = parsed["label"]
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: unparseable output for {entry['image_id']}, skipping")
            continue
        predictions.append({
            "image_id": entry["image_id"],
            "label": pred_label,
            "ground_truth": entry["label"],
        })
    return predictions


def run_prediction(
    eval_dir: Path,
    output_path: Path,
    provider: str = "dummy",
    model_dir: Path | None = None,
    base_model: str = "Qwen/Qwen3.5-8B",
    dummy_accuracy: float = 0.8,
    dummy_seed: int = 42,
) -> list[dict]:
    """Run prediction pipeline end-to-end.

    Args:
        eval_dir: Directory with labels.csv + images/.
        output_path: Where to write predictions JSONL.
        provider: "dummy" or "vllm".
        model_dir: Path to models/loop_{N}/ (required for vllm provider).
        base_model: HuggingFace base model ID (vllm provider).
        dummy_accuracy: Simulated accuracy (dummy provider).
        dummy_seed: Random seed (dummy provider).

    Returns:
        The predictions list.
    """
    eval_entries = load_eval_set(eval_dir)

    if provider == "dummy":
        predictions = predict_dummy(
            eval_entries, accuracy=dummy_accuracy, seed=dummy_seed
        )
    elif provider == "vllm":
        if model_dir is None:
            raise ValueError("provider=vllm requires --model-dir")
        predictions = predict_vllm(
            eval_entries, model_dir=model_dir, base_model=base_model
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM inference on eval images"
    )
    parser.add_argument(
        "--eval-dir", type=str, required=True,
        help="Directory with labels.csv + images/",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write predictions JSONL",
    )
    parser.add_argument(
        "--provider", type=str, choices=["dummy", "vllm"], default="dummy",
        help="Inference provider",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to models/loop_{N}/ (vllm provider)",
    )
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen3.5-8B",
        help="Base model ID (vllm provider)",
    )
    parser.add_argument(
        "--dummy-accuracy", type=float, default=0.8,
        help="Simulated accuracy (dummy provider)",
    )
    parser.add_argument(
        "--dummy-seed", type=int, default=42,
        help="Random seed (dummy provider)",
    )
    args = parser.parse_args()

    predictions = run_prediction(
        eval_dir=Path(args.eval_dir),
        output_path=Path(args.output),
        provider=args.provider,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        base_model=args.base_model,
        dummy_accuracy=args.dummy_accuracy,
        dummy_seed=args.dummy_seed,
    )

    correct = sum(1 for p in predictions if p["label"] == p["ground_truth"])
    total = len(predictions)
    print(f"Predictions: {correct}/{total} correct ({correct/total:.2%})")


if __name__ == "__main__":
    main()
