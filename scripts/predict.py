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


def predict_transformers(
    eval_entries: list[dict],
    model_dir: Path,
    base_model: str = "Qwen/Qwen3.5-9B",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> list[dict]:
    """Run inference using HuggingFace Transformers + PEFT LoRA adapter.

    Mirrors the loading path used in [docker/step_d_finetune/finetune.py] so the
    adapter is consumed exactly the way it was trained. Use this provider when
    vLLM does not register the base model architecture.

    Performance:
      - LoRA is merged into the base weights (`merge_and_unload`) to remove
        per-step PEFT overhead.
      - Flash Attention 2 is enabled when available; falls back silently.
      - Inputs are processed in batches with left-padding so all rows can be
        decoded with one slice.
      - `temperature=0` selects greedy decoding (fastest, deterministic).

    Heavy imports are deferred so tests don't need GPU/torch.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    lora_dir = model_dir / "lora_weights"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[predict] WARNING: no CUDA device detected, running on CPU.")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    # Left padding is required so all sequences in a batch align at the right
    # edge — then `output_ids[:, prompt_len:]` slices new tokens for every row.
    processor.tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("[predict] using flash_attention_2")
    except (ValueError, ImportError, RuntimeError) as e:
        print(f"[predict] flash_attention_2 unavailable ({e}); using default attention")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, str(lora_dir))
    # Collapse LoRA matrices into the base weights so generate() runs without
    # PEFT's per-layer hook overhead.
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    predictions: list[dict] = []
    for start in range(0, len(eval_entries), batch_size):
        batch = eval_entries[start:start + batch_size]
        images = [Image.open(e["image_path"]).convert("RGB") for e in batch]
        texts = [
            processor.apply_chat_template(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_WITH_RATIONALE},
                    ],
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _ in batch
        ]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        for entry, text_out in zip(batch, decoded):
            text_out = text_out.strip()
            try:
                parsed = json.loads(text_out)
                pred_label = parsed["label"]
                if pred_label not in ("tight", "loose"):
                    raise ValueError(f"invalid label: {pred_label!r}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(
                    f"Warning: unparseable output for {entry['image_id']} "
                    f"({e}): {text_out!r}"
                )
                continue
            predictions.append({
                "image_id": entry["image_id"],
                "label": pred_label,
                "ground_truth": entry["label"],
            })
    return predictions


def predict_vllm(
    eval_entries: list[dict],
    model_dir: Path,
    base_model: str = "Qwen/Qwen3.5-9B",
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

    # Collect the eval image directory so we can allowlist it for vLLM's
    # local-file security check (required since vLLM 0.19).
    image_paths = [Path(e["image_path"]).resolve() for e in eval_entries]
    image_dirs = {p.parent for p in image_paths}
    # All eval images live under one directory; use the common parent.
    allowed_media_path = str(image_paths[0].parent) if len(image_dirs) == 1 else str(image_paths[0].parents[1])

    llm = LLM(
        model=base_model,
        enable_lora=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        allowed_local_media_path=allowed_media_path,
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
                {"type": "image_url", "image_url": {"url": f"file://{Path(entry['image_path']).resolve()}"}},
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
    base_model: str = "Qwen/Qwen3.5-9B",
    batch_size: int = 8,
    dummy_accuracy: float = 0.8,
    dummy_seed: int = 42,
) -> list[dict]:
    """Run prediction pipeline end-to-end.

    Args:
        eval_dir: Directory with labels.csv + images/.
        output_path: Where to write predictions JSONL.
        provider: "dummy", "vllm", or "transformers".
        model_dir: Path to models/loop_{N}/ (required for vllm/transformers).
        base_model: HuggingFace base model ID (vllm/transformers).
        batch_size: Mini-batch size (transformers provider).
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
    elif provider == "transformers":
        if model_dir is None:
            raise ValueError("provider=transformers requires --model-dir")
        predictions = predict_transformers(
            eval_entries,
            model_dir=model_dir,
            base_model=base_model,
            batch_size=batch_size,
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
        "--provider", type=str,
        choices=["dummy", "vllm", "transformers"], default="dummy",
        help="Inference provider",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to models/loop_{N}/ (vllm/transformers provider)",
    )
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen3.5-9B",
        help="Base model ID (vllm/transformers provider)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Mini-batch size (transformers provider; lower if OOM)",
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
        batch_size=args.batch_size,
        dummy_accuracy=args.dummy_accuracy,
        dummy_seed=args.dummy_seed,
    )

    correct = sum(1 for p in predictions if p["label"] == p["ground_truth"])
    total = len(predictions)
    print(f"Predictions: {correct}/{total} correct ({correct/total:.2%})")


if __name__ == "__main__":
    main()
