"""Step B: Teacher Screening & Labeling

Reads generated/loop_{N}/meta.csv + images,
runs Teacher VLM inference (or accepts pre-computed responses),
outputs screened/loop_{N}/screening.csv + labeled.jsonl.
"""

import argparse
import csv
import json
from pathlib import Path


TEACHER_PROMPT = (
    "You are a safety inspection assistant.\n"
    "Look at the image and determine the helmet chinstrap status.\n"
    "\n"
    "Respond ONLY in the following JSON format:\n"
    '{\n'
    '  "label": "tight" | "loose",\n'
    '  "rationale": "<one sentence explaining visible evidence>"\n'
    '}'
)

# Must match the question strings in scripts/build_dataset.py (student was trained on these).
_STUDENT_QUESTION_WITH_OBSERVATION = (
    "Inspect the helmet chinstrap in this image. "
    "Describe exactly what you see for each of the following:\n"
    "1. Chin-strap contact: is the strap touching the chin skin, or is there a visible gap?\n"
    "2. Strap tension: does the strap appear taut and straight, or slack and drooping?\n"
    "3. Buckle state: is the buckle fastened or unfastened?\n"
    "4. Buckle position: is the buckle centered under the chin, or displaced to the side?\n"
    "5. Strap shape: does the strap run straight from helmet to chin, or does it sag or curve?\n"
    "After describing these observations, classify the chinstrap as tight or loose.\n"
    'Answer with JSON: {"observation": "<your observations>", "label": "tight"|"loose"}'
)
_STUDENT_QUESTION_LABEL_ONLY = (
    'Is the helmet chinstrap tight or loose?\n'
    'Answer with JSON: {"label": "tight"|"loose"}'
)


def load_meta(meta_path: Path) -> list[dict]:
    """Load meta.csv into a list of dicts."""
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def screen_and_label(
    meta_path: Path,
    teacher_responses: dict[str, dict],
    output_dir: Path,
    rationale: bool = True,
) -> dict:
    """Screen generated images using teacher predictions.

    Args:
        meta_path: Path to meta.csv with seed_label column.
        teacher_responses: Dict mapping image_id -> {"label": ..., "rationale": ...}
        output_dir: Where to write screening.csv and labeled.jsonl.
        rationale: Whether to include rationale in labeled.jsonl.

    Returns:
        Dict with screening statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = load_meta(meta_path)

    screening_rows = []
    labeled_entries = []

    for row in meta_rows:
        image_id = row["image_id"]
        seed_label = row["seed_label"]

        if image_id not in teacher_responses:
            continue

        teacher_pred = teacher_responses[image_id]
        pred_label = teacher_pred["label"]
        keep = pred_label == seed_label

        screening_rows.append({
            "image_id": image_id,
            "seed_label": seed_label,
            "pred_teacher": pred_label,
            "keep": keep,
        })

        if keep:
            entry = {"image_id": image_id, "label": pred_label}
            if rationale and "observation" in teacher_pred:
                entry["observation"] = teacher_pred["observation"]
            labeled_entries.append(entry)

    # Write screening.csv
    screening_path = output_dir / "screening.csv"
    with open(screening_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "seed_label", "pred_teacher", "keep"]
        )
        writer.writeheader()
        for row in screening_rows:
            writer.writerow(row)

    # Write labeled.jsonl (keep=True only)
    labeled_path = output_dir / "labeled.jsonl"
    with open(labeled_path, "w", encoding="utf-8") as f:
        for entry in labeled_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(screening_rows)
    keep_count = len(labeled_entries)
    stats = {
        "total": total,
        "keep_count": keep_count,
        "reject_count": total - keep_count,
        "keep_rate": keep_count / total if total > 0 else 0.0,
    }

    return stats


def load_teacher_responses_from_jsonl(path: Path) -> dict[str, dict]:
    """Load pre-computed teacher responses from a JSONL file."""
    responses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            responses[entry["image_id"]] = {
                "label": entry["label"],
                "observation": entry.get("observation", ""),
            }
    return responses


def _passthrough_responses(meta_path: Path) -> dict[str, dict]:
    """Return seed labels directly without any VLM inference (Loop 1).

    All generated images inherit their seed's label. No observation field is
    included because no VLM was called — the screened labeled.jsonl for Loop 1
    will contain only {image_id, label}.
    """
    meta_rows = load_meta(meta_path)
    return {row["image_id"]: {"label": row["seed_label"]} for row in meta_rows}


def _run_vllm_teacher(meta_path: Path, image_dir: Path, config: dict) -> dict[str, dict]:
    """Run local vLLM Teacher inference. Imports deferred so tests don't need vllm."""
    from vllm_teacher import run_teacher_inference

    meta_rows = load_meta(meta_path)
    image_paths = [image_dir / f"{row['image_id']}.png" for row in meta_rows]
    image_ids = [row["image_id"] for row in meta_rows]

    return run_teacher_inference(
        image_paths=image_paths,
        image_ids=image_ids,
        model_id=config.get("teacher_model", "Qwen/Qwen3.5-27B"),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        max_model_len=config.get("max_model_len", 4096),
        enable_thinking=config.get("enable_thinking", False),
        temperature=config.get("sampling", {}).get("temperature", 0.7),
        top_p=config.get("sampling", {}).get("top_p", 0.8),
        max_tokens=config.get("sampling", {}).get("max_tokens", 256),
        rationale=config.get("rationale", True),
        dtype=config.get("dtype", "auto"),
        allowed_local_media_path=image_dir,
    )


def _run_student_inference(
    meta_path: Path,
    image_dir: Path,
    model_dir: Path,
    base_model: str,
    rationale: bool = True,
    batch_size: int = 4,
) -> dict[str, dict]:
    """Run batched inference using the fine-tuned Student LoRA model. Deferred imports."""
    import json as _json

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    meta_rows = load_meta(meta_path)
    lora_dir = model_dir / "lora_weights"
    question = _STUDENT_QUESTION_WITH_OBSERVATION if rationale else _STUDENT_QUESTION_LABEL_ONLY

    processor = AutoProcessor.from_pretrained(str(lora_dir))
    # Left-padding aligns all batch samples at the generation start point.
    # Some VL processors ignore the tokenizer's padding_side, so set both.
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.eval()

    # Pre-load all images; skip missing files up front.
    valid_items: list[tuple[str, Image.Image]] = []
    for row in meta_rows:
        image_id = row["image_id"]
        image_path = image_dir / f"{image_id}.png"
        try:
            valid_items.append((image_id, Image.open(image_path).convert("RGB")))
        except FileNotFoundError:
            print(f"Warning: image not found for student inference: {image_path}")

    responses: dict[str, dict] = {}
    total = len(valid_items)

    for batch_start in range(0, total, batch_size):
        batch = valid_items[batch_start:batch_start + batch_size]
        image_ids = [item[0] for item in batch]
        images = [item[1] for item in batch]

        texts = []
        for _ in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            # Prefix-force the assistant response into a JSON object. The base
            # model defaults to chain-of-thought and the LoRA is not strong
            # enough to suppress it, so we pre-fill the first two tokens.
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(prompt_text + '{"')

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        done = min(batch_start + batch_size, total)
        print(f"Student inference: {done}/{total}", flush=True)

        for i, image_id in enumerate(image_ids):
            generated = processor.decode(
                output_ids[i][prompt_len:],
                skip_special_tokens=True,
            )
            # Prepend the forced prefix so the model's continuation becomes
            # a complete JSON object.
            full_output = '{"' + generated
            if batch_start == 0:
                print(f"  [debug] raw output for {image_id}: {full_output[:300]!r}", flush=True)
            start = full_output.find('{')
            end = full_output.rfind('}')
            if start == -1 or end < start:
                print(f"Warning: no JSON object found for {image_id} (raw: {full_output[:120]!r}), skipping")
                continue
            try:
                parsed = _json.loads(full_output[start:end + 1])
            except _json.JSONDecodeError:
                print(f"Warning: malformed JSON for {image_id} (raw: {full_output[start:end+1][:120]!r}), skipping")
                continue
            label = parsed.get("label", "")
            if label not in ("tight", "loose"):
                print(f"Warning: invalid label '{label}' from student for {image_id}, skipping")
                continue
            responses[image_id] = {
                "label": label,
                "observation": parsed.get("observation", "") if rationale else "",
            }

    return responses


def main():
    parser = argparse.ArgumentParser(
        description="Step B: Teacher Screening & Labeling"
    )
    parser.add_argument("--loop", type=int, required=True, help="Loop number")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["passthrough", "vllm", "precomputed", "student"],
        default="precomputed",
        help=(
            "Inference provider: "
            "'passthrough' to inherit seed labels with no VLM (loop 1), "
            "'vllm' for Teacher GPU inference, "
            "'precomputed' for JSONL fixture, "
            "'student' for fine-tuned Student LoRA (loop 2+)"
        ),
    )
    parser.add_argument(
        "--teacher-responses",
        type=str,
        help="Path to pre-computed teacher responses JSONL (provider=precomputed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/step_b.yaml",
        help="Path to step_b.yaml (provider=vllm)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to previous loop's model dir, e.g. models/loop_1 (provider=student)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="HuggingFace base model ID used during fine-tuning (provider=student)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for student model inference (provider=student)",
    )
    parser.add_argument(
        "--no-rationale",
        action="store_true",
        help="Omit rationale from labeled output",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default="generated",
        help="Base directory for generated data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screened",
        help="Base directory for screened output",
    )
    args = parser.parse_args()

    loop_name = f"loop_{args.loop}"
    meta_path = Path(args.generated_dir) / loop_name / "meta.csv"
    image_dir = Path(args.generated_dir) / loop_name / "images"
    output_dir = Path(args.output_dir) / loop_name

    if args.provider == "passthrough":
        teacher_responses = _passthrough_responses(meta_path)
    elif args.provider == "precomputed":
        if not args.teacher_responses:
            raise ValueError(
                "provider=precomputed requires --teacher-responses path"
            )
        teacher_responses = load_teacher_responses_from_jsonl(
            Path(args.teacher_responses)
        )
    elif args.provider == "vllm":
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        teacher_responses = _run_vllm_teacher(meta_path, image_dir, config)
    elif args.provider == "student":
        if not args.model_dir:
            raise ValueError("provider=student requires --model-dir")
        if not args.base_model:
            raise ValueError("provider=student requires --base-model")
        teacher_responses = _run_student_inference(
            meta_path=meta_path,
            image_dir=image_dir,
            model_dir=Path(args.model_dir),
            base_model=args.base_model,
            rationale=not args.no_rationale,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    stats = screen_and_label(
        meta_path=meta_path,
        teacher_responses=teacher_responses,
        output_dir=output_dir,
        rationale=not args.no_rationale,
    )

    print(
        f"Screening done: {stats['total']} images, "
        f"{stats['keep_count']} kept, {stats['reject_count']} rejected "
        f"(keep_rate={stats['keep_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
