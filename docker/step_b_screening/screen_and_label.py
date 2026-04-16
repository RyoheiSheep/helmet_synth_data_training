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
            if rationale and "rationale" in teacher_pred:
                entry["rationale"] = teacher_pred["rationale"]
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
                "rationale": entry.get("rationale", ""),
            }
    return responses


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


def main():
    parser = argparse.ArgumentParser(
        description="Step B: Teacher Screening & Labeling"
    )
    parser.add_argument("--loop", type=int, required=True, help="Loop number")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["vllm", "precomputed"],
        default="precomputed",
        help="Teacher inference provider: 'vllm' for local GPU, 'precomputed' for JSONL",
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

    if args.provider == "precomputed":
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
