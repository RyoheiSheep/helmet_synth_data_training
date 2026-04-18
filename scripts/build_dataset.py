"""Step C: Dataset Build

Reads screened/loop_{N}/labeled.jsonl + generated images,
produces dataset/loop_{N}/train.jsonl + stats.json.
"""

import argparse
import json
from pathlib import Path


QUESTION_WITH_OBSERVATION = (
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

QUESTION_LABEL_ONLY = (
    'Is the helmet chinstrap tight or loose?\n'
    'Answer with JSON: {"label": "tight"|"loose"}'
)


def build_dataset(
    labeled_jsonl: Path,
    image_dir: Path,
    output_dir: Path,
    rationale: bool = True,
    prior_sources: list[tuple[Path, Path]] | None = None,
) -> dict:
    """Build a fine-tuning dataset from screened labels.

    Args:
        labeled_jsonl: Path to labeled.jsonl for the current loop (Step B output).
        image_dir: Directory containing the generated images for the current loop.
        output_dir: Where to write train.jsonl and stats.json.
        rationale: If True, include rationale in question/answer.
        prior_sources: Optional list of (labeled_jsonl, image_dir) pairs from
            previous loops to accumulate into the dataset.

    Returns:
        The stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect (entry, image_dir) pairs from prior loops first, then current.
    all_sources: list[tuple[Path, Path]] = []
    if prior_sources:
        all_sources.extend(prior_sources)
    all_sources.append((labeled_jsonl, image_dir))

    train_path = output_dir / "train.jsonl"
    tight_count = 0
    loose_count = 0
    total_samples = 0

    with open(train_path, "w", encoding="utf-8") as out:
        for src_jsonl, src_image_dir in all_sources:
            with open(src_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    total_samples += 1

                    image_path = str(src_image_dir / f"{entry['image_id']}.png")

                    # Use observation-style answer when the entry has an observation
                    # field (produced by VLM in loop 2+). Passthrough entries from
                    # loop 1 have no observation field → fall back to label-only.
                    if rationale and "observation" in entry:
                        question = QUESTION_WITH_OBSERVATION
                        answer = {
                            "observation": entry["observation"],
                            "label": entry["label"],
                        }
                    else:
                        question = QUESTION_LABEL_ONLY
                        answer = {"label": entry["label"]}

                    record = {
                        "image_path": image_path,
                        "question": question,
                        "answer": answer,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if entry["label"] == "tight":
                        tight_count += 1
                    else:
                        loose_count += 1

    total = tight_count + loose_count
    stats = {
        "total": total,
        "tight_count": tight_count,
        "loose_count": loose_count,
        "keep_rate": total / total_samples if total_samples else 0.0,
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Step C: Build fine-tuning dataset")
    parser.add_argument("--loop", type=int, required=True, help="Loop number")
    parser.add_argument(
        "--no-rationale",
        action="store_true",
        help="Omit rationale from question/answer (label-only mode)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Include screened data from all prior loops (loop_1 .. loop_{N-1})",
    )
    parser.add_argument(
        "--screened-dir",
        type=str,
        default="screened",
        help="Base directory for screened data",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default="generated",
        help="Base directory for generated images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Base directory for output dataset",
    )
    args = parser.parse_args()

    loop_name = f"loop_{args.loop}"
    labeled_jsonl = Path(args.screened_dir) / loop_name / "labeled.jsonl"
    image_dir = Path(args.generated_dir) / loop_name / "images"
    output_dir = Path(args.output_dir) / loop_name

    prior_sources = None
    if args.accumulate and args.loop > 1:
        prior_sources = []
        for prior_loop in range(1, args.loop):
            prior_name = f"loop_{prior_loop}"
            prior_jsonl = Path(args.screened_dir) / prior_name / "labeled.jsonl"
            prior_images = Path(args.generated_dir) / prior_name / "images"
            if prior_jsonl.exists():
                prior_sources.append((prior_jsonl, prior_images))

    stats = build_dataset(
        labeled_jsonl=labeled_jsonl,
        image_dir=image_dir,
        output_dir=output_dir,
        rationale=not args.no_rationale,
        prior_sources=prior_sources,
    )

    print(f"Dataset built: {stats['total']} samples "
          f"(tight={stats['tight_count']}, loose={stats['loose_count']})")


if __name__ == "__main__":
    main()
