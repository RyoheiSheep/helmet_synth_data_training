"""Step C: Dataset Build

Reads screened/loop_{N}/labeled.jsonl + generated images,
produces dataset/loop_{N}/train.jsonl + stats.json.
"""

import argparse
import json
from pathlib import Path


QUESTION_WITH_RATIONALE = (
    'Is the helmet chinstrap properly fastened?\n'
    'Answer with JSON: {"label": "tight"|"loose", "rationale": "<reason>"}'
)

QUESTION_LABEL_ONLY = (
    'Is the helmet chinstrap properly fastened?\n'
    'Answer with JSON: {"label": "tight"|"loose"}'
)


def build_dataset(
    labeled_jsonl: Path,
    image_dir: Path,
    output_dir: Path,
    rationale: bool = True,
) -> dict:
    """Build a fine-tuning dataset from screened labels.

    Args:
        labeled_jsonl: Path to labeled.jsonl from Step B.
        image_dir: Directory containing the generated images.
        output_dir: Where to write train.jsonl and stats.json.
        rationale: If True, include rationale in question/answer.

    Returns:
        The stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    question = QUESTION_WITH_RATIONALE if rationale else QUESTION_LABEL_ONLY

    samples = []
    with open(labeled_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            samples.append(entry)

    train_path = output_dir / "train.jsonl"
    tight_count = 0
    loose_count = 0

    with open(train_path, "w", encoding="utf-8") as out:
        for entry in samples:
            image_path = str(image_dir / f"{entry['image_id']}.png")

            if rationale:
                answer = {
                    "label": entry["label"],
                    "rationale": entry["rationale"],
                }
            else:
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
        "keep_rate": total / len(samples) if samples else 0.0,
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

    stats = build_dataset(
        labeled_jsonl=labeled_jsonl,
        image_dir=image_dir,
        output_dir=output_dir,
        rationale=not args.no_rationale,
    )

    print(f"Dataset built: {stats['total']} samples "
          f"(tight={stats['tight_count']}, loose={stats['loose_count']})")


if __name__ == "__main__":
    main()
