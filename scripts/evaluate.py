"""Evaluation script for helmet chinstrap VLM.

Computes accuracy, per-class metrics, and McNemar test between loops.
"""

import argparse
import json
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    """Load predictions JSONL: {"image_id": ..., "label": ..., "ground_truth": ...}"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_accuracy(predictions: list[dict]) -> dict:
    """Compute overall and per-class accuracy.

    Args:
        predictions: List of dicts with 'label' (predicted) and 'ground_truth'.

    Returns:
        Dict with accuracy, per-class metrics, confusion counts.
    """
    if not predictions:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "tight": {"total": 0, "correct": 0, "accuracy": 0.0},
            "loose": {"total": 0, "correct": 0, "accuracy": 0.0},
        }

    total = len(predictions)
    correct = sum(1 for p in predictions if p["label"] == p["ground_truth"])

    class_stats = {}
    for cls in ("tight", "loose"):
        cls_preds = [p for p in predictions if p["ground_truth"] == cls]
        cls_correct = sum(1 for p in cls_preds if p["label"] == cls)
        cls_total = len(cls_preds)
        class_stats[cls] = {
            "total": cls_total,
            "correct": cls_correct,
            "accuracy": cls_correct / cls_total if cls_total > 0 else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        **class_stats,
    }


def mcnemar_test(
    preds_a: list[dict], preds_b: list[dict]
) -> dict:
    """McNemar's test comparing two models on the same test set.

    Both prediction lists must have the same image_ids in the same order.

    Args:
        preds_a: Predictions from model A (loop N-1).
        preds_b: Predictions from model B (loop N).

    Returns:
        Dict with b (A wrong, B right), c (A right, B wrong), p_value.
    """
    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Prediction lists must have same length: {len(preds_a)} vs {len(preds_b)}"
        )

    # b = A wrong, B right; c = A right, B wrong
    b = 0  # discordant: A wrong, B right
    c = 0  # discordant: A right, B wrong

    for pa, pb in zip(preds_a, preds_b):
        if pa["image_id"] != pb["image_id"]:
            raise ValueError(
                f"Mismatched image_ids: {pa['image_id']} vs {pb['image_id']}"
            )

        a_correct = pa["label"] == pa["ground_truth"]
        b_correct = pb["label"] == pb["ground_truth"]

        if not a_correct and b_correct:
            b += 1
        elif a_correct and not b_correct:
            c += 1

    # McNemar's chi-squared statistic (with continuity correction)
    n = b + c
    if n == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
        # Approximate p-value from chi2(1) using a simple lookup
        # For proper stats, use scipy.stats.chi2.sf(chi2, 1)
        p_value = _chi2_p_value_approx(chi2)

    return {
        "b_a_wrong_b_right": b,
        "c_a_right_b_wrong": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def _chi2_p_value_approx(chi2: float) -> float:
    """Rough p-value approximation for chi2 with 1 degree of freedom.

    For production use, replace with scipy.stats.chi2.sf(chi2, 1).
    """
    # Critical values for chi2(1): 3.841 -> p=0.05, 6.635 -> p=0.01
    if chi2 >= 10.828:
        return 0.001
    elif chi2 >= 6.635:
        return 0.01
    elif chi2 >= 3.841:
        return 0.05
    elif chi2 >= 2.706:
        return 0.10
    elif chi2 >= 1.323:
        return 0.25
    else:
        return 1.0


def evaluate(
    predictions_path: Path,
    prev_predictions_path: Path | None = None,
) -> dict:
    """Run full evaluation.

    Args:
        predictions_path: Path to current loop predictions JSONL.
        prev_predictions_path: Path to previous loop predictions (for McNemar).

    Returns:
        Dict with accuracy metrics and optional McNemar results.
    """
    preds = load_predictions(predictions_path)
    results = {"metrics": compute_accuracy(preds)}

    if prev_predictions_path and prev_predictions_path.exists():
        prev_preds = load_predictions(prev_predictions_path)
        results["mcnemar"] = mcnemar_test(prev_preds, preds)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions")
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSONL",
    )
    parser.add_argument(
        "--prev-predictions", type=str, default=None,
        help="Path to previous loop predictions (for McNemar test)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results JSON",
    )
    args = parser.parse_args()

    results = evaluate(
        predictions_path=Path(args.predictions),
        prev_predictions_path=Path(args.prev_predictions) if args.prev_predictions else None,
    )

    metrics = results["metrics"]
    print(
        f"Accuracy: {metrics['accuracy']:.2%} "
        f"({metrics['correct']}/{metrics['total']})"
    )
    print(f"  tight: {metrics['tight']['accuracy']:.2%} ({metrics['tight']['correct']}/{metrics['tight']['total']})")
    print(f"  loose: {metrics['loose']['accuracy']:.2%} ({metrics['loose']['correct']}/{metrics['loose']['total']})")

    if "mcnemar" in results:
        mc = results["mcnemar"]
        print(f"McNemar: chi2={mc['chi2']}, p={mc['p_value']}, significant={mc['significant']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
