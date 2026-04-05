"""Tests for evaluate.py"""

import json
from pathlib import Path

import pytest

from scripts.evaluate import (
    compute_accuracy,
    evaluate,
    load_predictions,
    mcnemar_test,
)


@pytest.fixture
def perfect_preds():
    return [
        {"image_id": "img_001", "label": "tight", "ground_truth": "tight"},
        {"image_id": "img_002", "label": "loose", "ground_truth": "loose"},
        {"image_id": "img_003", "label": "tight", "ground_truth": "tight"},
        {"image_id": "img_004", "label": "loose", "ground_truth": "loose"},
    ]


@pytest.fixture
def mixed_preds():
    return [
        {"image_id": "img_001", "label": "tight", "ground_truth": "tight"},
        {"image_id": "img_002", "label": "tight", "ground_truth": "loose"},  # wrong
        {"image_id": "img_003", "label": "loose", "ground_truth": "tight"},  # wrong
        {"image_id": "img_004", "label": "loose", "ground_truth": "loose"},
    ]


@pytest.fixture
def preds_jsonl(tmp_path, mixed_preds):
    path = tmp_path / "preds.jsonl"
    with open(path, "w") as f:
        for p in mixed_preds:
            f.write(json.dumps(p) + "\n")
    return path


class TestComputeAccuracy:
    def test_perfect_accuracy(self, perfect_preds):
        result = compute_accuracy(perfect_preds)
        assert result["accuracy"] == 1.0
        assert result["correct"] == 4
        assert result["total"] == 4

    def test_mixed_accuracy(self, mixed_preds):
        result = compute_accuracy(mixed_preds)
        assert result["accuracy"] == 0.5
        assert result["correct"] == 2
        assert result["total"] == 4

    def test_per_class_accuracy(self, mixed_preds):
        result = compute_accuracy(mixed_preds)
        assert result["tight"]["total"] == 2
        assert result["tight"]["correct"] == 1
        assert result["tight"]["accuracy"] == 0.5
        assert result["loose"]["total"] == 2
        assert result["loose"]["correct"] == 1
        assert result["loose"]["accuracy"] == 0.5

    def test_empty_predictions(self):
        result = compute_accuracy([])
        assert result["total"] == 0
        assert result["accuracy"] == 0.0


class TestMcNemarTest:
    def test_identical_models(self, perfect_preds):
        result = mcnemar_test(perfect_preds, perfect_preds)
        assert result["b_a_wrong_b_right"] == 0
        assert result["c_a_right_b_wrong"] == 0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_b_improves_over_a(self, mixed_preds, perfect_preds):
        # A gets 2 wrong, B gets all right
        result = mcnemar_test(mixed_preds, perfect_preds)
        assert result["b_a_wrong_b_right"] == 2
        assert result["c_a_right_b_wrong"] == 0

    def test_mismatched_lengths_raises(self):
        a = [{"image_id": "1", "label": "tight", "ground_truth": "tight"}]
        b = []
        with pytest.raises(ValueError, match="same length"):
            mcnemar_test(a, b)

    def test_mismatched_ids_raises(self):
        a = [{"image_id": "1", "label": "tight", "ground_truth": "tight"}]
        b = [{"image_id": "2", "label": "tight", "ground_truth": "tight"}]
        with pytest.raises(ValueError, match="Mismatched"):
            mcnemar_test(a, b)


class TestLoadPredictions:
    def test_loads_from_file(self, preds_jsonl):
        preds = load_predictions(preds_jsonl)
        assert len(preds) == 4
        for p in preds:
            assert "image_id" in p
            assert "label" in p
            assert "ground_truth" in p


class TestEvaluate:
    def test_without_prev(self, preds_jsonl):
        results = evaluate(preds_jsonl)
        assert "metrics" in results
        assert results["metrics"]["accuracy"] == 0.5
        assert "mcnemar" not in results

    def test_with_prev(self, tmp_path, perfect_preds, mixed_preds):
        prev_path = tmp_path / "prev.jsonl"
        curr_path = tmp_path / "curr.jsonl"
        with open(prev_path, "w") as f:
            for p in mixed_preds:
                f.write(json.dumps(p) + "\n")
        with open(curr_path, "w") as f:
            for p in perfect_preds:
                f.write(json.dumps(p) + "\n")

        results = evaluate(curr_path, prev_path)
        assert "metrics" in results
        assert "mcnemar" in results
        assert results["metrics"]["accuracy"] == 1.0
