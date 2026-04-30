"""Tests for scripts/predict.py"""

import csv
import json
from pathlib import Path

import pytest

from scripts.predict import load_eval_set, predict_dummy, run_prediction

FIXTURES = Path(__file__).parent / "fixtures" / "eval"


class TestLoadEvalSet:
    def test_loads_all_entries(self):
        entries = load_eval_set(FIXTURES)
        assert len(entries) == 6

    def test_entry_schema(self):
        entries = load_eval_set(FIXTURES)
        for e in entries:
            assert "image_id" in e
            assert "label" in e
            assert "image_path" in e
            assert e["label"] in ("tight", "loose")

    def test_image_path_constructed(self):
        entries = load_eval_set(FIXTURES)
        for e in entries:
            assert e["image_path"] == FIXTURES / "images" / f"{e['image_id']}.png"

    def test_invalid_label_raises(self, tmp_path):
        labels = tmp_path / "labels.csv"
        labels.write_text("image_id,label\nbad_01,unknown\n")
        (tmp_path / "images").mkdir()
        with pytest.raises(ValueError, match="Invalid ground-truth label"):
            load_eval_set(tmp_path)


class TestPredictDummy:
    def test_returns_correct_count(self):
        entries = load_eval_set(FIXTURES)
        preds = predict_dummy(entries, accuracy=1.0)
        assert len(preds) == 6

    def test_prediction_schema(self):
        entries = load_eval_set(FIXTURES)
        preds = predict_dummy(entries)
        for p in preds:
            assert "image_id" in p
            assert "label" in p
            assert "ground_truth" in p
            assert p["label"] in ("tight", "loose")
            assert p["ground_truth"] in ("tight", "loose")

    def test_perfect_accuracy(self):
        entries = load_eval_set(FIXTURES)
        preds = predict_dummy(entries, accuracy=1.0)
        for p in preds:
            assert p["label"] == p["ground_truth"]

    def test_zero_accuracy(self):
        entries = load_eval_set(FIXTURES)
        preds = predict_dummy(entries, accuracy=0.0)
        for p in preds:
            assert p["label"] != p["ground_truth"]

    def test_deterministic_with_seed(self):
        entries = load_eval_set(FIXTURES)
        preds_a = predict_dummy(entries, accuracy=0.5, seed=99)
        preds_b = predict_dummy(entries, accuracy=0.5, seed=99)
        assert preds_a == preds_b


class TestRunPrediction:
    def test_writes_output_jsonl(self, tmp_path):
        output = tmp_path / "preds.jsonl"
        run_prediction(
            eval_dir=FIXTURES,
            output_path=output,
            provider="dummy",
            dummy_accuracy=0.8,
        )
        assert output.exists()
        lines = [
            json.loads(l)
            for l in output.read_text().strip().split("\n")
            if l.strip()
        ]
        assert len(lines) == 6

    def test_output_schema(self, tmp_path):
        output = tmp_path / "preds.jsonl"
        run_prediction(
            eval_dir=FIXTURES,
            output_path=output,
            provider="dummy",
        )
        for line in output.read_text().strip().split("\n"):
            rec = json.loads(line)
            assert set(rec.keys()) == {"image_id", "label", "ground_truth"}
            assert rec["label"] in ("tight", "loose")
            assert rec["ground_truth"] in ("tight", "loose")

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "sub" / "dir" / "preds.jsonl"
        run_prediction(
            eval_dir=FIXTURES,
            output_path=output,
            provider="dummy",
        )
        assert output.exists()

    def test_unknown_provider_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown provider"):
            run_prediction(
                eval_dir=FIXTURES,
                output_path=tmp_path / "out.jsonl",
                provider="invalid",
            )

    def test_vllm_without_model_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires --model-dir"):
            run_prediction(
                eval_dir=FIXTURES,
                output_path=tmp_path / "out.jsonl",
                provider="vllm",
                model_dir=None,
            )
