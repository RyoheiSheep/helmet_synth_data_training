"""Tests for Step C: build_dataset.py"""

import json
from pathlib import Path

import pytest

from scripts.build_dataset import (
    QUESTION_LABEL_ONLY,
    QUESTION_WITH_RATIONALE,
    build_dataset,
)

FIXTURES = Path(__file__).parent / "fixtures" / "step_c"


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "dataset_out"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestBuildDatasetWithRationale:
    """Tests with rationale=True (default)."""

    def test_train_jsonl_schema(self, output_dir):
        build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
        )
        records = _load_jsonl(output_dir / "train.jsonl")
        assert len(records) == 5

        for rec in records:
            assert set(rec.keys()) == {"image_path", "question", "answer"}
            assert rec["question"] == QUESTION_WITH_RATIONALE
            assert rec["answer"]["label"] in ("tight", "loose")
            assert "rationale" in rec["answer"]
            assert isinstance(rec["answer"]["rationale"], str)

    def test_stats_json_fields(self, output_dir):
        stats = build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
        )
        stats_path = output_dir / "stats.json"
        assert stats_path.exists()

        with open(stats_path) as f:
            saved_stats = json.load(f)

        for key in ("total", "tight_count", "loose_count", "keep_rate"):
            assert key in saved_stats

        assert saved_stats["total"] == 5
        assert saved_stats["tight_count"] == 3
        assert saved_stats["loose_count"] == 2
        assert saved_stats["keep_rate"] == 1.0

        # Return value matches saved file
        assert stats == saved_stats

    def test_image_paths_reference_image_dir(self, output_dir):
        build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
        )
        records = _load_jsonl(output_dir / "train.jsonl")
        for rec in records:
            assert rec["image_path"].endswith(".png")
            assert "images" in rec["image_path"]


class TestBuildDatasetLabelOnly:
    """Tests with rationale=False (label-only mode)."""

    def test_no_rationale_in_answer(self, output_dir):
        build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=False,
        )
        records = _load_jsonl(output_dir / "train.jsonl")
        assert len(records) == 5

        for rec in records:
            assert rec["question"] == QUESTION_LABEL_ONLY
            assert set(rec["answer"].keys()) == {"label"}
            assert rec["answer"]["label"] in ("tight", "loose")


class TestBuildDatasetEdgeCases:
    """Edge case tests."""

    def test_empty_input(self, output_dir, tmp_path):
        empty_jsonl = tmp_path / "empty.jsonl"
        empty_jsonl.write_text("")

        stats = build_dataset(
            labeled_jsonl=empty_jsonl,
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
        )
        assert stats["total"] == 0
        assert stats["tight_count"] == 0
        assert stats["loose_count"] == 0

    def test_output_dir_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()

        build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=nested,
            rationale=True,
        )
        assert nested.exists()
        assert (nested / "train.jsonl").exists()
        assert (nested / "stats.json").exists()
