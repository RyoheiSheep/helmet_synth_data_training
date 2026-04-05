"""Tests for Step B: screen_and_label.py"""

import csv
import json
from pathlib import Path

import pytest

from docker.step_b_screening.screen_and_label import (
    load_meta,
    load_teacher_responses_from_jsonl,
    screen_and_label,
)

FIXTURES = Path(__file__).parent / "fixtures" / "step_b"


@pytest.fixture
def teacher_responses():
    return load_teacher_responses_from_jsonl(FIXTURES / "teacher_responses.jsonl")


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "screened_out"


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestScreeningCSV:
    """Tests for screening.csv output."""

    def test_schema(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        rows = _load_csv(output_dir / "screening.csv")
        assert len(rows) == 6

        for row in rows:
            assert set(row.keys()) == {
                "image_id", "seed_label", "pred_teacher", "keep"
            }

    def test_labels_only_tight_or_loose(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        rows = _load_csv(output_dir / "screening.csv")
        for row in rows:
            assert row["seed_label"] in ("tight", "loose")
            assert row["pred_teacher"] in ("tight", "loose")

    def test_keep_reflects_label_match(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        rows = _load_csv(output_dir / "screening.csv")
        for row in rows:
            expected_keep = row["seed_label"] == row["pred_teacher"]
            assert row["keep"] == str(expected_keep)

    def test_img_003_rejected(self, teacher_responses, output_dir):
        """img_003: seed_label=tight, teacher=loose -> rejected."""
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        rows = _load_csv(output_dir / "screening.csv")
        img_003 = [r for r in rows if r["image_id"] == "img_003"][0]
        assert img_003["seed_label"] == "tight"
        assert img_003["pred_teacher"] == "loose"
        assert img_003["keep"] == "False"


class TestLabeledJSONL:
    """Tests for labeled.jsonl output."""

    def test_only_keep_entries(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        entries = _load_jsonl(output_dir / "labeled.jsonl")
        # 6 total, 1 rejected (img_003) -> 5 kept
        assert len(entries) == 5

    def test_rejected_not_in_labeled(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        entries = _load_jsonl(output_dir / "labeled.jsonl")
        image_ids = [e["image_id"] for e in entries]
        assert "img_003" not in image_ids

    def test_label_values(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        entries = _load_jsonl(output_dir / "labeled.jsonl")
        for entry in entries:
            assert entry["label"] in ("tight", "loose")

    def test_rationale_included_by_default(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
            rationale=True,
        )
        entries = _load_jsonl(output_dir / "labeled.jsonl")
        for entry in entries:
            assert "rationale" in entry
            assert isinstance(entry["rationale"], str)

    def test_rationale_omitted_when_disabled(self, teacher_responses, output_dir):
        screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
            rationale=False,
        )
        entries = _load_jsonl(output_dir / "labeled.jsonl")
        for entry in entries:
            assert "rationale" not in entry


class TestScreeningStats:
    """Tests for returned statistics."""

    def test_keep_rate(self, teacher_responses, output_dir):
        stats = screen_and_label(
            meta_path=FIXTURES / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=output_dir,
        )
        assert stats["total"] == 6
        assert stats["keep_count"] == 5
        assert stats["reject_count"] == 1
        assert abs(stats["keep_rate"] - 5 / 6) < 1e-9

    def test_empty_input(self, output_dir, tmp_path):
        empty_meta = tmp_path / "empty_meta.csv"
        empty_meta.write_text("image_id,seed_id,seed_label,prompt,loop_num\n")

        stats = screen_and_label(
            meta_path=empty_meta,
            teacher_responses={},
            output_dir=output_dir,
        )
        assert stats["total"] == 0
        assert stats["keep_count"] == 0
        assert stats["keep_rate"] == 0.0
