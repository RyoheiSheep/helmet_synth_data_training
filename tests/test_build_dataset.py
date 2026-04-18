"""Tests for Step C: build_dataset.py"""

import json
from pathlib import Path

import pytest

from scripts.build_dataset import (
    QUESTION_LABEL_ONLY,
    QUESTION_WITH_OBSERVATION,
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
            assert rec["question"] == QUESTION_WITH_OBSERVATION
            assert rec["answer"]["label"] in ("tight", "loose")
            assert "observation" in rec["answer"]
            assert isinstance(rec["answer"]["observation"], str)

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


class TestBuildDatasetAccumulation:
    """Tests for multi-loop data accumulation via prior_sources."""

    def _make_prior_source(self, tmp_path: Path, name: str) -> tuple[Path, Path]:
        """Create a minimal prior-loop labeled.jsonl + images dir."""
        src_dir = tmp_path / name
        images_dir = src_dir / "images"
        images_dir.mkdir(parents=True)
        jsonl = src_dir / "labeled.jsonl"
        # No observation field: simulates passthrough (loop 1) data
        entries = [
            {"image_id": f"{name}_tight", "label": "tight"},
            {"image_id": f"{name}_loose", "label": "loose"},
        ]
        with open(jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        for e in entries:
            (images_dir / f"{e['image_id']}.png").write_bytes(b"\x89PNG")
        return jsonl, images_dir

    def test_accumulation_combines_all_loops(self, tmp_path):
        """train.jsonl must contain samples from prior loops AND current loop."""
        prior1_jsonl, prior1_images = self._make_prior_source(tmp_path, "loop_1")
        prior2_jsonl, prior2_images = self._make_prior_source(tmp_path, "loop_2")
        output_dir = tmp_path / "dataset_loop_3"

        stats = build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",  # 5 current samples
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
            prior_sources=[(prior1_jsonl, prior1_images), (prior2_jsonl, prior2_images)],
        )

        assert stats["total"] == 5 + 2 + 2  # current + loop_1 + loop_2
        assert stats["tight_count"] + stats["loose_count"] == stats["total"]

        records = _load_jsonl(output_dir / "train.jsonl")
        assert len(records) == 9

        image_paths = [r["image_path"] for r in records]
        assert any("loop_1" in p for p in image_paths)
        assert any("loop_2" in p for p in image_paths)

    def test_no_prior_sources_behaves_like_single_loop(self, output_dir):
        """Passing prior_sources=None must produce the same result as before."""
        stats = build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
            prior_sources=None,
        )
        assert stats["total"] == 5

    def test_prior_sources_order_preserved(self, tmp_path):
        """Prior loop records must appear before current loop records in train.jsonl."""
        prior_jsonl, prior_images = self._make_prior_source(tmp_path, "loop_1")
        output_dir = tmp_path / "out"

        build_dataset(
            labeled_jsonl=FIXTURES / "labeled.jsonl",
            image_dir=FIXTURES / "images",
            output_dir=output_dir,
            rationale=True,
            prior_sources=[(prior_jsonl, prior_images)],
        )

        records = _load_jsonl(output_dir / "train.jsonl")
        # Prior loop records come first
        assert "loop_1" in records[0]["image_path"]
        assert "loop_1" in records[1]["image_path"]
