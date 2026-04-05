"""Tests for Step A: generate.py"""

import csv
from pathlib import Path

import pytest

from docker.step_a_imagegen.generate import (
    build_prompt,
    generate_prompts,
    load_seeds,
    run_step_a,
    write_meta_csv,
)

FIXTURES = Path(__file__).parent / "fixtures" / "step_a"


@pytest.fixture
def seeds():
    return load_seeds(FIXTURES)


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "generated_out"


@pytest.fixture
def dummy_config():
    return {
        "loop": 1,
        "variations_per_seed": 2,
        "diversity": {
            "person": "low",
            "background": "low",
            "angle_deg": 5,
            "chinstrap_delta": 0,
        },
        "api": {
            "provider": "dummy",
            "model": "test-model",
        },
    }


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestLoadSeeds:
    def test_loads_all_seeds(self, seeds):
        assert len(seeds) == 3

    def test_seed_schema(self, seeds):
        for seed in seeds:
            assert "image_id" in seed
            assert "label" in seed
            assert seed["label"] in ("tight", "loose")


class TestGeneratePrompts:
    def test_correct_count(self, seeds):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=5,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=1,
        )
        assert len(meta) == 3 * 5  # 3 seeds x 5 variations

    def test_meta_schema(self, seeds):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=2,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=1,
        )
        for row in meta:
            assert set(row.keys()) == {
                "image_id", "seed_id", "seed_label", "prompt", "loop_num"
            }

    def test_seed_label_inherited(self, seeds):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=2,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=1,
        )
        seed_map = {s["image_id"]: s["label"] for s in seeds}
        for row in meta:
            assert row["seed_label"] == seed_map[row["seed_id"]]

    def test_loop_num_propagated(self, seeds):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=1,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=3,
        )
        for row in meta:
            assert row["loop_num"] == 3

    def test_unique_image_ids(self, seeds):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=5,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=1,
        )
        ids = [r["image_id"] for r in meta]
        assert len(ids) == len(set(ids))


class TestBuildPrompt:
    def test_contains_components(self):
        prompt = build_prompt("office", "safety vest", "front view")
        assert "office" in prompt
        assert "safety vest" in prompt
        assert "front view" in prompt
        assert "chinstrap" in prompt.lower()

    def test_keeps_chinstrap(self):
        prompt = build_prompt("park", "uniform", "left profile")
        assert "exactly as-is" in prompt


class TestWriteMetaCSV:
    def test_schema(self, seeds, output_dir):
        meta = generate_prompts(
            seeds=seeds, variations_per_seed=2,
            diversity_background="low", diversity_person="low",
            angle_deg=5, loop_num=1,
        )
        meta_path = write_meta_csv(meta, output_dir)
        rows = _load_csv(meta_path)
        assert len(rows) == 6
        for row in rows:
            assert set(row.keys()) == {
                "image_id", "seed_id", "seed_label", "prompt", "loop_num"
            }


class TestRunStepA:
    def test_full_pipeline(self, output_dir, dummy_config):
        stats = run_step_a(
            seeds_dir=FIXTURES,
            output_dir=output_dir,
            config=dummy_config,
        )
        assert stats["total_generated"] == 6  # 3 seeds x 2 variations
        assert stats["seeds_used"] == 3
        assert stats["loop_num"] == 1

        # meta.csv exists and has correct count
        rows = _load_csv(output_dir / "meta.csv")
        assert len(rows) == 6

        # image files exist for each entry
        for row in rows:
            img_path = output_dir / "images" / f"{row['image_id']}.png"
            assert img_path.exists(), f"Missing image: {img_path}"

    def test_images_are_files(self, output_dir, dummy_config):
        run_step_a(
            seeds_dir=FIXTURES,
            output_dir=output_dir,
            config=dummy_config,
        )
        image_dir = output_dir / "images"
        images = list(image_dir.glob("*.png"))
        assert len(images) == 6
        for img in images:
            assert img.stat().st_size > 0
