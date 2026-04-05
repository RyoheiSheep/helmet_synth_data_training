"""System test: run the full pipeline A → B → C → D → Evaluate for multiple loops.

Uses dummy/dry-run modes throughout. Simulates teacher responses by echoing
seed_label back (with a configurable error rate to test rejection).
"""

import csv
import json
import random
from pathlib import Path

import pytest

from docker.step_a_imagegen.generate import run_step_a
from docker.step_b_screening.screen_and_label import screen_and_label
from docker.step_d_finetune.finetune import run_finetuning
from scripts.build_dataset import build_dataset
from scripts.evaluate import compute_accuracy, evaluate, mcnemar_test


def _create_seed_data(seeds_dir: Path, num_seeds: int = 6):
    """Create seed images and labels.csv."""
    images_dir = seeds_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, num_seeds + 1):
        image_id = f"seed_{i:02d}"
        label = "tight" if i % 2 == 1 else "loose"
        rows.append({"image_id": image_id, "label": label})
        # Create a tiny placeholder file
        (images_dir / f"{image_id}.png").write_bytes(b"\x89PNG_PLACEHOLDER")

    labels_path = seeds_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _simulate_teacher_responses(
    meta_path: Path, error_rate: float = 0.1, seed: int = 42
) -> dict[str, dict]:
    """Simulate teacher VLM by echoing seed_label, with some errors.

    Args:
        meta_path: Path to meta.csv from Step A.
        error_rate: Fraction of images where teacher "disagrees" with seed_label.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    responses = {}
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            seed_label = row["seed_label"]

            if rng.random() < error_rate:
                # Teacher disagrees
                pred = "loose" if seed_label == "tight" else "tight"
                rationale = "Simulated teacher error — label flipped."
            else:
                pred = seed_label
                rationale = f"Simulated teacher agrees: chinstrap is {pred}."

            responses[image_id] = {"label": pred, "rationale": rationale}

    return responses


def _simulate_predictions(
    train_jsonl: Path, accuracy: float = 0.8, seed: int = 42
) -> list[dict]:
    """Simulate model predictions on a 'test set' derived from training data.

    Reads train.jsonl, treats ground_truth = answer.label,
    and flips some predictions to simulate imperfect model.
    """
    rng = random.Random(seed)
    preds = []
    with open(train_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt = rec["answer"]["label"]
            if rng.random() < accuracy:
                pred = gt
            else:
                pred = "loose" if gt == "tight" else "tight"
            preds.append({
                "image_id": Path(rec["image_path"]).stem,
                "label": pred,
                "ground_truth": gt,
            })
    return preds


class TestFullPipelineSingleLoop:
    """Run A → B → C → D → Eval for a single loop."""

    def test_loop_1_runs_end_to_end(self, tmp_path):
        # --- Setup seeds ---
        seeds_dir = tmp_path / "seeds"
        _create_seed_data(seeds_dir, num_seeds=6)

        # --- Step A: Image Generation ---
        gen_dir = tmp_path / "generated" / "loop_1"
        config_a = {
            "loop": 1,
            "variations_per_seed": 3,
            "diversity": {
                "person": "low",
                "background": "low",
                "angle_deg": 5,
                "chinstrap_delta": 0,
            },
            "api": {"provider": "dummy", "model": "test"},
        }
        stats_a = run_step_a(seeds_dir, gen_dir, config_a)
        assert stats_a["total_generated"] == 18  # 6 seeds x 3 variations
        assert (gen_dir / "meta.csv").exists()
        assert len(list((gen_dir / "images").glob("*.png"))) == 18

        # --- Step B: Teacher Screening ---
        teacher_responses = _simulate_teacher_responses(
            gen_dir / "meta.csv", error_rate=0.15, seed=42
        )
        screened_dir = tmp_path / "screened" / "loop_1"
        stats_b = screen_and_label(
            meta_path=gen_dir / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=screened_dir,
        )
        assert stats_b["total"] == 18
        assert stats_b["keep_count"] > 0
        assert stats_b["keep_count"] < 18  # Some should be rejected at 15% error
        assert 0 < stats_b["keep_rate"] < 1.0
        assert (screened_dir / "screening.csv").exists()
        assert (screened_dir / "labeled.jsonl").exists()

        # --- Step C: Dataset Build ---
        dataset_dir = tmp_path / "dataset" / "loop_1"
        stats_c = build_dataset(
            labeled_jsonl=screened_dir / "labeled.jsonl",
            image_dir=gen_dir / "images",
            output_dir=dataset_dir,
        )
        assert stats_c["total"] == stats_b["keep_count"]
        assert stats_c["tight_count"] + stats_c["loose_count"] == stats_c["total"]
        assert (dataset_dir / "train.jsonl").exists()
        assert (dataset_dir / "stats.json").exists()

        # --- Step D: Fine-tuning (dry run) ---
        model_dir = tmp_path / "models" / "loop_1"
        config_d = {
            "base_model": "test-model",
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
        }
        log = run_finetuning(
            train_jsonl=dataset_dir / "train.jsonl",
            output_dir=model_dir,
            config=config_d,
            dry_run=True,
        )
        assert log["status"] == "completed"
        assert log["num_samples"] == stats_c["total"]
        assert len(log["loss_history"]) == 2
        assert (model_dir / "lora_weights" / "adapter_config.json").exists()
        assert (model_dir / "training_log.json").exists()

        # --- Evaluation ---
        preds = _simulate_predictions(
            dataset_dir / "train.jsonl", accuracy=0.8, seed=100
        )
        metrics = compute_accuracy(preds)
        assert metrics["total"] == stats_c["total"]
        assert 0 < metrics["accuracy"] <= 1.0
        assert metrics["tight"]["total"] + metrics["loose"]["total"] == metrics["total"]


class TestMultiLoopPipeline:
    """Run 2 loops and verify McNemar comparison works."""

    def _run_one_loop(
        self, tmp_path, seeds_dir, loop_num, error_rate, pred_accuracy, pred_seed
    ):
        """Helper to run a single loop, returning paths and stats."""
        gen_dir = tmp_path / "generated" / f"loop_{loop_num}"
        config_a = {
            "loop": loop_num,
            "variations_per_seed": 3,
            "diversity": {
                "person": "low",
                "background": "low",
                "angle_deg": 5,
                "chinstrap_delta": 0,
            },
            "api": {"provider": "dummy", "model": "test"},
        }
        run_step_a(seeds_dir, gen_dir, config_a)

        teacher_responses = _simulate_teacher_responses(
            gen_dir / "meta.csv", error_rate=error_rate, seed=loop_num
        )
        screened_dir = tmp_path / "screened" / f"loop_{loop_num}"
        screen_and_label(
            meta_path=gen_dir / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=screened_dir,
        )

        dataset_dir = tmp_path / "dataset" / f"loop_{loop_num}"
        build_dataset(
            labeled_jsonl=screened_dir / "labeled.jsonl",
            image_dir=gen_dir / "images",
            output_dir=dataset_dir,
        )

        model_dir = tmp_path / "models" / f"loop_{loop_num}"
        config_d = {
            "base_model": "test-model",
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
        }
        run_finetuning(
            train_jsonl=dataset_dir / "train.jsonl",
            output_dir=model_dir,
            config=config_d,
            dry_run=True,
        )

        # Simulate predictions and save to file
        preds = _simulate_predictions(
            dataset_dir / "train.jsonl", accuracy=pred_accuracy, seed=pred_seed
        )
        preds_path = tmp_path / "eval" / f"preds_loop_{loop_num}.jsonl"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        return preds_path

    def test_two_loops_with_mcnemar(self, tmp_path):
        seeds_dir = tmp_path / "seeds"
        _create_seed_data(seeds_dir, num_seeds=6)

        # Loop 1: lower accuracy
        preds_1 = self._run_one_loop(
            tmp_path, seeds_dir,
            loop_num=1, error_rate=0.1, pred_accuracy=0.7, pred_seed=100,
        )

        # Loop 2: higher accuracy (simulating improvement)
        preds_2 = self._run_one_loop(
            tmp_path, seeds_dir,
            loop_num=2, error_rate=0.1, pred_accuracy=0.9, pred_seed=200,
        )

        # Evaluate loop 1
        results_1 = evaluate(preds_1)
        assert "metrics" in results_1
        assert results_1["metrics"]["total"] > 0

        # Evaluate loop 2 with McNemar vs loop 1
        # Note: McNemar requires same image_ids, which won't match across loops
        # since each loop generates different image IDs. So we test evaluate()
        # individually and McNemar separately with matching data.
        results_2 = evaluate(preds_2)
        assert results_2["metrics"]["total"] > 0

        # Verify loop 2 accuracy >= loop 1 (by design of our simulation)
        assert results_2["metrics"]["accuracy"] >= results_1["metrics"]["accuracy"]

    def test_mcnemar_on_same_test_set(self, tmp_path):
        """McNemar requires same test set. Simulate two models on identical images."""
        seeds_dir = tmp_path / "seeds"
        _create_seed_data(seeds_dir, num_seeds=6)

        # Run one loop to get a test set
        gen_dir = tmp_path / "generated" / "loop_1"
        config_a = {
            "loop": 1,
            "variations_per_seed": 3,
            "diversity": {
                "person": "low",
                "background": "low",
                "angle_deg": 5,
                "chinstrap_delta": 0,
            },
            "api": {"provider": "dummy", "model": "test"},
        }
        run_step_a(seeds_dir, gen_dir, config_a)

        teacher_responses = _simulate_teacher_responses(
            gen_dir / "meta.csv", error_rate=0.05, seed=42
        )
        screened_dir = tmp_path / "screened" / "loop_1"
        screen_and_label(
            meta_path=gen_dir / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=screened_dir,
        )

        dataset_dir = tmp_path / "dataset" / "loop_1"
        build_dataset(
            labeled_jsonl=screened_dir / "labeled.jsonl",
            image_dir=gen_dir / "images",
            output_dir=dataset_dir,
        )

        # Simulate two models on the SAME test data
        preds_model_a = _simulate_predictions(
            dataset_dir / "train.jsonl", accuracy=0.6, seed=300
        )
        preds_model_b = _simulate_predictions(
            dataset_dir / "train.jsonl", accuracy=0.95, seed=400
        )

        # McNemar test
        result = mcnemar_test(preds_model_a, preds_model_b)
        assert "b_a_wrong_b_right" in result
        assert "c_a_right_b_wrong" in result
        assert "chi2" in result
        assert "p_value" in result
        assert "significant" in result
        # Model B is much better, so b >> c
        assert result["b_a_wrong_b_right"] >= result["c_a_right_b_wrong"]


class TestPipelineDataFlow:
    """Verify data flows correctly between steps — no schema mismatches."""

    def test_step_b_output_feeds_step_c(self, tmp_path):
        """Step B labeled.jsonl must be parseable by Step C."""
        seeds_dir = tmp_path / "seeds"
        _create_seed_data(seeds_dir, num_seeds=4)

        gen_dir = tmp_path / "generated" / "loop_1"
        config_a = {
            "loop": 1,
            "variations_per_seed": 2,
            "diversity": {
                "person": "low", "background": "low",
                "angle_deg": 5, "chinstrap_delta": 0,
            },
            "api": {"provider": "dummy", "model": "test"},
        }
        run_step_a(seeds_dir, gen_dir, config_a)

        teacher_responses = _simulate_teacher_responses(
            gen_dir / "meta.csv", error_rate=0.0, seed=42
        )
        screened_dir = tmp_path / "screened" / "loop_1"
        screen_and_label(
            meta_path=gen_dir / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=screened_dir,
        )

        # Step C should consume Step B output without errors
        dataset_dir = tmp_path / "dataset" / "loop_1"
        stats = build_dataset(
            labeled_jsonl=screened_dir / "labeled.jsonl",
            image_dir=gen_dir / "images",
            output_dir=dataset_dir,
        )
        assert stats["total"] == 8  # 4 seeds x 2 variations, 0% error

    def test_step_c_output_feeds_step_d(self, tmp_path):
        """Step C train.jsonl must be validated by Step D."""
        seeds_dir = tmp_path / "seeds"
        _create_seed_data(seeds_dir, num_seeds=4)

        gen_dir = tmp_path / "generated" / "loop_1"
        config_a = {
            "loop": 1,
            "variations_per_seed": 2,
            "diversity": {
                "person": "low", "background": "low",
                "angle_deg": 5, "chinstrap_delta": 0,
            },
            "api": {"provider": "dummy", "model": "test"},
        }
        run_step_a(seeds_dir, gen_dir, config_a)

        teacher_responses = _simulate_teacher_responses(
            gen_dir / "meta.csv", error_rate=0.0, seed=42
        )
        screened_dir = tmp_path / "screened" / "loop_1"
        screen_and_label(
            meta_path=gen_dir / "meta.csv",
            teacher_responses=teacher_responses,
            output_dir=screened_dir,
        )

        dataset_dir = tmp_path / "dataset" / "loop_1"
        build_dataset(
            labeled_jsonl=screened_dir / "labeled.jsonl",
            image_dir=gen_dir / "images",
            output_dir=dataset_dir,
        )

        # Step D should consume Step C output without errors
        model_dir = tmp_path / "models" / "loop_1"
        config_d = {
            "base_model": "test-model",
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
        }
        log = run_finetuning(
            train_jsonl=dataset_dir / "train.jsonl",
            output_dir=model_dir,
            config=config_d,
            dry_run=True,
        )
        assert log["status"] == "completed"
        assert log["num_samples"] == 8
