"""Tests for Step D: finetune.py"""

import json
from pathlib import Path

import pytest

from docker.step_d_finetune.finetune import (
    load_dataset,
    run_finetuning,
    validate_dataset,
)

FIXTURES = Path(__file__).parent / "fixtures" / "step_d"

DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen3.5-8B",
    "lora": {
        "r": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "quantization": "none",
    },
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "output_format": "structured_json",
}


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "model_out"


class TestLoadDataset:
    def test_loads_all_samples(self):
        samples = load_dataset(FIXTURES / "train.jsonl")
        assert len(samples) == 4

    def test_sample_schema(self):
        samples = load_dataset(FIXTURES / "train.jsonl")
        for s in samples:
            assert "image_path" in s
            assert "question" in s
            assert "answer" in s


class TestValidateDataset:
    def test_valid_data_passes(self):
        samples = load_dataset(FIXTURES / "train.jsonl")
        validate_dataset(samples)  # Should not raise

    def test_missing_keys_raises(self):
        bad_samples = [{"image_path": "x.png", "question": "q?"}]
        with pytest.raises(ValueError, match="missing keys"):
            validate_dataset(bad_samples)

    def test_invalid_label_raises(self):
        bad_samples = [{
            "image_path": "x.png",
            "question": "q?",
            "answer": {"label": "unknown"},
        }]
        with pytest.raises(ValueError, match="invalid label"):
            validate_dataset(bad_samples)


class TestRunFinetuning:
    def test_lora_weights_created(self, output_dir):
        run_finetuning(
            train_jsonl=FIXTURES / "train.jsonl",
            output_dir=output_dir,
            config=DEFAULT_CONFIG,
            dry_run=True,
        )
        lora_dir = output_dir / "lora_weights"
        assert lora_dir.exists()
        assert (lora_dir / "adapter_config.json").exists()

    def test_adapter_config_content(self, output_dir):
        run_finetuning(
            train_jsonl=FIXTURES / "train.jsonl",
            output_dir=output_dir,
            config=DEFAULT_CONFIG,
            dry_run=True,
        )
        with open(output_dir / "lora_weights" / "adapter_config.json") as f:
            config = json.load(f)
        assert config["r"] == 16
        assert config["alpha"] == 32
        assert "q_proj" in config["target_modules"]

    def test_training_log_created(self, output_dir):
        run_finetuning(
            train_jsonl=FIXTURES / "train.jsonl",
            output_dir=output_dir,
            config=DEFAULT_CONFIG,
            dry_run=True,
        )
        log_path = output_dir / "training_log.json"
        assert log_path.exists()

        with open(log_path) as f:
            log = json.load(f)

        assert log["status"] == "completed"
        assert log["num_samples"] == 4
        assert log["epochs"] == 3

    def test_loss_history(self, output_dir):
        log = run_finetuning(
            train_jsonl=FIXTURES / "train.jsonl",
            output_dir=output_dir,
            config=DEFAULT_CONFIG,
            dry_run=True,
        )
        assert len(log["loss_history"]) == 3
        for entry in log["loss_history"]:
            assert "epoch" in entry
            assert "loss" in entry
            assert isinstance(entry["loss"], float)

        # Loss should decrease over epochs
        losses = [e["loss"] for e in log["loss_history"]]
        assert losses == sorted(losses, reverse=True)

    def test_live_raises_not_implemented(self, output_dir):
        with pytest.raises(NotImplementedError, match="GPU"):
            run_finetuning(
                train_jsonl=FIXTURES / "train.jsonl",
                output_dir=output_dir,
                config=DEFAULT_CONFIG,
                dry_run=False,
            )
