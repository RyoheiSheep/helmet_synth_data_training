"""Step D: Fine-tuning

Fine-tunes a lightweight VLM with LoRA on the dataset built by Step C.
Outputs LoRA adapter weights and a training log.
"""

import argparse
import json
import time
from pathlib import Path

import yaml


def load_dataset(train_jsonl: Path) -> list[dict]:
    """Load the training dataset from JSONL."""
    samples = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def validate_dataset(samples: list[dict]) -> None:
    """Validate that training samples have the expected schema."""
    for i, sample in enumerate(samples):
        required_keys = {"image_path", "question", "answer"}
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {i} missing keys: {missing}")

        answer = sample["answer"]
        if "label" not in answer:
            raise ValueError(f"Sample {i} answer missing 'label'")
        if answer["label"] not in ("tight", "loose"):
            raise ValueError(
                f"Sample {i} has invalid label: {answer['label']}"
            )


def run_finetuning(
    train_jsonl: Path,
    output_dir: Path,
    config: dict,
    dry_run: bool = False,
) -> dict:
    """Run the fine-tuning pipeline.

    Args:
        train_jsonl: Path to training JSONL from Step C.
        output_dir: Where to save lora_weights/ and training_log.json.
        config: Parsed step_d.yaml config dict.
        dry_run: If True, skip actual training (for testing schema/IO).

    Returns:
        The training log dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / "lora_weights"
    lora_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate
    samples = load_dataset(train_jsonl)
    validate_dataset(samples)

    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 2e-4)
    base_model = config.get("base_model", "llava-hf/llava-1.5-7b-hf")

    training_log = {
        "base_model": base_model,
        "lora_config": config.get("lora", {}),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_samples": len(samples),
        "loss_history": [],
        "status": "pending",
    }

    if dry_run:
        # Simulate training with fake loss curve
        for epoch in range(1, epochs + 1):
            # Simulate decreasing loss
            epoch_loss = 1.0 / epoch + 0.1
            training_log["loss_history"].append({
                "epoch": epoch,
                "loss": round(epoch_loss, 4),
            })

        # Write a dummy adapter config to lora_weights/
        adapter_config = {
            "base_model": base_model,
            "r": config.get("lora", {}).get("r", 16),
            "alpha": config.get("lora", {}).get("alpha", 32),
            "target_modules": config.get("lora", {}).get(
                "target_modules", ["q_proj", "v_proj"]
            ),
        }
        with open(lora_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        training_log["status"] = "completed"
    else:
        raise NotImplementedError(
            "Live fine-tuning requires GPU and transformers/peft. "
            "Use --dry-run for testing."
        )

    # Write training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    return training_log


def main():
    parser = argparse.ArgumentParser(description="Step D: Fine-tuning")
    parser.add_argument("--loop", type=int, required=True, help="Loop number")
    parser.add_argument(
        "--config", type=str, default="config/step_d.yaml",
        help="Path to step_d.yaml",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset",
        help="Base directory for dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Base directory for model output",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate training without GPU (for testing)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    loop_name = f"loop_{args.loop}"
    train_jsonl = Path(args.dataset_dir) / loop_name / "train.jsonl"
    output_dir = Path(args.output_dir) / loop_name

    log = run_finetuning(
        train_jsonl=train_jsonl,
        output_dir=output_dir,
        config=config,
        dry_run=args.dry_run,
    )

    print(
        f"Fine-tuning {log['status']}: {log['num_samples']} samples, "
        f"{log['epochs']} epochs, final loss={log['loss_history'][-1]['loss']}"
    )


if __name__ == "__main__":
    main()
