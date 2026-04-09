"""Step D: Fine-tuning

Fine-tunes a lightweight VLM with LoRA on the dataset built by Step C.
Outputs LoRA adapter weights and a training log.
"""

import argparse
import json
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


def format_answer(answer: dict) -> str:
    """Serialize an answer dict to the canonical JSON string the model must emit."""
    # Keep key order stable: label first, rationale second if present.
    ordered = {"label": answer["label"]}
    if "rationale" in answer:
        ordered["rationale"] = answer["rationale"]
    return json.dumps(ordered, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Live training backend (GPU-only; heavy imports deferred)
# ---------------------------------------------------------------------------

def _build_collator(processor, max_length: int):
    """Create a collate_fn that turns samples into model inputs.

    Each sample is `{image_path, question, answer}`. The answer is stringified
    JSON and concatenated to the question to form the training target; only
    the answer tokens contribute to the loss.
    """
    import torch
    from PIL import Image

    IGNORE_INDEX = -100

    def collate_fn(batch: list[dict]) -> dict:
        images = [Image.open(s["image_path"]).convert("RGB") for s in batch]
        prompts = [s["question"] for s in batch]
        answers = [format_answer(s["answer"]) for s in batch]
        full_texts = [p + "\n" + a for p, a in zip(prompts, answers)]

        encoded = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"]
        labels = input_ids.clone()

        # Mask everything before the answer so loss is computed on the answer only.
        prompt_lens = [
            len(
                processor.tokenizer(
                    p + "\n", add_special_tokens=False
                )["input_ids"]
            )
            for p in prompts
        ]
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = IGNORE_INDEX
        # Also mask padding.
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = IGNORE_INDEX

        encoded["labels"] = labels
        return encoded

    return collate_fn


def _run_live_training(
    samples: list[dict],
    lora_dir: Path,
    base_model: str,
    lora_cfg: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> list[dict]:
    """Real LoRA fine-tune loop. All heavy imports deferred to here.

    Returns per-epoch loss history as `[{"epoch": N, "loss": float}, ...]`.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model

    class _JsonListDataset(Dataset):
        def __init__(self, items: list[dict]):
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict:
            return self._items[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[step_d] WARNING: no CUDA device detected, running on CPU.")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.train()

    dataset = _JsonListDataset(samples)
    collate_fn = _build_collator(processor, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_history: list[dict] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += float(loss.detach().cpu())
            n_batches += 1
        avg_loss = running / max(n_batches, 1)
        print(f"[step_d] epoch={epoch} loss={avg_loss:.4f}")
        loss_history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    return loss_history


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
    base_model = config.get("base_model", "Qwen/Qwen3.5-8B")
    max_length = config.get("max_length", 1024)

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
        training_log["loss_history"] = _run_live_training(
            samples=samples,
            lora_dir=lora_dir,
            base_model=base_model,
            lora_cfg=config.get("lora", {}),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )
        training_log["status"] = "completed"

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
