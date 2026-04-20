"""Step A: Image Generation

Reads seed images + labels, generates diverse variations using a local
FLUX.2 Klein model (via diffusers) or dummy mode for testing.

Container I/O:
  Input (ro):
    /data/seeds/images/*.png       — seed images
    /data/seeds/labels.csv         — image_id,label
    /data/config/step_a.yaml       — config
  Env:
    HF_TOKEN                       — HuggingFace token (for gated models)
    LOOP_NUM                       — optional loop number override
  Output:
    /data/generated/loop_{N}/images/*.png  — generated images
    /data/generated/loop_{N}/meta.csv      — image_id,seed_id,seed_label,prompt,loop_num
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import yaml


BACKGROUNDS = {
    "low": ["office", "warehouse", "parking lot"],
    "medium": ["park", "street", "rooftop", "beach", "subway station"],
    "high": [
        "park", "street", "rooftop", "beach", "subway station",
        "forest", "desert", "snow field", "night scene", "rainy day",
    ],
}

CLOTHING = {
    "low": ["work uniform", "safety vest"],
    "medium": ["work uniform", "safety vest", "casual clothes", "winter jacket"],
    "high": [
        "work uniform", "safety vest", "casual clothes", "winter jacket",
        "rain coat", "high-visibility suit", "polo shirt",
    ],
}

ANGLE_DESCRIPTIONS = {
    5: ["front view", "slightly left", "slightly right"],
    15: ["front view", "left profile", "right profile", "slightly above", "slightly below"],
    30: [
        "front view", "left profile", "right profile",
        "from above", "from below", "three-quarter left", "three-quarter right",
    ],
}


def build_prompt(background: str, clothing: str, angle_desc: str) -> str:
    """Build the image edit prompt from components."""
    return (
        f"Edit this image: change the background to {background} and "
        f"clothing to {clothing}. Keep the helmet chinstrap condition "
        f"exactly as-is. Camera angle: {angle_desc}."
    )


def get_angle_descriptions(angle_deg: int) -> list[str]:
    """Get angle description options for the given max angle."""
    for threshold in sorted(ANGLE_DESCRIPTIONS.keys()):
        if angle_deg <= threshold:
            return ANGLE_DESCRIPTIONS[threshold]
    return ANGLE_DESCRIPTIONS[max(ANGLE_DESCRIPTIONS.keys())]


def generate_prompts(
    seeds: list[dict],
    variations_per_seed: int,
    diversity_background: str,
    diversity_person: str,
    angle_deg: int,
    loop_num: int,
) -> list[dict]:
    """Generate prompt metadata for all seed variations.

    Args:
        seeds: List of dicts with image_id and label.
        variations_per_seed: Number of variations per seed image.
        diversity_background: "low", "medium", or "high".
        diversity_person: "low", "medium", or "high".
        angle_deg: Max camera angle deviation.
        loop_num: Current loop number.

    Returns:
        List of dicts with image_id, seed_id, seed_label, prompt, loop_num.
    """
    backgrounds = BACKGROUNDS.get(diversity_background, BACKGROUNDS["low"])
    clothes = CLOTHING.get(diversity_person, CLOTHING["low"])
    angles = get_angle_descriptions(angle_deg)

    meta_rows = []
    img_counter = 0

    for seed in seeds:
        for _ in range(variations_per_seed):
            img_counter += 1
            image_id = f"gen_{loop_num:02d}_{img_counter:04d}"

            bg = random.choice(backgrounds)
            cl = random.choice(clothes)
            angle = random.choice(angles)
            prompt = build_prompt(bg, cl, angle)

            meta_rows.append({
                "image_id": image_id,
                "seed_id": seed["image_id"],
                "seed_label": seed["label"],
                "prompt": prompt,
                "loop_num": loop_num,
            })

    return meta_rows


def load_seeds(seeds_dir: Path) -> list[dict]:
    """Load seed labels.csv."""
    labels_path = seeds_dir / "labels.csv"
    seeds = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append(row)
    return seeds


# ---------------------------------------------------------------------------
# Image generation backends
# ---------------------------------------------------------------------------

def _load_diffusers_pipeline(model_id: str):
    """Load a Flux2KleinPipeline. Imports are deferred so tests don't need torch."""
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    return pipe


def _generate_one_diffusers(
    pipe,
    prompt: str,
    seed_image_path: Path,
    output_path: Path,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    output_format: str = "png",
    seed: int | None = None,
):
    """Generate a single image with a loaded diffusers pipeline.

    Args:
        pipe: A loaded Flux2KleinPipeline.
        prompt: Edit prompt.
        seed_image_path: Path to the seed image.
        output_path: Where to save the generated image.
        num_inference_steps: Denoising steps.
        guidance_scale: CFG scale.
        output_format: "png" or "jpeg".
        seed: Optional torch generator seed.
    """
    import torch
    from PIL import Image

    input_image = Image.open(seed_image_path).convert("RGB")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=input_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    out_image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(output_path, format=output_format.upper())


def _generate_batch_diffusers(
    pipe,
    prompts: list[str],
    seed_image_paths: list[Path],
    output_paths: list[Path],
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    output_format: str = "png",
):
    """Generate a batch of images in one pipeline call.

    Each item can have a completely different seed image and prompt.
    Images are resized to a common size before batching (required because
    the pipeline stacks them into a single tensor).
    """
    from PIL import Image

    raw_images = [Image.open(p).convert("RGB") for p in seed_image_paths]

    # All images in a batch must share the same spatial dimensions.
    target_size = raw_images[0].size  # (width, height)
    images = [
        img.resize(target_size, Image.LANCZOS) if img.size != target_size else img
        for img in raw_images
    ]

    result = pipe(
        prompt=prompts,
        image=images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    for out_image, output_path in zip(result.images, output_paths):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_image.save(output_path, format=output_format.upper())


def generate_images(
    meta_rows: list[dict],
    seed_image_dir: Path,
    output_image_dir: Path,
    api_provider: str = "diffusers",
    api_model: str = "black-forest-labs/FLUX.2-klein-base-9B",
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    output_format: str = "png",
    batch_size: int = 1,
) -> list[dict]:
    """Generate images for all prompt rows.

    Args:
        meta_rows: Prompt metadata from generate_prompts().
        seed_image_dir: Directory containing seed images.
        output_image_dir: Where to save generated images.
        api_provider: "diffusers" for local GPU, "dummy" for testing.
        api_model: HuggingFace model ID (used when provider is "diffusers").
        num_inference_steps: Denoising steps (diffusers only).
        guidance_scale: CFG scale (diffusers only).
        output_format: "png" or "jpeg".
        batch_size: Number of images to generate per pipeline call (diffusers only).

    Returns:
        The meta_rows (unmodified, for chaining).
    """
    output_image_dir.mkdir(parents=True, exist_ok=True)

    ext = "png" if output_format == "png" else "jpg"

    if api_provider == "dummy":
        for row in meta_rows:
            seed_image = seed_image_dir / f"{row['seed_id']}.png"
            output_image = output_image_dir / f"{row['image_id']}.{ext}"
            if seed_image.exists():
                shutil.copy2(seed_image, output_image)
            else:
                output_image.write_bytes(b"PLACEHOLDER")

    elif api_provider == "diffusers":
        pipe = _load_diffusers_pipeline(api_model)

        for batch_start in range(0, len(meta_rows), batch_size):
            batch = meta_rows[batch_start : batch_start + batch_size]
            batch_end = batch_start + len(batch)

            print(f"  [{batch_start + 1}-{batch_end}/{len(meta_rows)}] "
                  f"{[r['image_id'] for r in batch]}")

            _generate_batch_diffusers(
                pipe=pipe,
                prompts=[r["prompt"] for r in batch],
                seed_image_paths=[seed_image_dir / f"{r['seed_id']}.png" for r in batch],
                output_paths=[output_image_dir / f"{r['image_id']}.{ext}" for r in batch],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_format=output_format,
            )
    else:
        raise ValueError(
            f"Unknown api.provider: '{api_provider}'. "
            "Use 'diffusers' for local GPU or 'dummy' for testing."
        )

    return meta_rows


def write_meta_csv(meta_rows: list[dict], output_dir: Path) -> Path:
    """Write meta.csv to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "seed_id", "seed_label", "prompt", "loop_num"],
        )
        writer.writeheader()
        for row in meta_rows:
            writer.writerow(row)
    return meta_path


def run_step_a(
    seeds_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Run the full Step A pipeline.

    Args:
        seeds_dir: Path to seeds/ directory (with images/ and labels.csv).
        output_dir: Path to generated/loop_{N}/ output directory.
        config: Parsed step_a.yaml config dict.

    Returns:
        Dict with generation statistics.
    """
    seeds = load_seeds(seeds_dir)
    loop_num = config["loop"]
    variations = config["variations_per_seed"]
    diversity = config["diversity"]
    api_cfg = config["api"]

    meta_rows = generate_prompts(
        seeds=seeds,
        variations_per_seed=variations,
        diversity_background=diversity["background"],
        diversity_person=diversity["person"],
        angle_deg=diversity["angle_deg"],
        loop_num=loop_num,
    )

    image_dir = output_dir / "images"
    generate_images(
        meta_rows=meta_rows,
        seed_image_dir=seeds_dir / "images",
        output_image_dir=image_dir,
        api_provider=api_cfg["provider"],
        api_model=api_cfg["model"],
        num_inference_steps=api_cfg.get("num_inference_steps", 50),
        guidance_scale=api_cfg.get("guidance_scale", 4.0),
        output_format=api_cfg.get("output_format", "png"),
        batch_size=api_cfg.get("batch_size", 1),
    )

    write_meta_csv(meta_rows, output_dir)

    stats = {
        "total_generated": len(meta_rows),
        "seeds_used": len(seeds),
        "variations_per_seed": variations,
        "loop_num": loop_num,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Step A: Image Generation")
    parser.add_argument(
        "--config", type=str, default="config/step_a.yaml", help="Path to step_a.yaml"
    )
    parser.add_argument(
        "--seeds-dir", type=str, default="seeds", help="Path to seeds directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: generated/loop_{N})",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    loop_num = config["loop"]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("generated") / f"loop_{loop_num}"
    )

    stats = run_step_a(
        seeds_dir=Path(args.seeds_dir),
        output_dir=output_dir,
        config=config,
    )

    print(
        f"Generated {stats['total_generated']} images from "
        f"{stats['seeds_used']} seeds (loop {stats['loop_num']})"
    )


if __name__ == "__main__":
    main()
