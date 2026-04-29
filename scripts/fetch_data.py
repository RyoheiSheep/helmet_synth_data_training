"""Fetch helmet dataset tarballs from Cloudflare R2 into seeds/ and eval/test_real/.

Per-split policy (hardcoded — see CLAUDE.md context):
    train -> seeds/            (merge: keep existing committed images, append new)
    test  -> eval/test_real/   (replace: wipe existing placeholders)

Tarball layout (members at archive root):
    images/{image_id}.png ...
    labels.csv            (image_id,label)

Credentials are read from env: CLOUDFLARE_R2_ACCOUNT_ID,
CLOUDFLARE_R2_ACCESS_KEY_ID, CLOUDFLARE_R2_SECRET_ACCESS_KEY,
CLOUDFLARE_R2_BUCKET (CLI --bucket overrides the bucket env var).
"""

import argparse
import csv
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

VALID_LABELS = ("tight", "loose")

SPLITS: dict[str, dict] = {
    "train": {
        "object_key": "train_dataset.tar",
        "target": Path("seeds"),
        "mode": "merge",
    },
    "test": {
        "object_key": "test_dataset.tar",
        "target": Path("eval/test_real"),
        "mode": "replace",
    },
}


# ---------------------------------------------------------------------------
# Tar handling
# ---------------------------------------------------------------------------


def _validate_tar_members(tar: tarfile.TarFile) -> None:
    for m in tar.getmembers():
        name = m.name
        # Cross-platform absolute-path detection: tar archives use POSIX paths,
        # but Path.is_absolute() is platform-aware. Check explicitly for both
        # POSIX-style ("/...") and Windows-style ("\\..." or "C:...") roots.
        if (
            name.startswith("/")
            or name.startswith("\\")
            or (len(name) >= 2 and name[1] == ":")
        ):
            raise ValueError(f"Unsafe tar member path (absolute): {name}")
        if any(part == ".." for part in Path(name).parts):
            raise ValueError(f"Unsafe tar member path (traversal): {name}")
        if m.islnk() or m.issym():
            raise ValueError(f"Tar member is a link (not allowed): {name}")


def _safe_extract(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        _validate_tar_members(tar)
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)


# ---------------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------------


def _read_labels_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if row.get("label") not in VALID_LABELS:
            raise ValueError(
                f"Invalid label in {path} for image_id={row.get('image_id')!r}: "
                f"{row.get('label')!r} (must be one of {VALID_LABELS})"
            )
        if not row.get("image_id"):
            raise ValueError(f"Missing image_id in {path}: {row}")
    return rows


def _write_labels_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"image_id": r["image_id"], "label": r["label"]})


# ---------------------------------------------------------------------------
# Per-mode placement
# ---------------------------------------------------------------------------


def _replace_split(target_dir: Path, tar_path: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    labels_path = target_dir / "labels.csv"

    if images_dir.exists():
        for child in images_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    if labels_path.exists():
        labels_path.unlink()

    _safe_extract(tar_path, target_dir)
    return _verify_split(target_dir, prior_count=0)


def _merge_split(target_dir: Path, tar_path: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = target_dir / "labels.csv"

    existing_rows = _read_labels_csv(labels_path) if labels_path.exists() else []
    existing_ids = {r["image_id"] for r in existing_rows}

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        _safe_extract(tar_path, staging)
        new_labels_path = staging / "labels.csv"
        new_images_dir = staging / "images"
        if not new_labels_path.is_file():
            raise ValueError(f"Tarball missing labels.csv at root: {tar_path}")
        if not new_images_dir.is_dir():
            raise ValueError(f"Tarball missing images/ at root: {tar_path}")

        new_rows = _read_labels_csv(new_labels_path)
        new_ids = {r["image_id"] for r in new_rows}
        collisions = sorted(existing_ids & new_ids)
        if collisions:
            shown = collisions[:5]
            suffix = "..." if len(collisions) > 5 else ""
            raise ValueError(
                f"image_id collision in {target_dir}/labels.csv vs incoming tarball: "
                f"{shown}{suffix} (total={len(collisions)}). "
                f"Rename upstream and retry; merge will not overwrite."
            )

        for child in new_images_dir.iterdir():
            if child.is_file():
                shutil.move(str(child), str(images_dir / child.name))

        merged = existing_rows + new_rows
        _write_labels_csv(labels_path, merged)

    return _verify_split(target_dir, prior_count=len(existing_ids))


def _verify_split(target_dir: Path, prior_count: int) -> dict:
    labels_path = target_dir / "labels.csv"
    images_dir = target_dir / "images"
    rows = _read_labels_csv(labels_path)

    missing = [
        r["image_id"]
        for r in rows
        if not (images_dir / f"{r['image_id']}.png").is_file()
    ]
    if missing:
        shown = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"labels.csv references images that are absent under {images_dir}: "
            f"{shown}{suffix} (total={len(missing)})"
        )

    tight = sum(1 for r in rows if r["label"] == "tight")
    loose = sum(1 for r in rows if r["label"] == "loose")
    return {
        "total": len(rows),
        "existing": prior_count,
        "added": len(rows) - prior_count,
        "tight": tight,
        "loose": loose,
    }


# ---------------------------------------------------------------------------
# Default R2 downloader (deferred boto3 import)
# ---------------------------------------------------------------------------


def _r2_download(bucket: str, object_key: str) -> Path:
    """Download an R2 object to a temp file. Reads credentials from env."""
    import boto3  # deferred per CLAUDE.md §4.1

    account_id = os.environ.get("CLOUDFLARE_R2_ACCOUNT_ID")
    access_key = os.environ.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
    secret = os.environ.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
    missing = [
        name for name, val in [
            ("CLOUDFLARE_R2_ACCOUNT_ID", account_id),
            ("CLOUDFLARE_R2_ACCESS_KEY_ID", access_key),
            ("CLOUDFLARE_R2_SECRET_ACCESS_KEY", secret),
        ] if not val
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".tar", prefix=f"r2-{object_key}-")
    os.close(fd)
    client.download_file(bucket, object_key, tmp_path)
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_split(
    split: str,
    dest_root: Path,
    bucket: str,
    *,
    download_fn: Callable[[str, str], Path] = _r2_download,
    splits: dict | None = None,
) -> dict:
    """Fetch one split, place it on disk, and return verification stats.

    Args:
        split: "train" or "test".
        dest_root: Repo root (target dirs are resolved under this).
        bucket: R2 bucket name.
        download_fn: (bucket, object_key) -> local Path. Override in tests.
        splits: Override the SPLITS table (test injection point).
    """
    table = splits if splits is not None else SPLITS
    if split not in table:
        raise ValueError(f"Unknown split: {split!r} (expected one of {sorted(table)})")
    cfg = table[split]
    target = dest_root / cfg["target"]

    tar_path = download_fn(bucket, cfg["object_key"])
    try:
        if cfg["mode"] == "replace":
            stats = _replace_split(target, tar_path)
        elif cfg["mode"] == "merge":
            stats = _merge_split(target, tar_path)
        else:
            raise ValueError(f"Unknown mode for split {split}: {cfg['mode']!r}")
    finally:
        try:
            tar_path.unlink()
        except FileNotFoundError:
            pass

    print(
        f"[fetch] {split}: {stats['total']} images "
        f"(existing={stats['existing']}, added={stats['added']}), "
        f"label dist tight={stats['tight']} loose={stats['loose']}"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fetch helmet dataset tarballs from Cloudflare R2."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to fetch.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("CLOUDFLARE_R2_BUCKET"),
        help="R2 bucket name (defaults to CLOUDFLARE_R2_BUCKET env var).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root under which seeds/ and eval/test_real/ live.",
    )
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set CLOUDFLARE_R2_BUCKET).")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for s in splits:
        fetch_split(s, dest_root=args.dest_root, bucket=args.bucket)


if __name__ == "__main__":
    main()
