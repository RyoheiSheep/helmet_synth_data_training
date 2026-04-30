"""Upload Loop N results to Cloudflare R2 as tarballs.

Bundles per-category directories into gzip tarballs, computes SHA256, and
uploads to R2 with timestamped keys. A separate manifest JSON lists every
uploaded item with its size, hash, and the originating git commit.

R2 layout:
    loop_{N}_model_{ts}.tar.gz       — models/loop_{N}/
    loop_{N}_eval_{ts}.tar.gz        — eval_out/loop_{N}/
    loop_{N}_data_{ts}.tar.gz        — generated/loop_{N}/ + screened/loop_{N}/ + dataset/loop_{N}/
    loop_{N}_manifest_{ts}.json      — index (sha256, size, git_commit)

Credentials are read from env: CLOUDFLARE_R2_ACCOUNT_ID,
CLOUDFLARE_R2_ACCESS_KEY_ID, CLOUDFLARE_R2_SECRET_ACCESS_KEY,
CLOUDFLARE_R2_BUCKET (CLI --bucket overrides the bucket env var).
"""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ITEM_SPECS: dict[str, list[str]] = {
    "model": ["models/loop_{loop}"],
    "eval": ["eval_out/loop_{loop}"],
    "data": [
        "generated/loop_{loop}",
        "screened/loop_{loop}",
        "dataset/loop_{loop}",
    ],
}

CATEGORIES = list(ITEM_SPECS.keys())
CHUNK_SIZE = 1024 * 1024  # 1 MiB


# ---------------------------------------------------------------------------
# Hashing & tarballing
# ---------------------------------------------------------------------------


def _hash_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _make_tarball(src_dirs: list[Path], dst: Path, root: Path) -> dict:
    """Create dst (gzip tar) containing src_dirs (paths relative to root).

    Returns a dict: {sha256, size_bytes, file_count}.
    """
    for d in src_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Source directory not found: {d}")

    file_count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tar:
        for d in src_dirs:
            arcname = d.relative_to(root).as_posix()
            tar.add(d, arcname=arcname)
            for sub in d.rglob("*"):
                if sub.is_file():
                    file_count += 1

    sha, size = _hash_and_size(dst)
    return {"sha256": sha, "size_bytes": size, "file_count": file_count}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _get_git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def _build_manifest(
    loop: int,
    timestamp: str,
    git_commit: str,
    items: list[dict],
) -> dict:
    return {
        "loop": loop,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "items": items,
    }


# ---------------------------------------------------------------------------
# Default R2 uploader (deferred boto3 import)
# ---------------------------------------------------------------------------


def _r2_upload(bucket: str, key: str, path: Path) -> None:
    """Upload a local file to an R2 object key. Reads credentials from env."""
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
    client.upload_file(str(path), bucket, key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_includes(include: list[str]) -> list[str]:
    if "all" in include:
        return list(CATEGORIES)
    unknown = [x for x in include if x not in ITEM_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown --include values: {unknown} (expected: {CATEGORIES + ['all']})"
        )
    seen: set[str] = set()
    out: list[str] = []
    for x in include:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def run_upload(
    loop: int,
    include: list[str],
    bucket: str,
    *,
    dest_root: Path,
    work_dir: Path,
    dry_run: bool = False,
    cleanup: bool = True,
    upload_fn: Callable[[str, str, Path], None] = _r2_upload,
) -> dict:
    """Build per-category tarballs + a manifest, optionally upload to R2.

    Args:
        loop: Loop number.
        include: Subset of CATEGORIES, or ["all"].
        bucket: R2 bucket name (unused when dry_run=True).
        dest_root: Repo root (source dirs are resolved relative to this).
        work_dir: Local staging directory for built tarballs.
        dry_run: If True, build artifacts locally but skip upload.
        cleanup: If True (and not dry_run), remove local tarballs after upload.
        upload_fn: (bucket, key, path) -> None. Override in tests.

    Returns: summary dict.
    """
    categories = _resolve_includes(include)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    git_commit = _get_git_commit(dest_root)
    work_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for cat in categories:
        src_dirs = [dest_root / s.format(loop=loop) for s in ITEM_SPECS[cat]]
        key = f"loop_{loop}_{cat}_{timestamp}.tar.gz"
        tar_path = work_dir / key
        info = _make_tarball(src_dirs, tar_path, root=dest_root)
        items.append({
            "category": cat,
            "key": key,
            "sha256": info["sha256"],
            "size_bytes": info["size_bytes"],
            "file_count": info["file_count"],
            "src_dirs": [s.relative_to(dest_root).as_posix() for s in src_dirs],
        })
        print(
            f"[upload] built {key} "
            f"({info['size_bytes']:,} bytes, {info['file_count']} files, "
            f"sha256={info['sha256'][:12]}...)"
        )

    manifest = _build_manifest(loop, timestamp, git_commit, items)
    manifest_key = f"loop_{loop}_manifest_{timestamp}.json"
    manifest_path = work_dir / manifest_key
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not dry_run:
        for item in items:
            upload_fn(bucket, item["key"], work_dir / item["key"])
            print(f"[upload] -> r2://{bucket}/{item['key']}")
        upload_fn(bucket, manifest_key, manifest_path)
        print(f"[upload] -> r2://{bucket}/{manifest_key}")

        if cleanup:
            for item in items:
                (work_dir / item["key"]).unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
    else:
        print(f"[upload] dry-run: skipped R2 upload, artifacts in {work_dir}")

    return {
        "loop": loop,
        "timestamp": timestamp,
        "manifest_key": manifest_key,
        "items": items,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Upload Loop N results (tarballs + manifest) to Cloudflare R2."
    )
    parser.add_argument("--loop", type=int, required=True, help="Loop number to upload.")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["all"],
        choices=CATEGORIES + ["all"],
        help="Categories to upload (default: all).",
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
        help="Repo root containing models/, eval_out/, etc.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tmp/upload_results"),
        help="Local staging directory for tarballs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build tarballs and manifest locally, but skip R2 upload.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep local tarballs after a successful upload (for inspection).",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.bucket:
        parser.error("--bucket is required (or set CLOUDFLARE_R2_BUCKET).")

    result = run_upload(
        loop=args.loop,
        include=args.include,
        bucket=args.bucket or "",
        dest_root=args.dest_root,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
        cleanup=not args.no_cleanup,
    )

    print(
        f"Uploaded loop={result['loop']} "
        f"keys={len(result['items'])} "
        f"manifest={result['manifest_key']} "
        f"dry_run={result['dry_run']}"
    )


if __name__ == "__main__":
    main()
