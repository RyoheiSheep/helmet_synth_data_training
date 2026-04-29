"""Tests for scripts/fetch_data.py.

These tests do not import boto3 or hit the network: they inject a fake
download_fn that returns a locally-built tarball.
"""

import csv
import io
import tarfile
from pathlib import Path

import pytest

from scripts.fetch_data import fetch_split


PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cf00000003000000180001270b8c000000004945"
    "4e44ae426082"
)


def _build_tarball(tar_path: Path, image_ids_with_labels: list[tuple[str, str]],
                   extra_members: list[tuple[str, bytes]] | None = None) -> None:
    """Build a tarball with images/<id>.png and labels.csv at the root."""
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["image_id", "label"])
    writer.writeheader()
    for image_id, label in image_ids_with_labels:
        writer.writerow({"image_id": image_id, "label": label})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("labels.csv")
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))

        for image_id, _ in image_ids_with_labels:
            info = tarfile.TarInfo(f"images/{image_id}.png")
            info.size = len(PNG_BYTES)
            tar.addfile(info, io.BytesIO(PNG_BYTES))

        for name, payload in (extra_members or []):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _make_download_fn(tar_path: Path):
    def _download(bucket: str, key: str) -> Path:
        # Return a copy so fetch_split can unlink it without affecting the source.
        copy = tar_path.with_suffix(".copy.tar")
        copy.write_bytes(tar_path.read_bytes())
        return copy
    return _download


def _read_labels(target: Path) -> list[dict]:
    with open(target / "labels.csv", "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Replace mode (test split)
# ---------------------------------------------------------------------------


class TestReplaceMode:
    def test_wipes_existing_and_extracts(self, tmp_path):
        target = tmp_path / "eval" / "test_real"
        (target / "images").mkdir(parents=True)
        (target / "images" / "stale.png").write_bytes(b"old")
        (target / "images" / ".gitkeep").write_text("")
        (target / "labels.csv").write_text("image_id,label\nstale,tight\n")

        tar = tmp_path / "test_dataset.tar"
        _build_tarball(tar, [("realA", "tight"), ("realB", "loose")])

        stats = fetch_split(
            "test",
            dest_root=tmp_path,
            bucket="bucket",
            download_fn=_make_download_fn(tar),
        )

        assert stats == {"total": 2, "existing": 0, "added": 2, "tight": 1, "loose": 1}
        assert not (target / "images" / "stale.png").exists()
        assert (target / "images" / ".gitkeep").exists()  # preserved
        ids = {r["image_id"] for r in _read_labels(target)}
        assert ids == {"realA", "realB"}

    def test_works_when_target_does_not_exist(self, tmp_path):
        tar = tmp_path / "test_dataset.tar"
        _build_tarball(tar, [("only", "tight")])

        stats = fetch_split(
            "test",
            dest_root=tmp_path,
            bucket="bucket",
            download_fn=_make_download_fn(tar),
        )

        assert stats["total"] == 1
        assert (tmp_path / "eval" / "test_real" / "images" / "only.png").is_file()


# ---------------------------------------------------------------------------
# Merge mode (train split)
# ---------------------------------------------------------------------------


class TestMergeMode:
    def test_appends_new_rows_keeps_existing(self, tmp_path):
        target = tmp_path / "seeds"
        (target / "images").mkdir(parents=True)
        (target / "images" / "tight1.png").write_bytes(PNG_BYTES)
        (target / "images" / "loose1.png").write_bytes(PNG_BYTES)
        (target / "labels.csv").write_text(
            "image_id,label\ntight1,tight\nloose1,loose\n"
        )

        tar = tmp_path / "train_dataset.tar"
        _build_tarball(tar, [("genA", "tight"), ("genB", "loose")])

        stats = fetch_split(
            "train",
            dest_root=tmp_path,
            bucket="bucket",
            download_fn=_make_download_fn(tar),
        )

        assert stats == {"total": 4, "existing": 2, "added": 2, "tight": 2, "loose": 2}
        ids = {r["image_id"] for r in _read_labels(target)}
        assert ids == {"tight1", "loose1", "genA", "genB"}
        for stem in ("tight1", "loose1", "genA", "genB"):
            assert (target / "images" / f"{stem}.png").is_file()

    def test_collision_aborts_and_preserves_existing(self, tmp_path):
        target = tmp_path / "seeds"
        (target / "images").mkdir(parents=True)
        (target / "images" / "tight1.png").write_bytes(PNG_BYTES)
        (target / "labels.csv").write_text("image_id,label\ntight1,tight\n")

        tar = tmp_path / "train_dataset.tar"
        _build_tarball(tar, [("tight1", "loose"), ("genB", "loose")])

        with pytest.raises(ValueError, match="collision"):
            fetch_split(
                "train",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=_make_download_fn(tar),
            )

        rows = _read_labels(target)
        assert len(rows) == 1
        assert rows[0] == {"image_id": "tight1", "label": "tight"}
        assert not (target / "images" / "genB.png").exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_path_traversal_rejected(self, tmp_path):
        tar = tmp_path / "evil.tar"
        with tarfile.open(tar, "w") as t:
            payload = b"image_id,label\nx,tight\n"
            info = tarfile.TarInfo("labels.csv")
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
            evil = tarfile.TarInfo("../escape.png")
            evil.size = len(PNG_BYTES)
            t.addfile(evil, io.BytesIO(PNG_BYTES))

        with pytest.raises(ValueError, match="Unsafe tar member"):
            fetch_split(
                "test",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=_make_download_fn(tar),
            )

    def test_absolute_path_rejected(self, tmp_path):
        tar = tmp_path / "evil.tar"
        with tarfile.open(tar, "w") as t:
            info = tarfile.TarInfo("/etc/passwd")
            info.size = 0
            t.addfile(info, io.BytesIO(b""))

        with pytest.raises(ValueError, match="Unsafe tar member"):
            fetch_split(
                "test",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=_make_download_fn(tar),
            )

    def test_bad_label_rejected(self, tmp_path):
        tar = tmp_path / "bad.tar"
        _build_tarball(tar, [("x", "weird")])
        with pytest.raises(ValueError, match="Invalid label"):
            fetch_split(
                "test",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=_make_download_fn(tar),
            )

    def test_missing_image_rejected(self, tmp_path):
        tar = tmp_path / "missing.tar"
        csv_payload = b"image_id,label\nphantom,tight\n"
        with tarfile.open(tar, "w") as t:
            info = tarfile.TarInfo("labels.csv")
            info.size = len(csv_payload)
            t.addfile(info, io.BytesIO(csv_payload))
            # No images/ at all.

        with pytest.raises(FileNotFoundError, match="absent"):
            fetch_split(
                "test",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=_make_download_fn(tar),
            )

    def test_unknown_split_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown split"):
            fetch_split(
                "garbage",
                dest_root=tmp_path,
                bucket="bucket",
                download_fn=lambda b, k: tmp_path / "never.tar",
            )
