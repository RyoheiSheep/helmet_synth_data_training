"""Tests for scripts/upload_results.py.

These tests do not import boto3 or hit the network: they inject a fake
upload_fn that records calls.
"""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.upload_results import run_upload


def _make_loop_tree(root: Path, loop: int) -> None:
    """Create a dummy Loop N output tree with a few files in each category."""
    (root / f"models/loop_{loop}/lora_weights").mkdir(parents=True)
    (root / f"models/loop_{loop}/lora_weights/adapter_config.json").write_text("{}")
    (root / f"models/loop_{loop}/training_log.json").write_text('{"status": "ok"}')

    (root / f"eval_out/loop_{loop}").mkdir(parents=True)
    (root / f"eval_out/loop_{loop}/test_real.jsonl").write_text('{"a":1}\n')
    (root / f"eval_out/loop_{loop}/metrics.json").write_text('{"acc": 0.9}')

    (root / f"generated/loop_{loop}/images").mkdir(parents=True)
    (root / f"generated/loop_{loop}/images/gen_01_0001.png").write_bytes(b"PNG")
    (root / f"generated/loop_{loop}/meta.csv").write_text("image_id,label\nx,tight\n")

    (root / f"screened/loop_{loop}").mkdir(parents=True)
    (root / f"screened/loop_{loop}/screening.csv").write_text("image_id,keep\nx,True\n")
    (root / f"screened/loop_{loop}/labeled.jsonl").write_text(
        '{"image_id":"x","label":"tight"}\n'
    )

    (root / f"dataset/loop_{loop}").mkdir(parents=True)
    (root / f"dataset/loop_{loop}/train.jsonl").write_text('{"image_path":"x"}\n')
    (root / f"dataset/loop_{loop}/stats.json").write_text('{"total":1}')


def _recording_upload_fn():
    calls: list[tuple[str, str, bytes]] = []

    def _fn(bucket: str, key: str, path: Path) -> None:
        # Capture bytes because cleanup may delete the source after upload.
        calls.append((bucket, key, Path(path).read_bytes()))

    return _fn, calls


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_dry_run_builds_tarballs_and_manifest(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "tmp/upload_results"

        upload_fn, calls = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["all"],
            bucket="bucket",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=True,
            upload_fn=upload_fn,
        )

        assert calls == []  # dry-run skips upload
        assert len(result["items"]) == 3
        cats = {it["category"] for it in result["items"]}
        assert cats == {"model", "eval", "data"}

        for item in result["items"]:
            tar_path = work / item["key"]
            assert tar_path.is_file()
            assert tar_path.stat().st_size == item["size_bytes"]
            sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
            assert sha == item["sha256"]
            assert item["file_count"] >= 1

        manifest = json.loads((work / result["manifest_key"]).read_text())
        assert manifest["loop"] == 1
        assert manifest["timestamp"] == result["timestamp"]
        assert len(manifest["items"]) == 3

    def test_real_upload_path_and_cleanup(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, calls = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["all"],
            bucket="my-bucket",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=False,
            cleanup=True,
            upload_fn=upload_fn,
        )

        # 3 tarballs + 1 manifest
        assert len(calls) == 4
        keys = [c[1] for c in calls]
        assert any(k.startswith("loop_1_model_") and k.endswith(".tar.gz") for k in keys)
        assert any(k.startswith("loop_1_eval_") and k.endswith(".tar.gz") for k in keys)
        assert any(k.startswith("loop_1_data_") and k.endswith(".tar.gz") for k in keys)
        assert any(k.startswith("loop_1_manifest_") and k.endswith(".json") for k in keys)
        assert all(c[0] == "my-bucket" for c in calls)

        # cleanup removed local artifacts
        for item in result["items"]:
            assert not (work / item["key"]).exists()
        assert not (work / result["manifest_key"]).exists()

    def test_no_cleanup_keeps_artifacts(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, _ = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["all"],
            bucket="b",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=False,
            cleanup=False,
            upload_fn=upload_fn,
        )

        for item in result["items"]:
            assert (work / item["key"]).is_file()
        assert (work / result["manifest_key"]).is_file()


# ---------------------------------------------------------------------------
# Tar contents
# ---------------------------------------------------------------------------


class TestTarContent:
    def test_model_tar_contains_expected_paths(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, _ = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["model"],
            bucket="b",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=True,
            upload_fn=upload_fn,
        )

        item = result["items"][0]
        with tarfile.open(work / item["key"], "r:gz") as tar:
            names = set(tar.getnames())
        assert "models/loop_1/training_log.json" in names
        assert "models/loop_1/lora_weights/adapter_config.json" in names

    def test_data_tar_contains_three_subtrees(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, _ = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["data"],
            bucket="b",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=True,
            upload_fn=upload_fn,
        )

        item = result["items"][0]
        with tarfile.open(work / item["key"], "r:gz") as tar:
            names = set(tar.getnames())
        assert "generated/loop_1/meta.csv" in names
        assert "generated/loop_1/images/gen_01_0001.png" in names
        assert "screened/loop_1/screening.csv" in names
        assert "dataset/loop_1/train.jsonl" in names


# ---------------------------------------------------------------------------
# Filters & errors
# ---------------------------------------------------------------------------


class TestFilters:
    def test_include_eval_only(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, _ = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["eval"],
            bucket="b",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=True,
            upload_fn=upload_fn,
        )

        assert len(result["items"]) == 1
        assert result["items"][0]["category"] == "eval"

    def test_unknown_include_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown --include"):
            run_upload(
                loop=1,
                include=["bogus"],
                bucket="b",
                dest_root=tmp_path,
                work_dir=tmp_path / "w",
                dry_run=True,
            )


class TestErrors:
    def test_missing_source_dir_raises(self, tmp_path):
        upload_fn, _ = _recording_upload_fn()
        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            run_upload(
                loop=1,
                include=["model"],
                bucket="b",
                dest_root=tmp_path,
                work_dir=tmp_path / "w",
                dry_run=True,
                upload_fn=upload_fn,
            )


# ---------------------------------------------------------------------------
# Manifest integrity
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_sha_matches_tar_bytes(self, tmp_path):
        _make_loop_tree(tmp_path, loop=1)
        work = tmp_path / "work"

        upload_fn, _ = _recording_upload_fn()
        result = run_upload(
            loop=1,
            include=["all"],
            bucket="b",
            dest_root=tmp_path,
            work_dir=work,
            dry_run=True,
            upload_fn=upload_fn,
        )

        manifest = json.loads((work / result["manifest_key"]).read_text())
        for item in manifest["items"]:
            tar_path = work / item["key"]
            actual_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
            assert actual_sha == item["sha256"]
            assert tar_path.stat().st_size == item["size_bytes"]
        assert "git_commit" in manifest
