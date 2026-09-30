from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.services.serving_snapshot import (
    REQUIRED_SERVING_FILES,
    SNAPSHOT_MANIFEST_FILENAME,
    SnapshotValidationError,
    build_snapshot_manifest,
    load_snapshot_manifest,
    validate_manifest_document,
    validate_snapshot_directory,
    write_snapshot_manifest,
)


def _write_required_files(root: Path) -> None:
    for index, name in enumerate(REQUIRED_SERVING_FILES):
        (root / name).write_bytes(f"fixture-{index}-{name}".encode())


def test_frozen_snapshot_hashes_match_bytes_stored_in_git() -> None:
    root = Path(__file__).resolve().parents[2]
    bootstrap = root / "backend" / "data" / "bootstrap" / "v1"
    manifest = load_snapshot_manifest(bootstrap / SNAPSHOT_MANIFEST_FILENAME)
    for item in manifest["files"]:
        # 작업 폴더에서만 맞는 해시가 아니라 Linux 배포에 쓰는 Git 객체도 검사한다.
        stored = subprocess.run(
            ["git", "cat-file", "blob", f":backend/data/bootstrap/v1/{item['name']}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        assert len(stored) == item["size_bytes"], item["name"]
        assert hashlib.sha256(stored).hexdigest() == item["sha256"], item["name"]


def test_build_and_validate_snapshot_manifest(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    created_at = datetime(2026, 9, 22, 3, 4, 5, tzinfo=timezone.utc)

    manifest = build_snapshot_manifest(tmp_path, created_at=created_at)
    summary = validate_snapshot_directory(manifest, tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["snapshot_id"].startswith("20260922T030405Z-")
    assert manifest["created_at"] == "2026-09-22T03:04:05Z"
    assert tuple(item["name"] for item in manifest["files"]) == REQUIRED_SERVING_FILES
    assert summary["file_count"] == len(REQUIRED_SERVING_FILES)
    assert summary["total_size_bytes"] == sum(
        (tmp_path / name).stat().st_size for name in REQUIRED_SERVING_FILES
    )


def test_write_and_load_manifest_atomically(tmp_path: Path) -> None:
    _write_required_files(tmp_path)

    target, written = write_snapshot_manifest(
        tmp_path,
        snapshot_id="daily-2026-09-22",
    )
    loaded = load_snapshot_manifest(target)

    assert target == tmp_path / SNAPSHOT_MANIFEST_FILENAME
    assert loaded == written
    assert json.loads(target.read_text(encoding="utf-8"))["snapshot_id"] == "daily-2026-09-22"
    assert list(tmp_path.glob(f".{SNAPSHOT_MANIFEST_FILENAME}.*.tmp")) == []


def test_missing_required_file_is_rejected(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    (tmp_path / REQUIRED_SERVING_FILES[-1]).unlink()

    with pytest.raises(SnapshotValidationError, match="필수 서빙 파일"):
        build_snapshot_manifest(tmp_path)


def test_same_size_file_corruption_is_rejected(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = build_snapshot_manifest(tmp_path)
    target = tmp_path / REQUIRED_SERVING_FILES[0]
    target.write_bytes(b"x" * target.stat().st_size)

    with pytest.raises(SnapshotValidationError, match="파일 해시"):
        validate_snapshot_directory(manifest, tmp_path)


def test_manifest_file_list_is_strict(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = build_snapshot_manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["files"][0]["name"] = "../outside.json"

    with pytest.raises(SnapshotValidationError, match="파일 목록이나 순서"):
        validate_manifest_document(changed)


def test_manifest_content_fingerprint_is_checked(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = build_snapshot_manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["files"][0]["sha256"] = "0" * 64

    with pytest.raises(SnapshotValidationError, match="content_sha256"):
        validate_manifest_document(changed)
