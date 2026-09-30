from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from app.config import R2Config
from app.services.r2_storage import (
    R2PublishError,
    build_publish_plan,
    publish_snapshot,
    serialize_latest_pointer,
    snapshot_object_keys,
    validate_latest_pointer,
)
from app.services.serving_snapshot import REQUIRED_SERVING_FILES, build_snapshot_manifest


class _NotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("not found")
        self.response = {
            "Error": {"Code": "404"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _FakeR2Client:
    def __init__(self, *, fail_upload_suffix: str | None = None) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_upload_suffix = fail_upload_suffix

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("head", Key))
        try:
            item = self.objects[Key]
        except KeyError as exc:
            raise _NotFoundError() from exc
        return {
            "ContentLength": len(item["Body"]),
            "Metadata": item.get("Metadata", {}),
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        self.calls.append(("upload", key))
        if self.fail_upload_suffix and key.endswith(self.fail_upload_suffix):
            raise RuntimeError("의도한 업로드 실패")
        self.objects[key] = {
            "Body": Path(filename).read_bytes(),
            **ExtraArgs,
        }

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> None:
        self.calls.append(("put", Key))
        self.objects[Key] = {"Body": bytes(Body), **kwargs}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("get", Key))
        return {"Body": io.BytesIO(self.objects[Key]["Body"])}


def _config() -> R2Config:
    return R2Config(
        LENS_R2_ENDPOINT_URL="https://example.r2.invalid",
        LENS_R2_BUCKET="lens-private",
        LENS_R2_ACCESS_KEY_ID="access",
        LENS_R2_SECRET_ACCESS_KEY="secret",
        LENS_R2_PREFIX="serving/v1",
    )


def _write_required_files(root: Path) -> None:
    for index, name in enumerate(REQUIRED_SERVING_FILES):
        (root / name).write_bytes(f"r2-fixture-{index}-{name}".encode())


def _manifest(root: Path) -> dict[str, Any]:
    return build_snapshot_manifest(
        root,
        snapshot_id="daily-2026-09-22",
        created_at=datetime(2026, 9, 22, 1, 2, 3, tzinfo=timezone.utc),
    )


def test_publish_uploads_latest_pointer_last(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = _manifest(tmp_path)
    config = _config()
    client = _FakeR2Client()

    result = publish_snapshot(
        tmp_path,
        manifest,
        config=config,
        client=client,
        published_at=datetime(2026, 9, 22, 4, 5, 6, tzinfo=timezone.utc),
    )

    keys = snapshot_object_keys(config, manifest["snapshot_id"])
    mutations = [call for call in client.calls if call[0] in {"upload", "put"}]
    assert mutations[-1] == ("put", keys["latest_key"])
    assert result["uploaded_object_count"] == len(REQUIRED_SERVING_FILES) + 1
    assert result["skipped_object_count"] == 0
    pointer = json.loads(client.objects[keys["latest_key"]]["Body"])
    validate_latest_pointer(pointer)
    assert pointer["snapshot_id"] == manifest["snapshot_id"]
    assert pointer["manifest_key"] == keys["manifest_key"]
    assert pointer["published_at"] == "2026-09-22T04:05:06Z"


def test_upload_failure_does_not_publish_latest(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = _manifest(tmp_path)
    config = _config()
    client = _FakeR2Client(fail_upload_suffix="predictions_band_1d.parquet")

    with pytest.raises(R2PublishError, match="완료되지 않았습니다"):
        publish_snapshot(tmp_path, manifest, config=config, client=client)

    latest_key = snapshot_object_keys(config, manifest["snapshot_id"])["latest_key"]
    assert latest_key not in client.objects


def test_existing_different_release_object_is_not_overwritten(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = _manifest(tmp_path)
    config = _config()
    client = _FakeR2Client()
    keys = snapshot_object_keys(config, manifest["snapshot_id"])
    first_key = f"{keys['release_prefix']}/{REQUIRED_SERVING_FILES[0]}"
    client.objects[first_key] = {
        "Body": b"different",
        "Metadata": {"sha256": "0" * 64},
    }

    with pytest.raises(R2PublishError, match="다른 객체가 이미"):
        publish_snapshot(tmp_path, manifest, config=config, client=client)

    assert client.objects[first_key]["Body"] == b"different"
    assert keys["latest_key"] not in client.objects


def test_republish_same_snapshot_skips_immutable_objects(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = _manifest(tmp_path)
    config = _config()
    client = _FakeR2Client()
    publish_snapshot(tmp_path, manifest, config=config, client=client)
    client.calls.clear()

    result = publish_snapshot(tmp_path, manifest, config=config, client=client)

    assert not [call for call in client.calls if call[0] == "upload"]
    assert result["uploaded_object_count"] == 0
    assert result["skipped_object_count"] == len(REQUIRED_SERVING_FILES) + 1


def test_publish_plan_needs_no_credentials_or_network(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    manifest = _manifest(tmp_path)
    config = R2Config(LENS_R2_PREFIX="custom/v1")

    plan = build_publish_plan(config, manifest, tmp_path)

    assert plan["bucket"] is None
    assert plan["release_prefix"] == "custom/v1/releases/daily-2026-09-22"
    assert plan["latest_key"] == "custom/v1/latest.json"
    assert len(plan["release_objects"]) == len(REQUIRED_SERVING_FILES) + 1


def test_latest_pointer_rejects_parent_path() -> None:
    pointer = {
        "schema_version": 1,
        "snapshot_id": "daily-2026-09-22",
        "published_at": "2026-09-22T04:05:06Z",
        "manifest_key": "serving/v1/../snapshot_manifest.json",
        "manifest_sha256": "1" * 64,
        "content_sha256": "2" * 64,
    }

    with pytest.raises(R2PublishError, match="manifest_key"):
        serialize_latest_pointer(pointer)
