from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from app.config import R2Config
from app.services.r2_storage import publish_snapshot, snapshot_object_keys
from app.services.serving_paths import (
    activate_serving_data_dir,
    get_serving_data_dir,
    reset_active_serving_data_dir,
)
from app.services.serving_snapshot import (
    REQUIRED_SERVING_FILES,
    SnapshotValidationError,
    build_snapshot_manifest,
)
from app.services.serving_sync import (
    R2SyncError,
    activate_bootstrap_serving_snapshot,
    get_serving_sync_state,
    inspect_latest_serving_snapshot,
    reset_serving_sync_state,
    sync_latest_serving_snapshot,
)
from fastapi.testclient import TestClient


class _NotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("not found")
        self.response = {
            "Error": {"Code": "404"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _FakeR2Client:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []

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
        self.objects[key] = {"Body": Path(filename).read_bytes(), **ExtraArgs}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> None:
        self.calls.append(("put", Key))
        self.objects[Key] = {"Body": bytes(Body), **kwargs}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("get", Key))
        payload = self.objects[Key]["Body"]
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append(("download", key))
        Path(filename).write_bytes(self.objects[key]["Body"])


def _config() -> R2Config:
    return R2Config(
        LENS_R2_ENDPOINT_URL="https://example.r2.invalid",
        LENS_R2_BUCKET="lens-private",
        LENS_R2_ACCESS_KEY_ID="access",
        LENS_R2_SECRET_ACCESS_KEY="secret",
        LENS_R2_PREFIX="serving/v1",
    )


def _write_required_files(root: Path) -> None:
    root.mkdir(parents=True)
    for index, name in enumerate(REQUIRED_SERVING_FILES):
        (root / name).write_bytes(f"sync-fixture-{index}-{name}".encode())


def _published_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], R2Config, _FakeR2Client]:
    source = tmp_path / "source"
    _write_required_files(source)
    manifest = build_snapshot_manifest(
        source,
        snapshot_id="daily-2026-09-22",
        created_at=datetime(2026, 9, 22, 1, 2, 3, tzinfo=timezone.utc),
    )
    config = _config()
    client = _FakeR2Client()
    publish_snapshot(
        source,
        manifest,
        config=config,
        client=client,
        published_at=datetime(2026, 9, 22, 4, 5, 6, tzinfo=timezone.utc),
    )
    client.calls.clear()
    return source, manifest, config, client


def setup_function() -> None:
    reset_active_serving_data_dir()
    reset_serving_sync_state()


def teardown_function() -> None:
    reset_active_serving_data_dir()
    reset_serving_sync_state()


def test_sync_downloads_validates_and_activates_snapshot(tmp_path: Path) -> None:
    source, manifest, config, client = _published_fixture(tmp_path)
    cache_root = tmp_path / "cache"

    result = sync_latest_serving_snapshot(
        config=config,
        client=client,
        cache_dir=cache_root,
    )

    active_dir = cache_root / manifest["snapshot_id"]
    assert get_serving_data_dir() == active_dir.resolve()
    assert result["status"] == "ready"
    assert result["changed"] is True
    assert result["reused_cache"] is False
    assert len([call for call in client.calls if call[0] == "download"]) == len(
        REQUIRED_SERVING_FILES
    )
    for name in REQUIRED_SERVING_FILES:
        assert (active_dir / name).read_bytes() == (source / name).read_bytes()
    assert get_serving_sync_state()["snapshot_id"] == manifest["snapshot_id"]


def test_bootstrap_snapshot_is_verified_before_activation(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    _write_required_files(bootstrap)
    manifest = build_snapshot_manifest(
        bootstrap,
        snapshot_id="bootstrap-frozen",
        created_at=datetime(2026, 9, 22, 0, 0, 0, tzinfo=timezone.utc),
    )
    (bootstrap / "snapshot_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"LENS_SERVING_BOOTSTRAP_DIR": str(bootstrap)}):
        result = activate_bootstrap_serving_snapshot()

    assert result["status"] == "fallback_ready"
    assert result["source"] == "bootstrap"
    assert result["fallback_ready"] is True
    assert get_serving_data_dir() == bootstrap.resolve()


def test_corrupt_bootstrap_snapshot_is_not_marked_ready(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    _write_required_files(bootstrap)
    manifest = build_snapshot_manifest(
        bootstrap,
        snapshot_id="bootstrap-frozen",
        created_at=datetime(2026, 9, 22, 0, 0, 0, tzinfo=timezone.utc),
    )
    (bootstrap / "snapshot_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (bootstrap / REQUIRED_SERVING_FILES[0]).write_bytes(b"corrupt")

    with (
        patch.dict(os.environ, {"LENS_SERVING_BOOTSTRAP_DIR": str(bootstrap)}),
        pytest.raises(SnapshotValidationError, match="스냅샷 파일"),
    ):
        activate_bootstrap_serving_snapshot()

    state = get_serving_sync_state()
    assert state["status"] == "fallback_error"
    assert state["fallback_ready"] is False


def test_r2_failure_keeps_verified_bootstrap_and_reports_degraded(tmp_path: Path) -> None:
    from app.main import app

    bootstrap = tmp_path / "bootstrap"
    _write_required_files(bootstrap)
    manifest = build_snapshot_manifest(bootstrap, snapshot_id="bootstrap-frozen")
    (bootstrap / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    env = {
        "LENS_DATA_BACKEND": "local",
        "LENS_SERVING_DATA_DIR": str(bootstrap),
        "LENS_SERVING_BOOTSTRAP_DIR": str(bootstrap),
        "LENS_R2_ENDPOINT_URL": "https://example.r2.invalid",
        "LENS_R2_BUCKET": "lens-private",
        "LENS_R2_ACCESS_KEY_ID": "access",
        "LENS_R2_SECRET_ACCESS_KEY": "secret",
    }
    with patch.dict(os.environ, env, clear=True):
        activate_bootstrap_serving_snapshot()
        with pytest.raises(R2SyncError):
            sync_latest_serving_snapshot(
                config=_config(), client=_FakeR2Client(), cache_dir=tmp_path / "cache"
            )
        response = TestClient(app).get("/api/v1/health/ready")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["source"] == "bootstrap"
    assert data["snapshot_id"] == "bootstrap-frozen"
    assert data["checks"]["bootstrap_verified"] is True
    assert get_serving_data_dir() == bootstrap.resolve()
    assert get_serving_sync_state()["last_error"]


def test_corrupt_download_keeps_previous_active_directory(tmp_path: Path) -> None:
    _, manifest, config, client = _published_fixture(tmp_path)
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    activate_serving_data_dir(fallback)
    keys = snapshot_object_keys(config, manifest["snapshot_id"])
    corrupt_key = f"{keys['release_prefix']}/{REQUIRED_SERVING_FILES[0]}"
    client.objects[corrupt_key]["Body"] = b"corrupt"
    cache_root = tmp_path / "cache"

    with pytest.raises(R2SyncError, match="동기화에 실패"):
        sync_latest_serving_snapshot(
            config=config,
            client=client,
            cache_dir=cache_root,
        )

    assert get_serving_data_dir() == fallback.resolve()
    assert get_serving_sync_state()["status"] == "error"
    assert not list(cache_root.glob(".*.tmp"))


def test_pointer_cannot_redirect_to_another_manifest_key(tmp_path: Path) -> None:
    _, manifest, config, client = _published_fixture(tmp_path)
    keys = snapshot_object_keys(config, manifest["snapshot_id"])
    pointer = json.loads(client.objects[keys["latest_key"]]["Body"])
    pointer["manifest_key"] = "serving/v1/releases/other/snapshot_manifest.json"
    client.objects[keys["latest_key"]]["Body"] = (
        json.dumps(pointer, sort_keys=True) + "\n"
    ).encode()

    with pytest.raises(R2SyncError, match="예상한 버전 경로"):
        sync_latest_serving_snapshot(
            config=config,
            client=client,
            cache_dir=tmp_path / "cache",
        )

    assert not [call for call in client.calls if call[0] == "download"]


def test_second_sync_reuses_verified_cache(tmp_path: Path) -> None:
    _, _, config, client = _published_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    sync_latest_serving_snapshot(config=config, client=client, cache_dir=cache_root)
    client.calls.clear()

    result = sync_latest_serving_snapshot(config=config, client=client, cache_dir=cache_root)

    assert result["changed"] is False
    assert result["reused_cache"] is True
    assert not [call for call in client.calls if call[0] == "download"]


def test_remote_audit_verifies_bytes_without_changing_serving_state(tmp_path: Path) -> None:
    _, manifest, config, client = _published_fixture(tmp_path)
    state_before = get_serving_sync_state()
    directory_before = get_serving_data_dir()
    result = inspect_latest_serving_snapshot(config=config, client=client, verify_files=True)
    assert result["manifest"] == manifest
    assert result["files_verified"] is True
    assert get_serving_sync_state() == state_before
    assert get_serving_data_dir() == directory_before
    assert not [call for call in client.calls if call[0] in {"download", "put", "upload"}]


def test_remote_audit_rejects_corrupt_file_bytes(tmp_path: Path) -> None:
    _, manifest, config, client = _published_fixture(tmp_path)
    keys = snapshot_object_keys(config, manifest["snapshot_id"])
    client.objects[f"{keys['release_prefix']}/{REQUIRED_SERVING_FILES[0]}"]["Body"] = b"corrupt"
    with pytest.raises(R2SyncError, match="무결성"):
        inspect_latest_serving_snapshot(config=config, client=client, verify_files=True)


def test_startup_r2_failure_serves_real_frozen_price_and_predictions() -> None:
    from datetime import date

    from app.main import app

    from backend.scripts.verify_deployed_serving import verify_deployed_serving

    bootstrap = Path(__file__).resolve().parents[1] / "data" / "bootstrap" / "v1"
    env = {
        "LENS_DATA_BACKEND": "local",
        "LENS_SERVING_DATA_DIR": str(bootstrap),
        "LENS_SERVING_BOOTSTRAP_DIR": str(bootstrap),
        "LENS_R2_ENDPOINT_URL": "https://example.r2.invalid",
        "LENS_R2_BUCKET": "lens-private",
        "LENS_R2_ACCESS_KEY_ID": "access",
        "LENS_R2_SECRET_ACCESS_KEY": "secret",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("app.services.serving_sync.create_r2_client", return_value=_FakeR2Client()),
        TestClient(app) as client,
    ):

        def get_json(url: str) -> dict[str, Any]:
            parsed = urlsplit(url)
            response = client.get(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
            response.raise_for_status()
            return response.json()

        result = verify_deployed_serving(
            base_url="http://testserver",
            get_json=get_json,
            today=date(2026, 9, 26),
        )

    assert result["source"] == "bootstrap"
    assert result["fallback_active"] is True
    assert result["result"] == "FALLBACK"
    assert result["r2_sync_status"] == "error"


def test_admin_sync_endpoint_requires_token_and_returns_result() -> None:
    from app.main import app

    expected = {
        "status": "ready",
        "source": "r2",
        "snapshot_id": "daily-2026-09-22",
        "changed": True,
    }
    client = TestClient(app)
    with (
        patch.dict(os.environ, {"LENS_ADMIN_RELOAD_TOKEN": "secret"}, clear=True),
        patch(
            "app.routers.v1.admin.sync_latest_serving_snapshot",
            return_value=expected,
        ) as sync,
    ):
        denied = client.post("/api/v1/admin/sync-serving")
        response = client.post(
            "/api/v1/admin/sync-serving",
            headers={"X-Lens-Admin-Token": "secret"},
        )

    assert denied.status_code == 403
    assert response.status_code == 200
    assert response.json()["data"] == expected
    sync.assert_called_once_with()
