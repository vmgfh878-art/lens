"""원격 최신 스냅샷을 로컬 캐시에 완성한 뒤 원자적으로 활성화한다."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import GitHubReleaseConfig
from app.services.r2_storage import (
    R2ConfigurationError,
    create_r2_client,
    snapshot_object_keys,
    validate_latest_pointer,
)
from app.services.serving_paths import (
    activate_serving_data_dir,
    get_serving_bootstrap_dir,
    get_serving_cache_dir,
    get_serving_data_dir,
)
from app.services.serving_snapshot import (
    SNAPSHOT_MANIFEST_FILENAME,
    SnapshotValidationError,
    load_snapshot_manifest,
    parse_snapshot_manifest,
    validate_snapshot_directory,
)
from app.services.serving_storage import (
    RemoteServingConfig,
    create_serving_storage_client,
    get_remote_serving_config,
    remote_serving_source,
)

_LATEST_MAX_BYTES = 64 * 1024
_MANIFEST_MAX_BYTES = 1024 * 1024
_SYNC_LOCK = RLock()
_STATE_LOCK = RLock()
_SYNC_STATE: dict[str, Any] = {
    "status": "idle",
    "source": "local",
    "snapshot_id": None,
    "active_dir": None,
    "fallback_ready": False,
    "fallback_snapshot_id": None,
    "last_success_at": None,
    "last_error": None,
}


class R2SyncError(RuntimeError):
    """원격 스냅샷을 완전하게 내려받아 활성화하지 못했을 때 발생한다."""


def _create_remote_client(config: RemoteServingConfig) -> Any:
    if isinstance(config, GitHubReleaseConfig):
        return create_serving_storage_client(config)
    return create_r2_client(config)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _set_sync_state(**updates: Any) -> None:
    with _STATE_LOCK:
        _SYNC_STATE.update(updates)


def get_serving_sync_state() -> dict[str, Any]:
    with _STATE_LOCK:
        state = dict(_SYNC_STATE)
    if state["active_dir"] is None:
        state["active_dir"] = str(get_serving_data_dir())
    return state


def reset_serving_sync_state() -> None:
    """테스트와 프로세스 재초기화에서 동기화 상태만 기본값으로 되돌린다."""
    with _STATE_LOCK:
        _SYNC_STATE.clear()
        _SYNC_STATE.update(
            {
                "status": "idle",
                "source": "local",
                "snapshot_id": None,
                "active_dir": None,
                "fallback_ready": False,
                "fallback_snapshot_id": None,
                "last_success_at": None,
                "last_error": None,
            }
        )


def _read_object_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    max_bytes: int,
) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    content_length = response.get("ContentLength")
    if isinstance(content_length, int) and content_length > max_bytes:
        response["Body"].close()
        raise R2SyncError(f"원격 메타데이터 객체가 너무 큽니다: {key}")
    stream = response["Body"]
    try:
        payload = stream.read(max_bytes + 1)
    finally:
        stream.close()
    if len(payload) > max_bytes:
        raise R2SyncError(f"원격 메타데이터 객체가 너무 큽니다: {key}")
    return payload


def _load_latest_pointer(client: Any, config: RemoteServingConfig) -> dict[str, Any]:
    keys = snapshot_object_keys(config, "pointer-probe")
    payload = _read_object_bytes(
        client,
        bucket=str(config.bucket),
        key=keys["latest_key"],
        max_bytes=_LATEST_MAX_BYTES,
    )
    try:
        pointer = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R2SyncError("원격 최신 포인터 JSON이 손상됐습니다.") from exc
    if not isinstance(pointer, dict):
        raise R2SyncError("원격 최신 포인터는 JSON 객체여야 합니다.")
    validate_latest_pointer(pointer)
    expected = snapshot_object_keys(config, pointer["snapshot_id"])
    if pointer["manifest_key"] != expected["manifest_key"]:
        raise R2SyncError("원격 최신 포인터가 예상한 버전 경로를 가리키지 않습니다.")
    return pointer


def _load_remote_manifest(
    client: Any,
    config: RemoteServingConfig,
    pointer: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    payload = _read_object_bytes(
        client,
        bucket=str(config.bucket),
        key=str(pointer["manifest_key"]),
        max_bytes=_MANIFEST_MAX_BYTES,
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != pointer["manifest_sha256"]:
        raise R2SyncError("원격 스냅샷 매니페스트 해시가 최신 포인터와 다릅니다.")
    manifest = parse_snapshot_manifest(payload)
    if manifest["snapshot_id"] != pointer["snapshot_id"]:
        raise R2SyncError("원격 매니페스트와 최신 포인터의 snapshot_id가 다릅니다.")
    if manifest["content_sha256"] != pointer["content_sha256"]:
        raise R2SyncError("원격 매니페스트와 최신 포인터의 콘텐츠 지문이 다릅니다.")
    return payload, manifest


def inspect_latest_serving_snapshot(
    *,
    config: RemoteServingConfig | None = None,
    client: Any | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    """원격 latest와 매니페스트를 읽고, 선택적으로 파일 해시까지 검사한다.

    로컬 파일, 활성 경로, 동기화 상태, 원격 첨부파일을 변경하지 않는다.
    """
    resolved = config or get_remote_serving_config()
    if not resolved.configured:
        raise R2ConfigurationError("원격 읽기 전용 검사를 위한 설정이 없습니다.")
    r2 = client or _create_remote_client(resolved)
    try:
        pointer = _load_latest_pointer(r2, resolved)
        _, manifest = _load_remote_manifest(r2, resolved, pointer)
        if verify_files:
            keys = snapshot_object_keys(resolved, str(manifest["snapshot_id"]))
            for item in manifest["files"]:
                key = f"{keys['release_prefix']}/{item['name']}"
                response = r2.get_object(Bucket=str(resolved.bucket), Key=key)
                stream = response["Body"]
                size = 0
                digest = hashlib.sha256()
                try:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > item["size_bytes"]:
                            raise R2SyncError(
                                f"원격 파일 크기가 매니페스트를 초과합니다: {item['name']}"
                            )
                        digest.update(chunk)
                finally:
                    stream.close()
                if size != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
                    raise R2SyncError(f"원격 파일 무결성이 매니페스트와 다릅니다: {item['name']}")
        return {"pointer": pointer, "manifest": manifest, "files_verified": verify_files}
    finally:
        if client is None and callable(getattr(r2, "close", None)):
            r2.close()


def _safe_remove_cache_dir(path: Path, cache_root: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != cache_root.resolve():
        raise R2SyncError(f"캐시 루트 밖 디렉터리는 지울 수 없습니다: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _download_release(
    client: Any,
    config: RemoteServingConfig,
    manifest_payload: bytes,
    manifest: Mapping[str, Any],
    cache_root: Path,
) -> tuple[Path, bool]:
    snapshot_id = str(manifest["snapshot_id"])
    keys = snapshot_object_keys(config, snapshot_id)
    final_dir = cache_root / snapshot_id
    cache_root.mkdir(parents=True, exist_ok=True)

    if final_dir.is_dir():
        try:
            cached_manifest = load_snapshot_manifest(final_dir / SNAPSHOT_MANIFEST_FILENAME)
            if cached_manifest["content_sha256"] == manifest["content_sha256"]:
                validate_snapshot_directory(cached_manifest, final_dir)
                return final_dir, True
        except SnapshotValidationError:
            pass
        _safe_remove_cache_dir(final_dir, cache_root)

    staging_dir = cache_root / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
    staging_dir.mkdir(parents=False, exist_ok=False)
    try:
        for item in manifest["files"]:
            key = f"{keys['release_prefix']}/{item['name']}"
            client.download_file(str(config.bucket), key, str(staging_dir / item["name"]))
        (staging_dir / SNAPSHOT_MANIFEST_FILENAME).write_bytes(manifest_payload)
        validate_snapshot_directory(manifest, staging_dir)
        os.replace(staging_dir, final_dir)
    finally:
        if staging_dir.exists():
            _safe_remove_cache_dir(staging_dir, cache_root)
    return final_dir, False


def _clear_serving_caches() -> None:
    # 순환 import를 피하고 실제 활성화 시점에만 무거운 서비스 모듈을 불러온다.
    from app.repositories.ai_repo import _load_mock
    from app.services import local_market_svc, parquet_store
    from app.services.product_prediction_history_svc import clear_product_history_cache
    from app.services.strategy_backtest_svc import clear_strategy_cache

    parquet_store.clear_all()
    local_market_svc.clear_caches()
    clear_product_history_cache()
    clear_strategy_cache()
    _load_mock.cache_clear()


def _activate_snapshot(path: Path) -> None:
    # 활성 경로 교체 전후로 비워 전환 순간에 생성된 이전 세대 캐시도 제거한다.
    _clear_serving_caches()
    activate_serving_data_dir(path)
    _clear_serving_caches()


def activate_bootstrap_serving_snapshot() -> dict[str, Any]:
    """Git에 고정된 폴백 한 벌을 검증한 뒤 활성화한다."""
    bootstrap_dir = get_serving_bootstrap_dir()
    try:
        manifest = load_snapshot_manifest(bootstrap_dir / SNAPSHOT_MANIFEST_FILENAME)
        summary = validate_snapshot_directory(manifest, bootstrap_dir)
        changed = get_serving_data_dir().resolve() != bootstrap_dir.resolve()
        if changed:
            _activate_snapshot(bootstrap_dir)
        else:
            activate_serving_data_dir(bootstrap_dir)
        result = {
            "status": "fallback_ready",
            "source": "bootstrap",
            "snapshot_id": summary["snapshot_id"],
            "content_sha256": summary["content_sha256"],
            "active_dir": str(bootstrap_dir),
            "changed": changed,
            "fallback_ready": True,
            "fallback_snapshot_id": summary["snapshot_id"],
            "last_error": None,
        }
        _set_sync_state(**result)
        return result
    except Exception as exc:  # noqa: BLE001 — 부팅 상태에 폴백 검증 실패를 남긴다.
        error_text = f"{type(exc).__name__}: {str(exc)[:500]}"
        _set_sync_state(
            status="fallback_error",
            source="bootstrap",
            snapshot_id=None,
            active_dir=str(bootstrap_dir),
            fallback_ready=False,
            fallback_snapshot_id=None,
            last_error=error_text,
        )
        raise


def _prune_old_versions(cache_root: Path, active_dir: Path, *, keep: int = 2) -> None:
    versions = sorted(
        (
            path
            for path in cache_root.iterdir()
            if path.is_dir() and not path.name.startswith(".") and path != active_dir
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for old_dir in versions[max(keep - 1, 0) :]:
        _safe_remove_cache_dir(old_dir, cache_root)


def sync_latest_serving_snapshot(
    *,
    config: RemoteServingConfig | None = None,
    client: Any | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """latest가 가리키는 완전한 버전을 내려받고 검증 후 활성화한다."""
    resolved = config or get_remote_serving_config()
    if not resolved.configured:
        raise R2ConfigurationError(
            f"원격 서빙 필수 설정이 없습니다: {', '.join(resolved.missing_required_fields)}"
        )
    cache_root = Path(cache_dir).resolve() if cache_dir else get_serving_cache_dir()

    with _SYNC_LOCK:
        _set_sync_state(status="syncing", last_error=None)
        r2 = None
        try:
            r2 = client or _create_remote_client(resolved)
            pointer = _load_latest_pointer(r2, resolved)
            manifest_payload, manifest = _load_remote_manifest(r2, resolved, pointer)
            target_dir, reused = _download_release(
                r2,
                resolved,
                manifest_payload,
                manifest,
                cache_root,
            )
            changed = get_serving_data_dir().resolve() != target_dir.resolve()
            if changed:
                _activate_snapshot(target_dir)
            _prune_old_versions(cache_root, target_dir)
            completed_at = _utc_now_text()
            result = {
                "status": "ready",
                "source": remote_serving_source(resolved),
                "snapshot_id": manifest["snapshot_id"],
                "content_sha256": manifest["content_sha256"],
                "active_dir": str(target_dir),
                "changed": changed,
                "reused_cache": reused,
                "last_success_at": completed_at,
                "last_error": None,
            }
            _set_sync_state(**result)
            return result
        except Exception as exc:  # noqa: BLE001 — 활성 버전을 보존하고 원인을 상태에 남긴다.
            error_text = f"{type(exc).__name__}: {str(exc)[:500]}"
            _set_sync_state(
                status="error",
                active_dir=str(get_serving_data_dir()),
                last_error=error_text,
            )
            if isinstance(exc, R2ConfigurationError | R2SyncError):
                raise
            raise R2SyncError("원격 최신 스냅샷 동기화에 실패했습니다.") from exc
        finally:
            if client is None and r2 is not None and callable(getattr(r2, "close", None)):
                r2.close()


__all__ = [
    "R2SyncError",
    "activate_bootstrap_serving_snapshot",
    "get_serving_sync_state",
    "inspect_latest_serving_snapshot",
    "reset_serving_sync_state",
    "sync_latest_serving_snapshot",
]
