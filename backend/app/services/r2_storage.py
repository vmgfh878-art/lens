"""비공개 Cloudflare R2에 버전 스냅샷을 안전하게 발행한다."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import R2Config, get_r2_config
from app.services.serving_snapshot import (
    SNAPSHOT_MANIFEST_FILENAME,
    SnapshotValidationError,
    serialize_snapshot_manifest,
    validate_snapshot_directory,
)

LATEST_POINTER_FILENAME = "latest.json"
LATEST_POINTER_SCHEMA_VERSION = 1

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class R2ConfigurationError(RuntimeError):
    """R2 연결에 필요한 설정이 빠졌을 때 발생한다."""


class R2PublishError(RuntimeError):
    """R2 발행이 완결되지 않았을 때 발생한다."""


def require_r2_config(config: R2Config | None = None) -> R2Config:
    """필수 설정을 검사하되 자격 증명 값은 오류 메시지에 노출하지 않는다."""
    resolved = config or get_r2_config()
    missing = resolved.missing_required_fields
    if missing:
        raise R2ConfigurationError(f"R2 필수 설정이 없습니다: {', '.join(missing)}")
    return resolved


def create_r2_client(config: R2Config | None = None) -> Any:
    """공식 S3 호환 엔드포인트용 boto3 클라이언트를 만든다."""
    resolved = require_r2_config(config)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise R2ConfigurationError(
            "boto3가 설치되지 않았습니다. backend/requirements.txt를 설치하세요."
        ) from exc

    client_options: dict[str, Any] = {
        "service_name": "s3",
        "endpoint_url": resolved.endpoint_url,
        "aws_access_key_id": resolved.access_key_id,
        "aws_secret_access_key": resolved.secret_access_key,
        "region_name": resolved.region,
        "config": Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 4, "mode": "standard"},
        ),
    }
    if resolved.session_token:
        client_options["aws_session_token"] = resolved.session_token
    return boto3.client(**client_options)


def _utc_text(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise R2PublishError("published_at은 시간대 정보가 있는 datetime이어야 합니다.")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _object_key(prefix: str, *parts: str) -> str:
    return "/".join((prefix.strip("/"), *(part.strip("/") for part in parts)))


def snapshot_object_keys(config: R2Config, snapshot_id: str) -> dict[str, Any]:
    """한 버전의 불변 객체 경로와 가변 latest 포인터 경로를 계산한다."""
    if not _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise R2PublishError(f"snapshot_id가 올바르지 않습니다: {snapshot_id!r}")
    release_prefix = _object_key(config.object_prefix, "releases", snapshot_id)
    return {
        "release_prefix": release_prefix,
        "manifest_key": _object_key(release_prefix, SNAPSHOT_MANIFEST_FILENAME),
        "latest_key": _object_key(config.object_prefix, LATEST_POINTER_FILENAME),
    }


def build_latest_pointer(
    manifest: Mapping[str, Any],
    manifest_key: str,
    manifest_sha256: str,
    *,
    published_at: datetime | None = None,
) -> dict[str, Any]:
    pointer = {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "snapshot_id": manifest["snapshot_id"],
        "published_at": _utc_text(published_at),
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "content_sha256": manifest["content_sha256"],
    }
    validate_latest_pointer(pointer)
    return pointer


def validate_latest_pointer(pointer: Mapping[str, Any]) -> None:
    """다운로더가 신뢰할 수 있도록 latest 포인터의 최소 계약을 검사한다."""
    if not isinstance(pointer, Mapping):
        raise R2PublishError("latest 포인터는 JSON 객체여야 합니다.")
    if pointer.get("schema_version") != LATEST_POINTER_SCHEMA_VERSION:
        raise R2PublishError("latest 포인터 schema_version이 올바르지 않습니다.")
    snapshot_id = pointer.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise R2PublishError("latest 포인터 snapshot_id가 올바르지 않습니다.")
    manifest_key = pointer.get("manifest_key")
    if (
        not isinstance(manifest_key, str)
        or manifest_key.startswith("/")
        or any(part in {"", ".", ".."} for part in manifest_key.split("/"))
        or not manifest_key.endswith(f"/{SNAPSHOT_MANIFEST_FILENAME}")
    ):
        raise R2PublishError("latest 포인터 manifest_key가 올바르지 않습니다.")
    for field in ("manifest_sha256", "content_sha256"):
        value = pointer.get(field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise R2PublishError(f"latest 포인터 {field}가 올바르지 않습니다.")
    published_at = pointer.get("published_at")
    if not isinstance(published_at, str):
        raise R2PublishError("latest 포인터 published_at이 올바르지 않습니다.")
    try:
        parsed = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise R2PublishError("latest 포인터 published_at이 ISO 8601 형식이 아닙니다.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise R2PublishError("latest 포인터 published_at에는 시간대 정보가 필요합니다.")


def serialize_latest_pointer(pointer: Mapping[str, Any]) -> bytes:
    validate_latest_pointer(pointer)
    return (json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def build_publish_plan(
    config: R2Config,
    manifest: Mapping[str, Any],
    data_dir: str | Path,
) -> dict[str, Any]:
    """네트워크 호출 없이 로컬 무결성과 발행 대상 키를 계산한다."""
    summary = validate_snapshot_directory(manifest, data_dir)
    keys = snapshot_object_keys(config, str(manifest["snapshot_id"]))
    release_objects = [
        _object_key(keys["release_prefix"], item["name"]) for item in manifest["files"]
    ]
    release_objects.append(keys["manifest_key"])
    return {
        **summary,
        "bucket": config.bucket,
        "release_prefix": keys["release_prefix"],
        "latest_key": keys["latest_key"],
        "release_objects": release_objects,
    }


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    error = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = str(error.get("Code", "")) if isinstance(error, Mapping) else ""
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _head_or_none(client: Any, bucket: str, key: str) -> Mapping[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 — SDK별 not-found 예외형이 다르다.
        if _not_found(exc):
            return None
        raise


def _metadata_sha256(head: Mapping[str, Any]) -> str | None:
    metadata = head.get("Metadata")
    if not isinstance(metadata, Mapping):
        return None
    return next(
        (str(value) for key, value in metadata.items() if str(key).lower() == "sha256"),
        None,
    )


def _remote_matches(head: Mapping[str, Any], *, size_bytes: int, sha256: str) -> bool:
    return int(head.get("ContentLength", -1)) == size_bytes and _metadata_sha256(head) == sha256


def _verify_remote_object(
    client: Any,
    bucket: str,
    key: str,
    *,
    size_bytes: int,
    sha256: str,
) -> None:
    head = client.head_object(Bucket=bucket, Key=key)
    if not _remote_matches(head, size_bytes=size_bytes, sha256=sha256):
        raise R2PublishError(f"업로드 검증에 실패했습니다: {key}")


def _upload_immutable_file(
    client: Any,
    *,
    bucket: str,
    key: str,
    source: Path,
    sha256: str,
    snapshot_id: str,
) -> str:
    size_bytes = source.stat().st_size
    existing = _head_or_none(client, bucket, key)
    if existing is not None:
        if _remote_matches(existing, size_bytes=size_bytes, sha256=sha256):
            return "skipped"
        raise R2PublishError(f"같은 snapshot_id에 다른 객체가 이미 있습니다: {key}")

    content_type = (
        "application/json" if source.suffix.lower() == ".json" else "application/vnd.apache.parquet"
    )
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={
            "ContentType": content_type,
            "CacheControl": "private, max-age=31536000, immutable",
            "Metadata": {"sha256": sha256, "snapshot-id": snapshot_id},
        },
    )
    _verify_remote_object(
        client,
        bucket,
        key,
        size_bytes=size_bytes,
        sha256=sha256,
    )
    return "uploaded"


def _upload_immutable_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    sha256: str,
    snapshot_id: str,
) -> str:
    existing = _head_or_none(client, bucket, key)
    if existing is not None:
        if _remote_matches(existing, size_bytes=len(payload), sha256=sha256):
            return "skipped"
        raise R2PublishError(f"같은 snapshot_id에 다른 객체가 이미 있습니다: {key}")

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
        CacheControl="private, max-age=31536000, immutable",
        Metadata={"sha256": sha256, "snapshot-id": snapshot_id},
    )
    _verify_remote_object(
        client,
        bucket,
        key,
        size_bytes=len(payload),
        sha256=sha256,
    )
    return "uploaded"


def publish_snapshot(
    data_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    config: R2Config | None = None,
    client: Any | None = None,
    published_at: datetime | None = None,
) -> dict[str, Any]:
    """버전 객체를 모두 검증한 뒤 마지막에 latest 포인터만 교체한다."""
    resolved = require_r2_config(config)
    plan = build_publish_plan(resolved, manifest, data_dir)
    r2 = client or create_r2_client(resolved)
    bucket = str(resolved.bucket)
    snapshot_id = str(manifest["snapshot_id"])
    root = Path(data_dir).resolve()
    keys = snapshot_object_keys(resolved, snapshot_id)
    uploaded = 0
    skipped = 0

    try:
        for item in manifest["files"]:
            key = _object_key(keys["release_prefix"], item["name"])
            state = _upload_immutable_file(
                r2,
                bucket=bucket,
                key=key,
                source=root / item["name"],
                sha256=item["sha256"],
                snapshot_id=snapshot_id,
            )
            uploaded += state == "uploaded"
            skipped += state == "skipped"

        manifest_payload = serialize_snapshot_manifest(manifest)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        state = _upload_immutable_bytes(
            r2,
            bucket=bucket,
            key=keys["manifest_key"],
            payload=manifest_payload,
            sha256=manifest_sha256,
            snapshot_id=snapshot_id,
        )
        uploaded += state == "uploaded"
        skipped += state == "skipped"

        pointer = build_latest_pointer(
            manifest,
            keys["manifest_key"],
            manifest_sha256,
            published_at=published_at,
        )
        pointer_payload = serialize_latest_pointer(pointer)
        pointer_sha256 = hashlib.sha256(pointer_payload).hexdigest()
        r2.put_object(
            Bucket=bucket,
            Key=keys["latest_key"],
            Body=pointer_payload,
            ContentType="application/json",
            CacheControl="private, no-cache, max-age=0",
            Metadata={"sha256": pointer_sha256, "snapshot-id": snapshot_id},
        )
        _verify_remote_object(
            r2,
            bucket,
            keys["latest_key"],
            size_bytes=len(pointer_payload),
            sha256=pointer_sha256,
        )
        downloaded_pointer = r2.get_object(Bucket=bucket, Key=keys["latest_key"])["Body"].read()
        if downloaded_pointer != pointer_payload:
            raise R2PublishError("latest 포인터 재조회 검증에 실패했습니다.")
    except (R2PublishError, SnapshotValidationError):
        raise
    except Exception as exc:  # noqa: BLE001 — SDK 오류를 발행 단계 오류로 감싼다.
        raise R2PublishError(
            f"R2 발행이 완료되지 않았습니다. 대상 snapshot_id={snapshot_id}"
        ) from exc

    return {
        **plan,
        "uploaded_object_count": uploaded,
        "skipped_object_count": skipped,
        "published_at": pointer["published_at"],
        "manifest_sha256": manifest_sha256,
        "latest_pointer_sha256": pointer_sha256,
    }


__all__ = [
    "LATEST_POINTER_FILENAME",
    "LATEST_POINTER_SCHEMA_VERSION",
    "R2ConfigurationError",
    "R2PublishError",
    "build_latest_pointer",
    "build_publish_plan",
    "create_r2_client",
    "publish_snapshot",
    "require_r2_config",
    "serialize_latest_pointer",
    "snapshot_object_keys",
    "validate_latest_pointer",
]
