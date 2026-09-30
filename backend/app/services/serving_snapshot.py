"""버전 단위 서빙 스냅샷의 매니페스트 생성과 무결성 검증."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_FILENAME = "snapshot_manifest.json"
REQUIRED_SERVING_FILES = (
    "ai_runs_mock.json",
    "market_indicators_1d.parquet",
    "market_prices_1d.parquet",
    "market_prices_1w.parquet",
    "market_stock_info.parquet",
    "predictions_band_1d.parquet",
    "predictions_band_1w.parquet",
    "predictions_line_1d.parquet",
    "product_prediction_history_1D.manifest.json",
    "product_prediction_history_1D.parquet",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SnapshotValidationError(ValueError):
    """스냅샷 계약이나 파일 무결성이 맞지 않을 때 발생한다."""


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """큰 파일을 메모리에 한꺼번에 올리지 않고 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        line = f"{item['name']}\0{item['size_bytes']}\0{item['sha256']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _created_at_text(created_at: datetime | None) -> str:
    value = created_at or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise SnapshotValidationError("created_at은 시간대 정보가 있는 datetime이어야 합니다.")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_created_at(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotValidationError("created_at은 비어 있지 않은 문자열이어야 합니다.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SnapshotValidationError("created_at이 ISO 8601 형식이 아닙니다.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SnapshotValidationError("created_at에는 시간대 정보가 필요합니다.")
    return value


def _validate_snapshot_id(value: Any) -> str:
    if not isinstance(value, str) or not _SNAPSHOT_ID_PATTERN.fullmatch(value):
        raise SnapshotValidationError(
            "snapshot_id는 영문자·숫자로 시작하고 영문자·숫자·점·밑줄·하이픈만 써야 합니다."
        )
    return value


def validate_manifest_document(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """매니페스트 자체의 스키마와 콘텐츠 지문을 검증한다."""
    if not isinstance(manifest, Mapping):
        raise SnapshotValidationError("스냅샷 매니페스트는 JSON 객체여야 합니다.")

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotValidationError(f"지원하지 않는 schema_version입니다: {schema_version!r}")
    _validate_snapshot_id(manifest.get("snapshot_id"))
    _validate_created_at(manifest.get("created_at"))

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise SnapshotValidationError("files는 배열이어야 합니다.")

    files: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_files):
        if not isinstance(raw_item, Mapping):
            raise SnapshotValidationError(f"files[{index}]는 JSON 객체여야 합니다.")
        name = raw_item.get("name")
        size_bytes = raw_item.get("size_bytes")
        sha256 = raw_item.get("sha256")
        if not isinstance(name, str):
            raise SnapshotValidationError(f"files[{index}].name은 문자열이어야 합니다.")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise SnapshotValidationError(f"files[{index}].size_bytes는 양의 정수여야 합니다.")
        if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
            raise SnapshotValidationError(f"files[{index}].sha256 형식이 올바르지 않습니다.")
        files.append({"name": name, "size_bytes": size_bytes, "sha256": sha256})

    names = tuple(item["name"] for item in files)
    if names != REQUIRED_SERVING_FILES:
        missing = sorted(set(REQUIRED_SERVING_FILES) - set(names))
        unexpected = sorted(set(names) - set(REQUIRED_SERVING_FILES))
        raise SnapshotValidationError(
            "서빙 파일 목록이나 순서가 계약과 다릅니다. "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_total = sum(item["size_bytes"] for item in files)
    total_size_bytes = manifest.get("total_size_bytes")
    if type(total_size_bytes) is not int or total_size_bytes != expected_total:
        raise SnapshotValidationError(
            f"total_size_bytes가 파일 합계와 다릅니다: {total_size_bytes!r} != {expected_total}"
        )

    expected_content_sha256 = _content_sha256(files)
    content_sha256 = manifest.get("content_sha256")
    if content_sha256 != expected_content_sha256:
        raise SnapshotValidationError("content_sha256가 파일 목록 지문과 다릅니다.")
    return files


def build_snapshot_manifest(
    data_dir: str | Path,
    *,
    snapshot_id: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """필수 운영 파일 10개의 크기와 해시를 계산해 매니페스트를 만든다."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise SnapshotValidationError(f"서빙 데이터 디렉터리가 없습니다: {root}")

    missing = [name for name in REQUIRED_SERVING_FILES if not (root / name).is_file()]
    if missing:
        raise SnapshotValidationError(f"필수 서빙 파일이 없습니다: {missing}")

    files = []
    for name in REQUIRED_SERVING_FILES:
        path = root / name
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise SnapshotValidationError(f"빈 서빙 파일은 발행할 수 없습니다: {path}")
        files.append(
            {
                "name": name,
                "size_bytes": size_bytes,
                "sha256": file_sha256(path),
            }
        )

    content_sha256 = _content_sha256(files)
    created_at_text = _created_at_text(created_at)
    generated_id = f"{created_at_text.replace('-', '').replace(':', '')}-{content_sha256[:12]}"
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id or generated_id,
        "created_at": created_at_text,
        "content_sha256": content_sha256,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    validate_manifest_document(manifest)
    return manifest


def parse_snapshot_manifest(payload: bytes | str) -> dict[str, Any]:
    """JSON 바이트열이나 문자열을 해석하고 문서 계약을 검증한다."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        raw = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"스냅샷 매니페스트 JSON이 손상됐습니다: {exc}") from exc
    if not isinstance(raw, dict):
        raise SnapshotValidationError("스냅샷 매니페스트는 JSON 객체여야 합니다.")
    validate_manifest_document(raw)
    return raw


def load_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    """JSON 매니페스트 파일을 읽고 문서 계약을 검증한다."""
    manifest_path = Path(path)
    try:
        payload = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise SnapshotValidationError(f"스냅샷 매니페스트가 없습니다: {manifest_path}") from exc
    return parse_snapshot_manifest(payload)


def serialize_snapshot_manifest(manifest: Mapping[str, Any]) -> bytes:
    """검증된 매니페스트를 업로드와 로컬 저장이 공유하는 바이트열로 직렬화한다."""
    validate_manifest_document(manifest)
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def validate_snapshot_directory(
    manifest: Mapping[str, Any],
    data_dir: str | Path,
) -> dict[str, Any]:
    """디렉터리의 모든 파일이 매니페스트의 크기와 해시와 일치하는지 확인한다."""
    files = validate_manifest_document(manifest)
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise SnapshotValidationError(f"스냅샷 디렉터리가 없습니다: {root}")

    for item in files:
        path = root / item["name"]
        if not path.is_file():
            raise SnapshotValidationError(f"스냅샷 파일이 없습니다: {path}")
        actual_size = path.stat().st_size
        if actual_size != item["size_bytes"]:
            raise SnapshotValidationError(
                f"스냅샷 파일 크기가 다릅니다: {item['name']} "
                f"({actual_size} != {item['size_bytes']})"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256 != item["sha256"]:
            raise SnapshotValidationError(f"스냅샷 파일 해시가 다릅니다: {item['name']}")

    return {
        "snapshot_id": manifest["snapshot_id"],
        "file_count": len(files),
        "total_size_bytes": manifest["total_size_bytes"],
        "content_sha256": manifest["content_sha256"],
    }


def write_snapshot_manifest(
    data_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    snapshot_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """완성된 매니페스트만 보이도록 같은 디렉터리에서 원자적으로 교체한다."""
    root = Path(data_dir).resolve()
    target = Path(output_path).resolve() if output_path else root / SNAPSHOT_MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_snapshot_manifest(root, snapshot_id=snapshot_id, created_at=created_at)
    payload = serialize_snapshot_manifest(manifest)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)

    return target, manifest


__all__ = [
    "REQUIRED_SERVING_FILES",
    "SNAPSHOT_MANIFEST_FILENAME",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotValidationError",
    "build_snapshot_manifest",
    "file_sha256",
    "load_snapshot_manifest",
    "parse_snapshot_manifest",
    "serialize_snapshot_manifest",
    "validate_manifest_document",
    "validate_snapshot_directory",
    "write_snapshot_manifest",
]
