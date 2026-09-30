"""검증된 로컬 서빙 스냅샷을 비공개 Cloudflare R2에 발행한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.config import get_r2_config  # noqa: E402
from app.services.r2_storage import (  # noqa: E402
    R2ConfigurationError,
    R2PublishError,
    build_publish_plan,
    publish_snapshot,
)
from app.services.serving_paths import get_serving_data_dir  # noqa: E402
from app.services.serving_snapshot import (  # noqa: E402
    SnapshotValidationError,
    build_snapshot_manifest,
    load_snapshot_manifest,
)

DEFAULT_BACKEND_URL = "https://lens-backend-7stj.onrender.com"
SYNC_PATH = "/api/v1/admin/sync-serving"


def backend_sync_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith(SYNC_PATH):
        return cleaned
    return f"{cleaned}{SYNC_PATH}"


def notify_backend_sync(
    *,
    base_url: str,
    admin_token: str,
    expected_snapshot_id: str,
    expected_source: str = "r2",
    attempts: int = 4,
    timeout_seconds: float = 90.0,
    http_client: Any | None = None,
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    """Render에 최신 원격 버전 동기화를 요청하고 활성 버전까지 확인한다."""
    if not admin_token.strip():
        raise R2PublishError("LENS_ADMIN_RELOAD_TOKEN이 없어 Render 동기화를 요청할 수 없습니다.")
    url = backend_sync_url(base_url)
    owned_client = http_client is None
    if owned_client:
        import httpx

        http_client = httpx.Client(timeout=timeout_seconds, follow_redirects=True)
    try:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = http_client.post(
                    url,
                    headers={"X-Lens-Admin-Token": admin_token},
                )
                if 400 <= response.status_code < 500:
                    raise R2PublishError(
                        f"Render 동기화 요청이 거부됐습니다: HTTP {response.status_code}"
                    )
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                if not isinstance(data, dict):
                    raise R2PublishError("Render 동기화 응답 형식이 올바르지 않습니다.")
                if data.get("snapshot_id") != expected_snapshot_id:
                    raise R2PublishError(
                        "Render가 활성화한 snapshot_id가 방금 발행한 버전과 다릅니다."
                    )
                if data.get("source") != expected_source or data.get("status") != "ready":
                    raise R2PublishError("Render가 예상한 원격 스냅샷 준비 완료 상태가 아닙니다.")
                return {
                    "status": "ready",
                    "snapshot_id": data["snapshot_id"],
                    "changed": bool(data.get("changed")),
                    "attempt": attempt,
                    "url": url,
                }
            except R2PublishError:
                raise
            except Exception as exc:  # noqa: BLE001 — 콜드 스타트와 일시 5xx를 재시도한다.
                last_error = exc
                if attempt == attempts:
                    break
                sleep_fn(min(5 * attempt, 20))
        raise R2PublishError(
            f"Render 동기화 요청이 {attempts}회 실패했습니다: {type(last_error).__name__}"
        ) from last_error
    finally:
        if owned_client:
            http_client.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="로컬 서빙 스냅샷을 비공개 R2에 발행")
    parser.add_argument("--data-dir", default=str(get_serving_data_dir()))
    parser.add_argument("--manifest", help="기존 매니페스트 사용. 없으면 현재 파일로 생성")
    parser.add_argument("--snapshot-id", help="새 매니페스트를 만들 때 사용할 버전 ID")
    parser.add_argument("--result-path", help="발행 실패를 포함한 결과를 저장할 JSON 경로")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 무결성과 객체 키만 검사하고 네트워크 호출은 하지 않음",
    )
    parser.add_argument(
        "--skip-backend-sync",
        action="store_true",
        help="R2 발행 후 Render 동기화 요청을 생략함. 최초 수동 부트스트랩에만 사용",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    result: dict[str, Any] = {"status": "starting"}
    try:
        manifest = (
            load_snapshot_manifest(args.manifest)
            if args.manifest
            else build_snapshot_manifest(data_dir, snapshot_id=args.snapshot_id)
        )
        config = get_r2_config()
        if args.dry_run:
            result = build_publish_plan(config, manifest, data_dir)
            result["dry_run"] = True
        else:
            result = publish_snapshot(data_dir, manifest, config=config)
            result["status"] = "published"
            if args.skip_backend_sync:
                result["backend_sync"] = {"status": "skipped"}
            else:
                result["backend_sync"] = notify_backend_sync(
                    base_url=os.environ.get("LENS_DEPLOYED_BACKEND_URL", DEFAULT_BACKEND_URL),
                    admin_token=os.environ.get("LENS_ADMIN_RELOAD_TOKEN", ""),
                    expected_snapshot_id=str(manifest["snapshot_id"]),
                )
            result["dry_run"] = False
            result["status"] = "ready" if not args.skip_backend_sync else "published"
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (R2ConfigurationError, R2PublishError, SnapshotValidationError) as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        print(f"R2 발행 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.result_path:
            output = Path(args.result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
