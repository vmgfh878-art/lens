"""매일 데이터 커밋 없이 검증 스냅샷을 비공개 릴리스에 발행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "backend"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.config import get_github_release_config  # noqa: E402
from app.services.github_release_storage import (  # noqa: E402
    GitHubReleaseClient,
    GitHubReleaseError,
    resolve_github_config,
)
from app.services.r2_storage import (  # noqa: E402
    R2ConfigurationError,
    R2PublishError,
    build_publish_plan,
    publish_snapshot,
)
from app.services.serving_paths import configured_serving_data_dir  # noqa: E402
from app.services.serving_snapshot import (  # noqa: E402
    SnapshotValidationError,
    build_snapshot_manifest,
    load_snapshot_manifest,
)

from backend.scripts.publish_serving_to_r2 import (  # noqa: E402
    DEFAULT_BACKEND_URL,
    notify_backend_sync,
)


def wait_for_backend_snapshot(
    *,
    base_url: str,
    expected_snapshot_id: str,
    attempts: int = 33,
    sleep_seconds: float = 15.0,
    http_client: Any | None = None,
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    """관리 토큰 없이 Render의 주기적 동기화 완료를 읽기 전용으로 확인한다."""
    import httpx

    url = f"{base_url.rstrip('/')}/api/v1/health/ready"
    owned_client = http_client is None
    if owned_client:
        http_client = httpx.Client(timeout=45.0, follow_redirects=True)
    last_state = "응답 없음"
    try:
        for attempt in range(1, attempts + 1):
            try:
                response = http_client.get(url)
                if 400 <= response.status_code < 500:
                    raise R2PublishError(
                        f"Render 상태 조회가 거부됐습니다: HTTP {response.status_code}"
                    )
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                if isinstance(data, dict):
                    source = data.get("source")
                    snapshot_id = data.get("snapshot_id")
                    status = data.get("status")
                    checks = data.get("checks") or {}
                    if (
                        source == "github_release"
                        and snapshot_id == expected_snapshot_id
                        and status == "ok"
                        and checks.get("required_files")
                    ):
                        return {
                            "status": "ready",
                            "source": source,
                            "snapshot_id": snapshot_id,
                            "attempt": attempt,
                            "url": url,
                            "mode": "periodic_poll",
                        }
                    last_state = f"source={source} status={status} snapshot_id={snapshot_id}"
                else:
                    last_state = "응답 형식 오류"
            except R2PublishError:
                raise
            except Exception as exc:  # noqa: BLE001 — 콜드 스타트·일시 장애 뒤 재조회한다.
                last_state = type(exc).__name__
            if attempt < attempts:
                sleep_fn(sleep_seconds)
        raise R2PublishError(f"Render가 새 스냅샷을 활성화하지 않았습니다: {last_state}")
    finally:
        if owned_client:
            http_client.close()


@contextmanager
def _publisher_lock(repository: str, tag: str):
    """동일 PC에서 두 발행기가 동시에 최신 포인터를 바꾸지 못하게 한다."""
    digest = hashlib.sha256(f"{repository}/{tag}".encode()).hexdigest()
    lock_dir = Path(tempfile.gettempdir()) / "lens-serving-publish-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{digest}.lock").open("a+b") as lock_file:
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise GitHubReleaseError("다른 로컬 발행기가 실행 중입니다.") from exc
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="검증된 서빙 스냅샷을 비공개 GitHub 릴리스에 발행")
    parser.add_argument("--data-dir", default=str(configured_serving_data_dir()))
    parser.add_argument("--manifest")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--result-path")
    parser.add_argument(
        "--sync-mode",
        choices=("auto", "admin", "periodic"),
        default="auto",
        help="Render 즉시 동기화 또는 서버 주기 동기화 확인 방식",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="네트워크와 인증 없이 로컬 파일만 검사"
    )
    parser.add_argument(
        "--skip-backend-sync", action="store_true", help="최초 업로드 시에만 Render 동기화를 생략"
    )
    args = parser.parse_args(argv)
    result: dict[str, Any] = {"status": "starting", "source": "github_release"}
    client = None
    try:
        data_dir = Path(args.data_dir).resolve()
        manifest = (
            load_snapshot_manifest(args.manifest)
            if args.manifest
            else build_snapshot_manifest(data_dir, snapshot_id=args.snapshot_id)
        )
        config = get_github_release_config()
        result.update(build_publish_plan(config, manifest, data_dir))
        result["repository"] = config.repository
        result["release_tag"] = config.release_tag
        if args.dry_run:
            result.update(status="validated", dry_run=True)
        else:
            config = resolve_github_config(config, allow_cli_token=True)
            with _publisher_lock(config.repository, config.release_tag):
                client = GitHubReleaseClient(config, allow_create=True)
                # S3 발행기의 파일 검증·불변 이름·최신 포인터 마지막 교체 계약을 재사용한다.
                # 실제 저장과 통신은 위의 GitHub 클라이언트만 수행한다.
                result.update(publish_snapshot(data_dir, manifest, config=config, client=client))
                result["status"] = "published"
                result["retention"] = client.prune_old_snapshots()
            if args.skip_backend_sync:
                result["backend_sync"] = {"status": "skipped"}
            else:
                backend_url = os.environ.get("LENS_DEPLOYED_BACKEND_URL", DEFAULT_BACKEND_URL)
                admin_token = os.environ.get("LENS_ADMIN_RELOAD_TOKEN", "")
                if args.sync_mode == "admin" or (args.sync_mode == "auto" and admin_token.strip()):
                    result["backend_sync"] = notify_backend_sync(
                        base_url=backend_url,
                        admin_token=admin_token,
                        expected_snapshot_id=str(manifest["snapshot_id"]),
                        expected_source="github_release",
                    )
                else:
                    result["backend_sync"] = wait_for_backend_snapshot(
                        base_url=backend_url,
                        expected_snapshot_id=str(manifest["snapshot_id"]),
                    )
                result["status"] = "ready"
            result["dry_run"] = False
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (
        GitHubReleaseError,
        R2ConfigurationError,
        R2PublishError,
        SnapshotValidationError,
    ) as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        print(f"GitHub 릴리스 발행 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
        if args.result_path:
            output = Path(args.result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
