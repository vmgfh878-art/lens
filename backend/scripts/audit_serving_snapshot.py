"""로컬 산출물·고정 폴백·원격 릴리스·배포 사이의 계약을 변경 없이 검사한다."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "backend"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.config import GitHubReleaseConfig
from app.services.github_release_storage import GitHubReleaseError, resolve_github_config
from app.services.serving_paths import configured_serving_data_dir, get_serving_bootstrap_dir
from app.services.serving_snapshot import (
    SNAPSHOT_MANIFEST_FILENAME,
    build_snapshot_manifest,
    load_snapshot_manifest,
    validate_snapshot_directory,
)
from app.services.serving_storage import get_remote_serving_config, remote_serving_source
from app.services.serving_sync import inspect_latest_serving_snapshot

from backend.scripts.verify_deployed_serving import BASE, verify_deployed_serving


def audit_serving_snapshot(
    *,
    local_dir: Path,
    bootstrap_dir: Path,
    skip_remote: bool = False,
    skip_deployed: bool = False,
    verify_r2_files: bool = False,
    verify_remote_files: bool = False,
    base_url: str = BASE,
) -> dict[str, Any]:
    report: dict[str, Any] = {"checks": {}, "read_only": True}
    checks = report["checks"]
    local_hash = None
    remote_snapshot_id = None
    for label, directory in (("local", local_dir), ("bootstrap", bootstrap_dir)):
        try:
            # 고정 폴백은 저장된 매니페스트와 대조한다. 로컬은 현재 생성 파일의 지문을 계산한다.
            manifest = (
                load_snapshot_manifest(directory / SNAPSHOT_MANIFEST_FILENAME)
                if label == "bootstrap"
                else build_snapshot_manifest(directory, snapshot_id="local-audit")
            )
            summary = validate_snapshot_directory(manifest, directory)
            checks[label] = {"status": "PASS", **summary}
            if label == "local":
                local_hash = summary["content_sha256"]
        except Exception as exc:
            checks[label] = {"status": "FAIL", "error_type": type(exc).__name__}

    config = get_remote_serving_config()
    if not skip_remote and isinstance(config, GitHubReleaseConfig) and not config.configured:
        with suppress(GitHubReleaseError):
            config = resolve_github_config(config, allow_cli_token=True)
    remote_source = remote_serving_source(config)
    if skip_remote or not config.configured:
        checks["remote"] = {
            "status": "WARN",
            "source": remote_source,
            "reason": "검사 생략" if skip_remote else "원격 읽기 인증 없음",
        }
    else:
        try:
            remote = inspect_latest_serving_snapshot(
                config=config, verify_files=verify_remote_files or verify_r2_files
            )
            manifest = remote["manifest"]
            remote_snapshot_id = manifest["snapshot_id"]
            local_matches = local_hash == manifest["content_sha256"] if local_hash else None
            checks["remote"] = {
                "source": remote_source,
                "status": "PASS" if local_matches else "WARN",
                "snapshot_id": remote_snapshot_id,
                "content_sha256": manifest["content_sha256"],
                "published_at": remote["pointer"]["published_at"],
                "files_verified": remote["files_verified"],
                "local_content_matches": local_matches,
            }
        except Exception as exc:
            checks["remote"] = {
                "status": "FAIL",
                "source": remote_source,
                "error_type": type(exc).__name__,
            }

    if skip_deployed:
        checks["deployed"] = {"status": "WARN", "reason": "검사 생략"}
    else:
        deployed = verify_deployed_serving(
            base_url=base_url, expected_snapshot_id=remote_snapshot_id
        )
        verdict = deployed["result"]
        checks["deployed"] = {
            "status": "PASS"
            if verdict == "VERIFIED"
            else "WARN"
            if verdict in {"FALLBACK", "R2_DEGRADED", "REMOTE_DEGRADED", "LEGACY"}
            else "FAIL",
            **deployed,
        }
    statuses = {check["status"] for check in checks.values()}
    report["status"] = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="서빙 스냅샷 읽기 전용 품질 감사")
    parser.add_argument("--local-dir", default=str(configured_serving_data_dir()))
    parser.add_argument("--bootstrap-dir", default=str(get_serving_bootstrap_dir()))
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument(
        "--verify-remote-files",
        "--verify-r2-files",
        dest="verify_remote_files",
        action="store_true",
        help="원격 파일을 메모리에서 해시 검증",
    )
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--skip-deployed", action="store_true")
    args = parser.parse_args(argv)
    report = audit_serving_snapshot(
        local_dir=Path(args.local_dir),
        bootstrap_dir=Path(args.bootstrap_dir),
        skip_remote=args.skip_remote,
        skip_deployed=args.skip_deployed,
        verify_remote_files=args.verify_remote_files,
        base_url=args.base_url,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
