from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.services.serving_snapshot import REQUIRED_SERVING_FILES, write_snapshot_manifest

from backend.scripts.audit_serving_snapshot import audit_serving_snapshot


def test_offline_audit_is_read_only_and_reports_unverified_remote_checks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in REQUIRED_SERVING_FILES:
        (source / name).write_bytes(f"audit-fixture-{name}".encode())
    write_snapshot_manifest(source, snapshot_id="fixture-bootstrap")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with (
        patch("backend.scripts.audit_serving_snapshot.inspect_latest_serving_snapshot") as remote,
        patch("backend.scripts.audit_serving_snapshot.verify_deployed_serving") as deployed,
    ):
        result = audit_serving_snapshot(
            local_dir=source, bootstrap_dir=source, skip_remote=True, skip_deployed=True
        )
    assert result["status"] == "WARN"
    assert result["checks"]["local"]["status"] == "PASS"
    assert result["checks"]["bootstrap"]["status"] == "PASS"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    remote.assert_not_called()
    deployed.assert_not_called()


def test_audit_rejects_missing_source_and_corrupt_frozen_manifest(tmp_path: Path) -> None:
    result = audit_serving_snapshot(
        local_dir=tmp_path / "missing", bootstrap_dir=tmp_path, skip_remote=True, skip_deployed=True
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["local"]["status"] == "FAIL"
    assert result["checks"]["bootstrap"]["status"] == "FAIL"
