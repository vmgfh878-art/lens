from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from backend.scripts.verify_deployed_serving import verify_deployed_serving


def _responses(
    *,
    source: str = "r2",
    snapshot_id: str = "daily-1",
    health_status: str = "ok",
    price: str = "2026-09-25",
    predictions: str = "2026-09-25",
    bootstrap_verified: bool = True,
) -> Any:
    def get_json(url: str) -> dict[str, Any]:
        if url.endswith("/health/ready"):
            return {
                "data": {
                    "source": source,
                    "snapshot_id": snapshot_id,
                    "status": health_status,
                    "r2_sync_status": "ready" if health_status == "ok" else "error",
                    "checks": {"required_files": True, "bootstrap_verified": bootstrap_verified},
                }
            }
        column = "date" if "/stocks/" in url else "asof_date"
        return {"data": {"data": [{column: price if column == "date" else predictions}]}}

    return get_json


@pytest.mark.parametrize(
    ("source", "health_status", "snapshot_id", "expected", "verdict"),
    [
        ("r2", "ok", "daily-1", "daily-1", "VERIFIED"),
        ("github_release", "ok", "daily-1", "daily-1", "VERIFIED"),
        ("github_release", "degraded", "daily-1", None, "REMOTE_DEGRADED"),
        ("github_release", "ok", "daily-old", "daily-1", "MISMATCH"),
        ("bootstrap", "degraded", "bootstrap-1", "daily-1", "FALLBACK"),
        ("r2", "degraded", "daily-1", None, "R2_DEGRADED"),
        ("r2", "ok", "daily-old", "daily-1", "MISMATCH"),
        ("supabase", "ok", "daily-1", None, "LEGACY"),
    ],
)
def test_serving_sources_and_versions_are_not_reported_as_the_same_success(
    source: str,
    health_status: str,
    snapshot_id: str,
    expected: str | None,
    verdict: str,
) -> None:
    result = verify_deployed_serving(
        expected_snapshot_id=expected,
        today=date(2026, 9, 26),
        get_json=_responses(source=source, health_status=health_status, snapshot_id=snapshot_id),
    )
    assert result["result"] == verdict
    assert result["fallback_active"] == (source == "bootstrap")


def test_corrupt_bootstrap_is_an_error() -> None:
    result = verify_deployed_serving(
        today=date(2026, 9, 26),
        get_json=_responses(source="bootstrap", bootstrap_verified=False),
    )
    assert result["result"] == "ERROR"


def test_old_price_and_equally_old_predictions_are_stale() -> None:
    result = verify_deployed_serving(
        today=date(2026, 9, 26),
        get_json=_responses(price="2026-08-01", predictions="2026-08-01"),
    )
    assert result["result"] == "STALE"


def test_unreachable_backend_is_an_error_without_leaking_request_secrets() -> None:
    def failing_get_json(url: str) -> dict[str, Any]:
        raise RuntimeError("request secret-value")

    result = verify_deployed_serving(get_json=failing_get_json)
    assert result["result"] == "ERROR"
    assert "secret-value" not in str(result)
