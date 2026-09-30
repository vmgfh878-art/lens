from __future__ import annotations

from typing import Any

import pytest
from app.services.r2_storage import R2PublishError

from backend.scripts.publish_serving_to_github import wait_for_backend_snapshot


class _Response:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self.data = data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return {"data": self.data}


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str) -> _Response:
        self.urls.append(url)
        return self.responses.pop(0)


def _ready(snapshot_id: str) -> _Response:
    return _Response(
        {
            "source": "github_release",
            "snapshot_id": snapshot_id,
            "status": "ok",
            "checks": {"required_files": True},
        }
    )


def test_wait_for_backend_snapshot_requires_exact_new_version() -> None:
    client = _Client([_ready("old"), _ready("new")])
    waits: list[float] = []

    result = wait_for_backend_snapshot(
        base_url="https://lens.example/",
        expected_snapshot_id="new",
        attempts=2,
        sleep_seconds=2,
        http_client=client,
        sleep_fn=waits.append,
    )

    assert result["status"] == "ready"
    assert result["snapshot_id"] == "new"
    assert result["mode"] == "periodic_poll"
    assert waits == [2]
    assert client.urls == ["https://lens.example/api/v1/health/ready"] * 2


def test_wait_for_backend_snapshot_does_not_accept_fallback() -> None:
    client = _Client(
        [
            _Response(
                {
                    "source": "bootstrap",
                    "snapshot_id": "new",
                    "status": "ok",
                    "checks": {"required_files": True},
                }
            )
        ]
    )
    with pytest.raises(R2PublishError, match="활성화하지 않았습니다"):
        wait_for_backend_snapshot(
            base_url="https://lens.example",
            expected_snapshot_id="new",
            attempts=1,
            http_client=client,
        )


def test_wait_for_backend_snapshot_rejects_http_4xx_without_retry() -> None:
    client = _Client([_Response({}, status_code=403)])
    waits: list[float] = []
    with pytest.raises(R2PublishError, match="거부됐습니다"):
        wait_for_backend_snapshot(
            base_url="https://lens.example",
            expected_snapshot_id="new",
            attempts=2,
            http_client=client,
            sleep_fn=waits.append,
        )
    assert waits == []
