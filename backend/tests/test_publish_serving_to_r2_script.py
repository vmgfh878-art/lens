from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from app.services.r2_storage import R2PublishError
from app.services.serving_snapshot import REQUIRED_SERVING_FILES

from backend.scripts.publish_serving_to_r2 import backend_sync_url, notify_backend_sync


class _Response:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._body


class _HttpClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str]) -> _Response:
        self.calls.append({"url": url, "headers": headers})
        return self.responses.pop(0)


def _ready_response(snapshot_id: str) -> _Response:
    return _Response(
        200,
        {
            "data": {
                "status": "ready",
                "source": "r2",
                "snapshot_id": snapshot_id,
                "changed": True,
            }
        },
    )


def test_backend_sync_url_accepts_base_or_full_endpoint() -> None:
    expected = "https://lens.example/api/v1/admin/sync-serving"
    assert backend_sync_url("https://lens.example/") == expected
    assert backend_sync_url(expected) == expected


def test_notify_backend_sync_checks_activated_snapshot() -> None:
    client = _HttpClient([_ready_response("daily-1")])

    result = notify_backend_sync(
        base_url="https://lens.example",
        admin_token="secret",
        expected_snapshot_id="daily-1",
        http_client=client,
    )

    assert result["status"] == "ready"
    assert result["attempt"] == 1
    assert client.calls[0]["headers"] == {"X-Lens-Admin-Token": "secret"}
    assert "secret" not in str(result)


def test_notify_backend_sync_retries_server_error() -> None:
    client = _HttpClient([_Response(503, {}), _ready_response("daily-1")])
    waits: list[int] = []

    result = notify_backend_sync(
        base_url="https://lens.example",
        admin_token="secret",
        expected_snapshot_id="daily-1",
        http_client=client,
        sleep_fn=waits.append,
    )

    assert result["attempt"] == 2
    assert waits == [5]


def test_notify_backend_sync_does_not_retry_auth_failure() -> None:
    client = _HttpClient([_Response(403, {})])
    waits: list[int] = []

    with pytest.raises(R2PublishError, match="거부"):
        notify_backend_sync(
            base_url="https://lens.example",
            admin_token="wrong",
            expected_snapshot_id="daily-1",
            http_client=client,
            sleep_fn=waits.append,
        )

    assert waits == []


def test_notify_backend_sync_rejects_different_active_snapshot() -> None:
    client = _HttpClient([_ready_response("older")])

    with pytest.raises(R2PublishError, match="snapshot_id"):
        notify_backend_sync(
            base_url="https://lens.example",
            admin_token="secret",
            expected_snapshot_id="daily-1",
            http_client=client,
        )


def test_publish_result_keeps_uploaded_snapshot_when_render_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.scripts import publish_serving_to_r2 as script

    source = tmp_path / "source"
    source.mkdir()
    for name in REQUIRED_SERVING_FILES:
        (source / name).write_bytes(f"publish-fixture-{name}".encode())
    monkeypatch.setattr(
        script, "publish_snapshot", lambda *args, **kwargs: {"snapshot_id": "daily-1"}
    )

    def fail_sync(**kwargs: Any) -> dict[str, Any]:
        raise R2PublishError("동기화 실패")

    monkeypatch.setattr(script, "notify_backend_sync", fail_sync)
    output = tmp_path / "publish-result.json"
    exit_code = script.main(
        ["--data-dir", str(source), "--snapshot-id", "daily-1", "--result-path", str(output)]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert result["status"] == "failed"
    assert result["snapshot_id"] == "daily-1"
