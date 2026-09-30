from __future__ import annotations

from unittest.mock import Mock, patch

from app import main


def test_periodic_sync_activates_new_version_and_stops() -> None:
    stop_event = Mock()
    stop_event.wait.side_effect = [False, True]
    with patch.object(
        main, "sync_latest_serving_snapshot", return_value={"changed": True, "snapshot_id": "new"}
    ) as sync:
        main._periodic_remote_sync(stop_event, 300)
    assert sync.call_count == 1
    assert stop_event.wait.call_count == 2
    stop_event.wait.assert_called_with(300)


def test_periodic_sync_failure_retries_without_raising() -> None:
    stop_event = Mock()
    stop_event.wait.side_effect = [False, False, True]
    with patch.object(
        main,
        "sync_latest_serving_snapshot",
        side_effect=[RuntimeError("통신 실패"), {"changed": False}],
    ) as sync:
        main._periodic_remote_sync(stop_event, 300)
    assert sync.call_count == 2


def test_periodic_sync_does_not_start_without_opt_in() -> None:
    with (
        patch.object(main, "get_remote_sync_config", return_value=Mock(interval_seconds=0)),
        patch.object(main, "Thread") as thread,
    ):
        main._start_periodic_remote_sync()
    thread.assert_not_called()


def test_periodic_sync_starts_only_with_configured_remote() -> None:
    with (
        patch.object(main, "get_remote_sync_config", return_value=Mock(interval_seconds=300)),
        patch.object(main, "get_remote_serving_config", return_value=Mock(configured=True)),
        patch.object(main, "Thread") as thread_type,
    ):
        main._start_periodic_remote_sync()
        thread_type.return_value.start.assert_called_once_with()
        main._stop_periodic_remote_sync()
        thread_type.return_value.join.assert_called_once_with(timeout=1)
