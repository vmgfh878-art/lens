from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIFIED_REFRESH = ROOT / "scripts" / "run_v1_unified_refresh_local.ps1"


def test_unified_refresh_uses_releases_instead_of_git_or_supabase_publish() -> None:
    script = UNIFIED_REFRESH.read_text(encoding="utf-8")

    assert "backend\\scripts\\publish_serving_to_github.py" in script
    assert "--sync-mode periodic" in script
    assert "Invoke-V1ProductionPush" not in script
    assert "publish_serving_to_supabase.py" not in script
    assert '$PushStatus = "DISABLED_GITHUB_RELEASE"' in script
    assert "publish_serving_to_r2.py" not in script


def test_render_checks_the_release_every_five_minutes() -> None:
    config = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "- key: LENS_REMOTE_SYNC_INTERVAL_SECONDS\n        value: '300'" in config


def test_release_or_deployed_verification_failure_fails_scheduled_run() -> None:
    script = UNIFIED_REFRESH.read_text(encoding="utf-8")

    assert '$FinalStatus = "FAIL_GITHUB_PUBLISH"' in script
    assert '$FinalStatus = "FAIL_DEPLOYED_VERIFY"' in script
    assert "if (($FailedSteps.Count -gt 0) -or $ProductionFailed)" in script


def test_existing_collection_and_calculation_steps_remain_in_order() -> None:
    script = UNIFIED_REFRESH.read_text(encoding="utf-8")
    markers = [
        '"scripts\\cp151_yfinance_500_backfill.py"',
        '"backend\\scripts\\build_v1_market_local.py"',
        '"backend\\scripts\\cp210_band_forward_refresh.py"',
        '"backend\\scripts\\cp212_line_1d_export.py"',
        '"backend\\scripts\\rebuild_product_history_parquet.py"',
        '"backend\\scripts\\build_ai_runs_mock.py"',
        '"backend\\scripts\\compact_v1_parquets.py"',
    ]
    positions = [script.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert '"--start-date", "2015-01-01"' in script
    assert '"--incremental-lookback-days", "10"' in script
    assert '"--indicator-mode", "auto"' in script
    assert '"--batch-size", "2048"' in script
    assert '"--device", "auto"' in script
    assert '"backend\\scripts\\compact_v1_parquets.py", "--apply"' in script
    assert '$ModeArg = if ($Apply) { "--apply" } else { "--dry-run" }' in script


def test_slack_reports_data_and_remote_serving_outcome() -> None:
    script = UNIFIED_REFRESH.read_text(encoding="utf-8")
    assert "Send-SlackNotification -Text $SlackText -Color $SlackColor" in script
    assert "원격 동기화 상태=$($VerifyData.remote_sync_status)" in script
    assert "$Metrics | Add-Member -NotePropertyName slack_notification_status" in script
