from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase, supabase_is_configured


_MOCK_PATH = Path(__file__).resolve().parents[2] / "data" / "v1" / "ai_runs_mock.json"


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
    """v1 학교 데모용 mock AI runs. Supabase 미설정 시 사용."""
    if not _MOCK_PATH.exists():
        return {"runs": [], "evaluations": {}, "backtests": {}}
    try:
        return json.loads(_MOCK_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"runs": [], "evaluations": {}, "backtests": {}}


def _mock_runs() -> list[dict]:
    return list(_load_mock().get("runs", []))


def _mock_evaluations(run_id: str) -> list[dict]:
    return list(_load_mock().get("evaluations", {}).get(run_id, []))


def _mock_backtests(run_id: str) -> list[dict]:
    return list(_load_mock().get("backtests", {}).get(run_id, []))


RUN_COLUMNS = (
    "run_id, wandb_run_id, model_name, timeframe, horizon, val_metrics, "
    "test_metrics, config, checkpoint_path, status, feature_version, band_mode, created_at"
)
EVALUATION_COLUMNS = (
    "run_id, ticker, timeframe, asof_date, coverage, avg_band_width, "
    "lower_breach_rate, upper_breach_rate, direction_accuracy, mae, smape"
)
BACKTEST_COLUMNS = (
    "run_id, strategy_name, timeframe, return_pct, mdd, sharpe, win_rate, "
    "profit_factor, num_trades, meta, created_at"
)
LEGACY_COMPOSITE_MODEL_NAMES = {"line_band_composite"}


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "deprecated", "legacy"}
    return bool(value)


def is_legacy_composite_run(row: dict) -> bool:
    config = _as_dict(row.get("config"))
    model_name = str(row.get("model_name") or "").strip().lower()
    role = str(row.get("role") or config.get("role") or config.get("model_role") or "").strip().lower()
    if model_name in LEGACY_COMPOSITE_MODEL_NAMES:
        return True
    if role == "composite_model":
        return True
    if _truthy(config.get("deprecated_for_phase1_product_contract")):
        return True
    if config.get("indicator_layer_replacement") is not None and model_name in LEGACY_COMPOSITE_MODEL_NAMES:
        return True
    return False


def fetch_model_runs(
    *,
    model_name: str | None = None,
    timeframe: str | None = None,
    status: str | None = "completed",
    include_legacy: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    if not supabase_is_configured():
        rows = _mock_runs()
        if model_name:
            rows = [r for r in rows if (r.get("model_name") or "").lower() == model_name.lower()]
        if timeframe:
            rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
        if status is not None:
            rows = [r for r in rows if (r.get("status") or "") == status]
        if not include_legacy:
            rows = [r for r in rows if not is_legacy_composite_run(r)]
        return rows[offset : offset + limit]
    try:
        client = get_supabase()
        query = client.table("model_runs").select(RUN_COLUMNS).order("created_at", desc=True)
        if model_name:
            query = query.eq("model_name", model_name)
        if timeframe:
            query = query.eq("timeframe", timeframe)
        if status is not None:
            query = query.eq("status", status)
        if include_legacy:
            end = offset + limit - 1
            rows = query.range(offset, end).execute().data or []
            return rows
        filtered_rows: list[dict] = []
        scan_offset = 0
        scan_page_size = max(limit * 3, 50)
        while len(filtered_rows) < offset + limit:
            rows = query.range(scan_offset, scan_offset + scan_page_size - 1).execute().data or []
            if not rows:
                break
            filtered_rows.extend(row for row in rows if not is_legacy_composite_run(row))
            if len(rows) < scan_page_size:
                break
            scan_offset += scan_page_size
        return filtered_rows[offset : offset + limit]
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 목록을 조회할 수 없습니다.") from exc


def fetch_model_run(run_id: str) -> dict | None:
    if not supabase_is_configured():
        for r in _mock_runs():
            if r.get("run_id") == run_id:
                return r
        return None
    try:
        client = get_supabase()
        rows = (
            client.table("model_runs")
            .select(RUN_COLUMNS)
            .eq("run_id", run_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 상세를 조회할 수 없습니다.") from exc


def fetch_run_evaluations(
    run_id: str,
    *,
    ticker: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> list[dict]:
    if not supabase_is_configured():
        rows = _mock_evaluations(run_id)
        if ticker:
            rows = [r for r in rows if (r.get("ticker") or "").upper() == ticker.upper()]
        if timeframe:
            rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
        return rows[:limit]
    try:
        client = get_supabase()
        query = (
            client.table("prediction_evaluations")
            .select(EVALUATION_COLUMNS)
            .eq("run_id", run_id)
            .order("asof_date", desc=True)
        )
        if ticker:
            query = query.eq("ticker", ticker.upper())
        if timeframe:
            query = query.eq("timeframe", timeframe)
        return query.limit(limit).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 평가 결과를 조회할 수 없습니다.") from exc


def fetch_run_backtests(
    run_id: str,
    *,
    strategy_name: str | None = None,
    timeframe: str | None = None,
    limit: int = 50,
) -> list[dict]:
    if not supabase_is_configured():
        rows = _mock_backtests(run_id)
        if strategy_name:
            rows = [r for r in rows if (r.get("strategy_name") or "") == strategy_name]
        if timeframe:
            rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
        return rows[:limit]
    try:
        client = get_supabase()
        query = (
            client.table("backtest_results")
            .select(BACKTEST_COLUMNS)
            .eq("run_id", run_id)
            .order("created_at", desc=True)
        )
        if strategy_name:
            query = query.eq("strategy_name", strategy_name)
        if timeframe:
            query = query.eq("timeframe", timeframe)
        return query.limit(limit).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError("AI run 백테스트 결과를 조회할 수 없습니다.") from exc
