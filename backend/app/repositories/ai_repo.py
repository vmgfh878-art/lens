"""AI run / evaluation / backtest 조회.

v1 에서는 Supabase 가 비활성이라 mock parquet/JSON 만 사용한다.
mock 데이터는 backend/data/v1/ai_runs_mock.json 에서 로드.
Supabase 부활 필요 시 이전 git 이력 참조.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_MOCK_PATH = Path(__file__).resolve().parents[2] / "data" / "v1" / "ai_runs_mock.json"
LEGACY_COMPOSITE_MODEL_NAMES = {"line_band_composite"}


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
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


def fetch_model_run(run_id: str) -> dict | None:
    for r in _mock_runs():
        if r.get("run_id") == run_id:
            return r
    return None


def fetch_run_evaluations(
    run_id: str,
    *,
    ticker: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> list[dict]:
    rows = _mock_evaluations(run_id)
    if ticker:
        rows = [r for r in rows if (r.get("ticker") or "").upper() == ticker.upper()]
    if timeframe:
        rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
    return rows[:limit]


def fetch_run_backtests(
    run_id: str,
    *,
    strategy_name: str | None = None,
    timeframe: str | None = None,
    limit: int = 50,
) -> list[dict]:
    rows = _mock_backtests(run_id)
    if strategy_name:
        rows = [r for r in rows if (r.get("strategy_name") or "") == strategy_name]
    if timeframe:
        rows = [r for r in rows if (r.get("timeframe") or "") == timeframe]
    return rows[:limit]
