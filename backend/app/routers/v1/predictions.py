"""
v1 predictions endpoints — local parquet 직접 서빙.

Endpoints:
  GET /api/v1/predictions/line/{ticker}
  GET /api/v1/predictions/band/1d/{ticker}
  GET /api/v1/predictions/band/1w/{ticker}
  GET /api/v1/predictions/tickers
  GET /api/v1/predictions/health
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.core.http import success_response

router = APIRouter(prefix="/predictions", tags=["predictions"])


# ----- 캐시 -----
# main.py startup 에서 채워줌
_CACHE: dict[str, pd.DataFrame | None] = {
    "line_1d": None,
    "band_1d": None,
    "band_1w": None,
}


def load_caches(base_dir: Path) -> dict[str, Any]:
    """Startup 시 호출. 각 parquet 메모리 로드 + 캐시."""
    base = Path(base_dir)
    summary: dict[str, Any] = {}

    for slot, fname in [
        ("line_1d", "predictions_line_1d.parquet"),
        ("band_1d", "predictions_band_1d.parquet"),
        ("band_1w", "predictions_band_1w.parquet"),
    ]:
        path = base / fname
        if not path.exists():
            _CACHE[slot] = None
            summary[slot] = {"status": "missing", "path": str(path)}
            continue
        df = pd.read_parquet(path)
        _CACHE[slot] = df
        summary[slot] = {
            "status": "loaded",
            "rows": len(df),
            "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        }
    return summary


def _get_df(slot: str) -> pd.DataFrame:
    df = _CACHE.get(slot)
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=f"slot {slot} not loaded. Check backend/data/v1/ parquets and restart.",
        )
    return df


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas / nan to JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        try:
            v = value.item()
            return _jsonable(v)
        except Exception:
            return str(value)
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        records.append({k: _jsonable(v) for k, v in row.items()})
    return records


# ----- Endpoints -----


@router.get("/health")
def health(request: Request):
    """캐시 상태 확인."""
    status = {}
    for slot, df in _CACHE.items():
        if df is None:
            status[slot] = {"loaded": False}
        else:
            status[slot] = {
                "loaded": True,
                "rows": len(df),
                "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
            }
    return success_response(request, {"slots": status})


@router.get("/tickers")
def list_tickers(
    request: Request,
    search: str | None = Query(default=None, description="ticker prefix filter"),
    limit: int = Query(default=500, ge=1, le=1000),
):
    """가능한 ticker 목록 (1D Line 캐시 기준)."""
    df = _get_df("line_1d")
    tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())
    if search:
        s = search.upper()
        tickers = [t for t in tickers if t.startswith(s)]
    tickers = tickers[:limit]
    return success_response(request, {"tickers": tickers, "total": len(tickers)})


@router.get("/line/{ticker}")
def get_line(
    request: Request,
    ticker: str,
    days: int = Query(default=365, ge=1, le=730, description="과거 며칠 표시"),
):
    """1D Line — conservative h5 보수 예측선."""
    ticker = ticker.upper()
    df = _get_df("line_1d")
    sub = df[df["ticker"] == ticker].sort_values("asof_date")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"ticker {ticker} not found in 1D line cache")

    # filter by days
    cutoff = (date.fromisoformat(sub["asof_date"].max()) - timedelta(days=days)).isoformat()
    sub = sub[sub["asof_date"] >= cutoff]

    return success_response(
        request,
        {
            "ticker": ticker,
            "slot": "1D Line",
            "model_id": _jsonable(sub["model_id"].iloc[0]) if "model_id" in sub.columns else None,
            "source_cp": _jsonable(sub["source_cp"].iloc[0]) if "source_cp" in sub.columns else None,
            "rows": len(sub),
            "data": _df_to_records(sub),
        },
    )


@router.get("/band/1d/{ticker}")
def get_band_1d(
    request: Request,
    ticker: str,
    days: int = Query(default=365, ge=1, le=730),
    horizon: int | None = Query(default=None, ge=1, le=5, description="특정 horizon_step 만"),
):
    """1D Band — conformal q15/q85 historical."""
    ticker = ticker.upper()
    df = _get_df("band_1d")
    sub = df[df["ticker"] == ticker].sort_values(["asof_date", "horizon_step"])
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"ticker {ticker} not found in 1D band cache")

    cutoff = (date.fromisoformat(sub["asof_date"].max()) - timedelta(days=days)).isoformat()
    sub = sub[sub["asof_date"] >= cutoff]

    if horizon is not None:
        sub = sub[sub["horizon_step"] == horizon]

    return success_response(
        request,
        {
            "ticker": ticker,
            "slot": "1D Band",
            "model_id": _jsonable(sub["model_id"].iloc[0]) if "model_id" in sub.columns and len(sub) else None,
            "source_cp": _jsonable(sub["source_cp"].iloc[0]) if "source_cp" in sub.columns and len(sub) else None,
            "horizons": sorted(sub["horizon_step"].unique().tolist()) if len(sub) else [],
            "rows": len(sub),
            "data": _df_to_records(sub),
        },
    )


@router.get("/band/1w/{ticker}")
def get_band_1w(
    request: Request,
    ticker: str,
    days: int = Query(default=730, ge=1, le=1095),
    horizon: int | None = Query(default=None, ge=1, le=4),
):
    """1W Band — ensemble averaged."""
    ticker = ticker.upper()
    df = _get_df("band_1w")
    sub = df[df["ticker"] == ticker].sort_values(["asof_date", "horizon_step"])
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"ticker {ticker} not found in 1W band cache")

    cutoff = (date.fromisoformat(sub["asof_date"].max()) - timedelta(days=days)).isoformat()
    sub = sub[sub["asof_date"] >= cutoff]

    if horizon is not None:
        sub = sub[sub["horizon_step"] == horizon]

    return success_response(
        request,
        {
            "ticker": ticker,
            "slot": "1W Band",
            "model_id": _jsonable(sub["model_id"].iloc[0]) if "model_id" in sub.columns and len(sub) else None,
            "source_cp": _jsonable(sub["source_cp"].iloc[0]) if "source_cp" in sub.columns and len(sub) else None,
            "horizons": sorted(sub["horizon_step"].unique().tolist()) if len(sub) else [],
            "rows": len(sub),
            "data": _df_to_records(sub),
        },
    )
