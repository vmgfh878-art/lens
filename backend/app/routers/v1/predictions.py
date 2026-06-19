"""
v1 predictions endpoints — local parquet 또는 Supabase REST 서빙 (CP254 토글).

Endpoints:
  GET /api/v1/predictions/line/{ticker}
  GET /api/v1/predictions/band/1d/{ticker}
  GET /api/v1/predictions/band/1w/{ticker}
  GET /api/v1/predictions/tickers
  GET /api/v1/predictions/health

CP254 — data_backend.use_supabase() 가 True 면 prediction_repo 가 ticker 스코프
+ 날짜 윈도우 SQL 로 thin 조회한다. 응답 조립은 두 모드가 공유 (회귀 0).
프론트 실사용 엔드포인트라 Cache-Control + ETag 를 부여 (egress 가드).
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from app.core.http import success_response
from app.core.validators import TickerStr
from app.repositories import prediction_repo
from app.services import data_backend, parquet_store
from fastapi import APIRouter, HTTPException, Query, Request, Response

logger = logging.getLogger("lens.predictions")

router = APIRouter(prefix="/predictions", tags=["predictions"])

_SLOTS = ("line_1d", "band_1d", "band_1w")
_CACHE_HEADER = "public, max-age=3600"


def _build_slot_etag(slot: str, ticker: str, sub: pd.DataFrame, **params: Any) -> str:
    """슬롯 응답용 weak ETag — 최신 asof + 행수 + 쿼리 파라미터로 결정."""
    latest = ""
    if len(sub) and "asof_date" in sub.columns:
        latest = str(sub["asof_date"].max())
    param_text = "|".join(f"{k}={params[k]}" for k in sorted(params))
    raw = f"{slot}|{ticker}|{latest}|{len(sub)}|{param_text}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f'W/"{digest}"'


def load_caches(base_dir: Path) -> dict[str, Any]:
    """admin/reload 및 startup 에서 호출. 공유 store 를 비우고 각 슬롯 상태 반환.

    base_dir 인자는 하위 호환성을 위해 유지하지만 실제 경로는 parquet_store 가 관리.
    CP254 — REST 모드에서는 parquet 를 메모리에 올리지 않는다 (RSS 절감이 목적인데
    reload 가 도로 적재하면 무의미). 캐시 비움만 수행.
    """
    parquet_store.clear_all()
    if data_backend.use_supabase():
        return {slot: {"status": "skipped_supabase_mode"} for slot in _SLOTS}
    summary: dict[str, Any] = {}
    for slot in _SLOTS:
        try:
            df = parquet_store.get_raw(slot)
        except Exception as exc:  # noqa: BLE001 — admin reload 응답 유지. 원인은 로그로.
            logger.warning("load_caches slot '%s' 실패", slot, exc_info=exc)
            summary[slot] = {"status": "error", "error": str(exc)}
            continue
        if df is None:
            summary[slot] = {"status": "missing"}
        else:
            summary[slot] = {
                "status": "loaded",
                "rows": len(df),
                "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
                "mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1),
            }
    return summary


def _get_df(slot: str) -> pd.DataFrame:
    try:
        df = parquet_store.get_raw(slot)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    if isinstance(value, pd.Timestamp):
        # CP245 — asof/forecast 가 store 에서 datetime64 라 Timestamp 로 온다.
        # 기존 object("YYYY-MM-DD") 응답과 동일하도록 date 문자열로 직렬화한다.
        return None if pd.isna(value) else value.strftime("%Y-%m-%d")
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
    """캐시/백엔드 상태 확인.

    CP221: 정의되지 않은 모듈 변수 `_CACHE` 참조로 NameError(500)가 나던 버그 수정.
    parquet_store.stats() 가 동일 정보(slot별 status/rows/tickers/mb)를 thread-safe 하게 반환한다.
    CP254: REST 모드에서는 슬롯 테이블 행수(count 쿼리, payload ~0)를 보고한다.
    """
    if data_backend.use_supabase():
        counts = prediction_repo.slot_counts()
        slots = {slot: {"status": "supabase", "rows": count} for slot, count in counts.items()}
        return success_response(request, {"slots": slots})
    return success_response(request, {"slots": parquet_store.stats()})


@router.get("/tickers")
def list_tickers(
    request: Request,
    response: Response,
    search: str | None = Query(default=None, description="ticker prefix filter"),
    limit: int = Query(default=500, ge=1, le=1000),
):
    """가능한 ticker 목록 (로컬=1D Line 캐시 기준, REST=stock_info 기준)."""
    if data_backend.use_supabase():
        tickers = prediction_repo.list_tickers(search=search, limit=limit)
    else:
        df = _get_df("line_1d")
        tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())
        if search:
            s = search.upper()
            tickers = [t for t in tickers if t.startswith(s)]
        tickers = tickers[:limit]
    response.headers["Cache-Control"] = _CACHE_HEADER
    return success_response(request, {"tickers": tickers, "total": len(tickers)})


@router.get("/line/{ticker}")
def get_line(
    request: Request,
    response: Response,
    ticker: TickerStr,
    days: int = Query(default=365, ge=1, le=730, description="과거 며칠 표시"),
):
    """1D Line — conservative h5 보수 예측선."""
    ticker = ticker.upper()
    if data_backend.use_supabase():
        sub = prediction_repo.fetch_line_1d(ticker, days=days)
        if sub is None:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1D line cache"
            )
    else:
        df = _get_df("line_1d")
        sub = df[df["ticker"] == ticker].sort_values("asof_date")
        if sub.empty:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1D line cache"
            )
        # filter by days
        cutoff = sub["asof_date"].max() - pd.Timedelta(days=days)
        sub = sub[sub["asof_date"] >= cutoff]

    etag = _build_slot_etag("1D Line", ticker, sub, days=days)
    headers = {"Cache-Control": _CACHE_HEADER, "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return success_response(
        request,
        {
            "ticker": ticker,
            "slot": "1D Line",
            "model_id": _jsonable(sub["model_id"].iloc[0]) if "model_id" in sub.columns else None,
            "source_cp": _jsonable(sub["source_cp"].iloc[0])
            if "source_cp" in sub.columns
            else None,
            "rows": len(sub),
            "data": _df_to_records(sub),
        },
    )


def _band_response(
    request: Request,
    response: Response,
    *,
    slot_label: str,
    ticker: str,
    sub: pd.DataFrame,
    days: int,
    horizon: int | None,
):
    """1D/1W band 공용 응답 조립 — 두 모드(parquet/REST)가 동일 직렬화 경로 공유."""
    etag = _build_slot_etag(slot_label, ticker, sub, days=days, horizon=horizon)
    headers = {"Cache-Control": _CACHE_HEADER, "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return success_response(
        request,
        {
            "ticker": ticker,
            "slot": slot_label,
            "model_id": _jsonable(sub["model_id"].iloc[0])
            if "model_id" in sub.columns and len(sub)
            else None,
            "source_cp": _jsonable(sub["source_cp"].iloc[0])
            if "source_cp" in sub.columns and len(sub)
            else None,
            "horizons": sorted(sub["horizon_step"].unique().tolist()) if len(sub) else [],
            "rows": len(sub),
            "data": _df_to_records(sub),
        },
    )


@router.get("/band/1d/{ticker}")
def get_band_1d(
    request: Request,
    response: Response,
    ticker: TickerStr,
    days: int = Query(default=365, ge=1, le=730),
    horizon: int | None = Query(default=None, ge=1, le=5, description="특정 horizon_step 만"),
):
    """1D Band — conformal q15/q85 historical."""
    ticker = ticker.upper()
    if data_backend.use_supabase():
        sub = prediction_repo.fetch_band_1d(ticker, days=days, horizon=horizon)
        if sub is None:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1D band cache"
            )
    else:
        df = _get_df("band_1d")
        sub = df[df["ticker"] == ticker].sort_values(["asof_date", "horizon_step"])
        if sub.empty:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1D band cache"
            )
        cutoff = sub["asof_date"].max() - pd.Timedelta(days=days)
        sub = sub[sub["asof_date"] >= cutoff]
        if horizon is not None:
            sub = sub[sub["horizon_step"] == horizon]

    return _band_response(
        request, response, slot_label="1D Band", ticker=ticker, sub=sub, days=days, horizon=horizon
    )


@router.get("/band/1w/{ticker}")
def get_band_1w(
    request: Request,
    response: Response,
    ticker: TickerStr,
    days: int = Query(default=730, ge=1, le=1095),
    horizon: int | None = Query(default=None, ge=1, le=4),
):
    """1W Band — ensemble averaged."""
    ticker = ticker.upper()
    if data_backend.use_supabase():
        sub = prediction_repo.fetch_band_1w(ticker, days=days, horizon=horizon)
        if sub is None:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1W band cache"
            )
    else:
        df = _get_df("band_1w")
        sub = df[df["ticker"] == ticker].sort_values(["asof_date", "horizon_step"])
        if sub.empty:
            raise HTTPException(
                status_code=404, detail=f"ticker {ticker} not found in 1W band cache"
            )
        cutoff = sub["asof_date"].max() - pd.Timedelta(days=days)
        sub = sub[sub["asof_date"] >= cutoff]
        if horizon is not None:
            sub = sub[sub["horizon_step"] == horizon]

    return _band_response(
        request, response, slot_label="1W Band", ticker=ticker, sub=sub, days=days, horizon=horizon
    )
