"""Shared in-process parquet cache for prediction data.

predictions.py and strategy_backtest_svc both need line_1d and band_1d.
Without this store they each call pd.read_parquet() independently, keeping
two copies of the same 60-80 MB DataFrames alive simultaneously.

This module loads each prediction parquet exactly once and lets all callers
share the same object reference.  Thread-safe via per-store Lock.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd
import structlog
from app.schemas import frames as _frames

logger = structlog.get_logger("lens.parquet_store")

_BASE = Path(__file__).resolve().parents[2] / "data" / "v1"

# Only prediction parquets live here.
# market_prices / market_indicators stay in local_market_svc (different date
# format + different computed columns; sharing would add a third copy, not save one).
_FILE_MAP: dict[str, str] = {
    "line_1d": "predictions_line_1d.parquet",
    "band_1d": "predictions_band_1d.parquet",
    "band_1w": "predictions_band_1w.parquet",
}

# CP227 — read 경계 dtype 계약 검증 (CP214 회귀 방지).
# 매핑에 없는 슬롯은 검증을 거치지 않는다 (신규 슬롯 추가 시 의도적 통과).
_SLOT_MODELS = {
    "line_1d": _frames.LineDailyFrame,
    "band_1d": _frames.Band1dFrame,
    "band_1w": _frames.Band1wFrame,
}


def _validate_slot(name: str, df: pd.DataFrame) -> pd.DataFrame:
    model = _SLOT_MODELS.get(name)
    if model is None:
        return df
    return _frames.validate(model, df, name=name)


_FRAMES: dict[str, pd.DataFrame | None] = {}
_LOCK = Lock()


def get_raw(name: str) -> pd.DataFrame | None:
    """Return cached DataFrame for `name`, loading from disk on first access.

    Returns None if the file does not exist on disk.
    Raises ValueError for unknown slot names.
    """
    if name not in _FILE_MAP:
        raise ValueError(f"Unknown parquet slot: {name!r}. Valid: {sorted(_FILE_MAP)}")
    with _LOCK:
        if name not in _FRAMES:
            _FRAMES[name] = _load(name)
        return _FRAMES[name]


def _load(name: str) -> pd.DataFrame | None:
    path = _BASE / _FILE_MAP[name]
    if not path.exists():
        logger.warning("parquet missing: %s", path)
        return None
    mb_disk = path.stat().st_size / 1024 / 1024
    df = pd.read_parquet(path)
    df = _validate_slot(name, df)
    df = _compress_strings(df)
    mb_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info("loaded parquet %s (%.1f MB disk → %.1f MB memory)", name, mb_disk, mb_mem)
    return df


def _compress_strings(df: pd.DataFrame) -> pd.DataFrame:
    """object 컬럼은 ordered categorical 로, 날짜 컬럼은 datetime64 로 압축해 메모리 절감.

    model_id / source_cp 처럼 한 값이 597 k 행 반복되는 컬럼은 category 로 (44MB → <1MB).

    CP245 — `asof_date` / `forecast_date` 는 **datetime64 로 변환**한다. 이전엔 CP214 가
    category 비교 TypeError 때문에 object 로 뒀으나, 597 k 행 날짜 문자열이 수십 MB 를
    먹었다(band_1d 97MB 의 큰 비중). 라우터(predictions.py)의 `asof_date >= cutoff` 비교를
    문자열이 아니라 `pd.Timestamp` 로 올려 정공법으로 해결했고, strategy_scan 은
    `pd.to_datetime(asof_date)` 로 datetime64 를 그대로 받으므로 영향이 없다. 응답
    직렬화(_jsonable)는 Timestamp 를 date 문자열로 내보내 출력 JSON 은 기존과 동일하다.
    """
    cat_dtype = pd.CategoricalDtype(ordered=True)
    date_cols = {"asof_date", "forecast_date"}
    for col in df.select_dtypes(include="object").columns:
        if col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(cat_dtype)
    return df


def clear_all() -> dict[str, str]:
    """Evict all cached frames; each will be reloaded on next access."""
    with _LOCK:
        evicted = list(_FRAMES.keys())
        _FRAMES.clear()
    if evicted:
        logger.info("parquet_store cleared: %s", evicted)
    return {k: "cleared" for k in evicted}


def stats() -> dict[str, dict]:
    """Per-slot status and in-process memory usage (MB)."""
    with _LOCK:
        out: dict[str, dict] = {}
        for name, df in _FRAMES.items():
            if df is None:
                out[name] = {"status": "missing"}
            else:
                mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1)
                out[name] = {
                    "status": "loaded",
                    "rows": len(df),
                    "mb": mb,
                    "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
                }
    for name in _FILE_MAP:
        if name not in out:
            out[name] = {"status": "not_loaded"}
    return out
