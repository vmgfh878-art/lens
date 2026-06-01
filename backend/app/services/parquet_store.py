"""Shared in-process parquet cache for prediction data.

predictions.py and strategy_backtest_svc both need line_1d and band_1d.
Without this store they each call pd.read_parquet() independently, keeping
two copies of the same 60-80 MB DataFrames alive simultaneously.

This module loads each prediction parquet exactly once and lets all callers
share the same object reference.  Thread-safe via per-store Lock.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import pandas as pd

logger = logging.getLogger("lens.parquet_store")

_BASE = Path(__file__).resolve().parents[2] / "data" / "v1"

# Only prediction parquets live here.
# market_prices / market_indicators stay in local_market_svc (different date
# format + different computed columns; sharing would add a third copy, not save one).
_FILE_MAP: dict[str, str] = {
    "line_1d": "predictions_line_1d.parquet",
    "band_1d": "predictions_band_1d.parquet",
    "band_1w": "predictions_band_1w.parquet",
}

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


def require(name: str) -> pd.DataFrame:
    """Like get_raw but raises FileNotFoundError when the file is missing."""
    df = get_raw(name)
    if df is None:
        raise FileNotFoundError(
            f"Required parquet not found: {name!r} "
            f"(expected at {_BASE / _FILE_MAP[name]})"
        )
    return df


def _load(name: str) -> pd.DataFrame | None:
    path = _BASE / _FILE_MAP[name]
    if not path.exists():
        logger.warning("parquet missing: %s", path)
        return None
    mb_disk = path.stat().st_size / 1024 / 1024
    df = pd.read_parquet(path)
    df = _compress_strings(df)
    mb_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info("loaded parquet %s (%.1f MB disk → %.1f MB memory)", name, mb_disk, mb_mem)
    return df


def _compress_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to ordered categorical to save memory.

    Prediction parquets have columns like model_id / source_cp that store the
    same string in every row (e.g. 1 unique value × 597 k rows ≈ 44 MB as
    object, < 1 MB as ordered category).

    CP214 — `asof_date` / `forecast_date` 는 **categorical 에서 제외**한다.
    이유:
      - Categorical[str] vs str 비교 (`sub["asof_date"] >= cutoff`) 가 TypeError 발생
        (predictions.py 라우터에서 `Invalid comparison between dtype=category and str`).
      - `pd.to_datetime(Categorical[str])` 가 dtype 을 datetime64 가 아닌 **Categorical[datetime]**
        으로 유지 → strategy_backtest_svc 의 merge 가 `datetime64[ns] vs category` 충돌로 실패.
    날짜 컬럼은 unique 수가 700~2000 정도라 categorical 효과가 크지 않고, object 로 유지해도
    메모리 절감 효과의 대부분(ticker / model_id / source_cp)은 유지된다.
    """
    cat_dtype = pd.CategoricalDtype(ordered=True)
    skip = {"asof_date", "forecast_date"}
    for col in df.select_dtypes(include="object").columns:
        if col in skip:
            continue
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
