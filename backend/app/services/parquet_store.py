"""Shared in-process parquet cache for prediction data.

predictions.py and strategy_backtest_svc both need line_1d and band_1d.
Without this store they each call pd.read_parquet() independently, keeping
two copies of the same 60-80 MB DataFrames alive simultaneously.

This module loads each prediction parquet exactly once and lets all callers
share the same object reference.  Thread-safe via per-store Lock.
"""

from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import structlog
from app.schemas import frames as _frames
from pandas.api import types as pdt

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
    # CP246 — compress 먼저(object→category/datetime), 그 다음 경량 계약 검사.
    # 순서를 바꾼 이유: pandera coerce-validate(구버전)가 597k행을 object 로
    # 복제해 cold load peak +236MB → 512MB OOM. assert_contract 는 할당 0.
    df = _compress_strings(df)
    df = _frames.assert_contract(name, df)
    # read 디컴프레션이 남긴 transient 버퍼를 OS 로 반환(allocator 단편화 완화).
    gc.collect()
    try:
        import pyarrow as _pa

        _pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 — release 는 best-effort, 실패해도 무해.
        pass
    mb_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info("loaded parquet %s (%.1f MB disk → %.1f MB memory)", name, mb_disk, mb_mem)
    return df


def _compress_strings(df: pd.DataFrame) -> pd.DataFrame:
    """object 컬럼은 ordered categorical 로, 날짜 컬럼은 datetime64 로 압축해 메모리 절감.

    model_id / source_cp 처럼 한 값이 597 k 행 반복되는 컬럼은 category 로 (44MB → <1MB).

    CP245 — `asof_date` / `forecast_date` 는 **datetime64 로 변환**한다. 라우터
    (predictions.py)의 `asof_date >= cutoff` 비교를 `pd.Timestamp` 로 올려 정공법으로
    해결했고, strategy_scan 은 `pd.to_datetime` 로 그대로 받으므로 영향 없다. 응답
    직렬화(_jsonable)는 Timestamp 를 date 문자열로 내보내 출력 JSON 은 기존과 동일하다.

    CP246 — **재인코딩 parquet 대응**. compact 파일은 문자열을 category(dictionary),
    날짜를 datetime64 로 이미 저장하므로 read 직후 object 가 없다. 이 함수는 이제
    object 든 category 든 datetime 이든 **동일 결과**로 수렴하는 멱등 정규화다:
      · 날짜 = object→to_datetime, category→categories 매핑으로 datetime(object 미경유),
        datetime64→그대로.
      · 그 외 문자열 = object→ordered category, category→ordered 보장.
    핵심은 category 날짜를 `.astype(str)` 로 풀지 않는 것 — 그러면 597k object 가
    되살아나 스파이크가 재발한다. categories(수백 개)만 변환해 codes 로 take 한다.
    """
    cat_dtype = pd.CategoricalDtype(ordered=True)
    date_cols = {"asof_date", "forecast_date"}
    for col in df.columns:
        dt = df[col].dtype
        is_cat = isinstance(dt, pd.CategoricalDtype)
        is_obj = pdt.is_object_dtype(dt)
        if col in date_cols:
            if pdt.is_datetime64_any_dtype(dt):
                continue
            if is_cat:
                # categories(수백 개)만 datetime 으로 → codes 로 재구성(object 미경유).
                cats = pd.to_datetime(df[col].cat.categories, errors="coerce").to_numpy()
                codes = df[col].cat.codes.to_numpy()
                values = cats[codes]
                values[codes == -1] = np.datetime64("NaT")
                df[col] = values
            elif is_obj:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        elif is_obj:
            df[col] = df[col].astype(cat_dtype)
        elif is_cat and not dt.ordered:
            df[col] = df[col].cat.as_ordered()
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
