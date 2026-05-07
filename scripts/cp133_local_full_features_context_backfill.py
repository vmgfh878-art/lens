from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
CONTEXT_DIR = SNAPSHOT_DIR / "context"
BACKUP_DIR = SNAPSHOT_DIR / "backups"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp133_local_full_features_context_backfill"

PRICE_1D_PATH = SNAPSHOT_DIR / "price_data_yfinance.parquet"
STOCK_INFO_PATH = SNAPSHOT_DIR / "stock_info.parquet"
INDICATOR_1D_PATH = SNAPSHOT_DIR / "indicators_yfinance_1D.parquet"
INDICATOR_1W_PATH = SNAPSHOT_DIR / "indicators_yfinance_1W.parquet"
CP95_METRICS_PATH = DOCS_DIR / "cp95_yfinance_100ticker_long_history_validation_metrics.json"

CONTEXT_VERSION = "cp133_local_context_v1"
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_month_end",
    "is_quarter_end",
    "is_opex_friday",
]
MACRO_COLUMNS = ["us10y", "yield_spread", "vix_close", "credit_spread_hy"]
BREADTH_COLUMNS = ["nh_nl_index", "ma200_pct"]
FUNDAMENTAL_COLUMNS = ["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]
FLAG_COLUMNS = ["has_macro", "has_breadth", "has_fundamentals"]
REGIME_COLUMNS = ["regime_calm", "regime_neutral", "regime_stress"]
PVV_COLUMNS = [
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "vol_change",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "rsi",
    "macd_ratio",
    "bb_position",
]
FRED_SERIES = {
    "DGS10": "us10y",
    "DGS2": "us2y",
    "VIXCLS": "vix_close",
    "BAMLH0A0HYM2": "credit_spread_hy",
}


os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(SNAPSHOT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features  # noqa: E402
from backend.collector.sources.edgar import fetch_edgar_fundamentals  # noqa: E402
from backend.collector.sources.sector_returns import build_sector_returns  # noqa: E402


MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *CALENDAR_FEATURE_COLUMNS]
MODEL_N_FEATURES = len(MODEL_FEATURE_COLUMNS)
INDICATOR_CHECKSUM_COLUMNS = sorted({*FEATURE_COLUMNS, "atr_ratio", "volume"})


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_checksum(frame: pd.DataFrame, columns: list[str]) -> str | None:
    if frame.empty:
        return None
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    sort_columns = [column for column in ["ticker", "timeframe", "date", "source", "sector"] if column in frame.columns]
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if sort_columns:
        working = working.sort_values(sort_columns)
    payload = working[existing].to_dict(orient="records")
    return hashlib.sha256(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def indicator_value_checksum(frame: pd.DataFrame) -> str | None:
    columns = [column for column in INDICATOR_CHECKSUM_COLUMNS if column in frame.columns]
    return frame_checksum(frame, ["ticker", "date", *columns])


def load_cp95_tickers(limit: int | None) -> list[str]:
    if CP95_METRICS_PATH.exists():
        payload = json.loads(CP95_METRICS_PATH.read_text(encoding="utf-8"))
        tickers = [str(ticker).upper() for ticker in payload.get("final_tickers", []) if str(ticker).strip()]
    else:
        stock_info = pd.read_parquet(STOCK_INFO_PATH)
        tickers = sorted(stock_info["ticker"].astype(str).str.upper().unique().tolist())
    if limit is not None:
        return tickers[:limit]
    return tickers


def load_price_frame(tickers: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(PRICE_1D_PATH)
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["ticker"].isin(tickers)].dropna(subset=["date"]).sort_values(["ticker", "date"])
    return frame.reset_index(drop=True)


def load_stock_info(tickers: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(STOCK_INFO_PATH)
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame[frame["ticker"].isin(tickers)].drop_duplicates("ticker", keep="last").reset_index(drop=True)


def fetch_fred_macro(start_date: str, end_date: str | None, session: requests.Session) -> tuple[pd.DataFrame, dict[str, Any]]:
    series_ids = ",".join(FRED_SERIES.keys())
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_ids}&cosd={start_date}"
    if end_date:
        url = f"{url}&coed={end_date}"
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        content = response.content
        if content.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    raise RuntimeError("FRED zip 응답 안에서 CSV를 찾지 못했습니다.")
                frames = []
                for csv_name in csv_names:
                    with archive.open(csv_name) as handle:
                        current = pd.read_csv(handle)
                    if "observation_date" in current.columns:
                        frames.append(current)
                if not frames:
                    raise RuntimeError("FRED zip 응답 안에서 observation_date가 있는 CSV를 찾지 못했습니다.")
                raw = frames[0]
                for current in frames[1:]:
                    raw = raw.merge(current, on="observation_date", how="outer")
        else:
            raw = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS]), {
            "status": "FAIL",
            "source": "fred",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }

    if raw.empty or "observation_date" not in raw.columns:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS]), {"status": "FAIL", "source": "fred", "error": "empty_or_missing_date"}

    raw = raw.rename(columns={"observation_date": "date", **FRED_SERIES})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in ["us10y", "us2y", "vix_close", "credit_spread_hy"]:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column].replace(".", np.nan), errors="coerce")
    if "us10y" in raw.columns and "us2y" in raw.columns:
        raw["yield_spread"] = raw["us10y"] - raw["us2y"]
    frame = raw[["date", *[column for column in MACRO_COLUMNS if column in raw.columns]]].copy()
    for column in MACRO_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["source"] = "fred"
    frame["provider"] = "fred"
    frame["asof_date"] = frame["date"]
    frame["release_policy"] = "fred_observation_date_close_available"
    frame["context_version"] = CONTEXT_VERSION
    coverage = {column: float(frame[column].notna().mean()) if len(frame) else 0.0 for column in MACRO_COLUMNS}
    return frame, {
        "status": "PASS" if not frame.empty and any(value > 0 for value in coverage.values()) else "WARN",
        "source": "fred",
        "rows": int(len(frame)),
        "date_min": None if frame.empty else frame["date"].min().date().isoformat(),
        "date_max": None if frame.empty else frame["date"].max().date().isoformat(),
        "non_null_rate": coverage,
    }


def build_market_breadth_context(price_frame: pd.DataFrame, min_ticker_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if price_frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"]), {"status": "FAIL", "error": "empty_price"}
    working = price_frame[["ticker", "date", "close", "adjusted_close"]].copy()
    working["close"] = pd.to_numeric(working["adjusted_close"], errors="coerce").fillna(pd.to_numeric(working["close"], errors="coerce"))
    working = working.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])
    working["high_52w"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=252, min_periods=200).max())
    working["low_52w"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=252, min_periods=200).min())
    working["ma_200"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=200, min_periods=150).mean())
    working = working.dropna(subset=["high_52w", "low_52w", "ma_200"])
    if working.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"]), {"status": "FAIL", "error": "empty_after_roll"}
    working["is_nh"] = working["close"] >= working["high_52w"]
    working["is_nl"] = working["close"] <= working["low_52w"]
    working["is_above_ma200"] = working["close"] > working["ma_200"]
    stats = (
        working.groupby("date")
        .agg(
            total_count=("ticker", "count"),
            nh_count=("is_nh", "sum"),
            nl_count=("is_nl", "sum"),
            above_ma200_count=("is_above_ma200", "sum"),
        )
        .reset_index()
    )
    stats = stats[stats["total_count"] >= min_ticker_count].copy()
    stats["nh_nl_index"] = stats["nh_count"] - stats["nl_count"]
    stats["ma200_pct"] = (stats["above_ma200_count"] / stats["total_count"]) * 100.0
    stats["source"] = "local_yfinance_universe"
    stats["provider"] = "yfinance"
    stats["asof_date"] = stats["date"]
    stats["universe_ticker_count"] = int(price_frame["ticker"].nunique())
    stats["min_ticker_count"] = min_ticker_count
    stats["context_version"] = CONTEXT_VERSION
    result = stats[[
        "date",
        "nh_nl_index",
        "ma200_pct",
        "total_count",
        "source",
        "provider",
        "asof_date",
        "universe_ticker_count",
        "min_ticker_count",
        "context_version",
    ]].sort_values("date")
    return result.reset_index(drop=True), {
        "status": "PASS" if not result.empty else "FAIL",
        "rows": int(len(result)),
        "date_min": None if result.empty else result["date"].min().date().isoformat(),
        "date_max": None if result.empty else result["date"].max().date().isoformat(),
        "min_ticker_count": min_ticker_count,
        "source_policy": "100 ticker local yfinance universe breadth, not full-market breadth",
        "non_zero_rate": {
            "nh_nl_index": float((result["nh_nl_index"] != 0).mean()) if len(result) else 0.0,
            "ma200_pct": float((result["ma200_pct"] != 0).mean()) if len(result) else 0.0,
        },
    }


def build_sector_context(price_frame: pd.DataFrame, stock_info_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_price = price_frame[["ticker", "date", "close", "adjusted_close"]].copy()
    working_price["close"] = pd.to_numeric(working_price["adjusted_close"], errors="coerce").fillna(
        pd.to_numeric(working_price["close"], errors="coerce")
    )
    sector = build_sector_returns(working_price[["ticker", "date", "close"]], stock_info_frame)
    if sector.empty:
        return sector, {"status": "WARN", "rows": 0, "note": "sector_returns_empty"}
    sector["date"] = pd.to_datetime(sector["date"], errors="coerce")
    sector["source"] = "local_yfinance_universe"
    sector["provider"] = "yfinance"
    sector["asof_date"] = sector["date"]
    sector["context_version"] = CONTEXT_VERSION
    return sector.sort_values(["date", "sector"]).reset_index(drop=True), {
        "status": "PASS",
        "rows": int(len(sector)),
        "sector_count": int(sector["sector"].nunique()),
        "date_min": sector["date"].min().date().isoformat(),
        "date_max": sector["date"].max().date().isoformat(),
        "note": "현재 MODEL_FEATURE_COLUMNS에는 sector_returns가 포함되지 않음",
    }


def fetch_edgar_fundamentals_context(tickers: list[str], sleep_seconds: float, limit: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = tickers[:limit] if limit is not None else tickers
    rows: list[dict[str, Any]] = []
    ticker_metrics: dict[str, Any] = {}
    for index, ticker in enumerate(targets, start=1):
        started = time.perf_counter()
        try:
            ticker_rows = fetch_edgar_fundamentals(ticker)
        except Exception as exc:
            ticker_rows = []
            ticker_metrics[ticker] = {"status": "FAIL", "rows": 0, "error_type": type(exc).__name__, "error": str(exc)[:200]}
        else:
            ticker_metrics[ticker] = {
                "status": "PASS" if ticker_rows else "EMPTY",
                "rows": len(ticker_rows),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        for row in ticker_rows:
            enriched = dict(row)
            enriched["source"] = "sec_edgar"
            enriched["provider"] = "sec_edgar"
            enriched["asof_date"] = enriched.get("filing_date")
            enriched["asof_policy"] = "actual_sec_filing_date"
            enriched["context_version"] = CONTEXT_VERSION
            rows.append(enriched)
        if sleep_seconds > 0 and index < len(targets):
            time.sleep(sleep_seconds)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in ["date", "filing_date", "asof_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame = frame.dropna(subset=["ticker", "date", "filing_date"]).sort_values(["ticker", "date", "filing_date"])
        frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    success_tickers = [ticker for ticker, item in ticker_metrics.items() if item.get("status") == "PASS"]
    return frame, {
        "status": "PASS" if success_tickers else "WARN",
        "requested_tickers": len(targets),
        "success_ticker_count": len(success_tickers),
        "empty_ticker_count": sum(1 for item in ticker_metrics.values() if item.get("status") == "EMPTY"),
        "row_count": int(len(frame)),
        "date_min": None if frame.empty else frame["date"].min().date().isoformat(),
        "date_max": None if frame.empty else frame["date"].max().date().isoformat(),
        "ticker_metrics_sample": dict(list(ticker_metrics.items())[:20]),
    }


def normalize_indicator_frame(frame: pd.DataFrame, timeframe: str, context_hash: str, updated_at: str) -> pd.DataFrame:
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["timeframe"] = timeframe
    result["source"] = "yfinance"
    result["provider"] = "yfinance"
    result["context_version"] = CONTEXT_VERSION
    result["context_hash"] = context_hash
    result["updated_at"] = updated_at
    return result.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    ).reset_index(drop=True)


def feature_stats(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns and not frame.empty else 0,
        "date_min": None if frame.empty else str(pd.to_datetime(frame["date"]).min().date()),
        "date_max": None if frame.empty else str(pd.to_datetime(frame["date"]).max().date()),
    }
    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce") if feature_columns else pd.DataFrame()
    values = numeric.to_numpy(dtype=float) if not numeric.empty else np.array([])
    stats["feature_non_finite_count"] = int((~np.isfinite(values)).sum()) if values.size else 0
    stats["duplicate_ticker_timeframe_date_source"] = int(frame.duplicated(["ticker", "timeframe", "date", "source"]).sum())
    stats["flag_true_rate"] = {
        column: float(frame[column].astype(bool).mean()) if column in frame.columns and len(frame) else None
        for column in FLAG_COLUMNS
    }
    context_columns = [*MACRO_COLUMNS, *BREADTH_COLUMNS, *FUNDAMENTAL_COLUMNS, *REGIME_COLUMNS]
    stats["zero_rate"] = {
        column: float((pd.to_numeric(frame[column], errors="coerce").fillna(0) == 0).mean())
        if column in frame.columns and len(frame)
        else None
        for column in context_columns
    }
    stats["non_zero_rate"] = {
        column: (None if stats["zero_rate"][column] is None else 1.0 - float(stats["zero_rate"][column]))
        for column in context_columns
    }
    stats["indicator_value_checksum"] = indicator_value_checksum(frame)
    return stats


def backup_indicator_files(timestamp: str) -> dict[str, Any]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for timeframe, path in {"1D": INDICATOR_1D_PATH, "1W": INDICATOR_1W_PATH}.items():
        if path.exists():
            target = BACKUP_DIR / f"{path.stem}_before_cp133_{timestamp}.parquet"
            shutil.copy2(path, target)
            result[timeframe] = {
                "source": str(path.relative_to(ROOT_DIR)),
                "backup": str(target.relative_to(ROOT_DIR)),
                "source_sha256": file_sha256(path),
                "backup_sha256": file_sha256(target),
                "bytes": int(target.stat().st_size),
            }
    return result


def write_context_inventory_csv(path: Path, before_stats: dict[str, Any], after_stats: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    groups = {
        "macro": MACRO_COLUMNS,
        "market_breadth": BREADTH_COLUMNS,
        "fundamentals": FUNDAMENTAL_COLUMNS,
        "regime": REGIME_COLUMNS,
        "missing_flag": FLAG_COLUMNS,
        "price_volatility_volume": PVV_COLUMNS,
        "calendar": CALENDAR_FEATURE_COLUMNS,
    }
    for group, columns in groups.items():
        for column in columns:
            before_1d = before_stats.get("1D", {})
            before_1w = before_stats.get("1W", {})
            after_1d = after_stats.get("1D", {})
            after_1w = after_stats.get("1W", {})
            rows.append(
                {
                    "column": column,
                    "feature_group": group,
                    "in_model_features": column in MODEL_FEATURE_COLUMNS,
                    "in_pvv": column in PVV_COLUMNS,
                    "before_1d_zero_rate": before_1d.get("zero_rate", {}).get(column),
                    "after_1d_zero_rate": after_1d.get("zero_rate", {}).get(column),
                    "before_1w_zero_rate": before_1w.get("zero_rate", {}).get(column),
                    "after_1w_zero_rate": after_1w.get("zero_rate", {}).get(column),
                    "before_1d_flag_true_rate": before_1d.get("flag_true_rate", {}).get(column),
                    "after_1d_flag_true_rate": after_1d.get("flag_true_rate", {}).get(column),
                    "before_1w_flag_true_rate": before_1w.get("flag_true_rate", {}).get(column),
                    "after_1w_flag_true_rate": after_1w.get("flag_true_rate", {}).get(column),
                    "source_contract": source_contract_for(group),
                    "lookahead_policy": lookahead_policy_for(group),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def source_contract_for(group: str) -> str:
    if group == "macro":
        return "local parquet context, provider=fred, asof_date=observation date"
    if group == "market_breadth":
        return "local yfinance price-derived universe breadth, provider=yfinance"
    if group == "fundamentals":
        return "local parquet context, provider=sec_edgar, filing_date/asof_date actual SEC filing date"
    if group == "regime":
        return "derived from vix_close macro context"
    if group == "missing_flag":
        return "feature_svc context non-null flag"
    if group == "calendar":
        return "deterministic calendar feature"
    return "local yfinance price/indicator feature"


def lookahead_policy_for(group: str) -> str:
    if group == "fundamentals":
        return "merge_asof backward on filing_date; filing_date must be <= feature date"
    if group == "macro":
        return "daily observation date only; 1W resample uses completed W-FRI last observation"
    if group == "market_breadth":
        return "derived only from local prices up to same completed date"
    if group == "regime":
        return "derived from same-date vix_close after context merge"
    return "no additional lookahead source"


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    decision = metrics["final_decision"]["status"]
    macro = metrics["context_sources"]["macro"]
    breadth = metrics["context_sources"]["market_breadth"]
    fundamentals = metrics["context_sources"]["fundamentals"]
    after_1d = metrics["after_indicator_stats"]["1D"]
    after_1w = metrics["after_indicator_stats"]["1W"]
    report = f"""# CP133-DG local full_features context backfill 보고서

## 1. Executive Summary

판정은 **{decision}**이다.

CP132에서 0-fill이었던 full_features context 중 macro와 market breadth는 local parquet 기준으로 실제 값이 들어갔다. sector_returns도 local yfinance 가격과 stock_info에서 생성했지만, 현재 `MODEL_FEATURE_COLUMNS` 36개에는 포함되지 않는다. fundamentals는 SEC EDGAR 기반으로 시도했으며, 성공 범위와 true rate는 metrics에 기록했다.

중요한 제약은 그대로 남는다. market breadth는 전체 미국 시장 breadth가 아니라 현재 local 100티커 universe breadth다. macro는 FRED observation date 기준이다. fundamentals는 `filing_date` 기준 backward merge만 허용한다.

## 2. 생성/갱신한 local parquet

| 파일 | 역할 |
|---|---|
| `data/parquet/context/macroeconomic_indicators.parquet` | FRED 기반 macro context |
| `data/parquet/context/market_breadth.parquet` | local yfinance 100티커 breadth |
| `data/parquet/context/company_fundamentals.parquet` | SEC EDGAR filing_date 기반 fundamentals |
| `data/parquet/context/sector_returns.parquet` | local yfinance sector return, 현재 모델 feature 미포함 |
| `data/parquet/indicators_yfinance_1D.parquet` | context 반영 후 재생성 |
| `data/parquet/indicators_yfinance_1W.parquet` | context 반영 후 재생성 |

기존 1D/1W indicator parquet는 `data/parquet/backups` 아래에 백업했다.

## 3. context source 결과

| 계열 | 상태 | rows | date range | 비고 |
|---|---|---:|---|---|
| macro | {macro.get('status')} | {macro.get('rows')} | {macro.get('date_min')} ~ {macro.get('date_max')} | provider=fred |
| market_breadth | {breadth.get('status')} | {breadth.get('rows')} | {breadth.get('date_min')} ~ {breadth.get('date_max')} | local 100티커 breadth |
| fundamentals | {fundamentals.get('status')} | {fundamentals.get('row_count')} | {fundamentals.get('date_min')} ~ {fundamentals.get('date_max')} | provider=sec_edgar |

## 4. 1D/1W feature sanity

| 항목 | 1D | 1W |
|---|---:|---:|
| rows | {after_1d.get('rows')} | {after_1w.get('rows')} |
| tickers | {after_1d.get('ticker_count')} | {after_1w.get('ticker_count')} |
| date range | {after_1d.get('date_min')} ~ {after_1d.get('date_max')} | {after_1w.get('date_min')} ~ {after_1w.get('date_max')} |
| feature non-finite count | {after_1d.get('feature_non_finite_count')} | {after_1w.get('feature_non_finite_count')} |
| duplicate key count | {after_1d.get('duplicate_ticker_timeframe_date_source')} | {after_1w.get('duplicate_ticker_timeframe_date_source')} |
| has_macro true rate | {after_1d.get('flag_true_rate', {}).get('has_macro')} | {after_1w.get('flag_true_rate', {}).get('has_macro')} |
| has_breadth true rate | {after_1d.get('flag_true_rate', {}).get('has_breadth')} | {after_1w.get('flag_true_rate', {}).get('has_breadth')} |
| has_fundamentals true rate | {after_1d.get('flag_true_rate', {}).get('has_fundamentals')} | {after_1w.get('flag_true_rate', {}).get('has_fundamentals')} |

## 5. lookahead 방지 계약

- macro: FRED observation date만 사용한다. 1W는 완료된 W-FRI bucket에서 `last()`를 사용하므로, 주간 asof가 금요일 종가 이후라는 전제에서만 안전하다.
- market_breadth: 같은 날짜까지 존재하는 local yfinance adjusted close만 사용한다.
- fundamentals: `filing_date` 기준 `merge_asof(direction="backward")`만 허용한다. report period `date`가 아니라 filing date가 feature asof gate다.
- sector_returns: 현재 모델 feature가 아니며, 향후 승격 전 source/provider/asof 계약을 별도 모델 feature 계약에 추가해야 한다.

## 6. cache/hash 영향

context 반영 전후 indicator value checksum을 기록했다. CP96 이후 feature fingerprint는 indicator value checksum을 포함하므로, indicator 값이 바뀐 상태에서 기존 feature cache를 그대로 쓰면 안 된다.

- before 1D checksum: `{metrics['before_indicator_stats']['1D'].get('indicator_value_checksum')}`
- after 1D checksum: `{after_1d.get('indicator_value_checksum')}`
- before 1W checksum: `{metrics['before_indicator_stats']['1W'].get('indicator_value_checksum')}`
- after 1W checksum: `{after_1w.get('indicator_value_checksum')}`

## 7. 최종 판단

{metrics['final_decision']['summary']}

## 8. 금지 작업 미발생 확인

Supabase 대량 read/write, DB write/delete/update, 모델 학습, inference 실행, save-run, 프론트 수정, EODHD 호출, fake data 생성은 수행하지 않았다.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP133 local full_features context backfill")
    parser.add_argument("--limit-tickers", type=int, default=100)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-breadth-tickers", type=int, default=50)
    parser.add_argument("--fundamental-limit", type=int, default=100)
    parser.add_argument("--fundamental-sleep-seconds", type=float, default=0.12)
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp133_local_full_features_context_backfill_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp133_local_full_features_context_backfill_report.md"))
    parser.add_argument("--inventory-path", default=str(DOCS_DIR / "cp133_full_features_context_inventory.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_at = utc_now_iso()
    tickers = load_cp95_tickers(args.limit_tickers)
    price_frame = load_price_frame(tickers)
    stock_info = load_stock_info(tickers)
    before_1d = pd.read_parquet(INDICATOR_1D_PATH)
    before_1w = pd.read_parquet(INDICATOR_1W_PATH)
    before_stats = {"1D": feature_stats(before_1d, "1D"), "1W": feature_stats(before_1w, "1W")}
    backups = backup_indicator_files(timestamp)

    session = requests.Session()
    session.trust_env = False
    macro_frame, macro_metrics = fetch_fred_macro(args.start_date, args.end_date, session)
    breadth_frame, breadth_metrics = build_market_breadth_context(price_frame, args.min_breadth_tickers)
    sector_frame, sector_metrics = build_sector_context(price_frame, stock_info)
    if args.skip_fundamentals:
        fundamentals_frame = pd.DataFrame()
        fundamentals_metrics = {"status": "SKIPPED", "row_count": 0, "requested_tickers": 0}
    else:
        fundamentals_frame, fundamentals_metrics = fetch_edgar_fundamentals_context(
            tickers,
            sleep_seconds=args.fundamental_sleep_seconds,
            limit=args.fundamental_limit,
        )

    context_hash_payload = {
        "context_version": CONTEXT_VERSION,
        "macro": frame_checksum(macro_frame, ["date", *MACRO_COLUMNS]),
        "breadth": frame_checksum(breadth_frame, ["date", *BREADTH_COLUMNS, "total_count"]),
        "fundamentals": frame_checksum(fundamentals_frame, ["ticker", "date", "filing_date", *FUNDAMENTAL_COLUMNS]),
        "sector": frame_checksum(sector_frame, ["date", "sector", "return", "close"]),
    }
    context_hash = hashlib.sha256(json.dumps(_json_safe(context_hash_payload), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    context_paths = {
        "macro": CONTEXT_DIR / "macroeconomic_indicators.parquet",
        "market_breadth": CONTEXT_DIR / "market_breadth.parquet",
        "company_fundamentals": CONTEXT_DIR / "company_fundamentals.parquet",
        "sector_returns": CONTEXT_DIR / "sector_returns.parquet",
    }
    macro_frame.to_parquet(context_paths["macro"], index=False)
    breadth_frame.to_parquet(context_paths["market_breadth"], index=False)
    fundamentals_frame.to_parquet(context_paths["company_fundamentals"], index=False)
    sector_frame.to_parquet(context_paths["sector_returns"], index=False)

    indicator_1d = normalize_indicator_frame(
        build_features(price_df=price_frame, macro_df=macro_frame, breadth_df=breadth_frame, fundamentals_df=fundamentals_frame, timeframe="1D"),
        "1D",
        context_hash,
        updated_at,
    )
    indicator_1w = normalize_indicator_frame(
        build_features(price_df=price_frame, macro_df=macro_frame, breadth_df=breadth_frame, fundamentals_df=fundamentals_frame, timeframe="1W"),
        "1W",
        context_hash,
        updated_at,
    )
    indicator_1d.to_parquet(INDICATOR_1D_PATH, index=False)
    indicator_1w.to_parquet(INDICATOR_1W_PATH, index=False)

    after_stats = {"1D": feature_stats(indicator_1d, "1D"), "1W": feature_stats(indicator_1w, "1W")}
    write_context_inventory_csv(Path(args.inventory_path), before_stats, after_stats)

    context_file_metrics = {
        name: {
            "path": str(path.relative_to(ROOT_DIR)),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for name, path in context_paths.items()
    }
    indicator_file_metrics = {
        "1D": {
            "path": str(INDICATOR_1D_PATH.relative_to(ROOT_DIR)),
            "bytes": int(INDICATOR_1D_PATH.stat().st_size),
            "sha256": file_sha256(INDICATOR_1D_PATH),
        },
        "1W": {
            "path": str(INDICATOR_1W_PATH.relative_to(ROOT_DIR)),
            "bytes": int(INDICATOR_1W_PATH.stat().st_size),
            "sha256": file_sha256(INDICATOR_1W_PATH),
        },
    }

    failures = []
    warnings = []
    for timeframe, stats in after_stats.items():
        if stats["feature_non_finite_count"] != 0:
            failures.append(f"{timeframe} feature_non_finite_count={stats['feature_non_finite_count']}")
        if stats["duplicate_ticker_timeframe_date_source"] != 0:
            failures.append(f"{timeframe} duplicate={stats['duplicate_ticker_timeframe_date_source']}")
        if not stats["flag_true_rate"].get("has_macro"):
            warnings.append(f"{timeframe} has_macro true rate is zero")
        if not stats["flag_true_rate"].get("has_breadth"):
            warnings.append(f"{timeframe} has_breadth true rate is zero")
        if not stats["flag_true_rate"].get("has_fundamentals"):
            warnings.append(f"{timeframe} has_fundamentals true rate is zero")
        if (stats["non_zero_rate"].get("credit_spread_hy") or 0.0) < 0.8:
            warnings.append(f"{timeframe} credit_spread_hy coverage is partial")
        if (stats["flag_true_rate"].get("has_fundamentals") or 0.0) < 0.2:
            warnings.append(f"{timeframe} fundamentals coverage is partial due to filing_date gate")
    if macro_metrics.get("status") != "PASS":
        warnings.append("macro source incomplete")
    if breadth_metrics.get("status") != "PASS":
        warnings.append("market breadth source incomplete")
    else:
        warnings.append("market breadth is 100 ticker local universe breadth, not full-market breadth")
    if fundamentals_metrics.get("status") != "PASS":
        warnings.append("fundamentals source incomplete or partial")

    if failures:
        status = "FAIL"
        summary = "local full_features context backfill 후 feature sanity 실패가 있어 full_features 재실험을 막아야 한다."
    elif warnings:
        status = "WARN"
        summary = "macro/breadth/fundamentals context가 local parquet 기준으로 채워졌고 sanity도 통과했지만 일부 context coverage가 부분적이므로 full_features 재실험은 가능하되 해석 제약이 필요하다."
    else:
        status = "PASS"
        summary = "macro/breadth/fundamentals context가 local parquet 기준으로 채워졌고 1D/1W feature sanity가 통과했다."

    metrics: dict[str, Any] = {
        "cp": "CP133-DG",
        "created_at": updated_at,
        "context_version": CONTEXT_VERSION,
        "context_hash": context_hash,
        "context_hash_payload": context_hash_payload,
        "model_n_features": MODEL_N_FEATURES,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "tickers": tickers,
        "ticker_count": len(tickers),
        "source_scope": {
            "price": "data/parquet/price_data_yfinance.parquet",
            "stock_info": "data/parquet/stock_info.parquet",
            "supabase_bulk_read": False,
            "supabase_write": False,
            "eodhd_call": False,
        },
        "context_sources": {
            "macro": macro_metrics,
            "market_breadth": breadth_metrics,
            "fundamentals": fundamentals_metrics,
            "sector_returns": sector_metrics,
        },
        "context_files": context_file_metrics,
        "indicator_files": indicator_file_metrics,
        "backups": backups,
        "before_indicator_stats": before_stats,
        "after_indicator_stats": after_stats,
        "cache_hash_guard": {
            "indicator_value_checksum_changed_1D": before_stats["1D"].get("indicator_value_checksum") != after_stats["1D"].get("indicator_value_checksum"),
            "indicator_value_checksum_changed_1W": before_stats["1W"].get("indicator_value_checksum") != after_stats["1W"].get("indicator_value_checksum"),
            "note": "CP96 fingerprint는 indicator_value_checksum을 포함하므로 checksum 변경 시 기존 feature cache 재사용을 막아야 한다.",
        },
        "forbidden_actions": {
            "supabase_price_indicator_context_bulk_write": False,
            "supabase_bulk_read": False,
            "model_training": False,
            "inference_execution": False,
            "save_run": False,
            "frontend_change": False,
            "eodhd_call": False,
            "fake_data": False,
        },
        "final_decision": {
            "status": status,
            "failures": failures,
            "warnings": warnings,
            "summary": summary,
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_json(LOG_DIR / "cp133_local_full_features_context_backfill_metrics.json", metrics)
    write_report(Path(args.report_path), metrics)
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
