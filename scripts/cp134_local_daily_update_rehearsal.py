from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
CONTEXT_DIR = SNAPSHOT_DIR / "context"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp134_local_daily_update_pipeline"

PRICE_PATH = SNAPSHOT_DIR / "price_data_yfinance.parquet"
PRICE_1W_PATH = SNAPSHOT_DIR / "price_data_yfinance_1W.parquet"
INDICATOR_1D_PATH = SNAPSHOT_DIR / "indicators_yfinance_1D.parquet"
INDICATOR_1W_PATH = SNAPSHOT_DIR / "indicators_yfinance_1W.parquet"
STOCK_INFO_PATH = SNAPSHOT_DIR / "stock_info.parquet"
PRODUCT_HISTORY_1D_PATH = SNAPSHOT_DIR / "product_prediction_history_1D.parquet"
PRODUCT_HISTORY_1D_MANIFEST_PATH = SNAPSHOT_DIR / "product_prediction_history_1D.manifest.json"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
CONTEXT_VERSION = "cp133_local_context_v1"

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)
os.environ.setdefault("WANDB_MODE", "disabled")

# 로컬 데스크톱의 차단용 프록시가 yfinance 호출에 섞이면 fetch 실패를 신규 row 없음으로 오판할 수 있다.
for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    _proxy_value = os.environ.get(_proxy_key, "")
    if "127.0.0.1:9" in _proxy_value:
        os.environ.pop(_proxy_key, None)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import build_features, latest_complete_period_end  # noqa: E402
from backend.collector.sources.market_data_providers import fetch_market_data, provider_adjustment_policy  # noqa: E402
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402
from backend.collector.utils.network import sanitize_proxy_env  # noqa: E402
from scripts.cp133_local_full_features_context_backfill import (  # noqa: E402
    build_market_breadth_context,
    build_sector_context,
    fetch_edgar_fundamentals_context,
    fetch_fred_macro,
    indicator_value_checksum,
)

sanitize_proxy_env()


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


def frame_checksum(frame: pd.DataFrame, columns: list[str]) -> str | None:
    if frame.empty:
        return None
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_columns = [column for column in ["ticker", "timeframe", "date", "source", "sector"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns)
    payload = working[existing].to_dict(orient="records")
    return hashlib.sha256(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def snapshot_summary(path: Path, duplicate_keys: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT_DIR)), "exists": False}
    frame = load_parquet(path)
    summary = {
        "path": str(path.relative_to(ROOT_DIR)),
        "exists": True,
        "rows": int(len(frame)),
        "bytes": int(path.stat().st_size),
        "date_min": None,
        "date_max": None,
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "duplicate_count": int(frame.duplicated(duplicate_keys).sum()) if all(column in frame.columns for column in duplicate_keys) else None,
    }
    if "date" in frame.columns and not frame.empty:
        summary["date_min"] = frame["date"].min().date().isoformat()
        summary["date_max"] = frame["date"].max().date().isoformat()
    if "source" in frame.columns:
        summary["source_values"] = sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist())
    if "provider" in frame.columns:
        summary["provider_values"] = sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist())
    if "context_hash" in frame.columns:
        summary["context_hash_values"] = sorted(frame["context_hash"].dropna().astype(str).unique().tolist())
    return summary


def price_frame_to_records(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce")
    policy = provider_adjustment_policy("yfinance")
    records = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "date": working.index.normalize(),
            "open": pd.to_numeric(working["Open"], errors="coerce"),
            "high": pd.to_numeric(working["High"], errors="coerce"),
            "low": pd.to_numeric(working["Low"], errors="coerce"),
            "close": pd.to_numeric(working["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(working["Adj Close"], errors="coerce"),
            "volume": pd.to_numeric(working["Volume"], errors="coerce").fillna(0).astype("int64"),
            "amount": pd.to_numeric(working.get("Amount", working["Close"] * working["Volume"]), errors="coerce"),
            "source": "yfinance",
            "provider": "yfinance",
            "provider_adjustment_policy": policy,
            "updated_at": utc_now_iso(),
        }
    )
    return records.dropna(subset=["date"]).reset_index(drop=True)


def load_context_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in ["macroeconomic_indicators", "market_breadth", "company_fundamentals", "sector_returns"]:
        path = CONTEXT_DIR / f"{name}.parquet"
        frames[name] = load_parquet(path) if path.exists() else pd.DataFrame()
    return frames


def completed_price_update_dry_run(tickers: list[str], current_date: date, lookback_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    price = load_parquet(PRICE_PATH)
    snapshot_latest = price["date"].max().date()
    start_date = (snapshot_latest - timedelta(days=lookback_days)).isoformat()
    end_date = current_date.isoformat()
    completed_chunks: list[pd.DataFrame] = []
    ticker_metrics: dict[str, Any] = {}
    fallback_used_count = 0
    contract_failures: list[str] = []
    empty_fetch_tickers: list[str] = []

    for ticker in tickers:
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=None,
        )
        fallback_used_count += int(bool(result.fallback_used))
        if result.frame.empty:
            empty_fetch_tickers.append(ticker)
            ticker_metrics[ticker] = {
                "status": "WARN_EMPTY",
                "fallback_used": bool(result.fallback_used),
                "errors": result.errors,
                "fetched_rows": 0,
                "new_rows": 0,
                "completed_rows": 0,
                "partial_excluded_rows": 0,
            }
            continue
        contract = validate_adjusted_ohlc_contract(ticker, result.frame)
        if not contract.passed:
            contract_failures.append(ticker)
        records = price_frame_to_records(ticker, result.frame)
        records["date"] = pd.to_datetime(records["date"], errors="coerce")
        new_rows = records[pd.to_datetime(records["date"]).dt.date > snapshot_latest].copy()
        completed = new_rows[pd.to_datetime(new_rows["date"]).dt.date < current_date].copy()
        partial = new_rows[pd.to_datetime(new_rows["date"]).dt.date >= current_date].copy()
        if contract.passed and not completed.empty:
            completed_chunks.append(completed)
        ticker_metrics[ticker] = {
            "status": "PASS" if contract.passed else "FAIL_CONTRACT",
            "fallback_used": bool(result.fallback_used),
            "provider": result.provider,
            "fetched_rows": int(len(records)),
            "fetched_date_min": records["date"].min().date().isoformat(),
            "fetched_date_max": records["date"].max().date().isoformat(),
            "new_rows": int(len(new_rows)),
            "new_dates": sorted(pd.to_datetime(new_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()),
            "completed_rows": int(len(completed)),
            "completed_dates": sorted(pd.to_datetime(completed["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()),
            "partial_excluded_rows": int(len(partial)),
            "partial_excluded_dates": sorted(pd.to_datetime(partial["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()),
            "adjusted_ohlc_contract_passed": bool(contract.passed),
            "adjusted_ohlc_violations": contract.violations,
        }

    completed_rows = pd.concat(completed_chunks, ignore_index=True) if completed_chunks else pd.DataFrame()
    if not completed_rows.empty:
        completed_rows = completed_rows.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date", "source"], keep="last")
    total_fetched_rows = sum(int(item.get("fetched_rows", 0)) for item in ticker_metrics.values())
    total_new_rows = sum(int(item.get("new_rows", 0)) for item in ticker_metrics.values())
    total_partial_excluded_rows = sum(int(item.get("partial_excluded_rows", 0)) for item in ticker_metrics.values())
    if fallback_used_count or contract_failures or len(empty_fetch_tickers) == len(tickers):
        status = "FAIL"
        daily_update_state = "fetch_failed"
    elif empty_fetch_tickers:
        status = "WARN_PARTIAL_FETCH_EMPTY"
        daily_update_state = "fetch_failed_partial"
    elif not completed_rows.empty:
        status = "PASS_NEW_ROWS_AVAILABLE"
        daily_update_state = "append_ready"
    elif total_new_rows > 0 and total_partial_excluded_rows == total_new_rows:
        status = "PASS_PARTIAL_DAY_FILTERED"
        daily_update_state = "partial_day_filtered"
    else:
        status = "PASS_WITH_NO_NEW_DAY"
        daily_update_state = "no_new_rows"
    return completed_rows.reset_index(drop=True), {
        "status": status,
        "daily_update_state": daily_update_state,
        "snapshot_latest_date": snapshot_latest.isoformat(),
        "current_date_gate": current_date.isoformat(),
        "fetch_start_date": start_date,
        "fetch_end_date": end_date,
        "tickers": tickers,
        "ticker_metrics": ticker_metrics,
        "completed_append_candidate_rows": int(len(completed_rows)),
        "completed_append_candidate_dates": []
        if completed_rows.empty
        else sorted(pd.to_datetime(completed_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()),
        "total_fetched_rows": total_fetched_rows,
        "total_new_rows": total_new_rows,
        "total_partial_excluded_rows": total_partial_excluded_rows,
        "fallback_used_count": fallback_used_count,
        "contract_failures": contract_failures,
        "empty_fetch_tickers": empty_fetch_tickers,
        "empty_fetch_count": len(empty_fetch_tickers),
        "dry_run_only": True,
    }


def context_update_dry_run(price_with_candidates: pd.DataFrame, stock_info: pd.DataFrame, tickers: list[str]) -> dict[str, Any]:
    existing_context = load_context_frames()
    latest_price_date = price_with_candidates["date"].max().date()
    session = requests.Session()
    session.trust_env = False
    fred_start = (latest_price_date - timedelta(days=20)).isoformat()
    macro_latest, macro_fetch = fetch_fred_macro(fred_start, latest_price_date.isoformat(), session)

    existing_macro = existing_context.get("macroeconomic_indicators", pd.DataFrame())
    existing_macro_latest = None if existing_macro.empty else pd.to_datetime(existing_macro["date"]).max().date().isoformat()
    macro_new_dates = []
    if not macro_latest.empty and not existing_macro.empty:
        existing_dates = set(pd.to_datetime(existing_macro["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().tolist())
        macro_new_dates = sorted(
            value
            for value in pd.to_datetime(macro_latest["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()
            if value not in existing_dates
        )

    breadth_frame, breadth_metrics = build_market_breadth_context(price_with_candidates, min_ticker_count=50)
    sector_frame, sector_metrics = build_sector_context(price_with_candidates, stock_info)
    fundamentals_frame, fundamentals_metrics = fetch_edgar_fundamentals_context(tickers, sleep_seconds=0.05, limit=len(tickers))
    existing_fundamentals = existing_context.get("company_fundamentals", pd.DataFrame())
    new_fundamental_rows = 0
    if not fundamentals_frame.empty:
        if existing_fundamentals.empty:
            new_fundamental_rows = int(len(fundamentals_frame))
        else:
            existing_keys = set(
                zip(
                    existing_fundamentals["ticker"].astype(str).str.upper(),
                    pd.to_datetime(existing_fundamentals["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                    pd.to_datetime(existing_fundamentals["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                )
            )
            candidate_keys = zip(
                fundamentals_frame["ticker"].astype(str).str.upper(),
                pd.to_datetime(fundamentals_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                pd.to_datetime(fundamentals_frame["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            )
            new_fundamental_rows = sum(1 for key in candidate_keys if key not in existing_keys)

    return {
        "macro": {
            "existing_latest_date": existing_macro_latest,
            "fetch_status": macro_fetch.get("status"),
            "fetched_rows": macro_fetch.get("rows"),
            "fetched_date_min": macro_fetch.get("date_min"),
            "fetched_date_max": macro_fetch.get("date_max"),
            "new_dates_not_in_local_context": macro_new_dates,
            "update_policy": "append/update local context parquet only when FRED observation changes",
        },
        "market_breadth": {
            **breadth_metrics,
            "candidate_checksum": frame_checksum(breadth_frame, ["date", "nh_nl_index", "ma200_pct", "total_count"]),
            "existing_checksum": frame_checksum(existing_context.get("market_breadth", pd.DataFrame()), ["date", "nh_nl_index", "ma200_pct", "total_count"]),
            "update_policy": "recompute from local yfinance price universe after price append",
        },
        "sector_returns": {
            **sector_metrics,
            "candidate_checksum": frame_checksum(sector_frame, ["date", "sector", "return", "close"]),
            "existing_checksum": frame_checksum(existing_context.get("sector_returns", pd.DataFrame()), ["date", "sector", "return", "close"]),
            "model_feature_inclusion": False,
            "update_policy": "recompute from local yfinance price universe after price append",
        },
        "fundamentals": {
            **fundamentals_metrics,
            "new_rows_vs_local_context_for_rehearsal_tickers": new_fundamental_rows,
            "update_policy": "daily lightweight check for target tickers; weekly or filing-alert based broader SEC refresh",
        },
    }


def indicator_refresh_dry_run(price_with_candidates: pd.DataFrame, context_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    macro = context_frames.get("macroeconomic_indicators", pd.DataFrame())
    breadth = context_frames.get("market_breadth", pd.DataFrame())
    fundamentals = context_frames.get("company_fundamentals", pd.DataFrame())
    current_1d = load_parquet(INDICATOR_1D_PATH)
    current_1w = load_parquet(INDICATOR_1W_PATH)
    rebuilt_1d = build_features(price_df=price_with_candidates, macro_df=macro, breadth_df=breadth, fundamentals_df=fundamentals, timeframe="1D")
    rebuilt_1w = build_features(price_df=price_with_candidates, macro_df=macro, breadth_df=breadth, fundamentals_df=fundamentals, timeframe="1W")
    latest_daily_date = price_with_candidates["date"].max()
    complete_week_end = latest_complete_period_end(latest_daily_date, "1W")
    one_week_partial_rows = 0
    if not rebuilt_1w.empty and complete_week_end is not None:
        one_week_partial_rows = int((pd.to_datetime(rebuilt_1w["date"], errors="coerce") > complete_week_end).sum())

    def stats(frame: pd.DataFrame, current: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        feature_columns = [column for column in frame.columns if column in current.columns and column not in {"ticker", "date", "timeframe", "source", "provider"}]
        numeric = frame[feature_columns].select_dtypes(include=[np.number]).to_numpy(dtype=float) if feature_columns else np.array([])
        return {
            "timeframe": timeframe,
            "candidate_rows": int(len(frame)),
            "current_rows": int(len(current)),
            "candidate_date_max": None if frame.empty else pd.to_datetime(frame["date"], errors="coerce").max().date().isoformat(),
            "current_date_max": None if current.empty else pd.to_datetime(current["date"], errors="coerce").max().date().isoformat(),
            "feature_non_finite_count": int((~np.isfinite(numeric)).sum()) if numeric.size else 0,
            "candidate_checksum": indicator_value_checksum(frame),
            "current_checksum": indicator_value_checksum(current),
            "checksum_would_change": indicator_value_checksum(frame) != indicator_value_checksum(current),
        }

    return {
        "1D": stats(rebuilt_1d, current_1d, "1D"),
        "1W": {
            **stats(rebuilt_1w, current_1w, "1W"),
            "latest_complete_week_end": None if complete_week_end is None else complete_week_end.date().isoformat(),
            "partial_week_rows": one_week_partial_rows,
        },
        "1M": {
            "policy": "skip in CP134 daily pipeline; price-only or separate monthly completed-period refresh",
        },
        "dry_run_only": True,
    }


def product_and_readiness_plan() -> dict[str, Any]:
    history_summary = snapshot_summary(PRODUCT_HISTORY_1D_PATH, ["ticker", "timeframe", "role", "asof_date", "run_id"]) if PRODUCT_HISTORY_1D_PATH.exists() else {"exists": False}
    manifest = {}
    if PRODUCT_HISTORY_1D_MANIFEST_PATH.exists():
        manifest = json.loads(PRODUCT_HISTORY_1D_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "product_inference": {
            "execution_in_cp134": "SKIPPED_DRY_RUN_ONLY",
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "required_storage_helper": "save_product_latest_predictions",
            "required_storage_contract": "product_latest_only",
            "thin_upload_target_rows_for_5_tickers": {"predictions": 10, "prediction_evaluations": 10},
            "forbidden": ["ai.inference --save bulk", "composite save", "prediction history bulk upload"],
        },
        "rolling_history": {
            "current_snapshot": history_summary,
            "manifest": manifest,
            "daily_policy": "latest product forecast와 별개로 rolling replay는 별도 주기 또는 별도 CP에서 갱신",
        },
        "scanner": {
            "execution_in_cp134": "SKIPPED",
            "future_hook": "500티커 forward-only inference 이후 top-k scanner summary만 Supabase thin upload",
        },
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_decision"]
    price = metrics["price_update_dry_run"]
    readiness = metrics["readiness"]
    report = f"""# CP134-DG 로컬 일일 시장/컨텍스트 업데이트 파이프라인 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

local parquet 원천 구조에 맞춘 일일 업데이트 순서를 고정했고, 5티커 yfinance 완료 거래일 dry-run을 수행했다. 이번 실행에서는 새 완료 거래일 row가 있으면 append 후보로 기록하고, 없으면 `PASS_WITH_NO_NEW_DAY`로 처리한다. 단, yfinance fetch가 빈 응답이면 신규 거래일 없음으로 보지 않고 중단 조건으로 분리한다. Supabase 대량 read/write, EODHD fallback, 모델 학습, inference 저장은 수행하지 않았다.

## 2. 일일 파이프라인 계약

1. 가격 업데이트: yfinance만 사용하고 `row.date < current_date` 완료 거래일만 append한다.
2. 컨텍스트 업데이트: breadth/sector는 local price universe에서 재계산하고, macro는 FRED 최신 observation만 append/update하며, fundamentals는 SEC `filing_date` 기준 신규 filing만 반영한다.
3. 지표 갱신: 1D는 incremental refresh, 1W는 완료 주만 refresh, 1M은 daily job에서 skip한다.
4. 캐시/해시: indicator/context 값이 바뀌면 source_data_hash와 manifest mismatch로 기존 feature cache 재사용을 막는다.
5. 제품 추론: 별도 단계에서 1D line/band latest inference만 실행하고 `save_product_latest_predictions()`만 사용한다.
6. scanner: 500티커 운영 전까지 skip한다.
7. readiness: local price/indicator/context/product latest date와 EODHD fallback 0, Supabase bulk read/write 0을 확인한다.

## 3. 이번 리허설 결과

| 항목 | 값 |
|---|---|
| price dry-run status | {price["status"]} |
| daily update state | {price.get("daily_update_state")} |
| snapshot latest date | {price["snapshot_latest_date"]} |
| current date gate | {price["current_date_gate"]} |
| completed append candidate rows | {price["completed_append_candidate_rows"]} |
| completed append candidate dates | {price["completed_append_candidate_dates"]} |
| total fetched rows | {price.get("total_fetched_rows")} |
| total new rows | {price.get("total_new_rows")} |
| total partial excluded rows | {price.get("total_partial_excluded_rows")} |
| fallback used count | {price["fallback_used_count"]} |
| contract failures | {price["contract_failures"]} |
| empty fetch count | {price.get("empty_fetch_count", 0)} |
| empty fetch tickers | {price.get("empty_fetch_tickers", [])} |

## 4. 컨텍스트 업데이트 설계 검증

| context | update policy |
|---|---|
| macro | {metrics["context_update_dry_run"]["macro"]["update_policy"]} |
| market_breadth | {metrics["context_update_dry_run"]["market_breadth"]["update_policy"]} |
| sector_returns | {metrics["context_update_dry_run"]["sector_returns"]["update_policy"]} |
| fundamentals | {metrics["context_update_dry_run"]["fundamentals"]["update_policy"]} |

## 5. 지표 갱신 게이트

| 항목 | 1D | 1W |
|---|---:|---:|
| current rows | {metrics["indicator_refresh_dry_run"]["1D"]["current_rows"]} | {metrics["indicator_refresh_dry_run"]["1W"]["current_rows"]} |
| candidate rows | {metrics["indicator_refresh_dry_run"]["1D"]["candidate_rows"]} | {metrics["indicator_refresh_dry_run"]["1W"]["candidate_rows"]} |
| current latest | {metrics["indicator_refresh_dry_run"]["1D"]["current_date_max"]} | {metrics["indicator_refresh_dry_run"]["1W"]["current_date_max"]} |
| candidate latest | {metrics["indicator_refresh_dry_run"]["1D"]["candidate_date_max"]} | {metrics["indicator_refresh_dry_run"]["1W"]["candidate_date_max"]} |
| non-finite count | {metrics["indicator_refresh_dry_run"]["1D"]["feature_non_finite_count"]} | {metrics["indicator_refresh_dry_run"]["1W"]["feature_non_finite_count"]} |
| checksum would change | {metrics["indicator_refresh_dry_run"]["1D"]["checksum_would_change"]} | {metrics["indicator_refresh_dry_run"]["1W"]["checksum_would_change"]} |
| partial week rows | - | {metrics["indicator_refresh_dry_run"]["1W"]["partial_week_rows"]} |

## 6. 제품 추론과 얇은 업로드 경계

CP134에서는 product checkpoint inference와 Supabase thin upload를 실행하지 않았다. 운영 단계에서는 1D line/band latest inference 후 `save_product_latest_predictions()`만 사용하고, 5티커 기준 predictions 10행과 evaluations 10행 수준을 예상한다. `ai.inference --save` bulk 저장, composite 저장, history bulk upload는 금지한다.

## 7. 준비 상태

| 항목 | 값 |
|---|---|
| local price latest | {readiness["local_price_latest_date"]} |
| local indicator 1D latest | {readiness["local_indicator_1D_latest_date"]} |
| local indicator 1W latest | {readiness["local_indicator_1W_latest_date"]} |
| macro latest | {readiness["context_latest_dates"].get("macroeconomic_indicators")} |
| breadth latest | {readiness["context_latest_dates"].get("market_breadth")} |
| fundamentals latest filing_date | {readiness["context_latest_dates"].get("company_fundamentals_filing_date")} |
| sector latest | {readiness["context_latest_dates"].get("sector_returns")} |
| product history latest | {readiness["product_history_latest_asof_date"]} |
| EODHD fallback used | {readiness["eodhd_fallback_used"]} |
| Supabase bulk read/write | {readiness["supabase_bulk_read_write"]} |

## 8. 최종 판단

{final["summary"]}

실패 목록: {final["failures"]}

경고 목록: {final["warnings"]}

## 9. 금지 작업 미발생 확인

Supabase price_data/indicators/context 대량 write, Supabase 대량 read, 모델 학습, full inference 저장, DB row delete/update, EODHD 호출, 프론트 수정은 수행하지 않았다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_runbook(path: Path) -> None:
    runbook = r"""# 로컬 일일 업데이트 런북

## 목적

local parquet를 원천 저장소로 유지하고, Supabase는 제품 표시용 thin DB로만 사용한다.

## 매일 실행 순서

1. 환경 고정
   - `MARKET_DATA_PROVIDER=yfinance`
   - `MARKET_DATA_FALLBACK_PROVIDER=`
   - `EODHD_API_KEY=`
   - `LENS_DATA_BACKEND=local`
   - `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`

2. dry-run
   - `.\scripts\run_local_daily_update.ps1 -DryRun`
   - 신규 완료 거래일이 없으면 `PASS_WITH_NO_NEW_DAY`로 종료해도 정상이다.
   - yfinance fetch가 빈 응답이면 신규 거래일 없음이 아니라 중단 조건으로 본다.
   - 상태값은 `fetch_failed`, `no_new_rows`, `partial_day_filtered`, `append_ready`, `append_done`으로 분리한다.

3. price append
   - yfinance에서 `row.date < current_date`인 완료 거래일만 append한다.
   - duplicate `(ticker,date,source)`는 0이어야 한다.
   - adjusted OHLC contract가 실패하면 즉시 중단한다.

4. context refresh
   - breadth: local price universe 기준 재계산
   - sector_returns: local price + stock_info 기준 재계산
   - macro: FRED 최신 observation만 append/update
   - fundamentals: SEC EDGAR `filing_date` 기준 신규 filing만 반영

5. indicator refresh
   - 1D: append된 ticker/date 주변 lookback만 refresh
   - 1W: 완료된 W-FRI period만 refresh
   - 1M: daily job에서는 skip

6. cache/hash gate
   - indicator/context checksum이 바뀌면 기존 feature cache를 재사용하지 않는다.
   - manifest provider/source/context_hash mismatch면 재생성한다.

7. product inference와 thin upload
   - 1D line/band latest inference만 실행한다.
   - 저장은 `save_product_latest_predictions()`만 허용한다.
   - bulk prediction history와 composite 저장은 금지한다.

8. readiness
   - local price latest date
   - local indicator latest date
   - context latest/asof coverage
   - product prediction latest asof_date
   - EODHD fallback 0
   - Supabase bulk read/write 0

## 실패 시 중단 조건

- EODHD fallback 발생
- adjusted OHLC contract 위반
- partial current-date row append 후보 발생
- 1W partial period row 발생
- feature NaN/Inf 발생
- Supabase bulk read/write 발생
- product latest-only row 수 제한 초과
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(runbook, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP134 local daily update dry-run rehearsal")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--current-date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp134_local_daily_update_pipeline_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp134_local_daily_update_pipeline_report.md"))
    parser.add_argument("--runbook-path", default=str(DOCS_DIR / "local_daily_update_runbook.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    current_date = pd.to_datetime(args.current_date).date()
    tickers = [ticker.upper() for ticker in args.tickers]
    created_at = utc_now_iso()

    price = load_parquet(PRICE_PATH)
    stock_info = load_parquet(STOCK_INFO_PATH)
    completed_rows, price_metrics = completed_price_update_dry_run(tickers, current_date, args.lookback_days)
    if completed_rows.empty:
        price_with_candidates = price.copy()
    else:
        price_with_candidates = pd.concat([price, completed_rows], ignore_index=True)
        price_with_candidates = price_with_candidates.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date", "source"], keep="last")

    context_metrics = context_update_dry_run(price_with_candidates, stock_info, tickers)
    indicator_metrics = indicator_refresh_dry_run(price_with_candidates, load_context_frames())
    product_plan = product_and_readiness_plan()

    context_frames = load_context_frames()
    context_latest_dates = {}
    for name, frame in context_frames.items():
        if frame.empty or "date" not in frame.columns:
            context_latest_dates[name] = None
            continue
        context_latest_dates[name] = pd.to_datetime(frame["date"], errors="coerce").max().date().isoformat()
    fundamentals = context_frames.get("company_fundamentals", pd.DataFrame())
    if not fundamentals.empty and "filing_date" in fundamentals.columns:
        context_latest_dates["company_fundamentals_filing_date"] = pd.to_datetime(fundamentals["filing_date"], errors="coerce").max().date().isoformat()

    product_history_latest = None
    if PRODUCT_HISTORY_1D_PATH.exists():
        history = load_parquet(PRODUCT_HISTORY_1D_PATH)
        if not history.empty and "asof_date" in history.columns:
            product_history_latest = pd.to_datetime(history["asof_date"], errors="coerce").max().date().isoformat()

    readiness = {
        "local_price_latest_date": price["date"].max().date().isoformat(),
        "local_indicator_1D_latest_date": snapshot_summary(INDICATOR_1D_PATH, ["ticker", "timeframe", "date", "source"]).get("date_max"),
        "local_indicator_1W_latest_date": snapshot_summary(INDICATOR_1W_PATH, ["ticker", "timeframe", "date", "source"]).get("date_max"),
        "context_latest_dates": context_latest_dates,
        "product_history_latest_asof_date": product_history_latest,
        "line_run_id": LINE_RUN_ID,
        "band_run_id": BAND_RUN_ID,
        "eodhd_fallback_used": bool(price_metrics.get("fallback_used_count")),
        "supabase_bulk_read_write": False,
    }

    failures = []
    warnings = []
    if price_metrics["status"] == "FAIL":
        failures.append("price update dry-run failed")
    if price_metrics.get("empty_fetch_count", 0):
        warnings.append(f"{price_metrics['empty_fetch_count']} yfinance fetches returned empty")
    if indicator_metrics["1D"]["feature_non_finite_count"] != 0:
        failures.append("1D indicator candidate has non-finite features")
    if indicator_metrics["1W"]["feature_non_finite_count"] != 0:
        failures.append("1W indicator candidate has non-finite features")
    if indicator_metrics["1W"]["partial_week_rows"] != 0:
        failures.append("1W indicator candidate contains partial week rows")
    if price_metrics["status"] == "PASS_WITH_NO_NEW_DAY":
        warnings.append("no new completed market day rows available")
    if context_metrics["fundamentals"].get("new_rows_vs_local_context_for_rehearsal_tickers", 0) == 0:
        warnings.append("no new SEC filing rows for rehearsal tickers")
    if product_plan["product_inference"]["execution_in_cp134"] != "EXECUTED":
        warnings.append("product inference/thin upload left as dry-run boundary")

    if failures:
        status = "FAIL"
        if price_metrics.get("empty_fetch_count") == len(tickers):
            summary = "yfinance 최신 가격 조회가 리허설 5티커 모두 빈 응답으로 끝났다. 신규 거래일 없음으로 판정하지 않고, daily append와 EODHD 해지 gate를 보류한다."
        else:
            summary = "daily update pipeline 리허설 중 중단 조건이 발생했다."
    elif price_metrics["status"] == "PASS_WITH_NO_NEW_DAY":
        status = "PASS_WITH_NO_NEW_DAY"
        summary = "신규 완료 거래일 row는 없었지만 daily update 순서, context refresh 경계, partial period gate, thin upload 경계가 문서와 dry-run으로 고정됐다."
    elif warnings:
        status = "WARN"
        summary = "daily update pipeline은 동작 가능하지만 일부 단계가 dry-run 또는 no-op으로 남아 있다."
    else:
        status = "PASS"
        summary = "신규 완료 거래일 후보까지 포함해 local daily update pipeline 리허설이 통과했다."

    metrics: dict[str, Any] = {
        "cp": "CP134-DG",
        "created_at": created_at,
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "EODHD_API_KEY_SET": bool(os.environ.get("EODHD_API_KEY")),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
        },
        "snapshots": {
            "price_1D": snapshot_summary(PRICE_PATH, ["ticker", "date", "source"]),
            "price_1W": snapshot_summary(PRICE_1W_PATH, ["ticker", "date", "source"]),
            "indicators_1D": snapshot_summary(INDICATOR_1D_PATH, ["ticker", "timeframe", "date", "source"]),
            "indicators_1W": snapshot_summary(INDICATOR_1W_PATH, ["ticker", "timeframe", "date", "source"]),
        },
        "price_update_dry_run": price_metrics,
        "context_update_dry_run": context_metrics,
        "indicator_refresh_dry_run": indicator_metrics,
        "product_and_scanner_plan": product_plan,
        "readiness": readiness,
        "forbidden_actions_observed": {
            "supabase_price_data_indicators_context_bulk_write": False,
            "supabase_bulk_read": False,
            "model_training": False,
            "full_inference_save": False,
            "db_row_delete_update": False,
            "eodhd_call": False,
            "frontend_modify": False,
        },
        "final_decision": {
            "status": status,
            "failures": failures,
            "warnings": warnings,
            "summary": summary,
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_json(LOG_DIR / "cp134_local_daily_update_pipeline_metrics.json", metrics)
    write_report(Path(args.report_path), metrics)
    write_runbook(Path(args.runbook_path))
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
