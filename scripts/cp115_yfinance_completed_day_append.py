from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
BACKUP_DIR = SNAPSHOT_DIR / "backups"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp115_yfinance_completed_day_append_metrics.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "cp115_yfinance_completed_day_append_report.md"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp115_yfinance_completed_day_append"

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features  # noqa: E402
from backend.collector.sources.market_data_providers import (  # noqa: E402
    fetch_market_data,
    provider_adjustment_policy,
)
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402


PRICE_PATH = SNAPSHOT_DIR / "price_data_yfinance.parquet"
INDICATOR_PATH = SNAPSHOT_DIR / "indicators_yfinance_1D.parquet"


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
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_summary(path: Path, duplicate_keys: list[str]) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    source_values = sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist()) if "source" in frame.columns else []
    provider_values = (
        sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist()) if "provider" in frame.columns else []
    )
    return {
        "path": str(path.relative_to(ROOT_DIR)),
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": pd.to_datetime(frame["date"]).min().strftime("%Y-%m-%d") if "date" in frame.columns else None,
        "date_max": pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d") if "date" in frame.columns else None,
        "duplicate_count": int(frame.duplicated(subset=duplicate_keys).sum())
        if all(column in frame.columns for column in duplicate_keys)
        else None,
        "source_values": source_values,
        "provider_values": provider_values,
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def price_frame_to_records(ticker: str, frame: pd.DataFrame, *, updated_at: str) -> pd.DataFrame:
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce")
    policy = provider_adjustment_policy("yfinance")
    records = pd.DataFrame(
        {
            "ticker": ticker,
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
            "updated_at": updated_at,
        }
    )
    return records.dropna(subset=["date"]).reset_index(drop=True)


def validate_local_price_contract(frame: pd.DataFrame, tickers: list[str]) -> dict[str, Any]:
    subset = frame[frame["ticker"].isin(tickers)].copy()
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    required_columns = ["open", "high", "low", "close", "adjusted_close"]
    required_values = subset[required_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_count = int(np.isfinite(required_values).sum())
    duplicate_count = int(subset.duplicated(subset=["ticker", "date", "source"]).sum())

    close = pd.to_numeric(subset["close"], errors="coerce")
    adjusted_close = pd.to_numeric(subset["adjusted_close"], errors="coerce")
    factor = adjusted_close / close.where(close.abs() > 1e-9)
    adjusted_open = pd.to_numeric(subset["open"], errors="coerce") * factor
    adjusted_high = pd.to_numeric(subset["high"], errors="coerce") * factor
    adjusted_low = pd.to_numeric(subset["low"], errors="coerce") * factor
    high_violation = adjusted_high + 1e-9 < pd.concat([adjusted_open, adjusted_close], axis=1).max(axis=1)
    low_violation = adjusted_low - 1e-9 > pd.concat([adjusted_open, adjusted_close], axis=1).min(axis=1)
    high_low_violation = adjusted_high + 1e-9 < adjusted_low
    source_missing = int(subset["source"].isna().sum()) if "source" in subset.columns else len(subset)
    provider_missing = int(subset["provider"].isna().sum()) if "provider" in subset.columns else len(subset)
    policy_missing = (
        int(subset["provider_adjustment_policy"].isna().sum())
        if "provider_adjustment_policy" in subset.columns
        else len(subset)
    )
    violations = []
    if finite_count != int(required_values.size):
        violations.append("required_price_non_finite")
    if duplicate_count:
        violations.append("duplicate_ticker_date_source")
    if bool(high_violation.any()):
        violations.append("adjusted_high_below_open_or_close")
    if bool(low_violation.any()):
        violations.append("adjusted_low_above_open_or_close")
    if bool(high_low_violation.any()):
        violations.append("adjusted_high_below_low")
    if source_missing or provider_missing or policy_missing:
        violations.append("source_provider_policy_missing")
    return {
        "passed": not violations,
        "violations": violations,
        "row_count": int(len(subset)),
        "duplicate_ticker_date_source": duplicate_count,
        "required_finite_count": finite_count,
        "required_value_count": int(required_values.size),
        "adjusted_high_violation_count": int(high_violation.sum()),
        "adjusted_low_violation_count": int(low_violation.sum()),
        "adjusted_high_low_violation_count": int(high_low_violation.sum()),
        "source_missing_count": source_missing,
        "provider_missing_count": provider_missing,
        "provider_adjustment_policy_missing_count": policy_missing,
    }


def backup_snapshots(timestamp: str) -> dict[str, Any]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    price_backup = BACKUP_DIR / f"price_data_yfinance_before_cp115_{timestamp}.parquet"
    indicator_backup = BACKUP_DIR / f"indicators_yfinance_1D_before_cp115_{timestamp}.parquet"
    shutil.copy2(PRICE_PATH, price_backup)
    shutil.copy2(INDICATOR_PATH, indicator_backup)
    return {
        "created": True,
        "price_backup": str(price_backup.relative_to(ROOT_DIR)),
        "indicator_backup": str(indicator_backup.relative_to(ROOT_DIR)),
        "price_backup_bytes": int(price_backup.stat().st_size),
        "indicator_backup_bytes": int(indicator_backup.stat().st_size),
    }


def fetch_completed_rows(
    tickers: list[str],
    *,
    snapshot_latest: date,
    current_date: date,
    lookback_days: int,
    updated_at: str,
) -> tuple[list[pd.DataFrame], dict[str, Any], list[str]]:
    start_date = (snapshot_latest - timedelta(days=lookback_days)).isoformat()
    end_date = current_date.isoformat()
    chunks: list[pd.DataFrame] = []
    ticker_metrics: dict[str, Any] = {}
    excluded_partial_dates: set[str] = set()

    for ticker in tickers:
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=None,
        )
        if result.frame.empty:
            ticker_metrics[ticker] = {
                "status": "FAIL_EMPTY",
                "provider": result.provider,
                "fallback_used": bool(result.fallback_used),
                "errors": result.errors,
                "fetched_rows": 0,
                "new_rows_after_snapshot_latest": 0,
                "completed_append_rows": 0,
            }
            continue

        contract = validate_adjusted_ohlc_contract(ticker, result.frame)
        records = price_frame_to_records(ticker, result.frame, updated_at=updated_at)
        records["date"] = pd.to_datetime(records["date"], errors="coerce")
        new_records = records[pd.to_datetime(records["date"]).dt.date > snapshot_latest].copy()
        partial_rows = new_records[pd.to_datetime(new_records["date"]).dt.date >= current_date].copy()
        completed_rows = new_records[pd.to_datetime(new_records["date"]).dt.date < current_date].copy()
        for value in pd.to_datetime(partial_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique():
            excluded_partial_dates.add(str(value))
        if contract.passed and not completed_rows.empty:
            chunks.append(completed_rows)
        ticker_metrics[ticker] = {
            "status": "PASS" if contract.passed else "FAIL_CONTRACT",
            "provider": result.provider,
            "fallback_provider": result.fallback_provider,
            "fallback_used": bool(result.fallback_used),
            "errors": result.errors,
            "fetched_rows": int(len(records)),
            "fetched_date_min": records["date"].min().strftime("%Y-%m-%d"),
            "fetched_date_max": records["date"].max().strftime("%Y-%m-%d"),
            "new_rows_after_snapshot_latest": int(len(new_records)),
            "new_dates": sorted(pd.to_datetime(new_records["date"], errors="coerce").dt.strftime("%Y-%m-%d").unique().tolist()),
            "partial_excluded_rows": int(len(partial_rows)),
            "partial_excluded_dates": sorted(
                pd.to_datetime(partial_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").unique().tolist()
            ),
            "completed_append_rows": int(len(completed_rows)),
            "completed_append_dates": sorted(
                pd.to_datetime(completed_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").unique().tolist()
            ),
            "adjusted_ohlc_passed": bool(contract.passed),
            "adjusted_ohlc_violations": contract.violations,
            "adjusted_ohlc_metrics": contract.metrics,
        }
    return chunks, ticker_metrics, sorted(excluded_partial_dates)


def refresh_indicator_rows(price_df: pd.DataFrame, tickers: list[str], *, updated_at: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    existing = pd.read_parquet(INDICATOR_PATH)
    existing["ticker"] = existing["ticker"].astype(str).str.upper()
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")

    price_subset = price_df[price_df["ticker"].isin(tickers)].copy()
    rebuilt = build_features(price_df=price_subset, timeframe="1D")
    rebuilt["source"] = "yfinance"
    rebuilt["provider"] = "yfinance"
    rebuilt["updated_at"] = updated_at
    rebuilt["date"] = pd.to_datetime(rebuilt["date"], errors="coerce")

    kept = existing[~existing["ticker"].isin(tickers)].copy()
    combined = pd.concat([kept, rebuilt], ignore_index=True)
    combined = combined.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    )
    duplicate_count = int(combined.duplicated(subset=["ticker", "timeframe", "date", "source"]).sum())
    source_values = sorted(combined["source"].dropna().astype(str).str.lower().unique().tolist())
    provider_values = sorted(combined["provider"].dropna().astype(str).str.lower().unique().tolist())
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_parquet(INDICATOR_PATH, index=False)
    return combined, {
        "rebuilt_tickers": tickers,
        "rebuilt_rows": int(len(rebuilt)),
        "rebuilt_date_min": rebuilt["date"].min().strftime("%Y-%m-%d") if not rebuilt.empty else None,
        "rebuilt_date_max": rebuilt["date"].max().strftime("%Y-%m-%d") if not rebuilt.empty else None,
        "combined_rows": int(len(combined)),
        "duplicate_ticker_timeframe_date_source": duplicate_count,
        "source_values": source_values,
        "provider_values": provider_values,
    }


def feature_target_finite_check(price_df: pd.DataFrame, indicator_df: pd.DataFrame, tickers: list[str], horizon: int = 5) -> dict[str, Any]:
    subset = indicator_df[indicator_df["ticker"].isin(tickers)].copy()
    feature_values = subset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    feature_non_finite = int((~np.isfinite(feature_values)).sum())

    price_subset = price_df[price_df["ticker"].isin(tickers)].copy()
    price_subset["date"] = pd.to_datetime(price_subset["date"], errors="coerce")
    target_frames = []
    for ticker, ticker_price in price_subset.groupby("ticker", sort=True):
        ordered = ticker_price.sort_values("date").copy()
        close = pd.to_numeric(ordered["adjusted_close"].fillna(ordered["close"]), errors="coerce")
        future = close.shift(-horizon)
        target = (future - close) / close
        target_frames.append(pd.DataFrame({"ticker": ticker, "date": ordered["date"], "target": target}))
    target_frame = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame(columns=["ticker", "date", "target"])
    target_frame = target_frame.dropna(subset=["target"])
    target_values = target_frame["target"].to_numpy(dtype=float)
    target_non_finite = int((~np.isfinite(target_values)).sum())
    return {
        "feature_columns_checked": len(FEATURE_COLUMNS),
        "feature_rows_checked": int(len(subset)),
        "feature_non_finite_count": feature_non_finite,
        "target_horizon": horizon,
        "target_rows_checked": int(len(target_frame)),
        "target_non_finite_count": target_non_finite,
        "passed": feature_non_finite == 0 and target_non_finite == 0,
    }


def _ticker_table(ticker_metrics: dict[str, Any]) -> str:
    rows = []
    for ticker, item in sorted(ticker_metrics.items()):
        rows.append(
            "| {ticker} | {status} | {fetched_rows} | {new_rows} | {append_rows} | {append_dates} | {partial_dates} | {fallback} | {contract} |".format(
                ticker=ticker,
                status=item.get("status"),
                fetched_rows=item.get("fetched_rows"),
                new_rows=item.get("new_rows_after_snapshot_latest"),
                append_rows=item.get("completed_append_rows"),
                append_dates=item.get("completed_append_dates"),
                partial_dates=item.get("partial_excluded_dates"),
                fallback=item.get("fallback_used"),
                contract=item.get("adjusted_ohlc_passed"),
            )
        )
    return "\n".join(rows)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    before = metrics["snapshots_before"]
    after = metrics.get("snapshots_after", {})
    fetch = metrics["yfinance_fetch"]
    append = metrics.get("append_result", {})
    indicator = metrics.get("indicator_refresh", {})
    hash_change = metrics.get("hash_and_cache", {})
    finite = metrics.get("feature_target_finite_check", {})
    final = metrics["final_decision"]
    report = f"""# CP115-D yfinance 완료 거래일 append + 1D indicator refresh 리허설

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: {final["status"]}

{final["summary"]}

이번 CP는 DB/Supabase를 쓰지 않고 local parquet만 업데이트했다. 모델 학습, inference, thin upload, checkpoint load, torch import는 수행하지 않았다.

## 2. Append 전후 Snapshot

| 항목 | Before | After |
|---|---:|---:|
| 1D price latest date | {before["price_1d"]["date_max"]} | {after.get("price_1d", {}).get("date_max")} |
| 1D price rows | {before["price_1d"]["rows"]} | {after.get("price_1d", {}).get("rows")} |
| 1D indicator latest date | {before["indicators_1d"]["date_max"]} | {after.get("indicators_1d", {}).get("date_max")} |
| 1D indicator rows | {before["indicators_1d"]["rows"]} | {after.get("indicators_1d", {}).get("rows")} |
| price duplicate ticker/date/source | {before["price_1d"]["duplicate_count"]} | {after.get("price_1d", {}).get("duplicate_count")} |
| indicator duplicate ticker/timeframe/date/source | {before["indicators_1d"]["duplicate_count"]} | {after.get("indicators_1d", {}).get("duplicate_count")} |

## 3. Backup

| 항목 | 값 |
|---|---|
| created | {metrics.get("backup", {}).get("created")} |
| price backup | {metrics.get("backup", {}).get("price_backup")} |
| indicator backup | {metrics.get("backup", {}).get("indicator_backup")} |

## 4. yfinance Fetch 및 Partial 제외

요청:
- tickers: {", ".join(metrics["scope"]["tickers"])}
- snapshot_latest_date: {fetch["request"]["snapshot_latest_date"]}
- current_date: {fetch["request"]["current_date"]}
- start_date: {fetch["request"]["start_date"]}
- end_date: {fetch["request"]["end_date"]}
- completed row rule: row.date < current_date

| ticker | status | fetched rows | new rows | appended rows | appended dates | partial excluded dates | fallback | adjusted OHLC |
|---|---:|---:|---:|---:|---|---|---:|---:|
{_ticker_table(fetch["ticker_metrics"])}

partial 제외:
- excluded_partial_dates: {fetch["excluded_partial_dates"]}
- expected_append_dates: {append.get("expected_append_dates")}
- appended_dates: {append.get("appended_dates")}

## 5. Price Append 검증

| 항목 | 값 |
|---|---:|
| price_rows_appended | {append.get("price_rows_appended")} |
| expected_price_rows_appended | {append.get("expected_price_rows_appended")} |
| appended_ticker_count | {append.get("appended_ticker_count")} |
| duplicate_after_append | {append.get("local_price_contract", {}).get("duplicate_ticker_date_source")} |
| adjusted_ohlc_passed | {append.get("local_price_contract", {}).get("passed")} |
| adjusted_ohlc_violations | {append.get("local_price_contract", {}).get("violations")} |

## 6. 1D Indicator Incremental Refresh

| 항목 | 값 |
|---|---:|
| rebuilt_tickers | {indicator.get("rebuilt_tickers")} |
| rebuilt_rows | {indicator.get("rebuilt_rows")} |
| rebuilt_date_min | {indicator.get("rebuilt_date_min")} |
| rebuilt_date_max | {indicator.get("rebuilt_date_max")} |
| combined_rows | {indicator.get("combined_rows")} |
| duplicate_ticker_timeframe_date_source | {indicator.get("duplicate_ticker_timeframe_date_source")} |
| source_values | {indicator.get("source_values")} |
| provider_values | {indicator.get("provider_values")} |

## 7. Source/Hash/Cache 변화

| 항목 | 값 |
|---|---|
| price sha before | {hash_change.get("price_sha256_before")} |
| price sha after | {hash_change.get("price_sha256_after")} |
| price hash changed | {hash_change.get("price_hash_changed")} |
| indicator sha before | {hash_change.get("indicator_sha256_before")} |
| indicator sha after | {hash_change.get("indicator_sha256_after")} |
| indicator hash changed | {hash_change.get("indicator_hash_changed")} |
| feature/index cache touched | {hash_change.get("feature_index_cache_touched")} |

주의: 이번 CP는 torch import와 `ai.preprocessing` import가 금지되어 있어서 모델 feature/index cache를 생성하지 않았다. 대신 local parquet file hash 변화로 source snapshot 변화만 기록했다.

## 8. Feature/Target Finite

| 항목 | 값 |
|---|---:|
| feature_rows_checked | {finite.get("feature_rows_checked")} |
| feature_columns_checked | {finite.get("feature_columns_checked")} |
| feature_non_finite_count | {finite.get("feature_non_finite_count")} |
| target_horizon | {finite.get("target_horizon")} |
| target_rows_checked | {finite.get("target_rows_checked")} |
| target_non_finite_count | {finite.get("target_non_finite_count")} |
| passed | {finite.get("passed")} |

## 9. EODHD 해지 가능 여부 업데이트

{metrics.get("eodhd_cancellation_update")}

## 10. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| DB write | {metrics["forbidden_actions_observed"]["db_write"]} |
| Supabase price_data/indicators 대량 read/write | {metrics["forbidden_actions_observed"]["supabase_price_data_indicators_bulk_read_write"]} |
| 모델 학습 | {metrics["forbidden_actions_observed"]["model_training"]} |
| inference | {metrics["forbidden_actions_observed"]["inference"]} |
| thin upload | {metrics["forbidden_actions_observed"]["thin_upload"]} |
| checkpoint load | {metrics["forbidden_actions_observed"]["checkpoint_load"]} |
| torch import | {metrics["forbidden_actions_observed"]["torch_import"]} |
| 프론트 수정 | {metrics["forbidden_actions_observed"]["frontend_modify"]} |
| EODHD API call | {metrics["forbidden_actions_observed"]["eodhd_api_call"]} |
| EODHD fallback | {metrics["forbidden_actions_observed"]["eodhd_fallback"]} |

## 11. 실행 명령

```powershell
python scripts\\cp115_yfinance_completed_day_append.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP115 yfinance 완료 거래일 local append 리허설")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--current-date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    current_date = pd.to_datetime(args.current_date).date()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_at = utc_now_iso()

    snapshots_before = {
        "price_1d": snapshot_summary(PRICE_PATH, ["ticker", "date", "source"]),
        "indicators_1d": snapshot_summary(INDICATOR_PATH, ["ticker", "timeframe", "date", "source"]),
    }
    snapshot_latest = pd.to_datetime(snapshots_before["price_1d"]["date_max"]).date()
    price_sha_before = snapshots_before["price_1d"]["sha256"]
    indicator_sha_before = snapshots_before["indicators_1d"]["sha256"]

    chunks, ticker_metrics, excluded_partial_dates = fetch_completed_rows(
        tickers,
        snapshot_latest=snapshot_latest,
        current_date=current_date,
        lookback_days=args.lookback_days,
        updated_at=updated_at,
    )
    fallback_used_any = bool(any(item.get("fallback_used") for item in ticker_metrics.values()))
    fetch_failed_count = int(sum(1 for item in ticker_metrics.values() if not str(item.get("status")).startswith("PASS")))
    completed_rows = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    appended_dates = (
        sorted(pd.to_datetime(completed_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d").unique().tolist())
        if not completed_rows.empty
        else []
    )
    expected_append_dates = [
        value
        for value in sorted(
            {
                new_date
                for item in ticker_metrics.values()
                for new_date in item.get("new_dates", [])
            }
        )
        if pd.to_datetime(value).date() < current_date
    ]

    metrics: dict[str, Any] = {
        "cp": "CP115-D",
        "generated_at": updated_at,
        "scope": {
            "tickers": tickers,
            "provider": "yfinance",
            "source": "yfinance",
            "timeframe": "1D",
            "current_date": current_date.isoformat(),
        },
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "EODHD_API_KEY_present": bool(os.environ.get("EODHD_API_KEY")),
        },
        "snapshots_before": snapshots_before,
        "yfinance_fetch": {
            "request": {
                "tickers": tickers,
                "snapshot_latest_date": snapshot_latest.isoformat(),
                "current_date": current_date.isoformat(),
                "start_date": (snapshot_latest - timedelta(days=args.lookback_days)).isoformat(),
                "end_date": current_date.isoformat(),
                "provider": "yfinance",
                "fallback_provider": None,
            },
            "ticker_metrics": ticker_metrics,
            "excluded_partial_dates": excluded_partial_dates,
            "fallback_used_any": fallback_used_any,
            "fetch_failed_count": fetch_failed_count,
        },
        "forbidden_actions_observed": {
            "db_write": False,
            "supabase_price_data_indicators_bulk_read_write": False,
            "model_training": False,
            "inference": False,
            "thin_upload": False,
            "checkpoint_load": False,
            "torch_import": "torch" in sys.modules,
            "frontend_modify": False,
            "eodhd_api_call": False,
            "eodhd_fallback": fallback_used_any,
        },
    }

    if fallback_used_any or fetch_failed_count:
        metrics["final_decision"] = {
            "status": "FAIL",
            "summary": "yfinance fetch 실패 또는 fallback 발생으로 append를 수행하지 않았다.",
        }
        metrics["eodhd_cancellation_update"] = "EODHD 해지 보류. yfinance fetch/fallback 문제를 먼저 해결해야 한다."
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        raise SystemExit(1)

    if completed_rows.empty:
        metrics["final_decision"] = {
            "status": "WARN",
            "summary": "완료 거래일 append 대상 row가 없어 local parquet를 변경하지 않았다.",
        }
        metrics["append_result"] = {
            "price_rows_appended": 0,
            "expected_price_rows_appended": len(tickers),
            "expected_append_dates": expected_append_dates,
            "appended_dates": [],
        }
        metrics["eodhd_cancellation_update"] = "EODHD 해지 gate 보류. 완료 거래일 row가 확보된 뒤 다시 실행해야 한다."
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        return

    metrics["backup"] = backup_snapshots(run_ts)

    price_df = pd.read_parquet(PRICE_PATH)
    price_df["ticker"] = price_df["ticker"].astype(str).str.upper()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    combined_price = pd.concat([price_df, completed_rows], ignore_index=True)
    combined_price["ticker"] = combined_price["ticker"].astype(str).str.upper()
    combined_price["date"] = pd.to_datetime(combined_price["date"], errors="coerce")
    combined_price = combined_price.sort_values(["ticker", "date", "source"]).drop_duplicates(
        subset=["ticker", "date", "source"],
        keep="last",
    )
    local_price_contract = validate_local_price_contract(combined_price, tickers)
    combined_price_out = combined_price.copy()
    combined_price_out["date"] = combined_price_out["date"].dt.strftime("%Y-%m-%d")
    combined_price_out.to_parquet(PRICE_PATH, index=False)

    combined_indicator, indicator_refresh = refresh_indicator_rows(combined_price, tickers, updated_at=updated_at)
    indicator_df_after = combined_indicator.copy()
    indicator_df_after["date"] = pd.to_datetime(indicator_df_after["date"], errors="coerce")
    finite_check = feature_target_finite_check(combined_price, indicator_df_after, tickers)

    snapshots_after = {
        "price_1d": snapshot_summary(PRICE_PATH, ["ticker", "date", "source"]),
        "indicators_1d": snapshot_summary(INDICATOR_PATH, ["ticker", "timeframe", "date", "source"]),
    }
    price_sha_after = snapshots_after["price_1d"]["sha256"]
    indicator_sha_after = snapshots_after["indicators_1d"]["sha256"]

    append_result = {
        "price_rows_appended": int(len(completed_rows)),
        "expected_price_rows_appended": len(tickers),
        "appended_ticker_count": int(completed_rows["ticker"].nunique()),
        "expected_append_dates": expected_append_dates,
        "appended_dates": appended_dates,
        "partial_current_date_appended": bool(
            (pd.to_datetime(completed_rows["date"], errors="coerce").dt.date >= current_date).any()
        ),
        "local_price_contract": local_price_contract,
    }
    metrics["append_result"] = append_result
    metrics["indicator_refresh"] = indicator_refresh
    metrics["feature_target_finite_check"] = finite_check
    metrics["snapshots_after"] = snapshots_after
    metrics["hash_and_cache"] = {
        "price_sha256_before": price_sha_before,
        "price_sha256_after": price_sha_after,
        "price_hash_changed": price_sha_before != price_sha_after,
        "indicator_sha256_before": indicator_sha_before,
        "indicator_sha256_after": indicator_sha_after,
        "indicator_hash_changed": indicator_sha_before != indicator_sha_after,
        "feature_index_cache_touched": False,
        "model_feature_cache_touched": False,
    }

    pass_conditions = [
        appended_dates == ["2026-05-04"] or (len(appended_dates) == 1 and pd.to_datetime(appended_dates[0]).date() < current_date),
        int(len(completed_rows)) == len(tickers),
        append_result["partial_current_date_appended"] is False,
        local_price_contract["passed"] is True,
        snapshots_after["price_1d"]["duplicate_count"] == 0,
        snapshots_after["indicators_1d"]["duplicate_count"] == 0,
        snapshots_after["price_1d"]["date_max"] == appended_dates[-1],
        snapshots_after["indicators_1d"]["date_max"] == appended_dates[-1],
        indicator_refresh["duplicate_ticker_timeframe_date_source"] == 0,
        finite_check["passed"] is True,
        not fallback_used_any,
        "torch" not in sys.modules,
    ]
    status = "PASS" if all(pass_conditions) else "WARN"
    metrics["final_decision"] = {
        "status": status,
        "summary": "완료 거래일만 local price parquet에 append했고 5티커 1D indicator refresh까지 통과했다."
        if status == "PASS"
        else "append/refresh는 수행됐지만 일부 검증 조건이 미흡해 추가 확인이 필요하다.",
        "pass_conditions": pass_conditions,
    }
    metrics["eodhd_cancellation_update"] = (
        "데이터 append/refresh gate는 PASS다. CP101의 EODHD-off 제품 루프 통과 이력과 결합하면 EODHD 해지 가능 후보로 볼 수 있다. "
        "다만 이번 CP에서는 inference/thin upload가 금지였으므로, 해지 직전에는 최신 asof 기준 제품 루프를 한 번 더 얇게 확인하는 것이 안전하다."
        if status == "PASS"
        else "EODHD 해지 보류. WARN 조건을 해소한 뒤 다시 판단한다."
    )

    write_json(log_dir / "run_summary.json", metrics)
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
