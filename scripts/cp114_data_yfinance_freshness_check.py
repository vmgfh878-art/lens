from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp114_data_yfinance_freshness_check_metrics.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "cp114_data_yfinance_freshness_check_report.md"

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.sources.market_data_providers import fetch_market_data  # noqa: E402
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"local snapshot 파일이 없습니다: {path}")
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def snapshot_summary(path: Path, *, duplicate_keys: list[str]) -> dict[str, Any]:
    frame = _load_snapshot(path)
    latest_date = None
    earliest_date = None
    if "date" in frame.columns and not frame.empty:
        latest_date = pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d")
        earliest_date = pd.to_datetime(frame["date"]).min().strftime("%Y-%m-%d")
    source_values = []
    provider_values = []
    if "source" in frame.columns:
        source_values = sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist())
    if "provider" in frame.columns:
        provider_values = sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist())
    return {
        "path": str(path.relative_to(ROOT_DIR)),
        "exists": path.exists(),
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": earliest_date,
        "date_max": latest_date,
        "source_values": source_values,
        "provider_values": provider_values,
        "duplicate_count": int(frame.duplicated(subset=duplicate_keys).sum())
        if all(column in frame.columns for column in duplicate_keys)
        else None,
        "bytes": int(path.stat().st_size),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def fetch_ticker_freshness(ticker: str, *, start_date: str, end_date: str, snapshot_latest: date) -> dict[str, Any]:
    result = fetch_market_data(
        ticker,
        start_date=start_date,
        end_date=end_date,
        provider_name="yfinance",
        fallback_provider_name=None,
        eodhd_api_key=None,
    )
    if result.frame.empty:
        return {
            "ticker": ticker,
            "status": "FAIL_EMPTY",
            "provider": result.provider,
            "fallback_provider": result.fallback_provider,
            "fallback_used": bool(result.fallback_used),
            "errors": result.errors,
            "fetched_rows": 0,
            "new_rows_after_snapshot_latest": 0,
            "adjusted_ohlc_passed": False,
            "adjusted_ohlc_violations": ["empty_frame"],
        }

    contract = validate_adjusted_ohlc_contract(ticker, result.frame)
    dates = pd.to_datetime(result.frame.index, errors="coerce").normalize()
    new_rows = result.frame[dates.date > snapshot_latest]
    new_dates = sorted(pd.to_datetime(new_rows.index, errors="coerce").strftime("%Y-%m-%d").unique().tolist())
    return {
        "ticker": ticker,
        "status": "PASS" if contract.passed else "FAIL_CONTRACT",
        "provider": result.provider,
        "fallback_provider": result.fallback_provider,
        "fallback_used": bool(result.fallback_used),
        "errors": result.errors,
        "fetched_rows": int(len(result.frame)),
        "fetched_date_min": dates.min().strftime("%Y-%m-%d"),
        "fetched_date_max": dates.max().strftime("%Y-%m-%d"),
        "new_rows_after_snapshot_latest": int(len(new_rows)),
        "new_dates": new_dates,
        "new_date_min": None if new_rows.empty else pd.to_datetime(new_rows.index).min().strftime("%Y-%m-%d"),
        "new_date_max": None if new_rows.empty else pd.to_datetime(new_rows.index).max().strftime("%Y-%m-%d"),
        "adjusted_ohlc_passed": bool(contract.passed),
        "adjusted_ohlc_violations": contract.violations,
        "adjusted_ohlc_metrics": contract.metrics,
    }


def _ticker_table(ticker_metrics: dict[str, Any]) -> str:
    rows = []
    for ticker, item in sorted(ticker_metrics.items()):
        rows.append(
            "| {ticker} | {status} | {fetched_rows} | {fetched_max} | {new_rows} | {new_max} | {fallback} | {contract} |".format(
                ticker=ticker,
                status=item.get("status"),
                fetched_rows=item.get("fetched_rows"),
                fetched_max=item.get("fetched_date_max"),
                new_rows=item.get("new_rows_after_snapshot_latest"),
                new_max=item.get("new_date_max"),
                fallback=item.get("fallback_used"),
                contract=item.get("adjusted_ohlc_passed"),
            )
        )
    return "\n".join(rows)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    snapshots = metrics["snapshots"]
    fetch = metrics["yfinance_fetch"]
    final_decision = metrics["final_decision"]
    report = f"""# CP114-D yfinance 데이터 freshness 체크

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: {final_decision["status"]}

{final_decision["summary"]}

이번 체크는 CP103 스크립트를 사용하지 않았다. 모델, inference, checkpoint, torch import 없이 local parquet 최신일과 yfinance 5티커 최신 row만 확인했다.

## 2. Local Snapshot 최신일

| snapshot | rows | tickers | date min | date max | duplicate | source | provider |
|---|---:|---:|---|---|---:|---|---|
| 1D price | {snapshots["price_1d"]["rows"]} | {snapshots["price_1d"]["ticker_count"]} | {snapshots["price_1d"]["date_min"]} | {snapshots["price_1d"]["date_max"]} | {snapshots["price_1d"]["duplicate_count"]} | {snapshots["price_1d"]["source_values"]} | {snapshots["price_1d"]["provider_values"]} |
| 1D indicators | {snapshots["indicators_1d"]["rows"]} | {snapshots["indicators_1d"]["ticker_count"]} | {snapshots["indicators_1d"]["date_min"]} | {snapshots["indicators_1d"]["date_max"]} | {snapshots["indicators_1d"]["duplicate_count"]} | {snapshots["indicators_1d"]["source_values"]} | {snapshots["indicators_1d"]["provider_values"]} |
| 1W price | {snapshots["price_1w"]["rows"]} | {snapshots["price_1w"]["ticker_count"]} | {snapshots["price_1w"]["date_min"]} | {snapshots["price_1w"]["date_max"]} | {snapshots["price_1w"]["duplicate_count"]} | {snapshots["price_1w"]["source_values"]} | {snapshots["price_1w"]["provider_values"]} |
| 1W indicators | {snapshots["indicators_1w"]["rows"]} | {snapshots["indicators_1w"]["ticker_count"]} | {snapshots["indicators_1w"]["date_min"]} | {snapshots["indicators_1w"]["date_max"]} | {snapshots["indicators_1w"]["duplicate_count"]} | {snapshots["indicators_1w"]["source_values"]} | {snapshots["indicators_1w"]["provider_values"]} |

## 3. yfinance 최신 조회

요청:
- tickers: {", ".join(metrics["scope"]["tickers"])}
- start_date: {fetch["request"]["start_date"]}
- end_date: {fetch["request"]["end_date"]}
- snapshot_latest_date: {fetch["request"]["snapshot_latest_date"]}
- fallback_used_any: {fetch["fallback_used_any"]}
- torch_imported: {metrics["torch_imported"]}
- potential_partial_current_day_rows: {fetch["potential_partial_current_day_rows"]}

| ticker | status | fetched rows | fetched max | new rows | new max | fallback | adjusted OHLC |
|---|---:|---:|---|---:|---|---:|---:|
{_ticker_table(fetch["ticker_metrics"])}

## 4. 판정

| 항목 | 값 |
|---|---|
| total_new_rows_after_snapshot_latest | {fetch["total_new_rows_after_snapshot_latest"]} |
| all_adjusted_ohlc_passed | {fetch["all_adjusted_ohlc_passed"]} |
| fallback_used_any | {fetch["fallback_used_any"]} |
| fetch_failed_count | {fetch["fetch_failed_count"]} |
| new_dates_all | {fetch["new_dates_all"]} |
| partial_day_warning | {fetch["partial_day_warning"]} |
| 1D/1W 모델 실험 계속 가능 여부 | {metrics["model_experiment_continuation"]} |
| EODHD 해지 gate 상태 | {metrics["eodhd_cancellation_gate_status"]} |

## 5. 다음 순서

사용자가 지정한 다음 순서를 유지한다.

1. 데이터 freshness 체크 결과 받기
2. CP114-BM/LM 결과 기반 1W 후보 정리
3. 1W 저장 후보 재현 CP
4. 그 다음 1D 추가 개선 또는 1W 제품 표시

## 6. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| 모델 학습 | {metrics["forbidden_actions_observed"]["model_training"]} |
| inference 실행 | {metrics["forbidden_actions_observed"]["inference"]} |
| DB write | {metrics["forbidden_actions_observed"]["db_write"]} |
| Supabase price_data/indicators 대량 read | {metrics["forbidden_actions_observed"]["supabase_price_data_indicators_bulk_read"]} |
| thin upload | {metrics["forbidden_actions_observed"]["thin_upload"]} |
| checkpoint load | {metrics["forbidden_actions_observed"]["checkpoint_load"]} |
| torch import | {metrics["forbidden_actions_observed"]["torch_import"]} |
| 프론트 수정 | {metrics["forbidden_actions_observed"]["frontend_modify"]} |
| EODHD API call | {metrics["forbidden_actions_observed"]["eodhd_api_call"]} |
| EODHD fallback | {metrics["forbidden_actions_observed"]["eodhd_fallback"]} |

## 7. 실행 명령

```powershell
python scripts\\cp114_data_yfinance_freshness_check.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP114 yfinance local snapshot freshness check")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--end-date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    snapshots = {
        "price_1d": snapshot_summary(
            SNAPSHOT_DIR / "price_data_yfinance.parquet",
            duplicate_keys=["ticker", "date", "source"],
        ),
        "indicators_1d": snapshot_summary(
            SNAPSHOT_DIR / "indicators_yfinance_1D.parquet",
            duplicate_keys=["ticker", "timeframe", "date", "source"],
        ),
        "price_1w": snapshot_summary(
            SNAPSHOT_DIR / "price_data_yfinance_1W.parquet",
            duplicate_keys=["ticker", "date", "source"],
        ),
        "indicators_1w": snapshot_summary(
            SNAPSHOT_DIR / "indicators_yfinance_1W.parquet",
            duplicate_keys=["ticker", "timeframe", "date", "source"],
        ),
    }
    snapshot_latest = pd.to_datetime(snapshots["price_1d"]["date_max"]).date()
    start_date = (snapshot_latest - timedelta(days=args.lookback_days)).isoformat()
    end_date = args.end_date

    ticker_metrics = {
        ticker: fetch_ticker_freshness(ticker, start_date=start_date, end_date=end_date, snapshot_latest=snapshot_latest)
        for ticker in tickers
    }
    total_new_rows = int(sum(item.get("new_rows_after_snapshot_latest", 0) for item in ticker_metrics.values()))
    new_dates_all = sorted(
        {
            new_date
            for item in ticker_metrics.values()
            for new_date in item.get("new_dates", [])
        }
    )
    potential_partial_current_day_rows = int(
        sum(
            1
            for item in ticker_metrics.values()
            for new_date in item.get("new_dates", [])
            if new_date == end_date
        )
    )
    fallback_used_any = bool(any(item.get("fallback_used") for item in ticker_metrics.values()))
    fetch_failed_count = int(sum(1 for item in ticker_metrics.values() if not str(item.get("status")).startswith("PASS")))
    all_adjusted_ohlc_passed = bool(all(item.get("adjusted_ohlc_passed") for item in ticker_metrics.values()))
    torch_imported = "torch" in sys.modules

    if fallback_used_any or fetch_failed_count:
        status = "FAIL"
        summary = "yfinance fetch 실패 또는 fallback 발생으로 EODHD 해지 gate를 보류한다."
        model_continue = "보류 권장: fetch 실패 원인 확인 전 추가 모델 실험을 멈추는 편이 안전하다."
        eodhd_gate = "보류"
    elif total_new_rows <= 0:
        status = "WARN_NO_NEW_MARKET_DAY"
        summary = "yfinance가 snapshot 최신일보다 새로운 거래일 row를 아직 제공하지 않았다. 모델 실험은 기존 snapshot 기준으로 계속 가능하지만 EODHD 해지 최종 append gate는 보류한다."
        model_continue = "가능: 기존 1D/1W local snapshot 기준 실험은 계속 가능하다. 최신 거래일 반영 실험은 append CP 이후 권장한다."
        eodhd_gate = "보류"
    else:
        status = "NEW_ROWS_AVAILABLE"
        summary = "yfinance가 snapshot 최신일보다 새로운 거래일 row를 제공했다. local parquet append와 1D indicator incremental refresh를 별도 CP로 진행할 수 있다."
        model_continue = "조건부 가능: 기존 snapshot 기준 실험은 가능하지만, 최신 거래일 반영은 append/refresh CP 이후 권장한다."
        eodhd_gate = "append/refresh 검증 전까지 보류"
        if potential_partial_current_day_rows:
            summary += " 단, end_date와 같은 날짜의 row가 포함되어 있어 장중 partial daily row 가능성을 append CP에서 걸러야 한다."

    metrics: dict[str, Any] = {
        "cp": "CP114-D",
        "generated_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "provider": "yfinance",
            "source": "yfinance",
            "timeframes_checked": ["1D", "1W"],
        },
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "EODHD_API_KEY_present": bool(os.environ.get("EODHD_API_KEY")),
        },
        "snapshots": snapshots,
        "yfinance_fetch": {
            "request": {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "snapshot_latest_date": snapshot_latest.isoformat(),
                "provider": "yfinance",
                "fallback_provider": None,
            },
            "ticker_metrics": ticker_metrics,
            "total_new_rows_after_snapshot_latest": total_new_rows,
            "new_dates_all": new_dates_all,
            "potential_partial_current_day_rows": potential_partial_current_day_rows,
            "partial_day_warning": (
                "end_date와 같은 신규 row가 있어 미국장 마감 전 partial daily row일 수 있다. append CP에서는 완료 거래일만 반영해야 한다."
                if potential_partial_current_day_rows
                else None
            ),
            "fallback_used_any": fallback_used_any,
            "fetch_failed_count": fetch_failed_count,
            "all_adjusted_ohlc_passed": all_adjusted_ohlc_passed,
        },
        "torch_imported": torch_imported,
        "model_experiment_continuation": model_continue,
        "eodhd_cancellation_gate_status": eodhd_gate,
        "final_decision": {
            "status": status,
            "summary": summary,
        },
        "forbidden_actions_observed": {
            "model_training": False,
            "inference": False,
            "db_write": False,
            "supabase_price_data_indicators_bulk_read": False,
            "thin_upload": False,
            "checkpoint_load": False,
            "torch_import": torch_imported,
            "frontend_modify": False,
            "eodhd_api_call": False,
            "eodhd_fallback": fallback_used_any,
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)

    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
