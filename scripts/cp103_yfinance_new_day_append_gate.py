from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cp101_eodhd_off_1d_operation_rehearsal import (
    BAND_CHECKPOINT,
    BAND_RUN_ID,
    LINE_CHECKPOINT,
    LINE_RUN_ID,
    TARGET_TICKERS,
    _json_safe,
    count_existing_rows,
    fetch_latest_prediction_by_run,
    infer_latest_layer,
    latest_update_rehearsal,
    save_latest_records,
    snapshot_freshness,
    snapshot_gate,
    verify_bulk_read_guard,
    verify_model_run_exists,
)
from backend.collector.config import get_settings


DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp103_yfinance_new_day_append_gate"
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp103_yfinance_new_day_append_gate_metrics.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "cp103_yfinance_new_day_append_gate_report.md"


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _status_line(metrics: dict[str, Any]) -> str:
    return str((metrics.get("final_decision") or {}).get("status") or "UNKNOWN")


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    latest_update = metrics.get("latest_yfinance_update") or {}
    before = metrics.get("snapshot_before") or {}
    after = metrics.get("snapshot_after") or {}
    local_update = latest_update.get("local_update") or {}
    inference = metrics.get("inference_dry_run") or {}
    api_check = metrics.get("api_check") or {}
    final_decision = metrics.get("final_decision") or {}
    forbidden = metrics.get("forbidden_actions_observed") or {}

    ticker_lines = []
    for ticker, item in sorted((latest_update.get("ticker_metrics") or {}).items()):
        ticker_lines.append(
            "| {ticker} | {status} | {fetched_rows} | {new_rows} | {date_min} | {date_max} | {fallback} | {contract} |".format(
                ticker=ticker,
                status=item.get("status"),
                fetched_rows=item.get("fetched_rows", item.get("row_count")),
                new_rows=item.get("new_rows_after_snapshot_latest", 0),
                date_min=item.get("fetched_date_min"),
                date_max=item.get("fetched_date_max"),
                fallback=item.get("fallback_used"),
                contract=item.get("adjusted_ohlc_contract_passed"),
            )
        )

    report = f"""# CP103-D yfinance 신규 거래일 append 운영 게이트

생성일: 2026-05-04

## 1. Executive Summary

최종 판정: {_status_line(metrics)}

{final_decision.get("summary", "")}

이번 CP는 EODHD 해지 전 마지막 운영 gate로, yfinance가 local snapshot 최신일보다 새로운 1D row를 주는지 확인하고, 새 row가 있을 때 local parquet append, 1D indicator incremental refresh, 제품 inference, Supabase latest-only thin upload, API 조회까지 이어지는지 검증한다.

## 2. 환경

| 항목 | 값 |
|---|---|
| provider | {metrics.get("scope", {}).get("provider")} |
| source | {metrics.get("scope", {}).get("source")} |
| timeframe | {metrics.get("scope", {}).get("timeframe")} |
| horizon | {metrics.get("scope", {}).get("horizon")} |
| tickers | {", ".join(metrics.get("scope", {}).get("tickers", []))} |
| line run | {metrics.get("scope", {}).get("line_run_id")} |
| band run | {metrics.get("scope", {}).get("band_run_id")} |
| fallback provider | {metrics.get("environment", {}).get("settings_market_data_fallback_provider")} |
| EODHD key present | {metrics.get("environment", {}).get("settings_eodhd_api_key_present")} |

## 3. Snapshot 현재 상태

| 항목 | Before | After |
|---|---:|---:|
| price latest date | {before.get("price_date_max")} | {after.get("price_date_max")} |
| indicator latest date | {before.get("indicator_date_max")} | {after.get("indicator_date_max")} |
| price rows | {before.get("price_rows")} | {after.get("price_rows")} |
| indicator rows | {before.get("indicator_rows")} | {after.get("indicator_rows")} |
| price ticker count | {before.get("price_ticker_count")} | {after.get("price_ticker_count")} |
| indicator ticker count | {before.get("indicator_ticker_count")} | {after.get("indicator_ticker_count")} |
| source_data_hash | {before.get("source_data_hash")} | {after.get("source_data_hash")} |

## 4. yfinance 최신 조회

요청:
- start_date: {latest_update.get("request", {}).get("start_date")}
- end_date: {latest_update.get("request", {}).get("end_date")}
- fallback_used: {latest_update.get("fallback_used")}
- successful_fetch_count: {latest_update.get("successful_fetch_count")}
- failed_or_empty_fetch_count: {latest_update.get("failed_or_empty_fetch_count")}

| ticker | status | fetched rows | new rows | fetched min | fetched max | fallback | adjusted contract |
|---|---:|---:|---:|---|---|---:|---:|
{chr(10).join(ticker_lines)}

## 5. Local Parquet Append

| 항목 | 값 |
|---|---:|
| attempted | {local_update.get("attempted")} |
| price_rows_written | {local_update.get("price_rows_written")} |
| reason | {local_update.get("reason")} |
| indicator_rebuild_tickers | {", ".join(local_update.get("indicator_rebuild_tickers") or [])} |
| indicator_rows_after_rebuild | {local_update.get("indicator_rows_after_rebuild")} |

신규 row가 없으면 이 CP는 `WARN_NO_NEW_MARKET_DAY`로 종료한다. 이는 구조 문제라기보다 Yahoo Finance가 아직 snapshot 최신일보다 새 거래일 row를 제공하지 않았다는 의미다.

## 6. Indicator Refresh 및 Feature Gate

| 항목 | 값 |
|---|---:|
| snapshot_gate_status | {(metrics.get("snapshot_gate") or {}).get("split_gate", {}).get("status")} |
| MODEL_N_FEATURES | {(metrics.get("snapshot_gate") or {}).get("MODEL_N_FEATURES")} |
| atr_ratio_in_MODEL_FEATURE_COLUMNS | {(metrics.get("snapshot_gate") or {}).get("atr_ratio_in_MODEL_FEATURE_COLUMNS")} |
| feature_non_finite_count | {(metrics.get("snapshot_gate") or {}).get("split_gate", {}).get("feature_non_finite_count")} |
| target_non_finite_count | {(metrics.get("snapshot_gate") or {}).get("split_gate", {}).get("target_non_finite_count")} |

## 7. 제품 Inference 및 Thin Upload

| 항목 | 값 |
|---|---:|
| inference attempted | {bool(inference)} |
| prediction_rows | {inference.get("prediction_rows")} |
| evaluation_rows | {inference.get("evaluation_rows")} |
| all_forecast_horizon_5 | {inference.get("all_forecast_horizon_5")} |
| all_series_length_5 | {inference.get("all_series_length_5")} |
| lower_lte_upper | {inference.get("lower_lte_upper")} |
| thin_upload_attempted | {(metrics.get("thin_upload_result") or {}).get("attempted")} |
| prediction_rows_written | {(metrics.get("thin_upload_result") or {}).get("prediction_rows_written")} |
| evaluation_rows_written | {(metrics.get("thin_upload_result") or {}).get("evaluation_rows_written")} |

## 8. API 확인

| 항목 | 값 |
|---|---|
| attempted | {api_check.get("attempted")} |
| AAPL line found | {api_check.get("aapl_line_found")} |
| AAPL band found | {api_check.get("aapl_band_found")} |
| line asof_date | {api_check.get("line_asof_date")} |
| band asof_date | {api_check.get("band_asof_date")} |
| expected latest asof_date | {api_check.get("expected_latest_asof_date")} |
| asof_date matched | {api_check.get("asof_date_matched")} |

## 9. 금지 항목 확인

| 항목 | 발생 |
|---|---:|
| EODHD API call | {forbidden.get("eodhd_api_call")} |
| EODHD fallback | {forbidden.get("eodhd_fallback")} |
| Supabase price_data/indicators bulk read | {forbidden.get("supabase_price_data_indicators_bulk_read")} |
| Supabase price_data/indicators write | {forbidden.get("supabase_price_data_indicators_write")} |
| full model training | {forbidden.get("full_model_training")} |
| 1W/1M processing | {forbidden.get("one_week_or_one_month_processing")} |
| composite save | {forbidden.get("composite_save")} |
| DB row delete | {forbidden.get("db_row_delete")} |
| cache/checkpoint delete | {forbidden.get("cache_checkpoint_delete")} |

## 10. 해지 전 판단

- PASS이면 EODHD 해지 가능 후보로 본다.
- WARN_NO_NEW_MARKET_DAY이면 해지 판정은 보류하되 구조 문제는 아니다. 다음 거래일 데이터가 yfinance에 반영된 뒤 같은 명령을 재실행한다.
- FAIL이면 EODHD 해지 보류다.

현재 판정: {_status_line(metrics)}

## 11. 실행 명령

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER=""
$env:LENS_DATA_BACKEND="local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\\Users\\user\\lens\\data\\parquet"
python scripts\\cp103_yfinance_new_day_append_gate.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _total_new_rows(latest_update: dict[str, Any]) -> int:
    return int(
        sum(
            int(item.get("new_rows_after_snapshot_latest") or 0)
            for item in (latest_update.get("ticker_metrics") or {}).values()
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP103 yfinance 신규 거래일 append 운영 게이트")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tickers", nargs="*", default=TARGET_TICKERS)
    parser.add_argument("--skip-thin-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    metrics: dict[str, Any] = {
        "cp": "CP103-D",
        "generated_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "timeframe": "1D",
            "horizon": 5,
            "provider": "yfinance",
            "source": "yfinance",
            "composite_saved": False,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": "yfinance",
            "MARKET_DATA_FALLBACK_PROVIDER": "",
            "LENS_DATA_BACKEND": "local",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(ROOT_DIR / "data" / "parquet"),
            "settings_market_data_provider": settings.market_data_provider,
            "settings_market_data_fallback_provider": settings.market_data_fallback_provider,
            "settings_eodhd_api_key_present": bool(settings.eodhd_api_key),
        },
        "forbidden_actions_observed": {
            "eodhd_api_call": False,
            "eodhd_fallback": False,
            "supabase_price_data_indicators_bulk_read": False,
            "supabase_price_data_indicators_write": False,
            "full_model_training": False,
            "one_week_or_one_month_processing": False,
            "composite_save": False,
            "db_row_delete": False,
            "cache_checkpoint_delete": False,
        },
    }

    metrics["snapshot_before"] = snapshot_freshness(tickers)
    latest_update = latest_update_rehearsal(tickers, apply_local_update=True)
    metrics["latest_yfinance_update"] = latest_update
    metrics["snapshot_after"] = snapshot_freshness(tickers)
    metrics["new_rows_after_snapshot_latest"] = _total_new_rows(latest_update)

    if latest_update.get("fallback_used"):
        metrics["forbidden_actions_observed"]["eodhd_fallback"] = True
        metrics["final_decision"] = {
            "status": "FAIL",
            "summary": "yfinance 최신 조회 중 fallback이 발생했다. EODHD 해지 전 gate 실패다.",
        }
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        raise SystemExit(1)

    if metrics["new_rows_after_snapshot_latest"] <= 0:
        metrics["final_decision"] = {
            "status": "WARN_NO_NEW_MARKET_DAY",
            "summary": "yfinance가 snapshot 최신일보다 새로운 거래일 row를 아직 제공하지 않았다. local append와 indicator incremental refresh는 다음 거래일 데이터 반영 후 재검증해야 한다.",
        }
        metrics["snapshot_gate"] = snapshot_gate(tickers)
        metrics["bulk_read_guard"] = verify_bulk_read_guard()
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        return

    metrics["snapshot_gate"] = snapshot_gate(tickers)
    metrics["bulk_read_guard"] = verify_bulk_read_guard()

    model_run_exists = {
        LINE_RUN_ID: verify_model_run_exists(LINE_RUN_ID),
        BAND_RUN_ID: verify_model_run_exists(BAND_RUN_ID),
    }
    metrics["model_run_exists"] = model_run_exists
    if not all(model_run_exists.values()):
        metrics["final_decision"] = {
            "status": "FAIL",
            "summary": "제품 model_run을 찾지 못했다.",
        }
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        raise SystemExit(1)

    source_data_hash = metrics["snapshot_after"]["source_data_hash"]
    line_predictions, line_evaluations, line_diag = infer_latest_layer(
        layer="line",
        run_id=LINE_RUN_ID,
        checkpoint_path=LINE_CHECKPOINT,
        tickers=tickers,
        source_data_hash=source_data_hash,
    )
    band_predictions, band_evaluations, band_diag = infer_latest_layer(
        layer="band",
        run_id=BAND_RUN_ID,
        checkpoint_path=BAND_CHECKPOINT,
        tickers=tickers,
        source_data_hash=source_data_hash,
    )
    predictions = line_predictions + band_predictions
    evaluations = line_evaluations + band_evaluations
    metrics["inference_dry_run"] = {
        "line": line_diag,
        "band": band_diag,
        "prediction_rows": len(predictions),
        "evaluation_rows": len(evaluations),
        "all_forecast_horizon_5": all(len(record["forecast_dates"]) == 5 for record in predictions),
        "all_series_length_5": all(
            len(record["line_series"]) == 5
            and len(record["conservative_series"]) == 5
            and len(record["lower_band_series"]) == 5
            and len(record["upper_band_series"]) == 5
            for record in predictions
        ),
        "lower_lte_upper": line_diag["lower_lte_upper"] and band_diag["lower_lte_upper"],
        "composite_rows": 0,
        "payload_bytes": len(
            json.dumps(_json_safe({"predictions": predictions, "evaluations": evaluations}), ensure_ascii=False).encode(
                "utf-8"
            )
        ),
    }
    write_json(log_dir / "dry_run_predictions.json", {"predictions": predictions, "evaluations": evaluations})

    before_counts = {
        "line_predictions": count_existing_rows("predictions", LINE_RUN_ID, tickers),
        "band_predictions": count_existing_rows("predictions", BAND_RUN_ID, tickers),
        "line_evaluations": count_existing_rows("prediction_evaluations", LINE_RUN_ID, tickers),
        "band_evaluations": count_existing_rows("prediction_evaluations", BAND_RUN_ID, tickers),
    }
    metrics["thin_upload_plan"] = {
        "skip_thin_upload": bool(args.skip_thin_upload),
        "target_prediction_rows": len(predictions),
        "target_evaluation_rows": len(evaluations),
        "latest_only": True,
        "composite_rows": 0,
        "before_counts": before_counts,
    }

    if args.skip_thin_upload:
        metrics["thin_upload_result"] = {"attempted": False}
    else:
        save_latest_records(predictions, evaluations)
        after_counts = {
            "line_predictions": count_existing_rows("predictions", LINE_RUN_ID, tickers),
            "band_predictions": count_existing_rows("predictions", BAND_RUN_ID, tickers),
            "line_evaluations": count_existing_rows("prediction_evaluations", LINE_RUN_ID, tickers),
            "band_evaluations": count_existing_rows("prediction_evaluations", BAND_RUN_ID, tickers),
        }
        metrics["thin_upload_result"] = {
            "attempted": True,
            "prediction_rows_written": len(predictions),
            "evaluation_rows_written": len(evaluations),
            "after_counts": after_counts,
            "count_deltas": {
                key: (after_counts[key] - before_counts[key])
                if after_counts[key] is not None and before_counts[key] is not None
                else None
                for key in before_counts
            },
        }

    expected_latest = metrics["snapshot_after"]["indicator_date_max"]
    api_line = fetch_latest_prediction_by_run("AAPL", LINE_RUN_ID) if not args.skip_thin_upload else None
    api_band = fetch_latest_prediction_by_run("AAPL", BAND_RUN_ID) if not args.skip_thin_upload else None
    metrics["api_check"] = {
        "attempted": not args.skip_thin_upload,
        "aapl_line_found": api_line is not None,
        "aapl_band_found": api_band is not None,
        "line_run_id": api_line.get("run_id") if api_line else None,
        "band_run_id": api_band.get("run_id") if api_band else None,
        "line_asof_date": api_line.get("asof_date") if api_line else None,
        "band_asof_date": api_band.get("asof_date") if api_band else None,
        "line_meta_layer": (api_line.get("meta") or {}).get("layer") if api_line else None,
        "band_meta_layer": (api_band.get("meta") or {}).get("layer") if api_band else None,
        "expected_latest_asof_date": expected_latest,
        "asof_date_matched": bool(
            api_line
            and api_band
            and api_line.get("asof_date") == expected_latest
            and api_band.get("asof_date") == expected_latest
        ),
    }

    pass_conditions = [
        settings.market_data_provider == "yfinance",
        settings.market_data_fallback_provider is None,
        not settings.eodhd_api_key,
        not latest_update.get("fallback_used"),
        metrics["new_rows_after_snapshot_latest"] > 0,
        (latest_update.get("local_update") or {}).get("price_rows_written", 0) > 0,
        metrics["snapshot_gate"]["split_gate"].get("status") == "PASS",
        metrics["bulk_read_guard"].get("price_data") == "PASS_BLOCKED",
        metrics["bulk_read_guard"].get("indicators") == "PASS_BLOCKED",
        metrics["inference_dry_run"]["prediction_rows"] == 10,
        metrics["inference_dry_run"]["evaluation_rows"] == 10,
        metrics["inference_dry_run"]["lower_lte_upper"],
        metrics["thin_upload_result"].get("attempted") is True,
        metrics["api_check"].get("asof_date_matched") is True,
    ]
    metrics["final_decision"] = {
        "status": "PASS" if all(pass_conditions) else "FAIL",
        "summary": "신규 거래일 yfinance row append부터 latest-only thin upload/API 조회까지 운영 gate를 통과했다."
        if all(pass_conditions)
        else "신규 거래일 append gate 중 하나 이상의 조건이 실패했다.",
        "pass_conditions": pass_conditions,
    }

    write_json(log_dir / "run_summary.json", metrics)
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    if metrics["final_decision"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
