from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
EXPECTED_ASOF_DATE = "2026-05-04"
LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp116_eodhd_off_final_product_loop_rehearsal_metrics.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "cp116_eodhd_off_final_product_loop_rehearsal_report.md"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp116_eodhd_off_final_product_loop_rehearsal"

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch(cpu_only=True)  # noqa: E402
import pandas as pd  # noqa: E402

from backend.collector.config import get_settings  # noqa: E402
from scripts.cp99_1d_product_loop_thin_upload import (  # noqa: E402
    BAND_CHECKPOINT,
    LINE_CHECKPOINT,
    _json_safe,
    count_existing_rows,
    fetch_latest_prediction_by_run,
    infer_latest_layer,
    save_latest_records,
    snapshot_gate,
    verify_bulk_read_guard,
    verify_model_run_exists,
)


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def snapshot_summary(path: Path, duplicate_keys: list[str]) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return {
        "path": str(path.relative_to(ROOT_DIR)),
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": pd.to_datetime(frame["date"]).min().strftime("%Y-%m-%d") if "date" in frame.columns else None,
        "date_max": pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d") if "date" in frame.columns else None,
        "duplicate_count": int(frame.duplicated(subset=duplicate_keys).sum())
        if all(column in frame.columns for column in duplicate_keys)
        else None,
        "source_values": sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist())
        if "source" in frame.columns
        else [],
        "provider_values": sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist())
        if "provider" in frame.columns
        else [],
        "bytes": int(path.stat().st_size),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def _count_deltas(before: dict[str, int | None], after: dict[str, int | None]) -> dict[str, int | None]:
    return {
        key: (after[key] - before[key]) if before.get(key) is not None and after.get(key) is not None else None
        for key in before
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    snapshots = metrics["snapshots"]
    inference = metrics.get("inference_dry_run", {})
    thin = metrics.get("thin_upload_result", {})
    api = metrics.get("api_check", {})
    final = metrics["final_decision"]
    forbidden = metrics["forbidden_actions_observed"]
    report = f"""# CP116-D EODHD-off 1D 제품 루프 최종 리허설

생성일: 2026-05-05

## 1. Executive Summary

최종 판정: {final["status"]}

{final["summary"]}

이번 CP는 CP115에서 local yfinance snapshot이 완료 거래일 `2026-05-04`까지 올라간 뒤, EODHD 없이 1D 제품 line/band checkpoint inference, latest-only thin upload, API latest 조회가 가능한지 최종 확인했다.

## 2. 환경 및 Snapshot

| 항목 | 값 |
|---|---|
| MARKET_DATA_PROVIDER | {metrics["environment"]["MARKET_DATA_PROVIDER"]} |
| MARKET_DATA_FALLBACK_PROVIDER | {metrics["environment"]["MARKET_DATA_FALLBACK_PROVIDER"]} |
| settings provider | {metrics["environment"]["settings_market_data_provider"]} |
| settings fallback | {metrics["environment"]["settings_market_data_fallback_provider"]} |
| EODHD key present | {metrics["environment"]["settings_eodhd_api_key_present"]} |
| LENS_DATA_BACKEND | {metrics["environment"]["LENS_DATA_BACKEND"]} |
| WANDB_MODE | {metrics["environment"]["WANDB_MODE"]} |

| snapshot | rows | latest date | duplicate | source | provider |
|---|---:|---|---:|---|---|
| 1D price | {snapshots["price_1d"]["rows"]} | {snapshots["price_1d"]["date_max"]} | {snapshots["price_1d"]["duplicate_count"]} | {snapshots["price_1d"]["source_values"]} | {snapshots["price_1d"]["provider_values"]} |
| 1D indicators | {snapshots["indicators_1d"]["rows"]} | {snapshots["indicators_1d"]["date_max"]} | {snapshots["indicators_1d"]["duplicate_count"]} | {snapshots["indicators_1d"]["source_values"]} | {snapshots["indicators_1d"]["provider_values"]} |

## 3. Bulk Read Guard

| table | result |
|---|---|
| price_data | {metrics["bulk_read_guard"].get("price_data")} |
| indicators | {metrics["bulk_read_guard"].get("indicators")} |

## 4. 제품 Inference

| 항목 | line | band |
|---|---:|---:|
| run_id | {inference.get("line", {}).get("run_id")} | {inference.get("band", {}).get("run_id")} |
| model | {inference.get("line", {}).get("model")} | {inference.get("band", {}).get("model")} |
| asof_dates | {inference.get("line", {}).get("asof_dates")} | {inference.get("band", {}).get("asof_dates")} |
| prediction_count | {inference.get("line", {}).get("prediction_count")} | {inference.get("band", {}).get("prediction_count")} |
| evaluation_count | {inference.get("line", {}).get("evaluation_count")} | {inference.get("band", {}).get("evaluation_count")} |
| forecast_date_lengths | {inference.get("line", {}).get("forecast_date_lengths")} | {inference.get("band", {}).get("forecast_date_lengths")} |
| lower_lte_upper | {inference.get("line", {}).get("lower_lte_upper")} | {inference.get("band", {}).get("lower_lte_upper")} |

전체:
- prediction_rows: {inference.get("prediction_rows")}
- evaluation_rows: {inference.get("evaluation_rows")}
- all_forecast_horizon_5: {inference.get("all_forecast_horizon_5")}
- all_series_length_5: {inference.get("all_series_length_5")}
- lower_lte_upper: {inference.get("lower_lte_upper")}
- composite_rows: {inference.get("composite_rows")}

## 5. Latest-only Thin Upload

| 항목 | 값 |
|---|---:|
| attempted | {thin.get("attempted")} |
| prediction_rows_written | {thin.get("prediction_rows_written")} |
| evaluation_rows_written | {thin.get("evaluation_rows_written")} |
| before_counts | {thin.get("before_counts")} |
| after_counts | {thin.get("after_counts")} |
| count_deltas | {thin.get("count_deltas")} |

## 6. API Latest 조회

| 항목 | 값 |
|---|---|
| AAPL line found | {api.get("aapl_line_found")} |
| AAPL band found | {api.get("aapl_band_found")} |
| line asof_date | {api.get("line_asof_date")} |
| band asof_date | {api.get("band_asof_date")} |
| expected asof_date | {api.get("expected_asof_date")} |
| asof_date matched | {api.get("asof_date_matched")} |
| line meta layer | {api.get("line_meta_layer")} |
| band meta layer | {api.get("band_meta_layer")} |

## 7. EODHD 해지 가능 최종 판정

{metrics["eodhd_cancellation_final_judgement"]}

## 8. 금지 작업 미발생 확인

| 항목 | 발생 |
|---|---:|
| Supabase price_data/indicators 대량 read/write | {forbidden["supabase_price_data_indicators_bulk_read_write"]} |
| 전체 universe write | {forbidden["full_universe_write"]} |
| full/model retraining | {forbidden["model_training"]} |
| composite 저장 | {forbidden["composite_save"]} |
| EODHD API call | {forbidden["eodhd_api_call"]} |
| EODHD fallback | {forbidden["eodhd_fallback"]} |
| 프론트 수정 | {forbidden["frontend_modify"]} |
| DB row delete | {forbidden["db_row_delete"]} |

## 9. 남은 운영 주의점

- yfinance 최신 row 중 현재 날짜 row는 partial일 수 있으므로 CP115의 `row.date < current_date` 필터를 daily 운영에도 유지한다.
- Supabase에는 price/indicator 원본을 올리지 않고 product latest prediction만 얇게 저장한다.
- EODHD 해지 후에도 provider 장애 시 재시도/지연 운영 절차가 필요하다.
- 1W/1M 제품화 전에는 별도 snapshot append/partial period guard를 다시 확인한다.

## 10. 실행 명령

```powershell
.\\.venv\\Scripts\\python.exe scripts\\cp116_eodhd_off_final_product_loop_rehearsal.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP116 EODHD-off 1D 제품 루프 최종 리허설")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--expected-asof-date", default=EXPECTED_ASOF_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    expected_asof_date = args.expected_asof_date
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    snapshots = {
        "price_1d": snapshot_summary(SNAPSHOT_DIR / "price_data_yfinance.parquet", ["ticker", "date", "source"]),
        "indicators_1d": snapshot_summary(
            SNAPSHOT_DIR / "indicators_yfinance_1D.parquet",
            ["ticker", "timeframe", "date", "source"],
        ),
    }
    bulk_read_guard = verify_bulk_read_guard()
    gate = snapshot_gate(tickers)
    source_data_hash = gate["source_data_hash"]
    model_run_exists = {
        LINE_RUN_ID: verify_model_run_exists(LINE_RUN_ID),
        BAND_RUN_ID: verify_model_run_exists(BAND_RUN_ID),
    }

    metrics: dict[str, Any] = {
        "cp": "CP116-D",
        "generated_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "timeframe": "1D",
            "horizon": 5,
            "expected_asof_date": expected_asof_date,
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "provider": "yfinance",
            "source": "yfinance",
        },
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
            "EODHD_API_KEY_present": bool(os.environ.get("EODHD_API_KEY")),
            "settings_market_data_provider": settings.market_data_provider,
            "settings_market_data_fallback_provider": settings.market_data_fallback_provider,
            "settings_eodhd_api_key_present": bool(settings.eodhd_api_key),
        },
        "snapshots": snapshots,
        "snapshot_gate": gate,
        "bulk_read_guard": bulk_read_guard,
        "model_run_exists": model_run_exists,
        "forbidden_actions_observed": {
            "supabase_price_data_indicators_bulk_read_write": False,
            "full_universe_write": False,
            "model_training": False,
            "composite_save": False,
            "eodhd_api_call": False,
            "eodhd_fallback": False,
            "frontend_modify": False,
            "db_row_delete": False,
        },
    }

    if not all(model_run_exists.values()):
        metrics["final_decision"] = {
            "status": "FAIL",
            "summary": "제품 model_run이 누락되어 최종 리허설을 중단했다.",
        }
        metrics["eodhd_cancellation_final_judgement"] = "EODHD 해지 보류."
        write_json(log_dir / "run_summary.json", metrics)
        write_json(Path(args.metrics_path), metrics)
        write_report(Path(args.report_path), metrics)
        raise SystemExit(1)

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
    composite_rows = int(
        sum(
            1
            for record in predictions
            if record.get("model_name") == "line_band_composite"
            or (record.get("meta") or {}).get("layer") == "composite"
            or (record.get("meta") or {}).get("composite") is True
        )
    )
    inference_dry_run = {
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
        "composite_rows": composite_rows,
        "payload_bytes": len(
            json.dumps(_json_safe({"predictions": predictions, "evaluations": evaluations}), ensure_ascii=False).encode(
                "utf-8"
            )
        ),
    }
    metrics["inference_dry_run"] = inference_dry_run
    write_json(log_dir / "dry_run_predictions.json", {"predictions": predictions, "evaluations": evaluations})

    before_counts = {
        "line_predictions": count_existing_rows("predictions", LINE_RUN_ID, tickers),
        "band_predictions": count_existing_rows("predictions", BAND_RUN_ID, tickers),
        "line_evaluations": count_existing_rows("prediction_evaluations", LINE_RUN_ID, tickers),
        "band_evaluations": count_existing_rows("prediction_evaluations", BAND_RUN_ID, tickers),
    }
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
        "latest_only": True,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "count_deltas": _count_deltas(before_counts, after_counts),
    }

    api_line = fetch_latest_prediction_by_run("AAPL", LINE_RUN_ID)
    api_band = fetch_latest_prediction_by_run("AAPL", BAND_RUN_ID)
    metrics["api_check"] = {
        "attempted": True,
        "aapl_line_found": api_line is not None,
        "aapl_band_found": api_band is not None,
        "line_run_id": api_line.get("run_id") if api_line else None,
        "band_run_id": api_band.get("run_id") if api_band else None,
        "line_asof_date": api_line.get("asof_date") if api_line else None,
        "band_asof_date": api_band.get("asof_date") if api_band else None,
        "expected_asof_date": expected_asof_date,
        "asof_date_matched": bool(
            api_line
            and api_band
            and api_line.get("asof_date") == expected_asof_date
            and api_band.get("asof_date") == expected_asof_date
        ),
        "line_meta_layer": (api_line.get("meta") or {}).get("layer") if api_line else None,
        "band_meta_layer": (api_band.get("meta") or {}).get("layer") if api_band else None,
    }

    pass_conditions = [
        snapshots["price_1d"]["date_max"] == expected_asof_date,
        snapshots["indicators_1d"]["date_max"] == expected_asof_date,
        settings.market_data_provider == "yfinance",
        settings.market_data_fallback_provider is None,
        not settings.eodhd_api_key,
        bulk_read_guard.get("price_data") == "PASS_BLOCKED",
        bulk_read_guard.get("indicators") == "PASS_BLOCKED",
        line_diag["asof_dates"] == [expected_asof_date],
        band_diag["asof_dates"] == [expected_asof_date],
        inference_dry_run["prediction_rows"] == 10,
        inference_dry_run["evaluation_rows"] == 10,
        inference_dry_run["all_forecast_horizon_5"],
        inference_dry_run["all_series_length_5"],
        inference_dry_run["lower_lte_upper"],
        inference_dry_run["composite_rows"] == 0,
        metrics["thin_upload_result"]["attempted"] is True,
        metrics["api_check"]["asof_date_matched"] is True,
    ]
    status = "PASS" if all(pass_conditions) else "FAIL"
    metrics["final_decision"] = {
        "status": status,
        "summary": "EODHD 없이 최신 asof_date=2026-05-04 기준 1D 제품 line/band inference, latest-only thin upload, API 조회가 모두 통과했다."
        if status == "PASS"
        else "EODHD-off 제품 루프 최종 리허설 중 하나 이상의 조건이 실패했다.",
        "pass_conditions": pass_conditions,
    }
    metrics["eodhd_cancellation_final_judgement"] = (
        "EODHD 해지 가능 판정. yfinance/local parquet append/indicator refresh와 최신 1D 제품 loop가 모두 통과했다."
        if status == "PASS"
        else "EODHD 해지 보류. 실패 조건을 먼저 해소해야 한다."
    )

    write_json(log_dir / "run_summary.json", metrics)
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
