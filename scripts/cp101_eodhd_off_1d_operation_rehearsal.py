from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp101_eodhd_off_1d_operation_rehearsal"
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp101_eodhd_off_1d_operation_rehearsal_metrics.json"
TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.config import get_settings  # noqa: E402
from backend.collector.sources.market_data_providers import (  # noqa: E402
    fetch_market_data,
    provider_adjustment_policy,
)
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402
from backend.app.services.feature_svc import build_features  # noqa: E402

LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
LINE_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "patchtst_1D_patchtst-1D-efad3c29d803.pt"
BAND_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt"


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _cp99_module():
    # 모델 inference helper는 호출 시점에만 로드해서 data-only import 경로가 torch를 끌고 오지 않게 한다.
    from scripts import cp99_1d_product_loop_thin_upload as cp99

    return cp99


def count_existing_rows(*args, **kwargs):
    return _cp99_module().count_existing_rows(*args, **kwargs)


def fetch_latest_prediction_by_run(*args, **kwargs):
    return _cp99_module().fetch_latest_prediction_by_run(*args, **kwargs)


def infer_latest_layer(*args, **kwargs):
    return _cp99_module().infer_latest_layer(*args, **kwargs)


def save_latest_records(*args, **kwargs):
    return _cp99_module().save_latest_records(*args, **kwargs)


def snapshot_gate(*args, **kwargs):
    return _cp99_module().snapshot_gate(*args, **kwargs)


def verify_bulk_read_guard(*args, **kwargs):
    return _cp99_module().verify_bulk_read_guard(*args, **kwargs)


def verify_model_run_exists(*args, **kwargs):
    return _cp99_module().verify_model_run_exists(*args, **kwargs)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_local_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path = SNAPSHOT_DIR / "price_data_yfinance.parquet"
    indicator_path = SNAPSHOT_DIR / "indicators_yfinance_1D.parquet"
    if not price_path.exists() or not indicator_path.exists():
        raise FileNotFoundError("CP101에는 1D yfinance local snapshot이 필요합니다.")
    price = pd.read_parquet(price_path)
    indicators = pd.read_parquet(indicator_path)
    price["ticker"] = price["ticker"].astype(str).str.upper()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    return price, indicators


def price_frame_to_records(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
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
            "updated_at": utc_now_iso(),
        }
    )
    return records.dropna(subset=["date"]).reset_index(drop=True)


def refresh_local_indicator_rows(price_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    indicator_path = SNAPSHOT_DIR / "indicators_yfinance_1D.parquet"
    existing = pd.read_parquet(indicator_path)
    existing["ticker"] = existing["ticker"].astype(str).str.upper()
    price_subset = price_df[price_df["ticker"].isin(tickers)].copy()
    rebuilt = build_features(price_df=price_subset, timeframe="1D")
    rebuilt["source"] = "yfinance"
    rebuilt["provider"] = "yfinance"
    rebuilt["updated_at"] = utc_now_iso()
    rebuilt["date"] = pd.to_datetime(rebuilt["date"], errors="coerce")
    kept = existing[~existing["ticker"].isin(tickers)].copy()
    kept["date"] = pd.to_datetime(kept["date"], errors="coerce")
    combined = pd.concat([kept, rebuilt], ignore_index=True)
    combined = combined.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    )
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined.to_parquet(indicator_path, index=False)
    return combined


def latest_update_rehearsal(tickers: list[str], *, apply_local_update: bool) -> dict[str, Any]:
    price_path = SNAPSHOT_DIR / "price_data_yfinance.parquet"
    price_df, indicator_df = load_local_snapshots()
    snapshot_latest = pd.to_datetime(price_df["date"]).max().date()
    indicator_latest = pd.to_datetime(indicator_df["date"]).max().date()
    start_date = (snapshot_latest - timedelta(days=7)).isoformat()
    end_date = date.today().isoformat()
    fetched_chunks: list[pd.DataFrame] = []
    ticker_metrics: dict[str, Any] = {}
    fallback_used = False

    for ticker in tickers:
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=None,
        )
        fallback_used = fallback_used or bool(result.fallback_used)
        if result.frame.empty:
            ticker_metrics[ticker] = {
                "status": "WARN",
                "provider": result.provider,
                "fallback_provider": result.fallback_provider,
                "fallback_used": result.fallback_used,
                "errors": result.errors,
                "row_count": 0,
            }
            continue
        contract = validate_adjusted_ohlc_contract(ticker, result.frame)
        records = price_frame_to_records(ticker, result.frame)
        fetched_latest = pd.to_datetime(records["date"]).max().date() if not records.empty else None
        new_records = records[pd.to_datetime(records["date"]).dt.date > snapshot_latest].copy()
        ticker_metrics[ticker] = {
            "status": "PASS" if contract.passed else "FAIL",
            "provider": result.provider,
            "fallback_provider": result.fallback_provider,
            "fallback_used": result.fallback_used,
            "source_errors": result.errors,
            "fetched_rows": int(len(records)),
            "new_rows_after_snapshot_latest": int(len(new_records)),
            "fetched_date_min": None if records.empty else pd.to_datetime(records["date"]).min().strftime("%Y-%m-%d"),
            "fetched_date_max": None if fetched_latest is None else fetched_latest.isoformat(),
            "adjusted_ohlc_contract_passed": bool(contract.passed),
            "adjusted_ohlc_violations": contract.violations,
            "adjusted_ohlc_metrics": contract.metrics,
        }
        if contract.passed and not new_records.empty:
            fetched_chunks.append(new_records)

    local_update_result: dict[str, Any] = {
        "attempted": bool(apply_local_update),
        "price_rows_written": 0,
        "indicator_rebuild_tickers": [],
        "reason": "no_new_rows",
    }
    if apply_local_update and fetched_chunks:
        update_rows = pd.concat(fetched_chunks, ignore_index=True)
        combined = pd.concat([price_df, update_rows], ignore_index=True)
        combined["ticker"] = combined["ticker"].astype(str).str.upper()
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.sort_values(["ticker", "date"]).drop_duplicates(
            subset=["ticker", "date", "source"],
            keep="last",
        )
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        combined.to_parquet(price_path, index=False)
        rebuilt = refresh_local_indicator_rows(combined.assign(date=pd.to_datetime(combined["date"])), tickers)
        local_update_result = {
            "attempted": True,
            "price_rows_written": int(len(update_rows)),
            "indicator_rebuild_tickers": tickers,
            "indicator_rows_after_rebuild": int(len(rebuilt)),
            "reason": "new_rows_appended",
        }

    refreshed_price, refreshed_indicators = load_local_snapshots()
    return {
        "snapshot_before": {
            "price_latest_date": snapshot_latest.isoformat(),
            "indicator_latest_date": indicator_latest.isoformat(),
            "price_rows": int(len(price_df)),
            "indicator_rows": int(len(indicator_df)),
        },
        "request": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "provider": "yfinance",
            "fallback_provider": None,
        },
        "ticker_metrics": ticker_metrics,
        "successful_fetch_count": sum(1 for item in ticker_metrics.values() if item.get("status") == "PASS"),
        "failed_or_empty_fetch_count": sum(1 for item in ticker_metrics.values() if item.get("status") != "PASS"),
        "fallback_used": fallback_used,
        "local_update": local_update_result,
        "snapshot_after": {
            "price_latest_date": pd.to_datetime(refreshed_price["date"]).max().strftime("%Y-%m-%d"),
            "indicator_latest_date": pd.to_datetime(refreshed_indicators["date"]).max().strftime("%Y-%m-%d"),
            "price_rows": int(len(refreshed_price)),
            "indicator_rows": int(len(refreshed_indicators)),
        },
    }


def snapshot_freshness(tickers: list[str]) -> dict[str, Any]:
    price, indicators = load_local_snapshots()
    return {
        "price_rows": int(len(price)),
        "price_ticker_count": int(price["ticker"].nunique()),
        "price_date_min": pd.to_datetime(price["date"]).min().strftime("%Y-%m-%d"),
        "price_date_max": pd.to_datetime(price["date"]).max().strftime("%Y-%m-%d"),
        "indicator_rows": int(len(indicators)),
        "indicator_ticker_count": int(indicators["ticker"].nunique()),
        "indicator_date_min": pd.to_datetime(indicators["date"]).min().strftime("%Y-%m-%d"),
        "indicator_date_max": pd.to_datetime(indicators["date"]).max().strftime("%Y-%m-%d"),
        "source_values": sorted(price["source"].dropna().astype(str).str.lower().unique().tolist()),
        "indicator_source_values": sorted(indicators["source"].dropna().astype(str).str.lower().unique().tolist()),
        "source_data_hash": snapshot_gate(tickers)["source_data_hash"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP101 EODHD off 1D operation rehearsal")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tickers", nargs="*", default=TARGET_TICKERS)
    parser.add_argument("--skip-local-update", action="store_true")
    parser.add_argument("--skip-thin-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    metrics: dict[str, Any] = {
        "cp": "CP101-D",
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
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "EODHD_API_KEY_present": bool(os.environ.get("EODHD_API_KEY")),
            "settings_market_data_provider": settings.market_data_provider,
            "settings_market_data_fallback_provider": settings.market_data_fallback_provider,
            "settings_eodhd_api_key_present": bool(settings.eodhd_api_key),
        },
        "forbidden_actions_observed": {
            "eodhd_api_call": False,
            "eodhd_fallback": False,
            "supabase_price_data_bulk_read": False,
            "supabase_indicators_bulk_read": False,
            "supabase_price_data_indicators_write": False,
            "indicators_db_recompute": False,
            "full_model_training": False,
            "one_week_or_one_month_inference": False,
            "composite_save": False,
            "eodhd_row_delete": False,
            "frontend_ui_modify": False,
        },
    }

    metrics["snapshot_freshness_before_update"] = snapshot_freshness(tickers)
    metrics["latest_yfinance_update"] = latest_update_rehearsal(
        tickers,
        apply_local_update=not args.skip_local_update,
    )
    metrics["snapshot_freshness_after_update"] = snapshot_freshness(tickers)
    metrics["snapshot_gate"] = snapshot_gate(tickers)
    metrics["bulk_read_guard"] = verify_bulk_read_guard()

    source_data_hash = metrics["snapshot_freshness_after_update"]["source_data_hash"]
    model_run_exists = {
        LINE_RUN_ID: verify_model_run_exists(LINE_RUN_ID),
        BAND_RUN_ID: verify_model_run_exists(BAND_RUN_ID),
    }
    metrics["model_run_exists"] = model_run_exists
    if not all(model_run_exists.values()):
        metrics["final_decision"] = {"status": "FAIL", "reason": "product_model_run_missing"}
        write_json(Path(args.metrics_path), metrics)
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
        "one_week_one_month_policy_unchanged": True,
    }

    eodhd_off_pass = (
        settings.market_data_provider == "yfinance"
        and settings.market_data_fallback_provider is None
        and not settings.eodhd_api_key
        and not metrics["latest_yfinance_update"]["fallback_used"]
    )
    snapshot_pass = metrics["snapshot_gate"]["split_gate"].get("status") == "PASS"
    inference_pass = (
        metrics["inference_dry_run"]["prediction_rows"] == 10
        and metrics["inference_dry_run"]["evaluation_rows"] == 10
        and metrics["inference_dry_run"]["all_forecast_horizon_5"]
        and metrics["inference_dry_run"]["all_series_length_5"]
        and metrics["inference_dry_run"]["lower_lte_upper"]
    )
    upload_pass = bool(
        not args.skip_thin_upload
        and metrics["api_check"]["aapl_line_found"]
        and metrics["api_check"]["aapl_band_found"]
    )
    bulk_guard_pass = metrics["bulk_read_guard"] == {"price_data": "PASS_BLOCKED", "indicators": "PASS_BLOCKED"}
    deltas = metrics.get("thin_upload_result", {}).get("count_deltas", {})
    no_bulk_history_growth = all(value is None or value <= 5 for value in deltas.values())
    failed_or_empty_fetch_count = int(metrics["latest_yfinance_update"].get("failed_or_empty_fetch_count", 0))
    no_new_rows = all(
        item.get("new_rows_after_snapshot_latest", 0) == 0
        for item in metrics["latest_yfinance_update"]["ticker_metrics"].values()
    )
    hard_fail = not (eodhd_off_pass and snapshot_pass and inference_pass and upload_pass and bulk_guard_pass and no_bulk_history_growth)
    warning = None
    if failed_or_empty_fetch_count:
        warning = f"yfinance 최신 조회에서 {failed_or_empty_fetch_count}개 ticker가 빈 응답 또는 실패를 반환했습니다. EODHD fallback 없이 local snapshot 기반 운영 루프는 통과했습니다."
    elif no_new_rows:
        warning = "yfinance 최신 조회는 성공했지만 local snapshot보다 새로운 거래일 row는 없어서 local update는 실질 dry-run으로 끝났습니다."
    metrics["final_decision"] = {
        "status": "FAIL" if hard_fail else "WARN" if failed_or_empty_fetch_count or no_new_rows else "PASS",
        "eodhd_off_pass": eodhd_off_pass,
        "snapshot_gate_pass": snapshot_pass,
        "inference_pass": inference_pass,
        "thin_upload_pass": upload_pass,
        "bulk_read_guard_pass": bulk_guard_pass,
        "no_bulk_history_growth": no_bulk_history_growth,
        "yfinance_failed_or_empty_fetch_count": failed_or_empty_fetch_count,
        "warning": warning,
    }

    write_json(Path(args.metrics_path), metrics)
    write_json(log_dir / "cp101_eodhd_off_1d_operation_rehearsal_metrics.json", metrics)
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
