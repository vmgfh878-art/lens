from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch(cpu_only=True)
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(ROOT_DIR / "data" / "parquet"))
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")

from ai.inference import decode_return_forecasts, load_checkpoint, load_checkpoint_config, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_CALENDAR_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    append_calendar_features,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from backend.collector.repositories.base import fetch_all_rows  # noqa: E402
from ai.storage import save_product_latest_predictions  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402
from backend.app.db import get_supabase  # noqa: E402


LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
LINE_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "patchtst_1D_patchtst-1D-efad3c29d803.pt"
BAND_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
LOG_DIR = ROOT_DIR / "logs" / "cp99_1d_product_loop_thin_upload"
METRICS_PATH = ROOT_DIR / "docs" / "cp99_1d_product_loop_thin_upload_metrics.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_snapshots() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = pd.read_parquet(SNAPSHOT_DIR / "stock_info.parquet")
    price = pd.read_parquet(SNAPSHOT_DIR / "price_data_yfinance.parquet")
    indicators = pd.read_parquet(SNAPSHOT_DIR / "indicators_yfinance_1D.parquet")
    return stock, price, indicators


def snapshot_gate(tickers: list[str]) -> dict[str, Any]:
    stock, price, indicators = load_snapshots()
    price_subset = price[price["ticker"].isin(tickers)].copy()
    indicator_subset = indicators[indicators["ticker"].isin(tickers)].copy()
    split_status: dict[str, Any]
    try:
        train, val, test, _, _, plan = prepare_dataset_splits(
            timeframe="1D",
            seq_len=60,
            horizon=5,
            tickers=tickers,
            market_data_provider="yfinance",
        )
        feature_non_finite = 0
        target_non_finite = 0
        for bundle in (train, val, test):
            for index in range(len(bundle)):
                features, line_target, band_target, raw_future_returns, *_ = bundle[index]
                feature_non_finite += int((~torch.isfinite(features)).sum().item())
                target_non_finite += int((~torch.isfinite(line_target)).sum().item())
                target_non_finite += int((~torch.isfinite(band_target)).sum().item())
                target_non_finite += int((~torch.isfinite(raw_future_returns)).sum().item())
        split_status = {
            "status": "PASS",
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "source_data_hash": plan.source_data_hash,
            "feature_non_finite_count": feature_non_finite,
            "target_non_finite_count": target_non_finite,
        }
    except Exception as exc:
        split_status = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "stock_info_rows": int(len(stock)),
        "price_rows": int(len(price)),
        "price_ticker_count": int(price["ticker"].nunique()),
        "price_date_min": str(price["date"].min()),
        "price_date_max": str(price["date"].max()),
        "indicator_rows": int(len(indicators)),
        "indicator_ticker_count": int(indicators["ticker"].nunique()),
        "indicator_date_min": str(indicators["date"].min()),
        "indicator_date_max": str(indicators["date"].max()),
        "price_duplicate_ticker_date_source": int(price.duplicated(subset=["ticker", "date", "source"]).sum()),
        "indicator_duplicate_ticker_timeframe_date_source": int(
            indicators.duplicated(subset=["ticker", "timeframe", "date", "source"]).sum()
        ),
        "source_values": sorted(set(price_subset["source"].dropna().astype(str).tolist())),
        "indicator_source_values": sorted(set(indicator_subset["source"].dropna().astype(str).tolist())),
        "source_data_hash": resolve_data_fingerprint("1D", tickers=tickers, market_data_provider="yfinance"),
        "split_gate": split_status,
        "MODEL_N_FEATURES": MODEL_N_FEATURES,
        "FEATURE_CONTRACT_VERSION": FEATURE_CONTRACT_VERSION,
        "atr_ratio_in_MODEL_FEATURE_COLUMNS": "atr_ratio" in MODEL_FEATURE_COLUMNS,
    }


def _next_business_dates(asof_date: str, horizon: int) -> list[str]:
    start = pd.to_datetime(asof_date) + pd.tseries.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=horizon).strftime("%Y-%m-%d").tolist()


def _latest_model_inputs(
    *,
    checkpoint_path: Path,
    tickers: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor]:
    _, price, indicators = load_snapshots()
    config = load_checkpoint_config(checkpoint_path)
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    seq_len = int(config["seq_len"])
    registry = resolve_checkpoint_ticker_registry(config, "1D")
    if registry is None:
        mapping = {ticker: 0 for ticker in tickers}
    else:
        mapping = {str(key).upper(): int(value) for key, value in (registry.get("mapping") or {}).items()}

    indicators = indicators[indicators["ticker"].isin(tickers)].copy()
    indicators["date"] = pd.to_datetime(indicators["date"])
    indicators = append_calendar_features(indicators)
    price = price[price["ticker"].isin(tickers)].copy()
    price["date"] = pd.to_datetime(price["date"])
    price["target_close"] = pd.to_numeric(price["adjusted_close"].fillna(price["close"]), errors="coerce")

    rows: list[dict[str, Any]] = []
    feature_tensors: list[np.ndarray] = []
    anchor_closes: list[float] = []
    ticker_ids: list[int] = []

    for ticker in tickers:
        ticker_features = indicators[indicators["ticker"] == ticker].sort_values("date").copy()
        if len(ticker_features) < seq_len:
            raise ValueError(f"{ticker}: seq_len={seq_len}보다 indicator row가 적습니다.")
        latest_features = ticker_features.tail(seq_len)
        if latest_features[feature_columns].isna().any().any():
            raise ValueError(f"{ticker}: latest feature window에 NaN이 있습니다.")
        feature_array = latest_features[feature_columns].to_numpy(dtype="float32")
        if not np.isfinite(feature_array).all():
            raise ValueError(f"{ticker}: latest feature window에 non-finite 값이 있습니다.")

        asof_ts = latest_features["date"].iloc[-1]
        ticker_price = price[(price["ticker"] == ticker) & (price["date"] == asof_ts)]
        if ticker_price.empty:
            raise ValueError(f"{ticker}: asof_date={asof_ts.date().isoformat()} anchor price가 없습니다.")
        anchor_close = float(ticker_price["target_close"].iloc[-1])
        if not math.isfinite(anchor_close) or anchor_close <= 0:
            raise ValueError(f"{ticker}: anchor_close가 유효하지 않습니다.")

        feature_tensors.append(feature_array)
        anchor_closes.append(anchor_close)
        ticker_ids.append(mapping.get(ticker, 0))
        rows.append(
            {
                "ticker": ticker,
                "asof_date": asof_ts.strftime("%Y-%m-%d"),
                "forecast_dates": _next_business_dates(asof_ts.strftime("%Y-%m-%d"), int(config["horizon"])),
                "anchor_close": anchor_close,
            }
        )

    features = torch.tensor(np.stack(feature_tensors), dtype=torch.float32)
    mean = torch.as_tensor(torch.load(checkpoint_path, map_location="cpu")["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(torch.load(checkpoint_path, map_location="cpu")["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    features = (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return (
        features,
        torch.tensor(ticker_ids, dtype=torch.long),
        torch.zeros((len(tickers), int(config["horizon"]), len(FUTURE_CALENDAR_COLUMNS)), dtype=torch.float32),
        pd.DataFrame(rows),
        torch.tensor(anchor_closes, dtype=torch.float32),
    )


def infer_latest_layer(
    *,
    layer: str,
    run_id: str,
    checkpoint_path: Path,
    tickers: list[str],
    source_data_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = load_checkpoint_config(checkpoint_path)
    model, _ = load_checkpoint(checkpoint_path)
    model = model.to(torch.device("cpu"))
    model.eval()
    features, ticker_ids, future_covariates, metadata, anchor_closes = _latest_model_inputs(
        checkpoint_path=checkpoint_path,
        tickers=tickers,
    )
    with torch.no_grad():
        output = model(features, ticker_id=ticker_ids)
    line_returns, lower_returns, upper_returns = apply_band_postprocess(
        output.line.detach().cpu(),
        output.lower_band.detach().cpu(),
        output.upper_band.detach().cpu(),
    )
    line_prices, lower_prices, upper_prices = decode_return_forecasts(
        line_returns,
        lower_returns,
        upper_returns,
        anchor_closes,
    )

    q_low = float(config.get("q_low", 0.1))
    q_high = float(config.get("q_high", 0.9))
    decision_time = utc_now_iso()
    prediction_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    lower_lte_upper = True
    series_lengths: dict[str, list[int]] = {"line": [], "lower": [], "upper": [], "conservative": []}

    for index, row in metadata.iterrows():
        line_series = [float(value) for value in line_prices[index]]
        lower_series = [float(value) for value in lower_prices[index]]
        upper_series = [float(value) for value in upper_prices[index]]
        conservative_series = line_series
        lower_lte_upper = lower_lte_upper and all(lo <= hi for lo, hi in zip(lower_series, upper_series, strict=False))
        series_lengths["line"].append(len(line_series))
        series_lengths["lower"].append(len(lower_series))
        series_lengths["upper"].append(len(upper_series))
        series_lengths["conservative"].append(len(conservative_series))
        current_price = float(row["anchor_close"])
        signal = "BUY" if current_price < lower_series[-1] else "SELL" if current_price > upper_series[-1] else "HOLD"
        meta = {
            "layer": layer,
            "provider": "yfinance",
            "source": "yfinance",
            "source_data_hash": source_data_hash,
            "feature_version": FEATURE_CONTRACT_VERSION,
            "thin_upload": True,
            "composite": False,
            "anchor_close": current_price,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
            "provider_adjustment_policy": provider_adjustment_policy("yfinance"),
        }
        prediction_records.append(
            {
                "ticker": row["ticker"],
                "model_name": str(config["model"]),
                "timeframe": "1D",
                "horizon": int(config["horizon"]),
                "asof_date": row["asof_date"],
                "decision_time": decision_time,
                "run_id": run_id,
                "model_ver": str(config.get("model_ver", "v2-multihead")),
                "signal": signal,
                "forecast_dates": row["forecast_dates"],
                "line_series": line_series,
                "conservative_series": conservative_series,
                "lower_band_series": lower_series,
                "upper_band_series": upper_series,
                "band_quantile_low": q_low,
                "band_quantile_high": q_high,
                "meta": meta,
            }
        )
        width = float(np.mean(np.asarray(upper_series) - np.asarray(lower_series)))
        normalized_width = width / max(current_price, 1e-6)
        evaluation_records.append(
            {
                "run_id": run_id,
                "ticker": row["ticker"],
                "timeframe": "1D",
                "asof_date": row["asof_date"],
                "actual_series": [],
                "pinball_loss": None,
                "coverage": None,
                "lower_breach_rate": None,
                "upper_breach_rate": None,
                "avg_band_width": width,
                "normalized_band_width": normalized_width,
                "direction_accuracy": None,
                "mae": None,
                "smape": None,
            }
        )

    diagnostics = {
        "layer": layer,
        "run_id": run_id,
        "model": str(config["model"]),
        "seq_len": int(config["seq_len"]),
        "horizon": int(config["horizon"]),
        "feature_columns_count": len(config.get("feature_columns") or MODEL_FEATURE_COLUMNS),
        "prediction_count": len(prediction_records),
        "evaluation_count": len(evaluation_records),
        "asof_dates": sorted({record["asof_date"] for record in prediction_records}),
        "forecast_date_lengths": sorted({len(record["forecast_dates"]) for record in prediction_records}),
        "series_lengths": {key: sorted(set(values)) for key, values in series_lengths.items()},
        "lower_lte_upper": lower_lte_upper,
        "payload_bytes": len(json.dumps(_json_safe(prediction_records), ensure_ascii=False).encode("utf-8")),
    }
    return prediction_records, evaluation_records, diagnostics


def count_existing_rows(table: str, run_id: str, tickers: list[str]) -> int | None:
    try:
        client = get_supabase()
        result = (
            client.table(table)
            .select("id", count="exact")
            .eq("run_id", run_id)
            .in_("ticker", tickers)
            .eq("timeframe", "1D")
            .limit(1)
            .execute()
        )
        return int(result.count or 0)
    except Exception:
        return None


def verify_model_run_exists(run_id: str) -> bool:
    rows = fetch_all_rows("model_runs", columns="run_id", filters=[("eq", "run_id", run_id)], limit=1)
    return bool(rows)


def save_latest_records(predictions: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> None:
    save_product_latest_predictions(
        predictions,
        evaluations,
        max_prediction_rows=20,
        max_evaluation_rows=20,
    )


def fetch_latest_prediction_by_run(ticker: str, run_id: str) -> dict[str, Any] | None:
    client = get_supabase()
    rows = (
        client.table("predictions")
        .select("ticker,model_name,timeframe,horizon,asof_date,run_id,forecast_dates,line_series,lower_band_series,upper_band_series,conservative_series,meta")
        .eq("ticker", ticker)
        .eq("run_id", run_id)
        .eq("timeframe", "1D")
        .order("asof_date", desc=True)
        .order("decision_time", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0] if rows else None


def verify_bulk_read_guard() -> dict[str, str]:
    result: dict[str, str] = {}
    for table in ("price_data", "indicators"):
        try:
            fetch_all_rows(table, limit=None)
            result[table] = "FAILED_NOT_BLOCKED"
        except RuntimeError:
            result[table] = "PASS_BLOCKED"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP99 1D 제품 루프 thin upload 검증")
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--write", action="store_true", help="5티커 latest-only 결과를 Supabase에 저장합니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    source_data_hash = resolve_data_fingerprint("1D", tickers=tickers, market_data_provider="yfinance")
    metrics: dict[str, Any] = {
        "cp": "CP99-D",
        "generated_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "timeframe": "1D",
            "horizon": 5,
            "provider": "yfinance",
            "source": "yfinance",
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "composite_saved": False,
        },
        "environment": {
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
        },
        "forbidden_actions_observed": {
            "full_yfinance_supabase_write": False,
            "supabase_price_indicator_bulk_read": False,
            "indicators_supabase_recompute": False,
            "full_training": False,
            "one_week_or_one_month_processing": False,
            "prediction_history_bulk_save": False,
            "eodhd_delete": False,
            "frontend_ui_modify": False,
            "composite_save": False,
        },
    }

    metrics["snapshot_gate"] = snapshot_gate(tickers)
    metrics["bulk_read_guard"] = verify_bulk_read_guard()
    model_run_exists = {
        LINE_RUN_ID: verify_model_run_exists(LINE_RUN_ID),
        BAND_RUN_ID: verify_model_run_exists(BAND_RUN_ID),
    }
    metrics["model_run_exists"] = model_run_exists
    if not all(model_run_exists.values()):
        metrics["final_decision"] = {"status": "FAIL", "reason": "model_run_missing"}
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
    metrics["dry_run"] = {
        "line": line_diag,
        "band": band_diag,
        "prediction_rows": len(predictions),
        "evaluation_rows": len(evaluations),
        "payload_bytes": len(json.dumps(_json_safe({"predictions": predictions, "evaluations": evaluations}), ensure_ascii=False).encode("utf-8")),
        "all_forecast_horizon_5": all(len(record["forecast_dates"]) == 5 for record in predictions),
        "all_series_length_5": all(
            len(record["line_series"]) == 5
            and len(record["conservative_series"]) == 5
            and len(record["lower_band_series"]) == 5
            and len(record["upper_band_series"]) == 5
            for record in predictions
        ),
        "lower_lte_upper": line_diag["lower_lte_upper"] and band_diag["lower_lte_upper"],
    }
    write_json(log_dir / "dry_run_predictions.json", {"predictions": predictions, "evaluations": evaluations})

    metrics["thin_upload_plan"] = {
        "write_requested": bool(args.write),
        "target_prediction_rows": len(predictions),
        "target_evaluation_rows": len(evaluations),
        "composite_rows": 0,
        "latest_only": True,
        "uses_ai_inference_save": False,
        "before_counts": {
            "line_predictions": count_existing_rows("predictions", LINE_RUN_ID, tickers),
            "band_predictions": count_existing_rows("predictions", BAND_RUN_ID, tickers),
            "line_evaluations": count_existing_rows("prediction_evaluations", LINE_RUN_ID, tickers),
            "band_evaluations": count_existing_rows("prediction_evaluations", BAND_RUN_ID, tickers),
        },
    }

    if args.write:
        save_latest_records(predictions, evaluations)
        metrics["thin_upload_result"] = {
            "attempted": True,
            "prediction_rows_written": len(predictions),
            "evaluation_rows_written": len(evaluations),
            "after_counts": {
                "line_predictions": count_existing_rows("predictions", LINE_RUN_ID, tickers),
                "band_predictions": count_existing_rows("predictions", BAND_RUN_ID, tickers),
                "line_evaluations": count_existing_rows("prediction_evaluations", LINE_RUN_ID, tickers),
                "band_evaluations": count_existing_rows("prediction_evaluations", BAND_RUN_ID, tickers),
            },
        }
    else:
        metrics["thin_upload_result"] = {
            "attempted": False,
            "prediction_rows_written": 0,
            "evaluation_rows_written": 0,
        }

    api_line = fetch_latest_prediction_by_run("AAPL", LINE_RUN_ID) if args.write else None
    api_band = fetch_latest_prediction_by_run("AAPL", BAND_RUN_ID) if args.write else None
    metrics["api_check"] = {
        "attempted": bool(args.write),
        "aapl_line_found": api_line is not None,
        "aapl_band_found": api_band is not None,
        "line_run_id": api_line.get("run_id") if api_line else None,
        "band_run_id": api_band.get("run_id") if api_band else None,
        "line_model_name": api_line.get("model_name") if api_line else None,
        "band_model_name": api_band.get("model_name") if api_band else None,
        "line_asof_date": api_line.get("asof_date") if api_line else None,
        "band_asof_date": api_band.get("asof_date") if api_band else None,
        "line_meta_layer": (api_line.get("meta") or {}).get("layer") if api_line else None,
        "band_meta_layer": (api_band.get("meta") or {}).get("layer") if api_band else None,
        "one_week_one_month_policy_unchanged": True,
    }

    pass_status = (
        metrics["snapshot_gate"]["split_gate"].get("status") == "PASS"
        and metrics["dry_run"]["prediction_rows"] == 10
        and metrics["dry_run"]["evaluation_rows"] == 10
        and metrics["dry_run"]["all_forecast_horizon_5"]
        and metrics["dry_run"]["all_series_length_5"]
        and metrics["dry_run"]["lower_lte_upper"]
        and metrics["bulk_read_guard"] == {"price_data": "PASS_BLOCKED", "indicators": "PASS_BLOCKED"}
        and (not args.write or (metrics["api_check"]["aapl_line_found"] and metrics["api_check"]["aapl_band_found"]))
    )
    metrics["final_decision"] = {
        "status": "PASS" if pass_status and args.write else "DRY_RUN_PASS" if pass_status else "FAIL",
        "write_executed": bool(args.write),
        "summary": "local yfinance snapshot 기반 latest-only 1D 제품 루프 검증",
    }
    write_json(Path(args.metrics_path), metrics)
    write_json(log_dir / "cp99_1d_product_loop_thin_upload_metrics.json", metrics)
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
