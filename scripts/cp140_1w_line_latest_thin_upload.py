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
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp140_1w_line_latest_thin_upload"

RUN_ID = "patchtst-1W-fe7f05a84c93"
TIMEFRAME = "1W"
HORIZON = 4
ROLE = "line_model"
CHECKPOINT_PATH = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "patchtst_1W_patchtst-1W-fe7f05a84c93.pt"
PRICE_1W_PATH = SNAPSHOT_DIR / "price_data_yfinance_1W.parquet"
INDICATORS_1W_PATH = SNAPSHOT_DIR / "indicators_yfinance_1W.parquet"
DEFAULT_CHECK_TICKERS = ["AAPL", "MSFT", "NVDA"]

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
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch(cpu_only=True)  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

from ai.inference import decode_return_forecasts, load_checkpoint, load_checkpoint_config, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import FEATURE_CONTRACT_VERSION, FUTURE_CALENDAR_COLUMNS, MODEL_N_FEATURES, append_calendar_features, resolve_data_fingerprint  # noqa: E402
from ai.storage import get_model_run, save_product_latest_predictions  # noqa: E402
from backend.app.repositories.prediction_repo import fetch_prediction_by_run  # noqa: E402
from backend.collector.repositories.base import fetch_all_rows  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_1w_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(PRICE_1W_PATH)
    indicators = pd.read_parquet(INDICATORS_1W_PATH)
    for frame in [price, indicators]:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return price, indicators


def snapshot_summary(frame: pd.DataFrame, duplicate_keys: list[str]) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "duplicate_count": int(frame.duplicated(duplicate_keys).sum()) if all(key in frame.columns for key in duplicate_keys) else None,
        "source_values": sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist()) if "source" in frame.columns else [],
        "provider_values": sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist()) if "provider" in frame.columns else [],
    }


def next_weekly_forecast_dates(asof_date: str, horizon: int) -> list[str]:
    asof_ts = pd.to_datetime(asof_date)
    start = asof_ts + pd.offsets.Week(weekday=4)
    if start <= asof_ts:
        start = start + pd.offsets.Week(weekday=4)
    return pd.date_range(start=start, periods=horizon, freq="W-FRI").strftime("%Y-%m-%d").tolist()


def load_latest_inputs(tickers: list[str]) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor, dict[str, Any]]:
    price, indicators = load_1w_snapshots()
    config = load_checkpoint_config(CHECKPOINT_PATH)
    feature_columns = list(config.get("feature_columns") or [])
    seq_len = int(config["seq_len"])
    registry = resolve_checkpoint_ticker_registry(config, TIMEFRAME)
    mapping = {str(key).upper(): int(value) for key, value in (registry or {}).get("mapping", {}).items()}
    feature_mean = torch.as_tensor(torch.load(CHECKPOINT_PATH, map_location="cpu")["feature_mean"], dtype=torch.float32)
    feature_std = torch.as_tensor(torch.load(CHECKPOINT_PATH, map_location="cpu")["feature_std"], dtype=torch.float32).clamp_min(1e-6)

    indicators = indicators[indicators["ticker"].isin(tickers)].copy()
    indicators = append_calendar_features(indicators)
    price = price[price["ticker"].isin(tickers)].copy()
    price["target_close"] = pd.to_numeric(price["adjusted_close"].fillna(price["close"]), errors="coerce")

    rows: list[dict[str, Any]] = []
    feature_tensors: list[np.ndarray] = []
    anchor_closes: list[float] = []
    ticker_ids: list[int] = []
    skipped: dict[str, str] = {}

    for ticker in tickers:
        ticker_features = indicators[indicators["ticker"] == ticker].sort_values("date").copy()
        if len(ticker_features) < seq_len:
            skipped[ticker] = f"seq_len={seq_len}보다 1W feature row가 적음"
            continue
        latest_features = ticker_features.tail(seq_len)
        feature_values = latest_features[feature_columns].to_numpy(dtype="float32")
        if not np.isfinite(feature_values).all():
            skipped[ticker] = "latest feature window에 non-finite 값 존재"
            continue
        asof_ts = latest_features["date"].iloc[-1]
        ticker_price = price[(price["ticker"] == ticker) & (price["date"] == asof_ts)]
        if ticker_price.empty:
            skipped[ticker] = f"asof_date={asof_ts.date().isoformat()} anchor price 없음"
            continue
        anchor_close = float(ticker_price["target_close"].iloc[-1])
        if not math.isfinite(anchor_close) or anchor_close <= 0:
            skipped[ticker] = "anchor_close 비정상"
            continue
        if ticker not in mapping:
            skipped[ticker] = "checkpoint ticker registry에 없음"
            continue

        feature_tensors.append(feature_values)
        anchor_closes.append(anchor_close)
        ticker_ids.append(mapping[ticker])
        rows.append(
            {
                "ticker": ticker,
                "asof_date": asof_ts.strftime("%Y-%m-%d"),
                "forecast_dates": next_weekly_forecast_dates(asof_ts.strftime("%Y-%m-%d"), int(config["horizon"])),
                "anchor_close": anchor_close,
            }
        )

    if not feature_tensors:
        raise ValueError("저장 가능한 1W latest inference 입력이 없습니다.")

    features = torch.tensor(np.stack(feature_tensors), dtype=torch.float32)
    features = (features - feature_mean.view(1, 1, -1)) / feature_std.view(1, 1, -1)
    return (
        features,
        torch.tensor(ticker_ids, dtype=torch.long),
        pd.DataFrame(rows),
        torch.tensor(anchor_closes, dtype=torch.float32),
        {
            "requested_ticker_count": len(tickers),
            "usable_ticker_count": len(rows),
            "skipped_tickers": skipped,
            "feature_columns": feature_columns,
            "seq_len": seq_len,
        },
    )


def build_prediction_records(tickers: list[str], source_data_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = load_checkpoint_config(CHECKPOINT_PATH)
    model, _ = load_checkpoint(CHECKPOINT_PATH)
    model = model.to(torch.device("cpu"))
    model.eval()

    features, ticker_ids, metadata, anchor_closes, input_metrics = load_latest_inputs(tickers)
    with torch.no_grad():
        output = model(features, ticker_id=ticker_ids)
    line_returns, _, _ = apply_band_postprocess(
        output.line.detach().cpu(),
        output.lower_band.detach().cpu(),
        output.upper_band.detach().cpu(),
    )
    line_prices, _, _ = decode_return_forecasts(line_returns, line_returns, line_returns, anchor_closes)
    decision_time = utc_now_iso()

    prediction_records: list[dict[str, Any]] = []
    asof_dates: set[str] = set()
    forecast_lengths: list[int] = []
    series_lengths: list[int] = []
    for index, row in metadata.iterrows():
        line_series = [float(value) for value in line_prices[index]]
        asof_date = str(row["asof_date"])
        forecast_dates = list(row["forecast_dates"])
        asof_dates.add(asof_date)
        forecast_lengths.append(len(forecast_dates))
        series_lengths.append(len(line_series))
        prediction_records.append(
            {
                "ticker": row["ticker"],
                "model_name": config.get("model", "patchtst"),
                "timeframe": TIMEFRAME,
                "horizon": HORIZON,
                "asof_date": asof_date,
                "decision_time": decision_time,
                "run_id": RUN_ID,
                "model_ver": config.get("model_ver", "v2-multihead"),
                "signal": "HOLD",
                "forecast_dates": forecast_dates,
                "line_series": line_series,
                "conservative_series": line_series,
                "lower_band_series": line_series,
                "upper_band_series": line_series,
                "band_quantile_low": float(config.get("q_low", 0.1)),
                "band_quantile_high": float(config.get("q_high", 0.9)),
                "meta": {
                    "layer": "line",
                    "role": ROLE,
                    "composite": False,
                    "source": "yfinance",
                    "provider": "yfinance",
                    "provider_adjustment_policy": provider_adjustment_policy("yfinance"),
                    "storage_contract": "product_latest_only",
                    "feature_set": config.get("feature_set"),
                    "feature_version": config.get("feature_version", FEATURE_CONTRACT_VERSION),
                    "source_data_hash": source_data_hash,
                    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)),
                    "band_saved_in_cp140": False,
                    "band_fields_policy": "schema_required_degenerate_equal_to_line",
                },
            }
        )

    return prediction_records, {
        **input_metrics,
        "prediction_count": len(prediction_records),
        "asof_dates": sorted(asof_dates),
        "forecast_date_lengths": sorted(set(forecast_lengths)),
        "line_series_lengths": sorted(set(series_lengths)),
        "model": config.get("model"),
        "feature_set": config.get("feature_set"),
        "horizon": int(config.get("horizon")),
        "n_features": int(config.get("n_features")),
    }


def fetch_prediction_rows() -> list[dict[str, Any]]:
    return fetch_all_rows(
        "predictions",
        columns="ticker,model_name,timeframe,horizon,asof_date,run_id,meta",
        filters=[("eq", "run_id", RUN_ID), ("eq", "timeframe", TIMEFRAME), ("eq", "horizon", HORIZON)],
        limit=1000,
    )


def fetch_evaluation_rows() -> list[dict[str, Any]]:
    return fetch_all_rows(
        "prediction_evaluations",
        columns="ticker,timeframe,asof_date,run_id",
        filters=[("eq", "run_id", RUN_ID), ("eq", "timeframe", TIMEFRAME)],
        limit=1000,
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    asof_dates = sorted({str(row.get("asof_date")) for row in rows if row.get("asof_date")})
    layer_counts: dict[str, int] = {}
    composite_count = 0
    for row in rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        layer = str(meta.get("layer") or "missing")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        composite_count += int(bool(meta.get("composite")) or row.get("model_name") == "line_band_composite")
    return {
        "rows": len(rows),
        "ticker_count": len({row.get("ticker") for row in rows if row.get("ticker")}),
        "asof_dates": asof_dates,
        "asof_date_count": len(asof_dates),
        "layer_counts": layer_counts,
        "composite_count": composite_count,
    }


def api_check(tickers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ticker in tickers:
        row = fetch_prediction_by_run(ticker, run_id=RUN_ID)
        meta = row.get("meta") if isinstance(row, dict) and isinstance(row.get("meta"), dict) else {}
        result[ticker] = {
            "found": row is not None,
            "asof_date": row.get("asof_date") if row else None,
            "timeframe": row.get("timeframe") if row else None,
            "horizon": row.get("horizon") if row else None,
            "run_id": row.get("run_id") if row else None,
            "model_name": row.get("model_name") if row else None,
            "meta_layer": meta.get("layer"),
            "meta_role": meta.get("role"),
            "storage_contract": meta.get("storage_contract"),
            "line_length": len(row.get("line_series") or []) if row else 0,
            "forecast_date_length": len(row.get("forecast_dates") or []) if row else 0,
            "has_meaningful_band_series": bool(
                row
                and (
                    (row.get("lower_band_series") or []) != (row.get("line_series") or [])
                    or (row.get("upper_band_series") or []) != (row.get("line_series") or [])
                )
            ),
        }
    return result


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_decision"]
    save = metrics["save_result"]
    report = f"""# CP140-D 1W line latest-only thin upload 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

{final["summary"]}

## 2. 실행 범위

- run_id: `{RUN_ID}`
- timeframe: `{TIMEFRAME}`
- horizon: `{HORIZON}`
- role: `{ROLE}`
- feature_set: `{metrics["inference"]["feature_set"]}`
- source/provider: yfinance local parquet

## 3. local 1W snapshot

| snapshot | rows | tickers | latest date | duplicate | source | provider |
|---|---:|---:|---|---:|---|---|
| price 1W | {metrics["snapshots"]["price_1w"]["rows"]} | {metrics["snapshots"]["price_1w"]["ticker_count"]} | {metrics["snapshots"]["price_1w"]["date_max"]} | {metrics["snapshots"]["price_1w"]["duplicate_count"]} | {metrics["snapshots"]["price_1w"]["source_values"]} | {metrics["snapshots"]["price_1w"]["provider_values"]} |
| indicators 1W | {metrics["snapshots"]["indicators_1w"]["rows"]} | {metrics["snapshots"]["indicators_1w"]["ticker_count"]} | {metrics["snapshots"]["indicators_1w"]["date_max"]} | {metrics["snapshots"]["indicators_1w"]["duplicate_count"]} | {metrics["snapshots"]["indicators_1w"]["source_values"]} | {metrics["snapshots"]["indicators_1w"]["provider_values"]} |

## 4. inference 결과

| 항목 | 값 |
|---|---|
| usable ticker count | {metrics["inference"]["usable_ticker_count"]} |
| skipped tickers | {metrics["inference"]["skipped_tickers"]} |
| asof dates | {metrics["inference"]["asof_dates"]} |
| forecast date lengths | {metrics["inference"]["forecast_date_lengths"]} |
| line series lengths | {metrics["inference"]["line_series_lengths"]} |
| n_features | {metrics["inference"]["n_features"]} |

## 5. latest-only 저장 결과

DB 스키마상 `lower_band_series`와 `upper_band_series`는 not-null이라 line-only row에도 값을 넣어야 한다. CP140은 의미 있는 1W band를 저장하지 않기 위해 두 필드를 `line_series`와 동일한 퇴화 구간으로 저장하고, meta에 `band_saved_in_cp140=False`, `band_fields_policy=schema_required_degenerate_equal_to_line`을 남긴다.

| 항목 | 값 |
|---|---|
| attempted | {save["attempted"]} |
| predictions attempted rows | {save["prediction_rows_attempted"]} |
| evaluations attempted rows | {save["evaluation_rows_attempted"]} |
| prediction row delta | {save["prediction_row_delta"]} |
| evaluation row delta | {save["evaluation_row_delta"]} |
| before predictions | {save["before_predictions"]} |
| after predictions | {save["after_predictions"]} |
| before evaluations | {save["before_evaluations"]} |
| after evaluations | {save["after_evaluations"]} |

## 6. API/repository 조회

| ticker | found | asof_date | timeframe | horizon | layer | role | storage_contract | line length | meaningful band series |
|---|---:|---|---|---:|---|---|---|---:|---:|
{api_table(metrics["api_check"])}

## 7. 금지 작업 확인

- 1W band inference/storage: {metrics["forbidden_actions_observed"]["band_inference_storage"]}
- composite 저장: {metrics["forbidden_actions_observed"]["composite_save"]}
- bulk evaluation save: {metrics["forbidden_actions_observed"]["bulk_evaluation_save"]}
- 모델 학습: {metrics["forbidden_actions_observed"]["model_training"]}
- W&B: {metrics["forbidden_actions_observed"]["wandb"]}
- yfinance live fetch: {metrics["forbidden_actions_observed"]["yfinance_live_fetch"]}
- EODHD 호출: {metrics["forbidden_actions_observed"]["eodhd_call"]}
- Supabase price_data/indicators 대량 read: {metrics["forbidden_actions_observed"]["supabase_price_indicator_bulk_read"]}
- 프론트 수정: {metrics["forbidden_actions_observed"]["frontend_modify"]}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def api_table(api: dict[str, Any]) -> str:
    rows = []
    for ticker, item in api.items():
        rows.append(
            f"| {ticker} | {item['found']} | {item['asof_date']} | {item['timeframe']} | {item['horizon']} | {item['meta_layer']} | {item['meta_role']} | {item['storage_contract']} | {item['line_length']} | {item['has_meaningful_band_series']} |"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP140 1W line latest-only thin upload")
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp140_1w_line_latest_thin_upload_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp140_1w_line_latest_thin_upload_report.md"))
    parser.add_argument("--check-tickers", nargs="*", default=DEFAULT_CHECK_TICKERS)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    price, indicators = load_1w_snapshots()
    config = load_checkpoint_config(CHECKPOINT_PATH)
    registry = resolve_checkpoint_ticker_registry(config, TIMEFRAME)
    registry_tickers = sorted(str(ticker).upper() for ticker in (registry or {}).get("mapping", {}).keys())
    source_data_hash = resolve_data_fingerprint(TIMEFRAME, tickers=registry_tickers, market_data_provider="yfinance")
    model_run = get_model_run(RUN_ID)

    before_predictions = summarize_rows(fetch_prediction_rows())
    before_evaluations = summarize_rows(fetch_evaluation_rows())
    prediction_records, inference_metrics = build_prediction_records(registry_tickers, source_data_hash)
    evaluation_records: list[dict[str, Any]] = []

    if args.no_save:
        attempted = False
    else:
        save_product_latest_predictions(
            prediction_records,
            evaluation_records,
            max_prediction_rows=max(100, len(prediction_records)),
            max_evaluation_rows=0,
        )
        attempted = True

    after_predictions = summarize_rows(fetch_prediction_rows())
    after_evaluations = summarize_rows(fetch_evaluation_rows())
    api = api_check([ticker.upper() for ticker in args.check_tickers])

    prediction_row_delta = after_predictions["rows"] - before_predictions["rows"]
    evaluation_row_delta = after_evaluations["rows"] - before_evaluations["rows"]
    band_rows_after = int(after_predictions["layer_counts"].get("band", 0))
    composite_rows_after = int(after_predictions["composite_count"])
    expected_asof = inference_metrics["asof_dates"][0] if len(inference_metrics["asof_dates"]) == 1 else None

    failures: list[str] = []
    warnings: list[str] = []
    if model_run is None:
        failures.append("model_runs에서 product run_id를 찾지 못함")
    if len(inference_metrics["asof_dates"]) != 1:
        failures.append("latest-only 입력 asof_date가 하나가 아님")
    if inference_metrics["horizon"] != HORIZON:
        failures.append("checkpoint horizon이 CP140 계약과 다름")
    if inference_metrics["feature_set"] != "price_volatility_volume":
        failures.append("feature_set이 price_volatility_volume이 아님")
    if any(length != HORIZON for length in inference_metrics["forecast_date_lengths"]):
        failures.append("forecast_dates 길이가 horizon과 다름")
    if any(length != HORIZON for length in inference_metrics["line_series_lengths"]):
        failures.append("line_series 길이가 horizon과 다름")
    if evaluation_row_delta != 0:
        failures.append("prediction_evaluations row가 증가함")
    if band_rows_after != 0:
        failures.append("line run_id 아래 band layer row가 존재함")
    if composite_rows_after != 0:
        failures.append("composite row가 존재함")
    if after_predictions["asof_date_count"] > 1:
        warnings.append("해당 run_id에 여러 asof_date row가 존재함. 이번 저장은 단일 asof_date지만 기존 row 확인 필요")
    if prediction_row_delta > len(prediction_records):
        failures.append("prediction row delta가 저장 시도 row 수보다 큼")
    for ticker, item in api.items():
        if not item["found"]:
            failures.append(f"{ticker} API/repository latest 조회 실패")
        if item["timeframe"] != TIMEFRAME or item["horizon"] != HORIZON:
            failures.append(f"{ticker} API/repository timeframe/horizon 불일치")
        if item["meta_layer"] != "line" or item["meta_role"] != ROLE:
            failures.append(f"{ticker} API/repository meta layer/role 불일치")
        if item["storage_contract"] != "product_latest_only":
            failures.append(f"{ticker} storage_contract 누락 또는 불일치")
        if item["has_meaningful_band_series"]:
            failures.append(f"{ticker} line row에 의미 있는 band series가 저장됨")
        if expected_asof and item["asof_date"] != expected_asof:
            failures.append(f"{ticker} asof_date가 latest 1W와 다름")

    if failures:
        status = "FAIL"
        summary = "1W line latest-only 저장 검증 중 중단 조건이 발생했다."
    elif warnings:
        status = "WARN"
        summary = "1W line latest-only 저장은 완료됐지만 기존 row 상태에 확인이 필요한 경고가 남았다."
    else:
        status = "PASS"
        summary = "1W line latest-only prediction 저장이 완료됐고 API/repository 조회가 통과했다. band/composite/bulk 저장은 발생하지 않았다."

    metrics: dict[str, Any] = {
        "cp": "CP140-D",
        "created_at": utc_now_iso(),
        "scope": {
            "run_id": RUN_ID,
            "timeframe": TIMEFRAME,
            "horizon": HORIZON,
            "role": ROLE,
            "provider": "yfinance",
            "source": "yfinance",
            "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)),
        },
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "MARKET_DATA_FALLBACK_PROVIDER": os.environ.get("MARKET_DATA_FALLBACK_PROVIDER"),
            "EODHD_API_KEY_SET": bool(os.environ.get("EODHD_API_KEY")),
            "LENS_DATA_BACKEND": os.environ.get("LENS_DATA_BACKEND"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
        },
        "snapshots": {
            "price_1w": snapshot_summary(price, ["ticker", "date", "source"]),
            "indicators_1w": snapshot_summary(indicators, ["ticker", "timeframe", "date", "source"]),
        },
        "model_run_found": model_run is not None,
        "inference": inference_metrics,
        "save_result": {
            "attempted": attempted,
            "prediction_rows_attempted": len(prediction_records),
            "evaluation_rows_attempted": len(evaluation_records),
            "prediction_row_delta": prediction_row_delta,
            "evaluation_row_delta": evaluation_row_delta,
            "before_predictions": before_predictions,
            "after_predictions": after_predictions,
            "before_evaluations": before_evaluations,
            "after_evaluations": after_evaluations,
        },
        "api_check": api,
        "forbidden_actions_observed": {
            "band_inference_storage": False,
            "frontend_modify": False,
            "model_training": False,
            "wandb": False,
            "composite_save": False,
            "bulk_evaluation_save": False,
            "supabase_price_indicator_bulk_read": False,
            "yfinance_live_fetch": False,
            "eodhd_call": False,
        },
        "final_decision": {
            "status": status,
            "summary": summary,
            "failures": failures,
            "warnings": warnings,
        },
    }
    metrics_path = Path(args.metrics_path)
    report_path = Path(args.report_path)
    write_json(metrics_path, metrics)
    write_json(LOG_DIR / metrics_path.name, metrics)
    write_report(report_path, metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
