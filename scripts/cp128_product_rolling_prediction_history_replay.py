from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
OUTPUT_PARQUET = SNAPSHOT_DIR / "product_prediction_history_1D.parquet"
OUTPUT_MANIFEST = SNAPSHOT_DIR / "product_prediction_history_1D.manifest.json"
METRICS_PATH = ROOT_DIR / "docs" / "cp128_product_rolling_prediction_history_replay_metrics.json"
REPORT_PATH = ROOT_DIR / "docs" / "cp128_product_rolling_prediction_history_replay_report.md"
LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
LINE_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "patchtst_1D_patchtst-1D-efad3c29d803.pt"
BAND_CHECKPOINT = ROOT_DIR / "ai" / "artifacts" / "checkpoints" / "cnn_lstm_1D_cnn_lstm-1D-d0c780dee5e8.pt"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
DISPLAY_HORIZON = 5
TIMEFRAME = "1D"
SOURCE = "rolling_replay"
PROVIDER = "yfinance"

os.environ["MARKET_DATA_PROVIDER"] = PROVIDER
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.inference import decode_return_forecasts, load_checkpoint, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_CALENDAR_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    append_calendar_features,
    resolve_data_fingerprint,
)
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
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = pd.read_parquet(SNAPSHOT_DIR / "stock_info.parquet")
    price = pd.read_parquet(SNAPSHOT_DIR / "price_data_yfinance.parquet")
    indicators = pd.read_parquet(SNAPSHOT_DIR / "indicators_yfinance_1D.parquet")
    for frame in (stock, price, indicators):
        if "ticker" in frame.columns:
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    return stock, price, indicators


def resolve_tickers(args: argparse.Namespace, stock: pd.DataFrame) -> list[str]:
    if args.tickers:
        return list(dict.fromkeys(ticker.upper() for ticker in args.tickers))
    if args.limit_tickers:
        return stock["ticker"].dropna().astype(str).str.upper().drop_duplicates().head(args.limit_tickers).tolist()
    return DEFAULT_TICKERS


def asof_window(indicators: pd.DataFrame, *, latest_date: pd.Timestamp, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = latest_date - pd.Timedelta(days=lookback_days)
    available = indicators[(indicators["date"] >= start_date) & (indicators["date"] <= latest_date)]["date"]
    if available.empty:
        raise ValueError("최근 1년 replay 대상 asof_date를 찾을 수 없습니다.")
    return pd.Timestamp(available.min()), pd.Timestamp(available.max())


def checkpoint_registry_mapping(config: dict[str, Any]) -> dict[str, int]:
    registry = resolve_checkpoint_ticker_registry(config, TIMEFRAME)
    if registry is None:
        return {}
    return {str(key).upper(): int(value) for key, value in (registry.get("mapping") or {}).items()}


def build_samples(
    *,
    role: str,
    checkpoint: dict[str, Any],
    tickers: list[str],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    asof_start: pd.Timestamp,
    asof_end: pd.Timestamp,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor, dict[str, Any]]:
    config = dict(checkpoint.get("config") or {})
    seq_len = int(config["seq_len"])
    horizon = int(config["horizon"])
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    registry_mapping = checkpoint_registry_mapping(config)

    missing_features = [column for column in feature_columns if column not in append_calendar_features(indicators.head(1)).columns]
    if missing_features:
        raise ValueError(f"{role}: feature column 누락: {missing_features}")

    price = price.copy()
    price["target_close"] = pd.to_numeric(price["adjusted_close"].fillna(price["close"]), errors="coerce")
    price_lookup = (
        price[price["ticker"].isin(tickers)]
        .dropna(subset=["date", "target_close"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .set_index(["ticker", "date"])
    )

    indicators = append_calendar_features(indicators[indicators["ticker"].isin(tickers)].copy())
    indicators = indicators.sort_values(["ticker", "date"]).reset_index(drop=True)

    features: list[np.ndarray] = []
    ticker_ids: list[int] = []
    anchor_closes: list[float] = []
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    feature_non_finite = 0

    for ticker in tickers:
        if registry_mapping and ticker not in registry_mapping:
            excluded[ticker] = "checkpoint_ticker_registry_missing"
            continue
        ticker_features = indicators[indicators["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if ticker_features.empty:
            excluded[ticker] = "indicator_missing"
            continue
        candidate_indices = ticker_features.index[
            (ticker_features["date"] >= asof_start) & (ticker_features["date"] <= asof_end)
        ].tolist()
        ticker_sample_count = 0
        for end_idx in candidate_indices:
            start_idx = end_idx - seq_len + 1
            if start_idx < 0:
                continue
            asof_date = pd.Timestamp(ticker_features.loc[end_idx, "date"])
            price_key = (ticker, asof_date)
            if price_key not in price_lookup.index:
                continue
            anchor_close = float(price_lookup.loc[price_key, "target_close"])
            if not math.isfinite(anchor_close) or anchor_close <= 0:
                continue
            window = ticker_features.iloc[start_idx : end_idx + 1][feature_columns]
            feature_array = window.to_numpy(dtype="float32")
            non_finite_now = int((~np.isfinite(feature_array)).sum())
            feature_non_finite += non_finite_now
            if non_finite_now:
                continue
            features.append(feature_array)
            ticker_ids.append(registry_mapping.get(ticker, 0))
            anchor_closes.append(anchor_close)
            rows.append(
                {
                    "_sample_index": len(features) - 1,
                    "ticker": ticker,
                    "asof_date": asof_date.strftime("%Y-%m-%d"),
                    "display_date": asof_date.strftime("%Y-%m-%d"),
                    "anchor_close": anchor_close,
                }
            )
            ticker_sample_count += 1
        if ticker_sample_count == 0 and ticker not in excluded:
            excluded[ticker] = "no_replay_samples"

    if not features:
        raise ValueError(f"{role}: replay sample이 없습니다.")

    feature_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    feature_tensor = (feature_tensor - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    metadata_unsorted = pd.DataFrame(rows)
    sorted_metadata = metadata_unsorted.sort_values(["asof_date", "ticker"])
    order_tensor = torch.tensor(sorted_metadata["_sample_index"].to_numpy(dtype="int64"), dtype=torch.long)
    metadata = sorted_metadata.drop(columns=["_sample_index"]).reset_index(drop=True)
    diagnostics = {
        "role": role,
        "seq_len": seq_len,
        "horizon": horizon,
        "feature_columns_count": len(feature_columns),
        "sample_count": int(len(metadata)),
        "ticker_count": int(metadata["ticker"].nunique()),
        "date_min": str(metadata["asof_date"].min()),
        "date_max": str(metadata["asof_date"].max()),
        "excluded": excluded,
        "feature_non_finite_count": feature_non_finite,
    }
    return (
        feature_tensor.index_select(0, order_tensor),
        torch.tensor(ticker_ids, dtype=torch.long).index_select(0, order_tensor),
        torch.zeros((len(metadata), horizon, len(FUTURE_CALENDAR_COLUMNS)), dtype=torch.float32),
        metadata,
        torch.tensor(anchor_closes, dtype=torch.float32).index_select(0, order_tensor),
        diagnostics,
    )


def infer_role_history(
    *,
    role: str,
    run_id: str,
    checkpoint_path: Path,
    tickers: list[str],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    asof_start: pd.Timestamp,
    asof_end: pd.Timestamp,
    source_data_hash: str,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint.get("config") or {})
    model = model.to(torch.device("cpu"))
    model.eval()
    features, ticker_ids, _future_covariates, metadata, anchor_closes, sample_diag = build_samples(
        role=role,
        checkpoint=checkpoint,
        tickers=tickers,
        price=price,
        indicators=indicators,
        asof_start=asof_start,
        asof_end=asof_end,
    )

    h5_index = DISPLAY_HORIZON - 1
    output_rows: list[dict[str, Any]] = []
    lower_lte_upper = True
    output_non_finite = 0

    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            end = min(start + batch_size, len(metadata))
            batch_features = features[start:end]
            batch_ticker_ids = ticker_ids[start:end]
            batch_anchor_closes = anchor_closes[start:end]
            output = model(batch_features, ticker_id=batch_ticker_ids)
            line_returns, lower_returns, upper_returns = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            line_prices, lower_prices, upper_prices = decode_return_forecasts(
                line_returns,
                lower_returns,
                upper_returns,
                batch_anchor_closes,
            )
            for batch_index, (_, meta_row) in enumerate(metadata.iloc[start:end].iterrows()):
                line_value = float(line_prices[batch_index][h5_index])
                lower_value = float(lower_prices[batch_index][h5_index])
                upper_value = float(upper_prices[batch_index][h5_index])
                lower_lte_upper = lower_lte_upper and lower_value <= upper_value
                if role == "line":
                    values = [line_value]
                    row = {
                        "ticker": meta_row["ticker"],
                        "timeframe": TIMEFRAME,
                        "role": "line",
                        "run_id": run_id,
                        "asof_date": meta_row["asof_date"],
                        "display_horizon": DISPLAY_HORIZON,
                        "display_date": meta_row["display_date"],
                        "line_value": line_value,
                        "lower_value": np.nan,
                        "upper_value": np.nan,
                        "source": SOURCE,
                        "model_feature_hash": source_data_hash,
                        "created_at": utc_now_iso(),
                    }
                else:
                    values = [lower_value, upper_value]
                    row = {
                        "ticker": meta_row["ticker"],
                        "timeframe": TIMEFRAME,
                        "role": "band",
                        "run_id": run_id,
                        "asof_date": meta_row["asof_date"],
                        "display_horizon": DISPLAY_HORIZON,
                        "display_date": meta_row["display_date"],
                        "line_value": np.nan,
                        "lower_value": lower_value,
                        "upper_value": upper_value,
                        "source": SOURCE,
                        "model_feature_hash": source_data_hash,
                        "created_at": utc_now_iso(),
                    }
                if not all(math.isfinite(value) for value in values):
                    output_non_finite += 1
                    continue
                output_rows.append(row)

    result = pd.DataFrame(output_rows).sort_values(["ticker", "role", "asof_date"]).reset_index(drop=True)
    diagnostics = {
        **sample_diag,
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
        "model": str(config.get("model")),
        "output_rows": int(len(result)),
        "output_non_finite_count": output_non_finite,
        "lower_lte_upper": bool(lower_lte_upper),
        "display_horizon_unique": sorted(result["display_horizon"].dropna().astype(int).unique().tolist()),
    }
    return result, diagnostics


def max_gap_by_ticker(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for ticker, ticker_frame in frame.groupby("ticker"):
        dates = (
            pd.to_datetime(ticker_frame["asof_date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        if len(dates) <= 1:
            result[str(ticker)] = {"point_count": int(len(dates)), "max_gap_days": 0, "latest_asof_date": None if dates.empty else dates.iloc[-1].strftime("%Y-%m-%d")}
            continue
        gaps = dates.diff().dropna().dt.days
        result[str(ticker)] = {
            "point_count": int(len(dates)),
            "max_gap_days": int(gaps.max()),
            "latest_asof_date": dates.iloc[-1].strftime("%Y-%m-%d"),
            "date_min": dates.iloc[0].strftime("%Y-%m-%d"),
            "date_max": dates.iloc[-1].strftime("%Y-%m-%d"),
        }
    return result


def validate_history(frame: pd.DataFrame, tickers: list[str], latest_date: str) -> dict[str, Any]:
    required_columns = [
        "ticker",
        "timeframe",
        "role",
        "run_id",
        "asof_date",
        "display_horizon",
        "display_date",
        "line_value",
        "lower_value",
        "upper_value",
        "source",
        "model_feature_hash",
        "created_at",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    duplicate_role_date = int(frame.duplicated(subset=["ticker", "role", "asof_date", "display_horizon", "source"]).sum())
    line_frame = frame[frame["role"] == "line"]
    band_frame = frame[frame["role"] == "band"]
    line_value_non_finite = int((~np.isfinite(line_frame["line_value"].to_numpy(dtype="float64"))).sum())
    band_lower_non_finite = int((~np.isfinite(band_frame["lower_value"].to_numpy(dtype="float64"))).sum())
    band_upper_non_finite = int((~np.isfinite(band_frame["upper_value"].to_numpy(dtype="float64"))).sum())
    lower_lte_upper_violations = int((band_frame["lower_value"] > band_frame["upper_value"]).sum())
    role_counts = {str(key): int(value) for key, value in frame["role"].value_counts().sort_index().items()}
    coverage: dict[str, Any] = {}
    for ticker in tickers:
        ticker_frame = frame[frame["ticker"] == ticker]
        coverage[ticker] = {
            "line_rows": int((ticker_frame["role"] == "line").sum()),
            "band_rows": int((ticker_frame["role"] == "band").sum()),
            "latest_asof_date": str(ticker_frame["asof_date"].max()) if not ticker_frame.empty else None,
        }
    return {
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "role_counts": role_counts,
        "duplicate_ticker_role_asof_source": duplicate_role_date,
        "asof_monotonic_by_ticker_role": bool(
            all(
                group["asof_date"].is_monotonic_increasing
                for _, group in frame.sort_values(["ticker", "role", "asof_date"]).groupby(["ticker", "role"])
            )
        ),
        "display_horizon_unique": sorted(frame["display_horizon"].dropna().astype(int).unique().tolist()),
        "source_values": sorted(frame["source"].dropna().astype(str).unique().tolist()),
        "line_run_ids": sorted(line_frame["run_id"].dropna().astype(str).unique().tolist()),
        "band_run_ids": sorted(band_frame["run_id"].dropna().astype(str).unique().tolist()),
        "line_value_non_finite": line_value_non_finite,
        "band_lower_non_finite": band_lower_non_finite,
        "band_upper_non_finite": band_upper_non_finite,
        "lower_lte_upper_violations": lower_lte_upper_violations,
        "contains_forecast_arrays": any(column in frame.columns for column in ("forecast_dates", "line_series", "lower_band_series", "upper_band_series")),
        "contains_latest_live_source": bool((frame["source"] == "latest_live").any()),
        "contains_test_split_source": bool((frame["source"].astype(str).str.contains("test", case=False, na=False)).any()),
        "coverage": coverage,
        "continuity_by_ticker": max_gap_by_ticker(frame),
        "latest_date_expected": latest_date,
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    validation = metrics["validation"]
    manifest = metrics["manifest"]
    role_diag = metrics["role_diagnostics"]
    report = f"""# CP128-DG 1D Product Rolling Prediction History Replay

생성일: 2026-05-06

## 1. Executive Summary

최종 판정: {metrics["final_decision"]["status"]}

{metrics["final_decision"]["summary"]}

이번 CP는 기존 `predictions/history`나 test split bulk row를 사용하지 않고, 제품 checkpoint와 local yfinance parquet만으로 1D 제품용 rolling prediction history를 생성했다. 생성 결과는 h5 scalar display point만 포함하며, latest-only product row나 미래 forecast row를 섞지 않았다.

## 2. 범위

| 항목 | 값 |
|---|---|
| timeframe | 1D |
| line run | {LINE_RUN_ID} |
| band run | {BAND_RUN_ID} |
| tickers | {", ".join(metrics["scope"]["tickers"])} |
| source/provider | yfinance local parquet |
| display_horizon | {DISPLAY_HORIZON} |
| asof_start | {metrics["scope"]["asof_start"]} |
| asof_end | {metrics["scope"]["asof_end"]} |

## 3. 생성 파일

| 파일 | rows/bytes |
|---|---:|
| `{metrics["output"]["parquet_path"]}` | {validation["row_count"]} rows / {metrics["output"]["parquet_bytes"]} bytes |
| `{metrics["output"]["manifest_path"]}` | {metrics["output"]["manifest_bytes"]} bytes |

## 4. Role별 Inference 진단

| role | run_id | model | seq_len | samples | output rows | ticker count | date min | date max | non-finite |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| line | {role_diag["line"]["run_id"]} | {role_diag["line"]["model"]} | {role_diag["line"]["seq_len"]} | {role_diag["line"]["sample_count"]} | {role_diag["line"]["output_rows"]} | {role_diag["line"]["ticker_count"]} | {role_diag["line"]["date_min"]} | {role_diag["line"]["date_max"]} | {role_diag["line"]["output_non_finite_count"]} |
| band | {role_diag["band"]["run_id"]} | {role_diag["band"]["model"]} | {role_diag["band"]["seq_len"]} | {role_diag["band"]["sample_count"]} | {role_diag["band"]["output_rows"]} | {role_diag["band"]["ticker_count"]} | {role_diag["band"]["date_min"]} | {role_diag["band"]["date_max"]} | {role_diag["band"]["output_non_finite_count"]} |

## 5. Contract 검증

| 검증 | 결과 |
|---|---|
| required columns 누락 | {validation["missing_columns"]} |
| duplicate ticker/role/asof/source | {validation["duplicate_ticker_role_asof_source"]} |
| asof_date 오름차순 | {validation["asof_monotonic_by_ticker_role"]} |
| display_horizon unique | {validation["display_horizon_unique"]} |
| source values | {validation["source_values"]} |
| line run ids | {validation["line_run_ids"]} |
| band run ids | {validation["band_run_ids"]} |
| line value non-finite | {validation["line_value_non_finite"]} |
| band lower non-finite | {validation["band_lower_non_finite"]} |
| band upper non-finite | {validation["band_upper_non_finite"]} |
| lower > upper violations | {validation["lower_lte_upper_violations"]} |
| forecast array columns 포함 | {validation["contains_forecast_arrays"]} |
| latest_live source 포함 | {validation["contains_latest_live_source"]} |
| test split source 포함 | {validation["contains_test_split_source"]} |

## 6. AAPL/MSFT/NVDA 연속성

| ticker | line rows | band rows | latest asof | max gap days |
|---|---:|---:|---|---:|
| AAPL | {validation["coverage"]["AAPL"]["line_rows"]} | {validation["coverage"]["AAPL"]["band_rows"]} | {validation["coverage"]["AAPL"]["latest_asof_date"]} | {validation["continuity_by_ticker"]["AAPL"]["max_gap_days"]} |
| MSFT | {validation["coverage"]["MSFT"]["line_rows"]} | {validation["coverage"]["MSFT"]["band_rows"]} | {validation["coverage"]["MSFT"]["latest_asof_date"]} | {validation["continuity_by_ticker"]["MSFT"]["max_gap_days"]} |
| NVDA | {validation["coverage"]["NVDA"]["line_rows"]} | {validation["coverage"]["NVDA"]["band_rows"]} | {validation["coverage"]["NVDA"]["latest_asof_date"]} | {validation["continuity_by_ticker"]["NVDA"]["max_gap_days"]} |

## 7. Manifest

주요 manifest:
- feature_version: `{manifest["feature_version"]}`
- source_data_hash: `{manifest["source_data_hash"]}`
- price_snapshot_hash: `{manifest["input_price_hash"][:16]}...`
- indicator_snapshot_hash: `{manifest["input_indicator_hash"][:16]}...`
- row_count: `{manifest["row_count"]}`
- ticker_count: `{manifest["ticker_count"]}`

## 8. 금지 작업 확인

| 항목 | 발생 |
|---|---:|
| DB write | {metrics["forbidden_actions_observed"]["db_write"]} |
| Supabase upload | {metrics["forbidden_actions_observed"]["supabase_upload"]} |
| 기존 predictions 수정/삭제 | {metrics["forbidden_actions_observed"]["prediction_row_modify_delete"]} |
| 모델 학습 | {metrics["forbidden_actions_observed"]["model_training"]} |
| W&B | {metrics["forbidden_actions_observed"]["wandb"]} |
| composite | {metrics["forbidden_actions_observed"]["composite"]} |
| 프론트 수정 | {metrics["forbidden_actions_observed"]["frontend_modify"]} |
| fake data | {metrics["forbidden_actions_observed"]["fake_data"]} |

## 9. 다음 단계

이 parquet는 프론트 연결 CP로 넘길 수 있다. 다음 CP에서는 backend API가 `product_prediction_history_1D.parquet`를 ticker 단위로 읽어 `/product-predictions/history` 형태로 제공하고, 프론트는 기존 `/predictions/history` 대신 신규 제품 history endpoint를 사용하면 된다.

## 10. 실행 명령

```powershell
.\\.venv\\Scripts\\python.exe scripts\\cp128_product_rolling_prediction_history_replay.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP128 제품용 1D rolling prediction history replay 생성")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-parquet", default=str(OUTPUT_PARQUET))
    parser.add_argument("--output-manifest", default=str(OUTPUT_MANIFEST))
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    stock, price, indicators = load_local_frames()
    tickers = resolve_tickers(args, stock)
    latest_date = pd.Timestamp(indicators["date"].max())
    asof_start, asof_end = asof_window(indicators[indicators["ticker"].isin(tickers)], latest_date=latest_date, lookback_days=args.lookback_days)
    source_data_hash = resolve_data_fingerprint(TIMEFRAME, tickers=tickers, market_data_provider=PROVIDER)
    price_hash = file_sha256(SNAPSHOT_DIR / "price_data_yfinance.parquet")
    indicator_hash = file_sha256(SNAPSHOT_DIR / "indicators_yfinance_1D.parquet")

    line_frame, line_diag = infer_role_history(
        role="line",
        run_id=LINE_RUN_ID,
        checkpoint_path=LINE_CHECKPOINT,
        tickers=tickers,
        price=price,
        indicators=indicators,
        asof_start=asof_start,
        asof_end=asof_end,
        source_data_hash=source_data_hash,
        batch_size=args.batch_size,
    )
    band_frame, band_diag = infer_role_history(
        role="band",
        run_id=BAND_RUN_ID,
        checkpoint_path=BAND_CHECKPOINT,
        tickers=tickers,
        price=price,
        indicators=indicators,
        asof_start=asof_start,
        asof_end=asof_end,
        source_data_hash=source_data_hash,
        batch_size=args.batch_size,
    )

    history = pd.concat([line_frame, band_frame], ignore_index=True).sort_values(["ticker", "role", "asof_date"])
    history = history.reset_index(drop=True)
    output_parquet = Path(args.output_parquet)
    output_manifest = Path(args.output_manifest)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_parquet, index=False)

    validation = validate_history(history, tickers, asof_end.strftime("%Y-%m-%d"))
    manifest = {
        "schema_version": "product_prediction_history_manifest_v1",
        "timeframe": TIMEFRAME,
        "source": SOURCE,
        "provider": PROVIDER,
        "provider_adjustment_policy": provider_adjustment_policy(PROVIDER),
        "line_run_id": LINE_RUN_ID,
        "band_run_id": BAND_RUN_ID,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "source_data_hash": source_data_hash,
        "model_feature_hash": source_data_hash,
        "input_price_hash": price_hash,
        "input_indicator_hash": indicator_hash,
        "ticker_count": int(history["ticker"].nunique()),
        "tickers": tickers,
        "asof_start": asof_start.strftime("%Y-%m-%d"),
        "asof_end": asof_end.strftime("%Y-%m-%d"),
        "display_horizon": DISPLAY_HORIZON,
        "row_count": int(len(history)),
        "role_counts": validation["role_counts"],
        "created_at": utc_now_iso(),
        "output_parquet": str(output_parquet.relative_to(ROOT_DIR)),
        "contract": "docs/product_prediction_history_contract.md",
    }
    write_json(output_manifest, manifest)

    elapsed = time.perf_counter() - started
    pass_conditions = [
        not validation["missing_columns"],
        validation["duplicate_ticker_role_asof_source"] == 0,
        validation["asof_monotonic_by_ticker_role"] is True,
        validation["display_horizon_unique"] == [DISPLAY_HORIZON],
        validation["source_values"] == [SOURCE],
        validation["line_run_ids"] == [LINE_RUN_ID],
        validation["band_run_ids"] == [BAND_RUN_ID],
        validation["line_value_non_finite"] == 0,
        validation["band_lower_non_finite"] == 0,
        validation["band_upper_non_finite"] == 0,
        validation["lower_lte_upper_violations"] == 0,
        validation["contains_forecast_arrays"] is False,
        validation["contains_latest_live_source"] is False,
        validation["contains_test_split_source"] is False,
        all(item["latest_asof_date"] == asof_end.strftime("%Y-%m-%d") for item in validation["coverage"].values()),
    ]
    status = "PASS" if all(pass_conditions) else "FAIL"
    metrics = {
        "cp": "CP128-DG",
        "generated_at": utc_now_iso(),
        "scope": {
            "timeframe": TIMEFRAME,
            "tickers": tickers,
            "ticker_count": len(tickers),
            "asof_start": asof_start.strftime("%Y-%m-%d"),
            "asof_end": asof_end.strftime("%Y-%m-%d"),
            "lookback_days": args.lookback_days,
            "display_horizon": DISPLAY_HORIZON,
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "source": SOURCE,
            "provider": PROVIDER,
        },
        "role_diagnostics": {
            "line": line_diag,
            "band": band_diag,
        },
        "validation": validation,
        "manifest": manifest,
        "output": {
            "parquet_path": str(output_parquet.relative_to(ROOT_DIR)),
            "manifest_path": str(output_manifest.relative_to(ROOT_DIR)),
            "parquet_bytes": int(output_parquet.stat().st_size),
            "manifest_bytes": int(output_manifest.stat().st_size),
        },
        "forbidden_actions_observed": {
            "db_write": False,
            "supabase_upload": False,
            "prediction_row_modify_delete": False,
            "model_training": False,
            "wandb": False,
            "composite": False,
            "frontend_modify": False,
            "fake_data": False,
        },
        "final_decision": {
            "status": status,
            "summary": (
                "제품용 1D rolling prediction history parquet를 생성했고 CP127 계약을 통과했다."
                if status == "PASS"
                else "제품용 1D rolling prediction history 생성 검증 중 실패 조건이 있다."
            ),
            "elapsed_seconds": round(elapsed, 3),
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
