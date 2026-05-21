from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp100_yfinance_1w_1m_local_snapshot"
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp100_yfinance_1w_1m_local_snapshot_metrics.json"
REQUIRED_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
RATIO_COLUMNS = ["open_ratio", "high_ratio", "low_ratio"]
PRICE_NUMERIC_COLUMNS = ["open", "high", "low", "close", "adjusted_close", "volume"]


os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(DEFAULT_SNAPSHOT_DIR))
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai import preprocessing as ai_preprocessing  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SOURCE_FEATURE_COLUMNS,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from backend.app.services.feature_svc import (  # noqa: E402
    build_features,
    latest_complete_period_end,
    resample_price_frame,
)
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402


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
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def checksum_frame(frame: pd.DataFrame, columns: list[str]) -> str | None:
    if frame.empty:
        return None
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    ordered = frame.sort_values([column for column in ["ticker", "timeframe", "date", "source"] if column in frame.columns])
    payload = ordered[existing].to_dict(orient="records")
    return hashlib.sha256(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def load_price_snapshot(snapshot_dir: Path) -> pd.DataFrame:
    path = snapshot_dir / "price_data_yfinance.parquet"
    if not path.exists():
        raise FileNotFoundError(f"1D yfinance price snapshot이 없습니다: {path}")
    frame = pd.read_parquet(path)
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return frame


def normalize_snapshot_columns(frame: pd.DataFrame, timeframe: str, *, updated_at: str) -> pd.DataFrame:
    working = frame.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["timeframe"] = timeframe
    working["source"] = "yfinance"
    working["provider"] = "yfinance"
    working["provider_adjustment_policy"] = provider_adjustment_policy("yfinance")
    working["updated_at"] = updated_at
    if "amount" not in working.columns:
        working["amount"] = pd.to_numeric(working["close"], errors="coerce") * pd.to_numeric(
            working["volume"],
            errors="coerce",
        )
    return working.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date", "source"],
        keep="last",
    ).reset_index(drop=True)


def build_resampled_price_snapshot(price_1d: pd.DataFrame, timeframe: str, *, updated_at: str) -> pd.DataFrame:
    resampled = resample_price_frame(price_1d, timeframe)
    if resampled.empty:
        return resampled
    return normalize_snapshot_columns(resampled, timeframe, updated_at=updated_at)


def build_indicator_snapshot(price_1d: pd.DataFrame, timeframe: str, *, updated_at: str) -> pd.DataFrame:
    indicators = build_features(price_df=price_1d, timeframe=timeframe)
    if indicators.empty:
        return indicators
    indicators = indicators.copy()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    indicators["source"] = "yfinance"
    indicators["provider"] = "yfinance"
    indicators["updated_at"] = updated_at
    return indicators.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    ).reset_index(drop=True)


def validate_price_snapshot(frame: pd.DataFrame, timeframe: str, latest_daily_date: pd.Timestamp) -> dict[str, Any]:
    required = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        "provider",
        "provider_adjustment_policy",
    ]
    missing = [column for column in required if column not in frame.columns]
    if frame.empty or missing:
        return {"status": "FAIL", "row_count": int(len(frame)), "missing_columns": missing}

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    numeric = working[PRICE_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    high = numeric["high"]
    low = numeric["low"]
    open_ = numeric["open"]
    close = numeric["close"]
    adjusted_close = numeric["adjusted_close"]
    high_floor = pd.concat([open_, close, adjusted_close], axis=1).max(axis=1)
    low_ceiling = pd.concat([open_, close, adjusted_close], axis=1).min(axis=1)
    duplicate_count = int(working.duplicated(subset=["ticker", "date", "source"]).sum())
    source_missing = int((working["source"].astype(str).str.lower() != "yfinance").sum())
    provider_missing = int((working["provider"].astype(str).str.lower() != "yfinance").sum())
    policy_missing = int(
        (working["provider_adjustment_policy"].astype(str) != provider_adjustment_policy("yfinance")).sum()
    )
    cutoff = latest_complete_period_end(latest_daily_date, timeframe)
    date_max = working["date"].max()
    partial_violation = bool(cutoff is not None and date_max > cutoff)
    epsilon = 1e-8
    violations = {
        "date_null": int(working["date"].isna().sum()),
        "numeric_non_finite": int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()),
        "high_below_low": int((high + epsilon < low).sum()),
        "high_below_open_close_adjusted_close": int((high + epsilon < high_floor).sum()),
        "low_above_open_close_adjusted_close": int((low - epsilon > low_ceiling).sum()),
        "duplicate_ticker_date_source": duplicate_count,
        "source_not_yfinance": source_missing,
        "provider_not_yfinance": provider_missing,
        "provider_adjustment_policy_mismatch": policy_missing,
        "partial_period_rows": int((working["date"] > cutoff).sum()) if cutoff is not None else None,
    }
    passed = all((value == 0 or value is None) for value in violations.values()) and not partial_violation
    return {
        "status": "PASS" if passed else "FAIL",
        "row_count": int(len(working)),
        "ticker_count": int(working["ticker"].nunique()),
        "date_min": working["date"].min().strftime("%Y-%m-%d"),
        "date_max": date_max.strftime("%Y-%m-%d"),
        "latest_complete_period_end": None if cutoff is None else cutoff.strftime("%Y-%m-%d"),
        "violations": violations,
    }


def validate_indicator_snapshot(frame: pd.DataFrame, timeframe: str, latest_daily_date: pd.Timestamp) -> dict[str, Any]:
    if frame.empty:
        return {"status": "FAIL", "row_count": 0, "error": "empty_indicator_snapshot"}
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    feature_columns = [column for column in SOURCE_FEATURE_COLUMNS if column in working.columns]
    feature_values = working[feature_columns].apply(pd.to_numeric, errors="coerce")
    non_finite_count = int((~np.isfinite(feature_values.to_numpy(dtype=float))).sum())
    duplicate_count = int(working.duplicated(subset=["ticker", "timeframe", "date", "source"]).sum())
    ratio_frame = working[RATIO_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ratio_abs = ratio_frame.abs()
    ratio_p99 = {column: float(ratio_abs[column].quantile(0.99)) for column in RATIO_COLUMNS}
    ratio_max = {column: float(ratio_abs[column].max()) for column in RATIO_COLUMNS}
    cutoff = latest_complete_period_end(latest_daily_date, timeframe)
    date_max = working["date"].max()
    atr_non_null = int(working["atr_ratio"].notna().sum()) if "atr_ratio" in working.columns else 0
    atr_coverage = float(atr_non_null / len(working)) if len(working) else 0.0
    ratio_pass = all(value <= 1.0 for value in ratio_p99.values()) and all(value <= 5.0 for value in ratio_max.values())
    violations = {
        "date_null": int(working["date"].isna().sum()),
        "feature_non_finite": non_finite_count,
        "duplicate_ticker_timeframe_date_source": duplicate_count,
        "source_not_yfinance": int((working["source"].astype(str).str.lower() != "yfinance").sum()),
        "provider_not_yfinance": int((working["provider"].astype(str).str.lower() != "yfinance").sum()),
        "partial_period_rows": int((working["date"] > cutoff).sum()) if cutoff is not None else None,
        "ratio_sanity_failed": 0 if ratio_pass else 1,
    }
    passed = all((value == 0 or value is None) for value in violations.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "row_count": int(len(working)),
        "ticker_count": int(working["ticker"].nunique()),
        "date_min": working["date"].min().strftime("%Y-%m-%d"),
        "date_max": date_max.strftime("%Y-%m-%d"),
        "latest_complete_period_end": None if cutoff is None else cutoff.strftime("%Y-%m-%d"),
        "atr_ratio_non_null": atr_non_null,
        "atr_ratio_coverage": atr_coverage,
        "ratio_p99_abs": ratio_p99,
        "ratio_max_abs": ratio_max,
        "violations": violations,
    }


def run_split_gate(
    *,
    timeframe: str,
    tickers: list[str],
    seq_len: int,
    horizon: int,
    min_fold_samples: int,
) -> dict[str, Any]:
    ai_preprocessing._PREPARED_SPLITS_CACHE.clear()
    try:
        with patch(
            "ai.preprocessing.fetch_frame",
            side_effect=AssertionError("DB fallback 금지"),
        ), patch(
            "ai.preprocessing._postgres_engine",
            side_effect=AssertionError("DB fallback 금지"),
        ):
            train, val, test, _, _, plan = prepare_dataset_splits(
                timeframe=timeframe,
                seq_len=seq_len,
                horizon=horizon,
                tickers=tickers,
                min_fold_samples=min_fold_samples,
                market_data_provider="yfinance",
            )
        return {
            "status": "PASS",
            "timeframe": timeframe,
            "seq_len": seq_len,
            "horizon": horizon,
            "min_fold_samples": min_fold_samples,
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "eligible_ticker_count": len(plan.eligible_tickers),
            "eligible_tickers": plan.eligible_tickers,
            "excluded_reasons_sample": dict(list(plan.excluded_reasons.items())[:10]),
            "source_data_hash": plan.source_data_hash,
            "db_fallback_observed": False,
        }
    except AssertionError as exc:
        return {
            "status": "FAIL",
            "timeframe": timeframe,
            "seq_len": seq_len,
            "horizon": horizon,
            "min_fold_samples": min_fold_samples,
            "error": str(exc),
            "db_fallback_observed": True,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "timeframe": timeframe,
            "seq_len": seq_len,
            "horizon": horizon,
            "min_fold_samples": min_fold_samples,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "db_fallback_observed": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP100 yfinance 1W/1M local snapshot bootstrap")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--limit-tickers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    price_1d = load_price_snapshot(snapshot_dir)
    if args.limit_tickers is not None:
        selected_tickers = sorted(price_1d["ticker"].unique().tolist())[: args.limit_tickers]
        price_1d = price_1d[price_1d["ticker"].isin(selected_tickers)].copy()
    latest_daily_date = pd.to_datetime(price_1d["date"]).max()
    source_updated_at = (
        str(price_1d["updated_at"].max())
        if "updated_at" in price_1d.columns and price_1d["updated_at"].notna().any()
        else latest_daily_date.isoformat()
    )
    tickers = sorted(price_1d["ticker"].unique().tolist())
    split_tickers = [ticker for ticker in REQUIRED_TICKERS if ticker in set(tickers)]
    if len(split_tickers) < len(REQUIRED_TICKERS):
        split_tickers = tickers[:5]

    metrics: dict[str, Any] = {
        "cp": "CP100-D",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scope": {
            "provider": "yfinance",
            "source": "yfinance",
            "input_price_snapshot": str(snapshot_dir / "price_data_yfinance.parquet"),
            "stock_info_snapshot": str(snapshot_dir / "stock_info.parquet"),
            "ticker_count": len(tickers),
            "date_min": pd.to_datetime(price_1d["date"]).min().strftime("%Y-%m-%d"),
            "date_max": latest_daily_date.strftime("%Y-%m-%d"),
            "split_gate_tickers": split_tickers,
        },
        "forbidden_actions_observed": {
            "supabase_price_data_read": False,
            "supabase_indicators_read": False,
            "supabase_write": False,
            "indicators_db_recompute": False,
            "model_training": False,
            "inference_save": False,
            "eodhd_fallback": False,
            "frontend_modification": False,
        },
    }

    generated: dict[str, Any] = {}
    validation: dict[str, Any] = {}

    for timeframe in ("1W", "1M"):
        price_snapshot = build_resampled_price_snapshot(price_1d, timeframe, updated_at=source_updated_at)
        indicator_snapshot = build_indicator_snapshot(price_1d, timeframe, updated_at=source_updated_at)
        price_path = snapshot_dir / f"price_data_yfinance_{timeframe}.parquet"
        indicator_path = snapshot_dir / f"indicators_yfinance_{timeframe}.parquet"
        price_snapshot.to_parquet(price_path, index=False)
        indicator_snapshot.to_parquet(indicator_path, index=False)

        generated[timeframe] = {
            "price_path": str(price_path),
            "price_bytes": price_path.stat().st_size,
            "indicator_path": str(indicator_path),
            "indicator_bytes": indicator_path.stat().st_size,
            "price_checksum": checksum_frame(
                price_snapshot,
                ["ticker", "timeframe", "date", "open", "high", "low", "close", "adjusted_close", "volume", "source"],
            ),
            "indicator_checksum": checksum_frame(
                indicator_snapshot,
                ["ticker", "timeframe", "date", *SOURCE_FEATURE_COLUMNS, "atr_ratio", "source"],
            ),
        }
        validation[timeframe] = {
            "price": validate_price_snapshot(price_snapshot, timeframe, latest_daily_date),
            "indicators": validate_indicator_snapshot(indicator_snapshot, timeframe, latest_daily_date),
        }

    metrics["generated_snapshots"] = generated
    metrics["snapshot_validation"] = validation
    metrics["feature_contract"] = {
        "MODEL_N_FEATURES": MODEL_N_FEATURES,
        "FEATURE_CONTRACT_VERSION": FEATURE_CONTRACT_VERSION,
        "atr_ratio_in_MODEL_FEATURE_COLUMNS": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "MODEL_FEATURE_COLUMNS": list(MODEL_FEATURE_COLUMNS),
    }

    data_hashes: dict[str, Any] = {}
    for timeframe in ("1D", "1W", "1M"):
        try:
            data_hashes[timeframe] = resolve_data_fingerprint(
                timeframe,
                tickers=split_tickers,
                market_data_provider="yfinance",
            )
        except Exception as exc:
            data_hashes[timeframe] = {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    metrics["source_data_hashes"] = data_hashes
    metrics["source_data_hashes_separated"] = len({value for value in data_hashes.values() if isinstance(value, str)}) == 3

    metrics["split_gates"] = {
        "1W_strict": run_split_gate(
            timeframe="1W",
            tickers=split_tickers,
            seq_len=104,
            horizon=4,
            min_fold_samples=50,
        ),
        "1M_strict": run_split_gate(
            timeframe="1M",
            tickers=split_tickers,
            seq_len=24,
            horizon=3,
            min_fold_samples=50,
        ),
        "1M_experimental": run_split_gate(
            timeframe="1M",
            tickers=split_tickers,
            seq_len=24,
            horizon=3,
            min_fold_samples=5,
        ),
    }

    price_pass = all(item["price"]["status"] == "PASS" for item in validation.values())
    indicator_pass = all(item["indicators"]["status"] == "PASS" for item in validation.values())
    hash_pass = bool(metrics["source_data_hashes_separated"])
    split_1w_pass = metrics["split_gates"]["1W_strict"]["status"] == "PASS"
    split_1m_experimental_pass = metrics["split_gates"]["1M_experimental"]["status"] == "PASS"
    split_1m_strict_pass = metrics["split_gates"]["1M_strict"]["status"] == "PASS"
    db_fallback_fail = any(gate.get("db_fallback_observed") for gate in metrics["split_gates"].values())
    hard_fail = not (price_pass and indicator_pass and hash_pass and split_1w_pass and split_1m_experimental_pass) or db_fallback_fail
    metrics["final_decision"] = {
        "status": "FAIL" if hard_fail else "PASS" if split_1m_strict_pass else "WARN",
        "price_pass": price_pass,
        "indicator_pass": indicator_pass,
        "source_data_hashes_separated": hash_pass,
        "one_week_strict_split_pass": split_1w_pass,
        "one_month_strict_split_pass": split_1m_strict_pass,
        "one_month_experimental_split_pass": split_1m_experimental_pass,
        "db_fallback_observed": db_fallback_fail,
        "warning": None if split_1m_strict_pass else "1M은 snapshot/local split 경로는 동작하지만 기본 min_fold_samples=50 기준에서는 row 수가 부족합니다.",
    }

    write_json(Path(args.metrics_path), metrics)
    write_json(log_dir / "cp100_yfinance_1w_1m_local_snapshot_metrics.json", metrics)
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
