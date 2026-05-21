from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_CALENDAR_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    SOURCE_FEATURE_COLUMNS,
    SequenceDataset,
    _build_target_frame,
    _enforce_feature_finite_contract,
    append_calendar_features,
    build_dataset_plan,
    build_registry,
    default_horizon,
    normalize_sequence_splits,
    normalize_ai_timeframe,
    split_sequence_dataset_by_plan,
)
from ai.targets import normalize_target_type  # noqa: E402
from ai.ticker_registry import lookup_id  # noqa: E402
from ai.train import FUTURE_COVARIATE_DIM, TrainConfig, resolve_feature_columns, run_training, summarize_dataset_plan  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp146_lm_eodhd500_line_full_training_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp146_lm_eodhd500_line_full_training_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp146_lm_eodhd500_line_candidate_registry.json"
CSV_PATH = PROJECT_ROOT / "docs" / "cp146_lm_eodhd500_line_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp146_lm_eodhd500_line_full_training_logs"
ALIAS_DIR = LOG_DIR / "snapshot_alias"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"

CONTEXT_HASH = "1aa6452d82369cc6"
WANDB_PROJECT_DEFAULT = "lens-eodhd500"

LINE_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "direction_accuracy",
]

CANDIDATES = [
    {
        "name": "1d_patchtst_h5_pvv_p32_s16_beta2",
        "model": "patchtst",
        "timeframe": "1D",
        "horizon": 5,
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "feature_set": "price_volatility_volume",
        "epochs": 5,
        "optional": False,
    },
    {
        "name": "1d_patchtst_h5_no_fundamentals_p32_s16_beta2",
        "model": "patchtst",
        "timeframe": "1D",
        "horizon": 5,
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "feature_set": "no_fundamentals",
        "epochs": 5,
        "optional": False,
    },
    {
        "name": "1d_patchtst_h5_pvv_dense_beta2",
        "model": "patchtst",
        "timeframe": "1D",
        "horizon": 5,
        "seq_len": 252,
        "patch_len": 16,
        "patch_stride": 8,
        "feature_set": "price_volatility_volume",
        "epochs": 5,
        "optional": False,
    },
    {
        "name": "1w_patchtst_h4_pvv_p16_s8_beta2",
        "model": "patchtst",
        "timeframe": "1W",
        "horizon": 4,
        "seq_len": 104,
        "patch_len": 16,
        "patch_stride": 8,
        "feature_set": "price_volatility_volume",
        "epochs": 5,
        "optional": False,
    },
    {
        "name": "1w_patchtst_h4_no_fundamentals_p16_s8_beta2",
        "model": "patchtst",
        "timeframe": "1W",
        "horizon": 4,
        "seq_len": 104,
        "patch_len": 16,
        "patch_stride": 8,
        "feature_set": "no_fundamentals",
        "epochs": 5,
        "optional": False,
    },
    {
        "name": "1w_patchtst_h6_pvv_p16_s8_beta2",
        "model": "patchtst",
        "timeframe": "1W",
        "horizon": 6,
        "seq_len": 104,
        "patch_len": 16,
        "patch_stride": 8,
        "feature_set": "price_volatility_volume",
        "epochs": 5,
        "optional": True,
    },
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any, digits: int = 6) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _sha256_file(path: Path, *, short: int = 16) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:short]


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def ensure_snapshot_alias() -> dict[str, Any]:
    mappings = {
        "price_data_eodhd.parquet": PROJECT_ROOT / "data" / "parquet" / "price_data_eodhd_500.parquet",
        "indicators_eodhd_1D.parquet": PROJECT_ROOT / "data" / "parquet" / "indicators_eodhd_1D_500.parquet",
        "indicators_eodhd_1W.parquet": PROJECT_ROOT / "data" / "parquet" / "indicators_eodhd_1W_500.parquet",
        "stock_info.parquet": PROJECT_ROOT / "data" / "parquet" / "stock_info.parquet",
    }
    results: dict[str, Any] = {}
    for target_name, source in mappings.items():
        target = ALIAS_DIR / target_name
        if not source.exists():
            raise FileNotFoundError(str(source))
        mode = _link_or_copy(source, target)
        results[target_name] = {
            "source": str(source),
            "target": str(target),
            "mode": mode,
            "source_sha16": _sha256_file(source),
            "target_sha16": _sha256_file(target),
            "bytes": source.stat().st_size,
        }
    return results


def _set_local_env(*, wandb_mode: str | None = None, wandb_project: str | None = None) -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
    os.environ["LENS_USE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_DATA_BACKEND"] = "local"
    os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(ALIAS_DIR)
    if wandb_mode is not None:
        os.environ["WANDB_MODE"] = wandb_mode
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project


def _combined_hash(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _date_bounds(series: pd.Series) -> dict[str, Any]:
    dates = pd.to_datetime(series, errors="coerce")
    return {
        "date_min": dates.min().date().isoformat() if not dates.empty and pd.notna(dates.min()) else None,
        "date_max": dates.max().date().isoformat() if not dates.empty and pd.notna(dates.max()) else None,
    }


def _finite_frame_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if frame.empty or not columns:
        return {"element_count": 0, "nonfinite_count": 0, "missing_columns": columns}
    missing = [column for column in columns if column not in frame.columns]
    present = [column for column in columns if column in frame.columns]
    if not present:
        return {"element_count": 0, "nonfinite_count": 0, "missing_columns": missing}
    values = frame[present].to_numpy(dtype="float64", copy=False)
    nonfinite = int((~np.isfinite(values)).sum())
    return {"element_count": int(values.size), "nonfinite_count": nonfinite, "missing_columns": missing}


def _parquet_meta(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha16": _sha256_file(path),
        "num_rows": int(parquet.metadata.num_rows),
        "columns": parquet.schema.names,
    }


def _timeframe_quick_check(timeframe: str, *, horizon: int, seq_len: int) -> dict[str, Any]:
    indicator_path = ALIAS_DIR / f"indicators_eodhd_{timeframe}.parquet"
    price_path = ALIAS_DIR / "price_data_eodhd.parquet"
    indicator_meta = _parquet_meta(indicator_path)
    price_meta = _parquet_meta(price_path)
    feature_sets = {
        "price_volatility_volume": resolve_feature_columns("price_volatility_volume"),
        "no_fundamentals": resolve_feature_columns("no_fundamentals"),
    }
    required_feature_columns = sorted({column for columns in feature_sets.values() for column in columns})
    indicator_columns = ["ticker", "date", "timeframe", "source", "provider", *required_feature_columns]
    indicator_columns = [column for column in indicator_columns if column in indicator_meta["columns"]]
    price_columns = ["ticker", "date", "close", "adjusted_close", "source", "provider"]
    price_columns = [column for column in price_columns if column in price_meta["columns"]]
    indicator = pd.read_parquet(indicator_path, columns=indicator_columns)
    price = pd.read_parquet(price_path, columns=price_columns)
    if "timeframe" in indicator.columns:
        indicator = indicator[indicator["timeframe"].astype(str).str.upper() == timeframe.upper()].copy()
    indicator["ticker"] = indicator["ticker"].astype(str).str.upper()
    price["ticker"] = price["ticker"].astype(str).str.upper()
    indicator_counts = indicator.groupby("ticker").size()
    eligible_estimate = int((indicator_counts >= (seq_len + horizon + 12)).sum())
    price_ticker_count = int(price["ticker"].nunique()) if "ticker" in price.columns else 0
    source_data_hash = _combined_hash(
        {
            "provider": "eodhd",
            "timeframe": timeframe,
            "feature_version": FEATURE_CONTRACT_VERSION,
            "price_sha16": price_meta["sha16"],
            "indicator_sha16": indicator_meta["sha16"],
            "context_checksum": CONTEXT_HASH,
        }
    )
    return {
        "source_data_hash": source_data_hash,
        "indicator_meta": {key: value for key, value in indicator_meta.items() if key != "columns"},
        "price_meta": {key: value for key, value in price_meta.items() if key != "columns"},
        "indicator_rows": int(len(indicator)),
        "indicator_ticker_count": int(indicator["ticker"].nunique()),
        "price_ticker_count": price_ticker_count,
        "eligible_ticker_count_estimate": eligible_estimate,
        "date_bounds": {"indicator": _date_bounds(indicator["date"]), "price": _date_bounds(price["date"])},
        "feature_sets": {
            name: {
                "feature_column_count": len(columns),
                "finite": _finite_frame_summary(indicator, columns),
            }
            for name, columns in feature_sets.items()
        },
        "target_proxy": {
            "close_finite": _finite_frame_summary(price, ["close"]),
            "adjusted_close_finite": _finite_frame_summary(price, ["adjusted_close"]),
            "note": "빠른 preflight에서는 전체 시퀀스/타깃 텐서를 만들지 않고 가격 finite와 행 수로 target 생성 가능성을 확인한다.",
        },
    }


def build_preflight() -> dict[str, Any]:
    alias = ensure_snapshot_alias()
    _set_local_env(wandb_mode=os.environ.get("WANDB_MODE", "offline"), wandb_project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT))
    checks: dict[str, Any] = {}
    for timeframe, horizon, seq_len, feature_set in [
        ("1D", 5, 252, "price_volatility_volume"),
        ("1W", 4, 104, "price_volatility_volume"),
    ]:
        checks[timeframe] = _timeframe_quick_check(timeframe, horizon=horizon, seq_len=seq_len)
        checks[timeframe]["primary_feature_set"] = feature_set
    pass_gate = bool(
        FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and len(MODEL_FEATURE_COLUMNS) == 36
        and "atr_ratio" not in MODEL_FEATURE_COLUMNS
        and all(
            feature_set["finite"]["nonfinite_count"] == 0
            for check in checks.values()
            for feature_set in (check.get("feature_sets") or {}).values()
        )
        and all((check.get("target_proxy") or {}).get("close_finite", {}).get("nonfinite_count") == 0 for check in checks.values())
    )
    payload = {
        "cp": "CP146-LM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PREFLIGHT_PASS" if pass_gate else "PREFLIGHT_FAIL",
        "preflight_gate_pass": pass_gate,
        "provider": "eodhd",
        "context": "eodhd_500",
        "context_checksum": CONTEXT_HASH,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "model_n_features": len(MODEL_FEATURE_COLUMNS),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "snapshot_alias": alias,
        "checks": checks,
    }
    _write_json(PREFLIGHT_PATH, payload)
    return payload


def _candidate_by_name(name: str) -> dict[str, Any]:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return candidate
    raise KeyError(name)


def _run_name(candidate: dict[str, Any]) -> str:
    return (
        f"cp146_{candidate['timeframe']}_{candidate['model']}_{candidate['feature_set']}"
        f"_h{candidate['horizon']}_p{candidate['patch_len']}_s{candidate['patch_stride']}_beta2"
    )


def _log_progress(candidate_dir: Path, stage: str, payload: dict[str, Any] | None = None) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **(payload or {}),
    }
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with (candidate_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(record), ensure_ascii=False), flush=True)


def _filter_cp146_provider(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    if frame.empty or "source" not in frame.columns:
        return frame.copy()
    source = frame["source"].astype("string").str.lower()
    if provider == "eodhd":
        return frame[source.isna() | (source == "eodhd")].copy()
    return frame[source == provider].copy()


def _preflight_source_hash(timeframe: str) -> str:
    if PREFLIGHT_PATH.exists():
        payload = _read_json(PREFLIGHT_PATH)
        checks = payload.get("checks") if isinstance(payload, dict) else {}
        timeframe_check = checks.get(timeframe) if isinstance(checks, dict) else {}
        source_hash = timeframe_check.get("source_data_hash") if isinstance(timeframe_check, dict) else None
        if source_hash:
            return str(source_hash)
    source_paths = [
        ALIAS_DIR / "price_data_eodhd.parquet",
        ALIAS_DIR / f"indicators_eodhd_{timeframe}.parquet",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _read_cp146_index_frame(timeframe: str, provider: str, candidate_dir: Path) -> pd.DataFrame:
    indicator_path = ALIAS_DIR / f"indicators_eodhd_{timeframe}.parquet"
    price_path = ALIAS_DIR / "price_data_eodhd.parquet"
    _log_progress(candidate_dir, "index_parquet_read_start", {"indicator_path": str(indicator_path), "price_path": str(price_path)})
    indicators = pq.read_table(
        indicator_path,
        columns=["ticker", "timeframe", "date", "source", "provider"],
    ).to_pandas()
    prices = pq.read_table(
        price_path,
        columns=["ticker", "date", "source", "provider"],
    ).to_pandas()
    _log_progress(
        candidate_dir,
        "index_parquet_read_done",
        {"indicator_rows": int(len(indicators)), "price_rows": int(len(prices))},
    )
    indicators = _filter_cp146_provider(indicators, provider)
    prices = _filter_cp146_provider(prices, provider)
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["timeframe"] = indicators["timeframe"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"] == timeframe].dropna(subset=["date"])
    prices = prices.dropna(subset=["date"]).drop_duplicates(subset=["ticker", "date"])
    index_frame = indicators.merge(prices[["ticker", "date"]], on=["ticker", "date"], how="inner")
    return (
        index_frame[["ticker", "timeframe", "date"]]
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "timeframe", "date"])
        .reset_index(drop=True)
    )


def _read_cp146_training_frames(timeframe: str, provider: str, eligible_tickers: list[str], candidate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicator_path = ALIAS_DIR / f"indicators_eodhd_{timeframe}.parquet"
    price_path = ALIAS_DIR / "price_data_eodhd.parquet"
    ticker_set = {ticker.upper() for ticker in eligible_tickers}
    indicator_columns = ["ticker", "timeframe", "date", *SOURCE_FEATURE_COLUMNS, "source", "provider"]
    price_columns = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "provider", "provider_adjustment_policy", "updated_at"]
    _log_progress(candidate_dir, "training_parquet_read_start", {"indicator_path": str(indicator_path), "price_path": str(price_path)})
    feature_df = pq.read_table(indicator_path, columns=indicator_columns).to_pandas()
    price_df = pq.read_table(price_path, columns=price_columns).to_pandas()
    _log_progress(
        candidate_dir,
        "training_parquet_read_done",
        {"feature_rows_raw": int(len(feature_df)), "price_rows_raw": int(len(price_df))},
    )
    feature_df = _filter_cp146_provider(feature_df, provider)
    price_df = _filter_cp146_provider(price_df, provider)
    feature_df["ticker"] = feature_df["ticker"].astype(str).str.upper()
    price_df["ticker"] = price_df["ticker"].astype(str).str.upper()
    feature_df["timeframe"] = feature_df["timeframe"].astype(str).str.upper()
    feature_df = feature_df[(feature_df["timeframe"] == timeframe) & feature_df["ticker"].isin(ticker_set)].copy()
    price_df = price_df[price_df["ticker"].isin(ticker_set)].copy()
    return feature_df, price_df


def _append_cp146_calendar_features(features: pd.DataFrame, candidate_dir: Path) -> pd.DataFrame:
    enriched = features.copy()
    dates = pd.to_datetime(enriched["date"], errors="coerce")
    unique_dates = pd.DatetimeIndex(pd.Series(dates.dropna().unique()).sort_values())
    _log_progress(candidate_dir, "dataset_calendar_unique_start", {"unique_dates": int(len(unique_dates))})
    weekday = unique_dates.weekday.to_numpy()
    month = unique_dates.month.to_numpy()
    day = unique_dates.day.to_numpy()
    dow_angle = 2.0 * math.pi * weekday / 5.0
    month_angle = 2.0 * math.pi * month / 12.0

    month_end = pd.DatetimeIndex([timestamp + pd.offsets.BMonthEnd(0) for timestamp in unique_dates])
    quarter_end = pd.DatetimeIndex([timestamp + pd.offsets.BQuarterEnd(0) for timestamp in unique_dates])

    def _business_days_until(targets: pd.DatetimeIndex) -> np.ndarray:
        values = []
        for current, target in zip(unique_dates, targets, strict=False):
            values.append(len(pd.bdate_range(current.normalize(), target.normalize())) - 1)
        return np.asarray(values, dtype=np.int64)

    calendar = pd.DataFrame(
        {
            "date": unique_dates,
            "day_of_week_sin": np.sin(dow_angle).astype("float32"),
            "day_of_week_cos": np.cos(dow_angle).astype("float32"),
            "month_sin": np.sin(month_angle).astype("float32"),
            "month_cos": np.cos(month_angle).astype("float32"),
            "is_month_end": (_business_days_until(month_end) <= 4).astype("float32"),
            "is_quarter_end": (_business_days_until(quarter_end) <= 4).astype("float32"),
            "is_opex_friday": ((weekday == 4) & (day >= 15) & (day <= 21)).astype("float32"),
        }
    )
    enriched = enriched.merge(calendar, on="date", how="left", sort=False)
    _log_progress(candidate_dir, "dataset_calendar_unique_done", {"feature_rows": int(len(enriched))})
    return enriched


def _build_cp146_lazy_sequence_dataset_with_progress(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    horizon: int,
    ticker_registry: dict[str, Any],
    include_future_covariate: bool,
    line_target_type: str,
    band_target_type: str,
    candidate_dir: Path,
) -> SequenceDataset:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    resolved_horizon = horizon or default_horizon(normalized_timeframe)
    resolved_line_target_type = normalize_target_type(line_target_type)
    resolved_band_target_type = normalize_target_type(band_target_type)
    if feature_df.empty or price_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다.")

    _log_progress(candidate_dir, "dataset_feature_prepare_start", {"feature_rows": int(len(feature_df))})
    features = feature_df.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["ticker", "date"])
    _log_progress(candidate_dir, "dataset_feature_sort_done", {"feature_rows": int(len(features))})
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"cp146_build_lazy_sequence_dataset:{normalized_timeframe}",
        validate_columns=None,
    )
    before_drop = len(features)
    features = features.dropna(subset=REQUIRED_FEATURE_COLUMNS)
    _log_progress(
        candidate_dir,
        "dataset_feature_required_drop_done",
        {"before_rows": int(before_drop), "after_rows": int(len(features))},
    )
    features = _append_cp146_calendar_features(features, candidate_dir)
    _log_progress(candidate_dir, "dataset_calendar_done", {"feature_rows": int(len(features))})
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"cp146_build_lazy_sequence_dataset:{normalized_timeframe}:calendar",
        validate_columns=MODEL_FEATURE_COLUMNS,
    )

    _log_progress(candidate_dir, "dataset_target_build_start", {"price_rows": int(len(price_df))})
    targets = _build_target_frame(price_df, normalized_timeframe)
    _log_progress(candidate_dir, "dataset_target_build_done", {"target_rows": int(len(targets))})

    _log_progress(candidate_dir, "dataset_merge_start", {"feature_rows": int(len(features)), "target_rows": int(len(targets))})
    merged = features.merge(targets, on=["ticker", "date"], how="inner")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = merged.dropna(subset=["target_close"])
    _log_progress(
        candidate_dir,
        "dataset_merge_done",
        {
            "merged_rows": int(len(merged)),
            "ticker_count": int(merged["ticker"].nunique()) if "ticker" in merged.columns else None,
        },
    )

    ticker_arrays: dict[str, dict[str, Any]] = {}
    sample_refs: list[tuple[str, int]] = []
    metadata_tickers: list[str] = []
    metadata_asof_dates: list[str] = []
    metadata_sample_indices: list[int] = []
    total_tickers = int(merged["ticker"].nunique()) if "ticker" in merged.columns else 0
    _log_progress(candidate_dir, "dataset_ticker_loop_start", {"ticker_count": total_tickers})

    processed = 0
    for ticker, ticker_frame in merged.groupby("ticker", sort=True):
        processed += 1
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        if len(ticker_frame) < (seq_len + resolved_horizon):
            continue

        ticker_key = str(ticker)
        feature_values = ticker_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
        closes = ticker_frame["target_close"].to_numpy(dtype="float32")
        dates = pd.to_datetime(ticker_frame["date"]).to_numpy()
        date_strings = pd.to_datetime(ticker_frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        ticker_arrays[ticker_key] = {
            "features": feature_values,
            "closes": closes,
            "dates": dates,
            "calendar": (
                ticker_frame[FUTURE_CALENDAR_COLUMNS].to_numpy(dtype="float32")
                if include_future_covariate
                else None
            ),
            "ticker_id": lookup_id(ticker_key, ticker_registry) if ticker_registry is not None else 0,
        }

        end_indices = np.arange(seq_len - 1, len(ticker_frame) - resolved_horizon, dtype=np.int64)
        if end_indices.size:
            valid_end_indices = end_indices[closes[end_indices] != 0.0]
        else:
            valid_end_indices = end_indices
        if valid_end_indices.size:
            sample_refs.extend((ticker_key, int(end_idx)) for end_idx in valid_end_indices.tolist())
            metadata_tickers.extend([ticker_key] * int(valid_end_indices.size))
            metadata_asof_dates.extend(date_strings[int(end_idx)] for end_idx in valid_end_indices.tolist())
            metadata_sample_indices.extend(range(int(valid_end_indices.size)))

        if processed == 1 or processed % 50 == 0 or processed == total_tickers:
            _log_progress(
                candidate_dir,
                "dataset_ticker_loop_progress",
                {
                    "processed_tickers": processed,
                    "total_tickers": total_tickers,
                    "sample_refs": len(sample_refs),
                },
            )

    if not sample_refs:
        raise ValueError("지정한 조건에서 시퀀스 샘플을 만들지 못했습니다.")

    metadata = pd.DataFrame(
        {
            "ticker": metadata_tickers,
            "timeframe": normalized_timeframe,
            "asof_date": metadata_asof_dates,
            "sample_index": metadata_sample_indices,
        }
    )
    if not metadata.empty:
        metadata["ticker"] = metadata["ticker"].astype("category")
    _log_progress(
        candidate_dir,
        "dataset_metadata_done",
        {"sample_refs": len(sample_refs), "metadata_rows": int(len(metadata)), "ticker_arrays": len(ticker_arrays)},
    )
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=sample_refs,
        metadata=metadata,
        seq_len=seq_len,
        horizon=resolved_horizon,
        include_future_covariate=include_future_covariate,
        line_target_type=resolved_line_target_type,
        band_target_type=resolved_band_target_type,
    )


def _prepare_dataset_splits_with_progress(config: TrainConfig, candidate_dir: Path) -> tuple:
    _log_progress(candidate_dir, "feature_index_start", {"timeframe": config.timeframe})
    index_frame = _read_cp146_index_frame(config.timeframe, config.market_data_provider, candidate_dir)
    _log_progress(
        candidate_dir,
        "feature_index_done",
        {
            "rows": int(len(index_frame)),
            "ticker_count": int(index_frame["ticker"].nunique()) if "ticker" in index_frame.columns else None,
        },
    )

    _log_progress(candidate_dir, "fingerprint_start", {"timeframe": config.timeframe})
    data_hash = _preflight_source_hash(config.timeframe)
    _log_progress(candidate_dir, "fingerprint_done", {"source_data_hash": data_hash})

    _log_progress(candidate_dir, "dataset_plan_start", {"seq_len": config.seq_len, "horizon": config.horizon})
    plan = build_dataset_plan(
        index_frame,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        market_data_provider=config.market_data_provider,
        source_data_hash=data_hash,
    )
    _log_progress(
        candidate_dir,
        "dataset_plan_done",
        {
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
            "estimated_usable_sample_count": plan.estimated_usable_sample_count,
            "ticker_registry_path": plan.ticker_registry_path,
        },
    )
    if not plan.eligible_tickers:
        raise ValueError("eligible_tickers가 비어 있습니다.")

    _log_progress(candidate_dir, "training_frames_start", {"eligible_ticker_count": len(plan.eligible_tickers)})
    feature_df, price_df = _read_cp146_training_frames(config.timeframe, config.market_data_provider, plan.eligible_tickers, candidate_dir)
    _log_progress(
        candidate_dir,
        "training_frames_done",
        {
            "feature_rows": int(len(feature_df)),
            "price_rows": int(len(price_df)),
            "feature_ticker_count": int(feature_df["ticker"].nunique()) if "ticker" in feature_df.columns else None,
            "price_ticker_count": int(price_df["ticker"].nunique()) if "ticker" in price_df.columns else None,
        },
    )

    _log_progress(candidate_dir, "dataset_build_start", {"include_future_covariate": config.model == "tide" and config.use_future_covariate})
    dataset = _build_cp146_lazy_sequence_dataset_with_progress(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        ticker_registry=build_registry(plan.eligible_tickers, plan.timeframe),
        include_future_covariate=config.model == "tide" and config.use_future_covariate,
        line_target_type=config.line_target_type,
        band_target_type=config.band_target_type,
        candidate_dir=candidate_dir,
    )
    _log_progress(
        candidate_dir,
        "dataset_build_done",
        {
            "sample_count": len(dataset),
            "metadata_rows": int(len(dataset.metadata)),
            "ticker_count": len(dataset.ticker_arrays),
        },
    )

    _log_progress(candidate_dir, "split_start", {})
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    _log_progress(
        candidate_dir,
        "split_done",
        {"train_samples": len(train_bundle), "val_samples": len(val_bundle), "test_samples": len(test_bundle)},
    )

    _log_progress(candidate_dir, "normalize_start", {})
    train_bundle, val_bundle, test_bundle, mean, std = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    _log_progress(candidate_dir, "normalize_done", {"feature_dim": int(mean.shape[0])})
    return train_bundle, val_bundle, test_bundle, mean, std, plan


def _make_config(candidate: dict[str, Any], *, wandb_project: str) -> TrainConfig:
    return TrainConfig(
        model=candidate["model"],
        timeframe=candidate["timeframe"],
        horizon=candidate["horizon"],
        seq_len=candidate["seq_len"],
        epochs=candidate["epochs"],
        batch_size=256,
        lr=1e-3,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-4,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.1,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=42,
        device="cuda",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=True,
        wandb_project=wandb_project,
        model_ver="v2-multihead",
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        checkpoint_selection="line_gate",
        amp_dtype="bf16",
        detect_anomaly=False,
        explicit_cuda_cleanup=True,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=candidate["patch_len"],
        patch_stride=candidate["patch_stride"],
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=candidate["feature_set"],
        feature_columns=None,
        n_features=36,
        market_data_provider="eodhd",
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
    )


def run_candidate(candidate_name: str, *, wandb_project: str, wandb_mode: str) -> dict[str, Any]:
    ensure_snapshot_alias()
    _set_local_env(wandb_mode=wandb_mode, wandb_project=wandb_project)
    candidate = _candidate_by_name(candidate_name)
    candidate_dir = LOG_DIR / candidate["name"]
    local_log_dir = candidate_dir / "train_local_logs"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(candidate, wandb_project=wandb_project)
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    error = None
    result: dict[str, Any] | None = None
    try:
        _log_progress(
            candidate_dir,
            "runtime_check",
            {
                "candidate": candidate["name"],
                "sys_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "wandb_mode": os.environ.get("WANDB_MODE"),
                "wandb_project": os.environ.get("WANDB_PROJECT"),
                "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
                "device": config.device,
            },
        )
        _log_progress(
            candidate_dir,
            "data_prepare_start",
            {
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "seq_len": config.seq_len,
                "feature_set": config.feature_set,
                "provider": config.market_data_provider,
            },
        )
        precomputed_bundles = _prepare_dataset_splits_with_progress(config, candidate_dir)
        train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed_bundles
        plan_summary = summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle)
        _log_progress(
            candidate_dir,
            "data_prepare_done",
            {
                "source_data_hash": plan_summary.get("source_data_hash"),
                "eligible_ticker_count": plan_summary.get("eligible_ticker_count"),
                "train_samples": plan_summary.get("train_samples"),
                "val_samples": plan_summary.get("val_samples"),
                "test_samples": plan_summary.get("test_samples"),
            },
        )
        _log_progress(candidate_dir, "training_start", {"run_name": _run_name(candidate)})
        result = run_training(
            config,
            save_run=False,
            precomputed_bundles=precomputed_bundles,
            local_log=True,
            local_log_dir=local_log_dir,
            wandb_name=_run_name(candidate),
            wandb_group="cp146_lm_eodhd500_line",
            wandb_config_override={**asdict(config), "cp": "CP146-LM", "candidate": candidate["name"], "context": "eodhd_500"},
        )
        _log_progress(candidate_dir, "training_done", {"run_id": result.get("run_id") if isinstance(result, dict) else None})
        exit_code = 0
    except Exception as exc:
        exit_code = 1
        error = str(exc)
        _log_progress(candidate_dir, "failed", {"error": error})
    elapsed = time.perf_counter() - started
    peak_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    process = {
        "candidate": candidate["name"],
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "epoch_seconds_estimate": round(elapsed / max(int(candidate["epochs"]), 1), 3),
        "peak_vram_bytes": peak_after,
        "peak_vram_bytes_before": peak_before,
        "local_log_dir": str(local_log_dir),
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
        "run_name": _run_name(candidate),
        "save_run": False,
        "error": error,
        "result_run_id": (result or {}).get("run_id") if isinstance(result, dict) else None,
    }
    _write_json(candidate_dir / "train_process.json", process)
    if exit_code != 0:
        raise RuntimeError(error or f"{candidate_name} failed")
    return process


def _find_run_dir(candidate: dict[str, Any]) -> Path | None:
    base = LOG_DIR / candidate["name"] / "train_local_logs"
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _line_source(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    nested = metrics.get("line_metrics")
    return nested if isinstance(nested, dict) else metrics


def _line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _line_source(metrics)
    return {key: source.get(key) for key in LINE_KEYS if source.get(key) is not None}


def _bucket_metrics(metrics: dict[str, Any] | None, horizon: int) -> dict[str, Any]:
    source = _line_source(metrics)
    buckets: dict[str, Any] = {}
    for prefix in ("h1_h5", "h6_h10", "h11_h20"):
        bucket = {key: source.get(f"{prefix}_{key}") for key in LINE_KEYS if source.get(f"{prefix}_{key}") is not None}
        if bucket:
            label = "h1_h4" if horizon <= 4 and prefix == "h1_h5" else prefix
            buckets[label] = bucket
    return buckets


def _product_ok(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and (_safe_float(metrics.get("ic_mean")) or -999.0) > 0
        and (_safe_float(metrics.get("long_short_spread")) or -999.0) > 0
        and (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) > 0
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.20
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.75
    )


def _classify(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [candidate for candidate in candidates if _product_ok(candidate)]
    default_by_timeframe: dict[str, str] = {}
    for timeframe in ("1D", "1W"):
        group = [candidate for candidate in eligible if (candidate.get("config") or {}).get("timeframe") == timeframe]
        if group:
            best = sorted(
                group,
                key=lambda item: (
                    _safe_float((item.get("validation_line_metrics") or {}).get("ic_mean")) or -999.0,
                    _safe_float((item.get("validation_line_metrics") or {}).get("long_short_spread")) or -999.0,
                    _safe_float((item.get("validation_line_metrics") or {}).get("fee_adjusted_return")) or -999.0,
                    -(_safe_float((item.get("validation_line_metrics") or {}).get("false_safe_tail_rate")) or 999.0),
                ),
                reverse=True,
            )[0]
            default_by_timeframe[timeframe] = best["candidate"]
    for candidate in candidates:
        timeframe = (candidate.get("config") or {}).get("timeframe")
        if candidate.get("candidate") == default_by_timeframe.get(timeframe):
            candidate["classification"] = "recommended_default"
        elif _product_ok(candidate):
            candidate["classification"] = "selectable_verified"
        elif candidate.get("status") == "missing":
            candidate["classification"] = "experiment_record"
        else:
            candidate["classification"] = "rejected"
    return candidates


def collect_candidates() -> list[dict[str, Any]]:
    records = []
    for spec in CANDIDATES:
        run_dir = _find_run_dir(spec)
        process_path = LOG_DIR / spec["name"] / "train_process.json"
        process = _read_json(process_path) if process_path.exists() else {}
        if run_dir is None:
            records.append({"candidate": spec["name"], "status": "missing", "spec": spec, "process": process})
            continue
        summary = _read_json(run_dir / "summary.json")
        config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        best = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
        test = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
        validation = _line_metrics(best)
        test_metrics = _line_metrics(test)
        status = "PASS" if process.get("exit_code") == 0 and validation else "FAIL"
        records.append(
            {
                "candidate": spec["name"],
                "status": status,
                "spec": spec,
                "process": process,
                "run_id": summary.get("run_id"),
                "run_dir": str(run_dir),
                "checkpoint_path": summary.get("checkpoint_path"),
                "checkpoint_exists": bool(summary.get("checkpoint_path") and Path(str(summary.get("checkpoint_path"))).exists()),
                "config": (config.get("config") if isinstance(config.get("config"), dict) else {}),
                "gate_status": {
                    "checkpoint_selection": best.get("checkpoint_selection"),
                    "gate_type": best.get("gate_type"),
                    "line_gate_pass": best.get("line_gate_pass"),
                    "gate_failed": best.get("gate_failed"),
                    "role": best.get("role"),
                    "best_epoch": best.get("best_epoch"),
                },
                "validation_line_metrics": validation,
                "test_line_metrics": test_metrics,
                "validation_bucket_metrics": _bucket_metrics(best, int(spec["horizon"])),
                "test_bucket_metrics": _bucket_metrics(test, int(spec["horizon"])),
                "wandb_status": summary.get("wandb_status") or {},
            }
        )
    return _classify(records)


def _registry(candidates: list[dict[str, Any]], preflight: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        spec = candidate.get("spec") or {}
        metrics = candidate.get("validation_line_metrics") or {}
        rows.append(
            {
                "candidate_id": candidate.get("candidate"),
                "display_name": candidate.get("candidate"),
                "version_label": f"cp146_{candidate.get('candidate')}",
                "classification": candidate.get("classification"),
                "role": "line_model",
                "timeframe": spec.get("timeframe"),
                "horizon": spec.get("horizon"),
                "model_family": spec.get("model"),
                "feature_set": spec.get("feature_set"),
                "line_loss": {"class": "AsymmetricHuberLoss", "alpha": 1.0, "beta": 2.0, "delta": 1.0},
                "validation_stability": {"line_gate_pass": (candidate.get("gate_status") or {}).get("line_gate_pass"), "basis": "validation"},
                "test_exposure_count": ((preflight.get("checks") or {}).get(str(spec.get("timeframe"))) or {}).get("test_exposure_count"),
                "strength_summary": (
                    f"IC {_fmt(metrics.get('ic_mean'))}, spread {_fmt(metrics.get('long_short_spread'))}, "
                    f"false_safe {_fmt(metrics.get('false_safe_tail_rate'))}, severe {_fmt(metrics.get('severe_downside_recall'))}"
                ),
                "weakness_summary": "실행 전" if candidate.get("status") == "missing" else "",
                "why_not_default": "recommended_default" if candidate.get("classification") == "recommended_default" else "",
                "key_metrics": {"validation": candidate.get("validation_line_metrics"), "test": candidate.get("test_line_metrics")},
                "source_data_hash": ((preflight.get("checks") or {}).get(str(spec.get("timeframe"))) or {}).get("source_data_hash"),
                "context_checksum": preflight.get("context_checksum"),
                "wandb_url": (candidate.get("wandb_status") or {}).get("run_url"),
            }
        )
    return rows


def _write_csv(candidates: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate", "status", "classification", "run_id", "wandb_url", *[f"val_{key}" for key in LINE_KEYS]]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            metrics = candidate.get("validation_line_metrics") or {}
            row = {
                "candidate": candidate.get("candidate"),
                "status": candidate.get("status"),
                "classification": candidate.get("classification"),
                "run_id": candidate.get("run_id"),
                "wandb_url": (candidate.get("wandb_status") or {}).get("run_url"),
            }
            row.update({f"val_{key}": metrics.get(key) for key in LINE_KEYS})
            writer.writerow(row)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, separator, *body])


def build_report() -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else {}
    candidates = collect_candidates()
    completed = [candidate for candidate in candidates if candidate.get("status") == "PASS"]
    registry = _registry(candidates, preflight)
    status = "PASS" if len(completed) >= 5 and any(item.get("classification") == "recommended_default" for item in completed) else "WAITING_FOR_TRAINING"
    payload = {
        "cp": "CP146-LM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "preflight": preflight,
        "candidates": candidates,
        "wandb_run_links": [
            {"candidate": item.get("candidate"), "url": (item.get("wandb_status") or {}).get("run_url")}
            for item in candidates
            if (item.get("wandb_status") or {}).get("run_url")
        ],
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, {"cp": "CP146-LM", "generated_at": payload["generated_at"], "candidates": registry})
    _write_csv(candidates)
    _write_report(payload)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    candidates = payload.get("candidates") or []
    rows = []
    for candidate in candidates:
        metrics = candidate.get("validation_line_metrics") or {}
        rows.append(
            {
                "candidate": candidate.get("candidate"),
                "status": candidate.get("status"),
                "class": candidate.get("classification"),
                "gate": (candidate.get("gate_status") or {}).get("line_gate_pass"),
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fee": _fmt(metrics.get("fee_adjusted_return")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "wandb": (candidate.get("wandb_status") or {}).get("run_url") or "",
            }
        )
    preflight = payload.get("preflight") or {}
    report = [
        "# CP146-LM EODHD 500 Line baseline full training",
        "",
        "## 상태",
        "",
        f"- status: `{payload.get('status')}`",
        f"- provider/context: `eodhd` / `eodhd_500`",
        f"- context_checksum: `{preflight.get('context_checksum')}`",
        f"- W&B run link 수: `{len(payload.get('wandb_run_links') or [])}`",
        "",
        "## 후보 결과",
        "",
        _table(
            rows,
            [
                ("candidate", "candidate"),
                ("status", "status"),
                ("class", "class"),
                ("line_gate", "gate"),
                ("ic_mean", "ic"),
                ("spread", "spread"),
                ("fee", "fee"),
                ("false_safe", "false_safe"),
                ("severe", "severe"),
                ("wandb", "wandb"),
            ],
        ),
        "",
        "## 메모",
        "",
        "- save-run, DB write, inference 저장, composite, 프론트 수정은 사용하지 않는다.",
        "- W&B online 실행은 사용자 터미널에서 수행한다.",
        "- local log와 summary.json을 기준으로 report/registry/csv를 재생성할 수 있다.",
        "- 터미널 실행 중 출력이 멈추거나 GPU compute가 보이지 않으면 정상 학습으로 가정하지 말고, 후보별 progress.log와 python 프로세스 CPU delta를 확인한다. 20~30초 동안 같은 stage에서 CPU delta가 0이면 잔여 실행을 컷하고 병목 stage를 수정한 뒤 재실행한다.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8-sig")


def write_runner_script() -> Path:
    ensure_snapshot_alias()
    path = LOG_DIR / "run_cp146_lm_eodhd500_line_training.ps1"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "cd C:\\Users\\user\\lens",
        "$env:PYTHONUTF8='1'",
        "$env:PYTHONPATH='C:\\Users\\user\\lens'",
        "$env:KMP_DUPLICATE_LIB_OK='TRUE'",
        "$env:TORCHDYNAMO_DISABLE='1'",
        "$env:MARKET_DATA_PROVIDER='eodhd'",
        "$env:LENS_USE_LOCAL_SNAPSHOTS='1'",
        "$env:LENS_REQUIRE_LOCAL_SNAPSHOTS='1'",
        "$env:LENS_DATA_BACKEND='local'",
        f"$env:LENS_LOCAL_SNAPSHOT_DIR='{ALIAS_DIR}'",
        "$env:WANDB_MODE='online'",
        "$env:WANDB_PROJECT='lens-eodhd500'",
        "",
        "function Invoke-CP146($Arguments) {",
        "  & .\\.venv\\Scripts\\python.exe @Arguments",
        "  if ($LASTEXITCODE -ne 0) { throw \"CP146 command failed: $($Arguments -join ' ')\" }",
        "}",
        "",
        "Invoke-CP146 @('-m','ai.cp146_lm_eodhd500_line_full_training','preflight')",
        "",
    ]
    for candidate in CANDIDATES[:5]:
        lines.extend(
            [
                f"Invoke-CP146 @('-m','ai.cp146_lm_eodhd500_line_full_training','run','--candidate','{candidate['name']}','--wandb-project','lens-eodhd500','--wandb-mode','online')",
                "",
            ]
        )
    lines.extend(
        [
            "# 선택 후보 h6까지 실행하려면 아래 주석을 해제하세요.",
            f"# Invoke-CP146 @('-m','ai.cp146_lm_eodhd500_line_full_training','run','--candidate','{CANDIDATES[5]['name']}','--wandb-project','lens-eodhd500','--wandb-mode','online')",
            "",
            "Invoke-CP146 @('-m','ai.cp146_lm_eodhd500_line_full_training','report')",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight()
    runner = write_runner_script()
    print(json.dumps({"status": payload["status"], "preflight_path": str(PREFLIGHT_PATH), "runner": str(runner)}, ensure_ascii=False, indent=2))


def run_train_candidate(args: argparse.Namespace) -> None:
    payload = run_candidate(args.candidate, wandb_project=args.wandb_project, wandb_mode=args.wandb_mode)
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))


def run_report(_: argparse.Namespace) -> None:
    payload = build_report()
    print(json.dumps({"status": payload.get("status"), "metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH), "csv_path": str(CSV_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP146-LM EODHD 500 line full training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    run = subparsers.add_parser("run")
    run.add_argument("--candidate", required=True)
    run.add_argument("--wandb-project", default=WANDB_PROJECT_DEFAULT)
    run.add_argument("--wandb-mode", default="online")
    run.set_defaults(func=run_train_candidate)
    report = subparsers.add_parser("report")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
