from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from statistics import NormalDist
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_PYTHON_PATH = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# 이번 CP는 yfinance 500 로컬 parquet만 사용한다.
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    SequenceDataset,
    append_calendar_features,
    apply_calendar_split_metadata,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    default_horizon,
    split_sequence_dataset_calendar_aligned,
)
from ai.ticker_registry import build_registry, load_registry  # noqa: E402
from ai.train import (  # noqa: E402
    MODEL_REGISTRY,
    autocast_context,
    forward_model,
    make_loader,
    resolve_device,
    resolve_feature_columns,
)


CP_NAME = "CP178-BM"
TIMEFRAME = "1W"
HORIZON = default_horizon(TIMEFRAME)
SEQ_LEN = 104
TARGET_TYPE = "raw_future_return"
PROVIDER = "yfinance"
FEATURE_SET = "price_volatility_volume"
SOURCE_DATA_HASH_FALLBACK = "90666b44cbfb8e5c"
CURRENT_DATE = pd.Timestamp("2026-05-15")
SEED = 42
STAGE2_EPOCHS = 2

PRICE_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.parquet"
PRICE_MANIFEST_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.manifest.json"
INDICATOR_PATH = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W_500.parquet"
INDICATOR_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W_500.manifest.json"
)

REPORT_STAGE0_PATH = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage0_preflight_report.md"
REPORT_STAGE1_PATH = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage1_baseline_report.md"
REPORT_STAGE2_PATH = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage2_model_zoo_smoke_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage0_2_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage0_2_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage0_2_logs"
OVERLAY_DIR = LOG_DIR / "snapshot_overlay"
TRAIN_LOG_BASE_DIR = LOG_DIR / "ai_train_local_logs"

Q_PAIRS = [
    ("q15_q85", 0.15, 0.85),
    ("q10_q90", 0.10, 0.90),
    ("q05_q95", 0.05, 0.95),
]

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "lower_breach_abs_error",
    "upper_breach_abs_error",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "asymmetric_interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
]

LOWER_IS_BETTER = {
    "coverage_abs_error",
    "lower_breach_abs_error",
    "upper_breach_abs_error",
    "asymmetric_interval_score",
    "median_band_width",
    "p90_band_width",
    "squeeze_breakout_rate",
}
HIGHER_IS_BETTER = {
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
}


@dataclass(frozen=True)
class SplitPayload:
    plan: Any
    dataset: SequenceDataset
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    registry: dict[str, Any]


@dataclass(frozen=True)
class Stage2Candidate:
    candidate_id: str
    model: str
    family: str
    seq_len: int
    q_label: str
    q_low: float
    q_high: float
    band_mode: str
    note: str
    batch_size: int = 256
    epochs: int = STAGE2_EPOCHS
    fp32_modules: str | None = None
    patch_len: int = 16
    patch_stride: int = 8
    lower_band_loss_weight: float = 1.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return None
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    corr = np.corrcoef(xr, yr)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else None


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    price_manifest = _read_json(PRICE_MANIFEST_PATH)
    indicator_manifest = _read_json(INDICATOR_MANIFEST_PATH)
    price = pd.read_parquet(PRICE_PATH)
    indicators = pd.read_parquet(INDICATOR_PATH)
    for frame in (price, indicators):
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"].astype(str).str.upper() == TIMEFRAME].copy()
    return price, indicators, price_manifest, indicator_manifest


def build_split_payload(
    *, price: pd.DataFrame, indicators: pd.DataFrame, source_data_hash: str
) -> SplitPayload:
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        market_data_provider=PROVIDER,
        source_data_hash=source_data_hash,
        split_mode="calendar_aligned",
    )
    registry = build_registry(plan.eligible_tickers, TIMEFRAME)
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price[price["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test, calendar_plan = split_sequence_dataset_calendar_aligned(
        dataset,
        purge_gap_trading_days=plan.h_max,
        min_fold_samples=plan.min_fold_samples,
    )
    apply_calendar_split_metadata(plan, calendar_plan)
    return SplitPayload(
        plan=plan, dataset=dataset, train=train, val=val, test=test, registry=registry
    )


def collect_targets(bundle: SequenceDataset) -> np.ndarray:
    targets = np.empty((len(bundle.sample_refs), bundle.horizon), dtype=np.float32)
    refs_by_ticker: dict[str, list[tuple[int, int]]] = {}
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        refs_by_ticker.setdefault(str(ticker), []).append((row_idx, int(end_idx)))
    for ticker, refs in refs_by_ticker.items():
        closes = np.asarray(bundle.ticker_arrays[ticker]["closes"], dtype=np.float64)
        for row_idx, end_idx in refs:
            anchor = float(closes[end_idx])
            future = closes[end_idx + 1 : end_idx + 1 + bundle.horizon]
            targets[row_idx] = ((future / anchor) - 1.0).astype(np.float32)
    return targets


def _split_dates(bundle: SequenceDataset) -> pd.Series:
    return pd.to_datetime(bundle.metadata["asof_date"], errors="coerce")


def _date_range_count(bundle: SequenceDataset, start: str, end: str) -> int:
    dates = _split_dates(bundle)
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    return int(mask.sum())


def _target_event_counts(targets: np.ndarray, low: float, high: float) -> dict[str, int]:
    lower = targets < low
    upper = targets > high
    return {
        "lower_event_points": int(lower.sum()),
        "upper_event_points": int(upper.sum()),
        "total_event_points": int((lower | upper).sum()),
        "total_target_points": int(targets.size),
    }


def _event_counts_for_range(
    bundle: SequenceDataset,
    targets: np.ndarray,
    *,
    low: float,
    high: float,
    start: str,
    end: str,
) -> dict[str, Any]:
    dates = _split_dates(bundle)
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    selected = targets[mask.to_numpy()]
    if selected.size == 0:
        return {
            "start": start,
            "end": end,
            "sample_rows": 0,
            "total_event_points": 0,
            "total_target_points": 0,
        }
    return {
        "start": start,
        "end": end,
        "sample_rows": int(mask.sum()),
        **_target_event_counts(selected, low, high),
        "realized_abs_mean": float(np.nanmean(np.abs(selected))),
        "downside_point_rate": float((selected < 0.0).mean()),
    }


def latest_week_status(indicators: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(indicators["date"], errors="coerce").dropna()
    latest = dates.max()
    weekday_counts = {
        str(int(key)): int(value)
        for key, value in dates.dt.weekday.value_counts().sort_index().items()
    }
    days_since_friday = (CURRENT_DATE.weekday() - 4) % 7
    if days_since_friday == 0:
        days_since_friday = 7
    latest_complete_friday = CURRENT_DATE - pd.Timedelta(days=days_since_friday)
    return {
        "date_min": str(dates.min().date()) if not dates.empty else None,
        "date_max": str(latest.date()) if not dates.empty else None,
        "latest_weekday": int(latest.weekday()) if pd.notna(latest) else None,
        "weekday_counts": weekday_counts,
        "expected_anchor": "W-FRI",
        "latest_complete_friday_cutoff": str(latest_complete_friday.date()),
        "latest_week_complete": bool(
            pd.notna(latest) and latest.normalize() <= latest_complete_friday.normalize()
        ),
        "incomplete_current_week_excluded": bool(
            pd.notna(latest) and latest.normalize() < CURRENT_DATE.normalize()
        ),
    }


def duplicate_summary(price: pd.DataFrame, indicators: pd.DataFrame) -> dict[str, Any]:
    price_subset = ["ticker", "date", "source"] if "source" in price.columns else ["ticker", "date"]
    indicator_subset = (
        ["ticker", "timeframe", "date", "source"]
        if "source" in indicators.columns
        else ["ticker", "timeframe", "date"]
    )
    return {
        "price_duplicate_ticker_date_source_rows": int(price.duplicated(price_subset).sum()),
        "indicator_duplicate_ticker_timeframe_date_source_rows": int(
            indicators.duplicated(indicator_subset).sum()
        ),
    }


def feature_quality_summary(
    indicators: pd.DataFrame, targets_by_split: dict[str, np.ndarray]
) -> dict[str, Any]:
    enriched = indicators.copy()
    if "has_fundamentals" not in enriched.columns:
        enriched["has_fundamentals"] = False
    for column in ["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]:
        if column not in enriched.columns:
            enriched[column] = np.nan
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["has_fundamentals"] = enriched["has_fundamentals"].fillna(False).astype(bool)
    enriched = append_calendar_features(enriched)
    missing_columns = [column for column in MODEL_FEATURE_COLUMNS if column not in enriched.columns]
    nonfinite_counts: dict[str, int] = {}
    total_nonfinite = 0
    if not missing_columns:
        for column in MODEL_FEATURE_COLUMNS:
            values = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=np.float64)
            count = int((~np.isfinite(values)).sum())
            if count:
                nonfinite_counts[column] = count
                total_nonfinite += count
    target_nonfinite_by_split = {
        split: int((~np.isfinite(values)).sum()) for split, values in targets_by_split.items()
    }
    return {
        "feature_version": FEATURE_CONTRACT_VERSION,
        "model_feature_column_count": len(MODEL_FEATURE_COLUMNS),
        "feature_set": FEATURE_SET,
        "feature_set_columns": resolve_feature_columns(FEATURE_SET),
        "feature_set_column_count": len(resolve_feature_columns(FEATURE_SET)),
        "missing_model_feature_columns": missing_columns,
        "feature_nonfinite_count_after_contract_impute": total_nonfinite,
        "feature_nonfinite_counts_after_contract_impute": nonfinite_counts,
        "target_nonfinite_by_split": target_nonfinite_by_split,
        "target_nonfinite_total": int(sum(target_nonfinite_by_split.values())),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "intraday_range_ratio_in_model_features": "intraday_range_ratio" in MODEL_FEATURE_COLUMNS,
    }


def split_overlap_summary(
    train: SequenceDataset, val: SequenceDataset, test: SequenceDataset, plan: Any
) -> dict[str, Any]:
    def keys(bundle: SequenceDataset) -> set[tuple[str, int]]:
        return {(str(ticker), int(end_idx)) for ticker, end_idx in bundle.sample_refs}

    train_keys = keys(train)
    val_keys = keys(val)
    test_keys = keys(test)
    return {
        "split_mode": plan.split_mode,
        "train_val_sample_overlap": len(train_keys & val_keys),
        "train_test_sample_overlap": len(train_keys & test_keys),
        "val_test_sample_overlap": len(val_keys & test_keys),
        "cross_split_date_overlap_count": plan.cross_split_date_overlap_count,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_min": str(pd.to_datetime(train.metadata["asof_date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train.metadata["asof_date"]).max().date()),
        "val_date_min": str(pd.to_datetime(val.metadata["asof_date"]).min().date()),
        "val_date_max": str(pd.to_datetime(val.metadata["asof_date"]).max().date()),
        "test_date_min": str(pd.to_datetime(test.metadata["asof_date"]).min().date()),
        "test_date_max": str(pd.to_datetime(test.metadata["asof_date"]).max().date()),
        "purge_gap_trading_days": plan.purge_gap_trading_days,
    }


def sample_sufficiency(
    *,
    payload: SplitPayload,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "thresholds": {},
        "fold_windows": {},
        "special_months": {},
        "pass": True,
        "fail_reasons": [],
    }
    fold_windows = {
        "fold_1_test": ("2024-11-01", "2025-05-01"),
        "fold_2_test": ("2025-05-01", "2025-11-01"),
        "fold_3_test": ("2025-11-01", "2026-05-09"),
    }
    for q_label, q_low, q_high in Q_PAIRS:
        low = float(np.nanquantile(train_targets.reshape(-1), q_low))
        high = float(np.nanquantile(train_targets.reshape(-1), q_high))
        split_counts = {
            "train": _target_event_counts(train_targets, low, high),
            "validation": _target_event_counts(val_targets, low, high),
            "test_diagnostic_only": _target_event_counts(test_targets, low, high),
        }
        fold_counts = {
            fold_name: _event_counts_for_range(
                payload.dataset,
                collect_targets(payload.dataset),
                low=low,
                high=high,
                start=start,
                end=end,
            )
            for fold_name, (start, end) in fold_windows.items()
        }
        checks = {
            "train_event_points_gte_500": split_counts["train"]["total_event_points"] >= 500,
            "validation_event_points_gte_200": split_counts["validation"]["total_event_points"]
            >= 200,
            "fold_test_event_points_gte_200_each": all(
                item["total_event_points"] >= 200 for item in fold_counts.values()
            ),
        }
        q_pass = all(checks.values())
        result["thresholds"][q_label] = {
            "q_low": q_low,
            "q_high": q_high,
            "train_quantile_low": low,
            "train_quantile_high": high,
            "split_event_counts": split_counts,
            "stage5_expected_fold_test_counts": fold_counts,
            "checks": checks,
            "pass": q_pass,
            "event_count_unit": "horizon_point",
        }
        if not q_pass:
            result["pass"] = False
            result["fail_reasons"].append(f"{q_label}_sample_thin")

    month_ranges = {
        "2024-12": ("2024-12-01", "2025-01-01"),
        "2025-04": ("2025-04-01", "2025-05-01"),
        "2026-02": ("2026-02-01", "2026-03-01"),
    }
    dataset_targets = collect_targets(payload.dataset)
    q15_info = result["thresholds"]["q15_q85"]
    low = float(q15_info["train_quantile_low"])
    high = float(q15_info["train_quantile_high"])
    for label, (start, end) in month_ranges.items():
        split_counts = {
            "train_rows": _date_range_count(payload.train, start, end),
            "validation_rows": _date_range_count(payload.val, start, end),
            "test_rows": _date_range_count(payload.test, start, end),
        }
        fold_membership = [
            name
            for name, (fold_start, fold_end) in fold_windows.items()
            if pd.Timestamp(start) < pd.Timestamp(fold_end)
            and pd.Timestamp(end) > pd.Timestamp(fold_start)
        ]
        result["special_months"][label] = {
            "range": {"start": start, "end": end},
            "calendar_split_rows": split_counts,
            "stage5_expected_fold_membership": fold_membership,
            "q15_event_profile": _event_counts_for_range(
                payload.dataset, dataset_targets, low=low, high=high, start=start, end=end
            ),
        }
    result["fold_windows"] = {
        name: {"start": start, "end": end} for name, (start, end) in fold_windows.items()
    }
    return result


def regime_in_train_summary(train: SequenceDataset) -> dict[str, Any]:
    ranges = {
        "2018_Q4": ("2018-10-01", "2019-01-01"),
        "2020_corona": ("2020-02-01", "2020-05-01"),
        "2022_bear": ("2022-01-01", "2023-01-01"),
    }
    return {
        label: {
            "start": start,
            "end": end,
            "train_sample_rows": _date_range_count(train, start, end),
            "included_in_train": _date_range_count(train, start, end) > 0,
        }
        for label, (start, end) in ranges.items()
    }


def build_stage0_metrics(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    price_manifest: dict[str, Any],
    indicator_manifest: dict[str, Any],
    payload: SplitPayload,
    source_data_hash: str,
) -> dict[str, Any]:
    train_targets = collect_targets(payload.train)
    val_targets = collect_targets(payload.val)
    test_targets = collect_targets(payload.test)
    targets_by_split = {"train": train_targets, "validation": val_targets, "test": test_targets}
    sufficiency = sample_sufficiency(
        payload=payload,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
    feature_quality = feature_quality_summary(indicators, targets_by_split)
    duplicates = duplicate_summary(price, indicators)
    overlap = split_overlap_summary(payload.train, payload.val, payload.test, payload.plan)
    contract_fail_reasons: list[str] = []
    if not sufficiency["pass"]:
        contract_fail_reasons.extend(sufficiency["fail_reasons"])
    if feature_quality["feature_nonfinite_count_after_contract_impute"] != 0:
        contract_fail_reasons.append("feature_nonfinite")
    if feature_quality["target_nonfinite_total"] != 0:
        contract_fail_reasons.append("target_nonfinite")
    if duplicates["price_duplicate_ticker_date_source_rows"] != 0:
        contract_fail_reasons.append("price_duplicate_rows")
    if duplicates["indicator_duplicate_ticker_timeframe_date_source_rows"] != 0:
        contract_fail_reasons.append("indicator_duplicate_rows")
    if overlap["cross_split_date_overlap_count"] != 0:
        contract_fail_reasons.append("split_date_overlap")
    return {
        "status": "PASS"
        if not contract_fail_reasons
        else ("WARN_SAMPLE_THIN" if not sufficiency["pass"] else "FAIL_CONTRACT"),
        "contract_fail_reasons": contract_fail_reasons,
        "provider": PROVIDER,
        "source": PROVIDER,
        "source_data_hash": source_data_hash,
        "price_path": str(PRICE_PATH),
        "indicator_path": str(INDICATOR_PATH),
        "price_manifest": price_manifest,
        "indicator_manifest": indicator_manifest,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "timeframe": TIMEFRAME,
        "horizon_contract": {
            "requested_note": "지시서의 h5 문구는 기존 1W band pipeline 계약 확인 대상으로 해석했다.",
            "resolved_horizon": HORIZON,
            "default_horizon_1w": default_horizon("1W"),
            "final_horizon_used": HORIZON,
            "target": TARGET_TYPE,
        },
        "week_status": latest_week_status(indicators),
        "dataset_plan": {
            "input_ticker_count": int(payload.plan.input_ticker_count),
            "eligible_ticker_count": len(payload.plan.eligible_tickers),
            "excluded_ticker_count": len(payload.plan.excluded_reasons),
            "excluded_reasons": payload.plan.excluded_reasons,
            "estimated_usable_sample_count": payload.plan.estimated_usable_sample_count,
            "num_tickers": payload.plan.num_tickers,
        },
        "split": overlap,
        "feature_quality": feature_quality,
        "duplicates": duplicates,
        "sample_sufficiency": sufficiency,
        "train_regime_coverage": regime_in_train_summary(payload.train),
        "test_policy": {
            "test_opened_for_candidate_selection": False,
            "test_counts_in_stage0": "diagnostic_only",
            "selection_basis": "validation_only_from_stage1_stage2",
        },
    }


def build_rolling_cache(dataset: SequenceDataset, quantiles: list[float]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    quantile_window = 104
    quantile_min_periods = 26
    boll_window = 52
    boll_min_periods = 13
    for ticker, arrays in dataset.ticker_arrays.items():
        closes = np.asarray(arrays["closes"], dtype=np.float64)
        n_rows = len(closes)
        returns = np.full((dataset.horizon, n_rows), np.nan, dtype=np.float32)
        for horizon_idx in range(dataset.horizon):
            step = horizon_idx + 1
            valid = n_rows - step
            if valid <= 0:
                continue
            returns[horizon_idx, :valid] = (closes[step:] / closes[:-step] - 1.0).astype(np.float32)

        quantile_payload: dict[str, np.ndarray] = {}
        for quantile in quantiles:
            values = np.full_like(returns, np.nan, dtype=np.float32)
            for horizon_idx in range(dataset.horizon):
                step = horizon_idx + 1
                series = pd.Series(returns[horizon_idx].astype(np.float64))
                values[horizon_idx] = (
                    series.shift(step)
                    .rolling(window=quantile_window, min_periods=quantile_min_periods)
                    .quantile(float(quantile))
                    .to_numpy(dtype=np.float32)
                )
            quantile_payload[f"{quantile:.2f}"] = values

        mean_values = np.full_like(returns, np.nan, dtype=np.float32)
        std_values = np.full_like(returns, np.nan, dtype=np.float32)
        for horizon_idx in range(dataset.horizon):
            step = horizon_idx + 1
            series = pd.Series(returns[horizon_idx].astype(np.float64))
            shifted = series.shift(step)
            mean_values[horizon_idx] = (
                shifted.rolling(window=boll_window, min_periods=boll_min_periods)
                .mean()
                .to_numpy(dtype=np.float32)
            )
            std_values[horizon_idx] = (
                shifted.rolling(window=boll_window, min_periods=boll_min_periods)
                .std(ddof=0)
                .to_numpy(dtype=np.float32)
            )

        cache[str(ticker)] = {
            "quantiles": quantile_payload,
            "mean": mean_values,
            "std": std_values,
        }
    return cache


def _bundle_end_indices(bundle: SequenceDataset) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        grouped.setdefault(str(ticker), []).append((row_idx, int(end_idx)))
    return {
        ticker: (
            np.asarray([row_idx for row_idx, _ in refs], dtype=np.int64),
            np.asarray([end_idx for _, end_idx in refs], dtype=np.int64),
        )
        for ticker, refs in grouped.items()
    }


def rolling_quantile_predictions(
    bundle: SequenceDataset,
    cache: dict[str, Any],
    *,
    q_low: float,
    q_high: float,
    fallback_low: np.ndarray,
    fallback_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    upper = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    lower_key = f"{q_low:.2f}"
    upper_key = f"{q_high:.2f}"
    for ticker, (rows, end_indices) in _bundle_end_indices(bundle).items():
        ticker_cache = cache[ticker]["quantiles"]
        low_values = ticker_cache[lower_key][:, end_indices].T
        high_values = ticker_cache[upper_key][:, end_indices].T
        lower[rows] = np.where(np.isfinite(low_values), low_values, fallback_low.reshape(1, -1))
        upper[rows] = np.where(np.isfinite(high_values), high_values, fallback_high.reshape(1, -1))
    return lower, upper


def bollinger_predictions(
    bundle: SequenceDataset,
    cache: dict[str, Any],
    *,
    q_low: float,
    q_high: float,
    fallback_mean: np.ndarray,
    fallback_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del q_low
    k_value = float(NormalDist().inv_cdf(q_high))
    lower = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    upper = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    for ticker, (rows, end_indices) in _bundle_end_indices(bundle).items():
        ticker_cache = cache[ticker]
        mean_values = ticker_cache["mean"][:, end_indices].T
        std_values = ticker_cache["std"][:, end_indices].T
        mean_values = np.where(np.isfinite(mean_values), mean_values, fallback_mean.reshape(1, -1))
        std_values = np.where(np.isfinite(std_values), std_values, fallback_std.reshape(1, -1))
        lower[rows] = mean_values - (k_value * std_values)
        upper[rows] = mean_values + (k_value * std_values)
    return lower, upper


def _date_cross_sectional_width_ic(
    *,
    metadata: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    frame = metadata[["asof_date"]].reset_index(drop=True).copy()
    width = np.maximum(upper - lower, 0.0)
    center = (lower + upper) / 2.0
    downside_width = np.maximum(center - lower, 0.0)
    realized_abs = np.abs(actual)
    downside = np.maximum(-actual, 0.0)
    band_values: list[float] = []
    downside_values: list[float] = []
    for _, group in frame.groupby("asof_date", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        band_corr = _spearman_corr(width[indices].reshape(-1), realized_abs[indices].reshape(-1))
        downside_corr = _spearman_corr(
            downside_width[indices].reshape(-1), downside[indices].reshape(-1)
        )
        if band_corr is not None:
            band_values.append(band_corr)
        if downside_corr is not None:
            downside_values.append(downside_corr)
    return {
        "band_width_ic_date_cs_mean": float(np.mean(band_values)) if band_values else None,
        "band_width_ic_date_cs_std": float(np.std(band_values, ddof=1))
        if len(band_values) > 1
        else None,
        "band_width_ic_date_cs_count": len(band_values),
        "downside_width_ic_date_cs_mean": float(np.mean(downside_values))
        if downside_values
        else None,
        "downside_width_ic_date_cs_std": float(np.std(downside_values, ddof=1))
        if len(downside_values) > 1
        else None,
        "downside_width_ic_date_cs_count": len(downside_values),
    }


def summarize_band_prediction(
    *,
    name: str,
    split_name: str,
    q_label: str,
    q_low: float,
    q_high: float,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    metadata: pd.DataFrame,
    squeeze_breakout_threshold: float,
) -> dict[str, Any]:
    lower_t = torch.from_numpy(lower.astype(np.float32))
    upper_t = torch.from_numpy(upper.astype(np.float32))
    actual_t = torch.from_numpy(actual.astype(np.float32))
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=metadata,
        line_predictions=line_t,
        lower_predictions=torch.minimum(lower_t, upper_t),
        upper_predictions=torch.maximum(lower_t, upper_t),
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=q_low,
        q_high=q_high,
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        include_legacy_overlay_diagnostics=False,
    )
    metrics = {key: _safe_float(summary.get(key)) for key in BAND_METRIC_KEYS if key in summary}
    date_cs = _date_cross_sectional_width_ic(
        metadata=metadata,
        lower=np.minimum(lower, upper),
        upper=np.maximum(lower, upper),
        actual=actual,
    )
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    metrics.update(
        {
            "baseline": name,
            "split": split_name,
            "q_label": q_label,
            "q_low": q_low,
            "q_high": q_high,
            "nominal_coverage": q_high - q_low,
            "diagnostic_only": split_name == "test",
        }
    )
    return metrics


def compute_baselines(payload: SplitPayload) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_targets = collect_targets(payload.train)
    val_targets = collect_targets(payload.val)
    test_targets = collect_targets(payload.test)
    train_abs_p80 = float(np.nanquantile(np.abs(train_targets.reshape(-1)), 0.80))
    quantiles = sorted({q for _, q_low, q_high in Q_PAIRS for q in (q_low, q_high)})
    rolling_cache = build_rolling_cache(payload.dataset, quantiles)
    rows: list[dict[str, Any]] = []
    for q_label, q_low, q_high in Q_PAIRS:
        global_low = float(np.nanquantile(train_targets.reshape(-1), q_low))
        global_high = float(np.nanquantile(train_targets.reshape(-1), q_high))
        horizon_low = np.nanquantile(train_targets, q_low, axis=0).astype(np.float32)
        horizon_high = np.nanquantile(train_targets, q_high, axis=0).astype(np.float32)
        horizon_mean = np.nanmean(train_targets, axis=0).astype(np.float32)
        horizon_std = np.nanstd(train_targets, axis=0).astype(np.float32)
        for split_name, bundle, actual in (
            ("val", payload.val, val_targets),
            ("test", payload.test, test_targets),
        ):
            constant_lower = np.full_like(actual, global_low, dtype=np.float32)
            constant_upper = np.full_like(actual, global_high, dtype=np.float32)
            rows.append(
                {
                    **summarize_band_prediction(
                        name="constant_width_train_quantile",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=constant_lower,
                        upper=constant_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "baseline_detail": "global_train_quantile_repeated_all_horizons",
                }
            )
            rolling_lower, rolling_upper = rolling_quantile_predictions(
                bundle,
                rolling_cache,
                q_low=q_low,
                q_high=q_high,
                fallback_low=horizon_low,
                fallback_high=horizon_high,
            )
            rows.append(
                {
                    **summarize_band_prediction(
                        name="rolling_historical_quantile_band",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=rolling_lower,
                        upper=rolling_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "baseline_detail": "w104_shifted_horizon_specific_quantile_min_periods_26",
                }
            )
            boll_lower, boll_upper = bollinger_predictions(
                bundle,
                rolling_cache,
                q_low=q_low,
                q_high=q_high,
                fallback_mean=horizon_mean,
                fallback_std=horizon_std,
            )
            rows.append(
                {
                    **summarize_band_prediction(
                        name="rolling_bollinger_return_band",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=boll_lower,
                        upper=boll_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "baseline_detail": "w52_shifted_mean_std_gaussian_k_from_q_high_min_periods_13",
                }
            )
    target_checks = {
        "train_nonfinite": int((~np.isfinite(train_targets)).sum()),
        "validation_nonfinite": int((~np.isfinite(val_targets)).sum()),
        "test_nonfinite": int((~np.isfinite(test_targets)).sum()),
        "train_abs_return_p80_for_squeeze": train_abs_p80,
    }
    return rows, target_checks


def baseline_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    val_rows = [row for row in rows if row.get("split") == "val"]
    by_q: dict[str, Any] = {}
    for q_label, q_low, q_high in Q_PAIRS:
        subset = [row for row in val_rows if row.get("q_label") == q_label]
        sota: dict[str, Any] = {}
        for metric in sorted(LOWER_IS_BETTER):
            finite = [row for row in subset if _safe_float(row.get(metric)) is not None]
            if finite:
                best = min(finite, key=lambda row: float(row[metric]))
                sota[metric] = {
                    "value": float(best[metric]),
                    "baseline": best["baseline"],
                    "direction": "lower",
                }
        for metric in sorted(HIGHER_IS_BETTER):
            finite = [row for row in subset if _safe_float(row.get(metric)) is not None]
            if finite:
                best = max(finite, key=lambda row: float(row[metric]))
                sota[metric] = {
                    "value": float(best[metric]),
                    "baseline": best["baseline"],
                    "direction": "higher",
                }
        coverage_pass_rows = [
            row
            for row in subset
            if _safe_float(row.get("coverage_abs_error")) is not None
            and float(row["coverage_abs_error"]) <= 0.05
            and _safe_float(row.get("p90_band_width")) is not None
        ]
        p90_base = coverage_pass_rows or [
            row for row in subset if _safe_float(row.get("p90_band_width")) is not None
        ]
        p90_min = min(float(row["p90_band_width"]) for row in p90_base) if p90_base else None
        p90_max = max(float(row["p90_band_width"]) for row in p90_base) if p90_base else None
        by_q[q_label] = {
            "q_low": q_low,
            "q_high": q_high,
            "baseline_sota_validation": sota,
            "p90_overwide_threshold": (p90_min * 1.50) if p90_min is not None else None,
            "p90_profile_max": p90_max,
            "validation_rows": subset,
            "baseline_profile_note": "metric별 SOTA는 oracle gate가 아니라 validation profile 비교 기준으로 사용한다.",
        }
    selected_q = select_stage2_q(by_q)
    return {
        "aggregation_lock": {
            "band_width_ic": "date_cross_sectional_mean",
            "downside_width_ic": "date_cross_sectional_mean",
            "flatten_values_retained_as_diagnostics": True,
        },
        "metric_direction_lock": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
            "coverage_interpreted_by_abs_error": True,
        },
        "by_q": by_q,
        "selected_stage2_q": selected_q,
        "selection_uses_test": False,
    }


def _rank_scores(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    finite = [(key, value) for key, value in values.items() if math.isfinite(value)]
    if not finite:
        return {key: 0.0 for key in values}
    finite_sorted = sorted(finite, key=lambda item: item[1], reverse=higher_is_better)
    count = len(finite_sorted)
    scores: dict[str, float] = {}
    for rank, (key, _) in enumerate(finite_sorted):
        scores[key] = 1.0 if count == 1 else 1.0 - (rank / (count - 1))
    for key in values:
        scores.setdefault(key, 0.0)
    return scores


def select_stage2_q(by_q: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "coverage_abs_error": {},
        "asymmetric_interval_score": {},
        "lower_breach_abs_error": {},
        "band_width_ic": {},
        "downside_width_ic": {},
        "p90_band_width": {},
    }
    for q_label, profile in by_q.items():
        sota = profile["baseline_sota_validation"]
        for metric in metrics:
            metrics[metric][q_label] = float(sota.get(metric, {}).get("value", math.nan))
    scores = {q_label: 0.0 for q_label in by_q}
    weights = {
        "coverage_abs_error": 0.20,
        "asymmetric_interval_score": 0.30,
        "lower_breach_abs_error": 0.15,
        "band_width_ic": 0.15,
        "downside_width_ic": 0.10,
        "p90_band_width": 0.10,
    }
    for metric, weight in weights.items():
        rank = _rank_scores(metrics[metric], higher_is_better=metric in HIGHER_IS_BETTER)
        for q_label, score in rank.items():
            scores[q_label] += score * weight
    selected = max(scores, key=lambda key: scores[key])
    return {
        "q_label": selected,
        "q_low": by_q[selected]["q_low"],
        "q_high": by_q[selected]["q_high"],
        "score": scores[selected],
        "all_q_scores": scores,
        "selection_basis": "validation_baseline_profile_rank_no_test_no_1d_bias",
    }


def build_stage1_metrics(payload: SplitPayload) -> dict[str, Any]:
    rows, target_checks = compute_baselines(payload)
    profiles = baseline_profiles(rows)
    return {
        "status": "PASS",
        "rows": rows,
        "target_checks": target_checks,
        "profiles": profiles,
        "test_policy": {
            "test_metrics_emitted": True,
            "test_metrics_usage": "diagnostic_only",
            "candidate_selection_uses_test": False,
        },
    }


def prepare_snapshot_overlay() -> dict[str, Any]:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    links = [
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance_1W.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance_1W.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators_yfinance_1W.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators_yfinance_1W.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators_yfinance.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators_yfinance.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators.manifest.json"),
    ]
    entries: list[dict[str, Any]] = []
    for source, target in links:
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            mode = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        entries.append({"source": str(source), "target": str(target), "mode": mode})
    return {"overlay_dir": str(OVERLAY_DIR), "entries": entries}


def build_stage2_candidates(selected_q: dict[str, Any]) -> list[Stage2Candidate]:
    q_label = str(selected_q["q_label"])
    q_low = float(selected_q["q_low"])
    q_high = float(selected_q["q_high"])
    q_tag = q_label.replace("_", "")
    return [
        Stage2Candidate(
            candidate_id=f"tide_s104_{q_tag}_param",
            model="tide",
            family="tide",
            seq_len=104,
            q_label=q_label,
            q_low=q_low,
            q_high=q_high,
            band_mode="param",
            note="1W 전용 TiDE param smoke",
        ),
        Stage2Candidate(
            candidate_id=f"tcn_s104_{q_tag}_param",
            model="tcn_quantile",
            family="tcn_quantile",
            seq_len=104,
            q_label=q_label,
            q_low=q_low,
            q_high=q_high,
            band_mode="param",
            note="1W 전용 TCNQuantile param smoke",
        ),
        Stage2Candidate(
            candidate_id=f"cnn_s104_{q_tag}_direct",
            model="cnn_lstm",
            family="cnn_lstm",
            seq_len=104,
            q_label=q_label,
            q_low=q_low,
            q_high=q_high,
            band_mode="direct",
            note="1W 전용 CNN-LSTM direct smoke",
            fp32_modules="lstm,heads",
        ),
        Stage2Candidate(
            candidate_id=f"patch_s104_{q_tag}_direct",
            model="patchtst",
            family="patchtst",
            seq_len=104,
            q_label=q_label,
            q_low=q_low,
            q_high=q_high,
            band_mode="direct",
            note="1W 전용 PatchTST reference smoke",
            batch_size=128,
        ),
    ]


def command_for_candidate(candidate: Stage2Candidate, *, device: str) -> list[str]:
    cmd = [
        str(VENV_PYTHON_PATH),
        "-m",
        "ai.train",
        "--model",
        candidate.model,
        "--model-role",
        "band",
        "--timeframe",
        TIMEFRAME,
        "--horizon",
        str(HORIZON),
        "--seq-len",
        str(candidate.seq_len),
        "--feature-set",
        FEATURE_SET,
        "--line-target-type",
        TARGET_TYPE,
        "--band-target-type",
        TARGET_TYPE,
        "--q-low",
        str(candidate.q_low),
        "--q-high",
        str(candidate.q_high),
        "--lambda-band",
        "2.0",
        "--band-mode",
        candidate.band_mode,
        "--checkpoint-selection",
        "band_gate",
        "--epochs",
        str(candidate.epochs),
        "--batch-size",
        str(candidate.batch_size),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--amp-dtype",
        "bf16",
        "--no-compile",
        "--no-wandb",
        "--num-workers",
        "0",
        "--split-mode",
        "calendar_aligned",
        "--local-log",
        "--local-log-dir",
        str(TRAIN_LOG_BASE_DIR),
        "--explicit-cuda-cleanup",
    ]
    if candidate.fp32_modules:
        cmd.extend(["--fp32-modules", candidate.fp32_modules])
    if candidate.lower_band_loss_weight != 1.0:
        cmd.extend(["--lower-band-loss-weight", str(candidate.lower_band_loss_weight)])
    if candidate.model == "patchtst":
        cmd.extend(
            ["--patch-len", str(candidate.patch_len), "--patch-stride", str(candidate.patch_stride)]
        )
    return cmd


def _should_echo_training_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if "epoch_seconds" in text or "checkpoint_selection" in text or "[EXIT-MARKER" in text:
        return True
    if text.startswith("GPU:") or text.startswith("amp_dtype"):
        return True
    if text.startswith("val_total="):
        return True
    return False


def _extract_run_id(text: str) -> str | None:
    patterns = [r'"run_id"\s*:\s*"([^"]+)"', r"run_id=([A-Za-z0-9_.:-]+)"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _read_local_summary(run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    path = TRAIN_LOG_BASE_DIR / run_id / "summary.json"
    if not path.exists():
        return None
    return _read_json(path)


def _resolve_checkpoint_path(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    raw = summary.get("checkpoint_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _read_epoch_logs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def classify_runtime_failure(lines: list[str]) -> dict[str, Any]:
    tail = "\n".join(lines[-100:])
    category = "runtime"
    lowered = tail.lower()
    if "nan" in lowered or "non-finite" in lowered or "nonfinite" in lowered:
        category = "contract"
    if "unsupported" in lowered or "not supported" in lowered:
        category = "model_family"
    if "out of memory" in lowered or "cuda" in lowered:
        category = "runtime"
    return {"category": category, "tail": tail[-4000:]}


def run_candidate(candidate: Stage2Candidate, *, device: str, force: bool) -> dict[str, Any]:
    candidate_dir = LOG_DIR / "stage2" / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    process_path = candidate_dir / "train_process.json"
    stdout_path = candidate_dir / "train_stdout.log"
    if not force and process_path.exists():
        existing = _read_json(process_path)
        checkpoint_path = existing.get("checkpoint_path")
        if existing.get("status") == "PASS" and checkpoint_path and Path(checkpoint_path).exists():
            return existing
    cmd = command_for_candidate(candidate, device=device)
    env = os.environ.copy()
    env.update(
        {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(OVERLAY_DIR),
            "WANDB_MODE": "disabled",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    start_ts = _now_utc()
    start = time.perf_counter()
    output_lines: list[str] = []
    run_id: str | None = None
    print(
        json.dumps(
            {"candidate": candidate.candidate_id, "event": "start", "time": start_ts},
            ensure_ascii=False,
        )
    )
    with stdout_path.open("w", encoding="utf-8", newline="") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line.rstrip("\n"))
            log_handle.write(line)
            log_handle.flush()
            maybe_run_id = _extract_run_id(line)
            if maybe_run_id:
                run_id = maybe_run_id
            if _should_echo_training_line(line):
                print(f"[{candidate.candidate_id}] {line}", end="")
        exit_code = proc.wait()
    elapsed = time.perf_counter() - start
    run_id = run_id or _extract_run_id("\n".join(output_lines))
    local_summary = _read_local_summary(run_id)
    checkpoint_path = _resolve_checkpoint_path(local_summary)
    metrics_jsonl_path = TRAIN_LOG_BASE_DIR / run_id / "metrics.jsonl" if run_id else None
    epoch_logs = _read_epoch_logs(metrics_jsonl_path) if metrics_jsonl_path else []
    status = (
        "PASS" if exit_code == 0 and checkpoint_path and Path(checkpoint_path).exists() else "FAIL"
    )
    process = {
        "candidate": asdict(candidate),
        "status": status,
        "failure_reason": None if status == "PASS" else classify_runtime_failure(output_lines),
        "failure_category": None
        if status == "PASS"
        else classify_runtime_failure(output_lines)["category"],
        "start_time_utc": start_ts,
        "end_time_utc": _now_utc(),
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": int(exit_code),
        "command": cmd,
        "stdout_path": str(stdout_path),
        "train_process_path": str(process_path),
        "run_id": run_id,
        "local_summary": local_summary,
        "checkpoint_path": checkpoint_path,
        "metrics_jsonl_path": str(metrics_jsonl_path) if metrics_jsonl_path else None,
        "epoch_logs": epoch_logs,
        "epoch_seconds": [
            float(row["epoch_seconds"])
            for row in epoch_logs
            if _safe_float(row.get("epoch_seconds")) is not None
        ],
        "vram_peak_allocated_mb": max(
            [
                float(row["vram_peak_allocated_mb"])
                for row in epoch_logs
                if _safe_float(row.get("vram_peak_allocated_mb")) is not None
            ],
            default=None,
        ),
        "full_universe": True,
        "limit_tickers": None,
        "save_run": False,
        "wandb": False,
        "db_write": False,
        "inference_saved": False,
    }
    process_path.write_text(
        json.dumps(_clean_json(process), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": candidate.candidate_id,
                "event": "end",
                "status": status,
                "exit_code": exit_code,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
        )
    )
    return process


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config") or {})
    model_cls = MODEL_REGISTRY[config["model"]]
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    model_role = str(config.get("model_role") or config.get("output_role") or "band").lower()
    kwargs = {
        "n_features": int(config.get("n_features") or len(feature_columns)),
        "seq_len": int(config["seq_len"]),
        "horizon": int(config["horizon"]),
        "dropout": float(config.get("dropout", 0.2)),
        "band_mode": config.get("band_mode", "direct"),
        "num_tickers": int(config.get("num_tickers", 0)),
        "ticker_emb_dim": int(config.get("ticker_emb_dim", 32)),
        "output_role": model_role,
    }
    if config["model"] == "cnn_lstm":
        kwargs["use_direction_head"] = bool(config.get("use_direction_head", False))
        kwargs["fp32_modules"] = str(config.get("fp32_modules", "none"))
    if config["model"] == "tide":
        use_future_covariate = bool(config.get("use_future_covariate", True))
        kwargs["future_cov_dim"] = (
            config.get("future_cov_dim", FUTURE_COVARIATE_DIM) if use_future_covariate else 0
        )
    if config["model"] == "patchtst":
        kwargs["use_revin"] = bool(config.get("use_revin", True))
        kwargs["ci_aggregate"] = config.get("ci_aggregate", "target")
        kwargs["target_channel_idx"] = int(config.get("target_channel_idx", 0))
        kwargs["ci_target_fast"] = bool(config.get("ci_target_fast", False))
        kwargs["patch_len"] = int(config.get("patch_len", 16))
        kwargs["stride"] = int(config.get("patch_stride", config.get("stride", 8)))
        kwargs["d_model"] = int(config.get("patchtst_d_model", 128))
        kwargs["n_heads"] = int(config.get("patchtst_n_heads", 8))
        kwargs["n_layers"] = int(config.get("patchtst_n_layers", 3))
    model = model_cls(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint["feature_mean"], checkpoint["feature_std"]


def build_split_for_checkpoint(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    checkpoint_config: dict[str, Any],
    source_data_hash: str,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset, Any]:
    registry_path = checkpoint_config.get("ticker_registry_path")
    if not registry_path:
        raise ValueError("checkpoint에 ticker_registry_path가 없다.")
    registry = load_registry(TIMEFRAME, Path(str(registry_path)))
    mapping = registry.get("mapping") or {}
    tickers = sorted(set(mapping).intersection(set(indicators["ticker"].unique())))
    feature_frame = indicators[indicators["ticker"].isin(tickers)].copy()
    price_frame = price[price["ticker"].isin(tickers)].copy()
    plan = build_dataset_plan(
        feature_frame,
        timeframe=TIMEFRAME,
        seq_len=int(checkpoint_config["seq_len"]),
        horizon=int(checkpoint_config["horizon"]),
        ticker_registry=registry,
        ticker_registry_path=str(registry_path),
        market_data_provider=PROVIDER,
        source_data_hash=source_data_hash,
        split_mode="calendar_aligned",
    )
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_frame[feature_frame["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price_frame[price_frame["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=int(checkpoint_config["seq_len"]),
        horizon=int(checkpoint_config["horizon"]),
        ticker_registry=registry,
        include_future_covariate=bool(checkpoint_config.get("use_future_covariate", True)),
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test, calendar_plan = split_sequence_dataset_calendar_aligned(
        dataset,
        purge_gap_trading_days=plan.h_max,
        min_fold_samples=plan.min_fold_samples,
    )
    apply_calendar_split_metadata(plan, calendar_plan)
    return train, val, test, plan


def select_bundle_features_with_checkpoint_stats(
    bundle: SequenceDataset,
    *,
    feature_columns: list[str],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> SequenceDataset:
    indices = [MODEL_FEATURE_COLUMNS.index(column) for column in feature_columns]
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        copied = dict(arrays)
        copied["features"] = arrays["features"][:, indices].copy()
        ticker_arrays[ticker] = copied
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=feature_mean.to(dtype=torch.float32).index_select(0, index_tensor),
        std=feature_std.to(dtype=torch.float32).index_select(0, index_tensor),
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )


def evaluate_band_bundle(
    *,
    model: Any,
    bundle: SequenceDataset,
    device: torch.device,
    config: dict[str, Any],
    squeeze_breakout_threshold: float | None,
) -> dict[str, Any]:
    loader = make_loader(bundle, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    actual_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for (
            features,
            line_target,
            band_target,
            raw_future_returns,
            ticker_id,
            future_covariates,
        ) in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = forward_model(model, features, ticker_id, future_covariates)
            if isinstance(output, ForecastOutput):
                _, lower, upper = apply_band_postprocess(
                    output.line.detach().cpu().to(torch.float32),
                    output.lower_band.detach().cpu().to(torch.float32),
                    output.upper_band.detach().cpu().to(torch.float32),
                )
            elif isinstance(output, BandOutput):
                raw_lower = output.lower_band.detach().cpu().to(torch.float32)
                raw_upper = output.upper_band.detach().cpu().to(torch.float32)
                lower = torch.minimum(raw_lower, raw_upper)
                upper = torch.maximum(raw_lower, raw_upper)
            else:
                raise TypeError(
                    f"band_model 평가에서 허용하지 않는 출력 타입: {type(output).__name__}"
                )
            lower_predictions.append(lower)
            upper_predictions.append(upper)
            actual_targets.append(raw_future_returns.detach().cpu().to(torch.float32))
    lower_t = torch.cat(lower_predictions, dim=0)
    upper_t = torch.cat(upper_predictions, dim=0)
    actual_t = torch.cat(actual_targets, dim=0)
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=bundle.metadata,
        line_predictions=line_t,
        lower_predictions=lower_t,
        upper_predictions=upper_t,
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=float(config.get("q_low", 0.15)),
        q_high=float(config.get("q_high", 0.85)),
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        include_legacy_overlay_diagnostics=False,
    )
    lower_np = lower_t.numpy()
    upper_np = upper_t.numpy()
    actual_np = actual_t.numpy()
    date_cs = _date_cross_sectional_width_ic(
        metadata=bundle.metadata,
        lower=lower_np,
        upper=upper_np,
        actual=actual_np,
    )
    metrics = {key: _safe_float(summary.get(key)) for key in BAND_METRIC_KEYS if key in summary}
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    return metrics


def evaluate_candidate_checkpoint(
    *,
    candidate: Stage2Candidate,
    process: dict[str, Any],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    device: str,
    source_data_hash: str,
) -> dict[str, Any]:
    if process.get("status") != "PASS":
        return {
            "candidate_id": candidate.candidate_id,
            "status": "skipped",
            "reason": "training_failed",
        }
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path:
        return {
            "candidate_id": candidate.candidate_id,
            "status": "failed",
            "reason": "checkpoint_path_missing",
        }
    model, config, feature_mean, feature_std = load_model_from_checkpoint(checkpoint_path)
    train, val, test, plan = build_split_for_checkpoint(
        price=price,
        indicators=indicators,
        checkpoint_config=config,
        source_data_hash=source_data_hash,
    )
    train_targets = collect_targets(train)
    squeeze_threshold = float(np.nanquantile(np.abs(train_targets.reshape(-1)), 0.80))
    feature_columns = list(config.get("feature_columns") or resolve_feature_columns(FEATURE_SET))
    val_selected = select_bundle_features_with_checkpoint_stats(
        val,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_selected = select_bundle_features_with_checkpoint_stats(
        test,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    eval_device = resolve_device(device)
    model = model.to(eval_device)
    try:
        val_metrics = evaluate_band_bundle(
            model=model,
            bundle=val_selected,
            device=eval_device,
            config=config,
            squeeze_breakout_threshold=squeeze_threshold,
        )
        test_metrics = evaluate_band_bundle(
            model=model,
            bundle=test_selected,
            device=eval_device,
            config=config,
            squeeze_breakout_threshold=squeeze_threshold,
        )
    finally:
        if eval_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "candidate_id": candidate.candidate_id,
        "status": "completed",
        "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "eligible_ticker_count": len(plan.eligible_tickers),
        "input_ticker_count": int(plan.input_ticker_count),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "checkpoint_config": {
            "model": config.get("model"),
            "model_role": config.get("model_role"),
            "output_role": config.get("output_role"),
            "feature_set": config.get("feature_set"),
            "q_low": config.get("q_low"),
            "q_high": config.get("q_high"),
            "band_mode": config.get("band_mode"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "checkpoint_selection": config.get("checkpoint_selection"),
            "split_mode": config.get("split_mode"),
        },
        "val_metrics": val_metrics,
        "test_metrics_readonly": test_metrics,
        "test_metric_usage": "diagnostic_only",
    }


def _better_than_any_baseline(
    value: float | None, baseline_values: list[float], *, higher: bool
) -> bool:
    if value is None or not baseline_values:
        return False
    return (
        any(value >= base for base in baseline_values)
        if higher
        else any(value <= base for base in baseline_values)
    )


def classify_candidate(
    *,
    candidate: Stage2Candidate,
    process: dict[str, Any],
    evaluation: dict[str, Any],
    stage1: dict[str, Any],
) -> dict[str, Any]:
    if process.get("status") != "PASS":
        return {
            "decision": "fail",
            "failure_category": process.get("failure_category") or "runtime",
            "failure_reasons": [process.get("failure_reason") or "training_failed"],
            "advantage_flags": {},
        }
    if evaluation.get("status") != "completed":
        return {
            "decision": "fail",
            "failure_category": "runtime",
            "failure_reasons": [evaluation.get("reason") or "evaluation_failed"],
            "advantage_flags": {},
        }
    metrics = evaluation.get("val_metrics") or {}
    q_profile = stage1["profiles"]["by_q"][candidate.q_label]
    baseline_rows = q_profile["validation_rows"]
    baseline_metric_values = {
        metric: [
            float(row[metric]) for row in baseline_rows if _safe_float(row.get(metric)) is not None
        ]
        for metric in [
            "coverage_abs_error",
            "lower_breach_abs_error",
            "asymmetric_interval_score",
            "band_width_ic",
            "downside_width_ic",
            "p90_band_width",
        ]
    }
    p90 = _safe_float(metrics.get("p90_band_width"))
    p90_overwide = _safe_float(q_profile.get("p90_overwide_threshold"))
    coverage_abs_error = _safe_float(metrics.get("coverage_abs_error"))
    lower_breach_rate = _safe_float(metrics.get("lower_breach_rate"))
    interval_score = _safe_float(metrics.get("asymmetric_interval_score"))
    fatal_reasons: list[str] = []
    if coverage_abs_error is None or lower_breach_rate is None or interval_score is None:
        fatal_reasons.append("metric_missing_or_nan")
    elif coverage_abs_error > 0.20:
        fatal_reasons.append("coverage_collapse")
    if lower_breach_rate is not None and lower_breach_rate > candidate.q_low + 0.20:
        fatal_reasons.append("lower_breach_collapse")
    if p90 is not None and p90_overwide is not None and p90 > p90_overwide * 2.0:
        fatal_reasons.append("band_width_overwide_collapse")

    advantage_flags = {
        "coverage_abs_error_better_than_baseline_profile": _better_than_any_baseline(
            coverage_abs_error,
            baseline_metric_values["coverage_abs_error"],
            higher=False,
        ),
        "lower_breach_near_nominal_or_profile": (
            lower_breach_rate is not None and lower_breach_rate <= candidate.q_low + 0.05
        )
        or _better_than_any_baseline(
            _safe_float(metrics.get("lower_breach_abs_error")),
            baseline_metric_values["lower_breach_abs_error"],
            higher=False,
        ),
        "interval_score_better_than_baseline_profile": _better_than_any_baseline(
            interval_score,
            baseline_metric_values["asymmetric_interval_score"],
            higher=False,
        ),
        "band_width_ic_at_or_above_baseline_profile": _better_than_any_baseline(
            _safe_float(metrics.get("band_width_ic")),
            baseline_metric_values["band_width_ic"],
            higher=True,
        ),
        "downside_width_ic_at_or_above_baseline_profile": _better_than_any_baseline(
            _safe_float(metrics.get("downside_width_ic")),
            baseline_metric_values["downside_width_ic"],
            higher=True,
        ),
        "p90_width_not_overwide": p90 is not None and (p90_overwide is None or p90 <= p90_overwide),
    }
    if (
        advantage_flags["coverage_abs_error_better_than_baseline_profile"]
        and not advantage_flags["p90_width_not_overwide"]
        and not advantage_flags["interval_score_better_than_baseline_profile"]
    ):
        advantage_flags["coverage_abs_error_better_than_baseline_profile"] = False
        fatal_reasons.append("coverage_only_by_overwide_band")

    if fatal_reasons:
        return {
            "decision": "fail",
            "failure_category": "metric",
            "failure_reasons": fatal_reasons,
            "advantage_flags": advantage_flags,
        }
    advantage_count = sum(1 for value in advantage_flags.values() if value)
    if advantage_count >= 3:
        decision = "profile_pass"
    elif advantage_count >= 1:
        decision = "research_reserve"
    else:
        decision = "fail"
    return {
        "decision": decision,
        "failure_category": None if decision != "fail" else "metric",
        "failure_reasons": [] if decision != "fail" else ["no_baseline_profile_signal"],
        "advantage_flags": advantage_flags,
    }


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if _safe_float(value) is not None]
    return float(np.mean(finite)) if finite else None


def attach_candidate_scores(rows: list[dict[str, Any]]) -> None:
    weights = {
        "asymmetric_interval_score": 0.30,
        "coverage_abs_error": 0.20,
        "lower_breach_abs_error": 0.15,
        "band_width_ic": 0.15,
        "downside_width_ic": 0.10,
        "p90_band_width": 0.05,
        "squeeze_breakout_rate": 0.05,
    }
    for metric, weight in weights.items():
        values = {
            row["candidate_id"]: float(row[metric])
            for row in rows
            if row.get("status") == "PASS" and _safe_float(row.get(metric)) is not None
        }
        ranks = _rank_scores(values, higher_is_better=metric in HIGHER_IS_BETTER)
        for row in rows:
            if row["candidate_id"] in ranks:
                row.setdefault("_score_components", {})[metric] = (
                    ranks[row["candidate_id"]] * weight
                )
    for row in rows:
        components = row.pop("_score_components", {})
        row["band_selection_score"] = sum(components.values()) if components else None
        row["band_selection_score_components"] = components


def build_stage2_metrics(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    stage1: dict[str, Any],
    source_data_hash: str,
    force: bool,
    candidates_filter: list[str] | None,
) -> dict[str, Any]:
    overlay = prepare_snapshot_overlay()
    selected_q = stage1["profiles"]["selected_stage2_q"]
    candidates = build_stage2_candidates(selected_q)
    if candidates_filter:
        wanted = set(candidates_filter)
        candidates = [candidate for candidate in candidates if candidate.candidate_id in wanted]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processes: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        process = run_candidate(candidate, device=device, force=force)
        processes[candidate.candidate_id] = process
        eval_path = LOG_DIR / "stage2" / candidate.candidate_id / "evaluation.json"
        existing_eval = _read_json(eval_path) if eval_path.exists() and not force else None
        if (
            process.get("status") == "PASS"
            and existing_eval
            and existing_eval.get("status") == "completed"
        ):
            evaluation = existing_eval
        else:
            try:
                evaluation = evaluate_candidate_checkpoint(
                    candidate=candidate,
                    process=process,
                    price=price,
                    indicators=indicators,
                    device=device,
                    source_data_hash=source_data_hash,
                )
            except Exception as exc:  # noqa: BLE001
                evaluation = {
                    "candidate_id": candidate.candidate_id,
                    "status": "failed",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            eval_path.write_text(
                json.dumps(_clean_json(evaluation), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        evaluations[candidate.candidate_id] = evaluation
        classification = classify_candidate(
            candidate=candidate,
            process=process,
            evaluation=evaluation,
            stage1=stage1,
        )
        row = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "model": candidate.model,
            "seq_len": candidate.seq_len,
            "q_label": candidate.q_label,
            "q_low": candidate.q_low,
            "q_high": candidate.q_high,
            "band_mode": candidate.band_mode,
            "status": process.get("status"),
            "exit_code": process.get("exit_code"),
            "elapsed_seconds": process.get("elapsed_seconds"),
            "epoch_seconds_mean": _mean(process.get("epoch_seconds") or []),
            "vram_peak_allocated_mb": process.get("vram_peak_allocated_mb"),
            "eligible_ticker_count": evaluation.get("eligible_ticker_count"),
            "train_rows": (evaluation.get("split_rows") or {}).get("train"),
            "val_rows": (evaluation.get("split_rows") or {}).get("val"),
            "test_rows": (evaluation.get("split_rows") or {}).get("test"),
            "run_id": process.get("run_id"),
            "checkpoint_path": process.get("checkpoint_path"),
            "decision": classification["decision"],
            "failure_category": classification["failure_category"],
            "failure_reasons": classification["failure_reasons"],
            "advantage_flags": classification["advantage_flags"],
        }
        for key in BAND_METRIC_KEYS:
            row[key] = (evaluation.get("val_metrics") or {}).get(key)
        rows.append(row)
    attach_candidate_scores(rows)
    stage3_candidates = [
        row["candidate_id"]
        for row in sorted(
            [row for row in rows if row["decision"] in {"profile_pass", "research_reserve"}],
            key=lambda item: float(item.get("band_selection_score") or -1),
            reverse=True,
        )
    ]
    final_status = "FAIL_NO_MODEL_SIGNAL"
    if any(row["decision"] == "profile_pass" for row in rows):
        final_status = "PASS_STAGE2_CANDIDATES_FOUND"
    elif any(row["decision"] == "research_reserve" for row in rows):
        final_status = "WARN_RESEARCH_RESERVE_ONLY"
    return {
        "status": "PASS",
        "final_status": final_status,
        "overlay": overlay,
        "selected_q": selected_q,
        "candidates": [asdict(candidate) for candidate in candidates],
        "processes": processes,
        "evaluations": evaluations,
        "summary_rows": rows,
        "stage3_research_candidates": stage3_candidates[:4],
        "full_universe_execution": True,
        "epochs": STAGE2_EPOCHS,
        "seed": SEED,
        "test_policy": {
            "test_metrics_emitted": True,
            "test_metrics_usage": "diagnostic_only",
            "candidate_selection_uses_test": False,
        },
        "forbidden_actions": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "supabase_bulk_read_write": False,
            "wandb": False,
            "composite": False,
            "line_warning_combined": False,
        },
    }


def write_metrics(metrics: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            _clean_json(metrics),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def write_summary_csv(metrics: dict[str, Any]) -> None:
    columns = [
        "stage",
        "record_type",
        "id",
        "family",
        "model",
        "split",
        "q_label",
        "decision",
        "band_selection_score",
        "elapsed_seconds",
        "epoch_seconds_mean",
        "vram_peak_allocated_mb",
        "empirical_coverage",
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "lower_breach_abs_error",
        "upper_breach_abs_error",
        "asymmetric_interval_score",
        "median_band_width",
        "p90_band_width",
        "band_width_ic",
        "downside_width_ic",
        "width_bucket_realized_vol_ratio",
        "squeeze_breakout_rate",
        "diagnostic_only",
        "failure_category",
        "failure_reasons",
        "run_id",
        "checkpoint_path",
    ]
    rows: list[dict[str, Any]] = []
    for row in metrics.get("stage1", {}).get("rows", []):
        rows.append(
            {
                "stage": "stage1",
                "record_type": "baseline",
                "id": row.get("baseline"),
                "split": row.get("split"),
                "q_label": row.get("q_label"),
                "diagnostic_only": row.get("diagnostic_only"),
                **{key: row.get(key) for key in BAND_METRIC_KEYS},
            }
        )
    stage2_payload = metrics.get("stage2") or {}
    for row in stage2_payload.get("summary_rows", []):
        rows.append(
            {
                "stage": "stage2",
                "record_type": "model",
                "id": row.get("candidate_id"),
                "family": row.get("family"),
                "model": row.get("model"),
                "split": "val",
                "q_label": row.get("q_label"),
                "decision": row.get("decision"),
                "band_selection_score": row.get("band_selection_score"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "epoch_seconds_mean": row.get("epoch_seconds_mean"),
                "vram_peak_allocated_mb": row.get("vram_peak_allocated_mb"),
                "diagnostic_only": False,
                "failure_category": row.get("failure_category"),
                "failure_reasons": row.get("failure_reasons"),
                "run_id": row.get("run_id"),
                "checkpoint_path": row.get("checkpoint_path"),
                **{key: row.get(key) for key in BAND_METRIC_KEYS},
            }
        )
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _baseline_table(stage1: dict[str, Any]) -> str:
    lines = [
        "| q | baseline | split | cov_abs | lower | upper | interval | width_ic | downside_ic | p90 | test용도 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in stage1.get("rows", []):
        lines.append(
            "| {q} | {base} | {split} | {cov} | {lower} | {upper} | {interval} | {wic} | {dic} | {p90} | {diag} |".format(
                q=row.get("q_label"),
                base=row.get("baseline"),
                split=row.get("split"),
                cov=_fmt(row.get("coverage_abs_error")),
                lower=_fmt(row.get("lower_breach_rate")),
                upper=_fmt(row.get("upper_breach_rate")),
                interval=_fmt(row.get("asymmetric_interval_score")),
                wic=_fmt(row.get("band_width_ic")),
                dic=_fmt(row.get("downside_width_ic")),
                p90=_fmt(row.get("p90_band_width")),
                diag="diagnostic_only" if row.get("diagnostic_only") else "selection",
            )
        )
    return "\n".join(lines)


def _model_table(stage2: dict[str, Any]) -> str:
    lines = [
        "| 후보 | family | q | 판정 | cov_abs | lower | upper | interval | width_ic | downside_ic | p90 | score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        stage2.get("summary_rows", []),
        key=lambda item: float(item.get("band_selection_score") or -1),
        reverse=True,
    ):
        lines.append(
            "| {cid} | {family} | {q} | {decision} | {cov} | {lower} | {upper} | {interval} | {wic} | {dic} | {p90} | {score} |".format(
                cid=row.get("candidate_id"),
                family=row.get("family"),
                q=row.get("q_label"),
                decision=row.get("decision"),
                cov=_fmt(row.get("coverage_abs_error")),
                lower=_fmt(row.get("lower_breach_rate")),
                upper=_fmt(row.get("upper_breach_rate")),
                interval=_fmt(row.get("asymmetric_interval_score")),
                wic=_fmt(row.get("band_width_ic")),
                dic=_fmt(row.get("downside_width_ic")),
                p90=_fmt(row.get("p90_band_width")),
                score=_fmt(row.get("band_selection_score")),
            )
        )
    return "\n".join(lines)


def write_reports(metrics: dict[str, Any]) -> None:
    stage0 = metrics["stage0"]
    stage1 = metrics.get("stage1", {})
    stage2 = metrics.get("stage2", {})
    feature_columns = stage0["feature_quality"]["feature_set_columns"]
    REPORT_STAGE0_PATH.write_text(
        "\n".join(
            [
                "# CP178-BM 1W Band 500 Stage 0 Preflight Report",
                "",
                f"- 생성 시각 UTC: `{metrics['created_at_utc']}`",
                f"- provider/source: `{PROVIDER}/{PROVIDER}`",
                f"- source_data_hash: `{stage0['source_data_hash']}`",
                f"- timeframe/horizon: `{TIMEFRAME}/h{HORIZON}`",
                f"- horizon 계약: 기존 1W pipeline `default_horizon(1W)=4`를 사용했다. h5로 변경하지 않았다.",
                f"- feature_version: `{FEATURE_CONTRACT_VERSION}`",
                f"- feature_set: `{FEATURE_SET}`",
                f"- feature columns ({len(feature_columns)}): `{', '.join(feature_columns)}`",
                f"- 최신 1W 완료 주: `{stage0['week_status']['latest_week_complete']}` latest=`{stage0['week_status']['date_max']}` cutoff=`{stage0['week_status']['latest_complete_friday_cutoff']}`",
                f"- eligible ticker: `{stage0['dataset_plan']['eligible_ticker_count']}` / input `{stage0['dataset_plan']['input_ticker_count']}`",
                f"- excluded ticker count: `{stage0['dataset_plan']['excluded_ticker_count']}`",
                f"- split rows: train `{stage0['split']['train_rows']}`, validation `{stage0['split']['val_rows']}`, test `{stage0['split']['test_rows']}`",
                f"- split overlap count: `{stage0['split']['cross_split_date_overlap_count']}`",
                f"- feature NaN/Inf: `{stage0['feature_quality']['feature_nonfinite_count_after_contract_impute']}`",
                f"- target NaN/Inf: `{stage0['feature_quality']['target_nonfinite_total']}`",
                f"- duplicate rows: price `{stage0['duplicates']['price_duplicate_ticker_date_source_rows']}`, indicator `{stage0['duplicates']['indicator_duplicate_ticker_timeframe_date_source_rows']}`",
                f"- Stage 0 status: `{stage0['status']}`",
                f"- fail reasons: `{stage0['contract_fail_reasons']}`",
                "",
                "## Sample Sufficiency",
                "",
                "event count 단위는 horizon point다.",
                "",
                "| q | train events | validation events | fold1 test | fold2 test | fold3 test | pass |",
                "|---|---:|---:|---:|---:|---:|---|",
                *[
                    "| {q} | {train} | {val} | {f1} | {f2} | {f3} | {ok} |".format(
                        q=q_label,
                        train=info["split_event_counts"]["train"]["total_event_points"],
                        val=info["split_event_counts"]["validation"]["total_event_points"],
                        f1=info["stage5_expected_fold_test_counts"]["fold_1_test"][
                            "total_event_points"
                        ],
                        f2=info["stage5_expected_fold_test_counts"]["fold_2_test"][
                            "total_event_points"
                        ],
                        f3=info["stage5_expected_fold_test_counts"]["fold_3_test"][
                            "total_event_points"
                        ],
                        ok=info["pass"],
                    )
                    for q_label, info in stage0["sample_sufficiency"]["thresholds"].items()
                ],
                "",
                "## Stress Month Mapping",
                "",
                json.dumps(
                    _clean_json(stage0["sample_sufficiency"]["special_months"]),
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "## Train Regime Coverage",
                "",
                json.dumps(
                    _clean_json(stage0["train_regime_coverage"]), ensure_ascii=False, indent=2
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_STAGE1_PATH.write_text(
        "\n".join(
            [
                "# CP178-BM 1W Band 500 Stage 1 Baseline Report",
                "",
                "- 1W는 1D와 별도 실험으로 처리했다.",
                "- q15/q85 편향 없이 q15/q85, q10/q90, q05/q95를 validation 기준으로 동등 비교했다.",
                "- test metric은 열렸더라도 diagnostic_only이며 기준 잠금과 후보 선정에 쓰지 않았다.",
                f"- Stage 2 선택 q: `{stage1.get('profiles', {}).get('selected_stage2_q', {}).get('q_label')}`",
                f"- 선택 근거: `{stage1.get('profiles', {}).get('selected_stage2_q', {}).get('selection_basis')}`",
                "",
                "## Baseline Metrics",
                "",
                _baseline_table(stage1),
                "",
                "## Baseline SOTA/Profile Lock",
                "",
                json.dumps(_clean_json(stage1.get("profiles", {})), ensure_ascii=False, indent=2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if stage2:
        REPORT_STAGE2_PATH.write_text(
            "\n".join(
                [
                    "# CP178-BM 1W Band 500 Stage 2 Model Zoo Smoke Report",
                    "",
                    f"- 500티커 full universe 실행 여부: `{stage2.get('full_universe_execution')}`",
                    f"- save-run/DB/inference/W&B: `false/false/false/false`",
                    f"- Stage 2 q: `{stage2.get('selected_q', {}).get('q_label')}`",
                    f"- epochs: `{stage2.get('epochs')}`",
                    f"- seed: `{stage2.get('seed')}`",
                    f"- final label: `{stage2.get('final_status')}`",
                    "- 후보 선정은 validation 기준만 사용했다. test는 diagnostic_only다.",
                    "",
                    "## Model Metrics",
                    "",
                    _model_table(stage2),
                    "",
                    "## Stage 3 Candidates",
                    "",
                    *[
                        f"- {candidate_id}"
                        for candidate_id in stage2.get("stage3_research_candidates", [])
                    ],
                    "",
                    "## 보내지 않을 후보와 이유",
                    "",
                    json.dumps(
                        _clean_json(
                            [
                                {
                                    "candidate_id": row.get("candidate_id"),
                                    "decision": row.get("decision"),
                                    "failure_category": row.get("failure_category"),
                                    "failure_reasons": row.get("failure_reasons"),
                                    "advantage_flags": row.get("advantage_flags"),
                                }
                                for row in stage2.get("summary_rows", [])
                                if row.get("decision") == "fail"
                            ]
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "",
                    "## Stage 3~5 시간 견적",
                    "",
                    "- Stage 3 calibration/full rescue: 2~4시간",
                    "- Stage 4 seed stability: 후보 2~4개 기준 6~12시간",
                    "- Stage 5 true walk-forward: 후보 1~2개 기준 12~24시간",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def current_python_process_snapshot() -> dict[str, Any]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CreationDate,CommandLine | "
        "ConvertTo-Json -Depth 3"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": type(exc).__name__, "message": str(exc)}


def current_cuda_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "nvidia-smi_not_found"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": type(exc).__name__, "message": str(exc)}


def determine_final_status(metrics: dict[str, Any]) -> str:
    stage0_status = metrics.get("stage0", {}).get("status")
    if stage0_status == "WARN_SAMPLE_THIN":
        return "WARN_SAMPLE_THIN"
    if stage0_status != "PASS":
        return "FAIL_CONTRACT"
    stage2 = metrics.get("stage2")
    if not stage2:
        return "PASS_STAGE1_BASELINE_READY"
    return str(stage2.get("final_status") or "FAIL_NO_MODEL_SIGNAL")


def run_cp(args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    price, indicators, price_manifest, indicator_manifest = load_source_frames()
    source_data_hash = str(
        indicator_manifest.get("source_data_hash")
        or indicator_manifest.get("price_source_data_hash")
        or price_manifest.get("source_data_hash")
        or SOURCE_DATA_HASH_FALLBACK
    )
    start_snapshot = {
        "python_processes": current_python_process_snapshot(),
        "cuda_compute_apps": current_cuda_snapshot(),
    }
    payload = build_split_payload(
        price=price, indicators=indicators, source_data_hash=source_data_hash
    )
    stage0 = build_stage0_metrics(
        price=price,
        indicators=indicators,
        price_manifest=price_manifest,
        indicator_manifest=indicator_manifest,
        payload=payload,
        source_data_hash=source_data_hash,
    )
    stage1: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None
    if stage0["status"] == "PASS":
        stage1 = build_stage1_metrics(payload)
        if not args.stop_after_stage1:
            stage2 = build_stage2_metrics(
                price=price,
                indicators=indicators,
                stage1=stage1,
                source_data_hash=source_data_hash,
                force=args.force,
                candidates_filter=args.candidates,
            )
    metrics: dict[str, Any] = {
        "cp": CP_NAME,
        "created_at_utc": _now_utc(),
        "stage_scope": "1W band 500 Stage 0~2 only",
        "stage3_or_later_executed": False,
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "process_snapshot_before": start_snapshot,
        "process_snapshot_after": {
            "python_processes": current_python_process_snapshot(),
            "cuda_compute_apps": current_cuda_snapshot(),
        },
        "outputs": {
            "stage0_report": str(REPORT_STAGE0_PATH),
            "stage1_report": str(REPORT_STAGE1_PATH),
            "stage2_report": str(REPORT_STAGE2_PATH),
            "metrics": str(METRICS_PATH),
            "summary_csv": str(SUMMARY_CSV_PATH),
            "logs": str(LOG_DIR),
        },
    }
    metrics["final_status"] = determine_final_status(metrics)
    write_metrics(metrics)
    write_summary_csv(metrics)
    write_reports(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP178 1W band 500 Stage 0~2 runner")
    parser.add_argument(
        "--stop-after-stage1", action="store_true", help="Stage 0/1까지만 실행한다."
    )
    parser.add_argument("--force", action="store_true", help="기존 Stage 2 PASS 후보도 재실행한다.")
    parser.add_argument(
        "--candidates", nargs="*", default=None, help="Stage 2 후보 id 일부만 실행한다."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_cp(args)
    print(
        json.dumps(
            {
                "cp": CP_NAME,
                "final_status": metrics.get("final_status"),
                "stage0_status": metrics.get("stage0", {}).get("status"),
                "stage2_status": (metrics.get("stage2") or {}).get("final_status"),
                "outputs": metrics.get("outputs"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
