from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "parquet"
DOCS_DIR = ROOT / "docs"

PRICE_PATH = DATA_DIR / "price_data_yfinance_1W.parquet"
INDICATOR_PATH = DATA_DIR / "indicators_yfinance_1W.parquet"
STOCK_INFO_PATH = DATA_DIR / "stock_info.parquet"
METRICS_PATH = DOCS_DIR / "cp123_1w_data_sufficiency_overfit_guard_metrics.json"
REPORT_PATH = DOCS_DIR / "cp123_1w_data_sufficiency_overfit_guard_report.md"

TIMEFRAME = "1W"
PROVIDER = "yfinance"
SOURCE = "yfinance"
SEQ_LEN = 104
HORIZON = 4
H_MAX = 12
MIN_FOLD_SAMPLES = 50
SPLIT_RATIO = (0.7, 0.15, 0.15)

# ai/splits.py의 현재 1W 후보 eligibility 정책을 read-only 감사 스크립트에 수동 반영한다.
FUNDAMENTAL_INSUFFICIENT_TICKERS = {
    "AMP",
    "APA",
    "BALL",
    "BK",
    "BXP",
    "CHTR",
    "CINF",
    "CPAY",
    "DUK",
    "EA",
    "EME",
    "EXE",
    "GLW",
    "HOOD",
    "INVH",
    "KEYS",
    "LMT",
    "MS",
    "PRU",
    "Q",
    "RF",
    "T",
    "TDG",
    "UBER",
    "VICI",
    "XYZ",
}

MODEL_FEATURE_COLUMNS = [
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "vol_change",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "rsi",
    "macd_ratio",
    "bb_position",
    "regime_calm",
    "regime_neutral",
    "regime_stress",
    "revenue",
    "net_income",
    "equity",
    "eps",
    "roe",
    "debt_ratio",
    "has_macro",
    "has_breadth",
    "has_fundamentals",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "month_end_distance",
    "quarter_end_distance",
    "is_month_end",
    "is_quarter_end",
    "month_start_distance",
    "is_month_start",
    "is_quarter_start",
]

REGIME_WINDOWS = {
    "2020_crash": ("2020-02-21", "2020-04-10"),
    "2022_bear": ("2022-01-07", "2022-12-30"),
    "2023_2024_bull": ("2023-01-06", "2024-12-27"),
}

CP_METRIC_FILES = [
    "cp112_bm_1w_band_smoke_metrics.json",
    "cp112_lm_1w_line_smoke_metrics.json",
    "cp113_bm_1w_band_limited_validation_metrics.json",
    "cp113_lm_1w_line_rescue_metrics.json",
    "cp114_bm_1w_band_candidate_expansion_metrics.json",
    "cp114_lm_1w_line_candidate_expansion_metrics.json",
    "cp118_bm_1w_band_feature_target_audit_metrics.json",
    "cp119_bm_1w_band_feature_group_experiment_metrics.json",
]


@dataclass(frozen=True)
class SplitSpec:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    sample_count: int
    effective_sample_count: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_date(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied["date"] = pd.to_datetime(copied["date"]).dt.normalize()
    return copied


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p10": float(numeric.quantile(0.10)),
        "p50": float(numeric.quantile(0.50)),
        "p90": float(numeric.quantile(0.90)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _make_split_spec(row_count: int) -> SplitSpec | None:
    sample_count = max(row_count - SEQ_LEN - H_MAX + 1, 0)
    effective = max(sample_count - (2 * H_MAX), 0)
    if sample_count < 1 or effective < 1:
        return None
    train_count = math.floor(effective * SPLIT_RATIO[0])
    val_count = math.floor(effective * SPLIT_RATIO[1])
    test_count = effective - train_count - val_count
    if train_count < MIN_FOLD_SAMPLES or val_count < MIN_FOLD_SAMPLES or test_count < MIN_FOLD_SAMPLES:
        return None
    train = (0, train_count)
    val = (train[1] + H_MAX, train[1] + H_MAX + val_count)
    test = (val[1] + H_MAX, val[1] + H_MAX + test_count)
    if test[1] > sample_count:
        return None
    return SplitSpec(train=train, val=val, test=test, sample_count=sample_count, effective_sample_count=effective)


def _split_name(index: int, spec: SplitSpec) -> str | None:
    for name in ("train", "val", "test"):
        start, end = getattr(spec, name)
        if start <= index < end:
            return name
    return None


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price = _to_date(pd.read_parquet(PRICE_PATH))
    indicators = _to_date(pd.read_parquet(INDICATOR_PATH))
    stock_info = pd.read_parquet(STOCK_INFO_PATH).copy()
    for frame in (price, indicators, stock_info):
        if "ticker" in frame.columns:
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
    price = price[(price["timeframe"].astype(str).str.upper() == TIMEFRAME) & (price["source"] == SOURCE)]
    indicators = indicators[
        (indicators["timeframe"].astype(str).str.upper() == TIMEFRAME) & (indicators["source"] == SOURCE)
    ]
    return price, indicators, stock_info


def _coverage(price: pd.DataFrame, indicators: pd.DataFrame, stock_info: pd.DataFrame) -> dict[str, Any]:
    price_counts = price.groupby("ticker").size().sort_values()
    indicator_counts = indicators.groupby("ticker").size().sort_values()
    stock_sectors = stock_info.assign(
        sector=stock_info.get("sector", pd.Series(dtype=object)).fillna("UNKNOWN").astype(str)
    )
    indicator_tickers = set(indicator_counts.index)
    eligible = sorted(t for t in indicator_tickers if t not in FUNDAMENTAL_INSUFFICIENT_TICKERS)
    sector_counts = (
        stock_sectors[stock_sectors["ticker"].isin(indicator_tickers)]
        .groupby("sector")["ticker"]
        .nunique()
        .sort_values(ascending=False)
    )
    eligible_sector_counts = (
        stock_sectors[stock_sectors["ticker"].isin(eligible)]
        .groupby("sector")["ticker"]
        .nunique()
        .sort_values(ascending=False)
    )
    return {
        "price": {
            "rows": int(len(price)),
            "ticker_count": int(price["ticker"].nunique()),
            "date_min": str(price["date"].min().date()),
            "date_max": str(price["date"].max().date()),
            "rows_per_ticker": _quantiles(price_counts),
            "duplicate_ticker_date_source": int(price.duplicated(["ticker", "date", "source"]).sum()),
        },
        "indicators": {
            "rows": int(len(indicators)),
            "ticker_count": int(indicators["ticker"].nunique()),
            "date_min": str(indicators["date"].min().date()),
            "date_max": str(indicators["date"].max().date()),
            "rows_per_ticker": _quantiles(indicator_counts),
            "duplicate_ticker_date_source": int(indicators.duplicated(["ticker", "date", "source"]).sum()),
        },
        "stock_info": {
            "rows": int(len(stock_info)),
            "ticker_count": int(stock_info["ticker"].nunique()),
            "sector_count": int(sector_counts.shape[0]),
            "sector_ticker_counts": {str(k): int(v) for k, v in sector_counts.items()},
            "eligible_sector_ticker_counts": {str(k): int(v) for k, v in eligible_sector_counts.items()},
            "missing_sector_tickers": sorted(
                stock_sectors.loc[
                    stock_sectors["ticker"].isin(indicator_tickers)
                    & (
                        stock_sectors["sector"].isna()
                        | (stock_sectors["sector"].astype(str).str.strip() == "")
                        | (stock_sectors["sector"].astype(str).str.upper() == "UNKNOWN")
                    ),
                    "ticker",
                ].astype(str)
            ),
        },
        "eligibility": {
            "input_ticker_count": int(len(indicator_tickers)),
            "eligible_ticker_count": int(len(eligible)),
            "excluded_tickers": sorted(indicator_tickers - set(eligible)),
            "excluded_reason": "Gate fundamentals",
        },
    }


def _build_samples(price: pd.DataFrame, indicators: pd.DataFrame, stock_info: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_prices = price[["ticker", "date", "adjusted_close"]].rename(columns={"adjusted_close": "target_close"})
    merged = indicators.merge(target_prices, on=["ticker", "date"], how="inner").sort_values(["ticker", "date"])
    sector_map = stock_info.set_index("ticker")["sector"].to_dict() if "sector" in stock_info.columns else {}
    records: list[dict[str, Any]] = []
    specs: dict[str, Any] = {}
    for ticker, ticker_frame in merged.groupby("ticker", sort=True):
        if ticker in FUNDAMENTAL_INSUFFICIENT_TICKERS:
            continue
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        spec = _make_split_spec(len(ticker_frame))
        if spec is None:
            continue
        closes = pd.to_numeric(ticker_frame["target_close"], errors="coerce").to_numpy(dtype=float)
        dates = pd.to_datetime(ticker_frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        specs[str(ticker)] = {
            "row_count": int(len(ticker_frame)),
            "sample_count": int(spec.sample_count),
            "effective_sample_count": int(spec.effective_sample_count),
            "train": {"start": spec.train[0], "end": spec.train[1], "count": spec.train[1] - spec.train[0]},
            "val": {"start": spec.val[0], "end": spec.val[1], "count": spec.val[1] - spec.val[0]},
            "test": {"start": spec.test[0], "end": spec.test[1], "count": spec.test[1] - spec.test[0]},
        }
        for sample_index in range(spec.sample_count):
            split = _split_name(sample_index, spec)
            if split is None:
                continue
            end_idx = SEQ_LEN - 1 + sample_index
            future_start = end_idx + 1
            future_end = future_start + HORIZON
            if future_end > len(closes):
                continue
            anchor_close = closes[end_idx]
            if not np.isfinite(anchor_close) or anchor_close == 0:
                continue
            future_returns = (closes[future_start:future_end] / anchor_close) - 1.0
            if len(future_returns) != HORIZON or not np.isfinite(future_returns).all():
                continue
            indicator_row = ticker_frame.iloc[end_idx]
            records.append(
                {
                    "ticker": str(ticker),
                    "sector": str(sector_map.get(str(ticker), "UNKNOWN") or "UNKNOWN"),
                    "split": split,
                    "sample_index": int(sample_index),
                    "asof_date": dates[end_idx],
                    "target_h1": float(future_returns[0]),
                    "target_h2": float(future_returns[1]),
                    "target_h3": float(future_returns[2]),
                    "target_h4": float(future_returns[3]),
                    "target_min_h1_h4": float(np.min(future_returns)),
                    "abs_log_return": abs(_finite_float(indicator_row.get("log_return")) or 0.0),
                    "atr_ratio": _finite_float(indicator_row.get("atr_ratio")),
                    "regime_label": str(indicator_row.get("regime_label", "UNKNOWN") or "UNKNOWN"),
                    "regime_stress": _finite_float(indicator_row.get("regime_stress")) or 0.0,
                    "regime_calm": _finite_float(indicator_row.get("regime_calm")) or 0.0,
                }
            )
    return pd.DataFrame(records), specs


def _sample_distribution(samples: pd.DataFrame) -> dict[str, Any]:
    ticker_split = samples.groupby(["ticker", "split"]).size().unstack(fill_value=0)
    sector_split = samples.groupby(["sector", "split"]).size().unstack(fill_value=0)
    for column in ("train", "val", "test"):
        if column not in ticker_split.columns:
            ticker_split[column] = 0
        if column not in sector_split.columns:
            sector_split[column] = 0
    ticker_total = samples.groupby("ticker").size()
    sector_total = samples.groupby("sector").size().sort_values(ascending=False)
    return {
        "by_split": {str(k): int(v) for k, v in samples.groupby("split").size().items()},
        "ticker_sample_distribution_total": _quantiles(ticker_total),
        "ticker_sample_distribution_by_split": {
            split: _quantiles(ticker_split[split]) for split in ("train", "val", "test")
        },
        "sector_sample_counts_total": {str(k): int(v) for k, v in sector_total.items()},
        "sector_sample_counts_by_split": {
            str(sector): {split: int(row.get(split, 0)) for split in ("train", "val", "test")}
            for sector, row in sector_split.sort_index().iterrows()
        },
    }


def _split_ranges(samples: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = samples.copy()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    for split, group in frame.groupby("split", sort=False):
        out[str(split)] = {
            "rows": int(len(group)),
            "ticker_count": int(group["ticker"].nunique()),
            "sector_count": int(group["sector"].nunique()),
            "asof_min": str(group["asof_date"].min().date()),
            "asof_max": str(group["asof_date"].max().date()),
        }
    return out


def _target_summary(samples: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    flattened = samples.melt(
        id_vars=["split", "ticker", "sector", "asof_date"],
        value_vars=["target_h1", "target_h2", "target_h3", "target_h4"],
        var_name="horizon",
        value_name="target",
    )
    for split, group in samples.groupby("split"):
        flat_group = flattened[flattened["split"] == split]
        h4 = group["target_h4"]
        rows[str(split)] = {
            "h1_h4_flattened": {
                **_quantiles(flat_group["target"]),
                "negative_rate": float((flat_group["target"] < 0).mean()),
                "severe_downside_le_-5pct_count": int((flat_group["target"] <= -0.05).sum()),
                "severe_downside_le_-5pct_rate": float((flat_group["target"] <= -0.05).mean()),
                "tail_20pct_cross_section_events": _tail_event_count(flat_group),
            },
            "h4_terminal": {
                **_quantiles(h4),
                "negative_rate": float((h4 < 0).mean()),
                "severe_downside_le_-5pct_count": int((h4 <= -0.05).sum()),
                "severe_downside_le_-8pct_count": int((h4 <= -0.08).sum()),
                "severe_downside_le_-12pct_count": int((h4 <= -0.12).sum()),
            },
        }
    return rows


def _tail_event_count(flattened: pd.DataFrame) -> dict[str, Any]:
    count = 0
    date_count = 0
    for _, group in flattened.groupby(["asof_date", "horizon"], sort=False):
        if len(group) < 2:
            continue
        cutoff = group["target"].quantile(0.20)
        count += int((group["target"] <= cutoff).sum())
        date_count += 1
    return {"event_count": int(count), "date_horizon_groups": int(date_count)}


def _regime_summary(samples: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    abs_return_q20 = float(samples["abs_log_return"].quantile(0.20))
    abs_return_q80 = float(samples["abs_log_return"].quantile(0.80))
    atr_q20 = float(samples["atr_ratio"].dropna().quantile(0.20))
    atr_q80 = float(samples["atr_ratio"].dropna().quantile(0.80))
    for split, group in samples.groupby("split"):
        out[str(split)] = {
            "rows": int(len(group)),
            "abs_log_return": _quantiles(group["abs_log_return"]),
            "atr_ratio": _quantiles(group["atr_ratio"]),
            "low_vol_abs_return_count": int((group["abs_log_return"] <= abs_return_q20).sum()),
            "high_vol_abs_return_count": int((group["abs_log_return"] >= abs_return_q80).sum()),
            "low_vol_atr_count": int((group["atr_ratio"] <= atr_q20).sum()),
            "high_vol_atr_count": int((group["atr_ratio"] >= atr_q80).sum()),
            "regime_label_counts": {
                str(k): int(v) for k, v in group["regime_label"].value_counts(dropna=False).sort_index().items()
            },
            "regime_stress_mean": float(group["regime_stress"].mean()),
            "regime_calm_mean": float(group["regime_calm"].mean()),
        }
    return {
        "thresholds": {
            "abs_log_return_q20": abs_return_q20,
            "abs_log_return_q80": abs_return_q80,
            "atr_ratio_q20": atr_q20,
            "atr_ratio_q80": atr_q80,
        },
        "by_split": out,
    }


def _historical_regime_mapping(samples: pd.DataFrame) -> dict[str, Any]:
    frame = samples.copy()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    out: dict[str, Any] = {}
    for name, (start, end) in REGIME_WINDOWS.items():
        mask = frame["asof_date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        group = frame[mask]
        out[name] = {
            "calendar_range": {"start": start, "end": end},
            "sample_rows": int(len(group)),
            "split_counts": {str(k): int(v) for k, v in group.groupby("split").size().items()},
            "ticker_count": int(group["ticker"].nunique()) if not group.empty else 0,
            "asof_min": str(group["asof_date"].min().date()) if not group.empty else None,
            "asof_max": str(group["asof_date"].max().date()) if not group.empty else None,
        }
    return out


def _overlap_summary(samples: pd.DataFrame) -> dict[str, Any]:
    split_rows = samples.groupby("split").size()
    total = int(len(samples))
    label_effective = int(math.ceil(total / HORIZON))
    sequence_block_effective = float(total / SEQ_LEN)
    return {
        "nominal_split_samples": {str(k): int(v) for k, v in split_rows.items()},
        "nominal_total_samples": total,
        "adjacent_input_window_overlap_ratio": float((SEQ_LEN - 1) / SEQ_LEN),
        "adjacent_horizon_label_overlap_ratio": float((HORIZON - 1) / HORIZON),
        "label_nonoverlap_effective_sample_estimate": label_effective,
        "label_effective_ratio": float(label_effective / total) if total else None,
        "sequence_block_effective_sample_estimate": sequence_block_effective,
        "sequence_block_effective_ratio": float(sequence_block_effective / total) if total else None,
        "interpretation": "nominal rows는 교차단면 ticker 수와 overlapping weekly windows 때문에 독립 표본 수보다 크게 보인다.",
    }


def _cp_test_usage() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unique_run_ids: set[str] = set()
    test_reference_count = 0
    new_model_run_count = 0
    for filename in CP_METRIC_FILES:
        path = DOCS_DIR / filename
        if not path.exists():
            rows.append({"file": filename, "exists": False})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        experiments = data.get("experiments")
        candidates = data.get("candidates")
        report_count = 0
        run_count = 0
        reused_count = 0
        audit_only_reference = False
        run_ids: list[str] = []
        if isinstance(experiments, list):
            report_count = len(experiments)
            for item in experiments:
                if isinstance(item, dict):
                    source = str(item.get("source") or "")
                    if source.startswith("reused"):
                        reused_count += 1
                    run_id = item.get("run_id")
                    if isinstance(run_id, str) and run_id:
                        run_ids.append(run_id)
        elif isinstance(candidates, list):
            report_count = len(candidates)
            for item in candidates:
                if isinstance(item, dict):
                    run_id = item.get("run_id")
                    if isinstance(run_id, str) and run_id:
                        run_ids.append(run_id)
        elif data.get("test_band_metrics") or data.get("line_metrics") or data.get("bucket_line_metrics"):
            report_count = 1
            execution = data.get("execution", {})
            if isinstance(execution, dict) and isinstance(execution.get("run_id"), str):
                run_ids.append(execution["run_id"])
        elif "target_summary" in data or "candidate_registry" in data:
            report_count = 1
            audit_only_reference = True
        run_count = 0 if audit_only_reference else max(report_count - reused_count, 0)
        test_reference_count += report_count
        new_model_run_count += run_count
        unique_run_ids.update(run_ids)
        rows.append(
            {
                "file": filename,
                "exists": True,
                "test_or_candidate_reference_count": report_count,
                "new_or_replayed_model_run_count_estimate": run_count,
                "reused_reference_count": reused_count,
                "run_ids": sorted(run_ids),
            }
        )
    return {
        "files": rows,
        "test_or_candidate_reference_count_total": int(test_reference_count),
        "new_or_replayed_model_run_count_estimate_total": int(new_model_run_count),
        "unique_run_id_count": int(len(unique_run_ids)),
        "unique_run_ids": sorted(unique_run_ids),
        "bias_level": "HIGH" if test_reference_count >= 10 else "MEDIUM",
        "interpretation": "CP112~CP119에서 동일 1W test window를 여러 후보 비교와 watch/product 후보 판단에 반복 사용했다.",
    }


def _decision(
    coverage: dict[str, Any],
    target_summary: dict[str, Any],
    regime_summary: dict[str, Any],
    cp_usage: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    status = "PASS_WITH_GUARDS"
    if coverage["eligibility"]["eligible_ticker_count"] < 80:
        warnings.append("eligible ticker가 80개 미만이다.")
        status = "WARN"
    unknown_sector_count = int(coverage["stock_info"]["eligible_sector_ticker_counts"].get("UNKNOWN", 0))
    eligible_count = int(coverage["eligibility"]["eligible_ticker_count"])
    if eligible_count and unknown_sector_count / eligible_count > 0.20:
        warnings.append("stock_info sector UNKNOWN 비중이 높아 sector holdout 감사 신뢰도가 제한된다.")
    for split in ("train", "val", "test"):
        severe = target_summary.get(split, {}).get("h4_terminal", {}).get("severe_downside_le_-5pct_count", 0)
        high_vol = regime_summary["by_split"].get(split, {}).get("high_vol_abs_return_count", 0)
        if _safe_int(severe) < 100:
            warnings.append(f"{split} severe downside 표본이 100개 미만이다.")
            status = "WARN"
        if _safe_int(high_vol) < 100:
            warnings.append(f"{split} high volatility 표본이 100개 미만이다.")
            status = "WARN"
    if cp_usage["test_or_candidate_reference_count_total"] >= 10:
        warnings.append("동일 test set 반복 참조가 많아 후보 선택 bias 위험이 높다.")
    return {
        "status": status,
        "summary": "1W 데이터는 후보 실험을 계속할 수 있을 만큼 충분하지만, test 반복 사용 때문에 제품 후보 확정은 validation stability와 재현 split 없이는 금지해야 한다.",
        "warnings": warnings,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _write_report(metrics: dict[str, Any]) -> None:
    split_ranges = metrics["split_ranges"]
    target = metrics["target_distribution"]
    regimes = metrics["regime_distribution"]["by_split"]
    coverage = metrics["coverage"]
    cp_usage = metrics["cp112_cp119_test_usage"]
    target_rows = []
    for split in ("train", "val", "test"):
        h4 = target[split]["h4_terminal"]
        flat = target[split]["h1_h4_flattened"]
        target_rows.append(
            {
                "split": split,
                "rows": split_ranges[split]["rows"],
                "asof": f'{split_ranges[split]["asof_min"]} ~ {split_ranges[split]["asof_max"]}',
                "h4_mean": h4["mean"],
                "h4_p05": h4["p05"],
                "h4_p50": h4["p50"],
                "h4_p95": h4["p95"],
                "h4_severe": h4["severe_downside_le_-5pct_count"],
                "flat_severe_rate": flat["severe_downside_le_-5pct_rate"],
            }
        )
    regime_rows = []
    for split in ("train", "val", "test"):
        row = regimes[split]
        regime_rows.append(
            {
                "split": split,
                "high_abs": row["high_vol_abs_return_count"],
                "low_abs": row["low_vol_abs_return_count"],
                "high_atr": row["high_vol_atr_count"],
                "low_atr": row["low_vol_atr_count"],
                "stress_mean": row["regime_stress_mean"],
                "calm_mean": row["regime_calm_mean"],
            }
        )
    cp_rows = []
    for row in cp_usage["files"]:
        cp_rows.append(
            {
                "file": row["file"],
                "refs": row.get("test_or_candidate_reference_count", 0),
                "runs": row.get("new_or_replayed_model_run_count_estimate", 0),
                "reused": row.get("reused_reference_count", 0),
            }
        )
    report = f"""# CP123-DG 1W 데이터 충분성 및 과적합 가드 감사

생성일: 2026-05-06

## 1. 요약

최종 판정: `{metrics["decision"]["status"]}`

현재 1W yfinance 로컬 snapshot은 1W line/band 후보 실험을 계속할 수 있을 만큼의 기간, ticker, sector coverage를 갖고 있다. 다만 CP112~CP119에서 동일 test window를 반복 참조하며 후보를 좁힌 흔적이 커서, 다음 단계부터는 test를 중간 선택 기준으로 쓰면 안 된다.

핵심 결론:
- 1W indicator 기간: `{coverage["indicators"]["date_min"]}` ~ `{coverage["indicators"]["date_max"]}`
- input ticker: `{coverage["eligibility"]["input_ticker_count"]}`, eligible ticker: `{coverage["eligibility"]["eligible_ticker_count"]}`
- sector 수: 전체 `{coverage["stock_info"]["sector_count"]}`
- split: train `{split_ranges["train"]["asof_min"]}` ~ `{split_ranges["train"]["asof_max"]}`, val `{split_ranges["val"]["asof_min"]}` ~ `{split_ranges["val"]["asof_max"]}`, test `{split_ranges["test"]["asof_min"]}` ~ `{split_ranges["test"]["asof_max"]}`
- CP112~CP119 test/candidate 참조 수: `{cp_usage["test_or_candidate_reference_count_total"]}`
- 과적합 위험: `{cp_usage["bias_level"]}`
- 주요 경고: `{"; ".join(metrics["decision"]["warnings"]) if metrics["decision"]["warnings"] else "없음"}`

## 2. 데이터 coverage

| 항목 | 값 |
|---|---:|
| 1W price rows | {coverage["price"]["rows"]} |
| 1W indicator rows | {coverage["indicators"]["rows"]} |
| price ticker 수 | {coverage["price"]["ticker_count"]} |
| indicator ticker 수 | {coverage["indicators"]["ticker_count"]} |
| eligible ticker 수 | {coverage["eligibility"]["eligible_ticker_count"]} |
| excluded ticker | {", ".join(coverage["eligibility"]["excluded_tickers"])} |
| price duplicate `(ticker,date,source)` | {coverage["price"]["duplicate_ticker_date_source"]} |
| indicator duplicate `(ticker,date,source)` | {coverage["indicators"]["duplicate_ticker_date_source"]} |

sector별 eligible ticker 수:

```json
{json.dumps(coverage["stock_info"]["eligible_sector_ticker_counts"], ensure_ascii=False, indent=2)}
```

주의: `UNKNOWN` sector가 `{coverage["stock_info"]["eligible_sector_ticker_counts"].get("UNKNOWN", 0)}`개라서 현재 stock_info만으로 sector holdout을 설계하면 편향이 생길 수 있다. sector holdout은 stock_info sector 보강 후 적용하는 편이 안전하다.

## 3. Split 기간과 sample 분포

{_markdown_table(
        [
            {
                "split": split,
                "rows": split_ranges[split]["rows"],
                "tickers": split_ranges[split]["ticker_count"],
                "sectors": split_ranges[split]["sector_count"],
                "asof_min": split_ranges[split]["asof_min"],
                "asof_max": split_ranges[split]["asof_max"],
            }
            for split in ("train", "val", "test")
        ],
        [("split", "split"), ("rows", "rows"), ("tickers", "tickers"), ("sectors", "sectors"), ("asof_min", "asof_min"), ("asof_max", "asof_max")],
    )}

티커별 총 sample 수 분포:

```json
{json.dumps(metrics["sample_distribution"]["ticker_sample_distribution_total"], ensure_ascii=False, indent=2)}
```

sector별 split sample 수:

```json
{json.dumps(metrics["sample_distribution"]["sector_sample_counts_by_split"], ensure_ascii=False, indent=2)}
```

## 4. Target 분포 차이

아래 표는 h4 terminal raw future return 기준이다. 모델은 h1~h4 벡터를 쓰므로 metrics JSON에는 h1~h4 flatten 분포도 같이 기록했다.

{_markdown_table(
        target_rows,
        [
            ("split", "split"),
            ("rows", "rows"),
            ("asof", "asof"),
            ("h4_mean", "h4_mean"),
            ("h4_p05", "h4_p05"),
            ("h4_p50", "h4_p50"),
            ("h4_p95", "h4_p95"),
            ("h4 severe <= -5%", "h4_severe"),
            ("h1~h4 severe rate", "flat_severe_rate"),
        ],
    )}

판단:
- severe downside는 train/val/test 모두 존재한다.
- test는 2024-12-20 이후라 2020 crash나 2022 bear를 직접 포함하지 않는다.
- 따라서 test tail 성능이 과거 crisis tail 일반화까지 보장한다고 해석하면 안 된다.

## 5. Volatility regime 분포

{_markdown_table(
        regime_rows,
        [
            ("split", "split"),
            ("high abs-return", "high_abs"),
            ("low abs-return", "low_abs"),
            ("high atr", "high_atr"),
            ("low atr", "low_atr"),
            ("stress mean", "stress_mean"),
            ("calm mean", "calm_mean"),
        ],
    )}

split마다 high/low volatility sample은 충분히 있다. 하지만 2020/2022 같은 큰 구조적 stress는 train에만 존재하므로, 제품 후보 판단에는 regime별 metric을 별도로 붙여야 한다.

## 6. 역사적 regime의 split 위치

```json
{json.dumps(metrics["historical_regime_mapping"], ensure_ascii=False, indent=2)}
```

해석:
- 2020 crash: train
- 2022 bear: train
- 2023~2024 bull: train 일부, val 대부분, test 극소 구간
- test는 주로 2025~2026 최근 regime이다.

## 7. Overlapping window에 따른 과대평가

```json
{json.dumps(metrics["overlap"], ensure_ascii=False, indent=2)}
```

`seq_len=104`인 1W 인접 입력 window는 103주를 공유한다. 인접 label h1~h4도 3/4가 겹친다. 따라서 nominal sample `{metrics["overlap"]["nominal_total_samples"]}`개를 독립 표본처럼 해석하면 과대평가다.

## 8. CP112~CP119 test 반복 사용 감사

{_markdown_table(
        cp_rows,
        [("metrics file", "file"), ("test/candidate refs", "refs"), ("run estimate", "runs"), ("reused refs", "reused")],
    )}

요약:
- test/candidate 참조 총계: `{cp_usage["test_or_candidate_reference_count_total"]}`
- 신규 또는 재실행 model run 추정: `{cp_usage["new_or_replayed_model_run_count_estimate_total"]}`
- unique run_id 수: `{cp_usage["unique_run_id_count"]}`
- bias 등급: `{cp_usage["bias_level"]}`

판단:
- CP112~CP119는 smoke/제한 검증 목적이었기 때문에 test 확인 자체는 이해 가능하다.
- 하지만 이미 test가 후보 좁히기에 반복 노출됐다.
- 다음 1W 후보 저장 CP에서 test가 좋은 후보만 고르는 방식은 제품 후보 금지로 봐야 한다.

## 9. 과적합 방지 제안

필수 가드:
1. test set은 최종 후보 확인용으로만 쓴다.
2. 중간 실험과 후보 narrowing은 validation 중심으로 판단한다.
3. candidate registry에 `validation_stability`, `seed_stability`, `regime_stability`, `test_exposure_count`를 추가한다.
4. 후보가 test에서만 좋고 validation/regime 안정성이 없으면 제품 후보 금지다.
5. 모든 1W 보고서에 train/val/test target 분포와 regime별 metric을 필수로 붙인다.

추가 권장:
- anchored split 2개 이상 또는 rolling time split 추가
- sector holdout 또는 ticker group holdout 소규모 검증
- 2020 crash, 2022 bear, 2023~2024 bull, 2025~2026 recent를 별도 regime bucket으로 평가
- final test는 한 번만 열고, 그 전 후보 선택은 val과 rolling validation으로 제한

## 10. 다음 CP 권장

1. 1W 후보 재현 CP는 validation stability 중심으로 설계한다.
2. CP119 추천 band 후보는 seed 2~3개에서 val coverage, interval, downside_width_ic 안정성을 먼저 본다.
3. line 후보도 test IC만 보지 말고 val IC, tail recall, fee-adjusted proxy가 같은 방향인지 확인한다.
4. 제품 저장 전 `candidate_registry`에 test 노출 횟수와 split stability를 기록한다.

## 11. 금지 작업 확인

| 금지 항목 | 발생 |
|---|---|
| 모델 학습 | false |
| DB write | false |
| inference 저장 | false |
| Supabase 대량 read | false |
| 프론트 수정 | false |
| EODHD 호출 | false |

## 12. 읽기 전용 근거

- local price: `{PRICE_PATH}`
- local indicators: `{INDICATOR_PATH}`
- stock info: `{STOCK_INFO_PATH}`
- split 근거: `ai/splits.py`, `ai/preprocessing.py`
- CP112~CP119 metrics: `docs/cp112*`, `docs/cp113*`, `docs/cp114*`, `docs/cp118*`, `docs/cp119*`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    if "torch" in sys.modules:
        raise RuntimeError("CP123 감사 스크립트는 torch import 없이 실행되어야 합니다.")
    price, indicators, stock_info = _load_frames()
    coverage = _coverage(price, indicators, stock_info)
    samples, split_specs = _build_samples(price, indicators, stock_info)
    metrics = {
        "cp": "CP123-DG",
        "generated_at": _now_iso(),
        "inputs": {
            "price_path": str(PRICE_PATH),
            "indicator_path": str(INDICATOR_PATH),
            "stock_info_path": str(STOCK_INFO_PATH),
            "timeframe": TIMEFRAME,
            "provider": PROVIDER,
            "source": SOURCE,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "h_max": H_MAX,
            "min_fold_samples": MIN_FOLD_SAMPLES,
            "split_ratio": SPLIT_RATIO,
        },
        "coverage": coverage,
        "split_specs_summary": {
            "ticker_count": int(len(split_specs)),
            "example": next(iter(split_specs.values())) if split_specs else None,
        },
        "split_ranges": _split_ranges(samples),
        "sample_distribution": _sample_distribution(samples),
        "target_distribution": _target_summary(samples),
        "regime_distribution": _regime_summary(samples),
        "historical_regime_mapping": _historical_regime_mapping(samples),
        "overlap": _overlap_summary(samples),
        "cp112_cp119_test_usage": _cp_test_usage(),
        "forbidden_actions_observed": {
            "model_training": False,
            "db_write": False,
            "inference_save": False,
            "supabase_bulk_read": False,
            "frontend_modified": False,
            "eodhd_call": False,
            "torch_imported": "torch" in sys.modules,
        },
    }
    metrics["decision"] = _decision(
        metrics["coverage"],
        metrics["target_distribution"],
        metrics["regime_distribution"],
        metrics["cp112_cp119_test_usage"],
    )
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(metrics)
    print(json.dumps({"status": metrics["decision"]["status"], "metrics": str(METRICS_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
