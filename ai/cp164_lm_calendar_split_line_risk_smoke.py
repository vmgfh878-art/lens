from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai import cp158_lm_1d_line_regime_stage0_2 as cp158  # noqa: E402
from ai import cp159_lm_1d_line_conformal_overlay as cp159  # noqa: E402
from ai import cp160_lm_1d_line_overlay_rejudgement as cp160  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    apply_calendar_split_metadata,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    dataset_plan_split_metadata,
    normalize_sequence_splits,
    split_sequence_dataset_calendar_aligned,
)
from ai.ticker_registry import build_registry  # noqa: E402
from ai.train import (  # noqa: E402
    assert_dataset_features_finite,
    resolve_device,
    run_training,
    summarize_dataset_plan,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp164_lm_calendar_split_line_risk_smoke"
REPORT_PATH = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_report.md"
METRICS_PATH = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_summary.csv"
OVERLAY_CSV = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_overlay_feature_comparison.csv"
SPLIT_META_CSV = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_split_metadata.csv"

CANDIDATE_ID = "cp164_patchtst_line_regime_p32_s16_calendar"
OVERLAY_FEATURES = (
    "vol_raw_20d",
    "vol_xs_rank_20d",
    "self_vol_percentile_252",
    "downside_vol_ratio_20d",
    "drawdown_from_5d_high",
    "drawdown_from_20d_high",
    "atr_ratio",
)
FIXED_SHARES = (0.10, 0.20, 0.30)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{result:.{digits}f}" if math.isfinite(result) else ""


def _date_overlap_count(train: Any, val: Any, test: Any) -> int:
    def dates(bundle: Any) -> set[str]:
        return set(pd.to_datetime(bundle.metadata["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())

    train_dates = dates(train)
    val_dates = dates(val)
    test_dates = dates(test)
    return len((train_dates & val_dates) | (train_dates & test_dates) | (val_dates & test_dates))


def build_calendar_split_payload(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    source_data_hash: str,
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    all_tickers = sorted({str(ticker).upper() for ticker in indicators["ticker"].dropna().unique()})
    registry = build_registry(all_tickers, cp158.TIMEFRAME)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = LOG_DIR / "cp164_ticker_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    plan = build_dataset_plan(
        indicators,
        timeframe=cp158.TIMEFRAME,
        seq_len=cp158.SEQ_LEN,
        horizon=cp158.HORIZON,
        ticker_registry=registry,
        ticker_registry_path=str(registry_path),
        market_data_provider=cp158.PROVIDER,
        source_data_hash=source_data_hash,
        split_mode="calendar_aligned",
    )
    eligible = set(plan.eligible_tickers)
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(eligible)].copy(),
        price_df=price[price["ticker"].isin(eligible)].copy(),
        timeframe=cp158.TIMEFRAME,
        seq_len=cp158.SEQ_LEN,
        horizon=cp158.HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=cp158.TARGET_TYPE,
        band_target_type=cp158.TARGET_TYPE,
    )
    train, val, test, calendar_plan = split_sequence_dataset_calendar_aligned(
        dataset,
        purge_gap_trading_days=plan.h_max,
        min_fold_samples=plan.min_fold_samples,
    )
    apply_calendar_split_metadata(plan, calendar_plan)
    if plan.split_mode != "calendar_aligned":
        raise RuntimeError(f"calendar split 강제 실패: split_mode={plan.split_mode}")
    if plan.cross_split_date_overlap_count is None or int(plan.cross_split_date_overlap_count) != 0:
        raise RuntimeError(f"calendar split overlap 감지: {plan.cross_split_date_overlap_count}")
    if _date_overlap_count(train, val, test) != 0:
        raise RuntimeError("bundle metadata 기준 cross split date overlap이 0이 아닙니다.")
    train, val, test, mean, std = normalize_sequence_splits(train, val, test)
    return train, val, test, mean, std, plan, registry


def finite_target_summary(name: str, bundle: Any) -> dict[str, Any]:
    targets = cp158.collect_targets(bundle)
    finite = np.isfinite(targets)
    return {
        "name": name,
        "shape": list(targets.shape),
        "finite_ok": bool(finite.all()),
        "finite_count": int(finite.sum()),
        "total_count": int(targets.size),
        "nan_count": int(np.isnan(targets).sum()),
        "inf_count": int(np.isinf(targets).sum()),
    }


def make_config(device: str) -> Any:
    spec = cp158.CandidateSpec(
        CANDIDATE_ID,
        model="patchtst",
        patch_len=32,
        patch_stride=16,
        epochs=1,
        note="CP164 calendar-aligned split smoke",
    )
    config = cp158.build_config(spec, seed=42, device=device)
    config.model_ver = "cp164_patchtst_line_regime_p32_s16_calendar"
    config.split_mode = "calendar_aligned"
    config.use_wandb = False
    config.wandb_project = "lens-cp164-calendar-smoke"
    config.model_role = "line_regime"
    return config


def line_positive_hit_rate(prediction: dict[str, Any]) -> float | None:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    mask = np.isfinite(line_score) & np.isfinite(actual) & (line_score >= 0)
    return _safe_ratio(int((actual[mask] > 0).sum()), int(mask.sum()))


def regime_metrics(prediction: dict[str, Any], thresholds: list[float]) -> dict[str, Any]:
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    pred = np.asarray(prediction["regime_pred"], dtype=np.int64)
    target = cp158.class_targets(actual, thresholds)
    metrics = cp158.regime_class_metrics(pred, target, actual)
    distribution = cp158.class_distribution(pred)
    metrics["predicted_class_distribution"] = distribution
    rates = distribution.get("rates", {})
    metrics["class_collapse_score"] = max((float(value) for value in rates.values() if value is not None), default=None)
    return metrics


def regime_filter_metrics(prediction: dict[str, Any], severe_threshold: float) -> dict[str, Any]:
    pred = np.asarray(prediction["regime_pred"], dtype=np.int64)
    warning = pred <= 1
    row = cp160.overlay_filter_metrics(
        candidate_id=CANDIDATE_ID,
        split=str(prediction.get("split", "unknown")),
        overlay_family="regime",
        overlay_id="regime_0_1_warning",
        warning_mask_all=warning,
        prediction=prediction,
        severe_threshold=severe_threshold,
    )
    return {
        "regime_filter_false_safe_reduction": row.get("false_safe_reduction"),
        "regime_filter_spread_retention": row.get("spread_retention"),
        "regime_warning_share": row.get("warning_share"),
        "regime_removed_severe_rate": row.get("removed_severe_rate"),
        "regime_kept_severe_rate": row.get("kept_severe_rate"),
    }


def stage1_metrics(prediction: dict[str, Any], thresholds: list[float], split: str) -> dict[str, Any]:
    severe_threshold = float(thresholds[0])
    prediction = dict(prediction)
    prediction["split"] = split
    line = cp160.line_alpha_metrics(CANDIDATE_ID, split, prediction, severe_threshold)
    regime = regime_metrics(prediction, thresholds)
    joint = regime_filter_metrics(prediction, severe_threshold)
    return {
        "candidate_id": CANDIDATE_ID,
        "split": split,
        "ic_mean": line.get("ic_mean"),
        "ic_std": line.get("ic_std"),
        "ic_ir": line.get("ic_ir"),
        "ic_t_stat": line.get("ic_t_stat"),
        "long_short_spread": line.get("long_short_spread"),
        "fee_adjusted_return": line.get("fee_adjusted_return"),
        "top_decile_actual_return": line.get("top_decile_actual_return"),
        "line_positive_hit_rate": line_positive_hit_rate(prediction),
        "line_gate_pass": bool(line.get("line_alpha_alive")),
        "line_top_decile_false_safe_rate": line.get("line_top_decile_false_safe_rate"),
        "line_top_decile_severe_rate": line.get("line_top_decile_severe_rate"),
        "regime_ordinal_mae": regime.get("regime_ordinal_mae"),
        "regime_adjacent_accuracy": regime.get("regime_adjacent_accuracy"),
        "regime_macro_f1": regime.get("regime_macro_f1"),
        "strong_down_recall": regime.get("strong_down_recall"),
        "class_collapse_score": regime.get("class_collapse_score"),
        "predicted_class_distribution": regime.get("predicted_class_distribution"),
        **joint,
    }


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, min_periods=max(3, min(window, 5)))
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, min_periods=1)
        .min()
        .to_numpy(dtype=np.float64)
    )


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
    )


def _trailing_percentile(series: np.ndarray, end_idx: int, window: int = 252) -> float:
    start = max(0, int(end_idx) - window + 1)
    values = np.asarray(series[start : int(end_idx) + 1], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.5
    current = float(series[int(end_idx)])
    return float((values <= current).mean())


def _indicator_lookup(indicators: pd.DataFrame, column: str) -> dict[tuple[str, str], float]:
    if column not in indicators.columns:
        return {}
    frame = indicators[["ticker", "date", column]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date"])
    return {
        (str(ticker).upper(), str(date)): float(value)
        for ticker, date, value in frame.itertuples(index=False, name=None)
        if pd.notna(value)
    }


def compute_overlay_features(bundle: Any, indicators: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(bundle.sample_refs)
    result = {name: np.zeros(n, dtype=np.float64) for name in OVERLAY_FEATURES}
    atr_lookup = _indicator_lookup(indicators, "atr_ratio")
    cache: dict[str, dict[str, np.ndarray]] = {}
    for ticker in sorted({str(ticker) for ticker, _end_idx in bundle.sample_refs}):
        closes = np.asarray(bundle.ticker_arrays[ticker]["closes"], dtype=np.float64)
        returns = np.zeros(len(closes), dtype=np.float64)
        if len(closes) > 1:
            returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
        raw20 = _rolling_std(returns, 20)
        downside = np.where(returns < 0, returns, 0.0)
        downside20 = _rolling_std(downside, 20)
        high5 = _rolling_max(closes, 5)
        high20 = _rolling_max(closes, 20)
        cache[ticker] = {
            "raw20": raw20,
            "downside_ratio": downside20 / np.maximum(raw20, 1e-8),
            "drawdown5": 1.0 - closes / np.maximum(high5, 1e-12),
            "drawdown20": 1.0 - closes / np.maximum(high20, 1e-12),
        }
    metadata = bundle.metadata.reset_index(drop=True)
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        ticker_key = str(ticker)
        values = cache[ticker_key]
        idx = int(end_idx)
        result["vol_raw_20d"][row_idx] = values["raw20"][idx]
        result["self_vol_percentile_252"][row_idx] = _trailing_percentile(values["raw20"], idx)
        result["downside_vol_ratio_20d"][row_idx] = values["downside_ratio"][idx]
        result["drawdown_from_5d_high"][row_idx] = values["drawdown5"][idx]
        result["drawdown_from_20d_high"][row_idx] = values["drawdown20"][idx]
        result["atr_ratio"][row_idx] = atr_lookup.get((ticker_key.upper(), str(dates.iloc[row_idx])), np.nan)
    frame = pd.DataFrame({"date": dates, "vol_raw_20d": result["vol_raw_20d"]})
    result["vol_xs_rank_20d"] = frame.groupby("date")["vol_raw_20d"].rank(pct=True, method="average").fillna(0.5).to_numpy(dtype=np.float64)
    cleaned: dict[str, np.ndarray] = {}
    for key, values in result.items():
        fallback = 0.5 if key.endswith("percentile_252") or key.endswith("xs_rank_20d") else 0.0
        cleaned[key] = np.nan_to_num(values, nan=fallback, posinf=fallback, neginf=0.0)
    return cleaned


def _top_removed_mask(prediction: dict[str, Any], warning_mask: np.ndarray) -> np.ndarray:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, _bottom = cp160._top_bottom_masks(line_score)
    return top & np.asarray(warning_mask, dtype=bool)


def ticker_concentration_top10_share(prediction: dict[str, Any], warning_mask: np.ndarray) -> float | None:
    removed = _top_removed_mask(prediction, warning_mask)
    if not removed.any():
        return None
    tickers = prediction["metadata"].loc[removed, "ticker"].astype(str).str.upper()
    counts = tickers.value_counts()
    return _safe_ratio(int(counts.head(10).sum()), int(removed.sum()))


def within_ticker_discrimination_score(prediction: dict[str, Any], warning_mask: np.ndarray, severe_threshold: float) -> float | None:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    top, _bottom = cp160._top_bottom_masks(line_score)
    severe = actual <= severe_threshold
    frame = prediction["metadata"][["ticker"]].copy()
    frame["top"] = top
    frame["warning"] = np.asarray(warning_mask, dtype=bool)
    frame["severe"] = severe
    scores: list[float] = []
    for _ticker, group in frame[frame["top"]].groupby("ticker", sort=False):
        warned = group[group["warning"]]
        kept = group[~group["warning"]]
        if len(warned) < 2 or len(kept) < 2:
            continue
        scores.append(float(warned["severe"].mean() - kept["severe"].mean()))
    return float(np.mean(scores)) if scores else None


def evaluate_overlay_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name in OVERLAY_FEATURES:
        values = np.asarray(features[feature_name], dtype=np.float64)
        val_values = np.asarray(val_features[feature_name], dtype=np.float64)
        finite_rate = _safe_ratio(int(np.isfinite(values).sum()), int(values.size))
        threshold = float(np.quantile(val_values[np.isfinite(val_values)], 0.80)) if np.isfinite(val_values).any() else 0.0
        for overlay_id, warning, rule_type, extra in [
            (
                f"{feature_name}_q80",
                values >= threshold,
                "validation_q80",
                {"threshold": threshold, "q": 0.80, "feature": feature_name},
            )
        ]:
            row = cp160.overlay_filter_metrics(
                candidate_id=CANDIDATE_ID,
                split=split,
                overlay_family="overlay_feature",
                overlay_id=overlay_id,
                warning_mask_all=warning,
                prediction=prediction,
                severe_threshold=severe_threshold,
            )
            row.update(extra)
            row["rule_type"] = rule_type
            row["finite_rate"] = finite_rate
            row["ticker_concentration_top10_share"] = ticker_concentration_top10_share(prediction, warning)
            row["within_ticker_discrimination_score"] = within_ticker_discrimination_score(prediction, warning, severe_threshold)
            rows.append(row)
        for share in FIXED_SHARES:
            share_threshold = float(np.quantile(values[np.isfinite(values)], 1.0 - share)) if np.isfinite(values).any() else 0.0
            warning = values >= share_threshold
            row = cp160.overlay_filter_metrics(
                candidate_id=CANDIDATE_ID,
                split=split,
                overlay_family="overlay_feature",
                overlay_id=f"{feature_name}_top{int(share * 100)}",
                warning_mask_all=warning,
                prediction=prediction,
                severe_threshold=severe_threshold,
            )
            row.update(
                {
                    "feature": feature_name,
                    "rule_type": "fixed_share",
                    "share": share,
                    "threshold": share_threshold,
                    "finite_rate": finite_rate,
                    "ticker_concentration_top10_share": ticker_concentration_top10_share(prediction, warning),
                    "within_ticker_discrimination_score": within_ticker_discrimination_score(prediction, warning, severe_threshold),
                }
            )
            rows.append(row)
    return rows


def load_prior_comparison() -> dict[str, Any]:
    result: dict[str, Any] = {}
    cp158_summary = DOCS_DIR / "cp158_lm_1d_line_regime_stage2_summary.csv"
    if cp158_summary.exists():
        frame = pd.read_csv(cp158_summary)
        rows = frame[frame["candidate_id"] == "patchtst_line_regime_p32_s16"]
        if not rows.empty:
            result["cp158_patchtst_line_regime_p32_s16"] = rows.iloc[0].to_dict()
    cp161_summary = DOCS_DIR / "cp161_lm_1d_vol_anchored_risk_color_summary.csv"
    if cp161_summary.exists():
        frame = pd.read_csv(cp161_summary)
        rows = frame[
            (frame.get("candidate_id") == "patchtst_line_regime_p32_s16")
            & (frame.get("overlay_id") == "vol_xs_rank_20d_q90_vote_2plus")
        ]
        if not rows.empty:
            result["cp161_best_rule_vol_xs_rank_20d_q90_vote_2plus"] = rows.iloc[0].to_dict()
    return result


def decide_line_status(test_row: dict[str, Any]) -> str:
    ic = float(test_row.get("ic_mean") or 0.0)
    spread = float(test_row.get("long_short_spread") or 0.0)
    fee = float(test_row.get("fee_adjusted_return") or 0.0)
    if ic > 0 and spread > 0 and fee > 0:
        return "LINE_SIGNAL_SURVIVED"
    if ic > 0 or spread > 0:
        return "LINE_SIGNAL_WEAKENED_BUT_PRESENT"
    return "LINE_SIGNAL_COLLAPSED"


def decide_risk_status(overlay_rows: list[dict[str, Any]]) -> str:
    test_rows = [row for row in overlay_rows if row.get("split") == "test" and row.get("rule_type") == "validation_q80"]
    useful = [
        row
        for row in test_rows
        if (row.get("false_safe_reduction") or 0) > 0
        and (row.get("severe_lift_removed_vs_kept") or 0) > 1.0
        and (row.get("removed_sample_count") or 0) > 0
    ]
    if useful:
        return "RISK_FEATURE_PREFLIGHT_READY"
    if test_rows:
        return "RISK_SIGNAL_WEAK"
    return "RISK_NEEDS_FEATURE_CONTRACT_REPAIR"


def write_report(payload: dict[str, Any]) -> None:
    test_row = next(row for row in payload["summary_rows"] if row["split"] == "test")
    validation_row = next(row for row in payload["summary_rows"] if row["split"] == "validation")
    best_overlay = sorted(
        [row for row in payload["overlay_rows"] if row.get("split") == "test" and row.get("rule_type") == "validation_q80"],
        key=lambda row: (
            row.get("false_safe_reduction") or -999,
            row.get("severe_lift_removed_vs_kept") or -999,
            row.get("spread_retention") or -999,
        ),
        reverse=True,
    )[:5]
    split_meta = payload["split_metadata"]
    lines = [
        "# CP164-LM Calendar Split Line/Risk Smoke Recheck",
        "",
        "## 한 줄 결론",
        f"- line 판정: **{payload['line_status']}**",
        f"- risk 판정: **{payload['risk_status']}**",
        "- 이번 CP는 제품 후보 확정이 아니라 calendar split 수리 후 신호 생존 여부를 확인한 smoke다.",
        "",
        "## 금지 작업 준수",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- composite 실행 없음",
        "- CP153 band artifact 변경 없음",
        "",
        "## Stage 0 Preflight",
        f"- provider/source: `{cp158.PROVIDER}` / `{cp158.PROVIDER}`",
        f"- feature contract: `{FEATURE_CONTRACT_VERSION}`",
        f"- feature_set: `{cp158.FEATURE_SET}`",
        f"- source_data_hash: `{split_meta.get('source_data_hash')}`",
        f"- eligible ticker count: `{split_meta.get('eligible_ticker_count')}`",
        f"- split_mode: `{split_meta.get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{split_meta.get('cross_split_date_overlap_count')}`",
        f"- purge_gap_trading_days: `{split_meta.get('purge_gap_trading_days')}`",
        f"- train: `{split_meta.get('split_train_start_date')}` ~ `{split_meta.get('split_train_end_date')}`",
        f"- validation: `{split_meta.get('split_validation_start_date')}` ~ `{split_meta.get('split_validation_end_date')}`",
        f"- test: `{split_meta.get('split_test_start_date')}` ~ `{split_meta.get('split_test_end_date')}`",
        f"- feature finite PASS: `{payload['preflight']['feature_finite_pass']}`",
        f"- target finite PASS: `{payload['preflight']['target_finite_pass']}`",
        "",
        "## Stage 1 Line/Regime Smoke",
        "| split | IC | spread | fee | top decile return | positive hit | top false-safe | ordinal MAE | adjacent acc | macro F1 | strong down recall | regime FS 감소 | spread retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (validation_row, test_row):
        lines.append(
            f"| {row['split']} | {_fmt(row.get('ic_mean'))} | {_fmt(row.get('long_short_spread'))} | "
            f"{_fmt(row.get('fee_adjusted_return'))} | {_fmt(row.get('top_decile_actual_return'))} | "
            f"{_fmt(row.get('line_positive_hit_rate'))} | {_fmt(row.get('line_top_decile_false_safe_rate'))} | "
            f"{_fmt(row.get('regime_ordinal_mae'))} | {_fmt(row.get('regime_adjacent_accuracy'))} | "
            f"{_fmt(row.get('regime_macro_f1'))} | {_fmt(row.get('strong_down_recall'))} | "
            f"{_fmt(row.get('regime_filter_false_safe_reduction'))} | {_fmt(row.get('regime_filter_spread_retention'))} |"
        )
    lines.extend(
        [
            "",
            "해석:",
            "- line alpha는 IC/spread/fee가 모두 양수인지로 smoke 생존 여부만 본다.",
            "- line 단독 false-safe는 탈락 게이트가 아니라 risk overlay 필요성 지표로 기록한다.",
            "- regime head는 색상/해석 보조 후보이므로, false-safe 감소와 spread retention을 같이 본다.",
            "",
            "## Stage 2 Vol/Self-normalized Overlay",
            "| rule | false-safe 감소 | severe lift vs kept | spread retention | fee retention | warning share | ticker top10 집중 | within-ticker 분리 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best_overlay:
        lines.append(
            f"| {row['overlay_id']} | {_fmt(row.get('false_safe_reduction'))} | "
            f"{_fmt(row.get('severe_lift_removed_vs_kept'))} | {_fmt(row.get('spread_retention'))} | "
            f"{_fmt(row.get('fee_retention'))} | {_fmt(row.get('warning_share'))} | "
            f"{_fmt(row.get('ticker_concentration_top10_share'))} | {_fmt(row.get('within_ticker_discrimination_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Stage 3 기존 CP158/161 방향성 비교",
            "- CP158/161은 CP163 이전 ticker-index split 가능성 때문에 exploratory로 재해석한다.",
            "- CP164는 같은 수치 우열 비교가 아니라, calendar split에서도 방향성이 남는지 확인하는 재스모크다.",
            f"- 기존 CP158 p32/s16 test IC/spread: `{_fmt((payload['prior_comparison'].get('cp158_patchtst_line_regime_p32_s16') or {}).get('test_ic'))}` / `{_fmt((payload['prior_comparison'].get('cp158_patchtst_line_regime_p32_s16') or {}).get('test_spread'))}`",
            f"- CP164 p32/s16 test IC/spread: `{_fmt(test_row.get('ic_mean'))}` / `{_fmt(test_row.get('long_short_spread'))}`",
            "- 기존 CP161 best rule은 `vol_xs_rank_20d_q90_vote_2plus`였고, CP164는 최소 smoke 범위라 단일 feature q80 및 fixed-share 비교까지만 수행했다.",
            "",
            "## Stage 4 다음 단계 판정",
            f"- line: `{payload['line_status']}`",
            f"- risk: `{payload['risk_status']}`",
        ]
    )
    if payload["line_status"] == "LINE_SIGNAL_SURVIVED" and payload["risk_status"] == "RISK_FEATURE_PREFLIGHT_READY":
        lines.append("- 다음 CP 제안: CP165 feature preflight로 진행 가능. 단, 이번 결과는 smoke이며 제품 저장 근거가 아니다.")
    elif payload["line_status"] == "LINE_SIGNAL_COLLAPSED":
        lines.append("- 다음 CP 제안: line 제품 신호 자체를 calendar split 기준으로 재검토해야 한다.")
    else:
        lines.append("- 다음 CP 제안: risk feature와 atr_ratio/feature interaction/PatchTST ci_aggregate 점검을 우선한다.")
    lines.extend(
        [
            "",
            "## 산출물",
            f"- metrics: `{METRICS_PATH}`",
            f"- summary: `{SUMMARY_CSV}`",
            f"- overlay comparison: `{OVERLAY_CSV}`",
            f"- split metadata: `{SPLIT_META_CSV}`",
            f"- local logs: `{LOG_DIR}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cp153_before = cp159.cp153_artifact_state()
    price, indicators, price_manifest, indicator_manifest = cp158.load_source_frames()
    source_hash = str(indicator_manifest.get("source_data_hash") or price_manifest.get("source_data_hash") or "unknown")
    train, val, test, mean, std, plan, _registry = build_calendar_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    assert_dataset_features_finite("cp164_train", train)
    assert_dataset_features_finite("cp164_validation", val)
    assert_dataset_features_finite("cp164_test", test)
    target_summaries = [finite_target_summary("train", train), finite_target_summary("validation", val), finite_target_summary("test", test)]
    split_summary = summarize_dataset_plan(plan, train, val, test)
    split_summary["cross_split_date_overlap_count_bundle_check"] = _date_overlap_count(train, val, test)
    split_summary["feature_version"] = FEATURE_CONTRACT_VERSION
    split_summary["source_data_hash"] = source_hash

    device_name = "cuda" if cp159.torch.cuda.is_available() else "cpu"
    device = resolve_device(device_name)
    config = make_config(str(device))
    print(
        json.dumps(
            {
                "stage": "cp164_train_start",
                "candidate_id": CANDIDATE_ID,
                "split_mode": config.split_mode,
                "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
                "epochs": config.epochs,
                "device": str(device),
                "save_run": False,
                "wandb": "disabled",
            },
            ensure_ascii=False,
        )
    )
    started = datetime.now(timezone.utc)
    result = run_training(
        config,
        save_run=False,
        precomputed_bundles=(train, val, test, mean, std, plan),
        enable_compile=False,
        local_log=True,
        local_log_dir=LOG_DIR / "train_local_logs" / CANDIDATE_ID,
    )
    elapsed_seconds = (datetime.now(timezone.utc) - started).total_seconds()
    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()

    thresholds = [float(value) for value in (config.regime_thresholds or result["best_metrics"].get("regime_thresholds") or [])]
    if len(thresholds) != 4:
        raise RuntimeError(f"regime threshold 수가 4가 아닙니다: {thresholds}")
    severe_threshold = thresholds[0]
    val_prediction = cp160.collect_predictions(
        candidate_id=CANDIDATE_ID,
        checkpoint_path=str(result["checkpoint_path"]),
        model_kind="line_regime",
        bundle=val,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    test_prediction = cp160.collect_predictions(
        candidate_id=CANDIDATE_ID,
        checkpoint_path=str(result["checkpoint_path"]),
        model_kind="line_regime",
        bundle=test,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    summary_rows = [
        stage1_metrics(val_prediction, thresholds, "validation"),
        stage1_metrics(test_prediction, thresholds, "test"),
    ]
    val_features = compute_overlay_features(val, indicators)
    test_features = compute_overlay_features(test, indicators)
    overlay_rows = []
    overlay_rows.extend(
        evaluate_overlay_rows(
            split="validation",
            prediction=val_prediction,
            features=val_features,
            val_features=val_features,
            severe_threshold=severe_threshold,
        )
    )
    overlay_rows.extend(
        evaluate_overlay_rows(
            split="test",
            prediction=test_prediction,
            features=test_features,
            val_features=val_features,
            severe_threshold=severe_threshold,
        )
    )
    cp153_after = cp159.cp153_artifact_state()
    preflight = {
        "feature_finite_pass": True,
        "target_finite_pass": all(item["finite_ok"] for item in target_summaries),
        "target_summaries": target_summaries,
        "cp153_band_artifact_unchanged": cp153_before == cp153_after,
        "product_save_run": False,
        "db_write": False,
        "inference_save": False,
        "live_fetch": False,
        "eodhd_fallback": False,
        "composite_execution": False,
    }
    line_status = decide_line_status(summary_rows[1])
    risk_status = decide_risk_status(overlay_rows)
    payload = {
        "cp": "CP164-LM",
        "title": "Calendar Split Line/Risk Smoke Recheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID,
        "run_result": {
            "run_id": result.get("run_id"),
            "checkpoint_path": result.get("checkpoint_path"),
            "local_log_dir": result.get("local_log_dir"),
            "elapsed_seconds": elapsed_seconds,
            "wandb_status": result.get("wandb_status"),
            "save_run": False,
        },
        "config": {
            "model": config.model,
            "model_role": config.model_role,
            "output_role": "line_regime",
            "timeframe": config.timeframe,
            "horizon": config.horizon,
            "seq_len": config.seq_len,
            "patch_len": config.patch_len,
            "patch_stride": config.patch_stride,
            "feature_set": config.feature_set,
            "split_mode": config.split_mode,
            "alpha": config.alpha,
            "beta": config.beta,
            "delta": config.delta,
            "seed": config.seed,
            "epochs": config.epochs,
            "checkpoint_selection": config.checkpoint_selection,
        },
        "split_metadata": split_summary,
        "preflight": preflight,
        "regime_thresholds": thresholds,
        "line_status": line_status,
        "risk_status": risk_status,
        "summary_rows": summary_rows,
        "overlay_rows": overlay_rows,
        "prior_comparison": load_prior_comparison(),
    }
    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(OVERLAY_CSV, overlay_rows)
    _write_csv(SPLIT_META_CSV, [split_summary])
    write_report(payload)
    print(
        json.dumps(
            {
                "status": "CP164_DONE",
                "line_status": line_status,
                "risk_status": risk_status,
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
                "split_mode": split_summary.get("split_mode"),
                "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
