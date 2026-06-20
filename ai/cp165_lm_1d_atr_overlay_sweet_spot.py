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

from ai import cp159_lm_1d_line_conformal_overlay as cp159  # noqa: E402
from ai import cp160_lm_1d_line_overlay_rejudgement as cp160  # noqa: E402
from ai import cp164_lm_calendar_split_line_risk_smoke as cp164  # noqa: E402
from ai.train import resolve_device  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp165_lm_1d_atr_overlay_sweet_spot"
REPORT_PATH = DOCS_DIR / "cp165_lm_1d_atr_overlay_sweet_spot_report.md"
METRICS_PATH = DOCS_DIR / "cp165_lm_1d_atr_overlay_sweet_spot_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp165_lm_1d_atr_overlay_sweet_spot_summary.csv"
REMOVED_DIAG_CSV = DOCS_DIR / "cp165_lm_1d_atr_overlay_removed_group_diagnostic.csv"
RULE_SWEEP_CSV = DOCS_DIR / "cp165_lm_1d_atr_self_rule_sweep.csv"
PARETO_CSV = DOCS_DIR / "cp165_lm_1d_atr_self_pareto_frontier.csv"
AUX_CSV = DOCS_DIR / "cp165_lm_1d_atr_overlay_auxiliary_feature_comparison.csv"

CP164_METRICS_PATH = DOCS_DIR / "cp164_lm_calendar_split_line_risk_smoke_metrics.json"
CANDIDATE_ID = "cp165_cp164_patchtst_line_regime_p32_s16_calendar"
THRESHOLD_QS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
AUX_FEATURES = (
    "vol_xs_rank_20d",
    "downside_vol_ratio_20d",
    "drawdown_from_5d_high",
    "drawdown_from_20d_high",
)


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
    path.write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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


def _safe_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{result:.{digits}f}" if math.isfinite(result) else ""


def _top_mask(prediction: dict[str, Any]) -> np.ndarray:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, _bottom = cp160._top_bottom_masks(line_score)
    return top


def _top_tickers(
    prediction: dict[str, Any], mask: np.ndarray, limit: int = 10
) -> list[dict[str, Any]]:
    frame = prediction["metadata"].loc[mask, ["ticker"]].copy()
    if frame.empty:
        return []
    counts = frame["ticker"].astype(str).str.upper().value_counts().head(limit)
    total = int(mask.sum())
    return [
        {"ticker": str(ticker), "count": int(count), "share": _safe_ratio(int(count), total)}
        for ticker, count in counts.items()
    ]


def _ticker_concentration_top10(prediction: dict[str, Any], mask: np.ndarray) -> float | None:
    tickers = prediction["metadata"].loc[mask, "ticker"].astype(str).str.upper()
    if tickers.empty:
        return None
    counts = tickers.value_counts()
    return _safe_ratio(int(counts.head(10).sum()), int(mask.sum()))


def _within_ticker_discrimination(
    prediction: dict[str, Any], warning_mask: np.ndarray, severe_threshold: float
) -> float | None:
    top = _top_mask(prediction)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    severe = actual <= severe_threshold
    frame = prediction["metadata"][["ticker"]].copy()
    frame["top"] = top
    frame["warning"] = np.asarray(warning_mask, dtype=bool)
    frame["severe"] = severe
    values: list[float] = []
    for _ticker, group in frame[frame["top"]].groupby("ticker", sort=False):
        removed = group[group["warning"]]
        kept = group[~group["warning"]]
        if len(removed) < 2 or len(kept) < 2:
            continue
        values.append(float(removed["severe"].mean() - kept["severe"].mean()))
    return float(np.mean(values)) if values else None


def load_cp164_reference() -> dict[str, Any]:
    if not CP164_METRICS_PATH.exists():
        raise FileNotFoundError(f"CP164 metrics가 없습니다: {CP164_METRICS_PATH}")
    payload = _read_json(CP164_METRICS_PATH)
    split_meta = payload.get("split_metadata") or {}
    if split_meta.get("split_mode") != "calendar_aligned":
        raise RuntimeError(
            f"CP164 split_mode가 calendar_aligned가 아닙니다: {split_meta.get('split_mode')}"
        )
    if int(split_meta.get("cross_split_date_overlap_count") or 0) != 0:
        raise RuntimeError(
            f"CP164 cross_split_date_overlap_count가 0이 아닙니다: {split_meta.get('cross_split_date_overlap_count')}"
        )
    checkpoint_path = PROJECT_ROOT / str(payload["run_result"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CP164 checkpoint가 없습니다: {checkpoint_path}")
    return payload


def build_predictions(
    cp164_payload: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
    list[float],
]:
    price, indicators, price_manifest, indicator_manifest = cp164.cp158.load_source_frames()
    source_hash = str(
        indicator_manifest.get("source_data_hash")
        or price_manifest.get("source_data_hash")
        or "unknown"
    )
    train, val, test, mean, std, plan, _registry = cp164.build_calendar_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    split_summary = cp164.summarize_dataset_plan(plan, train, val, test)
    split_summary["source_data_hash"] = source_hash
    split_summary["cross_split_date_overlap_count_bundle_check"] = cp164._date_overlap_count(
        train, val, test
    )
    overlap = split_summary.get("cross_split_date_overlap_count")
    if (
        split_summary.get("split_mode") != "calendar_aligned"
        or overlap is None
        or int(overlap) != 0
    ):
        raise RuntimeError(f"calendar split preflight 실패: {split_summary}")
    config = cp164.make_config("cuda" if cp159.torch.cuda.is_available() else "cpu")
    config.regime_thresholds = [
        float(value) for value in cp164_payload.get("regime_thresholds", [])
    ]
    device = resolve_device("cuda" if cp159.torch.cuda.is_available() else "cpu")
    checkpoint_path = str(PROJECT_ROOT / str(cp164_payload["run_result"]["checkpoint_path"]))
    val_prediction = cp160.collect_predictions(
        candidate_id=CANDIDATE_ID,
        checkpoint_path=checkpoint_path,
        model_kind="line_regime",
        bundle=val,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    test_prediction = cp160.collect_predictions(
        candidate_id=CANDIDATE_ID,
        checkpoint_path=checkpoint_path,
        model_kind="line_regime",
        bundle=test,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    val_features = cp164.compute_overlay_features(val, indicators)
    test_features = cp164.compute_overlay_features(test, indicators)
    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()
    return (
        val_prediction,
        test_prediction,
        val_features,
        test_features,
        split_summary,
        config.regime_thresholds,
    )


def removed_group_diagnostic(
    *,
    split: str,
    prediction: dict[str, Any],
    feature_values: np.ndarray,
    validation_feature_values: np.ndarray,
    severe_threshold: float,
    q: float,
) -> dict[str, Any]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    top = _top_mask(prediction)
    severe = actual <= severe_threshold
    finite_val = validation_feature_values[np.isfinite(validation_feature_values)]
    threshold = float(np.quantile(finite_val, q)) if len(finite_val) else 0.0
    warning = np.asarray(feature_values, dtype=np.float64) >= threshold
    removed = top & warning
    kept = top & ~warning
    top_count = int(top.sum())
    top_tickers = _top_tickers(prediction, removed, limit=10)
    return {
        "split": split,
        "rule": f"atr_ratio_q{int(q * 100)}",
        "q": q,
        "threshold": threshold,
        "top_decile_sample_count": top_count,
        "removed_sample_count": int(removed.sum()),
        "removed_sample_share": _safe_ratio(int(removed.sum()), top_count),
        "removed_actual_mean_return": _safe_mean(actual[removed]),
        "removed_actual_median_return": _safe_median(actual[removed]),
        "removed_positive_return_rate": _safe_ratio(
            int((actual[removed] > 0).sum()), int(removed.sum())
        ),
        "removed_severe_rate": _safe_ratio(int((removed & severe).sum()), int(removed.sum())),
        "removed_false_safe_rate": _safe_ratio(int((removed & severe).sum()), int(removed.sum())),
        "kept_actual_mean_return": _safe_mean(actual[kept]),
        "kept_actual_median_return": _safe_median(actual[kept]),
        "kept_positive_return_rate": _safe_ratio(int((actual[kept] > 0).sum()), int(kept.sum())),
        "kept_severe_rate": _safe_ratio(int((kept & severe).sum()), int(kept.sum())),
        "kept_false_safe_rate": _safe_ratio(int((kept & severe).sum()), int(kept.sum())),
        "removed_line_score_mean": _safe_mean(line_score[removed]),
        "removed_line_score_q50": _safe_quantile(line_score[removed], 0.50),
        "removed_line_score_q90": _safe_quantile(line_score[removed], 0.90),
        "kept_line_score_mean": _safe_mean(line_score[kept]),
        "kept_line_score_q50": _safe_quantile(line_score[kept], 0.50),
        "kept_line_score_q90": _safe_quantile(line_score[kept], 0.90),
        "removed_top_ticker_share": top_tickers[0]["share"] if top_tickers else None,
        "removed_top10_ticker_share": _ticker_concentration_top10(prediction, removed),
        "removed_top_tickers": json.dumps(top_tickers, ensure_ascii=False),
    }


def _row_with_extra(
    *,
    split: str,
    overlay_id: str,
    family: str,
    warning_mask: np.ndarray,
    prediction: dict[str, Any],
    severe_threshold: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = cp160.overlay_filter_metrics(
        candidate_id=CANDIDATE_ID,
        split=split,
        overlay_family=family,
        overlay_id=overlay_id,
        warning_mask_all=warning_mask,
        prediction=prediction,
        severe_threshold=severe_threshold,
    )
    removed = _top_mask(prediction) & warning_mask
    row.update(extra)
    row["removed_positive_return_rate"] = _safe_ratio(
        int((np.asarray(prediction["actual"], dtype=np.float64)[removed] > 0).sum()),
        int(removed.sum()),
    )
    row["ticker_concentration_top10_share"] = _ticker_concentration_top10(prediction, removed)
    row["within_ticker_discrimination_score"] = _within_ticker_discrimination(
        prediction, warning_mask, severe_threshold
    )
    return row


def classify_rule(row: dict[str, Any]) -> str:
    warning_share = row.get("warning_share") or 0.0
    spread_retention = row.get("spread_retention")
    fee_retention = row.get("fee_retention")
    fs_reduction = row.get("false_safe_reduction") or 0.0
    severe_lift = row.get("severe_lift_removed_vs_kept") or 0.0
    if warning_share < 0.03:
        return "too_narrow_unstable"
    if warning_share > 0.35:
        return "too_broad_alpha_damage"
    retention_ok = (
        spread_retention is not None
        and fee_retention is not None
        and spread_retention >= 0.80
        and fee_retention >= 0.80
    )
    if retention_ok and fs_reduction > 0 and severe_lift > 1.0:
        return "sweet_spot_candidate"
    if fs_reduction > 0 and severe_lift > 1.0 and not retention_ok:
        return "strong_warning_high_cost"
    return "weak_signal"


def build_rule_sweep(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    atr = np.asarray(features["atr_ratio"], dtype=np.float64)
    self_vol = np.asarray(features["self_vol_percentile_252"], dtype=np.float64)
    val_atr = np.asarray(validation_features["atr_ratio"], dtype=np.float64)
    val_self = np.asarray(validation_features["self_vol_percentile_252"], dtype=np.float64)
    atr_thresholds = {
        q: float(np.quantile(val_atr[np.isfinite(val_atr)], q))
        if np.isfinite(val_atr).any()
        else 0.0
        for q in THRESHOLD_QS
    }
    self_thresholds = {
        q: float(np.quantile(val_self[np.isfinite(val_self)], q))
        if np.isfinite(val_self).any()
        else 0.0
        for q in THRESHOLD_QS
    }
    rows: list[dict[str, Any]] = []
    for q, threshold in atr_thresholds.items():
        warning = atr >= threshold
        row = _row_with_extra(
            split=split,
            overlay_id=f"atr_only_q{int(q * 100)}",
            family="atr_self_sweep",
            warning_mask=warning,
            prediction=prediction,
            severe_threshold=severe_threshold,
            extra={
                "rule_family": "atr_only",
                "atr_q": q,
                "self_q": None,
                "atr_threshold": threshold,
                "self_threshold": None,
            },
        )
        row["classification"] = classify_rule(row)
        rows.append(row)
    for q, threshold in self_thresholds.items():
        warning = self_vol >= threshold
        row = _row_with_extra(
            split=split,
            overlay_id=f"self_only_q{int(q * 100)}",
            family="atr_self_sweep",
            warning_mask=warning,
            prediction=prediction,
            severe_threshold=severe_threshold,
            extra={
                "rule_family": "self_only",
                "atr_q": None,
                "self_q": q,
                "atr_threshold": None,
                "self_threshold": threshold,
            },
        )
        row["classification"] = classify_rule(row)
        rows.append(row)
    for atr_q, atr_threshold in atr_thresholds.items():
        atr_warning = atr >= atr_threshold
        for self_q, self_threshold in self_thresholds.items():
            self_warning = self_vol >= self_threshold
            masks = {
                "atr_and_self": atr_warning & self_warning,
                "atr_or_self": atr_warning | self_warning,
                "atr_and_not_self": atr_warning & ~self_warning,
                "self_and_not_atr": self_warning & ~atr_warning,
            }
            for rule_family, warning in masks.items():
                row = _row_with_extra(
                    split=split,
                    overlay_id=f"{rule_family}_atr{int(atr_q * 100)}_self{int(self_q * 100)}",
                    family="atr_self_sweep",
                    warning_mask=warning,
                    prediction=prediction,
                    severe_threshold=severe_threshold,
                    extra={
                        "rule_family": rule_family,
                        "atr_q": atr_q,
                        "self_q": self_q,
                        "atr_threshold": atr_threshold,
                        "self_threshold": self_threshold,
                    },
                )
                row["classification"] = classify_rule(row)
                rows.append(row)
    return rows


def pareto_frontier(
    validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in validation_rows
        if (row.get("spread_retention") is not None and row.get("spread_retention") >= 0.80)
        and (row.get("fee_retention") is not None and row.get("fee_retention") >= 0.80)
        and 0.03 <= (row.get("warning_share") or 0.0) <= 0.35
        and (row.get("false_safe_reduction") or 0.0) > 0
    ]
    if not candidates:
        candidates = [
            row
            for row in validation_rows
            if 0.03 <= (row.get("warning_share") or 0.0) <= 0.35
            and (row.get("false_safe_reduction") or 0.0) > 0
        ]
    non_dominated: list[dict[str, Any]] = []
    for row in candidates:
        row_obj = (
            row.get("false_safe_reduction") or -999,
            row.get("severe_lift_removed_vs_kept") or -999,
            -(row.get("removed_actual_mean_return") or 999),
            -(row.get("ticker_concentration_top10_share") or 999),
        )
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_obj = (
                other.get("false_safe_reduction") or -999,
                other.get("severe_lift_removed_vs_kept") or -999,
                -(other.get("removed_actual_mean_return") or 999),
                -(other.get("ticker_concentration_top10_share") or 999),
            )
            if all(o >= r for o, r in zip(other_obj, row_obj)) and any(
                o > r for o, r in zip(other_obj, row_obj)
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(row)
    sorted_rows = sorted(
        non_dominated or candidates,
        key=lambda row: (
            row.get("false_safe_reduction") or -999,
            row.get("severe_lift_removed_vs_kept") or -999,
            -(row.get("removed_actual_mean_return") or 999),
            -(row.get("ticker_concentration_top10_share") or 999),
        ),
        reverse=True,
    )[:5]
    test_by_id = {row["overlay_id"]: row for row in test_rows}
    result: list[dict[str, Any]] = []
    for row in sorted_rows:
        combined = dict(row)
        combined["pareto_selected_on"] = "validation"
        test = test_by_id.get(row["overlay_id"])
        if test:
            for key, value in test.items():
                if key in ("candidate_id", "overlay_id", "overlay_family"):
                    continue
                combined[f"test_{key}"] = value
        baseline_rows = cp160.baseline_rows_for_overlay(
            overlay_row=row,
            prediction=row["_prediction_ref"],
            severe_threshold=row["_severe_threshold"],
        )
        random_row = next(
            (
                item
                for item in baseline_rows
                if item.get("baseline_type") == "random_matched_warning_mean20"
            ),
            None,
        )
        line_trim_row = next(
            (
                item
                for item in baseline_rows
                if item.get("baseline_type") == "line_score_trim_matched_warning"
            ),
            None,
        )
        if random_row:
            combined["random_removed_severe_rate_mean"] = random_row.get("removed_severe_rate")
            combined["random_false_safe_reduction_mean"] = random_row.get("false_safe_reduction")
        if line_trim_row:
            combined["line_trim_removed_severe_rate"] = line_trim_row.get("removed_severe_rate")
            combined["line_trim_false_safe_reduction"] = line_trim_row.get("false_safe_reduction")
        for private_key in ("_prediction_ref", "_severe_threshold"):
            combined.pop(private_key, None)
        result.append(combined)
    return result


def attach_private_refs(
    rows: list[dict[str, Any]], prediction: dict[str, Any], severe_threshold: float
) -> None:
    for row in rows:
        row["_prediction_ref"] = prediction
        row["_severe_threshold"] = severe_threshold


def auxiliary_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
    severe_threshold: float,
    target_shares: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in AUX_FEATURES:
        values = np.asarray(features[feature], dtype=np.float64)
        val_values = np.asarray(validation_features[feature], dtype=np.float64)
        for share in target_shares:
            share = min(max(float(share), 0.001), 0.999)
            threshold = (
                float(np.quantile(val_values[np.isfinite(val_values)], 1.0 - share))
                if np.isfinite(val_values).any()
                else 0.0
            )
            warning = values >= threshold
            row = _row_with_extra(
                split=split,
                overlay_id=f"{feature}_matched_share_{share:.3f}",
                family="auxiliary_feature",
                warning_mask=warning,
                prediction=prediction,
                severe_threshold=severe_threshold,
                extra={"feature": feature, "target_warning_share": share, "threshold": threshold},
            )
            rows.append(row)
    return rows


def branch_recommendation(
    pareto_rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]
) -> str:
    test_sweet = [
        row
        for row in pareto_rows
        if (row.get("test_spread_retention") or 0) >= 0.80
        and (row.get("test_fee_retention") or 0) >= 0.80
        and (row.get("test_false_safe_reduction") or 0) > 0
        and (row.get("test_severe_lift_removed_vs_kept") or 0) > 1.0
    ]
    if test_sweet:
        return "A_RULE_PRODUCT_CANDIDATE_DIRECTION"
    high_cost = [
        row
        for row in rule_rows
        if row.get("split") == "test"
        and row.get("rule_family") in ("atr_only", "atr_or_self", "atr_and_self")
        and (row.get("false_safe_reduction") or 0) > 0.02
        and (row.get("severe_lift_removed_vs_kept") or 0) > 1.3
    ]
    if high_cost:
        return "B_CONTINUOUS_SCORE_ADJUST_DIRECTION"
    return "D_HOLD"


def write_report(payload: dict[str, Any]) -> None:
    removed_q80_test = next(
        row
        for row in payload["removed_diagnostic_rows"]
        if row["split"] == "test" and row["rule"] == "atr_ratio_q80"
    )
    pareto = payload["pareto_rows"]
    top_pareto = pareto[:5]
    lines = [
        "# CP165-LM 1D ATR Risk Overlay Sweet Spot 진단",
        "",
        "## 한 줄 결론",
        f"- 최종 분기 제안: **{payload['branch_recommendation']}**",
        "- 이번 CP는 ATR rule을 제품에 바로 붙이는 실험이 아니라, ATR이 잡는 위험과 alpha 비용을 분해한 진단이다.",
        "",
        "## 금지 작업 준수",
        "- 새 딥러닝 학습 없음",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- composite 실행 없음",
        "- CP153 band artifact 변경 없음",
        "",
        "## Stage 0 Preflight",
        f"- CP164 checkpoint: `{payload['cp164_checkpoint_path']}`",
        f"- split_mode: `{payload['split_metadata'].get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{payload['split_metadata'].get('cross_split_date_overlap_count')}`",
        f"- validation line top decile sample: `{payload['preflight']['validation_top_decile_count']}`",
        f"- test line top decile sample: `{payload['preflight']['test_top_decile_count']}`",
        f"- atr_ratio finite rate validation/test: `{_fmt(payload['preflight']['atr_finite_rate_validation'])}` / `{_fmt(payload['preflight']['atr_finite_rate_test'])}`",
        f"- self_vol_percentile_252 finite rate validation/test: `{_fmt(payload['preflight']['self_vol_finite_rate_validation'])}` / `{_fmt(payload['preflight']['self_vol_finite_rate_test'])}`",
        "",
        "## Stage 1 ATR Removed Group 진단",
        "| split | rule | removed share | removed mean | removed positive | removed severe | kept mean | kept positive | kept severe | removed score q50/q90 | top10 ticker share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["removed_diagnostic_rows"]:
        lines.append(
            f"| {row['split']} | {row['rule']} | {_fmt(row.get('removed_sample_share'))} | "
            f"{_fmt(row.get('removed_actual_mean_return'))} | {_fmt(row.get('removed_positive_return_rate'))} | "
            f"{_fmt(row.get('removed_severe_rate'))} | {_fmt(row.get('kept_actual_mean_return'))} | "
            f"{_fmt(row.get('kept_positive_return_rate'))} | {_fmt(row.get('kept_severe_rate'))} | "
            f"{_fmt(row.get('removed_line_score_q50'))}/{_fmt(row.get('removed_line_score_q90'))} | "
            f"{_fmt(row.get('removed_top10_ticker_share'))} |"
        )
    lines.extend(
        [
            "",
            "사람 말 해석:",
            f"- test 기준 `atr_ratio_q80`은 line top 후보 중 `{_fmt(removed_q80_test.get('removed_sample_share'))}`를 제거했다.",
            f"- 제거 그룹의 실제 평균 h5 수익률은 `{_fmt(removed_q80_test.get('removed_actual_mean_return'))}`, 양수 수익 비율은 `{_fmt(removed_q80_test.get('removed_positive_return_rate'))}`다.",
            f"- 즉 ATR은 위험한 종목을 잘 잡지만, 제거 그룹 안에 실제 양수 수익 후보도 상당히 섞인다. 이 때문에 CP164에서 spread/fee retention이 크게 훼손됐다.",
            "",
            "## Stage 2~3 ATR x Self-vol Pareto 후보",
            "| rule | class | val FS 감소 | val severe lift | val spread ret | val fee ret | val warning | test FS 감소 | test severe lift | test spread ret | test fee ret |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_pareto:
        lines.append(
            f"| {row['overlay_id']} | {row.get('classification')} | {_fmt(row.get('false_safe_reduction'))} | "
            f"{_fmt(row.get('severe_lift_removed_vs_kept'))} | {_fmt(row.get('spread_retention'))} | "
            f"{_fmt(row.get('fee_retention'))} | {_fmt(row.get('warning_share'))} | "
            f"{_fmt(row.get('test_false_safe_reduction'))} | {_fmt(row.get('test_severe_lift_removed_vs_kept'))} | "
            f"{_fmt(row.get('test_spread_retention'))} | {_fmt(row.get('test_fee_retention'))} |"
        )
    lines.extend(
        [
            "",
            "## Stage 4 보조 feature 비교",
            "- `vol_xs_rank_20d`, `downside_vol_ratio_20d`, `drawdown_from_5d_high`, `drawdown_from_20d_high`를 Pareto 후보 warning share와 같은 share 기준으로 비교했다.",
            f"- 상세 결과는 `{AUX_CSV}`에 기록했다.",
            "",
            "## 다음 분기",
        ]
    )
    if payload["branch_recommendation"] == "A_RULE_PRODUCT_CANDIDATE_DIRECTION":
        lines.append(
            "- A. Rule 제품 후보 방향: validation/test 모두 retention 조건과 false-safe 감소가 동시에 살아난 후보가 있어 2라벨 warning highlight 검증으로 넘길 수 있다."
        )
    elif payload["branch_recommendation"] == "B_CONTINUOUS_SCORE_ADJUST_DIRECTION":
        lines.append(
            "- B. Continuous score adjust 방향: ATR은 위험 신호가 강하지만 binary rule은 좋은 후보까지 많이 자르므로 `line_score * exp(-lambda * atr_z)` 같은 연속 감산을 먼저 보는 편이 낫다."
        )
    elif payload["branch_recommendation"] == "C_ATR_AWARE_LINE_RETRAIN_DIRECTION":
        lines.append(
            "- C. ATR-aware line 재학습 방향: 후처리만으로 trade-off가 안 좋으면 atr_ratio를 모델 feature에 재포함하는 calendar split smoke가 필요하다."
        )
    else:
        lines.append(
            "- D. 보류: ATR/self 조합도 lift와 retention의 균형이 약해 바로 다음 단계로 밀기 어렵다."
        )
    lines.extend(
        [
            "",
            "## 산출물",
            f"- metrics: `{METRICS_PATH}`",
            f"- summary: `{SUMMARY_CSV}`",
            f"- removed diagnostic: `{REMOVED_DIAG_CSV}`",
            f"- rule sweep: `{RULE_SWEEP_CSV}`",
            f"- pareto frontier: `{PARETO_CSV}`",
            f"- auxiliary comparison: `{AUX_CSV}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cp153_before = cp159.cp153_artifact_state()
    cp164_payload = load_cp164_reference()
    val_prediction, test_prediction, val_features, test_features, split_summary, thresholds = (
        build_predictions(cp164_payload)
    )
    severe_threshold = float(thresholds[0])

    removed_rows: list[dict[str, Any]] = []
    for split, prediction, features in (
        ("validation", val_prediction, val_features),
        ("test", test_prediction, test_features),
    ):
        for q in (0.70, 0.80, 0.90):
            removed_rows.append(
                removed_group_diagnostic(
                    split=split,
                    prediction=prediction,
                    feature_values=np.asarray(features["atr_ratio"], dtype=np.float64),
                    validation_feature_values=np.asarray(
                        val_features["atr_ratio"], dtype=np.float64
                    ),
                    severe_threshold=severe_threshold,
                    q=q,
                )
            )

    validation_rows = build_rule_sweep(
        split="validation",
        prediction=val_prediction,
        features=val_features,
        validation_features=val_features,
        severe_threshold=severe_threshold,
    )
    test_rows = build_rule_sweep(
        split="test",
        prediction=test_prediction,
        features=test_features,
        validation_features=val_features,
        severe_threshold=severe_threshold,
    )
    attach_private_refs(validation_rows, val_prediction, severe_threshold)
    pareto_rows = pareto_frontier(validation_rows, test_rows)
    for row in validation_rows:
        row.pop("_prediction_ref", None)
        row.pop("_severe_threshold", None)
    rule_rows = validation_rows + test_rows

    target_shares = sorted(
        {
            round(float(row.get("warning_share") or 0.0), 6)
            for row in pareto_rows
            if 0 < float(row.get("warning_share") or 0.0) < 1
        }
    )[:5]
    aux_rows: list[dict[str, Any]] = []
    if target_shares:
        aux_rows.extend(
            auxiliary_rows(
                split="validation",
                prediction=val_prediction,
                features=val_features,
                validation_features=val_features,
                severe_threshold=severe_threshold,
                target_shares=target_shares,
            )
        )
        aux_rows.extend(
            auxiliary_rows(
                split="test",
                prediction=test_prediction,
                features=test_features,
                validation_features=val_features,
                severe_threshold=severe_threshold,
                target_shares=target_shares,
            )
        )

    cp153_after = cp159.cp153_artifact_state()
    preflight = {
        "validation_top_decile_count": int(_top_mask(val_prediction).sum()),
        "test_top_decile_count": int(_top_mask(test_prediction).sum()),
        "atr_finite_rate_validation": _safe_ratio(
            int(np.isfinite(val_features["atr_ratio"]).sum()), len(val_features["atr_ratio"])
        ),
        "atr_finite_rate_test": _safe_ratio(
            int(np.isfinite(test_features["atr_ratio"]).sum()), len(test_features["atr_ratio"])
        ),
        "self_vol_finite_rate_validation": _safe_ratio(
            int(np.isfinite(val_features["self_vol_percentile_252"]).sum()),
            len(val_features["self_vol_percentile_252"]),
        ),
        "self_vol_finite_rate_test": _safe_ratio(
            int(np.isfinite(test_features["self_vol_percentile_252"]).sum()),
            len(test_features["self_vol_percentile_252"]),
        ),
        "cp153_band_artifact_unchanged": cp153_before == cp153_after,
        "new_training": False,
        "product_save_run": False,
        "db_write": False,
        "inference_save": False,
        "live_fetch": False,
        "eodhd_fallback": False,
        "composite_execution": False,
    }
    branch = branch_recommendation(pareto_rows, rule_rows)
    summary_rows = [
        {
            "stage": "preflight",
            "split_mode": split_summary.get("split_mode"),
            "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
            "validation_top_decile_count": preflight["validation_top_decile_count"],
            "test_top_decile_count": preflight["test_top_decile_count"],
            "branch_recommendation": branch,
        }
    ]
    payload = {
        "cp": "CP165-LM",
        "title": "1D ATR Risk Overlay Sweet Spot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cp164_checkpoint_path": cp164_payload["run_result"]["checkpoint_path"],
        "split_metadata": split_summary,
        "preflight": preflight,
        "severe_threshold": severe_threshold,
        "removed_diagnostic_rows": removed_rows,
        "rule_sweep_rows": rule_rows,
        "pareto_rows": pareto_rows,
        "auxiliary_rows": aux_rows,
        "branch_recommendation": branch,
    }
    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(REMOVED_DIAG_CSV, removed_rows)
    _write_csv(RULE_SWEEP_CSV, rule_rows)
    _write_csv(PARETO_CSV, pareto_rows)
    _write_csv(AUX_CSV, aux_rows)
    write_report(payload)
    print(
        json.dumps(
            {
                "status": "CP165_DONE",
                "branch_recommendation": branch,
                "split_mode": split_summary.get("split_mode"),
                "cross_split_date_overlap_count": split_summary.get(
                    "cross_split_date_overlap_count"
                ),
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
