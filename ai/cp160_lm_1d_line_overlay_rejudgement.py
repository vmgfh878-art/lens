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

from ai import cp159_lm_1d_line_conformal_overlay as cp159
from ai.models.common import ForecastOutput, LineRegimeOutput, LineV2Output
from ai.train import apply_feature_columns_to_splits, forward_model, make_loader, resolve_feature_columns

import numpy as np
import pandas as pd


DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "cp160_lm_1d_line_overlay_rejudgement_report.md"
METRICS_PATH = DOCS_DIR / "cp160_lm_1d_line_overlay_rejudgement_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp160_lm_1d_line_overlay_rejudgement_summary.csv"
OVERLAY_BASELINE_CSV = DOCS_DIR / "cp160_lm_1d_line_overlay_vs_baseline.csv"
REGIME_DETAIL_CSV = DOCS_DIR / "cp160_lm_1d_regime_overlay_detail.csv"
CONFORMAL_DETAIL_CSV = DOCS_DIR / "cp160_lm_1d_conformal_overlay_detail.csv"
BAND_BASELINE_CSV = DOCS_DIR / "cp160_lm_1d_band_overlap_baseline.csv"

RANDOM_SEEDS = tuple(range(20))
WARNING_THRESHOLDS = (0.0, -0.02, -0.05)


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


def _date_column(metadata: pd.DataFrame) -> str | None:
    for column in ("date", "asof_date", "end_date", "target_date"):
        if column in metadata.columns:
            return column
    return None


def _spearman_ic_by_date(line_score: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    date_column = _date_column(metadata)
    if date_column is None:
        return {"ic_mean": None, "ic_std": None, "ic_ir": None, "ic_t_stat": None, "ic_count": 0}
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(metadata[date_column], errors="coerce"),
            "line_score": line_score,
            "actual": actual,
        }
    ).dropna()
    values: list[float] = []
    for _date, group in frame.groupby("date", sort=False):
        if len(group) < 3:
            continue
        corr = group["line_score"].corr(group["actual"], method="spearman")
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    if not values:
        return {"ic_mean": None, "ic_std": None, "ic_ir": None, "ic_t_stat": None, "ic_count": 0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": None if std <= 0 else float(mean / std),
        "ic_t_stat": None if std <= 0 else float(mean / (std / math.sqrt(len(arr)))),
        "ic_count": int(len(arr)),
    }


def _daily_long_short_spread(line_score: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    date_column = _date_column(metadata)
    if date_column is None:
        return {"long_short_spread": None, "spread_count": 0}
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(metadata[date_column], errors="coerce"),
            "line_score": line_score,
            "actual": actual,
        }
    ).dropna()
    spreads: list[float] = []
    for _date, group in frame.groupby("date", sort=False):
        if len(group) < 20:
            continue
        q10 = group["line_score"].quantile(0.10)
        q90 = group["line_score"].quantile(0.90)
        top_mean = group.loc[group["line_score"] >= q90, "actual"].mean()
        bottom_mean = group.loc[group["line_score"] <= q10, "actual"].mean()
        if pd.notna(top_mean) and pd.notna(bottom_mean):
            spreads.append(float(top_mean - bottom_mean))
    if not spreads:
        return {"long_short_spread": None, "spread_count": 0}
    return {"long_short_spread": float(np.mean(spreads)), "spread_count": int(len(spreads))}


def collect_predictions(
    *,
    candidate_id: str,
    checkpoint_path: str,
    model_kind: str,
    bundle: Any,
    mean: Any,
    std: Any,
    device: Any,
    config: Any,
) -> dict[str, Any]:
    model, payload = cp159.load_model_for_prediction(config, checkpoint_path, device)
    feature_columns = list(payload.get("config", {}).get("feature_columns") or config.feature_columns or resolve_feature_columns(config.feature_set))
    _, selected_bundle, _, _, _ = apply_feature_columns_to_splits(bundle, bundle, bundle, mean, std, feature_columns)
    loader = make_loader(selected_bundle, batch_size=1024, shuffle=False, device=device, num_workers=0)
    line_chunks: list[Any] = []
    raw_chunks: list[Any] = []
    regime_chunks: list[Any] = []
    with cp159.torch.no_grad():
        for features, _line_target, _band_target, raw_returns, ticker_id, future_cov in loader:
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_cov = future_cov.to(device, non_blocking=True)
            output = forward_model(model, features, ticker_id, future_cov)
            if isinstance(output, (ForecastOutput, LineRegimeOutput, LineV2Output)):
                line_chunks.append(output.line.detach().cpu())
            else:
                raise TypeError(f"{candidate_id} 출력에서 line을 읽을 수 없습니다: {type(output).__name__}")
            if isinstance(output, LineRegimeOutput):
                regime_chunks.append(output.regime_logits.detach().cpu())
            raw_chunks.append(raw_returns.detach().cpu())
    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()
    line = cp159.torch.cat(line_chunks, dim=0).to(cp159.torch.float32).numpy()
    raw = cp159.torch.cat(raw_chunks, dim=0).to(cp159.torch.float32).numpy()
    regime_logits = None
    regime_pred = None
    if regime_chunks:
        regime_logits = cp159.torch.cat(regime_chunks, dim=0).to(cp159.torch.float32).numpy()
        regime_pred = np.argmax(regime_logits, axis=1).astype(np.int64)
    return {
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "checkpoint_path": checkpoint_path,
        "line_score": line[:, -1],
        "actual": raw[:, -1],
        "metadata": selected_bundle.metadata.reset_index(drop=True).copy(),
        "realized_vol_20d": cp159.collect_realized_vol_20d(selected_bundle),
        "regime_logits": regime_logits,
        "regime_pred": regime_pred,
        "checkpoint_config": payload.get("config", {}),
    }


def line_alpha_metrics(candidate_id: str, split: str, prediction: dict[str, Any], severe_threshold: float) -> dict[str, Any]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    q10 = float(np.quantile(line_score, 0.10))
    q90 = float(np.quantile(line_score, 0.90))
    top = line_score >= q90
    bottom = line_score <= q10
    positive = line_score >= 0
    severe = actual <= severe_threshold
    top_mean = _safe_mean(actual[top])
    bottom_mean = _safe_mean(actual[bottom])
    global_spread = None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
    ic = _spearman_ic_by_date(line_score, actual, prediction["metadata"])
    spread_payload = _daily_long_short_spread(line_score, actual, prediction["metadata"])
    spread = spread_payload["long_short_spread"]
    return {
        "candidate_id": candidate_id,
        "split": split,
        **ic,
        **spread_payload,
        "long_short_spread": spread,
        "fee_adjusted_return": None if spread is None else float(spread - 0.001),
        "top_decile_actual_return": top_mean,
        "bottom_decile_actual_return": bottom_mean,
        "global_top_bottom_spread": global_spread,
        "line_alpha_alive": bool((ic.get("ic_mean") or 0) > 0 and (spread or 0) > 0 and ((spread or 0) - 0.001) > 0),
        "line_top_decile_false_safe_rate": _safe_ratio(int((top & severe).sum()), int(top.sum())),
        "line_positive_false_safe_rate": _safe_ratio(int((positive & severe).sum()), int(positive.sum())),
        "line_top_decile_severe_rate": _safe_ratio(int((top & severe).sum()), int(top.sum())),
        "top_count": int(top.sum()),
        "bottom_count": int(bottom.sum()),
        "q10": q10,
        "q90": q90,
    }


def _top_bottom_masks(line_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q10 = float(np.quantile(line_score, 0.10))
    q90 = float(np.quantile(line_score, 0.90))
    return line_score >= q90, line_score <= q10


def overlay_filter_metrics(
    *,
    candidate_id: str,
    split: str,
    overlay_family: str,
    overlay_id: str,
    warning_mask_all: np.ndarray,
    prediction: dict[str, Any],
    severe_threshold: float,
) -> dict[str, Any]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    top, bottom = _top_bottom_masks(line_score)
    severe = actual <= severe_threshold
    warning_top = top & warning_mask_all
    kept = top & ~warning_mask_all
    removed = warning_top
    top_count = int(top.sum())
    removed_count = int(removed.sum())
    kept_count = int(kept.sum())
    top_mean = _safe_mean(actual[top])
    bottom_mean = _safe_mean(actual[bottom])
    base_spread = None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
    kept_mean = _safe_mean(actual[kept])
    spread_after = None if kept_mean is None or bottom_mean is None else float(kept_mean - bottom_mean)
    all_top_severe_rate = _safe_ratio(int((top & severe).sum()), top_count)
    kept_severe_rate = _safe_ratio(int((kept & severe).sum()), kept_count)
    removed_severe_rate = _safe_ratio(int((removed & severe).sum()), removed_count)
    base_fee = None if base_spread is None else float(base_spread - 0.001)
    fee_after = None if spread_after is None else float(spread_after - 0.001)
    return {
        "candidate_id": candidate_id,
        "split": split,
        "overlay_family": overlay_family,
        "overlay_id": overlay_id,
        "baseline_type": "overlay",
        "warning_share": _safe_ratio(removed_count, top_count),
        "kept_share": _safe_ratio(kept_count, top_count),
        "removed_share": _safe_ratio(removed_count, top_count),
        "kept_sample_count": kept_count,
        "removed_sample_count": removed_count,
        "removed_actual_mean_return": _safe_mean(actual[removed]),
        "kept_actual_mean_return": kept_mean,
        "removed_severe_rate": removed_severe_rate,
        "kept_severe_rate": kept_severe_rate,
        "removed_false_safe_rate": removed_severe_rate,
        "kept_false_safe_rate": kept_severe_rate,
        "line_top_decile_severe_rate": all_top_severe_rate,
        "severe_lift_removed_vs_all": None if removed_severe_rate is None or not all_top_severe_rate else float(removed_severe_rate / all_top_severe_rate),
        "severe_lift_removed_vs_kept": None if removed_severe_rate is None or not kept_severe_rate else float(removed_severe_rate / kept_severe_rate),
        "false_safe_reduction": None if all_top_severe_rate is None or kept_severe_rate is None else float(all_top_severe_rate - kept_severe_rate),
        "base_long_short_spread": base_spread,
        "long_short_spread_after_filter": spread_after,
        "spread_retention": None if spread_after is None or base_spread is None or abs(base_spread) < 1e-12 else float(spread_after / base_spread),
        "base_fee_proxy": base_fee,
        "fee_proxy_after_filter": fee_after,
        "fee_retention": None if fee_after is None or base_fee is None or abs(base_fee) < 1e-12 else float(fee_after / base_fee),
    }


def _metrics_for_removed_indices(
    *,
    candidate_id: str,
    split: str,
    overlay_family: str,
    overlay_id: str,
    baseline_type: str,
    top_indices: np.ndarray,
    removed_indices: np.ndarray,
    prediction: dict[str, Any],
    severe_threshold: float,
) -> dict[str, Any]:
    warning_mask = np.zeros(len(prediction["actual"]), dtype=bool)
    warning_mask[removed_indices] = True
    row = overlay_filter_metrics(
        candidate_id=candidate_id,
        split=split,
        overlay_family=overlay_family,
        overlay_id=overlay_id,
        warning_mask_all=warning_mask,
        prediction=prediction,
        severe_threshold=severe_threshold,
    )
    row["baseline_type"] = baseline_type
    row["top_pool_count"] = int(len(top_indices))
    return row


def baseline_rows_for_overlay(
    *,
    overlay_row: dict[str, Any],
    prediction: dict[str, Any],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    top, _bottom = _top_bottom_masks(line_score)
    top_indices = np.flatnonzero(top)
    removed_count = int(overlay_row.get("removed_sample_count") or 0)
    rows: list[dict[str, Any]] = []
    if removed_count <= 0 or removed_count >= len(top_indices):
        return rows

    rng_rows: list[dict[str, Any]] = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        removed = rng.choice(top_indices, size=removed_count, replace=False)
        row = _metrics_for_removed_indices(
            candidate_id=overlay_row["candidate_id"],
            split=overlay_row["split"],
            overlay_family=overlay_row["overlay_family"],
            overlay_id=overlay_row["overlay_id"],
            baseline_type=f"random_warning_seed{seed}",
            top_indices=top_indices,
            removed_indices=removed,
            prediction=prediction,
            severe_threshold=severe_threshold,
        )
        rng_rows.append(row)
    random_summary = {
        key: _safe_mean(np.asarray([row.get(key) for row in rng_rows if row.get(key) is not None], dtype=np.float64))
        for key in (
            "removed_actual_mean_return",
            "kept_actual_mean_return",
            "removed_severe_rate",
            "kept_severe_rate",
            "severe_lift_removed_vs_all",
            "severe_lift_removed_vs_kept",
            "false_safe_reduction",
            "spread_retention",
            "fee_retention",
        )
    }
    random_std = {
        f"{key}_std": (
            float(np.asarray([row.get(key) for row in rng_rows if row.get(key) is not None], dtype=np.float64).std(ddof=1))
            if len([row.get(key) for row in rng_rows if row.get(key) is not None]) > 1
            else None
        )
        for key in (
            "removed_severe_rate",
            "false_safe_reduction",
            "spread_retention",
            "fee_retention",
        )
    }
    random_row = dict(overlay_row)
    random_row.update(random_summary)
    random_row.update(random_std)
    random_row["baseline_type"] = "random_warning_mean20"
    rows.append(random_row)

    vol = np.asarray(prediction["realized_vol_20d"], dtype=np.float64)
    vol_top_order = top_indices[np.argsort(vol[top_indices])[::-1]]
    rows.append(
        _metrics_for_removed_indices(
            candidate_id=overlay_row["candidate_id"],
            split=overlay_row["split"],
            overlay_family=overlay_row["overlay_family"],
            overlay_id=overlay_row["overlay_id"],
            baseline_type="volatility_warning_matched_share",
            top_indices=top_indices,
            removed_indices=vol_top_order[:removed_count],
            prediction=prediction,
            severe_threshold=severe_threshold,
        )
    )

    line_top_order = top_indices[np.argsort(line_score[top_indices])]
    rows.append(
        _metrics_for_removed_indices(
            candidate_id=overlay_row["candidate_id"],
            split=overlay_row["split"],
            overlay_family=overlay_row["overlay_family"],
            overlay_id=overlay_row["overlay_id"],
            baseline_type="line_score_conservative_trim_matched_share",
            top_indices=top_indices,
            removed_indices=line_top_order[:removed_count],
            prediction=prediction,
            severe_threshold=severe_threshold,
        )
    )
    return rows


def no_overlay_row(candidate_id: str, split: str, prediction: dict[str, Any], severe_threshold: float) -> dict[str, Any]:
    warning_mask = np.zeros(len(prediction["actual"]), dtype=bool)
    row = overlay_filter_metrics(
        candidate_id=candidate_id,
        split=split,
        overlay_family="no_overlay",
        overlay_id="no_overlay",
        warning_mask_all=warning_mask,
        prediction=prediction,
        severe_threshold=severe_threshold,
    )
    row["baseline_type"] = "no_overlay"
    return row


def regime_detail_rows(candidate_id: str, split: str, prediction: dict[str, Any], severe_threshold: float) -> list[dict[str, Any]]:
    regime_pred = prediction.get("regime_pred")
    if regime_pred is None:
        return []
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    severe = actual <= severe_threshold
    rows: list[dict[str, Any]] = []
    for regime_class in range(5):
        mask = regime_pred == regime_class
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "regime_class": regime_class,
                "sample_count": int(mask.sum()),
                "class_share": float(mask.mean()),
                "actual_mean_return": _safe_mean(actual[mask]),
                "actual_median_return": None if not mask.any() else float(np.median(actual[mask])),
                "severe_rate": _safe_ratio(int((mask & severe).sum()), int(mask.sum())),
            }
        )
    shares = np.asarray([float((regime_pred == klass).mean()) for klass in range(5)], dtype=np.float64)
    entropy = float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum())
    rows.append(
        {
            "candidate_id": candidate_id,
            "split": split,
            "regime_class": "distribution_summary",
            "sample_count": int(len(regime_pred)),
            "class_share": None,
            "max_class_share": float(shares.max()),
            "prediction_entropy": entropy,
            "class_collapse_score": float(shares.max()),
        }
    )
    return rows


def conformal_detail_rows(
    *,
    candidate_id: str,
    split: str,
    overlay_id: str,
    lower: np.ndarray,
    actual: np.ndarray,
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    breach = actual < lower
    for threshold in WARNING_THRESHOLDS:
        warning = lower < threshold
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "overlay_id": overlay_id,
                "warning_threshold": threshold,
                "lower_breach_rate": float(breach.mean()),
                "target_breach_rate": alpha,
                "coverage_abs_error": abs(float(breach.mean()) - alpha),
                "warning_share": float(warning.mean()),
                "warning_lower_mean": _safe_mean(lower[warning]),
                "warning_lower_median": None if not warning.any() else float(np.median(lower[warning])),
            }
        )
    return rows


def classify_overlay(row: dict[str, Any], baseline_rows: list[dict[str, Any]], line_alpha_alive: bool) -> dict[str, Any]:
    if not line_alpha_alive:
        return {"final_label": "reject", "interpretability_score": "low", "baseline_excess_lift": None, "reason": "line alpha 실패"}
    removed_lift = row.get("severe_lift_removed_vs_all")
    false_safe_reduction = row.get("false_safe_reduction")
    spread_retention = row.get("spread_retention")
    comparable = [
        base
        for base in baseline_rows
        if base.get("baseline_type") in {"random_warning_mean20", "volatility_warning_matched_share", "line_score_conservative_trim_matched_share"}
        and base.get("severe_lift_removed_vs_all") is not None
    ]
    baseline_by_type = {str(base.get("baseline_type")): base for base in comparable}
    random_lift = baseline_by_type.get("random_warning_mean20", {}).get("severe_lift_removed_vs_all")
    volatility_lift = baseline_by_type.get("volatility_warning_matched_share", {}).get("severe_lift_removed_vs_all")
    line_trim_lift = baseline_by_type.get("line_score_conservative_trim_matched_share", {}).get("severe_lift_removed_vs_all")
    best_baseline_lift = max((float(base["severe_lift_removed_vs_all"]) for base in comparable), default=None)
    baseline_excess_lift = None if best_baseline_lift is None or removed_lift is None else float(removed_lift - best_baseline_lift)
    overlay_family = str(row.get("overlay_family"))
    interpretability_score = "high" if overlay_family == "conformal" else "medium"
    if overlay_family == "regime" and row.get("class_collapse_score") is not None and float(row["class_collapse_score"]) >= 0.90:
        interpretability_score = "low"
    beats_baseline = baseline_excess_lift is not None and baseline_excess_lift > 0
    beats_random = removed_lift is not None and random_lift is not None and float(removed_lift) > float(random_lift)
    beats_volatility = removed_lift is not None and volatility_lift is not None and float(removed_lift) > float(volatility_lift)
    beats_line_trim = removed_lift is not None and line_trim_lift is not None and float(removed_lift) > float(line_trim_lift)
    if (
        beats_baseline
        and (false_safe_reduction or 0) >= 0.015
        and (spread_retention or 0) >= 0.80
        and interpretability_score != "low"
    ):
        label = "product_warning_candidate"
    elif (
        interpretability_score != "low"
        and (false_safe_reduction or 0) > 0
        and (spread_retention or 0) >= 0.80
        and (beats_random or beats_line_trim or beats_volatility or (removed_lift or 0) > 1.0)
    ):
        label = "research_warning_candidate"
    elif (false_safe_reduction or 0) > 0:
        label = "weak_auxiliary_signal"
    else:
        label = "reject"
    reason = (
        f"removed lift={_fmt(removed_lift)}, best baseline lift={_fmt(best_baseline_lift)}, "
        f"false-safe reduction={_fmt(false_safe_reduction)}, spread retention={_fmt(spread_retention)}"
    )
    return {
        "final_label": label,
        "interpretability_score": interpretability_score,
        "baseline_excess_lift": baseline_excess_lift,
        "best_baseline_lift": best_baseline_lift,
        "random_baseline_lift": random_lift,
        "volatility_baseline_lift": volatility_lift,
        "line_trim_baseline_lift": line_trim_lift,
        "beats_random_baseline": beats_random,
        "beats_volatility_baseline": beats_volatility,
        "beats_line_trim_baseline": beats_line_trim,
        "reason": reason,
    }


def build_report(payload: dict[str, Any]) -> None:
    summary_rows = payload["summary_rows"]
    test_rows = [row for row in summary_rows if row["split"] == "test"]
    product = [row for row in test_rows if row["final_label"] == "product_warning_candidate"]
    research = [row for row in test_rows if row["final_label"] == "research_warning_candidate"]
    weak = [row for row in test_rows if row["final_label"] == "weak_auxiliary_signal"]
    lines = [
        "# CP160-LM 1D Line Overlay Re-judgement",
        "",
        "## 한 줄 결론",
        "이번 CP는 line을 버릴지 보는 실험이 아니다. line은 수익성/순위 신호로 재해석했고, regime/conformal overlay가 그 line의 하방 위험 해석을 보조하는지 같은 기준으로 다시 봤다. 결과적으로 product_warning_candidate는 없었다. p32/s16 regime overlay와 volatility-bucket conformal overlay는 random 또는 line-score trim보다 나은 부분이 있지만, 순수 volatility baseline보다 removed severe lift가 약해 research_warning_candidate로만 남긴다.",
        "",
        "## 역할 분리",
        "- line: 수익성/순위 신호",
        "- risk overlay: line 해석 보조 신호",
        "- band: 범위/하방 기준",
        "",
        "line 단독 false-safe는 line 탈락 게이트가 아니다. line만 보고 하방 리스크를 해석하면 안 된다는 근거이며, 별도 risk overlay 또는 band 확인이 필요하다는 신호다.",
        "",
        "## 최종 라벨 요약",
        f"- product_warning_candidate: {len(product)}개",
        f"- research_warning_candidate: {len(research)}개",
        f"- weak_auxiliary_signal: {len(weak)}개",
        "",
        "| 후보 | overlay | split | label | false-safe 감소 | spread retention | removed severe lift | random lift | vol lift | line-trim lift | 해석성 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(test_rows, key=lambda item: (item["final_label"], -(item.get("false_safe_reduction") or 0))):
        if row["final_label"] == "reject":
            continue
        lines.append(
            f"| {row['candidate_id']} | {row['overlay_id']} | {row['split']} | {row['final_label']} | "
            f"{_fmt(row.get('false_safe_reduction'))} | {_fmt(row.get('spread_retention'))} | "
            f"{_fmt(row.get('severe_lift_removed_vs_all'))} | {_fmt(row.get('random_baseline_lift'))} | "
            f"{_fmt(row.get('volatility_baseline_lift'))} | {_fmt(row.get('line_trim_baseline_lift'))} | "
            f"{row.get('interpretability_score')} |"
        )
    lines.extend(
        [
            "",
            "## 질문별 답변",
            "1. line은 수익성 신호로 살아 있는가?",
            "   - p32/s16 계열은 IC, spread, fee가 양수라 수익성/순위 신호는 살아 있다. 따라서 line 단독 false-safe만으로 line을 탈락시키지 않는다.",
            "2. line 단독 리스크 지표는 왜 탈락 게이트가 아닌가?",
            "   - line은 평균선이 아니라 보수적 point forecast지만, band나 risk classifier가 아니다. line top decile false-safe는 line의 역할 한계를 보여주는 지표다.",
            "3. regime overlay는 baseline보다 나은가?",
            "   - p32/s16 regime warning은 random warning과 line-score trim보다 낫고 spread를 보존했다. 다만 순수 volatility warning은 severe를 더 많이 잡았지만 spread retention이 낮아, regime은 제품 확정이 아니라 연구 후보로 둔다. p16/s8은 class 2 collapse가 있어 해석성이 약하다.",
            "4. conformal overlay는 baseline보다 나은가?",
            "   - volatility bucket alpha 0.05는 line top decile false-safe를 소폭 줄이고 spread를 보존했다. 그러나 같은 비율을 변동성 상위로 제거하는 baseline보다 removed severe lift가 약해 제품 경고 후보로는 부족하다.",
            "5. CP153 band lower baseline과 비교하면 새 정보가 있는가?",
            "   - 이번 CP에서는 같은 row-level split의 band lower를 안전하게 확보하지 못해 band baseline은 deferred로 기록했다. CP153 artifact는 읽기/해시 확인만 했고 수정하지 않았다.",
            "6. 프론트 warning badge로 붙일 만한가?",
            "   - 기본 제품 경고로 바로 붙이기에는 약하다. 다만 regime p32/s16은 연구용 warning badge 후보로 남길 수 있다.",
            "7. 문구 초안",
            "   - Lens line은 수익성/순위 중심의 보수 예측선입니다.",
            "   - 이 선만으로 하방 위험을 판단하지 마세요.",
            "   - 하방 위험은 밴드 하단과 별도 경고 신호를 함께 확인해야 합니다.",
            "   - regime badge: 방향 해석: 상승/중립/하락",
            "   - conformal badge: 통계적 하방 여유: 양호/주의/위험",
            "",
            "## 준수 사항",
            "- 새 학습 없음",
            "- product save-run 없음",
            "- DB write 없음",
            "- inference 저장 없음",
            "- live fetch / EODHD fallback 없음",
            "- composite / band 실행 없음",
            f"- CP153 band artifact 변경: {payload['cp153_artifact_guard']['changed']}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cp153_before = cp159.cp153_artifact_state()
    price, indicators, price_manifest, indicator_manifest = cp159.load_source_frames()
    source_hash = str(indicator_manifest.get("source_data_hash") or price_manifest.get("source_data_hash") or "unknown")
    train, val, test, mean, std, plan, _registry = cp159.build_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    del train, plan
    device = cp159.resolve_device("cuda" if cp159.torch.cuda.is_available() else "cpu")
    candidates = cp159.load_stage2_candidates()
    severe_threshold = float(_read_json(cp159.CP158_METRICS_PATH)["stage0"]["regime_thresholds"]["regime_threshold_q10"])
    split_bundles = {"validation": val, "test": test}

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    line_rows: list[dict[str, Any]] = []
    regime_detail: list[dict[str, Any]] = []
    conformal_detail: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    overlay_baseline_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in candidates:
        candidate_id = row["candidate_id"]
        config = cp159.candidate_config(row, device=device)
        checkpoint_path = str(row.get("checkpoint_path"))
        model_kind = str(row.get("model_kind") or "line_regime")
        for split, bundle in split_bundles.items():
            prediction = collect_predictions(
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                model_kind=model_kind,
                bundle=bundle,
                mean=mean,
                std=std,
                device=device,
                config=config,
            )
            predictions[(candidate_id, split)] = prediction
            line_rows.append(line_alpha_metrics(candidate_id, split, prediction, severe_threshold))
            regime_detail.extend(regime_detail_rows(candidate_id, split, prediction, severe_threshold))

    line_by_key = {(row["candidate_id"], row["split"]): row for row in line_rows}

    for candidate_id in {row["candidate_id"] for row in candidates}:
        for split in ("validation", "test"):
            prediction = predictions[(candidate_id, split)]
            no_row = no_overlay_row(candidate_id, split, prediction, severe_threshold)
            overlay_baseline_rows.append(no_row)

    for candidate_id in ("patchtst_line_regime_p32_s16", "patchtst_line_regime_p16_s8"):
        for split in ("validation", "test"):
            prediction = predictions[(candidate_id, split)]
            regime_pred = prediction.get("regime_pred")
            if regime_pred is None:
                continue
            warning_mask = np.isin(regime_pred, [0, 1])
            overlay = overlay_filter_metrics(
                candidate_id=candidate_id,
                split=split,
                overlay_family="regime",
                overlay_id="regime_class_0_1_warning",
                warning_mask_all=warning_mask,
                prediction=prediction,
                severe_threshold=severe_threshold,
            )
            collapse_rows = [row for row in regime_detail if row["candidate_id"] == candidate_id and row["split"] == split and row["regime_class"] == "distribution_summary"]
            if collapse_rows:
                overlay["class_collapse_score"] = collapse_rows[0].get("class_collapse_score")
            overlay_rows.append(overlay)
            baselines = baseline_rows_for_overlay(overlay_row=overlay, prediction=prediction, severe_threshold=severe_threshold)
            overlay_baseline_rows.extend([overlay, *baselines])
            judgement = classify_overlay(overlay, baselines, bool(line_by_key[(candidate_id, split)]["line_alpha_alive"]))
            summary_rows.append({**line_by_key[(candidate_id, split)], **overlay, **judgement})

    candidate_validation_predictions = {
        candidate_id: predictions[(candidate_id, "validation")]
        for candidate_id in {row["candidate_id"] for row in candidates}
    }
    conformal_params_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate_id, validation_prediction in candidate_validation_predictions.items():
        _params_rows, params = cp159.fit_params_for_candidate(validation_prediction)
        conformal_params_by_candidate[candidate_id] = params

    for candidate_id, params_by_id in conformal_params_by_candidate.items():
        for split in ("validation", "test"):
            prediction = predictions[(candidate_id, split)]
            actual = np.asarray(prediction["actual"], dtype=np.float64)
            for overlay_id, params in params_by_id.items():
                bucket_values = None
                if params["method"] == "volatility_bucket":
                    bucket_values = np.asarray(prediction["realized_vol_20d"], dtype=np.float64)
                elif params["method"] == "line_score_bucket":
                    bucket_values = np.asarray(prediction["line_score"], dtype=np.float64)
                lower = cp159.apply_overlay(np.asarray(prediction["line_score"], dtype=np.float64), bucket_values, params)
                conformal_detail.extend(
                    conformal_detail_rows(
                        candidate_id=candidate_id,
                        split=split,
                        overlay_id=overlay_id,
                        lower=lower,
                        actual=actual,
                        alpha=float(params["alpha"]),
                    )
                )
                for threshold in WARNING_THRESHOLDS:
                    label = str(threshold).replace("-", "neg_").replace(".", "p")
                    warning_mask = lower < threshold
                    overlay = overlay_filter_metrics(
                        candidate_id=candidate_id,
                        split=split,
                        overlay_family="conformal",
                        overlay_id=f"{overlay_id}_lower_lt_{label}",
                        warning_mask_all=warning_mask,
                        prediction=prediction,
                        severe_threshold=severe_threshold,
                    )
                    overlay["alpha"] = float(params["alpha"])
                    overlay["conformal_method"] = params["method"]
                    overlay["warning_threshold"] = threshold
                    overlay["lower_breach_rate"] = float((actual < lower).mean())
                    overlay["coverage_abs_error"] = abs(overlay["lower_breach_rate"] - float(params["alpha"]))
                    overlay_rows.append(overlay)
                    baselines = baseline_rows_for_overlay(overlay_row=overlay, prediction=prediction, severe_threshold=severe_threshold)
                    overlay_baseline_rows.extend([overlay, *baselines])
                    judgement = classify_overlay(overlay, baselines, bool(line_by_key[(candidate_id, split)]["line_alpha_alive"]))
                    summary_rows.append({**line_by_key[(candidate_id, split)], **overlay, **judgement})

    band_rows = [
        {
            "status": "deferred",
            "reason": "CP153 band lower의 동일 row-level split 값을 안전하게 확보하지 못해 이번 CP에서는 band artifact를 수정하지 않고 후속 비교로 남김",
            "cp153_artifact_changed": False,
        }
    ]
    cp153_after = cp159.cp153_artifact_state()
    cp153_changed = cp153_before != cp153_after
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "new_training": False,
        "save_run": False,
        "db_write": False,
        "inference_saved": False,
        "live_fetch": False,
        "composite_used": False,
        "band_experiment_used": False,
        "provider": cp159.PROVIDER,
        "source": cp159.PROVIDER,
        "timeframe": cp159.TIMEFRAME,
        "horizon": cp159.HORIZON,
        "source_data_hash": source_hash,
        "severe_threshold": severe_threshold,
        "line_rows": line_rows,
        "summary_rows": summary_rows,
        "overlay_baseline_rows": overlay_baseline_rows,
        "regime_detail_rows": regime_detail,
        "conformal_detail_rows": conformal_detail,
        "band_baseline_rows": band_rows,
        "cp153_artifact_guard": {
            "changed": cp153_changed,
            "before": cp153_before,
            "after": cp153_after,
        },
    }
    _write_json(METRICS_PATH, metrics)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(OVERLAY_BASELINE_CSV, overlay_baseline_rows)
    _write_csv(REGIME_DETAIL_CSV, regime_detail)
    _write_csv(CONFORMAL_DETAIL_CSV, conformal_detail)
    _write_csv(BAND_BASELINE_CSV, band_rows)
    build_report(metrics)
    print(
        json.dumps(
            {
                "status": "CP160_DONE",
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
                "summary": str(SUMMARY_CSV),
                "cp153_changed": cp153_changed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
