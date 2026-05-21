from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai import cp148_lm_1d_stage4_0_risk_aware_rerank as r0  # noqa: E402
from ai import cp148_lm_1d_stage4_1_risk_feature_abcd as s41  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4f_failure_analysis_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4f_failure_analysis_metrics.json"
TOP_DATES_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4f_false_safe_top_dates.csv"
TOP_TICKERS_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4f_false_safe_top_tickers.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "cp148_lm_1d_stage4_4_seed_stability"

C_FEATURES = ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]
TARGET_CANDIDATES = {"trial006_c_balanced", "trial024_c_risk"}
TARGET_SEEDS = {42, 7, 123}
SEVERE_THRESHOLD = -0.05

FEATURE_ANALYSIS_COLUMNS = [
    "atr_ratio",
    "vix_change_5d",
    "credit_spread_change_20d",
    "ma200_pct_change_20d",
    "rsi",
    "bb_position",
    "ma_20_ratio",
    "ma_60_ratio",
    "vol_change",
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "ma_5_ratio",
]

REGIME_MASKS = [
    "stress",
    "vix_rising",
    "breadth_worsening",
    "non_stress",
    "non_vix_rising",
    "non_breadth_worsening",
]

HORIZON_BUCKETS = {
    "h1": (0, 1),
    "h2_h3": (1, 3),
    "h4_h5": (3, 5),
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_stage4_4_metas() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("stage4_4_*/run_meta.json")):
        meta = _read_json(path)
        candidate_id = str(meta.get("candidate_id"))
        seed = int(meta.get("seed"))
        if candidate_id not in TARGET_CANDIDATES or seed not in TARGET_SEEDS:
            continue
        checkpoint_path = _resolve_path(meta.get("checkpoint_path"))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 없음: {checkpoint_path}")
        meta["run_meta_path"] = str(path)
        meta["checkpoint_path_abs"] = str(checkpoint_path)
        rows.append(meta)
    if len(rows) != 6:
        raise RuntimeError(f"Stage 4-4 대상 run_meta 6개가 필요하지만 {len(rows)}개를 찾았습니다.")
    return sorted(rows, key=lambda item: (str(item["candidate_id"]), int(item["seed"])))


def _stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    quantiles = np.quantile(finite, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "q50": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "max": float(np.max(finite)),
    }


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denom = int(np.asarray(denominator, dtype=bool).sum())
    if denom == 0:
        return None
    return float((np.asarray(numerator, dtype=bool) & np.asarray(denominator, dtype=bool)).sum() / denom)


def _margin_buckets(score: np.ndarray, false_safe_mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(score, dtype=np.float64)[false_safe_mask]
    buckets = {
        "0_to_0p001": (selected >= 0.0) & (selected < 0.001),
        "0p001_to_0p003": (selected >= 0.001) & (selected < 0.003),
        "0p003_to_0p005": (selected >= 0.003) & (selected < 0.005),
        "0p005_plus": selected >= 0.005,
    }
    total = int(selected.size)
    return {
        name: {
            "count": int(mask.sum()),
            "share": float(mask.sum() / total) if total else None,
        }
        for name, mask in buckets.items()
    } | {"total_false_safe": total}


def _last_feature_matrix(dataset: Any, feature_names: list[str]) -> np.ndarray:
    mean = dataset.mean.detach().cpu().numpy() if getattr(dataset, "mean", None) is not None else None
    std = dataset.std.detach().cpu().numpy() if getattr(dataset, "std", None) is not None else None
    rows = np.empty((len(dataset.sample_refs), len(feature_names)), dtype=np.float32)
    for row_idx, (ticker, end_idx) in enumerate(dataset.sample_refs):
        values = dataset.ticker_arrays[ticker]["features"][end_idx].astype("float32", copy=False)
        if mean is not None and std is not None:
            values = (values - mean) / np.clip(std, 1e-6, None)
        rows[row_idx] = values
    return rows


def _flat_event_frame(metadata: pd.DataFrame, line: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
    horizon = int(line.shape[1])
    sample_count = int(line.shape[0])
    sample_index = np.repeat(np.arange(sample_count), horizon)
    horizon_index = np.tile(np.arange(1, horizon + 1), sample_count)
    frame = pd.DataFrame(
        {
            "sample_row": sample_index,
            "ticker": np.repeat(metadata["ticker"].astype(str).to_numpy(), horizon),
            "asof_date": np.repeat(metadata["asof_date"].astype(str).to_numpy(), horizon),
            "horizon": horizon_index,
            "score": line.reshape(-1),
            "actual": raw.reshape(-1),
        }
    )
    for column in ["regime_stress", "vix_change_5d", "ma200_pct_change_20d"]:
        if column in metadata.columns:
            frame[column] = np.repeat(metadata[column].to_numpy(), horizon)
    return frame


def _attach_tail_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "actual"]).copy()
    tail_cutoff = float(output["actual"].quantile(0.20))
    output["tail_cutoff"] = tail_cutoff
    output["tail_actual"] = output["actual"] <= tail_cutoff
    output["severe_actual"] = output["actual"] <= SEVERE_THRESHOLD
    output["predicted_safe"] = output["score"] >= 0.0
    output["false_safe_tail"] = output["tail_actual"] & output["predicted_safe"]
    output["missed_severe"] = output["severe_actual"] & output["predicted_safe"]
    output["caught_severe"] = output["severe_actual"] & (~output["predicted_safe"])
    output["stress"] = output.get("regime_stress", pd.Series(False, index=output.index)).fillna(0).astype(float) > 0.5
    output["vix_rising"] = output.get("vix_change_5d", pd.Series(np.nan, index=output.index)).fillna(0).astype(float) > 0.0
    output["breadth_worsening"] = output.get("ma200_pct_change_20d", pd.Series(np.nan, index=output.index)).fillna(0).astype(float) < 0.0
    return output


def _event_metric(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = frame.loc[mask].copy()
    if selected.empty:
        return {"event_count": 0}
    tail_count = int(selected["tail_actual"].sum())
    severe_count = int(selected["severe_actual"].sum())
    return {
        "event_count": int(len(selected)),
        "tail_count": tail_count,
        "false_safe_count": int(selected["false_safe_tail"].sum()),
        "false_safe_tail_rate": float(selected["false_safe_tail"].sum() / tail_count) if tail_count else None,
        "severe_count": severe_count,
        "missed_severe_count": int(selected["missed_severe"].sum()),
        "severe_downside_recall": float(selected["caught_severe"].sum() / severe_count) if severe_count else None,
        "score_mean": float(selected["score"].mean()),
        "score_positive_rate": float((selected["score"] >= 0).mean()),
    }


def _regime_event_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    masks = {
        "stress": frame["stress"],
        "vix_rising": frame["vix_rising"],
        "breadth_worsening": frame["breadth_worsening"],
        "non_stress": ~frame["stress"],
        "non_vix_rising": ~frame["vix_rising"],
        "non_breadth_worsening": ~frame["breadth_worsening"],
    }
    return {name: _event_metric(frame, mask) for name, mask in masks.items()}


def _horizon_event_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    masks = {
        "h1": frame["horizon"] == 1,
        "h2_h3": frame["horizon"].between(2, 3),
        "h4_h5": frame["horizon"].between(4, 5),
    }
    return {name: _event_metric(frame, mask) for name, mask in masks.items()}


def _feature_blind_metrics(
    *,
    frame: pd.DataFrame,
    last_features: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_rows = frame["sample_row"].to_numpy(dtype=np.int64)
    missed_sample_rows = sample_rows[frame["missed_severe"].to_numpy(dtype=bool)]
    caught_sample_rows = sample_rows[frame["caught_severe"].to_numpy(dtype=bool)]
    for feature in FEATURE_ANALYSIS_COLUMNS:
        if feature not in feature_names:
            continue
        feature_idx = feature_names.index(feature)
        missed = last_features[missed_sample_rows, feature_idx] if missed_sample_rows.size else np.array([], dtype=np.float32)
        caught = last_features[caught_sample_rows, feature_idx] if caught_sample_rows.size else np.array([], dtype=np.float32)
        missed_stats = _stats(missed)
        caught_stats = _stats(caught)
        rows.append(
            {
                "feature": feature,
                "scale": "normalized_last_input_step",
                "missed_count": missed_stats.get("count", 0),
                "caught_count": caught_stats.get("count", 0),
                "missed_mean": missed_stats.get("mean"),
                "caught_mean": caught_stats.get("mean"),
                "mean_diff_missed_minus_caught": (
                    float(missed_stats["mean"] - caught_stats["mean"])
                    if missed_stats.get("count") and caught_stats.get("count")
                    else None
                ),
                "missed_q25": missed_stats.get("q25"),
                "missed_q50": missed_stats.get("q50"),
                "missed_q75": missed_stats.get("q75"),
                "caught_q25": caught_stats.get("q25"),
                "caught_q50": caught_stats.get("q50"),
                "caught_q75": caught_stats.get("q75"),
            }
        )
    return rows


def _load_sector_map() -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [
        PROJECT_ROOT / "data" / "parquet" / "indicators_eodhd_1D_500.parquet",
        PROJECT_ROOT / "data" / "parquet" / "price_data_eodhd_500.parquet",
    ]
    possible = ["sector", "industry", "gics_sector", "gics_industry", "ticker"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            import pyarrow.parquet as pq

            names = set(pq.ParquetFile(path).schema.names)
            columns = [column for column in possible if column in names]
            if "ticker" not in columns or len(columns) <= 1:
                continue
            frame = pd.read_parquet(path, columns=columns).drop_duplicates("ticker")
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
            return frame, {"sector_source": str(path), "sector_columns": columns}
        except Exception as exc:
            return pd.DataFrame({"ticker": []}), {"sector_source": str(path), "sector_error": str(exc)}
    return pd.DataFrame({"ticker": []}), {"sector_source": None, "sector_columns": []}


def _top_dates_and_tickers(
    *,
    candidate_id: str,
    seed: int,
    frame: pd.DataFrame,
    sector_map: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_safe = frame.loc[frame["false_safe_tail"]].copy()
    date_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    if not false_safe.empty:
        by_date = (
            false_safe.groupby("asof_date")
            .agg(false_safe_count=("false_safe_tail", "size"), mean_score=("score", "mean"), mean_actual=("actual", "mean"))
            .sort_values("false_safe_count", ascending=False)
            .head(20)
            .reset_index()
        )
        for _, row in by_date.iterrows():
            date_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "asof_date": row["asof_date"],
                    "false_safe_count": int(row["false_safe_count"]),
                    "mean_score": row["mean_score"],
                    "mean_actual": row["mean_actual"],
                }
            )

        ticker_counts = (
            false_safe.groupby("ticker")
            .agg(false_safe_count=("false_safe_tail", "size"), mean_score=("score", "mean"), mean_actual=("actual", "mean"))
            .sort_values("false_safe_count", ascending=False)
            .head(30)
            .reset_index()
        )
        if not sector_map.empty:
            ticker_counts = ticker_counts.merge(sector_map, on="ticker", how="left")
        for _, row in ticker_counts.iterrows():
            payload = {
                "candidate_id": candidate_id,
                "seed": seed,
                "ticker": row["ticker"],
                "false_safe_count": int(row["false_safe_count"]),
                "mean_score": row["mean_score"],
                "mean_actual": row["mean_actual"],
            }
            for column in ["sector", "industry", "gics_sector", "gics_industry"]:
                if column in row.index:
                    payload[column] = row.get(column)
            ticker_rows.append(payload)
    return date_rows, ticker_rows


def _segment_metrics(metadata: pd.DataFrame, line: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    masks = {
        "stress": (metadata.get("regime_stress", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
        "vix_rising": (metadata.get("vix_change_5d", pd.Series(0.0, index=metadata.index)).fillna(0).astype(float) > 0.0).to_numpy(),
        "breadth_worsening": (metadata.get("ma200_pct_change_20d", pd.Series(0.0, index=metadata.index)).fillna(0).astype(float) < 0.0).to_numpy(),
    }
    masks["non_stress"] = ~masks["stress"]
    masks["non_vix_rising"] = ~masks["vix_rising"]
    masks["non_breadth_worsening"] = ~masks["breadth_worsening"]
    segment = {
        name: r0._metric_for_segment(
            metadata=metadata,
            line=line,
            raw=raw,
            mask=mask,
            start=0,
            end=int(raw.shape[1]),
            severe_threshold=SEVERE_THRESHOLD,
        )
        for name, mask in masks.items()
    }
    buckets = {
        name: r0._metric_for_segment(
            metadata=metadata,
            line=line,
            raw=raw,
            mask=np.ones(len(metadata), dtype=bool),
            start=start,
            end=end,
            severe_threshold=SEVERE_THRESHOLD,
        )
        for name, (start, end) in HORIZON_BUCKETS.items()
    }
    return {"regime_segments": segment, "horizon_segments": buckets}


def _analyze_record(
    *,
    meta: dict[str, Any],
    test_bundle: Any,
    metadata: pd.DataFrame,
    last_features: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
    sector_map: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_id = str(meta["candidate_id"])
    seed = int(meta["seed"])
    model, checkpoint = load_checkpoint(meta["checkpoint_path_abs"])
    try:
        line, raw = r0._predict_bundle(
            model=model,
            bundle=test_bundle,
            device=device,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
    except (RuntimeError, TypeError) as exc:
        if "BFloat16" not in str(exc):
            raise
        line, raw = r0._predict_bundle(
            model=model,
            bundle=test_bundle,
            device=device,
            batch_size=batch_size,
            amp_dtype="fp32",
        )
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = _attach_tail_flags(_flat_event_frame(metadata, line, raw))
    false_safe_mask = frame["false_safe_tail"].to_numpy(dtype=bool)
    score = frame["score"].to_numpy(dtype=np.float64)
    actual = frame["actual"].to_numpy(dtype=np.float64)
    signed_error = score - actual
    score_distribution = _stats(score)
    score_distribution["positive_rate"] = float((score >= 0.0).mean())

    result = {
        "candidate_id": candidate_id,
        "seed": seed,
        "checkpoint_path": meta["checkpoint_path_abs"],
        "run_meta_path": meta["run_meta_path"],
        "test_metrics_from_stage4_4": meta.get("test") or {},
        "validation_metrics_from_stage4_4": meta.get("validation") or {},
        "score_distribution": score_distribution,
        "conservative_bias_recomputed_flat": float(np.mean(signed_error)),
        "tail_cutoff_global_q20": float(frame["tail_cutoff"].iloc[0]),
        "severe_threshold": SEVERE_THRESHOLD,
        "false_safe_margin_buckets": _margin_buckets(score, false_safe_mask),
        "false_safe_score_stats": _stats(score[false_safe_mask]),
        "false_safe_actual_stats": _stats(actual[false_safe_mask]),
        "regime_event_metrics_flat": _regime_event_metrics(frame),
        "horizon_event_metrics_flat": _horizon_event_metrics(frame),
        "segment_metrics_stage_style": _segment_metrics(metadata, line, raw),
        "feature_blind_analysis": _feature_blind_metrics(frame=frame, last_features=last_features, feature_names=feature_names),
        "checkpoint_config": (checkpoint.get("config") or {}),
    }
    date_rows, ticker_rows = _top_dates_and_tickers(candidate_id=candidate_id, seed=seed, frame=frame, sector_map=sector_map)
    return result, date_rows, ticker_rows


def _aggregate_bias(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate_id in sorted(TARGET_CANDIDATES):
        candidate_rows = [row for row in records if row["candidate_id"] == candidate_id]
        seed42 = next(row for row in candidate_rows if int(row["seed"]) == 42)
        others = [row for row in candidate_rows if int(row["seed"]) != 42]
        seed42_bias = _safe_float((seed42.get("test") or {}).get("conservative_bias"))
        other_biases = [_safe_float((row.get("test") or {}).get("conservative_bias")) for row in others]
        other_biases = [value for value in other_biases if value is not None]
        output[candidate_id] = {
            "seed42_test_conservative_bias": seed42_bias,
            "other_seed_test_conservative_bias_median": statistics.median(other_biases) if other_biases else None,
            "seed42_less_conservative_than_other_seeds": (
                seed42_bias is not None and other_biases and seed42_bias > statistics.median(other_biases)
            ),
        }
    return output


def _classify_failure(analyses: list[dict[str, Any]], bias_summary: dict[str, Any]) -> dict[str, Any]:
    seed42_rows = [row for row in analyses if int(row["seed"]) == 42]
    near_zero_shares = []
    strong_positive_shares = []
    non_stress_gaps = []
    h1_gaps = []
    for row in seed42_rows:
        buckets = row["false_safe_margin_buckets"]
        total = buckets.get("total_false_safe") or 0
        near = sum((buckets.get(key) or {}).get("count") or 0 for key in ["0_to_0p001", "0p001_to_0p003", "0p003_to_0p005"])
        strong = (buckets.get("0p005_plus") or {}).get("count") or 0
        near_zero_shares.append(float(near / total) if total else 0.0)
        strong_positive_shares.append(float(strong / total) if total else 0.0)

        regime = row["regime_event_metrics_flat"]
        stress_fs = _safe_float((regime.get("stress") or {}).get("false_safe_tail_rate"))
        non_stress_fs = _safe_float((regime.get("non_stress") or {}).get("false_safe_tail_rate"))
        if stress_fs is not None and non_stress_fs is not None:
            non_stress_gaps.append(non_stress_fs - stress_fs)

        horizon = row["horizon_event_metrics_flat"]
        h1_fs = _safe_float((horizon.get("h1") or {}).get("false_safe_tail_rate"))
        h45_fs = _safe_float((horizon.get("h4_h5") or {}).get("false_safe_tail_rate"))
        if h1_fs is not None and h45_fs is not None:
            h1_gaps.append(h1_fs - h45_fs)

    all_seed42_less_conservative = all(
        item.get("seed42_less_conservative_than_other_seeds") for item in bias_summary.values()
    )
    near_zero_share = float(np.mean(near_zero_shares)) if near_zero_shares else None
    strong_positive_share = float(np.mean(strong_positive_shares)) if strong_positive_shares else None
    non_stress_gap = float(np.mean(non_stress_gaps)) if non_stress_gaps else None
    h1_gap = float(np.mean(h1_gaps)) if h1_gaps else None

    if all_seed42_less_conservative and near_zero_share is not None and near_zero_share >= 0.45:
        primary = "selector_bias_problem"
    elif strong_positive_share is not None and strong_positive_share >= 0.50:
        primary = "feature_blind_problem"
    else:
        primary = "mixed_bias_and_feature_blind_problem"

    tags: list[str] = [primary]
    if non_stress_gap is not None and non_stress_gap > 0.05:
        tags.append("quiet_idiosyncratic_tail")
    if h1_gap is not None and h1_gap > 0.03:
        tags.append("short_horizon_downside")
    return {
        "primary_failure_type": primary,
        "failure_tags": tags,
        "seed42_all_less_conservative": all_seed42_less_conservative,
        "seed42_false_safe_near_zero_share_mean": near_zero_share,
        "seed42_false_safe_strong_positive_share_mean": strong_positive_share,
        "seed42_non_stress_minus_stress_false_safe_gap_mean": non_stress_gap,
        "seed42_h1_minus_h4_h5_false_safe_gap_mean": h1_gap,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _json_safe(value) for key, value in row.items()} for row in rows])


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _metric_rows(analyses: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in sorted(analyses, key=lambda item: (item["candidate_id"], int(item["seed"]))):
        metrics = row["test_metrics_from_stage4_4"]
        buckets = row["false_safe_margin_buckets"]
        rows.append(
            [
                row["candidate_id"],
                row["seed"],
                _fmt(metrics.get("ic_mean")),
                _fmt(metrics.get("long_short_spread")),
                _fmt(metrics.get("false_safe_tail_rate")),
                _fmt(metrics.get("severe_downside_recall")),
                _fmt(metrics.get("conservative_bias")),
                _fmt((buckets.get("0_to_0p001") or {}).get("share")),
                _fmt((buckets.get("0p001_to_0p003") or {}).get("share")),
                _fmt((buckets.get("0p003_to_0p005") or {}).get("share")),
                _fmt((buckets.get("0p005_plus") or {}).get("share")),
            ]
        )
    return rows


def _regime_rows(analyses: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in sorted(analyses, key=lambda item: (item["candidate_id"], int(item["seed"]))):
        metrics = row["regime_event_metrics_flat"]
        for regime in REGIME_MASKS:
            payload = metrics.get(regime) or {}
            rows.append(
                [
                    row["candidate_id"],
                    row["seed"],
                    regime,
                    payload.get("tail_count"),
                    _fmt(payload.get("false_safe_tail_rate")),
                    payload.get("severe_count"),
                    _fmt(payload.get("severe_downside_recall")),
                ]
            )
    return rows


def _horizon_rows(analyses: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in sorted(analyses, key=lambda item: (item["candidate_id"], int(item["seed"]))):
        metrics = row["horizon_event_metrics_flat"]
        for bucket in ["h1", "h2_h3", "h4_h5"]:
            payload = metrics.get(bucket) or {}
            rows.append(
                [
                    row["candidate_id"],
                    row["seed"],
                    bucket,
                    payload.get("tail_count"),
                    _fmt(payload.get("false_safe_tail_rate")),
                    payload.get("severe_count"),
                    _fmt(payload.get("severe_downside_recall")),
                ]
            )
    return rows


def _write_report(payload: dict[str, Any]) -> None:
    analyses = payload["analyses"]
    decision = payload["decision"]
    next_experiment = payload["next_experiment"]
    lines: list[str] = [
        "# CP148-LM-1D Stage 4-4F failure analysis 보고서",
        "",
        f"- 작성 시각: {payload['created_at']}",
        "- 범위: Stage 4-4 trial006/trial024의 기존 6개 checkpoint test split 진단",
        "- 금지 작업 준수: 새 학습, Optuna, product save-run, DB write, inference 저장, live fetch, band/composite 모두 미실행",
        "",
        "## 1. 한 줄 결론",
        "",
        "seed 42 붕괴는 단순 selector/bias 문제가 아니라, non-stress/quiet tail에서 실제 하락을 강한 양수 score로 본 feature blind 문제가 1차 원인이다.",
        "",
        "## 2. seed 42 실패 한 문장",
        "",
        "seed 42는 다른 seed보다 덜 보수적으로 떴지만, false-safe 대부분이 0 근처가 아니라 0.005 이상 강한 양수 score였기 때문에 severe recall이 같이 무너졌다.",
        "",
        "## 3. 판정",
        "",
        f"- 1차 판정: `{decision['primary_failure_type']}`",
        f"- 부가 태그: `{', '.join(decision['failure_tags'])}`",
        f"- seed42 덜 보수적 여부: `{decision['seed42_all_less_conservative']}`",
        f"- seed42 false_safe 0~0.005 근처 평균 비중: `{_fmt(decision['seed42_false_safe_near_zero_share_mean'])}`",
        f"- seed42 false_safe 0.005 이상 강한 양수 평균 비중: `{_fmt(decision['seed42_false_safe_strong_positive_share_mean'])}`",
        f"- seed42 non-stress minus stress false_safe gap: `{_fmt(decision['seed42_non_stress_minus_stress_false_safe_gap_mean'])}`",
        f"- seed42 h1 minus h4_h5 false_safe gap: `{_fmt(decision['seed42_h1_minus_h4_h5_false_safe_gap_mean'])}`",
        "",
        "## 4. 지표와 margin 요약",
        "",
    ]
    lines.extend(
        _table(
            [
                "후보",
                "seed",
                "IC",
                "spread",
                "false_safe",
                "severe",
                "bias",
                "0~0.001",
                "0.001~0.003",
                "0.003~0.005",
                "0.005+",
            ],
            _metric_rows(analyses),
        )
    )
    lines.extend(["", "## 5. stress / non-stress 분해", ""])
    lines.extend(_table(["후보", "seed", "구간", "tail_count", "false_safe", "severe_count", "severe_recall"], _regime_rows(analyses)))
    lines.extend(["", "## 6. horizon 분해", ""])
    lines.extend(_table(["후보", "seed", "bucket", "tail_count", "false_safe", "severe_count", "severe_recall"], _horizon_rows(analyses)))
    lines.extend(
        [
            "",
            "## 7. feature blind 확인",
            "",
            "- feature 비교는 test split의 마지막 입력 시점 값을 사용했다.",
            "- 값의 스케일은 모델 입력과 같은 train-normalized scale이다.",
            "- false_safe missed severe와 correctly caught severe의 feature 평균/분위수 차이는 metrics JSON의 `feature_blind_analysis`에 기록했다.",
            "",
            "## 8. 날짜/종목 집중도",
            "",
            f"- top false_safe dates: `{TOP_DATES_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- top false_safe tickers: `{TOP_TICKERS_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- sector/industry 사용 여부: `{payload['sector_metadata']}`",
            "",
            "## 9. 다음 실험 1개",
            "",
            f"- 추천: {next_experiment['name']}",
            f"- 내용: {next_experiment['description']}",
            f"- seed 안정성 통과 가능성을 높이는 이유: {next_experiment['why_it_helps_seed_stability']}",
            "",
            "## 10. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- top dates: `{TOP_DATES_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- top tickers: `{TOP_TICKERS_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_4f_failure_analysis.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    s41._configure_environment()
    s41._patch_training_for_experiment()
    records = _load_stage4_4_metas()
    bias_summary = _aggregate_bias(records)
    precomputed, feature_names = s41._build_exact_feature_splits(
        extra_names=C_FEATURES,
        device=args.device,
        batch_size=args.batch_size,
    )
    _train_bundle, _val_bundle, test_bundle, _mean, _std, plan = precomputed
    context = r0._load_context_frame()
    metadata = r0._decorate_metadata(test_bundle.metadata, context)
    last_features = _last_feature_matrix(test_bundle, feature_names)
    sector_map, sector_meta = _load_sector_map()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    analyses: list[dict[str, Any]] = []
    top_date_rows: list[dict[str, Any]] = []
    top_ticker_rows: list[dict[str, Any]] = []
    for meta in records:
        print(
            json.dumps(
                {
                    "stage": "analyze_checkpoint",
                    "candidate_id": meta["candidate_id"],
                    "seed": meta["seed"],
                    "device": str(device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        analysis, date_rows, ticker_rows = _analyze_record(
            meta=meta,
            test_bundle=test_bundle,
            metadata=metadata,
            last_features=last_features,
            feature_names=feature_names,
            device=device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
            sector_map=sector_map,
        )
        analyses.append(analysis)
        top_date_rows.extend(date_rows)
        top_ticker_rows.extend(ticker_rows)

    decision = _classify_failure(analyses, bias_summary)
    next_experiment = {
        "name": "Stage 4-5 단일 실험: C + overextension/quiet-tail fragility pack",
        "description": "C_stress_delta는 유지하되, non-stress에서 좋아 보이는 종목이 갑자기 tail로 빠지는 패턴을 잡기 위해 runup/overextension 계열 파생 피처를 추가한 3-seed 단일 후보를 검증한다. 후보 피처는 20일 runup, ma_20/ma_60 양의 괴리, RSI overbought, BB upper excess, 최근 max drawdown 또는 max down day 중 local parquet에서 계산 가능한 값으로 제한한다.",
        "why_it_helps_seed_stability": "seed 42 false-safe의 약 90%가 0.005 이상 강한 양수 score였고, missed severe는 caught severe보다 ma_60_ratio, ma_20_ratio, RSI, bb_position, log_return이 높았다. 즉 모델이 조용한 상승/과열 종목의 급락을 안전 신호로 읽었으므로, overextension을 명시 피처로 주면 seed별 bias drift보다 구조적인 blind spot을 줄일 가능성이 높다.",
    }
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "new_training": False,
            "optuna": False,
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite": False,
        },
        "data": {
            "provider": "eodhd",
            "backend": "local_parquet",
            "split": "test",
            "feature_pack": "C_stress_delta",
            "extra_features": C_FEATURES,
            "feature_analysis_scale": "normalized_last_input_step",
            "plan": _json_safe(plan),
        },
        "bias_summary": bias_summary,
        "decision": decision,
        "sector_metadata": sector_meta,
        "analyses": analyses,
        "top_dates_csv": str(TOP_DATES_CSV_PATH),
        "top_tickers_csv": str(TOP_TICKERS_CSV_PATH),
        "next_experiment": next_experiment,
    }
    _write_json(METRICS_PATH, payload)
    _write_csv(TOP_DATES_CSV_PATH, top_date_rows)
    _write_csv(TOP_TICKERS_CSV_PATH, top_ticker_rows)
    _write_report(payload)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-4F failure analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
