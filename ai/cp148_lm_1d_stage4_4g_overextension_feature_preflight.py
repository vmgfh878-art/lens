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
from ai import cp148_lm_1d_stage4_4f_failure_analysis as f4  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4g_overextension_feature_preflight_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4g_overextension_feature_preflight_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4g_overextension_feature_preflight_summary.csv"

NEW_FEATURES = [
    "runup_20d",
    "runup_20d_xs_z",
    "runup_20d_xs_rank",
    "ma60_extension_pos",
    "ma20_extension_pos",
    "max_down_day_20d",
    "pullback_from_20d_high",
]

CORRELATION_FEATURES = [
    "ma_20_ratio",
    "ma_60_ratio",
    "rsi",
    "bb_position",
    "log_return",
    "atr_ratio",
]

D_LIKE_FEATURES = {"max_down_day_20d", "pullback_from_20d_high"}


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
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "q75": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
    }


def _load_overextension_feature_frame() -> pd.DataFrame:
    alias = r0._snapshot_alias_dir()
    price_path = alias / "price_data_eodhd.parquet"
    indicator_path = alias / "indicators_eodhd_1D.parquet"

    prices = pd.read_parquet(price_path, columns=["ticker", "date", "close"])
    indicators = pd.read_parquet(
        indicator_path,
        columns=[
            "ticker",
            "date",
            "ma_20_ratio",
            "ma_60_ratio",
            "rsi",
            "bb_position",
            "log_return",
            "atr_ratio",
        ],
    )
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    indicators["date"] = pd.to_datetime(indicators["date"])

    frames: list[pd.DataFrame] = []
    for ticker, frame in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        local = frame.copy()
        close = local["close"].astype(float)
        returns = close.pct_change()
        rolling_high_20 = close.rolling(20, min_periods=1).max()
        rolling_min_return = returns.rolling(20, min_periods=2).min()
        local["runup_20d"] = close / close.shift(20) - 1.0
        local["max_down_day_20d"] = np.maximum(-rolling_min_return, 0.0)
        local["pullback_from_20d_high"] = np.maximum(1.0 - close / rolling_high_20, 0.0)
        local["ticker"] = ticker
        frames.append(local[["ticker", "date", "runup_20d", "max_down_day_20d", "pullback_from_20d_high"]])
    derived = pd.concat(frames, ignore_index=True)

    feature_frame = indicators.merge(derived, on=["ticker", "date"], how="left")
    feature_frame["ma60_extension_pos"] = np.maximum(feature_frame["ma_60_ratio"].astype(float), 0.0)
    feature_frame["ma20_extension_pos"] = np.maximum(feature_frame["ma_20_ratio"].astype(float), 0.0)
    grouped = feature_frame.groupby("date", sort=False)["runup_20d"]
    date_mean = grouped.transform("mean")
    date_std = grouped.transform("std").replace(0.0, np.nan)
    feature_frame["runup_20d_xs_z"] = (feature_frame["runup_20d"] - date_mean) / date_std
    feature_frame["runup_20d_xs_rank"] = grouped.rank(pct=True)
    for column in [*NEW_FEATURES, *CORRELATION_FEATURES]:
        feature_frame[column] = feature_frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    feature_frame["asof_date"] = feature_frame["date"].dt.strftime("%Y-%m-%d")
    keep = ["ticker", "asof_date", *NEW_FEATURES, *CORRELATION_FEATURES]
    return feature_frame[keep].drop_duplicates(["ticker", "asof_date"]).set_index(["ticker", "asof_date"])


def _feature_matrix_for_metadata(metadata: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "ticker": metadata["ticker"].astype(str).str.upper().to_numpy(),
                "asof_date": pd.to_datetime(metadata["asof_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            }
        )
    )
    columns = [*NEW_FEATURES, *CORRELATION_FEATURES]
    values = feature_frame.reindex(index)[columns].fillna(0.0).reset_index(drop=True)
    return values


def _predict_split(
    *,
    checkpoint_path: str,
    bundle: Any,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    model, _checkpoint = load_checkpoint(checkpoint_path)
    try:
        return r0._predict_bundle(
            model=model,
            bundle=bundle,
            device=device,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
    except (RuntimeError, TypeError) as exc:
        if "BFloat16" not in str(exc):
            raise
        return r0._predict_bundle(
            model=model,
            bundle=bundle,
            device=device,
            batch_size=batch_size,
            amp_dtype="fp32",
        )
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _event_feature_stats(frame: pd.DataFrame, sample_features: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_rows = frame["sample_row"].to_numpy(dtype=np.int64)
    missed_rows = sample_rows[frame["missed_severe"].to_numpy(dtype=bool)]
    caught_rows = sample_rows[frame["caught_severe"].to_numpy(dtype=bool)]
    for feature in NEW_FEATURES:
        missed = sample_features.iloc[missed_rows][feature].to_numpy(dtype=np.float64) if missed_rows.size else np.array([])
        caught = sample_features.iloc[caught_rows][feature].to_numpy(dtype=np.float64) if caught_rows.size else np.array([])
        missed_stats = _stats(missed)
        caught_stats = _stats(caught)
        mean_diff = (
            float(missed_stats["mean"] - caught_stats["mean"])
            if missed_stats.get("count") and caught_stats.get("count")
            else None
        )
        median_diff = (
            float(missed_stats["median"] - caught_stats["median"])
            if missed_stats.get("count") and caught_stats.get("count")
            else None
        )
        q75_diff = (
            float(missed_stats["q75"] - caught_stats["q75"])
            if missed_stats.get("count") and caught_stats.get("count")
            else None
        )
        rows.append(
            {
                "feature": feature,
                "missed_count": missed_stats.get("count"),
                "caught_count": caught_stats.get("count"),
                "missed_mean": missed_stats.get("mean"),
                "missed_median": missed_stats.get("median"),
                "missed_q75": missed_stats.get("q75"),
                "caught_mean": caught_stats.get("mean"),
                "caught_median": caught_stats.get("median"),
                "caught_q75": caught_stats.get("q75"),
                "mean_diff_missed_minus_caught": mean_diff,
                "median_diff_missed_minus_caught": median_diff,
                "q75_diff_missed_minus_caught": q75_diff,
                "positive_direction": bool(mean_diff is not None and mean_diff > 0.0),
            }
        )
    return rows


def _correlations(sample_features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for feature in NEW_FEATURES:
        corr_payload: dict[str, Any] = {}
        values = sample_features[feature].astype(float)
        for existing in CORRELATION_FEATURES:
            corr = values.corr(sample_features[existing].astype(float))
            corr_payload[existing] = float(corr) if pd.notna(corr) else None
        finite_corrs = [abs(value) for value in corr_payload.values() if value is not None and math.isfinite(value)]
        corr_payload["max_abs_corr"] = max(finite_corrs) if finite_corrs else None
        corr_payload["mean_abs_corr"] = float(np.mean(finite_corrs)) if finite_corrs else None
        output[feature] = corr_payload
    return output


def _analyze_split(
    *,
    meta: dict[str, Any],
    split_name: str,
    bundle: Any,
    metadata: pd.DataFrame,
    sample_features: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> dict[str, Any]:
    line, raw = _predict_split(
        checkpoint_path=meta["checkpoint_path_abs"],
        bundle=bundle,
        device=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    event_frame = f4._attach_tail_flags(f4._flat_event_frame(metadata, line, raw))
    feature_stats = _event_feature_stats(event_frame, sample_features)
    severe_count = int(event_frame["severe_actual"].sum())
    missed_count = int(event_frame["missed_severe"].sum())
    caught_count = int(event_frame["caught_severe"].sum())
    return {
        "candidate_id": meta["candidate_id"],
        "seed": int(meta["seed"]),
        "split": split_name,
        "severe_count": severe_count,
        "missed_severe_count": missed_count,
        "caught_severe_count": caught_count,
        "missed_severe_rate": float(missed_count / severe_count) if severe_count else None,
        "feature_stats": feature_stats,
    }


def _aggregate_feature_results(records: list[dict[str, Any]], correlations_by_split: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in NEW_FEATURES:
        feature_rows: list[dict[str, Any]] = []
        for record in records:
            for item in record["feature_stats"]:
                if item["feature"] == feature:
                    row = {**record, **item}
                    row.pop("feature_stats", None)
                    feature_rows.append(row)
                    break
        diffs = [_safe_float(row.get("mean_diff_missed_minus_caught")) for row in feature_rows]
        valid_diffs = [value for value in diffs if value is not None]
        positive_count = sum(1 for value in valid_diffs if value > 0.0)
        split_positive: dict[str, float | None] = {}
        for split in ["validation", "test"]:
            split_values = [
                _safe_float(row.get("mean_diff_missed_minus_caught"))
                for row in feature_rows
                if row.get("split") == split
            ]
            split_values = [value for value in split_values if value is not None]
            split_positive[split] = (
                float(sum(1 for value in split_values if value > 0.0) / len(split_values)) if split_values else None
            )
        seed_positive: dict[str, float | None] = {}
        for seed in [42, 7, 123]:
            seed_values = [
                _safe_float(row.get("mean_diff_missed_minus_caught"))
                for row in feature_rows
                if int(row.get("seed")) == seed
            ]
            seed_values = [value for value in seed_values if value is not None]
            seed_positive[str(seed)] = (
                float(sum(1 for value in seed_values if value > 0.0) / len(seed_values)) if seed_values else None
            )
        candidate_positive: dict[str, float | None] = {}
        for candidate_id in ["trial006_c_balanced", "trial024_c_risk"]:
            candidate_values = [
                _safe_float(row.get("mean_diff_missed_minus_caught"))
                for row in feature_rows
                if row.get("candidate_id") == candidate_id
            ]
            candidate_values = [value for value in candidate_values if value is not None]
            candidate_positive[candidate_id] = (
                float(sum(1 for value in candidate_values if value > 0.0) / len(candidate_values)) if candidate_values else None
            )
        corr_values = []
        corr_maxes = []
        corr_details: dict[str, Any] = {}
        for split, corr_payload in correlations_by_split.items():
            feature_corr = corr_payload.get(feature) or {}
            corr_details[split] = feature_corr
            max_corr = _safe_float(feature_corr.get("max_abs_corr"))
            mean_corr = _safe_float(feature_corr.get("mean_abs_corr"))
            if max_corr is not None:
                corr_maxes.append(max_corr)
            if mean_corr is not None:
                corr_values.append(mean_corr)
        positive_share = float(positive_count / len(valid_diffs)) if valid_diffs else None
        median_diff = statistics.median(valid_diffs) if valid_diffs else None
        mean_diff = float(np.mean(valid_diffs)) if valid_diffs else None
        max_abs_corr = max(corr_maxes) if corr_maxes else None
        mean_abs_corr = float(np.mean(corr_values)) if corr_values else None
        status, reason = _classify_feature(
            feature=feature,
            positive_share=positive_share,
            validation_share=split_positive.get("validation"),
            test_share=split_positive.get("test"),
            median_diff=median_diff,
            max_abs_corr=max_abs_corr,
        )
        rows.append(
            {
                "feature": feature,
                "status": status,
                "reason": reason,
                "valid_comparisons": len(valid_diffs),
                "positive_direction_share": positive_share,
                "validation_positive_share": split_positive.get("validation"),
                "test_positive_share": split_positive.get("test"),
                "seed42_positive_share": seed_positive.get("42"),
                "seed7_positive_share": seed_positive.get("7"),
                "seed123_positive_share": seed_positive.get("123"),
                "trial006_positive_share": candidate_positive.get("trial006_c_balanced"),
                "trial024_positive_share": candidate_positive.get("trial024_c_risk"),
                "mean_diff_missed_minus_caught": mean_diff,
                "median_diff_missed_minus_caught": median_diff,
                "max_abs_corr_existing": max_abs_corr,
                "mean_abs_corr_existing": mean_abs_corr,
                "d_like_risk": feature in D_LIKE_FEATURES,
                "correlations": corr_details,
                "record_rows": feature_rows,
            }
        )
    return _apply_family_constraints(rows)


def _apply_family_constraints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature = {row["feature"]: row for row in rows}
    xs_z = by_feature.get("runup_20d_xs_z")
    xs_rank = by_feature.get("runup_20d_xs_rank")
    if xs_z and xs_rank and xs_z["status"] == "recommend" and xs_rank["status"] == "recommend":
        xs_rank["status"] = "hold"
        xs_rank["reason"] = "runup_20d_xs_z와 같은 cross-sectional runup 계열이므로 둘 다 넣지 않고 xs_z를 우선함"

    ma60 = by_feature.get("ma60_extension_pos")
    ma20 = by_feature.get("ma20_extension_pos")
    if ma60 and ma20 and ma60["status"] == "recommend" and ma20["status"] == "recommend":
        ma20["status"] = "hold"
        ma20["reason"] = "ma60_extension_pos보다 단기/기존 ma_20_ratio 중복성이 커서 보조 후보로만 둠"
    return rows


def _classify_feature(
    *,
    feature: str,
    positive_share: float | None,
    validation_share: float | None,
    test_share: float | None,
    median_diff: float | None,
    max_abs_corr: float | None,
) -> tuple[str, str]:
    if positive_share is None or median_diff is None:
        return "exclude", "비교 가능한 severe 이벤트가 부족함"
    if positive_share < 0.58 or median_diff <= 0.0:
        return "exclude", "missed severe가 caught severe보다 높다는 방향성이 약하거나 반대임"
    if validation_share is not None and test_share is not None and (validation_share < 0.60 or test_share < 0.60):
        return "hold", "validation/test 중 한쪽 방향성이 약함"
    if max_abs_corr is not None and max_abs_corr >= 0.97:
        return "hold", "기존 feature와 상관이 지나치게 높아 신규 정보가 작을 수 있음"
    if feature in D_LIKE_FEATURES:
        return "hold", "D_stock_fragility 실패 계열과 가까워 과적합 위험이 큼"
    if positive_share >= 0.75 and median_diff > 0.0:
        return "recommend", "seed/candidate/split 전반에서 missed severe가 더 높고 해석이 명확함"
    return "hold", "방향성은 있으나 추천 후보로 보기에는 일관성이 부족함"


def _write_summary_csv(aggregates: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        rows.append(
            {
                "row_type": "aggregate",
                "feature": aggregate["feature"],
                "status": aggregate["status"],
                "reason": aggregate["reason"],
                "valid_comparisons": aggregate["valid_comparisons"],
                "positive_direction_share": aggregate["positive_direction_share"],
                "validation_positive_share": aggregate["validation_positive_share"],
                "test_positive_share": aggregate["test_positive_share"],
                "seed42_positive_share": aggregate["seed42_positive_share"],
                "seed7_positive_share": aggregate["seed7_positive_share"],
                "seed123_positive_share": aggregate["seed123_positive_share"],
                "trial006_positive_share": aggregate["trial006_positive_share"],
                "trial024_positive_share": aggregate["trial024_positive_share"],
                "mean_diff_missed_minus_caught": aggregate["mean_diff_missed_minus_caught"],
                "median_diff_missed_minus_caught": aggregate["median_diff_missed_minus_caught"],
                "max_abs_corr_existing": aggregate["max_abs_corr_existing"],
                "mean_abs_corr_existing": aggregate["mean_abs_corr_existing"],
                "d_like_risk": aggregate["d_like_risk"],
            }
        )
        for row in aggregate["record_rows"]:
            rows.append(
                {
                    "row_type": "record",
                    "feature": aggregate["feature"],
                    "status": "",
                    "reason": "",
                    "candidate_id": row.get("candidate_id"),
                    "seed": row.get("seed"),
                    "split": row.get("split"),
                    "missed_count": row.get("missed_count"),
                    "caught_count": row.get("caught_count"),
                    "missed_mean": row.get("missed_mean"),
                    "missed_median": row.get("missed_median"),
                    "missed_q75": row.get("missed_q75"),
                    "caught_mean": row.get("caught_mean"),
                    "caught_median": row.get("caught_median"),
                    "caught_q75": row.get("caught_q75"),
                    "mean_diff_missed_minus_caught": row.get("mean_diff_missed_minus_caught"),
                    "median_diff_missed_minus_caught": row.get("median_diff_missed_minus_caught"),
                    "q75_diff_missed_minus_caught": row.get("q75_diff_missed_minus_caught"),
                    "positive_direction": row.get("positive_direction"),
                }
            )
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _json_safe(value) for key, value in row.items()} for row in rows])


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _recommended_features(aggregates: list[dict[str, Any]]) -> list[str]:
    priority = {
        "runup_20d_xs_z": 0,
        "runup_20d": 1,
        "ma60_extension_pos": 2,
        "runup_20d_xs_rank": 3,
        "ma20_extension_pos": 4,
        "max_down_day_20d": 5,
        "pullback_from_20d_high": 6,
    }
    recommended = [
        row["feature"]
        for row in sorted(
            [item for item in aggregates if item["status"] == "recommend"],
            key=lambda item: priority.get(item["feature"], 99),
        )
    ]
    if len(recommended) >= 3:
        return recommended[:3]
    ranked = sorted(
        [row for row in aggregates if row["feature"] not in recommended and row["status"] == "hold"],
        key=lambda row: (
            -(row.get("positive_direction_share") or 0.0),
            -(row.get("median_diff_missed_minus_caught") or 0.0),
            row.get("max_abs_corr_existing") or 999.0,
        ),
    )
    return [*recommended, *[row["feature"] for row in ranked[: max(0, 3 - len(recommended))]]]


def _write_report(payload: dict[str, Any]) -> None:
    aggregates = payload["feature_aggregates"]
    recommended = payload["recommended_features"]
    hold = [row for row in aggregates if row["status"] == "hold"]
    excluded = [row for row in aggregates if row["status"] == "exclude"]
    rows = [
        [
            row["feature"],
            row["status"],
            _fmt(row["positive_direction_share"]),
            _fmt(row["validation_positive_share"]),
            _fmt(row["test_positive_share"]),
            _fmt(row["median_diff_missed_minus_caught"]),
            _fmt(row["max_abs_corr_existing"]),
            "Y" if row["d_like_risk"] else "N",
            row["reason"],
        ]
        for row in aggregates
    ]
    lines: list[str] = [
        "# CP148-LM-1D Stage 4-4G overextension feature preflight 보고서",
        "",
        f"- 작성 시각: {payload['created_at']}",
        "- 범위: Stage 4-4 기존 6개 checkpoint, validation/test split, forward-only 진단",
        "- 금지 작업 준수: 새 학습, Optuna, product save-run, DB write, inference 저장, live fetch, band/composite, core feature contract 변경 모두 미실행",
        "",
        "## 1. 핵심 결론",
        "",
        f"다음 학습 실험에 넣을 후보는 `{', '.join(recommended)}`로 제한한다. 이 후보들은 missed severe가 caught severe보다 높게 나타나는 방향성이 상대적으로 일관적이며, Stage 4-4F의 quiet-tail/overextension blind spot을 직접 겨냥한다.",
        "",
        "## 2. 후보 피처 판정표",
        "",
    ]
    lines.extend(
        _table(
            [
                "feature",
                "판정",
                "방향성",
                "val방향",
                "test방향",
                "median diff",
                "max corr",
                "D계열",
                "이유",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            "## 3. 추천 피처 2~3개",
            "",
            *[f"- `{feature}`" for feature in recommended],
            "",
            "## 4. 보류/제외 피처",
            "",
            *[f"- 보류 `{row['feature']}`: {row['reason']}" for row in hold],
            *[f"- 제외 `{row['feature']}`: {row['reason']}" for row in excluded],
            "",
            "## 5. 과적합 위험 평가",
            "",
            "- `max_down_day_20d`와 `pullback_from_20d_high`는 D_stock_fragility와 가까운 계열이라, 분리력이 있어도 바로 주력 pack에 넣기보다 보류한다.",
            "- `runup_20d_xs_z`와 `runup_20d_xs_rank`는 같은 원천의 상대 상승 피처라 둘 다 넣으면 중복될 수 있다. 둘 중 하나만 선택한다.",
            "- `ma20_extension_pos`와 `ma60_extension_pos`는 기존 ma ratio의 양수 부분만 분리한 값이라 상관이 높으면 설명력 추가가 제한된다.",
            "",
            "## 6. 다음 feature pack 제안",
            "",
            f"- 기준: C_stress_delta + `{', '.join(recommended)}`",
            "- beta=2.0, line_gate 의미, 모델 구조는 유지한다.",
            "- 3-seed 단일 후보로 먼저 확인하고, product save-run 표현은 쓰지 않는다.",
            "",
            "## 7. 학습 실행 판단 근거",
            "",
            "이 preflight에서 추천 후보가 2개 이상 나오면 다음 단계 학습을 돌릴 근거가 있다. 추천 후보가 0~1개면 feature blind를 피처로 해결하기 어렵다고 보고 selector 또는 horizon/target 재검토가 낫다.",
            "",
            "## 8. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_4g_overextension_feature_preflight.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    s41._configure_environment()
    s41._patch_training_for_experiment()
    metas = f4._load_stage4_4_metas()
    precomputed, _feature_names = s41._build_exact_feature_splits(
        extra_names=f4.C_FEATURES,
        device=args.device,
        batch_size=args.batch_size,
    )
    _train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    context = r0._load_context_frame()
    feature_frame = _load_overextension_feature_frame()
    split_payloads = {
        "validation": {
            "bundle": val_bundle,
            "metadata": r0._decorate_metadata(val_bundle.metadata, context),
        },
        "test": {
            "bundle": test_bundle,
            "metadata": r0._decorate_metadata(test_bundle.metadata, context),
        },
    }
    correlations_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, split_data in split_payloads.items():
        sample_features = _feature_matrix_for_metadata(split_data["metadata"], feature_frame)
        split_data["sample_features"] = sample_features
        correlations_by_split[split_name] = _correlations(sample_features)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    records: list[dict[str, Any]] = []
    for meta in metas:
        for split_name, split_data in split_payloads.items():
            print(
                json.dumps(
                    {
                        "stage": "overextension_preflight",
                        "candidate_id": meta["candidate_id"],
                        "seed": meta["seed"],
                        "split": split_name,
                        "device": str(device),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            records.append(
                _analyze_split(
                    meta=meta,
                    split_name=split_name,
                    bundle=split_data["bundle"],
                    metadata=split_data["metadata"],
                    sample_features=split_data["sample_features"],
                    device=device,
                    batch_size=args.batch_size,
                    amp_dtype=args.amp_dtype,
                )
            )

    aggregates = _aggregate_feature_results(records, correlations_by_split)
    recommended = _recommended_features(aggregates)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "new_training": False,
            "optuna": False,
            "seed_retraining": False,
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite": False,
            "core_feature_contract_changed": False,
            "model_structure_changed": False,
        },
        "data": {
            "provider": "eodhd",
            "backend": "local_parquet",
            "splits": ["validation", "test"],
            "base_feature_pack": "C_stress_delta",
            "candidate_features": NEW_FEATURES,
            "correlation_features": CORRELATION_FEATURES,
            "severe_threshold": f4.SEVERE_THRESHOLD,
            "plan": _json_safe(plan),
        },
        "split_records": records,
        "feature_aggregates": aggregates,
        "recommended_features": recommended,
        "proposed_next_feature_pack": [*f4.C_FEATURES, *recommended],
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(aggregates)
    _write_report(payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "metrics_path": str(METRICS_PATH),
                "summary_path": str(SUMMARY_CSV_PATH),
                "report_path": str(REPORT_PATH),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-4G overextension feature preflight")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
