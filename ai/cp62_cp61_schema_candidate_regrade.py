from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINE_METRIC_KEYS = (
    "spearman_ic",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_mean",
    "spread_std",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "mae",
    "smape",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
)

BAND_METRIC_KEYS = (
    "nominal_coverage",
    "empirical_coverage",
    "coverage_error",
    "coverage_abs_error",
    "empirical_q_low",
    "empirical_q_high",
    "lower_breach_rate",
    "upper_breach_rate",
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
)

LEGACY_OVERLAY_NAMES = (
    "line_band_composite",
    "risk_first_upper_buffer",
    "risk_first_lower_preserve",
    "include_line_clamp",
    "line_inside_band",
    "CP46 upper buffer",
)

LINE_TARGET_NAMES = {
    "h5_longer_context_seq252_p32_s16",
    "h5_baseline_seq252_p16_s8",
    "h5_dense_overlap_seq252_p16_s4",
}

BAND_TARGET_TERMS = (
    "s60_q15_b2_direct",
    "s60_q15_b2_direct_188",
    "s60_q20_b2_direct",
    "tide_param_scalar_width",
    "tide_direct_original",
    "q25",
    "q20",
    "q15",
)

CORE_BAND_COMPARE = (
    ("coverage_abs_error", "lower"),
    ("asymmetric_interval_score", "lower"),
    ("band_width_ic", "higher"),
    ("downside_width_ic", "higher"),
)


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    resolved = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_source(metrics: dict[str, Any], layer: str) -> dict[str, Any]:
    nested = metrics.get(layer)
    if isinstance(nested, dict):
        merged = dict(metrics)
        merged.update(nested)
        return merged
    return metrics


def _subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in keys}


def _normalize_line_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    source = _metric_source(metrics or {}, "line_metrics")
    normalized = _subset(source, LINE_METRIC_KEYS)
    if normalized.get("spearman_ic") is None and normalized.get("ic_mean") is not None:
        normalized["spearman_ic"] = normalized["ic_mean"]
    if normalized.get("ic_mean") is None and normalized.get("spearman_ic") is not None:
        normalized["ic_mean"] = normalized["spearman_ic"]
    if normalized.get("long_short_spread") is None and normalized.get("spread_mean") is not None:
        normalized["long_short_spread"] = normalized["spread_mean"]
    if normalized.get("spread_mean") is None and normalized.get("long_short_spread") is not None:
        normalized["spread_mean"] = normalized["long_short_spread"]
    return normalized


def _normalize_band_metrics(metrics: dict[str, Any], *, q_low: float | None, q_high: float | None) -> dict[str, Any]:
    source = _metric_source(metrics or {}, "band_metrics")
    normalized = _subset(source, BAND_METRIC_KEYS)
    if normalized.get("asymmetric_interval_score") is None and source.get("interval_score") is not None:
        normalized["asymmetric_interval_score"] = source.get("interval_score")
    if normalized.get("nominal_coverage") is None and q_low is not None and q_high is not None:
        normalized["nominal_coverage"] = float(q_high) - float(q_low)
    if normalized.get("empirical_coverage") is None:
        normalized["empirical_coverage"] = source.get("coverage")
    empirical = _finite_float(normalized.get("empirical_coverage"))
    nominal = _finite_float(normalized.get("nominal_coverage"))
    if normalized.get("coverage_error") is None and empirical is not None and nominal is not None:
        normalized["coverage_error"] = empirical - nominal
    if normalized.get("coverage_abs_error") is None and empirical is not None and nominal is not None:
        normalized["coverage_abs_error"] = abs(empirical - nominal)
    if normalized.get("empirical_q_low") is None and source.get("lower_breach_rate") is not None:
        normalized["empirical_q_low"] = source.get("lower_breach_rate")
    if normalized.get("empirical_q_high") is None and source.get("upper_breach_rate") is not None:
        upper_breach = _finite_float(source.get("upper_breach_rate"))
        normalized["empirical_q_high"] = None if upper_breach is None else 1.0 - upper_breach
    return normalized


def _config_from_candidate(item: dict[str, Any]) -> dict[str, Any]:
    config = dict(item.get("candidate") or item.get("config") or {})
    if not config and item.get("config_hint"):
        config = dict(item["config_hint"])
    return config


def _family_from_checkpoint(path: str | None, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback)
    name = Path(path or "").name.lower()
    if name.startswith("patchtst"):
        return "patchtst"
    if name.startswith("cnn_lstm"):
        return "cnn_lstm"
    if name.startswith("tide"):
        return "tide"
    return "unknown"


def _best_line_baseline(cp54: dict[str, Any], cp55_row: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if cp55_row:
        baseline = cp55_row.get("best_line_baseline_by_ic")
        if isinstance(baseline, dict) and baseline.get("metrics"):
            return {
                "name": baseline.get("name"),
                "metrics": _normalize_line_metrics(baseline.get("metrics") or {}),
                "source": "CP55 candidate fair baseline",
            }
    rows = cp54.get("line_baselines") or []
    best = None
    best_value = float("-inf")
    for row in rows:
        metrics = _normalize_line_metrics(row.get("test") or {})
        value = _finite_float(metrics.get("ic_mean"))
        if value is not None and value > best_value:
            best_value = value
            best = {"name": row.get("name"), "metrics": metrics, "source": "CP54 common statistical baseline"}
    return best


def _best_band_baseline(cp54: dict[str, Any], cp55_row: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if cp55_row:
        baseline = cp55_row.get("best_statistical_baseline")
        if isinstance(baseline, dict) and baseline.get("metrics"):
            return {
                "name": baseline.get("name"),
                "metrics": _normalize_band_metrics(baseline.get("metrics") or {}, q_low=None, q_high=None),
                "source": "CP55 candidate fair baseline",
            }
    rows = cp54.get("band_baselines") or []
    best = None
    best_value = float("inf")
    for row in rows:
        metrics = _normalize_band_metrics(row.get("test") or {}, q_low=None, q_high=None)
        value = _finite_float(metrics.get("asymmetric_interval_score"))
        if value is not None and value < best_value:
            best_value = value
            best = {"name": row.get("name"), "metrics": metrics, "source": "CP54 common statistical baseline"}
    return best


def _line_verdict(metrics: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    ic = _finite_float(metrics.get("spearman_ic"))
    spread = _finite_float(metrics.get("long_short_spread"))
    false_safe = _finite_float(metrics.get("false_safe_tail_rate"))
    severe_recall = _finite_float(metrics.get("severe_downside_recall"))
    baseline_metrics = (baseline or {}).get("metrics") or {}
    baseline_false_safe = _finite_float(baseline_metrics.get("false_safe_tail_rate"))
    baseline_severe = _finite_float(baseline_metrics.get("severe_downside_recall"))

    ic_good = ic is not None and ic > 0
    spread_good = spread is not None and spread > 0
    risk_beats_baseline = (
        false_safe is not None
        and baseline_false_safe is not None
        and false_safe < baseline_false_safe
    ) or (
        severe_recall is not None
        and baseline_severe is not None
        and severe_recall > baseline_severe
    )

    if ic_good and spread_good and risk_beats_baseline:
        return "line_survive"
    if ic_good or spread_good:
        return "line_watch"
    if risk_beats_baseline and severe_recall is not None and severe_recall >= 0.75:
        return "risk_only_watch"
    return "line_fail"


def _compare_band(ai_metrics: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    baseline_metrics = (baseline or {}).get("metrics") or {}
    per_metric: dict[str, str] = {}
    ai_wins = 0
    baseline_wins = 0
    ties = 0
    missing = 0
    for metric, direction in CORE_BAND_COMPARE:
        ai_value = _finite_float(ai_metrics.get(metric))
        baseline_value = _finite_float(baseline_metrics.get(metric))
        if ai_value is None or baseline_value is None:
            per_metric[metric] = "missing"
            missing += 1
            continue
        if abs(ai_value - baseline_value) <= 1e-12:
            per_metric[metric] = "tie"
            ties += 1
        elif (direction == "lower" and ai_value < baseline_value) or (
            direction == "higher" and ai_value > baseline_value
        ):
            per_metric[metric] = "ai"
            ai_wins += 1
        else:
            per_metric[metric] = "baseline"
            baseline_wins += 1
    return {
        "ai_wins": ai_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "missing": missing,
        "per_metric": per_metric,
    }


def _band_verdict(metrics: dict[str, Any], comparison: dict[str, Any]) -> str:
    coverage_error = _finite_float(metrics.get("coverage_abs_error"))
    lower_breach = _finite_float(metrics.get("lower_breach_rate"))
    upper_breach = _finite_float(metrics.get("upper_breach_rate"))
    width_ic = _finite_float(metrics.get("band_width_ic"))
    downside_ic = _finite_float(metrics.get("downside_width_ic"))
    ai_wins = int(comparison.get("ai_wins") or 0)
    baseline_wins = int(comparison.get("baseline_wins") or 0)

    dynamic_positive = (width_ic is not None and width_ic > 0) or (downside_ic is not None and downside_ic > 0)
    severe_skew = (
        lower_breach is not None
        and upper_breach is not None
        and max(lower_breach, upper_breach) > 3.0 * max(min(lower_breach, upper_breach), 0.01)
    )

    if coverage_error is None:
        return "band_watch"
    if coverage_error > 0.25 or (not dynamic_positive and baseline_wins >= 2) or severe_skew:
        return "band_fail"
    if ai_wins >= 2 and coverage_error <= 0.15 and dynamic_positive:
        return "band_survive"
    if coverage_error <= 0.20 and dynamic_positive:
        return "band_watch"
    return "band_fail"


def _cp55_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name") or "")
        indexed[name] = row
        for term in LINE_TARGET_NAMES:
            if term in name:
                indexed[term] = row
        for term in BAND_TARGET_TERMS:
            if term in name:
                indexed[term] = row
    return indexed


def _matching_cp55_row(name: str, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if name in index:
        return index[name]
    for key, row in index.items():
        if key and (key in name or name in key):
            return row
    return None


def _line_candidate_row(item: dict[str, Any], cp54: dict[str, Any], cp55_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = str(item.get("name"))
    config = _config_from_candidate(item)
    cp55_row = _matching_cp55_row(name, cp55_index)
    metrics = _normalize_line_metrics(item.get("test") or item.get("test_metrics") or {})
    validation = _normalize_line_metrics(item.get("validation") or item.get("validation_metrics") or {})
    baseline = _best_line_baseline(cp54, cp55_row)
    return {
        "model": _family_from_checkpoint(item.get("checkpoint_path"), config.get("model")),
        "candidate_name": name,
        "source": item.get("source"),
        "checkpoint_path": item.get("checkpoint_path"),
        "timeframe": config.get("timeframe", "1D"),
        "horizon": config.get("horizon", item.get("horizon", 5)),
        "seq_len": config.get("seq_len"),
        "patch_len": config.get("patch_len"),
        "patch_stride": config.get("patch_stride"),
        "structure_summary": _structure_summary(config),
        "line_metrics": metrics,
        "validation_line_metrics": validation,
        "best_baseline": baseline,
        "baseline_result": _line_baseline_result(metrics, baseline),
        "verdict": _line_verdict(metrics, baseline),
    }


def _line_baseline_result(metrics: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    baseline_metrics = (baseline or {}).get("metrics") or {}
    checks = {
        "ic_mean": "higher",
        "long_short_spread": "higher",
        "false_safe_tail_rate": "lower",
        "false_safe_severe_rate": "lower",
        "severe_downside_recall": "higher",
    }
    return _compare_directional(metrics, baseline_metrics, checks)


def _compare_directional(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    checks: dict[str, str],
) -> dict[str, Any]:
    ai_wins = 0
    baseline_wins = 0
    missing = 0
    per_metric: dict[str, str] = {}
    for key, direction in checks.items():
        ai_value = _finite_float(metrics.get(key))
        baseline_value = _finite_float(baseline_metrics.get(key))
        if ai_value is None or baseline_value is None:
            missing += 1
            per_metric[key] = "missing"
            continue
        if abs(ai_value - baseline_value) <= 1e-12:
            per_metric[key] = "tie"
            continue
        if (direction == "higher" and ai_value > baseline_value) or (
            direction == "lower" and ai_value < baseline_value
        ):
            ai_wins += 1
            per_metric[key] = "ai"
        else:
            baseline_wins += 1
            per_metric[key] = "baseline"
    return {"ai_wins": ai_wins, "baseline_wins": baseline_wins, "missing": missing, "per_metric": per_metric}


def _band_candidate_row(item: dict[str, Any], cp54: dict[str, Any], cp55_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = str(item.get("name"))
    config = _config_from_candidate(item)
    q_low = _finite_float(config.get("q_low"))
    q_high = _finite_float(config.get("q_high"))
    cp55_row = _matching_cp55_row(name, cp55_index)
    test_metrics = item.get("test") or item.get("test_metrics") or item.get("ai_metrics") or {}
    validation_metrics = item.get("validation") or item.get("validation_metrics") or {}
    metrics = _normalize_band_metrics(test_metrics, q_low=q_low, q_high=q_high)
    validation = _normalize_band_metrics(validation_metrics, q_low=q_low, q_high=q_high)
    baseline = _best_band_baseline(cp54, cp55_row)
    comparison = _compare_band(metrics, baseline)
    return {
        "model": _family_from_checkpoint(item.get("checkpoint_path"), config.get("model")),
        "candidate_name": name,
        "source": item.get("source"),
        "checkpoint_path": item.get("checkpoint_path"),
        "timeframe": config.get("timeframe", "1D"),
        "horizon": config.get("horizon", 5),
        "seq_len": config.get("seq_len"),
        "q_low": q_low,
        "q_high": q_high,
        "band_mode": config.get("band_mode"),
        "calibration_method": item.get("calibration_method", "scalar_width" if "scalar" in name else "none"),
        "band_metrics": metrics,
        "validation_band_metrics": validation,
        "best_baseline": baseline,
        "comparison": comparison,
        "verdict": _band_verdict(metrics, comparison),
    }


def _structure_summary(config: dict[str, Any]) -> str:
    bits = []
    for key in ("patch_len", "patch_stride", "d_model", "n_layers", "n_heads", "band_mode"):
        if config.get(key) is not None:
            bits.append(f"{key}={config[key]}")
    return ", ".join(bits) if bits else "구조 메타 없음"


def _baseline_rows(cp54: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    line_rows = []
    for item in cp54.get("line_baselines") or []:
        line_rows.append(
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "config": item.get("config"),
                "line_metrics": _normalize_line_metrics(item.get("test") or {}),
            }
        )
    band_rows = []
    for item in cp54.get("band_baselines") or []:
        band_rows.append(
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "config": item.get("config"),
                "band_metrics": _normalize_band_metrics(item.get("test") or {}, q_low=None, q_high=None),
            }
        )
    return line_rows, band_rows


def _collect_band_candidates(cp53: dict[str, Any], cp55: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(cp53.get("band_candidates") or [])
    existing_names = {str(item.get("name")) for item in candidates}
    for item in cp55.get("band_candidates") or []:
        name = str(item.get("name"))
        compact_name = name.split("::")[-2] if "::" in name and name.endswith("::scalar_width") else name
        if compact_name in existing_names:
            continue
        candidates.append(
            {
                "name": name,
                "checkpoint_path": item.get("checkpoint_path"),
                "candidate": item.get("config") or {},
                "source": item.get("source", "docs\\cp55_fair_baseline_and_band_candidate_expansion_metrics.json"),
                "test": item.get("ai_metrics") or {},
                "validation": {},
                "calibration_method": item.get("calibration_method", "none"),
            }
        )
    return candidates


def build_payload() -> dict[str, Any]:
    started = time.perf_counter()
    cp53 = _read_json("docs/cp53_existing_candidate_regrade_metrics.json")
    cp54 = _read_json("docs/cp54_baseline_metric_comparison_metrics.json")
    cp55 = _read_json("docs/cp55_fair_baseline_and_band_candidate_expansion_metrics.json")
    cp55_line_index = _cp55_index(cp55.get("line_candidates") or [])
    cp55_band_index = _cp55_index(cp55.get("band_candidates") or [])

    line_candidates = [
        _line_candidate_row(item, cp54, cp55_line_index)
        for item in cp53.get("line_candidates", [])
        if str(item.get("name")) in LINE_TARGET_NAMES or str(item.get("name", "")).startswith("h10") or str(item.get("name", "")).startswith("h20")
    ]

    band_candidates = [
        _band_candidate_row(item, cp54, cp55_band_index)
        for item in _collect_band_candidates(cp53, cp55)
        if any(term in str(item.get("name")) for term in BAND_TARGET_TERMS)
    ]

    line_baselines, band_baselines = _baseline_rows(cp54)
    legacy_overlay = {
        "excluded_from_model_ranking": True,
        "excluded_terms": list(LEGACY_OVERLAY_NAMES),
        "legacy_sources_seen": [
            "docs\\cp53_existing_candidate_regrade_metrics.json::composite_policies",
            "logs\\cp45\\*_composite_policy.json",
            "CP46 upper buffer 계열",
        ],
        "reason": "CP57~CP61 결정에 따라 overlay/composite 지표는 모델 생존 근거가 아니라 표시 진단값이다.",
    }

    return {
        "cp": "CP62-M",
        "purpose": "CP61 line_metrics / band_metrics 분리 schema 기준 기존 후보 재채점",
        "rules": {
            "new_training": False,
            "save_run": False,
            "db_write": False,
            "db_schema_change": False,
            "ui_change": False,
            "backend_api_change": False,
            "composite_candidate_restore": False,
        },
        "sources": {
            "candidate_regrade": "docs\\cp53_existing_candidate_regrade_metrics.json",
            "baseline_comparison": "docs\\cp54_baseline_metric_comparison_metrics.json",
            "fair_baseline": "docs\\cp55_fair_baseline_and_band_candidate_expansion_metrics.json",
        },
        "schema": {
            "line_metrics": list(LINE_METRIC_KEYS),
            "band_metrics": list(BAND_METRIC_KEYS),
            "legacy_overlay_diagnostics_excluded_from_ranking": True,
        },
        "line_candidates": line_candidates,
        "band_candidates": band_candidates,
        "line_baselines": line_baselines,
        "band_baselines": band_baselines,
        "legacy_overlay_diagnostics": legacy_overlay,
        "reexperiment_required": _reexperiment_required(cp55, line_candidates, band_candidates),
        "summary": _summary(line_candidates, band_candidates),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _summary(line_candidates: list[dict[str, Any]], band_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def _count(rows: list[dict[str, Any]], prefix: str) -> dict[str, int]:
        return {
            "survive": sum(1 for row in rows if row.get("verdict") == f"{prefix}_survive"),
            "watch": sum(1 for row in rows if row.get("verdict") in {f"{prefix}_watch", "risk_only_watch"}),
            "fail": sum(1 for row in rows if row.get("verdict") == f"{prefix}_fail"),
        }

    line_priority = {"line_survive": 3, "line_watch": 2, "risk_only_watch": 1, "line_fail": 0}
    ranked_lines = sorted(
        line_candidates,
        key=lambda row: (
            line_priority.get(str(row.get("verdict")), 0),
            _finite_float((row.get("line_metrics") or {}).get("spearman_ic")) or float("-inf"),
            _finite_float((row.get("line_metrics") or {}).get("long_short_spread")) or float("-inf"),
        ),
        reverse=True,
    )
    ranked_bands = sorted(
        band_candidates,
        key=lambda row: (
            {"band_survive": 2, "band_watch": 1, "band_fail": 0}.get(str(row.get("verdict")), 0),
            -(_finite_float((row.get("band_metrics") or {}).get("coverage_abs_error")) or float("inf")),
            -(_finite_float((row.get("band_metrics") or {}).get("asymmetric_interval_score")) or float("inf")),
        ),
        reverse=True,
    )
    return {
        "line_counts": _count(line_candidates, "line"),
        "band_counts": _count(band_candidates, "band"),
        "top_line_candidates": [row.get("candidate_name") for row in ranked_lines[:3]],
        "top_band_candidates": [row.get("candidate_name") for row in ranked_bands[:3]],
    }


def _reexperiment_required(
    cp55: dict[str, Any],
    line_candidates: list[dict[str, Any]],
    band_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in cp55.get("unsupported_or_reexperiment_required") or []:
        rows.append(
            {
                "name": item.get("name"),
                "role": item.get("role"),
                "reason": item.get("reason"),
                "checkpoint_path": item.get("checkpoint_path"),
            }
        )
    for row in line_candidates:
        if row.get("verdict") in {"line_watch", "risk_only_watch"}:
            rows.append(
                {
                    "name": row.get("candidate_name"),
                    "role": "line",
                    "reason": f"CP61 재채점 verdict={row.get('verdict')}",
                    "checkpoint_path": row.get("checkpoint_path"),
                }
            )
    for row in band_candidates:
        if row.get("verdict") == "band_watch":
            rows.append(
                {
                    "name": row.get("candidate_name"),
                    "role": "band",
                    "reason": "통계 baseline 대비 완승은 아니지만 dynamic width 또는 calibration 신호가 있어 재실험 필요",
                    "checkpoint_path": row.get("checkpoint_path"),
                }
            )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _line_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 후보 | horizon | 구조 | IC | spread | IC_IR | false_safe_tail | severe_recall | baseline | AI 승/패 | 판정 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        metrics = row.get("line_metrics") or {}
        baseline = row.get("best_baseline") or {}
        result = row.get("baseline_result") or {}
        lines.append(
            "| {name} | {horizon} | {structure} | {ic} | {spread} | {ic_ir} | {false_safe} | {severe} | {baseline} | {wins}/{losses} | {verdict} |".format(
                name=row.get("candidate_name"),
                horizon=row.get("horizon"),
                structure=row.get("structure_summary") or "-",
                ic=_fmt(metrics.get("spearman_ic")),
                spread=_fmt(metrics.get("long_short_spread")),
                ic_ir=_fmt(metrics.get("ic_ir")),
                false_safe=_fmt(metrics.get("false_safe_tail_rate")),
                severe=_fmt(metrics.get("severe_downside_recall")),
                baseline=baseline.get("name"),
                wins=result.get("ai_wins"),
                losses=result.get("baseline_wins"),
                verdict=row.get("verdict"),
            )
        )
    return "\n".join(lines)


def _band_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 후보 | 모델 | q | coverage err | lower/upper breach | width | interval | width_ic | downside_ic | baseline | AI 승/패 | 판정 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        metrics = row.get("band_metrics") or {}
        baseline = row.get("best_baseline") or {}
        comparison = row.get("comparison") or {}
        q_text = f"{_fmt(row.get('q_low'), 2)}~{_fmt(row.get('q_high'), 2)}"
        lines.append(
            "| {name} | {model} | {q} | {coverage_error} | {lower}/{upper} | {width} | {interval} | {width_ic} | {downside_ic} | {baseline} | {wins}/{losses} | {verdict} |".format(
                name=row.get("candidate_name"),
                model=row.get("model"),
                q=q_text,
                coverage_error=_fmt(metrics.get("coverage_abs_error")),
                lower=_fmt(metrics.get("lower_breach_rate")),
                upper=_fmt(metrics.get("upper_breach_rate")),
                width=_fmt(metrics.get("avg_band_width")),
                interval=_fmt(metrics.get("asymmetric_interval_score")),
                width_ic=_fmt(metrics.get("band_width_ic")),
                downside_ic=_fmt(metrics.get("downside_width_ic")),
                baseline=baseline.get("name"),
                wins=comparison.get("ai_wins"),
                losses=comparison.get("baseline_wins"),
                verdict=row.get("verdict"),
            )
        )
    return "\n".join(lines)


def _baseline_table(rows: list[dict[str, Any]], layer: str) -> str:
    if layer == "line":
        lines = [
            "| baseline | IC | spread | false_safe_tail | severe_recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            metrics = row.get("line_metrics") or {}
            lines.append(
                f"| {row.get('name')} | {_fmt(metrics.get('ic_mean'))} | {_fmt(metrics.get('long_short_spread'))} | {_fmt(metrics.get('false_safe_tail_rate'))} | {_fmt(metrics.get('severe_downside_recall'))} |"
            )
        return "\n".join(lines)
    lines = [
        "| baseline | coverage err | interval | width_ic | downside_ic |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("band_metrics") or {}
        lines.append(
            f"| {row.get('name')} | {_fmt(metrics.get('coverage_abs_error'))} | {_fmt(metrics.get('asymmetric_interval_score'))} | {_fmt(metrics.get('band_width_ic'))} | {_fmt(metrics.get('downside_width_ic'))} |"
        )
    return "\n".join(lines)


def build_report(payload: dict[str, Any]) -> str:
    line_rows = payload.get("line_candidates") or []
    band_rows = payload.get("band_candidates") or []
    summary = payload.get("summary") or {}
    reexperiment = payload.get("reexperiment_required") or []
    line_survivors = [row for row in line_rows if row.get("verdict") == "line_survive"]
    band_survivors = [row for row in band_rows if row.get("verdict") == "band_survive"]
    band_watch = [row for row in band_rows if row.get("verdict") == "band_watch"]
    return f"""# CP62 CP61 schema 기준 기존 후보 재채점 보고서

CP62는 새 학습 없이 기존 checkpoint / metrics JSON 산출물을 CP61의 line_metrics / band_metrics 분리 schema로 다시 해석한 CP다.

## 1. Executive Summary
- 새 학습, save-run, DB 쓰기, DB schema 변경, UI/API 수정은 수행하지 않았다.
- composite/overlay 계열은 모델 랭킹에서 제외했고, legacy_overlay_diagnostics로만 분리했다.
- line 후보 집계: {summary.get('line_counts')}.
- band 후보 집계: {summary.get('band_counts')}.
- top line 후보: {', '.join(summary.get('top_line_candidates') or [])}.
- top band 후보: {', '.join(summary.get('top_band_candidates') or [])}.

## 2. CP61 schema 기준 후보 목록
- line 후보는 `line_metrics`만으로 판정했다. coverage/breach/line_inside_band는 line 판정에 쓰지 않았다.
- band 후보는 `band_metrics`만으로 판정했다. IC/spread/line_inside_band는 band 판정에 쓰지 않았다.
- composite 정책 후보는 `line_band_composite`, `risk_first_*`, `include_line_clamp`, `upper_buffer` 전부 랭킹 제외다.

## 3. Line 후보 순위
{_line_table(line_rows)}

## 4. Band 후보 순위
{_band_table(band_rows)}

## 5. Baseline 대비 이긴 점 / 진 점
### Line Baseline
{_baseline_table(payload.get('line_baselines') or [], 'line')}

### Band Baseline
{_baseline_table(payload.get('band_baselines') or [], 'band')}

요약:
- line 후보는 IC/spread가 양수인 PatchTST h5/h20 계열이 남지만, random/historical baseline과의 격차가 크지 않은 후보가 있다.
- band 후보는 dynamic width 신호는 일부 있으나, rolling/Bollinger/constant 통계 baseline 대비 interval과 coverage_abs_error를 안정적으로 이겼다고 보기 어렵다.

## 6. Composite/Overlay 지표 제외 확인
- `line_inside_band_ratio`, `risk_first_lower_preserve`, `risk_first_upper_buffer`, `include_line_clamp`는 모델 생존 근거에서 제외했다.
- 기존 composite 산출물은 legacy demo 또는 표시 진단 자료로만 분류했다.
- CP62 JSON의 `legacy_overlay_diagnostics.excluded_from_model_ranking=true`로 기록했다.

## 7. 재실험 필요 후보
- 재실험 필요 후보 수: {len(reexperiment)}.
- 주요 band watch 후보: {', '.join(row.get('candidate_name') for row in band_watch[:5]) or '없음'}.
- line survive 후보: {', '.join(row.get('candidate_name') for row in line_survivors) or '없음'}.
- band survive 후보: {', '.join(row.get('candidate_name') for row in band_survivors) or '없음'}.

## 8. 기존 판정에서 바뀐 후보
- CP61 기준에서는 composite compatibility로 band 후보를 살리지 않는다.
- CNN-LSTM s60 계열은 단독 band 지표상 watch/survive 가능성이 있어도, 통계 baseline을 압도하지 못하면 재실험 필요로 분리했다.
- PatchTST dense overlap h5는 risk-only 참고 가능성이 있으나 line 주력 후보로는 약하다.

## 9. 다음 CP 추천
- line 실험: PatchTST h5 longer/baseline을 기준으로 TiDE line 후보가 있으면 같은 line_metrics로 재평가한다.
- band 실험: CNN-LSTM/TiDE/PatchTST band를 통계 baseline-aware residual/scale 방식으로 재설계할지 검토한다.
- 데이터/피처 점검: band_width_ic와 downside_width_ic가 baseline보다 약한 원인이 feature/target 계약인지 확인한다.
- baseline 강화: rolling historical quantile, Bollinger return band, volatility scaled band를 제품 baseline으로도 유지할 가치가 있다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP62 CP61 schema 기준 기존 후보 재채점")
    parser.add_argument("--output-json", default="docs/cp62_cp61_schema_candidate_regrade_metrics.json")
    parser.add_argument("--output-report", default="docs/cp62_cp61_schema_candidate_regrade_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    _write_json(args.output_json, payload)
    _write_text(args.output_report, build_report(payload))
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "output_report": args.output_report,
                "line_candidates": len(payload["line_candidates"]),
                "band_candidates": len(payload["band_candidates"]),
                "elapsed_seconds": round(float(payload["elapsed_seconds"]), 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
