from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402

from ai.inference import load_checkpoint  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp114_lm_1w_line_candidate_expansion_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp114_lm_1w_line_candidate_expansion_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp114_lm_1w_line_candidate_expansion_logs"
CP113_METRICS_PATH = PROJECT_ROOT / "docs" / "cp113_lm_1w_line_rescue_metrics.json"

CANDIDATES = [
    "h4_pvv_patch16_stride8",
    "h4_no_fundamentals_patch16_stride8",
    "h4_pvv_patch32_stride16",
    "h8_pvv_patch16_stride8",
]

LINE_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "mae",
    "smape",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]

HIGHER_IS_BETTER = {
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "severe_downside_recall",
    "downside_capture_rate",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
}

LOWER_IS_BETTER = {
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "upside_sacrifice",
    "mae",
    "smape",
}


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _line_source(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    nested = metrics.get("line_metrics")
    if isinstance(nested, dict):
        return nested
    return metrics


def extract_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _line_source(metrics)
    return {key: source.get(key) for key in LINE_KEYS}


def extract_bucket_metrics(metrics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    source = _line_source(metrics)
    buckets: dict[str, dict[str, Any]] = {}
    for prefix in ("h1_h5", "h6_h10", "h11_h20"):
        bucket = {key: source.get(f"{prefix}_{key}") for key in LINE_KEYS if source.get(f"{prefix}_{key}") is not None}
        if bucket:
            buckets[prefix] = bucket
    return buckets


def compare_metric(key: str, current: Any, baseline: Any) -> dict[str, Any]:
    left = _safe_float(current)
    right = _safe_float(baseline)
    if left is None or right is None:
        return {"delta": None, "direction": "unknown"}
    delta = left - right
    if abs(delta) < 1e-12:
        direction = "flat"
    elif key in HIGHER_IS_BETTER:
        direction = "improved" if delta > 0 else "worsened"
    elif key in LOWER_IS_BETTER:
        direction = "improved" if delta < 0 else "worsened"
    elif key == "conservative_bias":
        direction = "more_conservative" if delta < 0 else "less_conservative"
    else:
        direction = "changed"
    return {"delta": delta, "direction": direction}


def classify_candidate(line_metrics: dict[str, Any], *, line_gate_pass: bool, horizon: int) -> str:
    ic_mean = _safe_float(line_metrics.get("ic_mean"))
    spread = _safe_float(line_metrics.get("long_short_spread"))
    false_safe = _safe_float(line_metrics.get("false_safe_tail_rate"))
    severe_recall = _safe_float(line_metrics.get("severe_downside_recall"))
    bias = _safe_float(line_metrics.get("conservative_bias"))
    bias_ok = bias is None or abs(bias) <= 0.25
    product_ok = (
        line_gate_pass
        and ic_mean is not None
        and spread is not None
        and false_safe is not None
        and severe_recall is not None
        and ic_mean > 0
        and spread > 0
        and false_safe <= 0.20
        and severe_recall >= 0.75
        and bias_ok
    )
    risk_ok = (
        line_gate_pass
        and false_safe is not None
        and severe_recall is not None
        and false_safe <= 0.20
        and severe_recall >= 0.80
        and bias_ok
    )
    if int(horizon) == 8:
        if product_ok:
            return "h8_feasibility_product_watch"
        if risk_ok:
            return "h8_feasibility_risk_watch"
        if line_gate_pass:
            return "h8_feasibility_watch"
        return "h8_feasibility_fail"
    if product_ok:
        return "product_line_candidate"
    if risk_ok:
        return "risk_only_candidate"
    if line_gate_pass:
        return "phase1_watch"
    return "fail_gate"


def _find_run_dir(candidate: str) -> Path | None:
    local_root = LOG_DIR / candidate / "train_local_logs"
    if not local_root.exists():
        return None
    run_dirs = [path for path in local_root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _shape_check(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"pass": False, "checkpoint_exists": False, "error": "checkpoint_missing"}
    try:
        model, checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint.get("config") or {}
        n_features = int(config.get("n_features") or len(config.get("feature_columns") or []))
        seq_len = int(config.get("seq_len") or 104)
        horizon = int(config.get("horizon") or 4)
        features = torch.randn(2, seq_len, n_features, dtype=torch.float32)
        ticker_id = torch.zeros(2, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            output = model(features, ticker_id=ticker_id)
        expected = [2, horizon]
        shapes = {
            "line": list(output.line.shape),
            "lower_band": list(output.lower_band.shape),
            "upper_band": list(output.upper_band.shape),
        }
        return {
            "pass": all(shape == expected for shape in shapes.values()),
            "checkpoint_exists": True,
            "expected_shape": expected,
            "shapes": shapes,
        }
    except Exception as exc:
        return {"pass": False, "checkpoint_exists": True, "error": str(exc)}


def _load_cp113_best() -> dict[str, Any]:
    cp113 = _read_json(CP113_METRICS_PATH)
    best_name = (cp113.get("best_candidate") or {}).get("candidate")
    for candidate in cp113.get("candidates") or []:
        if candidate.get("candidate") == best_name:
            return candidate
    for candidate in cp113.get("candidates") or []:
        if candidate.get("candidate") == "price_volatility_volume":
            return candidate
    raise RuntimeError("CP113 best 후보를 찾지 못했습니다.")


def _load_candidate(candidate: str, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    process_path = LOG_DIR / candidate / "train_process.json"
    process = _read_json(process_path) if process_path.exists() else {}
    run_dir = _find_run_dir(candidate)
    if run_dir is None:
        return {
            "candidate": candidate,
            "status": "missing_run",
            "process": process,
            "line_metrics": {},
            "comparison_vs_cp113_best": {},
            "decision": {"classification": "missing_run"},
        }
    summary = _read_json(run_dir / "summary.json")
    config_payload = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    config = config_payload.get("config") if isinstance(config_payload.get("config"), dict) else {}
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    test_metrics = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    line_metrics = extract_line_metrics(test_metrics)
    bucket_metrics = extract_bucket_metrics(test_metrics)
    checkpoint_path = Path(str(summary.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    shape = _shape_check(checkpoint_path) if str(checkpoint_path) else {"pass": False, "error": "checkpoint_path_missing"}
    horizon = int(config.get("horizon") or summary.get("horizon") or 0)
    comparison = {
        key: {
            "current": line_metrics.get(key),
            "cp113_best": baseline_metrics.get(key),
            **compare_metric(key, line_metrics.get(key), baseline_metrics.get(key)),
        }
        for key in LINE_KEYS
    }
    status = "PASS" if int(process.get("exit_code") or 0) == 0 and shape.get("pass") and any(value is not None for value in line_metrics.values()) else "FAIL"
    classification = classify_candidate(
        line_metrics,
        line_gate_pass=bool(best_metrics.get("line_gate_pass")),
        horizon=horizon,
    )
    return {
        "candidate": candidate,
        "status": status,
        "run_id": summary.get("run_id") or process.get("run_id"),
        "process": process,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path) if str(checkpoint_path) else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "shape_check": shape,
        "config": {
            "model": config.get("model") or summary.get("model"),
            "timeframe": config.get("timeframe") or summary.get("timeframe"),
            "horizon": horizon,
            "seq_len": config.get("seq_len"),
            "patch_len": config.get("patch_len"),
            "patch_stride": config.get("patch_stride"),
            "feature_set": config.get("feature_set") or summary.get("feature_set"),
            "line_target_type": config.get("line_target_type"),
            "band_target_type": config.get("band_target_type"),
            "checkpoint_selection": config.get("checkpoint_selection") or best_metrics.get("checkpoint_selection"),
            "ci_aggregate": config.get("ci_aggregate"),
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "delta": config.get("delta"),
            "n_features": config.get("n_features"),
            "amp_dtype": config.get("amp_dtype"),
            "compile_model": config.get("compile_model"),
            "use_wandb": config.get("use_wandb"),
            "market_data_provider": config.get("market_data_provider"),
        },
        "dataset_plan": summary.get("dataset_plan") or config_payload.get("dataset_plan") or {},
        "gate_status": {
            "checkpoint_selection": best_metrics.get("checkpoint_selection"),
            "gate_type": best_metrics.get("gate_type"),
            "gate_failed": best_metrics.get("gate_failed"),
            "line_gate_pass": best_metrics.get("line_gate_pass"),
            "band_gate_pass": best_metrics.get("band_gate_pass"),
            "combined_gate_pass": best_metrics.get("combined_gate_pass"),
            "role": best_metrics.get("role"),
            "best_epoch": best_metrics.get("best_epoch"),
        },
        "line_metrics": line_metrics,
        "horizon_aggregate_label": "h1_h8" if horizon == 8 else "h1_h4",
        "horizon_aggregate_metrics": line_metrics,
        "bucket_line_metrics": bucket_metrics,
        "comparison_vs_cp113_best": comparison,
        "decision": {
            "classification": classification,
            "product_line_candidate": classification == "product_line_candidate",
            "risk_only_candidate": classification == "risk_only_candidate",
            "h8_watch": classification.startswith("h8_feasibility"),
        },
    }


def _select_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in candidates if item.get("status") == "PASS" and int((item.get("config") or {}).get("horizon") or 0) == 4]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda item: (
            1 if item.get("decision", {}).get("product_line_candidate") else 0,
            1 if item.get("decision", {}).get("risk_only_candidate") else 0,
            _safe_float(item.get("line_metrics", {}).get("ic_mean")) or -999.0,
            _safe_float(item.get("line_metrics", {}).get("long_short_spread")) or -999.0,
            -(_safe_float(item.get("line_metrics", {}).get("false_safe_tail_rate")) or 999.0),
        ),
        reverse=True,
    )[0]


def _summary_decision(candidates: list[dict[str, Any]], best: dict[str, Any] | None) -> dict[str, Any]:
    product = [item for item in candidates if item.get("decision", {}).get("product_line_candidate")]
    risk = [item for item in candidates if item.get("decision", {}).get("risk_only_candidate")]
    h8 = [item for item in candidates if item.get("decision", {}).get("h8_watch")]
    if product:
        storage = {
            "proceed_to_product_save_candidate": "yes_next_cp",
            "reason": "h4 후보가 제품 line 기준을 통과했다. CP114는 save-run 금지이므로 다음 CP에서 같은 설정으로 저장 후보 재현이 필요하다.",
        }
    elif risk:
        storage = {
            "proceed_to_product_save_candidate": "hold_as_risk_line",
            "reason": "제품 수익 line 기준은 부족하지만 risk-only 기준을 만족한다. 제품 저장 전 위험선 계약 분리가 필요하다.",
        }
    else:
        storage = {
            "proceed_to_product_save_candidate": "no",
            "reason": "제품 line 또는 risk-only 기준을 통과한 h4 후보가 없다.",
        }
    return {
        "best_h4_candidate": best.get("candidate") if best else None,
        "return_direction_line_available": bool(product),
        "risk_only_line_available": bool(risk),
        "h8_feasibility_recorded": bool(h8),
        "product_storage": storage,
        "next_cp_recommendation": _next_cp_recommendation(product, risk, h8),
    }


def _next_cp_recommendation(product: list[dict[str, Any]], risk: list[dict[str, Any]], h8: list[dict[str, Any]]) -> str:
    if product:
        best_name = product[0].get("candidate")
        return f"{best_name}를 1W h4 line 저장 전용 CP로 재현한다. 같은 yfinance snapshot/hash, epochs 5, save-run true 여부는 별도 지시가 필요하다."
    if risk:
        return "1W line을 제품 기본 수익선이 아니라 중기 위험/보수선으로 분리하는 계약 CP를 먼저 연다."
    if h8:
        return "h8은 feasibility/watch로만 남기고, h4 loss 또는 feature set을 더 좁히는 후보를 설계한다."
    return "1W line은 Phase 1 watch로 보류한다."


def build_payload() -> dict[str, Any]:
    cp113_best = _load_cp113_best()
    baseline_metrics = cp113_best.get("line_metrics") or {}
    candidates = [_load_candidate(candidate, baseline_metrics) for candidate in CANDIDATES]
    best = _select_best(candidates)
    return {
        "cp": "CP114-LM",
        "purpose": "1W PatchTST line candidate expansion",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if all(item.get("status") == "PASS" for item in candidates) else "PARTIAL_OR_FAIL",
        "environment": {
            "python_executable": sys.executable,
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "LENS_USE_LOCAL_SNAPSHOTS": os.environ.get("LENS_USE_LOCAL_SNAPSHOTS"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
        },
        "scope_guard": {
            "line_model_only": True,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "off",
            "composite": False,
            "frontend_modified": False,
        },
        "cp113_best_reference": {
            "candidate": cp113_best.get("candidate"),
            "run_id": cp113_best.get("run_id"),
            "line_metrics": baseline_metrics,
        },
        "candidates": candidates,
        "best_h4_candidate": {
            "candidate": best.get("candidate") if best else None,
            "run_id": best.get("run_id") if best else None,
            "classification": (best.get("decision") or {}).get("classification") if best else None,
        },
        "summary_decision": _summary_decision(candidates, best),
    }


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        metrics = item.get("line_metrics") or {}
        comparison = item.get("comparison_vs_cp113_best") or {}
        rows.append(
            {
                "candidate": item.get("candidate"),
                "run_id": item.get("run_id"),
                "gate": (item.get("gate_status") or {}).get("line_gate_pass"),
                "class": (item.get("decision") or {}).get("classification"),
                "ic": _fmt(metrics.get("ic_mean")),
                "d_ic": _fmt((comparison.get("ic_mean") or {}).get("delta")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "d_spread": _fmt((comparison.get("long_short_spread") or {}).get("delta")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "d_false_safe": _fmt((comparison.get("false_safe_tail_rate") or {}).get("delta")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "d_severe": _fmt((comparison.get("severe_downside_recall") or {}).get("delta")),
                "bias": _fmt(metrics.get("conservative_bias")),
                "upside": _fmt(metrics.get("upside_sacrifice")),
                "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
                "fee_sharpe": _fmt(metrics.get("fee_adjusted_sharpe")),
            }
        )
    return rows


def _comparison_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in LINE_KEYS:
        item = (candidate.get("comparison_vs_cp113_best") or {}).get(key) or {}
        rows.append(
            {
                "metric": key,
                "cp113_best": _fmt(item.get("cp113_best")),
                "current": _fmt(item.get("current")),
                "delta": _fmt(item.get("delta")),
                "direction": item.get("direction"),
            }
        )
    return rows


def _bucket_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        for bucket, metrics in (item.get("bucket_line_metrics") or {}).items():
            rows.append(
                {
                    "candidate": item.get("candidate"),
                    "bucket": bucket,
                    "ic": _fmt(metrics.get("ic_mean")),
                    "spread": _fmt(metrics.get("long_short_spread")),
                    "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                    "severe": _fmt(metrics.get("severe_downside_recall")),
                }
            )
    return rows


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = payload.get("candidates") or []
    decision = payload.get("summary_decision") or {}
    storage = decision.get("product_storage") or {}
    report = [
        "# CP114-LM 1W line 후보 확장",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}`",
        f"- CP113 기준 후보: `{((payload.get('cp113_best_reference') or {}).get('candidate'))}` / `{((payload.get('cp113_best_reference') or {}).get('run_id'))}`",
        f"- best h4 후보: `{((payload.get('best_h4_candidate') or {}).get('candidate'))}` / `{((payload.get('best_h4_candidate') or {}).get('run_id'))}` / `{((payload.get('best_h4_candidate') or {}).get('classification'))}`",
        "- 범위: PatchTST 1W line_model 전용. save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 실행하지 않았다.",
        "",
        "## 후보 결과",
        "",
        _table(
            _candidate_rows(candidates),
            [
                ("candidate", "candidate"),
                ("run_id", "run_id"),
                ("gate", "gate"),
                ("class", "class"),
                ("ic", "ic"),
                ("d_ic", "d_ic"),
                ("spread", "spread"),
                ("d_spread", "d_spread"),
                ("false_safe", "false_safe"),
                ("d_false_safe", "d_false_safe"),
                ("severe", "severe"),
                ("d_severe", "d_severe"),
                ("bias", "bias"),
                ("upside", "upside"),
                ("fee_ret", "fee_ret"),
                ("fee_sharpe", "fee_sharpe"),
            ],
        ),
        "",
        "## CP113 best 대비 개선/악화",
        "",
    ]
    for candidate in candidates:
        report.extend(
            [
                f"### {candidate.get('candidate')}",
                "",
                _table(
                    _comparison_rows(candidate),
                    [("metric", "metric"), ("CP113 best", "cp113_best"), ("current", "current"), ("delta", "delta"), ("판정", "direction")],
                ),
                "",
            ]
        )
    bucket_rows = _bucket_rows(candidates)
    report.extend(
        [
            "## horizon bucket",
            "",
            _table(
                bucket_rows,
                [
                    ("candidate", "candidate"),
                    ("bucket", "bucket"),
                    ("ic_mean", "ic"),
                    ("long_short_spread", "spread"),
                    ("false_safe_tail_rate", "false_safe"),
                    ("severe_downside_recall", "severe"),
                ],
            )
            if bucket_rows
            else "bucket 지표가 생성되지 않았다.",
            "",
            "## 판정",
            "",
            f"- 제품 line 후보 존재: `{decision.get('return_direction_line_available')}`",
            f"- risk-only 후보 존재: `{decision.get('risk_only_line_available')}`",
            f"- h8 feasibility/watch 기록: `{decision.get('h8_feasibility_recorded')}`",
            f"- 제품 후보 저장 진행: `{storage.get('proceed_to_product_save_candidate')}`",
            f"- 사유: {storage.get('reason')}",
            f"- 다음 CP 추천: {decision.get('next_cp_recommendation')}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run_report(_: argparse.Namespace) -> None:
    payload = build_payload()
    _write_json(METRICS_PATH, payload)
    _write_report(payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status"), "best_h4_candidate": payload.get("best_h4_candidate"), "summary_decision": payload.get("summary_decision")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP114-LM 1W line 후보 확장 보고서 생성")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
