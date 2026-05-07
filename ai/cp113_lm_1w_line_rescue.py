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


REPORT_PATH = PROJECT_ROOT / "docs" / "cp113_lm_1w_line_rescue_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp113_lm_1w_line_rescue_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp113_lm_1w_line_rescue_logs"
CP112_METRICS_PATH = PROJECT_ROOT / "docs" / "cp112_lm_1w_line_smoke_metrics.json"

CANDIDATES = [
    "full_features_baseline",
    "no_fundamentals",
    "price_volatility_volume",
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


def extract_h1_h4_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _line_source(metrics)
    return {key: source.get(f"h1_h5_{key}") for key in LINE_KEYS}


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


def classify_candidate(line_metrics: dict[str, Any], *, line_gate_pass: bool) -> str:
    ic_mean = _safe_float(line_metrics.get("ic_mean"))
    spread = _safe_float(line_metrics.get("long_short_spread"))
    false_safe = _safe_float(line_metrics.get("false_safe_tail_rate"))
    severe_recall = _safe_float(line_metrics.get("severe_downside_recall"))
    bias = _safe_float(line_metrics.get("conservative_bias"))
    if not line_gate_pass:
        return "phase1_watch_gate_failed"
    if ic_mean is not None and spread is not None and false_safe is not None:
        if ic_mean > 0 and spread > 0 and false_safe < 0.30:
            return "return_direction_line_candidate"
    if false_safe is not None and severe_recall is not None:
        bias_ok = bias is None or abs(bias) <= 0.25
        if severe_recall >= 0.70 and false_safe < 0.30 and bias_ok:
            return "risk_conservative_line_candidate"
    return "phase1_watch"


def _find_run_dir(candidate: str) -> Path | None:
    candidate_dir = LOG_DIR / candidate / "train_local_logs"
    if not candidate_dir.exists():
        return None
    run_dirs = [path for path in candidate_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()]
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
        shapes = {
            "line": list(output.line.shape),
            "lower_band": list(output.lower_band.shape),
            "upper_band": list(output.upper_band.shape),
        }
        expected = [2, horizon]
        return {
            "pass": shapes["line"] == expected and shapes["lower_band"] == expected and shapes["upper_band"] == expected,
            "checkpoint_exists": True,
            "expected_shape": expected,
            "shapes": shapes,
        }
    except Exception as exc:
        return {"pass": False, "checkpoint_exists": True, "error": str(exc)}


def _load_candidate(candidate: str, cp112_line: dict[str, Any], cp112_h1_h4: dict[str, Any]) -> dict[str, Any]:
    process_path = LOG_DIR / candidate / "train_process.json"
    process = _read_json(process_path) if process_path.exists() else {}
    run_dir = _find_run_dir(candidate)
    if run_dir is None:
        return {
            "candidate": candidate,
            "status": "missing_run",
            "process": process,
            "line_metrics": {},
            "h1_h4_line_metrics": {},
            "comparison_vs_cp112": {},
            "decision": {"classification": "phase1_watch_missing_run"},
        }
    summary = _read_json(run_dir / "summary.json")
    config_file = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    config = config_file.get("config") if isinstance(config_file.get("config"), dict) else {}
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    test_metrics = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    line_metrics = extract_line_metrics(test_metrics)
    h1_h4_metrics = extract_h1_h4_metrics(test_metrics)
    checkpoint_path = Path(str(summary.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    shape_check = _shape_check(checkpoint_path) if str(checkpoint_path) else {"pass": False, "error": "checkpoint_path_missing"}
    line_gate_pass = bool(best_metrics.get("line_gate_pass"))
    comparison = {
        key: {
            "current": line_metrics.get(key),
            "cp112": cp112_line.get(key),
            **compare_metric(key, line_metrics.get(key), cp112_line.get(key)),
        }
        for key in LINE_KEYS
    }
    h1_h4_comparison = {
        key: {
            "current": h1_h4_metrics.get(key),
            "cp112": cp112_h1_h4.get(key),
            **compare_metric(key, h1_h4_metrics.get(key), cp112_h1_h4.get(key)),
        }
        for key in LINE_KEYS
    }
    exit_code = int(process.get("exit_code") or 0)
    status = "PASS" if exit_code == 0 and any(value is not None for value in line_metrics.values()) and shape_check.get("pass") else "FAIL"
    classification = classify_candidate(line_metrics, line_gate_pass=line_gate_pass)
    return {
        "candidate": candidate,
        "status": status,
        "run_id": summary.get("run_id") or process.get("run_id"),
        "process": process,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path) if str(checkpoint_path) else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "shape_check": shape_check,
        "config": {
            "model": config.get("model") or summary.get("model"),
            "timeframe": config.get("timeframe") or summary.get("timeframe"),
            "horizon": config.get("horizon") or summary.get("horizon"),
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
        "dataset_plan": summary.get("dataset_plan") or config_file.get("dataset_plan") or {},
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
        "h1_h4_line_metrics": h1_h4_metrics,
        "comparison_vs_cp112": comparison,
        "h1_h4_comparison_vs_cp112": h1_h4_comparison,
        "decision": {
            "classification": classification,
            "return_direction_candidate": classification == "return_direction_line_candidate",
            "risk_conservative_candidate": classification in {"return_direction_line_candidate", "risk_conservative_line_candidate"},
        },
    }


def _select_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in candidates if item.get("status") == "PASS"]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda item: (
            1 if item.get("decision", {}).get("return_direction_candidate") else 0,
            1 if item.get("decision", {}).get("risk_conservative_candidate") else 0,
            _safe_float(item.get("line_metrics", {}).get("severe_downside_recall")) or -999.0,
            -(_safe_float(item.get("line_metrics", {}).get("false_safe_tail_rate")) or 999.0),
            _safe_float(item.get("line_metrics", {}).get("ic_mean")) or -999.0,
        ),
        reverse=True,
    )[0]


def _product_storage_recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return_candidates = [item for item in candidates if item.get("decision", {}).get("return_direction_candidate")]
    risk_candidates = [item for item in candidates if item.get("decision", {}).get("risk_conservative_candidate")]
    if return_candidates:
        return {
            "proceed_to_product_save_candidate": "not_yet",
            "reason": "수익 방향선 후보는 보이나 CP113은 save-run 금지 smoke 비교이므로 저장 전용 CP에서 재현 후 저장한다.",
        }
    if risk_candidates:
        return {
            "proceed_to_product_save_candidate": "hold",
            "reason": "수익 방향선보다는 중기 위험/보수선 후보 성격이다. 제품 저장은 기본 line이 아니라 별도 위험선 계약을 먼저 정해야 한다.",
        }
    return {
        "proceed_to_product_save_candidate": "no",
        "reason": "수익선/위험선 기준을 동시에 만족한 후보가 없다. Phase 1 watch로 보류한다.",
    }


def build_payload() -> dict[str, Any]:
    cp112 = _read_json(CP112_METRICS_PATH)
    cp112_line = cp112.get("line_metrics") or {}
    cp112_h1_h4 = (cp112.get("bucket_line_metrics") or {}).get("h1_h5") or {}
    candidates = [_load_candidate(candidate, cp112_line, cp112_h1_h4) for candidate in CANDIDATES]
    best = _select_best(candidates)
    product_storage = _product_storage_recommendation(candidates)
    payload = {
        "cp": "CP113-LM",
        "purpose": "1W PatchTST line feature set rescue comparison",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if all(candidate.get("status") == "PASS" for candidate in candidates) else "PARTIAL_OR_FAIL",
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
        "cp112_reference": {
            "run_id": (cp112.get("execution") or {}).get("run_id"),
            "source_data_hash": (cp112.get("preflight") or {}).get("source_data_hash"),
            "line_metrics": cp112_line,
            "h1_h4_line_metrics": cp112_h1_h4,
        },
        "candidates": candidates,
        "best_candidate": {
            "candidate": best.get("candidate") if best else None,
            "run_id": best.get("run_id") if best else None,
            "classification": (best.get("decision") or {}).get("classification") if best else None,
        },
        "summary_decision": {
            "return_direction_line_available": any(item.get("decision", {}).get("return_direction_candidate") for item in candidates),
            "risk_conservative_line_available": any(item.get("decision", {}).get("risk_conservative_candidate") for item in candidates),
            "product_storage": product_storage,
            "next_cp_recommendation": _next_cp_recommendation(candidates, product_storage),
        },
    }
    return payload


def _next_cp_recommendation(candidates: list[dict[str, Any]], product_storage: dict[str, Any]) -> str:
    if any(item.get("decision", {}).get("return_direction_candidate") for item in candidates):
        return "가장 좋은 1W 수익 방향선 후보를 같은 설정으로 epochs 5, save-run 전용 후보 CP에서 재현한다."
    if any(item.get("decision", {}).get("risk_conservative_candidate") for item in candidates):
        return "1W line은 중기 위험/보수선 계약으로 분리하고, 표시/저장 계약을 정한 뒤 별도 저장 CP로 넘긴다."
    return "1W line은 Phase 1 watch로 보류하고, beta 상향 또는 severe downside weighting 학습 후보를 별도 CP로 설계한다."


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        metrics = item.get("line_metrics") or {}
        comparison = item.get("comparison_vs_cp112") or {}
        rows.append(
            {
                "candidate": item.get("candidate"),
                "run_id": item.get("run_id"),
                "line_gate": (item.get("gate_status") or {}).get("line_gate_pass"),
                "class": (item.get("decision") or {}).get("classification"),
                "ic": _fmt(metrics.get("ic_mean")),
                "ic_delta": _fmt((comparison.get("ic_mean") or {}).get("delta")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "spread_delta": _fmt((comparison.get("long_short_spread") or {}).get("delta")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "false_safe_delta": _fmt((comparison.get("false_safe_tail_rate") or {}).get("delta")),
                "severe_recall": _fmt(metrics.get("severe_downside_recall")),
                "recall_delta": _fmt((comparison.get("severe_downside_recall") or {}).get("delta")),
                "bias": _fmt(metrics.get("conservative_bias")),
                "dir_acc": _fmt(metrics.get("direction_accuracy")),
            }
        )
    return rows


def _comparison_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in LINE_KEYS:
        item = (candidate.get("comparison_vs_cp112") or {}).get(key) or {}
        rows.append(
            {
                "metric": key,
                "cp112": _fmt(item.get("cp112")),
                "current": _fmt(item.get("current")),
                "delta": _fmt(item.get("delta")),
                "direction": item.get("direction"),
            }
        )
    return rows


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = payload.get("candidates") or []
    best = payload.get("best_candidate") or {}
    decision = payload.get("summary_decision") or {}
    product = decision.get("product_storage") or {}
    report = [
        "# CP113-LM 1W line rescue feature set 비교",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}`",
        f"- CP112 기준 run_id: `{((payload.get('cp112_reference') or {}).get('run_id'))}`",
        f"- 최고 후보: `{best.get('candidate')}` / run_id `{best.get('run_id')}` / 판단 `{best.get('classification')}`",
        "- 범위: PatchTST 1W h4 line_model 전용. save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 실행하지 않았다.",
        "",
        "## 후보별 결과",
        "",
        _table(
            _candidate_rows(candidates),
            [
                ("candidate", "candidate"),
                ("run_id", "run_id"),
                ("line_gate", "line_gate"),
                ("class", "class"),
                ("ic", "ic"),
                ("d_ic", "ic_delta"),
                ("spread", "spread"),
                ("d_spread", "spread_delta"),
                ("false_safe", "false_safe"),
                ("d_false_safe", "false_safe_delta"),
                ("severe_recall", "severe_recall"),
                ("d_recall", "recall_delta"),
                ("bias", "bias"),
                ("dir_acc", "dir_acc"),
            ],
        ),
        "",
        "## CP112 smoke 대비 개선/악화",
        "",
    ]
    for candidate in candidates:
        report.extend(
            [
                f"### {candidate.get('candidate')}",
                "",
                _table(
                    _comparison_rows(candidate),
                    [("metric", "metric"), ("CP112", "cp112"), ("current", "current"), ("delta", "delta"), ("판정", "direction")],
                ),
                "",
            ]
        )
    report.extend(
        [
            "## h1_h4 bucket",
            "",
            _table(
                [
                    {
                        "candidate": item.get("candidate"),
                        "ic": _fmt((item.get("h1_h4_line_metrics") or {}).get("ic_mean")),
                        "spread": _fmt((item.get("h1_h4_line_metrics") or {}).get("long_short_spread")),
                        "false_safe": _fmt((item.get("h1_h4_line_metrics") or {}).get("false_safe_tail_rate")),
                        "severe_recall": _fmt((item.get("h1_h4_line_metrics") or {}).get("severe_downside_recall")),
                    }
                    for item in candidates
                ],
                [
                    ("candidate", "candidate"),
                    ("ic_mean", "ic"),
                    ("long_short_spread", "spread"),
                    ("false_safe_tail_rate", "false_safe"),
                    ("severe_downside_recall", "severe_recall"),
                ],
            ),
            "",
            "## 제품/다음 CP 판단",
            "",
            f"- 1W 수익 방향선 가능 여부: `{decision.get('return_direction_line_available')}`",
            f"- 1W 중기 위험/보수선 가능 여부: `{decision.get('risk_conservative_line_available')}`",
            f"- 제품 후보 저장 진행: `{product.get('proceed_to_product_save_candidate')}`",
            f"- 저장 판단 사유: {product.get('reason')}",
            f"- 다음 CP 추천: {decision.get('next_cp_recommendation')}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run_report(_: argparse.Namespace) -> None:
    payload = build_payload()
    _write_json(METRICS_PATH, payload)
    _write_report(payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status"), "best_candidate": payload.get("best_candidate"), "summary_decision": payload.get("summary_decision")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP113-LM 1W line rescue 보고서 생성")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
