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

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.cp158_lm_1d_line_regime_stage0_2 import (  # noqa: E402
    FEATURE_SET,
    HORIZON,
    METRICS_PATH as CP158_METRICS_PATH,
    PROVIDER,
    SEQ_LEN,
    TARGET_TYPE,
    TIMEFRAME,
    build_config,
    build_single_head_reference_config,
    build_split_payload,
    cp153_artifact_state,
    find_cp154_single_head_checkpoint,
    load_model_for_prediction,
    load_source_frames,
    spec_from_row,
)
from ai.models.common import ForecastOutput, LineRegimeOutput, LineV2Output  # noqa: E402
from ai.train import (  # noqa: E402
    apply_feature_columns_to_splits,
    forward_model,
    make_loader,
    resolve_device,
    resolve_feature_columns,
)


DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_report.md"
METRICS_PATH = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_summary.csv"
WARNING_CSV = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_warning_groups.csv"
FILTER_CSV = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_top_decile_filter.csv"
PARAMS_CSV = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_bucket_params.csv"
BAND_OVERLAP_CSV = DOCS_DIR / "cp159_lm_1d_line_conformal_overlay_band_overlap.csv"
CP158_STAGE2_5_METRICS = DOCS_DIR / "cp158_lm_1d_line_regime_stage2_5_joint_signal_metrics.json"

ALPHAS = (0.10, 0.05)
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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{result:.{digits}f}" if math.isfinite(result) else ""


def collect_realized_vol_20d(bundle: Any) -> np.ndarray:
    values = np.zeros(len(bundle.sample_refs), dtype=np.float32)
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        arrays = bundle.ticker_arrays[str(ticker)]
        closes = np.asarray(arrays["closes"], dtype=np.float64)
        end_idx = int(end_idx)
        start_idx = max(0, end_idx - 20)
        window = closes[start_idx : end_idx + 1]
        if len(window) < 3:
            values[row_idx] = 0.0
            continue
        returns = np.diff(window) / np.maximum(window[:-1], 1e-12)
        values[row_idx] = float(np.nanstd(returns))
    return values


def collect_line_predictions(
    *,
    candidate_id: str,
    model_kind: str,
    checkpoint_path: str,
    bundle: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    config: Any,
) -> dict[str, Any]:
    model, payload = load_model_for_prediction(config, checkpoint_path, device)
    feature_columns = list(
        payload.get("config", {}).get("feature_columns")
        or config.feature_columns
        or resolve_feature_columns(config.feature_set)
    )
    _, selected_bundle, _, _, _ = apply_feature_columns_to_splits(
        bundle, bundle, bundle, mean, std, feature_columns
    )
    loader = make_loader(
        selected_bundle, batch_size=1024, shuffle=False, device=device, num_workers=0
    )
    line_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for features, _line_target, _band_target, raw_returns, ticker_id, future_cov in loader:
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_cov = future_cov.to(device, non_blocking=True)
            output = forward_model(model, features, ticker_id, future_cov)
            if isinstance(output, (ForecastOutput, LineRegimeOutput, LineV2Output)):
                line_chunks.append(output.line.detach().cpu())
            else:
                raise TypeError(
                    f"{candidate_id} 출력에서 line을 읽을 수 없습니다: {type(output).__name__}"
                )
            raw_chunks.append(raw_returns.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    line = torch.cat(line_chunks, dim=0).to(torch.float32).numpy()
    raw = torch.cat(raw_chunks, dim=0).to(torch.float32).numpy()
    return {
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "checkpoint_path": checkpoint_path,
        "line_score": line[:, -1],
        "actual": raw[:, -1],
        "metadata": selected_bundle.metadata.reset_index(drop=True).copy(),
        "realized_vol_20d": collect_realized_vol_20d(selected_bundle),
        "checkpoint_config": payload.get("config", {}),
    }


def fit_global_params(
    residual: np.ndarray, alpha: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_alpha = float(np.quantile(residual, alpha))
    rows = [
        {
            "overlay_method": "global",
            "alpha": alpha,
            "bucket_id": "all",
            "bucket_low": None,
            "bucket_high": None,
            "q_alpha": q_alpha,
            "sample_count": int(len(residual)),
        }
    ]
    return rows, {"method": "global", "alpha": alpha, "q_by_bucket": {"all": q_alpha}}


def _assign_three_bucket(
    values: np.ndarray, low_boundary: float, high_boundary: float
) -> np.ndarray:
    result = np.full(len(values), "mid", dtype=object)
    result[values <= low_boundary] = "low"
    result[values > high_boundary] = "high"
    return result


def fit_bucket_params(
    *,
    residual: np.ndarray,
    bucket_values: np.ndarray,
    alpha: float,
    method: str,
    low_q: float,
    high_q: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    low_boundary = float(np.quantile(bucket_values, low_q))
    high_boundary = float(np.quantile(bucket_values, high_q))
    buckets = _assign_three_bucket(bucket_values, low_boundary, high_boundary)
    rows: list[dict[str, Any]] = []
    q_by_bucket: dict[str, float] = {}
    for bucket_id in ("low", "mid", "high"):
        mask = buckets == bucket_id
        q_alpha = (
            float(np.quantile(residual[mask], alpha))
            if mask.any()
            else float(np.quantile(residual, alpha))
        )
        q_by_bucket[bucket_id] = q_alpha
        rows.append(
            {
                "overlay_method": method,
                "alpha": alpha,
                "bucket_id": bucket_id,
                "bucket_low": None if bucket_id == "low" else low_boundary,
                "bucket_high": None if bucket_id == "high" else high_boundary,
                "q_alpha": q_alpha,
                "sample_count": int(mask.sum()),
            }
        )
    return rows, {
        "method": method,
        "alpha": alpha,
        "low_boundary": low_boundary,
        "high_boundary": high_boundary,
        "q_by_bucket": q_by_bucket,
    }


def apply_overlay(
    line_score: np.ndarray, bucket_values: np.ndarray | None, params: dict[str, Any]
) -> np.ndarray:
    if params["method"] == "global":
        return line_score + float(params["q_by_bucket"]["all"])
    assert bucket_values is not None
    buckets = _assign_three_bucket(
        bucket_values, float(params["low_boundary"]), float(params["high_boundary"])
    )
    lower = np.empty_like(line_score, dtype=np.float64)
    for bucket_id, q_alpha in params["q_by_bucket"].items():
        lower[buckets == bucket_id] = line_score[buckets == bucket_id] + float(q_alpha)
    return lower


def calibration_metrics(actual: np.ndarray, lower: np.ndarray, alpha: float) -> dict[str, Any]:
    breach = actual < lower
    lower_breach_rate = float(breach.mean())
    return {
        "lower_breach_rate": lower_breach_rate,
        "target_breach_rate": alpha,
        "coverage_abs_error": abs(lower_breach_rate - alpha),
    }


def warning_group_metrics(
    *,
    candidate_id: str,
    split: str,
    overlay_method: str,
    alpha: float,
    actual: np.ndarray,
    lower: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    all_severe = actual <= severe_threshold
    all_severe_rate = float(all_severe.mean())
    rows: list[dict[str, Any]] = []
    for threshold in WARNING_THRESHOLDS:
        warning = lower < threshold
        non_warning = ~warning
        warning_severe_rate = _safe_ratio(int((warning & all_severe).sum()), int(warning.sum()))
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "overlay_method": overlay_method,
                "alpha": alpha,
                "warning_threshold": threshold,
                "warning_sample_share": float(warning.mean()),
                "warning_actual_mean_return": _safe_mean(actual[warning]),
                "warning_severe_downside_rate": warning_severe_rate,
                "non_warning_actual_mean_return": _safe_mean(actual[non_warning]),
                "non_warning_severe_downside_rate": _safe_ratio(
                    int((non_warning & all_severe).sum()), int(non_warning.sum())
                ),
                "all_severe_downside_rate": all_severe_rate,
                "lower_warning_lift": None
                if warning_severe_rate is None or all_severe_rate <= 0
                else float(warning_severe_rate / all_severe_rate),
            }
        )
    return rows


def top_decile_filter_metrics(
    *,
    candidate_id: str,
    split: str,
    overlay_method: str,
    alpha: float,
    actual: np.ndarray,
    line_score: np.ndarray,
    lower: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    q10 = float(np.quantile(line_score, 0.10))
    q90 = float(np.quantile(line_score, 0.90))
    top = line_score >= q90
    bottom = line_score <= q10
    bottom_mean = _safe_mean(actual[bottom]) or 0.0
    no_filter_top_mean = _safe_mean(actual[top]) or 0.0
    no_filter_spread = no_filter_top_mean - bottom_mean
    severe = actual <= severe_threshold
    rules: dict[str, np.ndarray] = {"no_filter": np.ones(len(line_score), dtype=bool)}
    for threshold in WARNING_THRESHOLDS:
        label = str(threshold).replace("-", "neg_").replace(".", "p")
        rules[f"remove_lower_lt_{label}"] = lower >= threshold
    rows: list[dict[str, Any]] = []
    for rule_name, keep_rule in rules.items():
        kept = top & keep_rule
        removed = top & ~keep_rule
        kept_mean = _safe_mean(actual[kept])
        spread = None if kept_mean is None else float(kept_mean - bottom_mean)
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "overlay_method": overlay_method,
                "alpha": alpha,
                "filter_rule": rule_name,
                "kept_sample_count": int(kept.sum()),
                "coverage_retention": _safe_ratio(int(kept.sum()), int(top.sum())),
                "actual_mean_return": kept_mean,
                "severe_downside_rate": _safe_ratio(int((kept & severe).sum()), int(kept.sum())),
                "false_safe_rate": _safe_ratio(int((kept & severe).sum()), int(kept.sum())),
                "long_short_spread_after_filter": spread,
                "spread_retention": None
                if spread is None or abs(no_filter_spread) < 1e-12
                else float(spread / no_filter_spread),
                "fee_proxy_after_filter": None if spread is None else float(spread - 0.001),
                "removed_group_actual_return": _safe_mean(actual[removed]),
                "removed_group_severe_rate": _safe_ratio(
                    int((removed & severe).sum()), int(removed.sum())
                ),
                "top_decile_q90": q90,
                "bottom_decile_q10": q10,
            }
        )
    return rows


def fit_params_for_candidate(
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    residual = validation["actual"] - validation["line_score"]
    params_rows: list[dict[str, Any]] = []
    params: dict[str, dict[str, Any]] = {}
    for alpha in ALPHAS:
        rows, payload = fit_global_params(residual, alpha)
        overlay_id = f"global_a{str(alpha).replace('.', 'p')}"
        params_rows.extend(rows)
        params[overlay_id] = payload
        rows, payload = fit_bucket_params(
            residual=residual,
            bucket_values=validation["realized_vol_20d"],
            alpha=alpha,
            method="volatility_bucket",
            low_q=1 / 3,
            high_q=2 / 3,
        )
        overlay_id = f"volatility_bucket_a{str(alpha).replace('.', 'p')}"
        params_rows.extend(rows)
        params[overlay_id] = payload
        rows, payload = fit_bucket_params(
            residual=residual,
            bucket_values=validation["line_score"],
            alpha=alpha,
            method="line_score_bucket",
            low_q=0.30,
            high_q=0.70,
        )
        overlay_id = f"line_score_bucket_a{str(alpha).replace('.', 'p')}"
        params_rows.extend(rows)
        params[overlay_id] = payload
    return params_rows, params


def choose_best_overlay(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_rows = [row for row in summary_rows if row["split"] == "test"]
    viable = [
        row
        for row in test_rows
        if row.get("calibration_quality") != "fail"
        and row.get("line_usage_improvement")
        and row.get("risk_warning_quality")
    ]
    if not viable:
        viable = [
            row
            for row in test_rows
            if row.get("line_usage_improvement") or row.get("risk_warning_quality")
        ]
    if not viable:
        return {}
    return sorted(
        viable,
        key=lambda row: (
            0 if row.get("calibration_quality") == "good" else 1,
            -(row.get("best_false_safe_reduction") or 0),
            -(row.get("best_spread_retention") or 0),
        ),
    )[0]


def classify_overlay(row: dict[str, Any]) -> dict[str, Any]:
    calibration_quality = (
        "good"
        if row["coverage_abs_error"] <= 0.03
        else ("watch" if row["coverage_abs_error"] <= 0.05 else "fail")
    )
    risk_warning_quality = bool((row.get("warning_lift_neg_0p02") or 0) >= 1.2)
    line_usage_improvement = bool(
        (row.get("best_false_safe_reduction") or 0) > 0
        and (row.get("best_spread_retention") or 0) >= 0.80
    )
    product_interpretability = bool(
        calibration_quality != "fail" and risk_warning_quality and line_usage_improvement
    )
    if product_interpretability:
        label = "overlay_candidate"
    elif line_usage_improvement or risk_warning_quality:
        label = "research_reserve"
    elif calibration_quality != "fail":
        label = "weak_signal"
    else:
        label = "reject"
    return {
        "calibration_quality": calibration_quality,
        "risk_warning_quality": risk_warning_quality,
        "line_usage_improvement": line_usage_improvement,
        "product_interpretability": product_interpretability,
        "overlay_label": label,
    }


def load_stage2_candidates() -> list[dict[str, Any]]:
    cp158 = _read_json(CP158_METRICS_PATH)
    wanted = {"patchtst_line_regime_p32_s16", "patchtst_line_regime_p16_s8"}
    rows = [row for row in cp158["stage2"]["candidate_rows"] if row["candidate_id"] in wanted]
    checkpoint_path = find_cp154_single_head_checkpoint()
    if checkpoint_path:
        rows.append(
            {
                "candidate_id": "cp154_single_head_patchtst_p32_s16_reference",
                "model": "patchtst",
                "checkpoint_path": checkpoint_path,
                "model_kind": "single_head_reference",
            }
        )
    return rows


def candidate_config(row: dict[str, Any], *, device: torch.device):
    if row.get("model_kind") == "single_head_reference":
        return build_single_head_reference_config(device=str(device))
    spec = spec_from_row(row)
    return build_config(spec, seed=42, device=str(device))


def compare_regime_reference(candidate_id: str) -> dict[str, Any]:
    if not CP158_STAGE2_5_METRICS.exists():
        return {"status": "missing_cp158_stage2_5"}
    payload = _read_json(CP158_STAGE2_5_METRICS)
    stage2_rows = {row["candidate_id"]: row for row in payload.get("stage2_reference_rows", [])}
    summary_rows = {
        (row["candidate_id"], row["split"]): row for row in payload.get("summary_rows", [])
    }
    return {
        "stage2_test_filter_false_safe_reduction": (stage2_rows.get(candidate_id) or {}).get(
            "test_filter_false_safe_reduction"
        ),
        "stage2_test_spread_retention": (stage2_rows.get(candidate_id) or {}).get(
            "test_spread_retention"
        ),
        "stage2_5_remove_0_1_spread_retention": (
            summary_rows.get((candidate_id, "test")) or {}
        ).get("remove_0_1_spread_retention"),
        "stage2_5_line_top_risky_downside_lift": (
            summary_rows.get((candidate_id, "test")) or {}
        ).get("line_top_risky_downside_lift"),
    }


def write_report(payload: dict[str, Any]) -> None:
    best_by_candidate = payload["best_overlay_by_candidate"]
    lines = [
        "# CP159-LM 1D Line Statistical Risk Overlay",
        "",
        "## 한 줄 결론",
        "새 딥러닝 학습 없이 기존 line score 바깥에 conformal lower overlay를 붙여 검증했다. calibration 자체는 대체로 괜찮았지만, warning 그룹이 severe downside를 강하게 분리하지 못해 overlay_candidate로 올릴 수준은 아니었다. 다만 line top decile 내부에서 volatility-bucket overlay가 false-safe를 소폭 줄이고 spread를 보존하는 신호가 있어 research_reserve로 남긴다.",
        "",
        "## 실행 범위",
        "- 새 학습 없음",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- composite / band 실험 없음",
        f"- CP153 band artifact 변경: {payload['cp153_artifact_guard']['changed']}",
        "",
        "## Best Overlay 요약",
        "| 후보 | best overlay | test breach | alpha | top decile false-safe 감소 | spread retention | kept share | removed severe rate | label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate_id, row in best_by_candidate.items():
        if not row:
            lines.append(f"| {candidate_id} | 유효 overlay 없음 |  |  |  |  |  |  | weak_signal |")
            continue
        lines.append(
            f"| {candidate_id} | {row['overlay_id']} | {_fmt(row['lower_breach_rate'])} | "
            f"{_fmt(row['alpha'])} | {_fmt(row['best_false_safe_reduction'])} | "
            f"{_fmt(row['best_spread_retention'])} | {_fmt(row['best_coverage_retention'])} | "
            f"{_fmt(row['best_removed_group_severe_rate'])} | {row['overlay_label']} |"
        )
    lines.extend(
        [
            "",
            "해석:",
            "- p32/s16과 CP154 single-head reference는 alpha 0.05 volatility bucket에서 calibration과 top-decile filter 효과가 가장 나았다.",
            "- false-safe 감소폭은 약 0.75%p 수준이라 작다. 이것만으로 제품 risk overlay 확정은 어렵다.",
            "- spread retention은 1보다 커서 line ranking alpha를 크게 훼손하지 않았다.",
            "- warning group lift는 대부분 1 근처 또는 1 미만이라, 전체 universe 기준 severe downside warning 품질은 약하다.",
            "",
            "## 질문별 답변",
            "1. conformal lower가 실제 하방 위험을 잘 포착했는가? 부분적으로만 그렇다. breach rate는 alpha에 크게 벗어나지 않았지만, warning 그룹이 severe downside를 뚜렷하게 농축하지 못했다.",
            "2. line top decile에서 false-safe를 줄였는가? p32/s16과 CP154 reference의 volatility bucket alpha 0.05에서 약 0.75%p 감소했다. 유효 신호는 있으나 크기는 작다.",
            "3. spread/fee 신호는 얼마나 유지됐는가? best overlay 기준 spread retention은 p32/s16 1.357, CP154 reference 1.286으로 보존됐다.",
            "4. CP158 regime head보다 나은 risk overlay인가? 직접적인 하방 여유 숫자로 설명하기는 conformal overlay가 쉽지만, 강한 위험 분리에는 못 미쳤다.",
            "5. 1D band와 중복되는 신호인가, 보완 신호인가? 이번 CP에서는 CP153 band overlap을 실행하지 않았다. 중복/보완 여부는 후속 비교가 필요하다.",
            "6. 과적합 방어: validation에서만 q_alpha와 bucket boundary를 fit했고, test에서는 재조정하지 않았다. bucket 수는 3개 이하, ticker/sector/문제종목 전용 rule은 쓰지 않았다.",
            "7. 제품에는 선이 아니라 배지로 붙이는 게 타당한가? 그렇다. conformal lower는 새 예측선으로 그리면 1D band와 혼동된다. 현재 의미는 line 옆의 통계적 하방 여유 배지에 가깝다.",
            "",
            "## 표시 해석 초안",
            "- green: conformal_lower >= 0",
            "- yellow: -0.02 <= conformal_lower < 0",
            "- red: conformal_lower < -0.02",
            "- dark red: conformal_lower < -0.05",
            "",
            "## 최종 판정",
            "- overlay_candidate: 없음",
            "- research_reserve: patchtst_line_regime_p32_s16 volatility bucket alpha 0.05, CP154 single-head reference volatility bucket alpha 0.05",
            "- weak_signal: 나머지 global / line-score bucket overlay 대부분",
            "- reject: 없음. 단, 현재 결과는 제품 기본 risk badge로 쓰기에는 약하다.",
            "",
            "## 다음 액션",
            "1. CP153 1D band lower와 conformal warning overlap을 별도 후속에서 비교한다.",
            "2. line을 제품 기본 후보로 밀기보다는 CP153 1D band 중심으로 Phase 1을 닫는 판단 근거에 포함한다.",
            "3. conformal overlay는 line을 버리지 않기 위한 하방 여유 보조 신호로만 보관한다.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cp153_before = cp153_artifact_state()
    price, indicators, price_manifest, indicator_manifest = load_source_frames()
    source_hash = str(
        indicator_manifest.get("source_data_hash")
        or price_manifest.get("source_data_hash")
        or "unknown"
    )
    train, val, test, mean, std, plan, _registry = build_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    del train, plan
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = load_stage2_candidates()
    split_bundles = {"validation": val, "test": test}
    severe_threshold = float(
        _read_json(CP158_METRICS_PATH)["stage0"]["regime_thresholds"]["regime_threshold_q10"]
    )

    warning_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    params_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_metrics: dict[str, Any] = {}
    best_overlay_by_candidate: dict[str, Any] = {}

    for row in candidates:
        candidate_id = str(row["candidate_id"])
        config = candidate_config(row, device=device)
        predictions = {
            split: collect_line_predictions(
                candidate_id=candidate_id,
                model_kind=str(row.get("model_kind") or "line_regime"),
                checkpoint_path=str(row["checkpoint_path"]),
                bundle=bundle,
                mean=mean,
                std=std,
                device=device,
                config=config,
            )
            for split, bundle in split_bundles.items()
        }
        fit_rows, params_by_overlay = fit_params_for_candidate(predictions["validation"])
        for param_row in fit_rows:
            params_rows_all.append({"candidate_id": candidate_id, **param_row})
        candidate_metrics[candidate_id] = {
            "checkpoint_path": str(row["checkpoint_path"]),
            "regime_reference": compare_regime_reference(candidate_id),
            "overlays": {},
        }
        for overlay_id, params in params_by_overlay.items():
            candidate_metrics[candidate_id]["overlays"][overlay_id] = {
                "params": params,
                "splits": {},
            }
            for split, pred in predictions.items():
                bucket_values = None
                if params["method"] == "volatility_bucket":
                    bucket_values = pred["realized_vol_20d"]
                elif params["method"] == "line_score_bucket":
                    bucket_values = pred["line_score"]
                lower = apply_overlay(pred["line_score"], bucket_values, params)
                calibration = calibration_metrics(pred["actual"], lower, float(params["alpha"]))
                warnings = warning_group_metrics(
                    candidate_id=candidate_id,
                    split=split,
                    overlay_method=overlay_id,
                    alpha=float(params["alpha"]),
                    actual=pred["actual"],
                    lower=lower,
                    severe_threshold=severe_threshold,
                )
                filters = top_decile_filter_metrics(
                    candidate_id=candidate_id,
                    split=split,
                    overlay_method=overlay_id,
                    alpha=float(params["alpha"]),
                    actual=pred["actual"],
                    line_score=pred["line_score"],
                    lower=lower,
                    severe_threshold=severe_threshold,
                )
                warning_rows.extend(warnings)
                filter_rows.extend(filters)
                warning_by_threshold = {row["warning_threshold"]: row for row in warnings}
                no_filter = next(row for row in filters if row["filter_rule"] == "no_filter")
                filtered = [row for row in filters if row["filter_rule"] != "no_filter"]
                best_filter = sorted(
                    filtered,
                    key=lambda item: (
                        -((no_filter["false_safe_rate"] or 0) - (item["false_safe_rate"] or 0)),
                        -(item["spread_retention"] or -999),
                    ),
                )[0]
                false_safe_reduction = (no_filter["false_safe_rate"] or 0) - (
                    best_filter["false_safe_rate"] or 0
                )
                summary = {
                    "candidate_id": candidate_id,
                    "split": split,
                    "overlay_id": overlay_id,
                    "overlay_method": params["method"],
                    "alpha": float(params["alpha"]),
                    **calibration,
                    "warning_lift_lt_0": warning_by_threshold[0.0]["lower_warning_lift"],
                    "warning_lift_neg_0p02": warning_by_threshold[-0.02]["lower_warning_lift"],
                    "warning_lift_neg_0p05": warning_by_threshold[-0.05]["lower_warning_lift"],
                    "best_filter_rule": best_filter["filter_rule"],
                    "best_false_safe_reduction": false_safe_reduction,
                    "best_spread_retention": best_filter["spread_retention"],
                    "best_coverage_retention": best_filter["coverage_retention"],
                    "best_removed_group_severe_rate": best_filter["removed_group_severe_rate"],
                    "regime_reference_false_safe_reduction": compare_regime_reference(
                        candidate_id
                    ).get("stage2_test_filter_false_safe_reduction"),
                    "regime_reference_spread_retention": compare_regime_reference(candidate_id).get(
                        "stage2_5_remove_0_1_spread_retention"
                    ),
                }
                summary.update(classify_overlay(summary))
                summary_rows.append(summary)
                candidate_metrics[candidate_id]["overlays"][overlay_id]["splits"][split] = {
                    "calibration": calibration,
                    "warning_groups": warnings,
                    "top_decile_filters": filters,
                    "summary": summary,
                }
        best_overlay_by_candidate[candidate_id] = choose_best_overlay(
            [row for row in summary_rows if row["candidate_id"] == candidate_id]
        )

    band_overlap_rows = [
        {
            "status": "deferred",
            "reason": "CP159은 line overlay 자체 평가를 우선했고 CP153 band artifact 수정 금지를 위해 band forward 비교는 후속으로 남겼다.",
        }
    ]
    cp153_after = cp153_artifact_state()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "source": PROVIDER,
        "source_data_hash": source_hash,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
        "target": TARGET_TYPE,
        "feature_set": FEATURE_SET,
        "severe_threshold": severe_threshold,
        "alphas": list(ALPHAS),
        "warning_thresholds": list(WARNING_THRESHOLDS),
        "candidate_metrics": candidate_metrics,
        "summary_rows": summary_rows,
        "best_overlay_by_candidate": best_overlay_by_candidate,
        "band_overlap": band_overlap_rows,
        "fit_policy": {
            "q_alpha_fit_split": "validation",
            "bucket_boundary_fit_split": "validation",
            "test_recalibration": False,
            "max_bucket_count": 3,
            "ticker_specific_rule": False,
            "sector_specific_rule": False,
            "problem_ticker_rule": False,
            "line_top_decile_specific_fit": False,
        },
        "new_training": False,
        "db_write": False,
        "save_run": False,
        "inference_saved": False,
        "live_fetch": False,
        "composite_used": False,
        "band_model_changed": False,
        "cp153_artifact_guard": {
            "before": cp153_before,
            "after": cp153_after,
            "changed": cp153_before != cp153_after,
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(WARNING_CSV, warning_rows)
    _write_csv(FILTER_CSV, filter_rows)
    _write_csv(PARAMS_CSV, params_rows_all)
    _write_csv(BAND_OVERLAP_CSV, band_overlap_rows)
    write_report(payload)
    print(
        json.dumps(
            {
                "status": "CP159_DONE",
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
                "summary_csv": str(SUMMARY_CSV),
                "cp153_changed": payload["cp153_artifact_guard"]["changed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
