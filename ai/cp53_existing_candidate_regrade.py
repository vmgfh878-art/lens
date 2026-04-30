from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.composite_inference import (  # noqa: E402
    _apply_scalar_width,
    _select_latest_common_indices,
    collect_predictions,
)
from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.inference import load_checkpoint, load_checkpoint_config, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.loss import ForecastCompositeLoss  # noqa: E402
from ai.preprocessing import prepare_dataset_splits  # noqa: E402
from ai.train import (  # noqa: E402
    autocast_context,
    estimate_train_risk_thresholds,
    evaluate_loader,
    make_loader,
    resolve_device,
)


LINE_METRIC_KEYS = (
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "spearman_ic",
    "long_short_spread",
    "spread_mean",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_sharpe",
    "direction_accuracy",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "mae",
    "smape",
)

BAND_METRIC_KEYS = (
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "empirical_q_low",
    "empirical_q_high",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
)

COMPOSITE_METRIC_KEYS = (
    "line_inside_band_ratio",
    "line_inside_band_point_ratio",
    "product_display_warning_rate",
    "conservative_series_false_safe_rate",
    "composite_width_increase_ratio",
    "coverage",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "interval_score",
    "band_width_ic",
    "downside_width_ic",
)

_SPLIT_CACHE: dict[str, tuple[Any, Any, Any, dict[str, float | None]]] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _repo_path(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _metric_subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in keys}


def _candidate_status(role: str, metrics: dict[str, Any]) -> str:
    if role == "line":
        if (metrics.get("ic_mean") or 0) > 0 and (metrics.get("long_short_spread") or 0) > 0:
            if (metrics.get("false_safe_tail_rate") or 1) < 0.15 and (metrics.get("severe_downside_recall") or 0) >= 0.80:
                return "후보"
            return "생존"
        if (metrics.get("false_safe_tail_rate") or 1) < 0.20 and (metrics.get("severe_downside_recall") or 0) >= 0.80:
            return "보류"
        return "탈락"
    if role == "band":
        if metrics.get("coverage_abs_error") is None:
            return "보류"
        width_ic = metrics.get("band_width_ic")
        downside_ic = metrics.get("downside_width_ic")
        if float(metrics["coverage_abs_error"]) <= 0.08 and (width_ic or 0) >= 0.05 and (downside_ic or 0) >= 0.05:
            return "후보"
        if float(metrics["coverage_abs_error"]) <= 0.15 and (width_ic or 0) > 0 and (downside_ic or 0) > 0:
            return "생존"
        return "보류"
    return "참고"


def _load_checkpoint_record(path: str | Path) -> tuple[Any, dict[str, Any]]:
    model, checkpoint = load_checkpoint(Path(path))
    return model, dict(checkpoint["config"])


def _split_cache_key(config: dict[str, Any], *, limit_tickers: int | None) -> str:
    key = {
        "timeframe": config.get("timeframe"),
        "seq_len": int(config.get("seq_len", 0)),
        "horizon": int(config.get("horizon", 0)),
        "limit_tickers": limit_tickers,
        "future": bool(config.get("use_future_covariate", config.get("model") == "tide")),
        "line_target_type": config.get("line_target_type", "raw_future_return"),
        "band_target_type": config.get("band_target_type", "raw_future_return"),
        "ticker_registry_path": config.get("ticker_registry_path"),
    }
    return json.dumps(key, ensure_ascii=False, sort_keys=True)


def _prepare_splits(config: dict[str, Any], *, limit_tickers: int | None):
    key = _split_cache_key(config, limit_tickers=limit_tickers)
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]

    ticker_registry = resolve_checkpoint_ticker_registry(config, str(config["timeframe"]))
    train_bundle, val_bundle, test_bundle, _, _, _ = prepare_dataset_splits(
        timeframe=str(config["timeframe"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        tickers=None,
        limit_tickers=limit_tickers,
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=ticker_registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    thresholds = estimate_train_risk_thresholds(train_bundle)
    _SPLIT_CACHE[key] = (train_bundle, val_bundle, test_bundle, thresholds)
    return _SPLIT_CACHE[key]


def _criterion_from_config(config: dict[str, Any]) -> ForecastCompositeLoss:
    return ForecastCompositeLoss(
        q_low=float(config.get("q_low", 0.1)),
        q_high=float(config.get("q_high", 0.9)),
        alpha=float(config.get("alpha", 1.0)),
        beta=float(config.get("beta", 2.0)),
        delta=float(config.get("delta", 1.0)),
        lambda_line=float(config.get("lambda_line", 1.0)),
        lambda_band=float(config.get("lambda_band", 1.0)),
        lambda_width=float(config.get("lambda_width", 0.0)),
        lambda_cross=float(config.get("lambda_cross", 1.0)),
        lambda_direction=float(config.get("lambda_direction", 0.1)),
        band_mode=str(config.get("band_mode", "direct")),
    )


def regrade_checkpoint(
    *,
    checkpoint_path: str | Path,
    limit_tickers: int | None,
    device_name: str,
    batch_size: int | None = None,
    amp_dtype: str = "bf16",
) -> dict[str, Any]:
    model, config = _load_checkpoint_record(checkpoint_path)
    _, val_bundle, test_bundle, thresholds = _prepare_splits(config, limit_tickers=limit_tickers)
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    eval_batch_size = int(batch_size or config.get("batch_size", 256))
    criterion = _criterion_from_config(config)

    def _eval(split_name: str, bundle: Any) -> dict[str, Any]:
        loader = make_loader(bundle, batch_size=eval_batch_size, shuffle=False, device=device, num_workers=0)
        with autocast_context(device, amp_dtype=amp_dtype):
            return evaluate_loader(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                metadata=bundle.metadata,
                line_target_type=str(config.get("line_target_type", "raw_future_return")),
                band_target_type=str(config.get("band_target_type", "raw_future_return")),
                q_low=float(config.get("q_low", 0.1)),
                q_high=float(config.get("q_high", 0.9)),
                severe_downside_threshold=thresholds.get("severe_downside_threshold"),
                squeeze_breakout_threshold=thresholds.get("squeeze_breakout_threshold"),
                amp_dtype=amp_dtype,
                phase=split_name,
                run_id=str(config.get("run_id", Path(checkpoint_path).stem)),
            )

    return {
        "checkpoint_path": _repo_path(checkpoint_path),
        "config": {
            "model": config.get("model"),
            "role": config.get("role"),
            "timeframe": config.get("timeframe"),
            "horizon": config.get("horizon"),
            "seq_len": config.get("seq_len"),
            "q_low": config.get("q_low"),
            "q_high": config.get("q_high"),
            "lambda_band": config.get("lambda_band"),
            "band_mode": config.get("band_mode"),
            "checkpoint_selection": config.get("checkpoint_selection"),
        },
        "risk_thresholds": thresholds,
        "validation": _eval("val", val_bundle),
        "test": _eval("test", test_bundle),
    }


def _summary_from_collected(
    *,
    predictions: Any,
    lower: torch.Tensor,
    upper: torch.Tensor,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    config = predictions.config
    return summarize_forecast_metrics(
        metadata=predictions.metadata,
        line_predictions=predictions.line,
        lower_predictions=lower,
        upper_predictions=upper,
        line_targets=predictions.line_target,
        band_targets=predictions.band_target,
        raw_future_returns=predictions.raw_future_returns,
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        q_low=float(config.get("q_low", 0.1)),
        q_high=float(config.get("q_high", 0.9)),
        severe_downside_threshold=thresholds.get("severe_downside_threshold"),
        squeeze_breakout_threshold=thresholds.get("squeeze_breakout_threshold"),
    )


def regrade_band_candidate(
    *,
    checkpoint_path: str | Path,
    limit_tickers: int | None,
    lower_scale: float | None,
    upper_scale: float | None,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
) -> dict[str, Any]:
    config = load_checkpoint_config(checkpoint_path)
    _, _, _, thresholds = _prepare_splits(config, limit_tickers=limit_tickers)

    result: dict[str, Any] = {
        "checkpoint_path": _repo_path(checkpoint_path),
        "config": {
            "model": config.get("model"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "q_low": config.get("q_low"),
            "q_high": config.get("q_high"),
            "lambda_band": config.get("lambda_band"),
            "band_mode": config.get("band_mode"),
        },
        "risk_thresholds": thresholds,
        "calibration": None,
    }
    for split in ("val", "test"):
        predictions = collect_predictions(
            checkpoint_path=Path(checkpoint_path),
            split=split,
            tickers=None,
            limit_tickers=limit_tickers,
            device_name=device_name,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
        original = _summary_from_collected(
            predictions=predictions,
            lower=predictions.lower,
            upper=predictions.upper,
            thresholds=thresholds,
        )
        result.setdefault("original", {})[split] = original
        if lower_scale is not None and upper_scale is not None:
            lower, upper = _apply_scalar_width(
                line=predictions.line,
                lower=predictions.lower,
                upper=predictions.upper,
                lower_scale=float(lower_scale),
                upper_scale=float(upper_scale),
            )
            result.setdefault("scalar_width", {})[split] = _summary_from_collected(
                predictions=predictions,
                lower=lower,
                upper=upper,
                thresholds=thresholds,
            )
            result["calibration"] = {
                "method": "scalar_width",
                "lower_scale": float(lower_scale),
                "upper_scale": float(upper_scale),
            }
    return result


def _composite_policy_bounds(
    *,
    policy: str,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy == "raw_composite":
        return lower, upper
    base_lower = torch.minimum(lower, line)
    base_upper = torch.maximum(upper, line)
    if policy == "risk_first_lower_preserve":
        return base_lower, base_upper
    if policy == "risk_first_upper_buffer_1.10":
        return base_lower, line + torch.clamp(base_upper - line, min=0.0) * 1.10
    raise ValueError(f"지원하지 않는 composite 정책입니다: {policy}")


def regrade_composite(
    *,
    line_checkpoint: str | Path,
    band_checkpoint: str | Path,
    limit_tickers: int,
    max_rows: int,
    lower_scale: float,
    upper_scale: float,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
) -> dict[str, Any]:
    line_config = load_checkpoint_config(line_checkpoint)
    band_config = load_checkpoint_config(band_checkpoint)
    _, _, _, thresholds = _prepare_splits(line_config, limit_tickers=limit_tickers)
    result: dict[str, Any] = {
        "line_checkpoint": _repo_path(line_checkpoint),
        "band_checkpoint": _repo_path(band_checkpoint),
        "limit_tickers": limit_tickers,
        "max_rows": max_rows,
        "calibration": {
            "method": "scalar_width",
            "lower_scale": float(lower_scale),
            "upper_scale": float(upper_scale),
        },
        "policies": {},
    }

    for split in ("val", "test"):
        line_predictions = collect_predictions(
            checkpoint_path=Path(line_checkpoint),
            split=split,
            tickers=None,
            limit_tickers=limit_tickers,
            device_name=device_name,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
        band_predictions = collect_predictions(
            checkpoint_path=Path(band_checkpoint),
            split=split,
            tickers=None,
            limit_tickers=limit_tickers,
            device_name=device_name,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
        selected = _select_latest_common_indices(line_predictions, band_predictions, max_rows=max_rows)
        line_indices = [line_index for line_index, _ in selected]
        band_indices = [band_index for _, band_index in selected]
        line = line_predictions.line[line_indices]
        band_line = band_predictions.line[band_indices]
        calibrated_lower, calibrated_upper = _apply_scalar_width(
            line=band_line,
            lower=band_predictions.lower[band_indices],
            upper=band_predictions.upper[band_indices],
            lower_scale=lower_scale,
            upper_scale=upper_scale,
        )
        metadata = line_predictions.metadata.iloc[line_indices].reset_index(drop=True)
        raw_width = torch.clamp(calibrated_upper - calibrated_lower, min=0.0).mean().item()
        for policy in ("raw_composite", "risk_first_lower_preserve", "risk_first_upper_buffer_1.10"):
            lower, upper = _composite_policy_bounds(
                policy=policy,
                line=line,
                lower=calibrated_lower,
                upper=calibrated_upper,
            )
            summary = summarize_forecast_metrics(
                metadata=metadata,
                line_predictions=line,
                lower_predictions=lower,
                upper_predictions=upper,
                line_targets=line_predictions.line_target[line_indices],
                band_targets=line_predictions.band_target[line_indices],
                raw_future_returns=line_predictions.raw_future_returns[line_indices],
                line_target_type=str(line_config.get("line_target_type", "raw_future_return")),
                band_target_type=str(line_config.get("band_target_type", "raw_future_return")),
                q_low=float(band_config.get("q_low", 0.1)),
                q_high=float(band_config.get("q_high", 0.9)),
                severe_downside_threshold=thresholds.get("severe_downside_threshold"),
                squeeze_breakout_threshold=thresholds.get("squeeze_breakout_threshold"),
            )
            width = float(torch.clamp(upper - lower, min=0.0).mean().item())
            summary["composite_width_increase_ratio"] = width / raw_width if raw_width else None
            result["policies"].setdefault(policy, {})[split] = summary
        result[f"{split}_row_count"] = len(selected)
    return result


def _load_cp49_line_candidates() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "docs" / "cp49_patchtst_horizon_rescue_metrics.json"
    if not path.exists():
        return []
    data = _read_json(path)
    return [
        {
            "name": record["candidate"]["name"],
            "branch": f"h{record['candidate']['horizon']}",
            "checkpoint_path": record["checkpoint_path"],
            "candidate": record["candidate"],
            "source": _repo_path(path),
            "limit_tickers": 50,
        }
        for record in data.get("records", [])
        if record.get("checkpoint_path")
    ]


def _band_records_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    records = []
    for record in data.get("records", []):
        train = record.get("train") or {}
        checkpoint_path = train.get("checkpoint_path")
        if not checkpoint_path:
            continue
        calibration = record.get("calibration") or {}
        records.append(
            {
                "name": record.get("candidate", {}).get("name"),
                "checkpoint_path": checkpoint_path,
                "candidate": record.get("candidate", {}),
                "source": _repo_path(path),
                "limit_tickers": int(data.get("limit_tickers") or 100),
                "lower_scale": calibration.get("lower_scale"),
                "upper_scale": calibration.get("upper_scale"),
            }
        )
    return records


def _load_band_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in (
        PROJECT_ROOT / "docs" / "cp45_cnn_lstm_band_sweep_metrics.json",
        PROJECT_ROOT / "logs" / "cp45" / "cp45_actual_metrics.json",
        PROJECT_ROOT / "logs" / "cp45" / "cp45_188_confirm_metrics.json",
    ):
        for record in _band_records_from_file(path):
            key = (str(record["name"]), int(record["limit_tickers"]))
            if key in seen:
                continue
            seen.add(key)
            if record["name"] in {"s60_q20_b2_direct", "s60_q15_b2_direct", "s60_q15_b2_direct_188"}:
                candidates.append(record)

    cp38_path = PROJECT_ROOT / "docs" / "cp38_band_calibration_gap_metrics.json"
    if cp38_path.exists():
        data = _read_json(cp38_path)
        tide = data.get("candidates", {}).get("tide_param")
        if tide:
            calibration = (tide.get("scalar_width") or {}).get("calibration") or {}
            candidates.append(
                {
                    "name": "tide_param_scalar_width",
                    "checkpoint_path": tide.get("checkpoint_path"),
                    "candidate": tide.get("config", {}),
                    "source": _repo_path(cp38_path),
                    "limit_tickers": 50,
                    "lower_scale": calibration.get("lower_scale"),
                    "upper_scale": calibration.get("upper_scale"),
                }
            )

    cp37_path = PROJECT_ROOT / "docs" / "cp37_role_based_model_recheck_metrics.json"
    if cp37_path.exists():
        data = _read_json(cp37_path)
        tide_direct = data.get("runs", {}).get("C_tide_band_gate_direct")
        if tide_direct:
            candidates.append(
                {
                    "name": "tide_direct_original",
                    "checkpoint_path": tide_direct.get("checkpoint_path"),
                    "candidate": tide_direct.get("config", {}),
                    "source": _repo_path(cp37_path),
                    "limit_tickers": 50,
                    "lower_scale": None,
                    "upper_scale": None,
                }
            )
    return [candidate for candidate in candidates if candidate.get("checkpoint_path")]


def build_regrade_payload(args: argparse.Namespace) -> dict[str, Any]:
    line_candidates = _load_cp49_line_candidates()
    band_candidates = _load_band_candidates()

    payload: dict[str, Any] = {
        "cp": "CP53-M",
        "purpose": "CP52 표준 지표판으로 기존 후보 재채점",
        "rules": {
            "new_training": False,
            "save_run": False,
            "ui_change": False,
            "schema_change": False,
            "model_structure_change": False,
            "fake_data": False,
        },
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "line_candidates": [],
        "band_candidates": [],
        "composite_policies": None,
        "limitations": [],
    }

    for candidate in line_candidates:
        regraded = regrade_checkpoint(
            checkpoint_path=candidate["checkpoint_path"],
            limit_tickers=int(candidate["limit_tickers"]),
            device_name=args.device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        )
        test = regraded["test"]
        payload["line_candidates"].append(
            {
                **candidate,
                "validation": _metric_subset(regraded["validation"], LINE_METRIC_KEYS),
                "test": _metric_subset(test, LINE_METRIC_KEYS),
                "test_band_reference": _metric_subset(test, BAND_METRIC_KEYS),
                "status": _candidate_status("line", test),
                "risk_thresholds": regraded["risk_thresholds"],
            }
        )

    for candidate in band_candidates:
        regraded = regrade_band_candidate(
            checkpoint_path=candidate["checkpoint_path"],
            limit_tickers=int(candidate["limit_tickers"]),
            lower_scale=candidate.get("lower_scale"),
            upper_scale=candidate.get("upper_scale"),
            device_name=args.device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        )
        source_block = "scalar_width" if regraded.get("scalar_width") else "original"
        test = regraded[source_block]["test"]
        payload["band_candidates"].append(
            {
                **candidate,
                "evaluation_block": source_block,
                "validation": _metric_subset(regraded[source_block]["val"], BAND_METRIC_KEYS),
                "test": _metric_subset(test, BAND_METRIC_KEYS),
                "line_reference": _metric_subset(test, LINE_METRIC_KEYS),
                "status": _candidate_status("band", test),
                "risk_thresholds": regraded["risk_thresholds"],
            }
        )

    cp46_path = PROJECT_ROOT / "docs" / "cp46_composite_upper_calibration_metrics.json"
    if cp46_path.exists():
        cp46 = _read_json(cp46_path)
        calibration = cp46.get("band_scalar_calibration") or {}
        payload["composite_policies"] = regrade_composite(
            line_checkpoint=cp46["line_checkpoint"],
            band_checkpoint=cp46["band_checkpoint"],
            limit_tickers=int(cp46.get("limit_tickers", 200)),
            max_rows=int(cp46.get("max_rows", 200)),
            lower_scale=float(calibration["lower_scale"]),
            upper_scale=float(calibration["upper_scale"]),
            device_name=args.device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        )
    else:
        payload["limitations"].append("CP46 composite 산출물이 없어 composite 정책 재채점은 제외했습니다.")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP53 기존 후보 CP52 지표판 재채점")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-json", default="docs/cp53_existing_candidate_regrade_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_regrade_payload(args)
    output_path = PROJECT_ROOT / args.output_json
    _write_json(output_path, payload)
    summary = {
        "output_json": _repo_path(output_path),
        "line_candidates": len(payload["line_candidates"]),
        "band_candidates": len(payload["band_candidates"]),
        "composite_policies": list((payload.get("composite_policies") or {}).get("policies", {}).keys()),
    }
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    # Windows CUDA 종료가 늦어지는 경우가 있어 결과 출력 뒤 참조를 명시적으로 해제한다.
    payload = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
