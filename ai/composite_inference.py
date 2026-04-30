from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.train import autocast_context, forward_model, make_loader, resolve_device, torch

import pandas as pd

from ai.evaluation import build_single_sample_evaluation, summarize_forecast_metrics
from ai.inference import (
    decode_return_forecasts,
    load_checkpoint,
    load_checkpoint_config,
    resolve_checkpoint_ticker_registry,
)
from ai.loss import PinballLoss
from ai.postprocess import apply_band_postprocess
from ai.preprocessing import FEATURE_CONTRACT_VERSION
from ai.preprocessing import SequenceDataset, SequenceDatasetBundle, prepare_dataset_splits
from ai.storage import (
    get_model_run,
    save_model_run,
    save_prediction_evaluations,
    save_predictions,
    utc_now_iso,
)


COMPOSITION_VERSION = "line_band_v1"
COMPOSITION_POLICIES = ("raw_composite", "include_line_clamp", "risk_first_lower_preserve")


@dataclass
class CollectedPredictions:
    checkpoint_path: Path
    config: dict[str, Any]
    line: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    line_target: torch.Tensor
    band_target: torch.Tensor
    raw_future_returns: torch.Tensor
    anchor_closes: torch.Tensor
    metadata: pd.DataFrame


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _forecast_key(row: pd.Series) -> tuple[str, str, tuple[str, ...]]:
    forecast_dates = row["forecast_dates"]
    if isinstance(forecast_dates, str):
        try:
            parsed = json.loads(forecast_dates)
            forecast_dates = parsed if isinstance(parsed, list) else [forecast_dates]
        except json.JSONDecodeError:
            forecast_dates = [forecast_dates]
    return (
        str(row["ticker"]),
        str(row["asof_date"]),
        tuple(str(item) for item in forecast_dates),
    )


def _anchor_closes_for_batch(
    bundle: SequenceDatasetBundle | SequenceDataset,
    *,
    offset: int,
    batch_size: int,
) -> torch.Tensor:
    if isinstance(bundle, SequenceDataset):
        anchors = [
            float(bundle.ticker_arrays[bundle.sample_refs[offset + index][0]]["closes"][bundle.sample_refs[offset + index][1]])
            for index in range(batch_size)
        ]
        return torch.tensor(anchors, dtype=torch.float32)
    return bundle.anchor_closes[offset : offset + batch_size].detach().cpu().to(dtype=torch.float32)


def _load_bundle_for_checkpoint(
    *,
    config: dict[str, Any],
    split: str,
    tickers: list[str] | None,
    limit_tickers: int | None,
) -> SequenceDatasetBundle | SequenceDataset:
    ticker_registry = resolve_checkpoint_ticker_registry(config, str(config["timeframe"]))
    train_bundle, val_bundle, test_bundle, _, _, _ = prepare_dataset_splits(
        timeframe=str(config["timeframe"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        tickers=tickers,
        limit_tickers=limit_tickers if tickers is None else None,
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=ticker_registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    return {"train": train_bundle, "val": val_bundle, "test": test_bundle}[split]


def collect_predictions(
    *,
    checkpoint_path: Path,
    split: str,
    tickers: list[str] | None,
    limit_tickers: int | None,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
) -> CollectedPredictions:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    bundle = _load_bundle_for_checkpoint(
        config=config,
        split=split,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=0)

    line_chunks: list[torch.Tensor] = []
    lower_chunks: list[torch.Tensor] = []
    upper_chunks: list[torch.Tensor] = []
    line_target_chunks: list[torch.Tensor] = []
    band_target_chunks: list[torch.Tensor] = []
    raw_target_chunks: list[torch.Tensor] = []
    anchor_chunks: list[torch.Tensor] = []

    offset = 0
    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_ids, future_covariates in loader:
            batch_size_now = int(features.shape[0])
            features = features.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype=amp_dtype):
                output = forward_model(model, features, ticker_ids, future_covariates)
            line, lower, upper = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            line_chunks.append(line)
            lower_chunks.append(lower)
            upper_chunks.append(upper)
            line_target_chunks.append(line_target.detach().cpu())
            band_target_chunks.append(band_target.detach().cpu())
            raw_target_chunks.append(raw_future_returns.detach().cpu())
            anchor_chunks.append(_anchor_closes_for_batch(bundle, offset=offset, batch_size=batch_size_now))
            offset += batch_size_now

    return CollectedPredictions(
        checkpoint_path=checkpoint_path,
        config=config,
        line=torch.cat(line_chunks, dim=0),
        lower=torch.cat(lower_chunks, dim=0),
        upper=torch.cat(upper_chunks, dim=0),
        line_target=torch.cat(line_target_chunks, dim=0),
        band_target=torch.cat(band_target_chunks, dim=0),
        raw_future_returns=torch.cat(raw_target_chunks, dim=0),
        anchor_closes=torch.cat(anchor_chunks, dim=0),
        metadata=bundle.metadata.reset_index(drop=True),
    )


def _assert_compatible(line_config: dict[str, Any], band_config: dict[str, Any]) -> None:
    required_equal = ("timeframe", "horizon", "line_target_type", "band_target_type")
    mismatches = {
        key: (line_config.get(key), band_config.get(key))
        for key in required_equal
        if str(line_config.get(key)) != str(band_config.get(key))
    }
    if mismatches:
        raise ValueError(f"line/band checkpoint 계약 불일치: {mismatches}")
    if str(line_config.get("line_target_type")) != "raw_future_return":
        raise ValueError("CP40 조합 저장 smoke는 raw_future_return checkpoint만 허용합니다.")
    if str(line_config.get("band_target_type")) != "raw_future_return":
        raise ValueError("CP40 조합 저장 smoke는 raw_future_return band checkpoint만 허용합니다.")


def _resolve_model_run_ref(config: dict[str, Any], checkpoint_path: Path) -> str:
    run_id = config.get("run_id")
    if run_id:
        return str(run_id)
    return f"checkpoint:{checkpoint_path.stem}"


def _apply_scalar_width(
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lower_scale: float,
    upper_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_width = torch.clamp(line - lower, min=1e-6)
    upper_width = torch.clamp(upper - line, min=1e-6)
    calibrated_lower = line - lower_width * float(lower_scale)
    calibrated_upper = line + upper_width * float(upper_scale)
    return torch.minimum(calibrated_lower, calibrated_upper), torch.maximum(calibrated_lower, calibrated_upper)


def _apply_composition_policy(
    *,
    policy: str,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy == "raw_composite":
        return lower, upper
    if policy == "include_line_clamp":
        return torch.minimum(lower, line), torch.maximum(upper, line)
    if policy == "risk_first_lower_preserve":
        # 하방 밴드는 더 보수적인 쪽을 택한다. 즉 line이 lower보다 낮으면 lower를 끌어올리지 않고 line까지 확장한다.
        return torch.minimum(lower, line), torch.maximum(upper, line)
    raise ValueError(f"지원하지 않는 composition policy입니다: {policy}")


def _select_latest_common_indices(
    line_predictions: CollectedPredictions,
    band_predictions: CollectedPredictions,
    *,
    max_rows: int,
) -> list[tuple[int, int]]:
    line_key_to_index = {
        _forecast_key(row): index
        for index, row in line_predictions.metadata.reset_index(drop=True).iterrows()
    }
    band_key_to_index = {
        _forecast_key(row): index
        for index, row in band_predictions.metadata.reset_index(drop=True).iterrows()
    }
    common_keys = set(line_key_to_index) & set(band_key_to_index)
    if not common_keys:
        raise ValueError("line/band checkpoint의 ticker/asof_date/forecast_dates 교집합이 없습니다.")

    latest_by_ticker: dict[str, tuple[str, tuple[str, ...]]] = {}
    for ticker, asof_date, forecast_dates in common_keys:
        current = latest_by_ticker.get(ticker)
        if current is None or asof_date > current[0]:
            latest_by_ticker[ticker] = (asof_date, forecast_dates)

    selected_keys = [
        (ticker, asof_date, forecast_dates)
        for ticker, (asof_date, forecast_dates) in sorted(latest_by_ticker.items())
    ][:max_rows]
    return [(line_key_to_index[key], band_key_to_index[key]) for key in selected_keys]


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        if value != value:
            return None
        if value == float("inf") or value == float("-inf"):
            return None
        return value
    return value


def _resolve_checkpoint_from_run_id(run_id: str) -> tuple[Path, dict[str, Any]]:
    model_run = get_model_run(run_id)
    if model_run is None:
        raise ValueError(f"model_runs에서 run_id={run_id}를 찾지 못했습니다.")
    status = str(model_run.get("status") or "")
    if status != "completed":
        raise ValueError(f"run_id={run_id} status={status}: completed run만 composite에 사용할 수 있습니다.")
    checkpoint_path = model_run.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError(f"run_id={run_id}에는 checkpoint_path가 없습니다.")
    resolved = Path(str(checkpoint_path))
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint_path가 존재하지 않습니다: {resolved}")
    return resolved, model_run


def _prediction_evaluation_records(
    *,
    run_id: str,
    metadata: pd.DataFrame,
    anchor_closes: torch.Tensor,
    line_returns: torch.Tensor,
    lower_returns: torch.Tensor,
    upper_returns: torch.Tensor,
    band_targets: torch.Tensor,
    q_low: float,
    q_high: float,
    line_target_type: str,
) -> list[dict[str, Any]]:
    pinball = PinballLoss((q_low, 0.5, q_high), sort_quantiles=True)
    records: list[dict[str, Any]] = []
    for row_index, row in metadata.iterrows():
        quantile_tensor = torch.stack(
            (lower_returns[row_index], line_returns[row_index], upper_returns[row_index]),
            dim=-1,
        ).unsqueeze(0)
        target_tensor = band_targets[row_index].unsqueeze(0)
        actual_prices = (anchor_closes[row_index] * (1.0 + band_targets[row_index])).tolist()
        records.append(
            {
                "run_id": run_id,
                "ticker": str(row["ticker"]),
                "timeframe": str(row["timeframe"]),
                "asof_date": str(row["asof_date"]),
                "actual_series": actual_prices,
                "pinball_loss": float(pinball(quantile_tensor, target_tensor).item()),
                **build_single_sample_evaluation(
                    actual_series=_to_float_list(band_targets[row_index]),
                    line_series=_to_float_list(line_returns[row_index]),
                    lower_series=_to_float_list(lower_returns[row_index]),
                    upper_series=_to_float_list(upper_returns[row_index]),
                    line_target_type=line_target_type,
                ),
                "normalized_band_width": float((upper_returns[row_index] - lower_returns[row_index]).mean().item()),
            }
        )
    return records


def _save_composite_run(
    *,
    run_id: str,
    line_config: dict[str, Any],
    band_config: dict[str, Any],
    line_model_run_id: str,
    band_model_run_id: str,
    summary: dict[str, float | None],
    calibration_params: dict[str, float],
    composition_policy: str,
) -> None:
    save_model_run(
        {
            "run_id": run_id,
            "wandb_run_id": None,
            "model_name": "line_band_composite",
            "timeframe": str(line_config["timeframe"]),
            "horizon": int(line_config["horizon"]),
            "feature_version": str(line_config.get("feature_version") or FEATURE_CONTRACT_VERSION),
            "band_quantile_low": float(band_config["q_low"]),
            "band_quantile_high": float(band_config["q_high"]),
            "band_mode": f"composite_{band_config.get('band_mode', 'direct')}",
            "alpha": None,
            "beta": None,
            "huber_delta": None,
            "lambda_line": None,
            "lambda_band": None,
            "lambda_width": None,
            "lambda_cross": None,
            "train_start": None,
            "train_end": None,
            "val_metrics": {},
            "test_metrics": summary,
            "config": {
                "role": "composite_model",
                "composition_policy": composition_policy,
                "line_model_run_id": line_model_run_id,
                "band_model_run_id": band_model_run_id,
                "band_calibration_method": "scalar_width",
                "band_calibration_params": calibration_params,
                "prediction_composition_version": COMPOSITION_VERSION,
            },
            "checkpoint_path": None,
            "status": "completed",
        }
    )


def build_composite_contract(
    *,
    line_checkpoint: Path,
    band_checkpoint: Path,
    line_model_run_id_override: str | None = None,
    band_model_run_id_override: str | None = None,
    composition_run_id: str | None = None,
    split: str,
    tickers: list[str] | None,
    limit_tickers: int | None,
    max_rows: int,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
    lower_scale: float,
    upper_scale: float,
    output_json: Path,
    composition_policy: str = "risk_first_lower_preserve",
    save: bool = False,
) -> dict[str, Any]:
    if composition_policy not in COMPOSITION_POLICIES:
        raise ValueError(f"지원하지 않는 composition policy입니다: {composition_policy}")
    line_config = load_checkpoint_config(line_checkpoint)
    band_config = load_checkpoint_config(band_checkpoint)
    _assert_compatible(line_config, band_config)

    line_predictions = collect_predictions(
        checkpoint_path=line_checkpoint,
        split=split,
        tickers=tickers,
        limit_tickers=limit_tickers,
        device_name=device_name,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    band_predictions = collect_predictions(
        checkpoint_path=band_checkpoint,
        split=split,
        tickers=tickers,
        limit_tickers=limit_tickers,
        device_name=device_name,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )

    selected_indices = _select_latest_common_indices(line_predictions, band_predictions, max_rows=max_rows)
    line_indices = [line_index for line_index, _ in selected_indices]
    band_indices = [band_index for _, band_index in selected_indices]

    selected_line_returns = line_predictions.line[line_indices]
    selected_line_targets = line_predictions.line_target[line_indices]
    selected_band_targets = line_predictions.band_target[line_indices]
    selected_raw_returns = line_predictions.raw_future_returns[line_indices]
    selected_anchor_closes = line_predictions.anchor_closes[line_indices]
    selected_metadata = line_predictions.metadata.iloc[line_indices].reset_index(drop=True)

    band_line = band_predictions.line[band_indices]
    band_lower = band_predictions.lower[band_indices]
    band_upper = band_predictions.upper[band_indices]
    calibrated_lower, calibrated_upper = _apply_scalar_width(
        line=band_line,
        lower=band_lower,
        upper=band_upper,
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )
    policy_lower, policy_upper = _apply_composition_policy(
        policy=composition_policy,
        line=selected_line_returns,
        lower=calibrated_lower,
        upper=calibrated_upper,
    )

    line_prices, lower_prices, upper_prices = decode_return_forecasts(
        selected_line_returns,
        policy_lower,
        policy_upper,
        selected_anchor_closes,
    )

    line_model_run_id = line_model_run_id_override or _resolve_model_run_ref(line_config, line_checkpoint)
    band_model_run_id = band_model_run_id_override or _resolve_model_run_ref(band_config, band_checkpoint)
    if composition_run_id is None:
        if save:
            composition_run_id = f"composite-{line_config['timeframe']}-{uuid4().hex[:12]}"
        else:
            composition_run_id = f"composite-smoke:{line_checkpoint.stem}:{band_checkpoint.stem}"
    calibration_params = {
        "target_coverage": 0.85,
        "target_each_tail_breach": 0.075,
        "lower_scale": float(lower_scale),
        "upper_scale": float(upper_scale),
    }

    records: list[dict[str, Any]] = []
    lower_le_upper_flags: list[bool] = []
    line_inside_band_flags: list[bool] = []
    for row_index, row in selected_metadata.iterrows():
        lower_tensor = policy_lower[row_index]
        upper_tensor = policy_upper[row_index]
        line_tensor = selected_line_returns[row_index]
        lower_le_upper = bool(torch.all(lower_tensor <= upper_tensor).item())
        line_inside_band = bool(torch.all((lower_tensor <= line_tensor) & (line_tensor <= upper_tensor)).item())
        lower_le_upper_flags.append(lower_le_upper)
        line_inside_band_flags.append(line_inside_band)

        record_meta = {
            "composition_policy": composition_policy,
            "line_model_run_id": line_model_run_id,
            "band_model_run_id": band_model_run_id,
            "line_model_name": str(line_config["model"]),
            "band_model_name": str(band_config["model"]),
            "line_checkpoint_path": _repo_path(line_checkpoint),
            "band_checkpoint_path": _repo_path(band_checkpoint),
            "band_calibration_method": "scalar_width",
            "band_calibration_params": calibration_params,
            "prediction_composition_version": COMPOSITION_VERSION,
            "line_seq_len": int(line_config["seq_len"]),
            "band_seq_len": int(band_config["seq_len"]),
            "target_contract": {
                "timeframe": str(line_config["timeframe"]),
                "horizon": int(line_config["horizon"]),
                "line_target_type": str(line_config["line_target_type"]),
                "band_target_type": str(line_config["band_target_type"]),
            },
            "validation": {
                "lower_le_upper": lower_le_upper,
                "line_inside_band": line_inside_band,
                "line_inside_band_is_guardrail_only": True,
            },
        }
        records.append(
            {
                "ticker": str(row["ticker"]),
                "model_name": "patchtst_line__cnn_lstm_calibrated_band",
                "timeframe": str(line_config["timeframe"]),
                "horizon": int(line_config["horizon"]),
                "asof_date": str(row["asof_date"]),
                "decision_time": utc_now_iso(),
                "run_id": composition_run_id,
                "model_ver": COMPOSITION_VERSION,
                "signal": "HOLD",
                "forecast_dates": [str(item) for item in row["forecast_dates"]],
                "line_series": line_prices[row_index],
                "conservative_series": lower_prices[row_index],
                "lower_band_series": lower_prices[row_index],
                "upper_band_series": upper_prices[row_index],
                "band_quantile_low": float(band_config["q_low"]),
                "band_quantile_high": float(band_config["q_high"]),
                "meta": record_meta,
            }
        )

    summary = summarize_forecast_metrics(
        metadata=selected_metadata,
        line_predictions=selected_line_returns,
        lower_predictions=policy_lower,
        upper_predictions=policy_upper,
        line_targets=selected_line_targets,
        band_targets=selected_band_targets,
        raw_future_returns=selected_raw_returns,
        line_target_type=str(line_config["line_target_type"]),
        band_target_type=str(line_config["band_target_type"]),
        q_low=float(band_config["q_low"]),
        q_high=float(band_config["q_high"]),
        severe_downside_threshold=line_config.get("severe_downside_threshold"),
        squeeze_breakout_threshold=band_config.get("squeeze_breakout_threshold") or line_config.get("squeeze_breakout_threshold"),
    )
    evaluation_records = _prediction_evaluation_records(
        run_id=composition_run_id,
        metadata=selected_metadata,
        anchor_closes=selected_anchor_closes,
        line_returns=selected_line_returns,
        lower_returns=policy_lower,
        upper_returns=policy_upper,
        band_targets=selected_band_targets,
        q_low=float(band_config["q_low"]),
        q_high=float(band_config["q_high"]),
        line_target_type=str(line_config["line_target_type"]),
    )

    if save:
        _save_composite_run(
            run_id=composition_run_id,
            line_config=line_config,
            band_config=band_config,
            line_model_run_id=line_model_run_id,
            band_model_run_id=band_model_run_id,
            summary=summary,
            calibration_params=calibration_params,
            composition_policy=composition_policy,
        )
        save_predictions(records)
        save_prediction_evaluations(evaluation_records)

    result = {
        "contract": "line_band_composite_inference",
        "composition_version": COMPOSITION_VERSION,
        "storage_contract": {
            "repository_schema_supports_predictions_meta": True,
            "runtime_db_meta_column_verified": False,
            "runtime_db_migration_required": "backend.db.scripts.ensure_runtime_schema 1회 실행 필요",
            "recommended_minimal_change": "predictions.meta JSONB NOT NULL DEFAULT '{}'::jsonb",
            "why_not_encode_in_existing_columns": "model_name/model_ver/run_id에 조합 메타를 문자열로 밀어 넣으면 검색성과 검증성이 떨어지고 calibration params를 안전하게 보존하기 어렵다.",
            "schema_columns_option": "line_model_run_id 등 개별 컬럼 추가는 쿼리에는 좋지만 CP40 최소 변경 범위를 넘는다.",
        },
        "line_checkpoint": _repo_path(line_checkpoint),
        "band_checkpoint": _repo_path(band_checkpoint),
        "line_model_run_id": line_model_run_id,
        "band_model_run_id": band_model_run_id,
        "composition_policy": composition_policy,
        "composition_run_id": composition_run_id,
        "saved_to_db": save,
        "split": split,
        "tickers": tickers,
        "max_rows": max_rows,
        "row_count": len(records),
        "contract_checks": {
            "lower_le_upper_all": all(lower_le_upper_flags),
            "line_inside_band_all": all(line_inside_band_flags),
            "line_inside_band_required": composition_policy != "raw_composite",
            "line_inside_band_failures_allowed": composition_policy == "raw_composite",
            "line_inside_band_ratio": (
                sum(1 for flag in line_inside_band_flags if flag) / len(line_inside_band_flags)
                if line_inside_band_flags
                else None
            ),
            "series_length_all_5": all(
                len(record["line_series"]) == 5
                and len(record["lower_band_series"]) == 5
                and len(record["upper_band_series"]) == 5
                and len(record["conservative_series"]) == 5
                for record in records
            ),
            "metadata_required_fields_present": all(
                all(
                    key in record["meta"]
                    for key in (
                        "line_model_run_id",
                        "band_model_run_id",
                        "composition_policy",
                        "line_model_name",
                        "band_model_name",
                        "band_calibration_method",
                        "band_calibration_params",
                        "prediction_composition_version",
                    )
                )
                for record in records
            ),
        },
        "calibration": {
            "method": "scalar_width",
            "params": calibration_params,
        },
        "metrics": summary,
        "prediction_records_smoke": records,
        "prediction_evaluations_smoke": evaluation_records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST line과 CNN-LSTM 보정 band 조합 smoke")
    parser.add_argument("--line-checkpoint", default=None)
    parser.add_argument("--band-checkpoint", default=None)
    parser.add_argument("--line-run-id", default=None)
    parser.add_argument("--band-run-id", default=None)
    parser.add_argument("--composition-run-id", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", default="off", choices=["off", "bf16", "fp16"])
    parser.add_argument("--lower-scale", type=float, required=True)
    parser.add_argument("--upper-scale", type=float, required=True)
    parser.add_argument("--composition-policy", default="risk_first_lower_preserve", choices=COMPOSITION_POLICIES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.line_checkpoint and not args.line_run_id:
        raise ValueError("--line-checkpoint 또는 --line-run-id 중 하나가 필요합니다.")
    if not args.band_checkpoint and not args.band_run_id:
        raise ValueError("--band-checkpoint 또는 --band-run-id 중 하나가 필요합니다.")

    line_model_run = None
    band_model_run = None
    line_checkpoint = Path(args.line_checkpoint) if args.line_checkpoint else None
    band_checkpoint = Path(args.band_checkpoint) if args.band_checkpoint else None
    if args.line_run_id:
        line_checkpoint, line_model_run = _resolve_checkpoint_from_run_id(args.line_run_id)
    if args.band_run_id:
        band_checkpoint, band_model_run = _resolve_checkpoint_from_run_id(args.band_run_id)
    if line_checkpoint is None or band_checkpoint is None:
        raise ValueError("line/band checkpoint 해석에 실패했습니다.")

    result = build_composite_contract(
        line_checkpoint=line_checkpoint,
        band_checkpoint=band_checkpoint,
        line_model_run_id_override=args.line_run_id or (str(line_model_run["run_id"]) if line_model_run else None),
        band_model_run_id_override=args.band_run_id or (str(band_model_run["run_id"]) if band_model_run else None),
        composition_run_id=args.composition_run_id,
        split=args.split,
        tickers=args.tickers,
        limit_tickers=args.limit_tickers,
        max_rows=args.max_rows,
        device_name=args.device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        lower_scale=args.lower_scale,
        upper_scale=args.upper_scale,
        output_json=Path(args.output_json),
        composition_policy=args.composition_policy,
        save=args.save,
    )
    print(json.dumps(_json_safe({
        "row_count": result["row_count"],
        "composition_run_id": result["composition_run_id"],
        "composition_policy": result["composition_policy"],
        "saved_to_db": result["saved_to_db"],
        "contract_checks": result["contract_checks"],
        "metrics": result["metrics"],
        "output_json": args.output_json,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
