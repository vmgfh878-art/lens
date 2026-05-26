from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.loss import PinballLoss
from ai.models.cnn_lstm import CNNLSTM
from ai.models.common import BandOutput, ForecastOutput, LineV2Output
from ai.models.patchtst import PatchTST
from ai.models.tide import TiDE
from ai.calibration_artifacts import apply_product_band_calibration, resolve_product_band_calibration
from ai.evaluation import (
    build_single_sample_evaluation,
    summarize_band_metrics,
    summarize_forecast_metrics,
    summarize_line_v2_metrics,
)
from ai.inference_contract import (
    INFERENCE_PAYLOAD_SUPPORTED_ROLES,
    assert_output_matches_model_role as contract_assert_output_matches_model_role,
    require_role_supported,
    resolve_execution_market_data_provider,
    resolve_model_role_from_config,
    role_contract_table,
    select_bundle_features_for_checkpoint,
    select_bundle_features_for_columns,
    validate_checkpoint_runtime_contract,
)
from ai.inference_save_guard import resolve_inference_save_contract
from ai.postprocess import apply_band_postprocess
from ai.preprocessing import (
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SequenceDataset,
    SequenceDatasetBundle,
    normalize_ai_timeframe,
    prepare_dataset_splits,
)
from ai.storage import (
    STORAGE_CONTRACT_EVALUATION_BULK,
    get_model_run,
    save_prediction_evaluations,
    save_predictions,
    save_product_latest_predictions,
    utc_now_iso,
    with_prediction_storage_contract,
)
from ai.ticker_registry import load_registry
from ai.train import make_loader, resolve_device

MODEL_REGISTRY = {
    "patchtst": PatchTST,
    "cnn_lstm": CNNLSTM,
    "tide": TiDE,
}
MODEL_ROLE_ALIASES = {
    "legacy": "legacy",
    "forecast": "legacy",
    "multihead": "legacy",
    "line": "line_v2",
    "line_v2": "line_v2",
    "line_model": "line_v2",
    "conservative_line": "line_v2",
    "line_regime": "line_regime",
    "regime": "line_regime",
    "line_warning": "line_warning",
    "warning": "line_warning",
    "line_distributional": "line_distributional",
    "distributional": "line_distributional",
    "line_v3": "line_distributional",
    "line_distributional_mono": "line_distributional_mono",
    "distributional_mono": "line_distributional_mono",
    "line_v3_mono": "line_distributional_mono",
    "band": "band",
    "band_model": "band",
}
def resolve_checkpoint_model_role(config: dict[str, Any]) -> str:
    raw_role_value = config.get("model_role") or config.get("output_role")
    if raw_role_value is None and str(config.get("role") or "").strip():
        raw_role_value = config.get("role")
    raw_role = str(raw_role_value or "legacy").strip().lower()
    model_role = MODEL_ROLE_ALIASES.get(raw_role)
    if model_role is None:
        raise ValueError(f"지원하지 않는 model_role입니다: {raw_role}")
    return model_role


def assert_output_matches_model_role(output: object, model_role: str) -> None:
    contract_assert_output_matches_model_role(output, model_role, context="inference.output")
    return
    if model_role == "line_v2":
        if not isinstance(output, LineV2Output):
            raise TypeError("model_role=line_v2 inference는 LineV2Output만 허용합니다.")
        return
    if model_role == "band":
        if not isinstance(output, BandOutput):
            raise TypeError("model_role=band inference는 BandOutput만 허용합니다.")
        return
    if model_role == "legacy":
        if not isinstance(output, ForecastOutput):
            raise TypeError("model_role=legacy inference는 ForecastOutput만 허용합니다.")
        return
    raise ValueError(f"지원하지 않는 model_role입니다: {model_role}")


def should_use_cuda_optimizations(device: torch.device) -> bool:
    return device.type == "cuda"


def maybe_compile_model(model, device: torch.device):
    if not should_use_cuda_optimizations(device) or not hasattr(torch, "compile"):
        return model
    if importlib.util.find_spec("triton") is None:
        print("triton이 없어 inference torch.compile을 건너뜁니다.")
        return model
    return torch.compile(model, mode="reduce-overhead")


def autocast_context(device: torch.device):
    if not should_use_cuda_optimizations(device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 Lens 모델로 예측 결과를 생성합니다.")
    parser.add_argument("--run-id", required=True, help="model_runs에 저장한 run_id")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="predictions와 prediction_evaluations에 저장합니다.")
    parser.add_argument(
        "--save-product-latest-only",
        action="store_true",
        help="제품 화면용 latest-only helper로 저장합니다. ticker/timeframe/horizon/layer별 최신 row만 남깁니다.",
    )
    parser.add_argument(
        "--allow-bulk-evaluation-save",
        action="store_true",
        help="평가/레거시 목적의 bulk predictions 저장을 명시적으로 허용합니다.",
    )
    parser.add_argument(
        "--disable-band-calibration",
        action="store_true",
        help="제품 band inference에서 calibration artifact 자동 적용을 끕니다. 저장 meta에는 calibration_disabled로 기록됩니다.",
    )
    args = parser.parse_args()
    if args.save_product_latest_only:
        args.save = True
    return args


def load_checkpoint(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model_role = resolve_checkpoint_model_role(config)
    model_cls = MODEL_REGISTRY[config["model"]]
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    n_features = int(config.get("n_features") or len(feature_columns) or MODEL_N_FEATURES)
    model_kwargs = {
        "n_features": n_features,
        "seq_len": config["seq_len"],
        "horizon": config["horizon"],
        "dropout": config["dropout"],
        "band_mode": config.get("band_mode", "direct"),
        "num_tickers": config.get("num_tickers", 0),
        "ticker_emb_dim": config.get("ticker_emb_dim", 32),
        "output_role": model_role,
    }
    if config["model"] == "patchtst":
        model_kwargs["use_revin"] = bool(config.get("use_revin", True))
        model_kwargs["ci_aggregate"] = config.get("ci_aggregate", "target")
        model_kwargs["target_channel_idx"] = config.get("target_channel_idx", 0)
        model_kwargs["ci_target_fast"] = bool(config.get("ci_target_fast", False))
        model_kwargs["patch_len"] = int(config.get("patch_len", 16))
        model_kwargs["stride"] = int(config.get("patch_stride", config.get("stride", 8)))
        model_kwargs["d_model"] = int(config.get("patchtst_d_model", 128))
        model_kwargs["n_heads"] = int(config.get("patchtst_n_heads", 8))
        model_kwargs["n_layers"] = int(config.get("patchtst_n_layers", 3))
    if config["model"] == "cnn_lstm":
        model_kwargs["use_direction_head"] = bool(config.get("use_direction_head", False))
        model_kwargs["fp32_modules"] = str(config.get("fp32_modules", "none"))
    if config["model"] == "tide":
        use_future_covariate = bool(config.get("use_future_covariate", True))
        model_kwargs["future_cov_dim"] = config.get("future_cov_dim", FUTURE_COVARIATE_DIM) if use_future_covariate else 0
    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _select_bundle_features(
    bundle: SequenceDatasetBundle | SequenceDataset,
    columns: list[str] | None,
) -> SequenceDatasetBundle | SequenceDataset:
    if not columns:
        return bundle
    return select_bundle_features_for_columns(bundle, list(columns)).bundle


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return dict(checkpoint.get("config") or {})


def resolve_checkpoint_ticker_registry(checkpoint_config: dict[str, Any], timeframe: str) -> dict[str, Any] | None:
    num_tickers = int(checkpoint_config.get("num_tickers") or 0)
    if num_tickers <= 0:
        return None
    registry_path = checkpoint_config.get("ticker_registry_path")
    if not registry_path:
        raise ValueError("ticker embedding checkpoint는 ticker_registry_path가 필요합니다.")

    normalized_timeframe = normalize_ai_timeframe(timeframe)
    registry = load_registry(normalized_timeframe, Path(str(registry_path)))
    registry_timeframe = str(registry.get("timeframe", "")).upper()
    if registry_timeframe != normalized_timeframe:
        raise ValueError(
            f"checkpoint ticker registry timeframe 불일치: checkpoint={normalized_timeframe}, registry={registry_timeframe}"
        )

    mapping = registry.get("mapping") or {}
    registry_num_tickers = int(registry.get("num_tickers") or -1)
    if registry_num_tickers != num_tickers or len(mapping) != num_tickers:
        raise ValueError(
            "checkpoint ticker registry mismatch: "
            f"config.num_tickers={num_tickers}, registry.num_tickers={registry_num_tickers}, mapping={len(mapping)}"
        )
    return registry


def decode_return_forecasts(
    line_returns: torch.Tensor,
    lower_returns: torch.Tensor,
    upper_returns: torch.Tensor,
    anchor_closes: torch.Tensor,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    anchor = anchor_closes.unsqueeze(-1)
    line_prices = (anchor * (1.0 + line_returns)).tolist()
    lower_prices = (anchor * (1.0 + lower_returns)).tolist()
    upper_prices = (anchor * (1.0 + upper_returns)).tolist()
    return line_prices, lower_prices, upper_prices


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _matches_calibration_target(payload: dict[str, Any], *, run_id: str, timeframe: str) -> bool:
    payload_timeframe = str(payload.get("timeframe") or "").upper()
    if payload_timeframe and payload_timeframe != str(timeframe).upper():
        return False
    ids = {
        str(payload.get("model_run_id") or ""),
        str(payload.get("run_id") or ""),
        str(payload.get("model_id") or ""),
    }
    return run_id in ids


def _calibration_from_payload(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    if payload.get("schema_version") == "calibration_artifact_manifest_v1":
        params = payload.get("calibration_params")
        method = payload.get("calibration_method")
    elif "calibration_method" in payload and "calibration_params" in payload:
        params = payload.get("calibration_params")
        method = payload.get("calibration_method")
    elif "method" in payload and "params" in payload:
        params = payload.get("params")
        method = payload.get("method")
    else:
        return None
    if not isinstance(params, dict) or not method:
        return None
    return {
        "status": "calibration_applied",
        "applied": True,
        "method": str(method),
        "params": params,
        "artifact_path": str(source_path),
        "source": "product_band_calibration_artifact",
    }


def _legacy_resolve_product_band_calibration_unused(
    *,
    checkpoint_config: dict[str, Any],
    run_id: str,
    timeframe: str,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "calibration_disabled", "applied": False}

    explicit_candidates = [
        checkpoint_config.get("calibration_artifact_path"),
        checkpoint_config.get("band_calibration_artifact"),
        checkpoint_config.get("calibration_path"),
    ]
    for raw_path in explicit_candidates:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        payload = _safe_json(path)
        if payload:
            artifact = _calibration_from_payload(payload, source_path=path)
            if artifact:
                return artifact

    try:
        manifest = resolve_default_calibration_artifact_manifest(
            role="band",
            timeframe=timeframe,
            model_run_id=run_id,
        )
        path = default_calibration_manifest_path(role="band", timeframe=timeframe, model_run_id=run_id)
        artifact = _calibration_from_payload(manifest, source_path=path)
        if artifact:
            return artifact
    except FileNotFoundError:
        pass

    docs_dir = PROJECT_ROOT / "docs"
    for pattern in ("*config_lock.json", "*calibration_params.json"):
        for path in sorted(docs_dir.glob(pattern)):
            payload = _safe_json(path)
            if not payload or not _matches_calibration_target(payload, run_id=run_id, timeframe=timeframe):
                continue
            artifact = _calibration_from_payload(payload, source_path=path)
            if artifact:
                return artifact

    return {
        "status": "calibration_missing_raw_band_output_used",
        "applied": False,
        "missing_reason": f"band calibration artifact not found for run_id={run_id}, timeframe={timeframe}",
    }


def _legacy_apply_product_band_calibration_unused(
    *,
    lower_returns: torch.Tensor,
    upper_returns: torch.Tensor,
    calibration: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not calibration.get("applied"):
        return lower_returns, upper_returns

    method = str(calibration.get("method") or "")
    params = calibration.get("params") if isinstance(calibration.get("params"), dict) else {}
    center = (lower_returns + upper_returns) / 2.0
    lower_width = torch.clamp(center - lower_returns, min=1e-6)
    upper_width = torch.clamp(upper_returns - center, min=1e-6)

    if method in {"scalar_width", "lower_focused", "separate_scale", "symmetric_expand", "upper_trimmed", "lower_breach_guard"}:
        scale = float(params.get("scale", 1.0))
        lower_scale = float(params.get("lower_scale", scale))
        upper_scale = float(params.get("upper_scale", scale))
        calibrated_lower = center - lower_width * lower_scale
        calibrated_upper = center + upper_width * upper_scale
    elif method == "conformal_residual":
        calibrated_lower = center + float(params["lower_offset"])
        calibrated_upper = center + float(params["upper_offset"])
    elif "global_shift" in params:
        calibrated_lower = lower_returns + float(params["global_shift"])
        calibrated_upper = upper_returns
    else:
        raise ValueError(f"지원하지 않는 band calibration method입니다: {method}")

    return torch.minimum(calibrated_lower, calibrated_upper), torch.maximum(calibrated_lower, calibrated_upper)


def resolve_bundle(
    *,
    split_name: str,
    timeframe: str,
    seq_len: int,
    horizon: int,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    include_future_covariate: bool = True,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
    ticker_registry: dict[str, Any] | None = None,
    ticker_registry_path: str | None = None,
    market_data_provider: str | None = None,
) -> SequenceDatasetBundle | SequenceDataset:
    train_bundle, val_bundle, test_bundle, _, _, _ = prepare_dataset_splits(
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        tickers=tickers,
        limit_tickers=limit_tickers,
        include_future_covariate=include_future_covariate,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        ticker_registry=ticker_registry,
        ticker_registry_path=ticker_registry_path,
        market_data_provider=market_data_provider,
    )
    bundle_map = {
        "train": train_bundle,
        "val": val_bundle,
        "test": test_bundle,
    }
    return bundle_map[split_name]


def infer_bundle(
    bundle: SequenceDatasetBundle | SequenceDataset,
    *,
    checkpoint_path: str | Path,
    model_name: str,
    timeframe: str,
    horizon: int,
    run_id: str,
    model_ver: str,
    q_low: float,
    q_high: float,
    enable_band_calibration: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    device = resolve_device(str(checkpoint.get("config", {}).get("device", "auto")))
    model = model.to(device)
    model = maybe_compile_model(model, device)
    model_config = checkpoint.get("config", {})
    use_future_covariate = bool(model_config.get("use_future_covariate", model_name == "tide"))
    line_target_type = str(model_config.get("line_target_type", "raw_future_return"))
    band_target_type = str(model_config.get("band_target_type", "raw_future_return"))
    raw_return_mode = line_target_type == "raw_future_return" and band_target_type == "raw_future_return"
    batch_size = int(model_config.get("batch_size", 64))
    num_workers = model_config.get("num_workers", "auto")
    feature_subset = select_bundle_features_for_checkpoint(bundle, model_config)
    bundle = feature_subset.bundle
    loader = make_loader(
        bundle,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )

    model_role = resolve_checkpoint_model_role(model_config)
    require_role_supported(
        model_role,
        INFERENCE_PAYLOAD_SUPPORTED_ROLES,
        context="inference.payload",
    )
    product_band_calibration = (
        resolve_product_band_calibration(
            checkpoint_config=model_config,
            run_id=run_id,
            timeframe=timeframe,
            enabled=enable_band_calibration,
        )
        if model_role == "band"
        else {"status": "not_band_model", "applied": False}
    )
    pinball = PinballLoss((q_low, 0.5, q_high), sort_quantiles=True)

    prediction_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    metadata = bundle.metadata.reset_index(drop=True)
    offset = 0
    summary_line_predictions: list[torch.Tensor] = []
    summary_lower_predictions: list[torch.Tensor] = []
    summary_upper_predictions: list[torch.Tensor] = []
    summary_risk_logits: list[torch.Tensor] = []
    summary_line_targets: list[torch.Tensor] = []
    summary_band_targets: list[torch.Tensor] = []
    summary_raw_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_ids, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            line_target = line_target.to(device, non_blocking=True)
            band_target = band_target.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)

            with autocast_context(device):
                if isinstance(getattr(model, "_orig_mod", model), TiDE) and use_future_covariate:
                    output = model(features, ticker_id=ticker_ids, future_covariate=future_covariates)
                else:
                    output = model(features, ticker_id=ticker_ids)

            assert_output_matches_model_role(output, model_role)
            line_target_cpu = line_target.detach().cpu()
            band_target_cpu = band_target.detach().cpu()
            raw_targets_cpu = raw_future_returns.detach().cpu()
            summary_line_targets.append(line_target_cpu)
            summary_band_targets.append(band_target_cpu)
            summary_raw_targets.append(raw_targets_cpu)

            if model_role == "line_v2":
                assert isinstance(output, LineV2Output)
                line_returns = output.line.detach().cpu()
                risk_logits = output.downside_risk_logit.detach().cpu()
                lower_returns = None
                upper_returns = None
                summary_line_predictions.append(line_returns)
                summary_risk_logits.append(risk_logits)
                batch_size_now = line_returns.shape[0]
            elif model_role == "band":
                assert isinstance(output, BandOutput)
                ordered_band = torch.sort(
                    torch.stack((output.lower_band.detach().cpu(), output.upper_band.detach().cpu()), dim=-1),
                    dim=-1,
                ).values
                line_returns = None
                lower_returns, upper_returns = apply_product_band_calibration(
                    lower_returns=ordered_band[..., 0],
                    upper_returns=ordered_band[..., 1],
                    calibration=product_band_calibration,
                )
                summary_lower_predictions.append(lower_returns)
                summary_upper_predictions.append(upper_returns)
                batch_size_now = lower_returns.shape[0]
            else:
                assert isinstance(output, ForecastOutput)
                line_returns, lower_returns, upper_returns = apply_band_postprocess(
                    output.line.detach().cpu(),
                    output.lower_band.detach().cpu(),
                    output.upper_band.detach().cpu(),
                )
                summary_line_predictions.append(line_returns)
                summary_lower_predictions.append(lower_returns)
                summary_upper_predictions.append(upper_returns)
                batch_size_now = line_returns.shape[0]

            if raw_return_mode:
                anchor_batch = torch.tensor(
                    [
                        float(
                            bundle.ticker_arrays[bundle.sample_refs[offset + batch_index][0]]["closes"][
                                bundle.sample_refs[offset + batch_index][1]
                            ]
                        )
                        if isinstance(bundle, SequenceDataset)
                        else float(bundle.anchor_closes[offset + batch_index].item())
                        for batch_index in range(batch_size_now)
                    ],
                    dtype=torch.float32,
                )
                if model_role == "line_v2" and line_returns is not None:
                    line_prices = (anchor_batch.unsqueeze(-1) * (1.0 + line_returns)).tolist()
                    lower_prices, upper_prices = [], []
                elif model_role == "band" and lower_returns is not None and upper_returns is not None:
                    lower_prices = (anchor_batch.unsqueeze(-1) * (1.0 + lower_returns)).tolist()
                    upper_prices = (anchor_batch.unsqueeze(-1) * (1.0 + upper_returns)).tolist()
                    line_prices = []
                else:
                    assert line_returns is not None and lower_returns is not None and upper_returns is not None
                    line_prices, lower_prices, upper_prices = decode_return_forecasts(
                        line_returns,
                        lower_returns,
                        upper_returns,
                        anchor_batch,
                    )
            else:
                anchor_batch = torch.zeros(batch_size_now, dtype=torch.float32)
                line_prices, lower_prices, upper_prices = [], [], []

            for batch_index in range(batch_size_now):
                row = metadata.iloc[offset + batch_index]
                if raw_return_mode and model_role == "line_v2" and line_returns is not None:
                    line_series = line_prices[batch_index]
                    current_price = float(anchor_batch[batch_index].item())
                    final_line = line_series[-1]
                    signal = "BUY" if final_line >= current_price else "SELL"
                    prediction_records.append(
                        {
                            "ticker": row["ticker"],
                            "model_name": model_name,
                            "timeframe": timeframe,
                            "horizon": horizon,
                            "asof_date": row["asof_date"],
                            "decision_time": utc_now_iso(),
                            "run_id": run_id,
                            "model_ver": model_ver,
                            "signal": signal,
                            "forecast_dates": row["forecast_dates"],
                            "line_series": line_series,
                            "conservative_series": [],
                            "lower_band_series": [],
                            "upper_band_series": [],
                            "band_quantile_low": None,
                            "band_quantile_high": None,
                            "meta": {"layer": "line", "model_role": "line_v2"},
                        }
                    )
                elif raw_return_mode and model_role == "band" and lower_returns is not None and upper_returns is not None:
                    lower_series = lower_prices[batch_index]
                    upper_series = upper_prices[batch_index]
                    prediction_records.append(
                        {
                            "ticker": row["ticker"],
                            "model_name": model_name,
                            "timeframe": timeframe,
                            "horizon": horizon,
                            "asof_date": row["asof_date"],
                            "decision_time": utc_now_iso(),
                            "run_id": run_id,
                            "model_ver": model_ver,
                            "signal": "HOLD",
                            "forecast_dates": row["forecast_dates"],
                            "line_series": [],
                            "conservative_series": [],
                            "lower_band_series": lower_series,
                            "upper_band_series": upper_series,
                            "band_quantile_low": q_low,
                            "band_quantile_high": q_high,
                            "meta": {
                                "layer": "band",
                                "model_role": "band",
                                "band_calibration_status": product_band_calibration.get("status"),
                                "band_calibration_applied": bool(product_band_calibration.get("applied")),
                                "band_calibration_method": product_band_calibration.get("method"),
                                "band_calibration_params": product_band_calibration.get("params") or {},
                                "band_calibration_artifact": product_band_calibration.get("artifact_path"),
                            },
                        }
                    )
                elif raw_return_mode:
                    assert line_returns is not None and lower_returns is not None and upper_returns is not None
                    line_series = line_prices[batch_index]
                    lower_series = lower_prices[batch_index]
                    upper_series = upper_prices[batch_index]
                    final_lower = lower_series[-1]
                    final_upper = upper_series[-1]
                    current_price = float(anchor_batch[batch_index].item())

                    if current_price < final_lower:
                        signal = "BUY"
                    elif current_price > final_upper:
                        signal = "SELL"
                    else:
                        signal = "HOLD"

                    prediction_records.append(
                        {
                            "ticker": row["ticker"],
                            "model_name": model_name,
                            "timeframe": timeframe,
                            "horizon": horizon,
                            "asof_date": row["asof_date"],
                            "decision_time": utc_now_iso(),
                            "run_id": run_id,
                            "model_ver": model_ver,
                            "signal": signal,
                            "forecast_dates": row["forecast_dates"],
                            "line_series": line_series,
                            "conservative_series": line_series,
                            "lower_band_series": lower_series,
                            "upper_band_series": upper_series,
                            "band_quantile_low": q_low,
                            "band_quantile_high": q_high,
                        }
                    )

                if raw_return_mode:
                    actual_series = (anchor_batch[batch_index] * (1.0 + band_target_cpu[batch_index])).tolist()
                else:
                    actual_series = band_target_cpu[batch_index].tolist()
                base_evaluation = {
                    "run_id": run_id,
                    "ticker": row["ticker"],
                    "timeframe": timeframe,
                    "asof_date": row["asof_date"],
                    "actual_series": actual_series,
                }
                if model_role == "line_v2" and line_returns is not None:
                    line_error = torch.abs(line_returns[batch_index] - line_target_cpu[batch_index])
                    evaluation_records.append(
                        {
                            **base_evaluation,
                            "mae": float(line_error.mean().item()),
                            "direction_accuracy": float(
                                (
                                    (line_returns[batch_index, -1] >= 0.0)
                                    == (line_target_cpu[batch_index, -1] >= 0.0)
                                )
                                .to(torch.float32)
                                .item()
                            ),
                            "meta": {"layer": "line", "model_role": "line_v2"},
                        }
                    )
                elif model_role == "band" and lower_returns is not None and upper_returns is not None:
                    actual_return_tensor = band_target_cpu[batch_index].unsqueeze(0)
                    mid_returns = (lower_returns[batch_index] + upper_returns[batch_index]) / 2.0
                    quantile_return_tensor = torch.stack(
                        (lower_returns[batch_index], mid_returns, upper_returns[batch_index]),
                        dim=-1,
                    ).unsqueeze(0)
                    evaluation_records.append(
                        {
                            **base_evaluation,
                            "pinball_loss": float(pinball(quantile_return_tensor, actual_return_tensor).item()),
                            "coverage": float(
                                (
                                    (band_target_cpu[batch_index] >= lower_returns[batch_index])
                                    & (band_target_cpu[batch_index] <= upper_returns[batch_index])
                                )
                                .to(torch.float32)
                                .mean()
                                .item()
                            ),
                            "lower_breach_rate": float((band_target_cpu[batch_index] < lower_returns[batch_index]).to(torch.float32).mean().item()),
                            "upper_breach_rate": float((band_target_cpu[batch_index] > upper_returns[batch_index]).to(torch.float32).mean().item()),
                            "normalized_band_width": float((upper_returns[batch_index] - lower_returns[batch_index]).mean().item()),
                            "meta": {"layer": "band", "model_role": "band"},
                        }
                    )
                else:
                    assert line_returns is not None and lower_returns is not None and upper_returns is not None
                    actual_return_tensor = band_target_cpu[batch_index].unsqueeze(0)
                    quantile_return_tensor = torch.stack(
                        (lower_returns[batch_index], line_returns[batch_index], upper_returns[batch_index]),
                        dim=-1,
                    ).unsqueeze(0)
                    evaluation_records.append(
                        {
                            **base_evaluation,
                            "pinball_loss": float(pinball(quantile_return_tensor, actual_return_tensor).item()),
                            **build_single_sample_evaluation(
                                actual_series=band_target_cpu[batch_index].tolist(),
                                line_series=line_returns[batch_index].tolist(),
                                lower_series=lower_returns[batch_index].tolist(),
                                upper_series=upper_returns[batch_index].tolist(),
                                line_target_type=line_target_type,
                            ),
                            "normalized_band_width": float((upper_returns[batch_index] - lower_returns[batch_index]).mean().item()),
                        }
                    )
            offset += batch_size_now

    metadata_slice = metadata.iloc[:offset].copy()
    if model_role == "line_v2":
        summary_metrics = summarize_line_v2_metrics(
            metadata=metadata_slice,
            line_predictions=torch.cat(summary_line_predictions, dim=0),
            risk_logits=torch.cat(summary_risk_logits, dim=0),
            line_targets=torch.cat(summary_line_targets, dim=0),
            raw_future_returns=torch.cat(summary_raw_targets, dim=0),
            line_target_type=line_target_type,
            severe_downside_threshold=model_config.get("severe_downside_threshold"),
            risk_decision_threshold=float(model_config.get("risk_decision_threshold", 0.5)),
        )
    elif model_role == "band":
        summary_metrics = summarize_band_metrics(
            lower_predictions=torch.cat(summary_lower_predictions, dim=0),
            upper_predictions=torch.cat(summary_upper_predictions, dim=0),
            band_targets=torch.cat(summary_band_targets, dim=0),
            raw_future_returns=torch.cat(summary_raw_targets, dim=0),
            q_low=q_low,
            q_high=q_high,
            squeeze_breakout_threshold=model_config.get("squeeze_breakout_threshold"),
        )
    else:
        summary_metrics = summarize_forecast_metrics(
            metadata=metadata_slice,
            line_predictions=torch.cat(summary_line_predictions, dim=0),
            lower_predictions=torch.cat(summary_lower_predictions, dim=0),
            upper_predictions=torch.cat(summary_upper_predictions, dim=0),
            line_targets=torch.cat(summary_line_targets, dim=0),
            band_targets=torch.cat(summary_band_targets, dim=0),
            raw_future_returns=torch.cat(summary_raw_targets, dim=0),
            line_target_type=line_target_type,
            band_target_type=band_target_type,
            q_low=q_low,
            q_high=q_high,
            severe_downside_threshold=model_config.get("severe_downside_threshold"),
            squeeze_breakout_threshold=model_config.get("squeeze_breakout_threshold"),
        )
    summary_metrics.update(feature_subset.report)
    if model_role == "band":
        summary_metrics["band_calibration"] = product_band_calibration
    return prediction_records, evaluation_records, summary_metrics


def run_inference(
    *,
    run_id: str,
    split_name: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    save: bool = False,
    save_product_latest_only: bool = False,
    allow_bulk_evaluation_save: bool = False,
    disable_band_calibration: bool = False,
) -> dict[str, Any]:
    model_run = get_model_run(run_id)
    if model_run is None:
        raise ValueError(f"run_id={run_id}에 해당하는 model_runs 기록이 없습니다.")
    # CP12: NaN으로 실패한 run에 대해서는 inference·결과 저장을 거부한다.
    run_status = str(model_run.get("status") or "completed")
    if run_status != "completed":
        raise ValueError(
            f"run_id={run_id} status={run_status}: completed 상태의 run에서만 inference를 실행할 수 있습니다."
        )

    normalize_ai_timeframe(str(model_run["timeframe"]))

    config = model_run.get("config") or {}
    checkpoint_path = model_run.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path가 model_runs에 저장되어 있지 않습니다.")

    checkpoint_config = load_checkpoint_config(checkpoint_path)
    runtime_contract = validate_checkpoint_runtime_contract(
        checkpoint_config,
        runtime_config=config,
        runtime_timeframe=str(model_run["timeframe"]),
        runtime_horizon=int(model_run["horizon"]),
        run_status=run_status,
    )
    provider, provider_contract = resolve_execution_market_data_provider(
        checkpoint_config,
        run_config=model_run,
    )
    line_target_type = str(checkpoint_config.get("line_target_type", config.get("line_target_type", "raw_future_return")))
    band_target_type = str(checkpoint_config.get("band_target_type", config.get("band_target_type", "raw_future_return")))
    raw_return_mode = line_target_type == "raw_future_return" and band_target_type == "raw_future_return"
    checkpoint_ticker_registry = resolve_checkpoint_ticker_registry(
        checkpoint_config,
        str(model_run["timeframe"]),
    )
    checkpoint_registry_path = checkpoint_config.get("ticker_registry_path")

    bundle = resolve_bundle(
        split_name=split_name,
        timeframe=model_run["timeframe"],
        seq_len=int(config.get("seq_len") or checkpoint_config["seq_len"]),
        horizon=int(model_run["horizon"]),
        tickers=tickers or config.get("tickers") or checkpoint_config.get("tickers"),
        limit_tickers=limit_tickers or config.get("limit_tickers") or checkpoint_config.get("limit_tickers"),
        include_future_covariate=bool(
            checkpoint_config.get("use_future_covariate", config.get("use_future_covariate", model_run["model_name"] == "tide"))
        ),
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        ticker_registry=checkpoint_ticker_registry,
        ticker_registry_path=str(checkpoint_registry_path) if checkpoint_registry_path else None,
        market_data_provider=provider,
    )
    prediction_records, evaluation_records, summary_metrics = infer_bundle(
        bundle,
        checkpoint_path=checkpoint_path,
        model_name=model_run["model_name"],
        timeframe=model_run["timeframe"],
        horizon=int(model_run["horizon"]),
        run_id=run_id,
        model_ver=config.get("model_ver", "v2-multihead"),
        q_low=float(model_run.get("band_quantile_low") or config.get("q_low", 0.1)),
        q_high=float(model_run.get("band_quantile_high") or config.get("q_high", 0.9)),
        enable_band_calibration=not disable_band_calibration,
    )
    summary_metrics.update(provider_contract)
    contract_metrics = {
        **provider_contract,
        **runtime_contract,
        "checkpoint_feature_count": summary_metrics.get("checkpoint_feature_count"),
        "bundle_feature_count_before": summary_metrics.get("bundle_feature_count_before"),
        "bundle_feature_count_after": summary_metrics.get("bundle_feature_count_after"),
        "missing_features": summary_metrics.get("missing_features"),
        "extra_features_ignored": summary_metrics.get("extra_features_ignored"),
    }

    effective_save = save or save_product_latest_only
    if effective_save and not raw_return_mode:
        raise ValueError("비 raw target 체크포인트는 predictions 저장과 시그널 생성을 지원하지 않습니다. score 모드로만 실행해 주세요.")

    storage_contract = resolve_inference_save_contract(
        save=effective_save,
        save_product_latest_only=save_product_latest_only,
        allow_bulk_evaluation_save=allow_bulk_evaluation_save,
        model_run=model_run,
        config=config,
    )
    storage_audit = None
    if storage_contract == "product_latest_only":
        storage_audit = save_product_latest_predictions(prediction_records, evaluation_records)
    elif storage_contract == STORAGE_CONTRACT_EVALUATION_BULK:
        save_predictions(with_prediction_storage_contract(prediction_records, STORAGE_CONTRACT_EVALUATION_BULK))
        save_prediction_evaluations(evaluation_records)

    return {
        "run_id": run_id,
        "split": split_name,
        "prediction_count": len(prediction_records),
        "evaluation_count": len(evaluation_records),
        "target_type": line_target_type,
        "band_target_type": band_target_type,
        "model_role": resolve_checkpoint_model_role(checkpoint_config),
        "mode": "raw_return" if raw_return_mode else "score_only",
        "storage_contract": storage_contract,
        "storage_audit": storage_audit,
        "role_contract_table": role_contract_table(),
        "contract_metrics": contract_metrics,
        "summary_metrics": summary_metrics,
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_inference(
        run_id=args.run_id,
        split_name=args.split,
        tickers=args.tickers,
        limit_tickers=args.limit_tickers,
        save=args.save,
        save_product_latest_only=args.save_product_latest_only,
        allow_bulk_evaluation_save=args.allow_bulk_evaluation_save,
        disable_band_calibration=args.disable_band_calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
