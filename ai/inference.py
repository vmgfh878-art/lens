from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import torch

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.loss import PinballLoss
from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.models.tide import TiDE
from ai.evaluation import build_single_sample_evaluation, summarize_forecast_metrics
from ai.postprocess import apply_band_postprocess
from ai.preprocessing import (
    FUTURE_COVARIATE_DIM,
    MODEL_N_FEATURES,
    SequenceDataset,
    SequenceDatasetBundle,
    normalize_ai_timeframe,
    prepare_dataset_splits,
)
from ai.storage import get_model_run, save_prediction_evaluations, save_predictions, utc_now_iso
from ai.ticker_registry import load_registry
from ai.train import make_loader, resolve_device

MODEL_REGISTRY = {
    "patchtst": PatchTST,
    "cnn_lstm": CNNLSTM,
    "tide": TiDE,
}


def should_use_cuda_optimizations(device: torch.device) -> bool:
    return device.type == "cuda"


def maybe_compile_model(model, device: torch.device):
    if not should_use_cuda_optimizations(device) or not hasattr(torch, "compile"):
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
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model_cls = MODEL_REGISTRY[config["model"]]
    model_kwargs = {
        "n_features": MODEL_N_FEATURES,
        "seq_len": config["seq_len"],
        "horizon": config["horizon"],
        "dropout": config["dropout"],
        "band_mode": config.get("band_mode", "direct"),
        "num_tickers": config.get("num_tickers", 0),
        "ticker_emb_dim": config.get("ticker_emb_dim", 32),
    }
    if config["model"] == "patchtst":
        model_kwargs["ci_aggregate"] = config.get("ci_aggregate", "target")
        model_kwargs["target_channel_idx"] = config.get("target_channel_idx", 0)
        model_kwargs["ci_target_fast"] = bool(config.get("ci_target_fast", False))
    if config["model"] == "cnn_lstm":
        model_kwargs["use_direction_head"] = bool(config.get("use_direction_head", False))
    if config["model"] == "tide":
        use_future_covariate = bool(config.get("use_future_covariate", True))
        model_kwargs["future_cov_dim"] = config.get("future_cov_dim", FUTURE_COVARIATE_DIM) if use_future_covariate else 0
    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float | None]]:
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
    loader = make_loader(
        bundle,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )

    pinball = PinballLoss((q_low, 0.5, q_high), sort_quantiles=True)

    prediction_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    metadata = bundle.metadata.reset_index(drop=True)
    offset = 0
    summary_line_predictions: list[torch.Tensor] = []
    summary_lower_predictions: list[torch.Tensor] = []
    summary_upper_predictions: list[torch.Tensor] = []
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

            line_returns, lower_returns, upper_returns = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            summary_line_predictions.append(line_returns)
            summary_lower_predictions.append(lower_returns)
            summary_upper_predictions.append(upper_returns)
            summary_line_targets.append(line_target.detach().cpu())
            summary_band_targets.append(band_target.detach().cpu())
            summary_raw_targets.append(raw_future_returns.detach().cpu())
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
                if raw_return_mode:
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

                actual_return_tensor = band_target[batch_index].unsqueeze(0)
                if raw_return_mode:
                    actual_series = (anchor_batch[batch_index] * (1.0 + band_target[batch_index])).tolist()
                else:
                    actual_series = band_target[batch_index].tolist()
                quantile_return_tensor = torch.stack(
                    (lower_returns[batch_index], line_returns[batch_index], upper_returns[batch_index]),
                    dim=-1,
                ).unsqueeze(0)
                evaluation_records.append(
                    {
                        "run_id": run_id,
                        "ticker": row["ticker"],
                        "timeframe": timeframe,
                        "asof_date": row["asof_date"],
                        "actual_series": actual_series,
                        "pinball_loss": float(pinball(quantile_return_tensor, actual_return_tensor).item()),
                        **build_single_sample_evaluation(
                            actual_series=band_target[batch_index].tolist(),
                            line_series=line_returns[batch_index].tolist(),
                            lower_series=lower_returns[batch_index].tolist(),
                            upper_series=upper_returns[batch_index].tolist(),
                            line_target_type=line_target_type,
                        ),
                        "normalized_band_width": float((upper_returns[batch_index] - lower_returns[batch_index]).mean().item()),
                    }
                )
            offset += batch_size_now

    summary_metrics = summarize_forecast_metrics(
        metadata=metadata.iloc[:offset].copy(),
        line_predictions=torch.cat(summary_line_predictions, dim=0),
        lower_predictions=torch.cat(summary_lower_predictions, dim=0),
        upper_predictions=torch.cat(summary_upper_predictions, dim=0),
        line_targets=torch.cat(summary_line_targets, dim=0),
        band_targets=torch.cat(summary_band_targets, dim=0),
        raw_future_returns=torch.cat(summary_raw_targets, dim=0),
        line_target_type=line_target_type,
        band_target_type=band_target_type,
    )
    return prediction_records, evaluation_records, summary_metrics


def run_inference(
    *,
    run_id: str,
    split_name: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    save: bool = False,
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
    line_target_type = str(config.get("line_target_type", "raw_future_return"))
    band_target_type = str(config.get("band_target_type", "raw_future_return"))
    raw_return_mode = line_target_type == "raw_future_return" and band_target_type == "raw_future_return"
    checkpoint_path = model_run.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path가 model_runs에 저장되어 있지 않습니다.")

    checkpoint_config = load_checkpoint_config(checkpoint_path)
    checkpoint_ticker_registry = resolve_checkpoint_ticker_registry(
        checkpoint_config,
        str(model_run["timeframe"]),
    )
    checkpoint_registry_path = checkpoint_config.get("ticker_registry_path")

    bundle = resolve_bundle(
        split_name=split_name,
        timeframe=model_run["timeframe"],
        seq_len=int(config["seq_len"]),
        horizon=int(model_run["horizon"]),
        tickers=tickers or config.get("tickers"),
        limit_tickers=limit_tickers or config.get("limit_tickers"),
        include_future_covariate=bool(config.get("use_future_covariate", model_run["model_name"] == "tide")),
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        ticker_registry=checkpoint_ticker_registry,
        ticker_registry_path=str(checkpoint_registry_path) if checkpoint_registry_path else None,
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
    )

    if save and not raw_return_mode:
        raise ValueError("비 raw target 체크포인트는 predictions 저장과 시그널 생성을 지원하지 않습니다. score 모드로만 실행해 주세요.")

    if save:
        save_predictions(prediction_records)
        save_prediction_evaluations(evaluation_records)

    return {
        "run_id": run_id,
        "split": split_name,
        "prediction_count": len(prediction_records),
        "evaluation_count": len(evaluation_records),
        "target_type": line_target_type,
        "band_target_type": band_target_type,
        "mode": "raw_return" if raw_return_mode else "score_only",
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
