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
from ai.preprocessing import SequenceDatasetBundle, apply_feature_stats, normalize_ai_timeframe, prepare_dataset_splits
from ai.storage import get_model_run, save_prediction_evaluations, save_predictions, utc_now_iso
from backend.app.services.feature_svc import FEATURE_COLUMNS

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
    parser = argparse.ArgumentParser(description="학습된 Lens 모델로 예측 결과를 생성한다")
    parser.add_argument("--run-id", required=True, help="model_runs에 저장된 run_id")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="predictions와 prediction_evaluations에 저장한다")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model_cls = MODEL_REGISTRY[config["model"]]
    model = model_cls(
        n_features=len(FEATURE_COLUMNS),
        seq_len=config["seq_len"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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


def sort_prediction_triplet(
    line_returns: torch.Tensor,
    lower_returns: torch.Tensor,
    upper_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantiles = torch.stack((lower_returns, line_returns, upper_returns), dim=-1)
    ordered = torch.sort(quantiles, dim=-1).values
    return ordered[..., 1], ordered[..., 0], ordered[..., 2]


def resolve_bundle(
    *,
    split_name: str,
    timeframe: str,
    seq_len: int,
    horizon: int,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> SequenceDatasetBundle:
    train_bundle, val_bundle, test_bundle, _, _, _ = prepare_dataset_splits(
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    bundle_map = {
        "train": train_bundle,
        "val": val_bundle,
        "test": test_bundle,
    }
    return bundle_map[split_name]


def infer_bundle(
    bundle: SequenceDatasetBundle,
    *,
    checkpoint_path: str | Path,
    model_name: str,
    timeframe: str,
    horizon: int,
    run_id: str,
    model_ver: str,
    q_low: float,
    q_high: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = maybe_compile_model(model, device)
    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()
    normalized = apply_feature_stats(bundle.features, mean, std).to(device, non_blocking=True)

    with torch.no_grad():
        with autocast_context(device):
            output = model(normalized)
    line_returns, lower_returns, upper_returns = sort_prediction_triplet(
        output.line.cpu(),
        output.lower_band.cpu(),
        output.upper_band.cpu(),
    )

    line_prices, lower_prices, upper_prices = decode_return_forecasts(
        line_returns,
        lower_returns,
        upper_returns,
        bundle.anchor_closes.cpu(),
    )
    pinball = PinballLoss((q_low, 0.5, q_high), sort_quantiles=True)

    prediction_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    for idx, row in bundle.metadata.reset_index(drop=True).iterrows():
        line_series = line_prices[idx]
        lower_series = lower_prices[idx]
        upper_series = upper_prices[idx]
        final_lower = lower_series[-1]
        final_upper = upper_series[-1]
        current_price = float(bundle.anchor_closes[idx].item())

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

        actual_returns = bundle.band_targets[idx]
        actual_prices = (bundle.anchor_closes[idx] * (1.0 + actual_returns)).tolist()
        actual_tensor = torch.tensor(actual_prices)
        lower_tensor = torch.tensor(lower_series)
        upper_tensor = torch.tensor(upper_series)
        line_tensor = torch.tensor(line_series)
        quantile_return_tensor = torch.stack(
            (lower_returns[idx], line_returns[idx], upper_returns[idx]),
            dim=-1,
        ).unsqueeze(0)
        actual_return_tensor = bundle.band_targets[idx].unsqueeze(0)
        evaluation_records.append(
            {
                "run_id": run_id,
                "ticker": row["ticker"],
                "timeframe": timeframe,
                "asof_date": row["asof_date"],
                "actual_series": actual_prices,
                "pinball_loss": float(pinball(quantile_return_tensor, actual_return_tensor).item()),
                "coverage": float(((actual_tensor >= lower_tensor) & (actual_tensor <= upper_tensor)).float().mean().item()),
                "avg_band_width": float((upper_tensor - lower_tensor).mean().item()),
                "normalized_band_width": float((upper_tensor - lower_tensor).mean().item() / max(current_price, 1e-6)),
                "direction_accuracy": float(((line_tensor >= current_price) == (actual_tensor >= current_price)).float().mean().item()),
                "mape": float((torch.abs(line_tensor - actual_tensor) / actual_tensor.abs().clamp_min(1e-6)).mean().item()),
            }
        )

    return prediction_records, evaluation_records


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

    normalize_ai_timeframe(str(model_run["timeframe"]))

    config = model_run.get("config") or {}
    checkpoint_path = model_run.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path가 model_runs에 저장되어 있지 않습니다.")

    bundle = resolve_bundle(
        split_name=split_name,
        timeframe=model_run["timeframe"],
        seq_len=int(config["seq_len"]),
        horizon=int(model_run["horizon"]),
        tickers=tickers or config.get("tickers"),
        limit_tickers=limit_tickers or config.get("limit_tickers"),
    )
    prediction_records, evaluation_records = infer_bundle(
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

    if save:
        save_predictions(prediction_records)
        save_prediction_evaluations(evaluation_records)

    return {
        "run_id": run_id,
        "split": split_name,
        "prediction_count": len(prediction_records),
        "evaluation_count": len(evaluation_records),
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
