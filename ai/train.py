from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.loss import ForecastCompositeLoss
from ai.models.cnn_lstm import CNNLSTM
from ai.models.patchtst import PatchTST
from ai.models.tide import TiDE
from ai.preprocessing import DatasetPlan, SequenceDatasetBundle, build_dataset_plan, default_horizon, fetch_feature_index_frame, prepare_dataset_splits
from ai.postprocess import apply_band_postprocess
from ai.storage import save_model_run
from backend.app.services.feature_svc import FEATURE_COLUMNS

try:
    import wandb
except ImportError:  # pragma: no cover - 로컬 환경에 wandb가 없을 수 있다.
    wandb = None


MODEL_REGISTRY = {
    "patchtst": PatchTST,
    "cnn_lstm": CNNLSTM,
    "tide": TiDE,
}
DEFAULT_NUM_WORKERS = 6


@dataclass
class TrainConfig:
    model: str
    timeframe: str
    horizon: int
    seq_len: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    q_low: float
    q_high: float
    alpha: float
    beta: float
    delta: float
    lambda_line: float
    lambda_band: float
    lambda_width: float
    lambda_cross: float
    dropout: float
    band_mode: str
    tickers: list[str] | None
    limit_tickers: int | None
    seed: int
    use_wandb: bool
    wandb_project: str
    model_ver: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 멀티헤드 시계열 모델 학습")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="patchtst")
    parser.add_argument("--timeframe", choices=["1D", "1W"], default="1D")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--q-low", type=float, default=0.1)
    parser.add_argument("--q-high", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--lambda-line", type=float, default=1.0)
    parser.add_argument("--lambda-band", type=float, default=1.0)
    parser.add_argument("--lambda-width", type=float, default=0.1)
    parser.add_argument("--lambda-cross", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--band-mode", choices=["direct", "param"], default="direct")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lens-batch4")
    parser.add_argument("--save-run", action="store_true")
    parser.add_argument("--model-ver", default="v2-multihead")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_model(config: TrainConfig):
    model_cls = MODEL_REGISTRY[config.model]
    common_kwargs = {
        "n_features": len(FEATURE_COLUMNS),
        "seq_len": config.seq_len,
        "horizon": config.horizon,
        "dropout": config.dropout,
        "band_mode": config.band_mode,
    }
    return model_cls(**common_kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config_hash(config: TrainConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def should_use_cuda_optimizations(device: torch.device) -> bool:
    return device.type == "cuda"


def maybe_compile_model(model, device: torch.device):
    if not should_use_cuda_optimizations(device) or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode="reduce-overhead")


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def autocast_context(device: torch.device):
    if not should_use_cuda_optimizations(device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def make_loader(bundle: SequenceDatasetBundle, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(bundle.features, bundle.line_targets, bundle.band_targets)
    use_cuda = torch.cuda.is_available()
    num_workers = DEFAULT_NUM_WORKERS if use_cuda else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    *,
    model,
    loader: DataLoader,
    criterion: ForecastCompositeLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "total_loss": 0.0,
        "line_loss": 0.0,
        "band_loss": 0.0,
        "width_loss": 0.0,
        "cross_loss": 0.0,
    }
    batch_count = 0

    for features, line_target, band_target in loader:
        features = features.to(device, non_blocking=True)
        line_target = line_target.to(device, non_blocking=True)
        band_target = band_target.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with autocast_context(device):
            prediction = model(features)
            losses = criterion(prediction, line_target, band_target)

        if is_train:
            losses.total.backward()
            optimizer.step()

        metrics = losses.to_log_dict()
        for key, value in metrics.items():
            totals[key] += value
        batch_count += 1

    return {key: value / max(batch_count, 1) for key, value in totals.items()}


def evaluate_bundle(model, bundle: SequenceDatasetBundle, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        features = bundle.features.to(device, non_blocking=True)
        with autocast_context(device):
            prediction = model(features)
        line_pred, lower, upper = apply_band_postprocess(
            prediction.line.detach().cpu(),
            prediction.lower_band.detach().cpu(),
            prediction.upper_band.detach().cpu(),
        )
        target = bundle.band_targets.cpu()

        coverage = ((target >= lower) & (target <= upper)).float().mean().item()
        avg_band_width = (upper - lower).mean().item()
        direction_accuracy = ((line_pred >= 0) == (bundle.line_targets >= 0)).float().mean().item()
        mape = ((line_pred - bundle.line_targets).abs() / bundle.line_targets.abs().clamp_min(1e-6)).mean().item()
        return {
            "coverage": coverage,
            "avg_band_width": avg_band_width,
            "direction_accuracy": direction_accuracy,
            "mape": mape,
        }


def maybe_init_wandb(config: TrainConfig, run_id: str):
    if not config.use_wandb or wandb is None:
        return None
    return wandb.init(
        project=config.wandb_project,
        config=asdict(config),
        name=f"{config.model}-{config.timeframe}-{run_id[:8]}",
    )


def save_checkpoint(
    model,
    config: TrainConfig,
    run_id: str,
    metrics: dict[str, float],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> Path:
    output_dir = Path("ai") / "artifacts" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.model}_{config.timeframe}_{run_id}.pt"
    state_model = unwrap_model(model)
    torch.save(
        {
            "model_state_dict": state_model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
        },
        path,
    )
    return path


def train(config: TrainConfig, *, save_run: bool) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{config.model}-{config.timeframe}-{uuid4().hex[:12]}"
    print(f"학습 디바이스: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        tickers=config.tickers,
        limit_tickers=config.limit_tickers,
    )

    train_loader = make_loader(train_bundle, batch_size=config.batch_size, shuffle=True)
    val_loader = make_loader(val_bundle, batch_size=config.batch_size, shuffle=False)

    model = build_model(config).to(device)
    model = maybe_compile_model(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = ForecastCompositeLoss(
        q_low=config.q_low,
        q_high=config.q_high,
        alpha=config.alpha,
        beta=config.beta,
        delta=config.delta,
        lambda_line=config.lambda_line,
        lambda_band=config.lambda_band,
        lambda_width=config.lambda_width,
        lambda_cross=config.lambda_cross,
        band_mode=config.band_mode,
    )

    wandb_run = maybe_init_wandb(config, run_id)
    best_val_loss = float("inf")
    best_summary: dict[str, float] = {}
    checkpoint_path = save_checkpoint(model, config, run_id, {}, mean, std)

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        val_quality = evaluate_bundle(model, val_bundle, device)

        summary = {
            "epoch": epoch,
            "epoch_seconds": round(time.perf_counter() - epoch_started_at, 4),
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
            **{f"val/{key}": value for key, value in val_quality.items()},
        }
        print(json.dumps(summary, ensure_ascii=False))

        if wandb_run is not None:
            wandb.log(summary)

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_summary = {
                **train_metrics,
                **val_metrics,
                **val_quality,
            }
            checkpoint_path = save_checkpoint(model, config, run_id, best_summary, mean, std)

    test_quality = evaluate_bundle(model, test_bundle, device)
    result = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "best_metrics": best_summary,
        "test_metrics": test_quality,
        "dataset_plan": summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle),
    }

    if wandb_run is not None:
        wandb.log({f"test/{key}": value for key, value in test_quality.items()})
        wandb_run.finish()

    if save_run:
        metadata = train_bundle.metadata.sort_values("asof_date")
        config_hash = build_config_hash(config)
        save_model_run(
            {
                "run_id": run_id,
                "wandb_run_id": getattr(wandb_run, "id", None),
                "model_name": config.model,
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "feature_version": "indicators_v1",
                "band_quantile_low": config.q_low,
                "band_quantile_high": config.q_high,
                "alpha": config.alpha,
                "beta": config.beta,
                "huber_delta": config.delta,
                "lambda_line": config.lambda_line,
                "lambda_band": config.lambda_band,
                "lambda_width": config.lambda_width,
                "lambda_cross": config.lambda_cross,
                "band_mode": config.band_mode,
                "train_start": metadata["asof_date"].min(),
                "train_end": metadata["asof_date"].max(),
                "val_metrics": best_summary,
                "test_metrics": test_quality,
                "config": {
                    **asdict(config),
                    "config_hash": config_hash,
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                    "best_val_loss": best_summary.get("total_loss"),
                },
                "checkpoint_path": str(checkpoint_path),
            }
        )

    return result


def summarize_dataset_plan(
    plan: DatasetPlan,
    train_bundle: SequenceDatasetBundle,
    val_bundle: SequenceDatasetBundle,
    test_bundle: SequenceDatasetBundle,
) -> dict[str, Any]:
    return {
        "timeframe": plan.timeframe,
        "seq_len": plan.seq_len,
        "horizon": plan.horizon,
        "h_max": plan.h_max,
        "min_fold_samples": plan.min_fold_samples,
        "input_ticker_count": plan.input_ticker_count,
        "eligible_ticker_count": len(plan.eligible_tickers),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "eligible_tickers": plan.eligible_tickers,
        "excluded_reasons": plan.excluded_reasons,
        "train_samples": len(train_bundle),
        "val_samples": len(val_bundle),
        "test_samples": len(test_bundle),
    }


def summarize_plan_only(plan: DatasetPlan) -> dict[str, Any]:
    train_samples = sum(spec.train.count for spec in plan.split_specs.values())
    val_samples = sum(spec.val.count for spec in plan.split_specs.values())
    test_samples = sum(spec.test.count for spec in plan.split_specs.values())
    return {
        "timeframe": plan.timeframe,
        "seq_len": plan.seq_len,
        "horizon": plan.horizon,
        "h_max": plan.h_max,
        "min_fold_samples": plan.min_fold_samples,
        "input_ticker_count": plan.input_ticker_count,
        "eligible_ticker_count": len(plan.eligible_tickers),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "eligible_tickers": plan.eligible_tickers,
        "excluded_reasons": plan.excluded_reasons,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }


def run_dry(config: TrainConfig) -> dict[str, Any]:
    feature_index_frame = fetch_feature_index_frame(
        timeframe=config.timeframe,
        tickers=config.tickers,
        limit_tickers=config.limit_tickers,
    )
    plan = build_dataset_plan(
        feature_index_frame,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
    )
    model = build_model(config)
    criterion = ForecastCompositeLoss(
        q_low=config.q_low,
        q_high=config.q_high,
        alpha=config.alpha,
        beta=config.beta,
        delta=config.delta,
        lambda_line=config.lambda_line,
        lambda_band=config.lambda_band,
        lambda_width=config.lambda_width,
        lambda_cross=config.lambda_cross,
        band_mode=config.band_mode,
    )
    sample_input = torch.randn(4, config.seq_len, len(FEATURE_COLUMNS))
    sample_line_target = torch.randn(4, config.horizon)
    sample_band_target = torch.randn(4, config.horizon)
    with torch.no_grad():
        output = model(sample_input)
        loss = criterion(output, sample_line_target, sample_band_target)
        line_pp, lower_pp, upper_pp = apply_band_postprocess(output.line, output.lower_band, output.upper_band)
    summary = summarize_plan_only(plan)
    summary["forward_smoke"] = {
        "line_shape": list(output.line.shape),
        "lower_shape": list(output.lower_band.shape),
        "upper_shape": list(output.upper_band.shape),
        "postprocess_lower_le_upper": bool(torch.all(lower_pp <= upper_pp).item()),
        "loss_total": float(loss.total.detach().cpu()),
        "line_preserved": bool(torch.equal(line_pp, output.line)),
    }
    return summary


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        model=args.model,
        timeframe=args.timeframe,
        horizon=args.horizon or default_horizon(args.timeframe),
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        q_low=args.q_low,
        q_high=args.q_high,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        lambda_line=args.lambda_line,
        lambda_band=args.lambda_band,
        lambda_width=args.lambda_width,
        lambda_cross=args.lambda_cross,
        dropout=args.dropout,
        band_mode=args.band_mode,
        tickers=args.tickers,
        limit_tickers=args.limit_tickers,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        model_ver=args.model_ver,
    )
    if args.dry_run:
        result = run_dry(config)
    else:
        result = train(config, save_run=args.save_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
