from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
from ai.postprocess import apply_band_postprocess
from ai.evaluation import summarize_forecast_metrics
from ai.preprocessing import (
    DatasetPlan,
    FUTURE_COVARIATE_DIM,
    MODEL_N_FEATURES,
    SequenceDataset,
    SequenceDatasetBundle,
    build_dataset_plan,
    default_horizon,
    fetch_feature_index_frame,
    prepare_dataset_splits,
)
from ai.storage import save_model_run

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

try:
    import wandb
except ImportError:  # pragma: no cover
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
    lr_schedule: str
    warmup_frac: float
    grad_clip: float
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
    lambda_direction: float
    dropout: float
    band_mode: str
    num_tickers: int
    ticker_emb_dim: int
    ci_aggregate: str
    target_channel_idx: int
    future_cov_dim: int
    use_future_covariate: bool
    line_target_type: str
    band_target_type: str
    ticker_registry_path: str | None
    tickers: list[str] | None
    limit_tickers: int | None
    seed: int
    device: str
    num_workers: int | str
    compile_model: bool
    ci_target_fast: bool
    use_direction_head: bool
    use_wandb: bool
    wandb_project: str
    model_ver: str
    early_stop_patience: int
    early_stop_min_delta: float


@dataclass
class EarlyStoppingState:
    enabled: bool
    should_stop: bool
    best_epoch: int | None
    best_value: float
    epochs_since_improve: int


class EarlyStopping:
    """검증 손실 기준으로 학습을 조기에 종료한다."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        if mode != "min":
            raise ValueError("현재 EarlyStopping은 mode='min'만 지원합니다.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.enabled = patience > 0
        self.best_value = float("inf")
        self.best_epoch: int | None = None
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.epochs_since_improve = 0
        self.should_stop = False

    def step(self, metric: float, epoch: int, model) -> bool:
        if not self.enabled:
            self.best_value = metric if self.best_epoch is None else min(self.best_value, metric)
            if self.best_epoch is None:
                self.best_epoch = epoch
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            return False

        improved = metric < (self.best_value - self.min_delta)
        if improved or self.best_epoch is None:
            self.best_value = metric
            self.best_epoch = epoch
            self.epochs_since_improve = 0
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.should_stop = False
            return False

        self.epochs_since_improve += 1
        self.should_stop = self.epochs_since_improve >= self.patience
        return self.should_stop

    def restore_best(self, model) -> None:
        if self.best_state_dict is None:
            return
        model.load_state_dict(self.best_state_dict)

    def snapshot(self) -> EarlyStoppingState:
        return EarlyStoppingState(
            enabled=self.enabled,
            should_stop=self.should_stop,
            best_epoch=self.best_epoch,
            best_value=self.best_value,
            epochs_since_improve=self.epochs_since_improve,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 멀티헤드 시계열 모델 학습")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="patchtst")
    parser.add_argument("--timeframe", choices=["1D", "1W"], default="1D")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--q-low", type=float, default=0.1)
    parser.add_argument("--q-high", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--lambda-line", type=float, default=1.0)
    parser.add_argument("--lambda-band", type=float, default=1.0)
    parser.add_argument("--lambda-width", type=float, default=0.1, help="레거시 호환용 인자이며 현재 손실 계산에는 사용하지 않습니다.")
    parser.add_argument("--lambda-cross", type=float, default=1.0)
    parser.add_argument("--lambda-direction", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--band-mode", choices=["direct", "param"], default="direct")
    parser.add_argument("--num-tickers", type=int, default=0)
    parser.add_argument("--ticker-emb-dim", type=int, default=32)
    parser.add_argument("--ci-aggregate", choices=["target", "mean", "attention"], default="target")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ci-target-fast", dest="ci_target_fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-direction-head", dest="use_direction_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="lens-ai")
    parser.add_argument("--line-target-type", default="raw_future_return")
    parser.add_argument("--band-target-type", default="raw_future_return")
    parser.add_argument("--save-run", action="store_true")
    parser.add_argument("--model-ver", default="v2-multihead")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--use-future-covariate", dest="use_future_covariate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_model(config: TrainConfig):
    model_cls = MODEL_REGISTRY[config.model]
    common_kwargs = {
        "n_features": MODEL_N_FEATURES,
        "seq_len": config.seq_len,
        "horizon": config.horizon,
        "dropout": config.dropout,
        "band_mode": config.band_mode,
        "num_tickers": config.num_tickers,
        "ticker_emb_dim": config.ticker_emb_dim,
    }
    if config.model == "patchtst":
        common_kwargs["ci_aggregate"] = config.ci_aggregate
        common_kwargs["target_channel_idx"] = config.target_channel_idx
        common_kwargs["ci_target_fast"] = config.ci_target_fast
    if config.model == "cnn_lstm":
        common_kwargs["use_direction_head"] = config.use_direction_head
    if config.model == "tide":
        common_kwargs["future_cov_dim"] = config.future_cov_dim if config.use_future_covariate else 0
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


def resolve_device(device_name: str) -> torch.device:
    normalized = (device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 요청했지만 현재 torch에서 사용할 수 없습니다.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"지원하지 않는 학습 디바이스입니다: {device_name}")


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


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def build_lr_lambda(*, total_steps: int, warmup_frac: float, schedule: str):
    clamped_total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(clamped_total_steps * warmup_frac), 0), clamped_total_steps - 1) if clamped_total_steps > 1 else 0

    def _lr_lambda(current_step: int) -> float:
        if schedule == "none":
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(float(current_step + 1) / float(warmup_steps), 1e-8)
        if clamped_total_steps <= warmup_steps + 1:
            return 1.0
        progress = float(current_step - warmup_steps) / float(clamped_total_steps - warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max((1e-2 + (1.0 - 1e-2) * cosine), 1e-8)

    return _lr_lambda


def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_frac: float, schedule: str):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(total_steps=total_steps, warmup_frac=warmup_frac, schedule=schedule),
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def resolve_num_workers(value: int | str, device: torch.device) -> int:
    if isinstance(value, int):
        return max(value, 0)
    normalized = str(value).strip().lower()
    if normalized != "auto":
        return max(int(normalized), 0)
    if platform.system() == "Windows":
        return 0
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or DEFAULT_NUM_WORKERS
    return max(1, min(DEFAULT_NUM_WORKERS, cpu_count // 2))


def make_loader(
    bundle: SequenceDatasetBundle | SequenceDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int | str,
) -> DataLoader:
    dataset = (
        bundle
        if isinstance(bundle, SequenceDataset)
        else TensorDataset(
            bundle.features,
            bundle.line_targets,
            bundle.band_targets,
            bundle.raw_future_returns,
            bundle.ticker_ids,
            bundle.future_covariates,
        )
    )
    use_cuda = device.type == "cuda"
    resolved_num_workers = resolve_num_workers(num_workers, device)

    def _build_loader(worker_count: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=worker_count,
            pin_memory=use_cuda,
            persistent_workers=worker_count > 0,
        )

    loader = _build_loader(resolved_num_workers)
    if resolved_num_workers <= 0:
        return loader

    try:
        iterator = iter(loader)
        next(iterator, None)
        del iterator
        return _build_loader(resolved_num_workers)
    except (PermissionError, OSError) as exc:
        print(f"경고: DataLoader worker 초기화에 실패해 num_workers=0으로 폴백합니다. ({exc})")
        return _build_loader(0)


def forward_model(model, features: torch.Tensor, ticker_id: torch.Tensor, future_covariates: torch.Tensor | None = None):
    raw_model = unwrap_model(model)
    if isinstance(raw_model, TiDE) and future_covariates is not None and raw_model.future_cov_dim > 0:
        return model(features, ticker_id=ticker_id, future_covariate=future_covariates)
    return model(features, ticker_id=ticker_id)


def run_epoch(
    *,
    model,
    loader: DataLoader,
    criterion: ForecastCompositeLoss,
    device: torch.device,
    epoch: int = 0,
    debug_label: str = "",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    grad_clip: float = 0.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "total_loss": 0.0,
        "forecast_loss": 0.0,
        "line_loss": 0.0,
        "band_loss": 0.0,
        "cross_loss": 0.0,
        "direction_loss": 0.0,
    }
    batch_count = 0
    grad_norm_total = 0.0

    for batch_index, (features, line_target, band_target, raw_future_returns, ticker_id, future_covariates) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        line_target = line_target.to(device, non_blocking=True)
        band_target = band_target.to(device, non_blocking=True)
        raw_future_returns = raw_future_returns.to(device, non_blocking=True)
        ticker_id = ticker_id.to(device, non_blocking=True)
        future_covariates = future_covariates.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with autocast_context(device):
            prediction = forward_model(model, features, ticker_id, future_covariates)
            losses = criterion(prediction, line_target, band_target, raw_future_returns)

        if not torch.isfinite(losses.total):
            message = (
                f"NaN/Inf loss at epoch {epoch} batch {batch_index}"
                + (f" [{debug_label}]" if debug_label else "")
            )
            print(message)
            raise RuntimeError(message)

        if is_train:
            losses.total.backward()
            if grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item())
            else:
                grad_norm = compute_grad_norm(model.parameters())
            grad_norm_total += grad_norm
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        metrics = losses.to_log_dict()
        for key, value in metrics.items():
            totals[key] += value
        batch_count += 1

    averaged = {key: value / max(batch_count, 1) for key, value in totals.items()}
    averaged["grad_norm_mean"] = grad_norm_total / max(batch_count, 1) if is_train else 0.0
    return averaged


def _summarize_predictions(
    *,
    line_predictions: list[torch.Tensor],
    lower_predictions: list[torch.Tensor],
    upper_predictions: list[torch.Tensor],
    line_targets: list[torch.Tensor],
    band_targets: list[torch.Tensor],
    raw_future_returns: list[torch.Tensor],
    metadata: Any | None = None,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
) -> dict[str, float]:
    return summarize_forecast_metrics(
        metadata=metadata if metadata is not None else None,
        line_predictions=torch.cat(line_predictions, dim=0),
        lower_predictions=torch.cat(lower_predictions, dim=0),
        upper_predictions=torch.cat(upper_predictions, dim=0),
        line_targets=torch.cat(line_targets, dim=0),
        band_targets=torch.cat(band_targets, dim=0),
        raw_future_returns=torch.cat(raw_future_returns, dim=0),
        line_target_type=line_target_type,
        band_target_type=band_target_type,
    )


def evaluate_loader(
    *,
    model,
    loader: DataLoader,
    criterion: ForecastCompositeLoss,
    device: torch.device,
    metadata: Any | None = None,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
) -> dict[str, float]:
    model.eval()
    totals = {
        "total_loss": 0.0,
        "forecast_loss": 0.0,
        "line_loss": 0.0,
        "band_loss": 0.0,
        "cross_loss": 0.0,
        "direction_loss": 0.0,
    }
    batch_count = 0
    line_predictions: list[torch.Tensor] = []
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    line_targets: list[torch.Tensor] = []
    band_targets: list[torch.Tensor] = []
    raw_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_id, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            line_target = line_target.to(device, non_blocking=True)
            band_target = band_target.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)

            with autocast_context(device):
                prediction = forward_model(model, features, ticker_id, future_covariates)
                losses = criterion(prediction, line_target, band_target, raw_future_returns)

            for key, value in losses.to_log_dict().items():
                totals[key] += value
            batch_count += 1

            line_pred, lower, upper = apply_band_postprocess(
                prediction.line.detach().cpu(),
                prediction.lower_band.detach().cpu(),
                prediction.upper_band.detach().cpu(),
            )
            line_predictions.append(line_pred)
            lower_predictions.append(lower)
            upper_predictions.append(upper)
            line_targets.append(line_target.detach().cpu())
            band_targets.append(band_target.detach().cpu())
            raw_targets.append(raw_future_returns.detach().cpu())

    averaged = {key: value / max(batch_count, 1) for key, value in totals.items()}
    averaged["grad_norm_mean"] = 0.0
    averaged.update(
        _summarize_predictions(
            line_predictions=line_predictions,
            lower_predictions=lower_predictions,
            upper_predictions=upper_predictions,
            line_targets=line_targets,
            band_targets=band_targets,
            raw_future_returns=raw_targets,
            metadata=metadata,
            line_target_type=line_target_type,
            band_target_type=band_target_type,
        )
    )
    return averaged


def evaluate_bundle(
    model,
    bundle: SequenceDatasetBundle | SequenceDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int | str,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
) -> dict[str, float]:
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=num_workers)
    criterion = ForecastCompositeLoss()
    metrics = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        metadata=bundle.metadata,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
    )
    return {
        "total_loss": metrics["total_loss"],
        "forecast_loss": metrics["forecast_loss"],
        "direction_loss": metrics["direction_loss"],
        "line_loss": metrics["line_loss"],
        "band_loss": metrics["band_loss"],
        "cross_loss": metrics["cross_loss"],
        "coverage": metrics["coverage"],
        "avg_band_width": metrics["avg_band_width"],
        "direction_accuracy": metrics["direction_accuracy"],
        "mae": metrics["mae"],
        "smape": metrics["smape"],
        "mean_signed_error": metrics["mean_signed_error"],
        "overprediction_rate": metrics["overprediction_rate"],
        "mean_overprediction": metrics["mean_overprediction"],
        "spearman_ic": metrics["spearman_ic"],
        "top_k_long_spread": metrics["top_k_long_spread"],
        "top_k_short_spread": metrics["top_k_short_spread"],
        "long_short_spread": metrics["long_short_spread"],
        "fee_adjusted_return": metrics["fee_adjusted_return"],
        "fee_adjusted_sharpe": metrics["fee_adjusted_sharpe"],
        "fee_adjusted_turnover": metrics["fee_adjusted_turnover"],
    }


def maybe_init_wandb(
    config: TrainConfig,
    run_id: str,
    *,
    group: str | None = None,
    name: str | None = None,
    config_override: dict[str, Any] | None = None,
):
    if not config.use_wandb:
        return None
    if wandb is None:
        print("wandb 패키지가 없어 W&B 기록을 건너뜁니다.")
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    return wandb.init(
        project=config.wandb_project,
        config=config_override or asdict(config),
        group=group,
        name=name or f"{config.model}-{config.timeframe}-{run_id[:8]}",
        reinit=True,
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


def run_training(
    config: TrainConfig,
    *,
    save_run: bool,
    precomputed_bundles: tuple | None = None,
    enable_compile: bool = True,
    trial: Any | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    wandb_config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_id = f"{config.model}-{config.timeframe}-{uuid4().hex[:12]}"
    print(f"학습 디바이스: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if precomputed_bundles is not None:
        train_bundle, val_bundle, test_bundle, mean, std, plan = precomputed_bundles
    else:
        train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
            timeframe=config.timeframe,
            seq_len=config.seq_len,
            horizon=config.horizon,
            tickers=config.tickers,
            limit_tickers=config.limit_tickers,
            include_future_covariate=config.model == "tide" and config.use_future_covariate,
            line_target_type=config.line_target_type,
            band_target_type=config.band_target_type,
        )
    config.num_tickers = plan.num_tickers
    config.ticker_registry_path = plan.ticker_registry_path

    train_loader = make_loader(
        train_bundle,
        batch_size=config.batch_size,
        shuffle=True,
        device=device,
        num_workers=config.num_workers,
    )
    val_loader = make_loader(
        val_bundle,
        batch_size=config.batch_size,
        shuffle=False,
        device=device,
        num_workers=config.num_workers,
    )

    model = build_model(config).to(device)
    if config.compile_model and enable_compile:
        model = maybe_compile_model(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_train_steps = max(len(train_loader) * config.epochs, 1)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_train_steps,
        warmup_frac=config.warmup_frac,
        schedule=config.lr_schedule,
    )
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
        lambda_direction=config.lambda_direction,
        band_mode=config.band_mode,
    )
    early_stopping = EarlyStopping(
        patience=0 if config.early_stop_patience < 0 else config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        mode="min",
    )

    wandb_run = maybe_init_wandb(
        config,
        run_id,
        group=wandb_group,
        name=wandb_name,
        config_override=wandb_config_override,
    )
    best_summary: dict[str, float] = {}
    grad_norm_history: list[float] = []

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            debug_label=f"run_id={run_id}" + (f" trial={trial.number}" if trial is not None else ""),
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=config.grad_clip,
        )
        val_summary = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metadata=val_bundle.metadata,
            line_target_type=config.line_target_type,
            band_target_type=config.band_target_type,
        )
        grad_norm_history.append(float(train_metrics.get("grad_norm_mean", 0.0)))

        val_total = val_summary["total_loss"]
        val_forecast = val_summary["forecast_loss"]
        stop_triggered = early_stopping.step(val_forecast, epoch, unwrap_model(model))
        early_state = early_stopping.snapshot()

        if early_state.best_epoch == epoch:
            best_summary = {
                **train_metrics,
                **val_summary,
            }

        summary = {
            "epoch": epoch,
            "epoch_seconds": round(time.perf_counter() - epoch_started_at, 4),
            "val_total": val_total,
            "val_forecast": val_forecast,
            "best_so_far": early_state.best_value,
            "epochs_since_improve": early_state.epochs_since_improve,
            "lr": current_lr(optimizer),
            "grad_norm_mean": train_metrics.get("grad_norm_mean", 0.0),
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_summary.items()},
        }
        print(json.dumps(summary, ensure_ascii=False))
        print(
            f"val_total={val_total:.6f} val_forecast={val_forecast:.6f} best_so_far={early_state.best_value:.6f} "
            f"epochs_since_improve={early_state.epochs_since_improve} "
            f"lr={current_lr(optimizer):.8f} grad_norm_mean={train_metrics.get('grad_norm_mean', 0.0):.6f}"
        )
        if train_metrics.get("grad_norm_mean", 0.0) > 100.0:
            print(f"경고: epoch {epoch}의 평균 grad_norm이 100을 초과했습니다. ({train_metrics['grad_norm_mean']:.6f})")

        if wandb_run is not None:
            wandb.log(summary)

        if trial is not None:
            trial.report(val_forecast, epoch)
            if trial.should_prune():
                if wandb_run is not None:
                    wandb_run.finish()
                if optuna is None:
                    raise RuntimeError("Optuna가 필요하지만 설치되지 않았습니다.")
                raise optuna.TrialPruned()

        if stop_triggered:
            print(
                "EarlyStopping triggered at epoch "
                f"{epoch} (best=epoch {early_state.best_epoch}, val={early_state.best_value:.6f})"
            )
            break

    early_stopping.restore_best(unwrap_model(model))
    checkpoint_metrics = {
        **best_summary,
        "best_epoch": early_stopping.best_epoch,
        "best_val_total": early_stopping.best_value,
        "best_val_forecast_loss": early_stopping.best_value,
        "best_val_total_including_direction": best_summary.get("total_loss"),
        "early_stopped": bool(early_stopping.should_stop),
        "grad_norm_history": grad_norm_history,
        "best_grad_norm_mean": best_summary.get("grad_norm_mean", 0.0),
    }
    checkpoint_path = save_checkpoint(model, config, run_id, checkpoint_metrics, mean, std)

    test_quality = evaluate_bundle(
        model,
        test_bundle,
        device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        line_target_type=config.line_target_type,
        band_target_type=config.band_target_type,
    )
    result = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "best_metrics": checkpoint_metrics,
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
                "val_metrics": checkpoint_metrics,
                "test_metrics": test_quality,
                "config": {
                    **asdict(config),
                    "config_hash": config_hash,
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                    "grad_norm_history": grad_norm_history,
                    "best_val_loss": checkpoint_metrics.get("best_val_total"),
                    "best_epoch": checkpoint_metrics.get("best_epoch"),
                    "best_val_total": checkpoint_metrics.get("best_val_total"),
                    "early_stopped": checkpoint_metrics.get("early_stopped"),
                },
                "checkpoint_path": str(checkpoint_path),
            }
        )

    return result


def train(config: TrainConfig, *, save_run: bool) -> dict[str, Any]:
    return run_training(config, save_run=save_run)


def summarize_dataset_plan(
    plan: DatasetPlan,
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
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
        "num_tickers": plan.num_tickers,
        "ticker_registry_path": plan.ticker_registry_path,
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
        "num_tickers": plan.num_tickers,
        "ticker_registry_path": plan.ticker_registry_path,
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
    config.num_tickers = plan.num_tickers
    config.ticker_registry_path = plan.ticker_registry_path
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
        lambda_direction=config.lambda_direction,
        band_mode=config.band_mode,
    )
    sample_input = torch.randn(4, config.seq_len, MODEL_N_FEATURES)
    sample_ticker_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    sample_future_covariates = torch.randn(4, config.horizon, FUTURE_COVARIATE_DIM)
    sample_line_target = torch.randn(4, config.horizon)
    sample_band_target = torch.randn(4, config.horizon)
    sample_raw_future_returns = sample_band_target.clone()
    with torch.no_grad():
        output = forward_model(
            model,
            sample_input,
            sample_ticker_ids,
            sample_future_covariates if config.model == "tide" and config.use_future_covariate else None,
        )
        loss = criterion(output, sample_line_target, sample_band_target, sample_raw_future_returns)
        line_pp, lower_pp, upper_pp = apply_band_postprocess(output.line, output.lower_band, output.upper_band)
    summary = summarize_plan_only(plan)
    summary["forward_smoke"] = {
        "line_shape": list(output.line.shape),
        "lower_shape": list(output.lower_band.shape),
        "upper_shape": list(output.upper_band.shape),
        "postprocess_lower_le_upper": bool(torch.all(lower_pp <= upper_pp).item()),
        "loss_total": float(loss.total.detach().cpu()),
        "line_preserved": bool(torch.equal(line_pp, output.line)),
        "ticker_id_shape": list(sample_ticker_ids.shape),
        "contains_nan_or_inf": bool(
            not torch.isfinite(output.line).all()
            or not torch.isfinite(output.lower_band).all()
            or not torch.isfinite(output.upper_band).all()
        ),
    }
    return summary


if __name__ == "__main__":
    args = parse_args()
    resolved_use_future_covariate = args.use_future_covariate
    if resolved_use_future_covariate is None:
        resolved_use_future_covariate = args.model == "tide"
    config = TrainConfig(
        model=args.model,
        timeframe=args.timeframe,
        horizon=args.horizon or default_horizon(args.timeframe),
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        grad_clip=args.grad_clip,
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
        lambda_direction=args.lambda_direction,
        dropout=args.dropout,
        band_mode=args.band_mode,
        num_tickers=args.num_tickers,
        ticker_emb_dim=args.ticker_emb_dim,
        ci_aggregate=args.ci_aggregate,
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=bool(resolved_use_future_covariate),
        line_target_type=args.line_target_type,
        band_target_type=args.band_target_type,
        ticker_registry_path=None,
        tickers=args.tickers,
        limit_tickers=args.limit_tickers,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        compile_model=args.compile_model,
        ci_target_fast=args.ci_target_fast,
        use_direction_head=args.use_direction_head,
        use_wandb=bool(args.use_wandb and not args.dry_run),
        wandb_project=args.wandb_project,
        model_ver=args.model_ver,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    if args.dry_run:
        result = run_dry(config)
    else:
        result = train(config, save_run=args.save_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
