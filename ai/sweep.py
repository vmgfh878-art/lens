from __future__ import annotations

# Windows에서는 torch DLL을 다른 네이티브 패키지보다 먼저 로드해야 한다.
from ai.preprocessing import FUTURE_COVARIATE_DIM, default_horizon, prepare_dataset_splits  # noqa: E402
from ai.train import TrainConfig, run_training  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import optuna  # noqa: E402
from optuna.importance import get_param_importances  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE_URL = "sqlite:///lens_optuna.db"
PLOT_DIR = PROJECT_ROOT / "docs" / "cp8_sweep_plots"


def build_study(
    study_name: str,
    storage_url: str = DEFAULT_STORAGE_URL,
    direction: str = "minimize",
    *,
    max_resource: int = 50,
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min(10, max_resource),
            max_resource=max_resource,
            reduction_factor=3,
        ),
        load_if_exists=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna 기반 Lens 학습 스윕")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage-url", default=DEFAULT_STORAGE_URL)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--model", choices=["patchtst", "cnn_lstm", "tide"], default="patchtst")
    parser.add_argument("--timeframe", choices=["1D", "1W"], default="1D")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-tickers", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ci-target-fast", dest="ci_target_fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="lens-ai")
    return parser.parse_args()


def _resolved_seq_len(timeframe: str, seq_len: int | None) -> int:
    if seq_len is not None:
        return seq_len
    return 252 if timeframe == "1D" else 104


def namespace_to_config(base_args: argparse.Namespace, *, lr: float, weight_decay: float, dropout: float) -> TrainConfig:
    timeframe = base_args.timeframe
    return TrainConfig(
        model=base_args.model,
        timeframe=timeframe,
        horizon=base_args.horizon or default_horizon(timeframe),
        seq_len=_resolved_seq_len(timeframe, base_args.seq_len),
        epochs=base_args.max_epoch,
        batch_size=base_args.batch_size,
        lr=lr,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=weight_decay,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=1.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=dropout,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=base_args.model == "tide",
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=base_args.limit_tickers,
        seed=base_args.seed,
        device=base_args.device,
        num_workers=base_args.num_workers,
        compile_model=base_args.compile_model,
        ci_target_fast=base_args.ci_target_fast,
        use_direction_head=False,
        use_wandb=bool(base_args.use_wandb and os.environ.get("WANDB_MODE") != "disabled"),
        wandb_project=base_args.wandb_project,
        model_ver="v2-multihead",
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
    )


def objective_lr_sweep(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
    *,
    precomputed_bundles: tuple | None = None,
) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    config = namespace_to_config(base_args, lr=lr, weight_decay=weight_decay, dropout=dropout)
    run_name = f"trial_{trial.number}"
    wandb_config = {
        **vars(base_args),
        **trial.params,
        "seq_len": config.seq_len,
        "horizon": config.horizon,
    }
    try:
        result = run_training(
            config,
            save_run=False,
            precomputed_bundles=precomputed_bundles,
            enable_compile=False,
            trial=trial,
            wandb_group=base_args.study_name,
            wandb_name=run_name,
            wandb_config_override=wandb_config,
        )
    except optuna.TrialPruned:
        raise

    best_val_total = float(result["best_metrics"]["best_val_total"])
    trial.set_user_attr("run_id", result["run_id"])
    trial.set_user_attr("checkpoint_path", result["checkpoint_path"])
    trial.set_user_attr("test_metrics", result["test_metrics"])
    return best_val_total


def _save_plot(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.clf()


def save_study_plots(study: optuna.Study, plot_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import (
            plot_contour,
            plot_parallel_coordinate,
            plot_slice,
        )
    except Exception:
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for plot_name, builder in (
        ("contour.png", plot_contour),
        ("parallel_coordinate.png", plot_parallel_coordinate),
        ("slice.png", plot_slice),
    ):
        try:
            axis = builder(study)
            _save_plot(axis.figure, plot_dir / plot_name)
            saved.append(str(plot_dir / plot_name))
        except Exception:
            continue
    return saved


def build_summary_payload(study: optuna.Study) -> dict[str, Any]:
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed, key=lambda trial: float(trial.value))[:5]
    try:
        param_importances = get_param_importances(study)
    except Exception:
        param_importances = {}
    return {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "top5": [
            {
                "number": trial.number,
                "value": float(trial.value),
                "params": trial.params,
            }
            for trial in top_trials
        ],
        "param_importances": param_importances,
        "completed_trials": len(completed),
        "pruned_trials": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
    }


def maybe_log_summary_to_wandb(base_args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not base_args.use_wandb or wandb is None or os.environ.get("WANDB_MODE") == "disabled":
        return
    run = wandb.init(
        project=base_args.wandb_project,
        group=base_args.study_name,
        name="study_summary",
        config=summary,
        reinit=True,
    )
    wandb.log(summary)
    run.finish()


def run_sweep(base_args: argparse.Namespace) -> dict[str, Any]:
    bootstrap_config = namespace_to_config(base_args, lr=1e-4, weight_decay=1e-2, dropout=0.2)
    dataset_started_at = time.perf_counter()
    precomputed_bundles = prepare_dataset_splits(
        timeframe=bootstrap_config.timeframe,
        seq_len=bootstrap_config.seq_len,
        horizon=bootstrap_config.horizon,
        tickers=bootstrap_config.tickers,
        limit_tickers=bootstrap_config.limit_tickers,
        include_future_covariate=bootstrap_config.model == "tide" and bootstrap_config.use_future_covariate,
    )
    dataset_build_seconds = round(time.perf_counter() - dataset_started_at, 4)
    print(json.dumps({"dataset_build_seconds": dataset_build_seconds, "dataset_build_mode": "shared_once"}, ensure_ascii=False))

    study = build_study(
        base_args.study_name,
        storage_url=base_args.storage_url,
        max_resource=base_args.max_epoch,
    )
    study.optimize(
        lambda trial: objective_lr_sweep(trial, base_args, precomputed_bundles=precomputed_bundles),
        n_trials=base_args.n_trials,
    )
    summary = build_summary_payload(study)
    summary["dataset_build_seconds"] = dataset_build_seconds
    summary["plot_paths"] = save_study_plots(study, PLOT_DIR)
    maybe_log_summary_to_wandb(base_args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
