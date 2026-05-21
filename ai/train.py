from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any
from uuid import uuid4

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.finite import (
    FiniteCheckResult,
    assert_finite_metrics,
    check_metrics_finite,
    is_nan_safe_better,
    tensor_finite_summary,
)
from ai.local_logging import LocalTrainingProgressLogger
from ai.loss import BandOnlyLoss, ForecastCompositeLoss, LineRegimeLoss, LineV2Loss, LineWarningLoss
from ai.models.cnn_lstm import CNNLSTM
from ai.models.common import BandOutput, ForecastOutput, LineRegimeOutput, LineV2Output, LineWarningOutput
from ai.models.patchtst import PatchTST
from ai.models.tcn_quantile import TCNQuantile
from ai.models.tide import TiDE
from ai.postprocess import apply_band_postprocess
from ai.evaluation import (
    summarize_band_metrics,
    summarize_forecast_metrics,
    summarize_line_regime_metrics,
    summarize_line_v2_metrics,
)
from ai.preprocessing import (
    DatasetPlan,
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SequenceDataset,
    SequenceDatasetBundle,
    build_dataset_plan,
    dataset_plan_split_metadata,
    default_horizon,
    fetch_feature_index_frame,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from ai.splits import SPLIT_MODE_CHOICES
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
    "tcn_quantile": TCNQuantile,
}
DEFAULT_NUM_WORKERS = 6

AMP_DTYPE_CHOICES = ("bf16", "fp16", "off")
AMP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED_NAN = "failed_nan"
RUN_STATUS_FAILED_QUALITY_GATE = "failed_quality_gate"
NAN_STREAK_LIMIT = 3
CLI_CUDA_CLEANUP_STATE: dict[str, Any] | None = None
CHECKPOINT_SELECTION_CHOICES = ("val_total", "line_gate", "band_gate", "combined_gate", "coverage_gate")
MODEL_ROLE_CHOICES = ("legacy", "line_v2", "line_regime", "line_warning", "band")
FEATURE_SET_PLAN_PATH = PROJECT_ROOT / "docs" / "cp63_bm_feature_set_plan.json"
DEFAULT_LOCAL_LOG_DIR = "logs/runs"
WANDB_STATUS_ONLINE_OK = "online_ok"
WANDB_STATUS_PACKAGE_MISSING = "package_missing"
WANDB_STATUS_DISABLED_BY_ENV = "disabled_by_env"
WANDB_STATUS_DISABLED_BY_CLI = "disabled_by_cli"
WANDB_STATUS_DISABLED_MISSING_KEY = "disabled_missing_key"
WANDB_STATUS_DISABLED_INIT_FAILED = "disabled_init_failed"

COVERAGE_GATE_MIN = 0.75
COVERAGE_GATE_MAX = 0.95
COVERAGE_GATE_MAX_UPPER_BREACH = 0.15
COVERAGE_GATE_MAX_LOWER_BREACH = 0.20
BAND_GATE_TARGET_COVERAGE = 0.85
BAND_GATE_MAX_COVERAGE_ABS_ERROR = 0.08
BAND_GATE_MAX_TAIL_ABS_ERROR = 0.08
BAND_GATE_MAX_TAIL_IMBALANCE = 0.12


def resolve_persisted_run_status(
    *,
    gate_failed: bool | None = None,
    coverage_gate_failed: bool | None = None,
) -> str:
    if gate_failed is None:
        gate_failed = bool(coverage_gate_failed)
    return RUN_STATUS_FAILED_QUALITY_GATE if gate_failed else RUN_STATUS_COMPLETED


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
    fp32_modules: str
    use_wandb: bool
    wandb_project: str
    model_ver: str
    early_stop_patience: int
    early_stop_min_delta: float
    checkpoint_selection: str = "val_total"
    amp_dtype: str = "bf16"
    detect_anomaly: bool = False
    explicit_cuda_cleanup: bool = False
    hard_exit_after_result: bool = False
    use_revin: bool = True
    patch_len: int = 16
    patch_stride: int = 8
    patchtst_d_model: int = 128
    patchtst_n_heads: int = 8
    patchtst_n_layers: int = 3
    feature_set: str = "full_features"
    feature_columns: list[str] | None = None
    n_features: int = MODEL_N_FEATURES
    market_data_provider: str | None = None
    split_mode: str = "calendar_aligned"
    lower_band_loss_weight: float = 1.0
    upper_band_loss_weight: float = 1.0
    model_role: str = "legacy"
    lambda_risk: float = 0.5
    lambda_regime: float = 0.5
    lambda_ordinal: float = 0.1
    risk_decision_threshold: float = 0.5
    regime_thresholds: list[float] | None = None


@dataclass(frozen=True)
class WandbInitOutcome:
    run: Any | None
    status: str
    message: str | None = None
    run_id: str | None = None
    run_url: str | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "run_id": self.run_id,
            "run_url": self.run_url,
        }


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
        # NaN/Inf metric은 절대 best가 될 수 없도록 가드한다 (CP12).
        improved = is_nan_safe_better(metric, self.best_value if self.best_epoch is not None else None, mode="min", min_delta=self.min_delta)
        if not self.enabled:
            if improved or self.best_epoch is None:
                # disabled여도 첫 valid metric이나 더 나은 metric이 있으면 best 갱신.
                if improved:
                    self.best_value = metric
                    self.best_epoch = epoch
                    self.best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                elif self.best_epoch is None and metric is not None and math.isfinite(metric):
                    # 첫 finite metric. best로 채택.
                    self.best_value = metric
                    self.best_epoch = epoch
                    self.best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            return False

        if improved:
            self.best_value = metric
            self.best_epoch = epoch
            self.epochs_since_improve = 0
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.should_stop = False
            return False

        # NaN인 경우 또는 개선 없음. patience 카운트.
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


@dataclass
class CheckpointCandidate:
    epoch: int
    metrics: dict[str, float]
    state_dict: dict[str, torch.Tensor]


@dataclass
class CheckpointSelectionResult:
    candidate: CheckpointCandidate
    checkpoint_selection: str
    selected_reason: str
    gate_type: str
    gate_failed: bool
    line_gate_pass: bool
    band_gate_pass: bool
    combined_gate_pass: bool
    coverage_gate_failed: bool
    best_val_total_epoch: int | None


def load_feature_set_plan(path: Path | None = None) -> dict[str, Any]:
    plan_path = path or FEATURE_SET_PLAN_PATH
    if not plan_path.exists():
        raise FileNotFoundError(f"feature set plan 파일이 없습니다: {plan_path}")
    return json.loads(plan_path.read_text(encoding="utf-8"))


def resolve_feature_columns(feature_set: str | None, *, plan_path: Path | None = None) -> list[str]:
    selected_set = str(feature_set or "full_features").strip()
    default_path = plan_path or FEATURE_SET_PLAN_PATH
    if selected_set == "full_features" and not default_path.exists():
        return list(MODEL_FEATURE_COLUMNS)

    plan = load_feature_set_plan(default_path)
    feature_sets = plan.get("feature_sets") or {}
    if selected_set not in feature_sets:
        allowed = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(f"알 수 없는 feature_set입니다: {selected_set}. 허용값: {allowed}")

    spec = feature_sets[selected_set]
    if spec.get("requires_contract_change"):
        raise ValueError(f"{selected_set}은 feature contract 변경이 필요한 후보라 CP64 smoke 입력으로 사용할 수 없습니다.")
    if spec.get("indicator_only_candidates"):
        raise ValueError(f"{selected_set}은 indicator-only 후보를 포함하므로 현재 모델 feature로 사용할 수 없습니다.")

    columns = [str(column) for column in spec.get("columns", [])]
    if not columns:
        raise ValueError(f"{selected_set} feature_set에 columns가 없습니다.")
    missing = [column for column in columns if column not in MODEL_FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"{selected_set}에 현재 MODEL_FEATURE_COLUMNS가 아닌 피처가 있습니다: {missing}")
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"{selected_set}에 중복 피처가 있습니다: {duplicates}")
    return columns


def apply_feature_set_to_config(config: TrainConfig) -> None:
    columns = config.feature_columns or resolve_feature_columns(config.feature_set)
    config.feature_columns = list(columns)
    config.n_features = len(columns)


def _feature_column_indices(columns: list[str]) -> list[int]:
    return [MODEL_FEATURE_COLUMNS.index(column) for column in columns]


def _select_bundle_features(
    bundle: SequenceDatasetBundle | SequenceDataset,
    *,
    indices: list[int],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> SequenceDatasetBundle | SequenceDataset:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    if isinstance(bundle, SequenceDatasetBundle):
        return SequenceDatasetBundle(
            features=bundle.features.index_select(2, index_tensor),
            line_targets=bundle.line_targets,
            band_targets=bundle.band_targets,
            raw_future_returns=bundle.raw_future_returns,
            anchor_closes=bundle.anchor_closes,
            ticker_ids=bundle.ticker_ids,
            future_covariates=bundle.future_covariates,
            metadata=bundle.metadata.copy(),
        )

    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        copied = dict(arrays)
        copied["features"] = arrays["features"][:, indices].copy()
        ticker_arrays[ticker] = copied
    selected_mean = mean.index_select(0, index_tensor) if mean is not None else None
    selected_std = std.index_select(0, index_tensor) if std is not None else None
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=selected_mean,
        std=selected_std,
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )


def apply_feature_columns_to_splits(
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_columns: list[str],
) -> tuple[
    SequenceDatasetBundle | SequenceDataset,
    SequenceDatasetBundle | SequenceDataset,
    SequenceDatasetBundle | SequenceDataset,
    torch.Tensor,
    torch.Tensor,
]:
    if feature_columns == list(MODEL_FEATURE_COLUMNS):
        return train_bundle, val_bundle, test_bundle, mean, std

    indices = _feature_column_indices(feature_columns)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    selected_mean = mean.index_select(0, index_tensor)
    selected_std = std.index_select(0, index_tensor)
    return (
        _select_bundle_features(train_bundle, indices=indices, mean=mean, std=std),
        _select_bundle_features(val_bundle, indices=indices, mean=mean, std=std),
        _select_bundle_features(test_bundle, indices=indices, mean=mean, std=std),
        selected_mean,
        selected_std,
    )


def clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _metric_view(metrics: dict[str, Any], layer: str) -> dict[str, Any]:
    nested = metrics.get(layer)
    return nested if isinstance(nested, dict) else metrics


def _finite_metric(
    metrics: dict[str, Any],
    key: str,
    default: float = float("nan"),
    *,
    layer: str | None = None,
) -> float:
    source = _metric_view(metrics, layer) if layer else metrics
    try:
        return float(source.get(key, metrics.get(key, default)))
    except (TypeError, ValueError):
        return float("nan")


def normalize_checkpoint_selection(mode: str) -> str:
    if mode == "coverage_gate":
        return "combined_gate"
    return mode


def checkpoint_role_for_gate(gate_type: str) -> str:
    if gate_type == "line_gate":
        return "line_model"
    if gate_type == "band_gate":
        return "band_model"
    if gate_type == "combined_gate":
        return "combined_model"
    return "legacy_val_total"


def line_gate_eligible(metrics: dict[str, Any]) -> bool:
    spearman_ic = _finite_metric(metrics, "spearman_ic", layer="line_metrics")
    long_short_spread = _finite_metric(metrics, "long_short_spread", layer="line_metrics")
    mae = _finite_metric(metrics, "mae", layer="line_metrics")
    smape = _finite_metric(metrics, "smape", layer="line_metrics")
    return (
        math.isfinite(spearman_ic)
        and spearman_ic > 0.0
        and math.isfinite(long_short_spread)
        and long_short_spread > 0.0
        and math.isfinite(mae)
        and math.isfinite(smape)
    )


def band_gate_eligible(metrics: dict[str, Any]) -> bool:
    coverage = _finite_metric(metrics, "empirical_coverage", layer="band_metrics")
    if not math.isfinite(coverage):
        coverage = _finite_metric(metrics, "coverage", layer="band_metrics")
    nominal_coverage = _finite_metric(metrics, "nominal_coverage", layer="band_metrics")
    if not math.isfinite(nominal_coverage):
        nominal_coverage = BAND_GATE_TARGET_COVERAGE
    coverage_abs_error = _finite_metric(metrics, "coverage_abs_error", layer="band_metrics")
    if not math.isfinite(coverage_abs_error) and math.isfinite(coverage) and math.isfinite(nominal_coverage):
        coverage_abs_error = abs(coverage - nominal_coverage)
    upper_breach = _finite_metric(metrics, "upper_breach_rate", layer="band_metrics")
    lower_breach = _finite_metric(metrics, "lower_breach_rate", layer="band_metrics")
    expected_tail = (1.0 - nominal_coverage) / 2.0 if 0.0 < nominal_coverage < 1.0 else float("nan")
    upper_breach_abs_error = _finite_metric(metrics, "upper_breach_abs_error", layer="band_metrics")
    if not math.isfinite(upper_breach_abs_error) and math.isfinite(upper_breach) and math.isfinite(expected_tail):
        upper_breach_abs_error = abs(upper_breach - expected_tail)
    lower_breach_abs_error = _finite_metric(metrics, "lower_breach_abs_error", layer="band_metrics")
    if not math.isfinite(lower_breach_abs_error) and math.isfinite(lower_breach) and math.isfinite(expected_tail):
        lower_breach_abs_error = abs(lower_breach - expected_tail)
    tail_imbalance = abs(lower_breach - upper_breach) if math.isfinite(lower_breach) and math.isfinite(upper_breach) else float("inf")
    avg_band_width = _finite_metric(metrics, "avg_band_width", layer="band_metrics")
    band_loss = _finite_metric(metrics, "band_loss")
    return (
        math.isfinite(coverage)
        and math.isfinite(nominal_coverage)
        and 0.0 < nominal_coverage < 1.0
        and math.isfinite(coverage_abs_error)
        and coverage_abs_error <= BAND_GATE_MAX_COVERAGE_ABS_ERROR
        and math.isfinite(upper_breach)
        and math.isfinite(upper_breach_abs_error)
        and upper_breach_abs_error <= BAND_GATE_MAX_TAIL_ABS_ERROR
        and math.isfinite(lower_breach)
        and math.isfinite(lower_breach_abs_error)
        and lower_breach_abs_error <= BAND_GATE_MAX_TAIL_ABS_ERROR
        and tail_imbalance <= BAND_GATE_MAX_TAIL_IMBALANCE
        and math.isfinite(avg_band_width)
        and avg_band_width > 0.0
        and math.isfinite(band_loss)
    )


def combined_gate_eligible(metrics: dict[str, float]) -> bool:
    return line_gate_eligible(metrics) and band_gate_eligible(metrics)


def coverage_gate_eligible(metrics: dict[str, float]) -> bool:
    return combined_gate_eligible(metrics)


def line_gate_sort_key(candidate: CheckpointCandidate) -> tuple[float, float, float, float]:
    metrics = candidate.metrics
    return (
        -_finite_metric(metrics, "spearman_ic", float("-inf"), layer="line_metrics"),
        -_finite_metric(metrics, "long_short_spread", float("-inf"), layer="line_metrics"),
        _finite_metric(metrics, "mae", float("inf"), layer="line_metrics"),
        _finite_metric(metrics, "forecast_loss", _finite_metric(metrics, "total_loss", float("inf"))),
    )


def band_gate_sort_key(candidate: CheckpointCandidate) -> tuple[float, float, float, float, float]:
    metrics = candidate.metrics
    coverage = _finite_metric(metrics, "empirical_coverage", layer="band_metrics")
    if not math.isfinite(coverage):
        coverage = _finite_metric(metrics, "coverage", layer="band_metrics")
    nominal_coverage = _finite_metric(metrics, "nominal_coverage", layer="band_metrics")
    if not math.isfinite(nominal_coverage):
        nominal_coverage = BAND_GATE_TARGET_COVERAGE
    coverage_abs_error = _finite_metric(metrics, "coverage_abs_error", layer="band_metrics")
    if not math.isfinite(coverage_abs_error) and math.isfinite(coverage) and math.isfinite(nominal_coverage):
        coverage_abs_error = abs(coverage - nominal_coverage)
    lower_breach = _finite_metric(metrics, "lower_breach_rate", float("inf"), layer="band_metrics")
    upper_breach = _finite_metric(metrics, "upper_breach_rate", float("inf"), layer="band_metrics")
    tail_imbalance = abs(lower_breach - upper_breach) if math.isfinite(lower_breach) and math.isfinite(upper_breach) else float("inf")
    interval_score = _finite_metric(metrics, "asymmetric_interval_score", layer="band_metrics")
    if not math.isfinite(interval_score):
        interval_score = _finite_metric(metrics, "interval_score", float("inf"), layer="band_metrics")
    return (
        coverage_abs_error,
        tail_imbalance,
        interval_score,
        _finite_metric(metrics, "avg_band_width", float("inf"), layer="band_metrics"),
        _finite_metric(metrics, "band_loss", float("inf")),
    )


def combined_gate_sort_key(candidate: CheckpointCandidate) -> tuple[float, float, float, float]:
    metrics = candidate.metrics
    return (
        _finite_metric(metrics, "upper_breach_rate", float("inf")),
        -_finite_metric(metrics, "spearman_ic", float("-inf")),
        -_finite_metric(metrics, "long_short_spread", float("-inf")),
        _finite_metric(metrics, "forecast_loss", _finite_metric(metrics, "total_loss", float("inf"))),
    )


class CheckpointSelector:
    """validation 목적에 맞는 checkpoint 후보를 epoch별로 보관하고 선택한다."""

    def __init__(self, mode: str = "val_total") -> None:
        if mode not in CHECKPOINT_SELECTION_CHOICES:
            raise ValueError(f"지원하지 않는 checkpoint_selection 값입니다: {mode}")
        self.mode = mode
        self.gate_type = normalize_checkpoint_selection(mode)
        self.best_val_total_candidate: CheckpointCandidate | None = None
        self.best_gate_candidate: CheckpointCandidate | None = None

    def _gate_eligible(self, metrics: dict[str, float]) -> bool:
        if self.gate_type == "line_gate":
            return line_gate_eligible(metrics)
        if self.gate_type == "band_gate":
            return band_gate_eligible(metrics)
        if self.gate_type == "combined_gate":
            return combined_gate_eligible(metrics)
        return False

    def _gate_sort_key(self, candidate: CheckpointCandidate):
        if self.gate_type == "line_gate":
            return line_gate_sort_key(candidate)
        if self.gate_type == "band_gate":
            return band_gate_sort_key(candidate)
        if self.gate_type == "combined_gate":
            return combined_gate_sort_key(candidate)
        return ()

    def _selected_reason(self, passed: bool) -> str:
        if self.mode == "coverage_gate":
            return "coverage_gate_eligible" if passed else "coverage_gate_failed_fallback_val_total"
        return f"{self.gate_type}_eligible" if passed else f"{self.gate_type}_failed_fallback_val_total"

    def update(self, *, epoch: int, metrics: dict[str, float], model) -> None:
        current_value = float(metrics.get("forecast_loss", metrics.get("total_loss", float("inf"))))
        is_best_val_total = False
        if self.best_val_total_candidate is None:
            is_best_val_total = True
        else:
            best_metrics = self.best_val_total_candidate.metrics
            best_value = float(best_metrics.get("forecast_loss", best_metrics.get("total_loss", float("inf"))))
            is_best_val_total = current_value < best_value

        is_best_gate = False
        if self.gate_type != "val_total" and self._gate_eligible(metrics):
            candidate_key = self._gate_sort_key(CheckpointCandidate(epoch=epoch, metrics=metrics, state_dict={}))
            if self.best_gate_candidate is None:
                is_best_gate = True
            else:
                is_best_gate = candidate_key < self._gate_sort_key(self.best_gate_candidate)

        if not (is_best_val_total or is_best_gate):
            return

        candidate = CheckpointCandidate(
            epoch=epoch,
            metrics=dict(metrics),
            state_dict=clone_state_dict(model),
        )
        if is_best_val_total:
            self.best_val_total_candidate = candidate
        if is_best_gate:
            self.best_gate_candidate = candidate

    def select(self) -> CheckpointSelectionResult:
        if self.best_val_total_candidate is None:
            raise RuntimeError("선택할 checkpoint 후보가 없습니다.")

        if self.mode == "val_total":
            metrics = self.best_val_total_candidate.metrics
            return CheckpointSelectionResult(
                candidate=self.best_val_total_candidate,
                checkpoint_selection=self.mode,
                selected_reason="val_total_best",
                gate_type="val_total",
                gate_failed=False,
                line_gate_pass=line_gate_eligible(metrics),
                band_gate_pass=band_gate_eligible(metrics),
                combined_gate_pass=combined_gate_eligible(metrics),
                coverage_gate_failed=False,
                best_val_total_epoch=self.best_val_total_candidate.epoch,
            )

        if self.best_gate_candidate is None:
            metrics = self.best_val_total_candidate.metrics
            return CheckpointSelectionResult(
                candidate=self.best_val_total_candidate,
                checkpoint_selection=self.mode,
                selected_reason=self._selected_reason(passed=False),
                gate_type=self.gate_type,
                gate_failed=True,
                line_gate_pass=line_gate_eligible(metrics),
                band_gate_pass=band_gate_eligible(metrics),
                combined_gate_pass=combined_gate_eligible(metrics),
                coverage_gate_failed=True,
                best_val_total_epoch=self.best_val_total_candidate.epoch,
            )

        metrics = self.best_gate_candidate.metrics
        return CheckpointSelectionResult(
            candidate=self.best_gate_candidate,
            checkpoint_selection=self.mode,
            selected_reason=self._selected_reason(passed=True),
            gate_type=self.gate_type,
            gate_failed=False,
            line_gate_pass=line_gate_eligible(metrics),
            band_gate_pass=band_gate_eligible(metrics),
            combined_gate_pass=combined_gate_eligible(metrics),
            coverage_gate_failed=False,
            best_val_total_epoch=self.best_val_total_candidate.epoch,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 멀티헤드 시계열 모델 학습")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="patchtst")
    parser.add_argument(
        "--model-role",
        choices=MODEL_ROLE_CHOICES,
        default="legacy",
        help="모델 출력 역할입니다. legacy는 기존 line+band, line_v2는 line+risk, band는 lower/upper만 학습합니다.",
    )
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
    parser.add_argument("--lambda-risk", type=float, default=0.5)
    parser.add_argument("--lambda-regime", type=float, default=0.5)
    parser.add_argument("--lambda-ordinal", type=float, default=0.1)
    parser.add_argument("--risk-decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--lower-band-loss-weight",
        type=float,
        default=1.0,
        help="lower quantile pinball loss 가중치입니다. 기본값은 기존 동작과 같은 1.0입니다.",
    )
    parser.add_argument(
        "--upper-band-loss-weight",
        type=float,
        default=1.0,
        help="upper quantile pinball loss 가중치입니다. 기본값은 기존 동작과 같은 1.0입니다.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--band-mode", choices=["direct", "param"], default="direct")
    parser.add_argument("--num-tickers", type=int, default=0)
    parser.add_argument("--ticker-emb-dim", type=int, default=32)
    parser.add_argument("--ci-aggregate", choices=["target", "mean", "attention"], default="target")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--market-data-provider", default=None)
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODE_CHOICES,
        default="calendar_aligned",
        help="학습 split 방식입니다. calendar_aligned는 같은 asof_date를 모든 ticker에서 같은 split에 배정합니다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ci-target-fast", dest="ci_target_fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-revin", dest="use_revin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--patchtst-d-model", type=int, default=128)
    parser.add_argument("--patchtst-n-heads", type=int, default=8)
    parser.add_argument("--patchtst-n-layers", type=int, default=3)
    parser.add_argument("--use-direction-head", dest="use_direction_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fp32-modules",
        choices=["none", "conv", "lstm", "heads", "lstm,heads", "conv,lstm,heads"],
        default="none",
        help="CNN-LSTM에서 선택한 블록만 autocast를 끄고 fp32로 강제합니다.",
    )
    parser.add_argument(
        "--wandb",
        dest="use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="W&B 기록 사용. full training에서는 init 실패 시 경고 후 자동 fallback하고, --no-wandb는 완전 비활성화합니다.",
    )
    parser.add_argument("--wandb-project", default="lens-ai")
    parser.add_argument("--line-target-type", default="raw_future_return")
    parser.add_argument("--band-target-type", default="raw_future_return")
    parser.add_argument(
        "--feature-set",
        "--feature-columns-preset",
        dest="feature_set",
        default="full_features",
        help="docs/cp63_bm_feature_set_plan.json에 정의된 모델 입력 피처 세트 이름입니다.",
    )
    parser.add_argument("--save-run", action="store_true")
    parser.add_argument("--model-ver", default="v2-multihead")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-selection",
        choices=CHECKPOINT_SELECTION_CHOICES,
        default="val_total",
        help="checkpoint 선택 기준. coverage_gate는 deprecated alias이며 combined_gate로 해석됩니다.",
    )
    parser.add_argument("--use-future-covariate", dest="use_future_covariate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=AMP_DTYPE_CHOICES, default="bf16",
                        help="CUDA autocast dtype. 'off'는 autocast 자체를 끈다 (NaN 디버그용).")
    parser.add_argument("--detect-anomaly", dest="detect_anomaly", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.autograd.set_detect_anomaly 활성화. NaN 첫 발생 backward op 추적 (느려짐).")
    parser.add_argument("--explicit-cuda-cleanup", dest="explicit_cuda_cleanup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hard-exit-after-result", dest="hard_exit_after_result", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-log",
        dest="local_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="epoch 진행률과 최종 요약을 로컬 logs/runs/{run_id}/에 기록합니다.",
    )
    parser.add_argument("--local-log-dir", default=DEFAULT_LOCAL_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_model(config: TrainConfig):
    apply_feature_set_to_config(config)
    if config.model_role not in MODEL_ROLE_CHOICES:
        raise ValueError(f"지원하지 않는 model_role입니다: {config.model_role}")
    if config.model == "tcn_quantile" and config.model_role in ("line_v2", "line_regime", "line_warning"):
        raise ValueError("TCNQuantile은 CP155 이후 band 전용 후보로만 사용합니다.")
    model_cls = MODEL_REGISTRY[config.model]
    common_kwargs = {
        "n_features": config.n_features,
        "seq_len": config.seq_len,
        "horizon": config.horizon,
        "dropout": config.dropout,
        "band_mode": config.band_mode,
        "num_tickers": config.num_tickers,
        "ticker_emb_dim": config.ticker_emb_dim,
        "output_role": config.model_role,
    }
    if config.model == "patchtst":
        common_kwargs["use_revin"] = config.use_revin
        common_kwargs["ci_aggregate"] = config.ci_aggregate
        common_kwargs["target_channel_idx"] = config.target_channel_idx
        common_kwargs["ci_target_fast"] = config.ci_target_fast
        common_kwargs["patch_len"] = config.patch_len
        common_kwargs["stride"] = config.patch_stride
        common_kwargs["d_model"] = config.patchtst_d_model
        common_kwargs["n_heads"] = config.patchtst_n_heads
        common_kwargs["n_layers"] = config.patchtst_n_layers
    if config.model == "cnn_lstm":
        common_kwargs["use_direction_head"] = config.use_direction_head
        common_kwargs["fp32_modules"] = config.fp32_modules
    if config.model == "tide":
        common_kwargs["future_cov_dim"] = config.future_cov_dim if config.use_future_covariate else 0
    return model_cls(**common_kwargs)


def build_criterion(
    config: TrainConfig,
    *,
    severe_downside_threshold: float | None = None,
) -> ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | LineWarningLoss | BandOnlyLoss:
    if config.model_role == "line_v2":
        return LineV2Loss(
            alpha=config.alpha,
            beta=config.beta,
            delta=config.delta,
            lambda_line=config.lambda_line,
            lambda_risk=config.lambda_risk,
            downside_threshold=(
                float(severe_downside_threshold)
                if severe_downside_threshold is not None
                else -0.05
            ),
        )
    if config.model_role == "line_regime":
        return LineRegimeLoss(
            alpha=config.alpha,
            beta=config.beta,
            delta=config.delta,
            lambda_line=config.lambda_line,
            lambda_regime=config.lambda_regime,
            lambda_ordinal=config.lambda_ordinal,
            regime_thresholds=getattr(config, "regime_thresholds", None),
        )
    if config.model_role == "line_warning":
        return LineWarningLoss(
            downside_threshold=(
                float(severe_downside_threshold)
                if severe_downside_threshold is not None
                else -0.03
            ),
            positive_weight=5.0,
            gamma=2.0,
            use_focal=True,
        )
    if config.model_role == "band":
        return BandOnlyLoss(
            q_low=config.q_low,
            q_high=config.q_high,
            lambda_band=config.lambda_band,
            lambda_cross=config.lambda_cross,
            band_mode=config.band_mode,
            lower_band_loss_weight=config.lower_band_loss_weight,
            upper_band_loss_weight=config.upper_band_loss_weight,
        )
    return ForecastCompositeLoss(
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
        lower_band_loss_weight=config.lower_band_loss_weight,
        upper_band_loss_weight=config.upper_band_loss_weight,
    )


def build_evaluation_criterion(
    *,
    model_role: str,
    q_low: float = 0.1,
    q_high: float = 0.9,
    band_mode: str = "direct",
    severe_downside_threshold: float | None = None,
    risk_decision_threshold: float = 0.5,
    regime_thresholds: tuple[float, float, float, float] | list[float] | None = None,
) -> ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | LineWarningLoss | BandOnlyLoss:
    """평가 루프용 criterion을 role 기준으로 한 번만 만든다."""
    if model_role == "line_v2":
        return LineV2Loss(
            downside_threshold=(
                float(severe_downside_threshold)
                if severe_downside_threshold is not None
                else -0.05
            )
        )
    if model_role == "band":
        return BandOnlyLoss(q_low=q_low, q_high=q_high, band_mode=band_mode)
    if model_role == "line_regime":
        return LineRegimeLoss(regime_thresholds=regime_thresholds)
    if model_role == "line_warning":
        return LineWarningLoss(
            downside_threshold=(
                float(severe_downside_threshold)
                if severe_downside_threshold is not None
                else -0.03
            )
        )
    return ForecastCompositeLoss(q_low=q_low, q_high=q_high, band_mode=band_mode)


def resolve_model_output_role(model) -> str:
    """compile wrapper를 벗겨 실제 모델의 output_role을 확인한다."""
    return str(getattr(unwrap_model(model), "output_role", "legacy") or "legacy")


def assert_prediction_matches_criterion(
    prediction: ForecastOutput | LineV2Output | LineRegimeOutput | LineWarningOutput | BandOutput,
    criterion: ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | LineWarningLoss | BandOnlyLoss,
    *,
    model_role: str,
    phase: str,
    run_id: str = "",
    epoch: int = -1,
    batch_index: int = -1,
) -> None:
    """출력 role과 loss role이 다르면 조용히 보정하지 않고 즉시 실패한다."""
    if isinstance(criterion, LineV2Loss):
        expected_type = LineV2Output
        expected_role = "line_v2"
    elif isinstance(criterion, LineRegimeLoss):
        expected_type = LineRegimeOutput
        expected_role = "line_regime"
    elif isinstance(criterion, LineWarningLoss):
        expected_type = LineWarningOutput
        expected_role = "line_warning"
    elif isinstance(criterion, BandOnlyLoss):
        expected_type = BandOutput
        expected_role = "band"
    else:
        expected_type = ForecastOutput
        expected_role = "legacy"
    if isinstance(prediction, expected_type):
        return
    raise TypeError(
        "model_role/criterion/output mismatch: "
        f"phase={phase} run_id={run_id} epoch={epoch} batch={batch_index} "
        f"model_role={model_role} criterion_role={expected_role} "
        f"prediction_type={type(prediction).__name__}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config_hash(config: TrainConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_local_log_base_dir(local_log_dir: str | Path) -> Path:
    path = Path(local_log_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _local_metric(metrics: dict[str, Any], key: str, *, layer: str | None = None) -> Any:
    if layer:
        nested = metrics.get(layer)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return metrics.get(key)


def _local_gate_status(checkpoint_selector: CheckpointSelector, metrics: dict[str, Any]) -> dict[str, Any]:
    line_pass = line_gate_eligible(metrics)
    band_pass = band_gate_eligible(metrics)
    combined_pass = combined_gate_eligible(metrics)
    gate_type = checkpoint_selector.gate_type
    if gate_type == "line_gate":
        gate_eligible: bool | None = line_pass
    elif gate_type == "band_gate":
        gate_eligible = band_pass
    elif gate_type == "combined_gate":
        gate_eligible = combined_pass
    else:
        gate_eligible = None
    return {
        "gate_type": gate_type,
        "gate_eligible": gate_eligible,
        "line_gate_pass": line_pass,
        "band_gate_pass": band_pass,
        "combined_gate_pass": combined_pass,
    }


def build_local_epoch_log_payload(
    *,
    run_id: str,
    epoch: int,
    epochs: int,
    elapsed_seconds: float,
    epoch_seconds: float,
    estimated_remaining_seconds: float,
    learning_rate: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    early_state: EarlyStoppingState,
    checkpoint_selector: CheckpointSelector,
    checkpoint_selection: str,
    vram_peak_allocated_mb: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "epoch": epoch,
        "epochs": epochs,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "epoch_seconds": round(epoch_seconds, 4),
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 4),
        "learning_rate": learning_rate,
        "train_total_loss": train_metrics.get("total_loss"),
        "val_total_loss": val_metrics.get("total_loss"),
        "val_forecast_loss": val_metrics.get("forecast_loss"),
        "best_so_far": early_state.best_value,
        "epochs_since_improve": early_state.epochs_since_improve,
        "checkpoint_selection": checkpoint_selection,
        "selected_reason": None,
        "gate_status": _local_gate_status(checkpoint_selector, val_metrics),
        "vram_peak_allocated_mb": vram_peak_allocated_mb,
        "ic_mean": _local_metric(val_metrics, "ic_mean", layer="line_metrics"),
        "long_short_spread": _local_metric(val_metrics, "long_short_spread", layer="line_metrics"),
        "false_safe_tail_rate": _local_metric(val_metrics, "false_safe_tail_rate", layer="line_metrics"),
        "severe_downside_recall": _local_metric(val_metrics, "severe_downside_recall", layer="line_metrics"),
        "nominal_coverage": _local_metric(val_metrics, "nominal_coverage", layer="band_metrics"),
        "empirical_coverage": _local_metric(val_metrics, "empirical_coverage", layer="band_metrics"),
        "coverage_abs_error": _local_metric(val_metrics, "coverage_abs_error", layer="band_metrics"),
        "asymmetric_interval_score": _local_metric(val_metrics, "asymmetric_interval_score", layer="band_metrics"),
    }


def build_local_config_payload(
    *,
    run_id: str,
    config: TrainConfig,
    local_log: bool,
    local_log_dir: str | Path,
    run_log_dir: Path,
    source_data_hash: str | None = None,
    dataset_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "source_data_hash": source_data_hash,
        "local_log": {
            "enabled": bool(local_log),
            "base_dir": str(resolve_local_log_base_dir(local_log_dir)),
            "run_dir": str(run_log_dir),
        },
        "config": asdict(config),
        "dataset_plan": dataset_plan,
    }


def build_local_summary_payload(
    *,
    run_id: str,
    status: str,
    config: TrainConfig,
    checkpoint_path: str | None,
    best_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    wandb_status: dict[str, Any] | None,
    total_elapsed_seconds: float,
    source_data_hash: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "model": config.model,
        "timeframe": config.timeframe,
        "horizon": config.horizon,
        "feature_set": config.feature_set,
        "source_data_hash": source_data_hash,
        "checkpoint_path": checkpoint_path,
        "best_metrics": best_metrics or {},
        "test_metrics": test_metrics or {},
        "wandb_status": wandb_status,
        "total_elapsed_seconds": round(total_elapsed_seconds, 4),
        "error": error,
    }


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


def autocast_context(device: torch.device, amp_dtype: str = "bf16"):
    """CUDA에서만 autocast를 켠다. amp_dtype='off'이거나 CPU면 nullcontext."""
    if not should_use_cuda_optimizations(device):
        return nullcontext()
    if amp_dtype == "off":
        return nullcontext()
    dtype = AMP_DTYPE_MAP.get(amp_dtype)
    if dtype is None:
        raise ValueError(f"지원하지 않는 amp_dtype입니다: {amp_dtype}")
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def _dump_nonfinite_diagnostics(
    *,
    phase: str,
    epoch: int,
    batch_index: int,
    debug_label: str,
    tensors: dict[str, torch.Tensor | None],
    losses_dict: dict[str, float] | None = None,
) -> None:
    """첫 NaN/Inf batch에서 입력 tensor와 loss component의 finite 통계를 stderr로 덤프한다."""
    summary = tensor_finite_summary(tensors)
    print(
        f"[NaN-DIAG phase={phase} epoch={epoch} batch={batch_index}"
        + (f" {debug_label}" if debug_label else "")
        + "]"
    )
    for name, info in summary.items():
        print(f"  tensor[{name}] = {info}")
    if losses_dict is not None:
        finite_flags = {key: math.isfinite(value) if isinstance(value, float) else True for key, value in losses_dict.items()}
        print(f"  loss_components_finite = {finite_flags}")
        print(f"  loss_components = {losses_dict}")


def run_epoch(
    *,
    model,
    loader: DataLoader,
    criterion: ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | BandOnlyLoss,
    device: torch.device,
    epoch: int = 0,
    debug_label: str = "",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    grad_clip: float = 0.0,
    amp_dtype: str = "bf16",
    run_id: str = "",
) -> dict[str, float]:
    is_train = optimizer is not None
    phase_label = "train" if is_train else "val"
    model.train(is_train)
    totals = {
        "total_loss": 0.0,
        "forecast_loss": 0.0,
        "line_loss": 0.0,
        "band_loss": 0.0,
        "cross_loss": 0.0,
        "direction_loss": 0.0,
        "risk_loss": 0.0,
        "regime_loss": 0.0,
    }
    batch_count = 0
    grad_norm_total = 0.0
    nan_streak = 0
    diagnostic_dumped = False
    model_role = resolve_model_output_role(model)

    for batch_index, (features, line_target, band_target, raw_future_returns, ticker_id, future_covariates) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        line_target = line_target.to(device, non_blocking=True)
        band_target = band_target.to(device, non_blocking=True)
        raw_future_returns = raw_future_returns.to(device, non_blocking=True)
        ticker_id = ticker_id.to(device, non_blocking=True)
        future_covariates = future_covariates.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with autocast_context(device, amp_dtype=amp_dtype):
            prediction = forward_model(model, features, ticker_id, future_covariates)
            assert_prediction_matches_criterion(
                prediction,
                criterion,
                model_role=model_role,
                phase=phase_label,
                run_id=run_id,
                epoch=epoch,
                batch_index=batch_index,
            )
            losses = criterion(prediction, line_target, band_target, raw_future_returns)

        loss_dict = losses.to_log_dict()
        component_finite = all(math.isfinite(value) for value in loss_dict.values())
        if not torch.isfinite(losses.total) or not component_finite:
            nan_streak += 1
            if not diagnostic_dumped:
                _dump_nonfinite_diagnostics(
                    phase=phase_label,
                    epoch=epoch,
                    batch_index=batch_index,
                    debug_label=debug_label,
                    tensors={
                        "features": features,
                        "line_target": line_target,
                        "band_target": band_target,
                        "raw_future_returns": raw_future_returns,
                        "prediction.line": getattr(prediction, "line", None),
                        "prediction.lower_band": getattr(prediction, "lower_band", None),
                        "prediction.upper_band": getattr(prediction, "upper_band", None),
                        "prediction.direction_logit": getattr(prediction, "direction_logit", None),
                        "prediction.downside_risk_logit": getattr(prediction, "downside_risk_logit", None),
                        "prediction.regime_logits": getattr(prediction, "regime_logits", None),
                    },
                    losses_dict=loss_dict,
                )
                diagnostic_dumped = True
            failed_metric = next(
                (key for key, value in loss_dict.items() if isinstance(value, float) and not math.isfinite(value)),
                "total_loss",
            )
            failed_value = loss_dict.get(failed_metric, float("nan"))
            result = FiniteCheckResult(
                ok=False,
                failed_metric=failed_metric,
                failed_value=failed_value,
                phase=phase_label,
                run_id=run_id,
                epoch=epoch,
                batch=batch_index,
            )
            message = result.format_message()
            print(message)
            if nan_streak >= NAN_STREAK_LIMIT or not is_train:
                raise RuntimeError(message)
            # train 단계에서 단발 NaN은 batch 스킵하고 계속 시도 (streak 한도까지).
            continue

        # 정상 batch이면 streak 리셋
        nan_streak = 0

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

        for key, value in loss_dict.items():
            totals.setdefault(key, 0.0)
            totals[key] += value
        batch_count += 1

    if batch_count == 0:
        # 모든 batch가 NaN이었거나 loader가 비어있다. NaN 통과를 막기 위해 즉시 실패.
        raise RuntimeError(
            f"[NaN-GATE phase={phase_label} run_id={run_id} epoch={epoch} "
            f"metric=batch_count value=0]: 정상 batch가 한 개도 없습니다."
        )

    averaged = {key: value / batch_count for key, value in totals.items()}
    averaged["grad_norm_mean"] = grad_norm_total / batch_count if is_train else 0.0
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
    q_low: float = 0.1,
    q_high: float = 0.9,
    severe_downside_threshold: float | None = None,
    squeeze_breakout_threshold: float | None = None,
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
        q_low=q_low,
        q_high=q_high,
        severe_downside_threshold=severe_downside_threshold,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
    )


def estimate_train_risk_thresholds(bundle: SequenceDatasetBundle | SequenceDataset) -> dict[str, float | None]:
    """train split raw return만 사용해 누수 없는 risk 기준값을 계산한다."""
    if isinstance(bundle, SequenceDatasetBundle):
        raw_values = bundle.raw_future_returns.detach().cpu().to(torch.float32).reshape(-1)
    else:
        chunks: list[np.ndarray] = []
        for ticker, end_idx in bundle.sample_refs:
            ticker_array = bundle.ticker_arrays[ticker]
            future_start = end_idx + 1
            future_end = future_start + bundle.horizon
            anchor_close = float(ticker_array["closes"][end_idx])
            future_returns = (ticker_array["closes"][future_start:future_end] / anchor_close) - 1.0
            chunks.append(np.asarray(future_returns, dtype="float32"))
        raw_values = torch.from_numpy(np.concatenate(chunks)).to(torch.float32) if chunks else torch.empty(0)

    finite = raw_values[torch.isfinite(raw_values)]
    if finite.numel() == 0:
        return {
            "severe_downside_threshold": None,
            "squeeze_breakout_threshold": None,
        }
    return {
        "severe_downside_threshold": float(torch.quantile(finite, 0.10).item()),
        "squeeze_breakout_threshold": float(torch.quantile(finite.abs(), 0.80).item()),
    }


def estimate_train_regime_thresholds(bundle: SequenceDatasetBundle | SequenceDataset) -> dict[str, Any]:
    """train split의 h5 raw_future_return 분위수로 line_regime class 경계를 고정한다."""
    if isinstance(bundle, SequenceDatasetBundle):
        raw_values = bundle.raw_future_returns.detach().cpu().to(torch.float32)
        h5_values = raw_values[:, -1] if raw_values.ndim == 2 else raw_values.reshape(-1)
    else:
        chunks: list[float] = []
        for ticker, end_idx in bundle.sample_refs:
            ticker_array = bundle.ticker_arrays[ticker]
            future_start = end_idx + 1
            future_end = future_start + bundle.horizon
            anchor_close = float(ticker_array["closes"][end_idx])
            future_returns = (ticker_array["closes"][future_start:future_end] / anchor_close) - 1.0
            if len(future_returns) > 0:
                chunks.append(float(future_returns[-1]))
        h5_values = torch.tensor(chunks, dtype=torch.float32) if chunks else torch.empty(0)

    finite = h5_values[torch.isfinite(h5_values)]
    if finite.numel() == 0:
        thresholds = (-0.05, -0.01, 0.01, 0.05)
    else:
        thresholds = tuple(float(torch.quantile(finite, q).item()) for q in (0.10, 0.35, 0.65, 0.90))
    return {
        "regime_thresholds": list(thresholds),
        "regime_threshold_q10": thresholds[0],
        "regime_threshold_q35": thresholds[1],
        "regime_threshold_q65": thresholds[2],
        "regime_threshold_q90": thresholds[3],
        "regime_threshold_source": "train_split_h5_quantile",
    }


def _find_first_nonfinite_tensor(
    named_tensors: dict[str, torch.Tensor | None],
) -> tuple[str | None, dict[str, dict[str, float | bool | int]]]:
    summary = tensor_finite_summary(named_tensors)
    for name, info in summary.items():
        finite_ratio = float(info.get("finite_ratio", 1.0))
        has_nan = bool(info.get("has_nan", False))
        has_inf = bool(info.get("has_inf", False))
        if finite_ratio < 1.0 or has_nan or has_inf:
            return name, summary
    return None, summary


def _dump_nonfinite_aggregate_diagnostics(
    *,
    phase: str,
    epoch: int,
    run_id: str,
    named_tensors: dict[str, torch.Tensor | None],
) -> str:
    first_name, summary = _find_first_nonfinite_tensor(named_tensors)
    print(f"[NaN-DIAG phase={phase} run_id={run_id} epoch={epoch} stage=aggregate]")
    for name, info in summary.items():
        print(f"  tensor[{name}] = {info}")
    return first_name or "unknown"


def evaluate_loader(
    *,
    model,
    loader: DataLoader,
    criterion: ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | BandOnlyLoss,
    device: torch.device,
    metadata: Any | None = None,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
    q_low: float = 0.1,
    q_high: float = 0.9,
    severe_downside_threshold: float | None = None,
    squeeze_breakout_threshold: float | None = None,
    risk_decision_threshold: float = 0.5,
    regime_thresholds: tuple[float, float, float, float] | list[float] | None = None,
    amp_dtype: str = "bf16",
    phase: str = "val",
    run_id: str = "",
    epoch: int = -1,
    model_role: str | None = None,
) -> dict[str, float]:
    model.eval()
    resolved_model_role = model_role or resolve_model_output_role(model)
    totals = {
        "total_loss": 0.0,
        "forecast_loss": 0.0,
        "line_loss": 0.0,
        "band_loss": 0.0,
        "cross_loss": 0.0,
        "direction_loss": 0.0,
        "risk_loss": 0.0,
        "regime_loss": 0.0,
    }
    batch_count = 0
    line_predictions: list[torch.Tensor] = []
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    raw_line_predictions: list[torch.Tensor] = []
    raw_lower_predictions: list[torch.Tensor] = []
    raw_upper_predictions: list[torch.Tensor] = []
    raw_direction_logits: list[torch.Tensor] = []
    raw_downside_risk_logits: list[torch.Tensor] = []
    raw_regime_logits: list[torch.Tensor] = []
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

            with autocast_context(device, amp_dtype=amp_dtype):
                prediction = forward_model(model, features, ticker_id, future_covariates)
                assert_prediction_matches_criterion(
                    prediction,
                    criterion,
                    model_role=resolved_model_role,
                    phase=phase,
                    run_id=run_id,
                    epoch=epoch,
                    batch_index=batch_count + 1,
                )
                losses = criterion(prediction, line_target, band_target, raw_future_returns)

            for key, value in losses.to_log_dict().items():
                totals.setdefault(key, 0.0)
                totals[key] += value
            batch_count += 1

            if isinstance(prediction, ForecastOutput):
                line_pred, lower, upper = apply_band_postprocess(
                    prediction.line.detach().cpu(),
                    prediction.lower_band.detach().cpu(),
                    prediction.upper_band.detach().cpu(),
                )
                raw_line_predictions.append(prediction.line.detach().cpu())
                raw_lower_predictions.append(prediction.lower_band.detach().cpu())
                raw_upper_predictions.append(prediction.upper_band.detach().cpu())
                if prediction.direction_logit is not None:
                    raw_direction_logits.append(prediction.direction_logit.detach().cpu())
                line_predictions.append(line_pred)
                lower_predictions.append(lower)
                upper_predictions.append(upper)
            elif isinstance(prediction, LineV2Output):
                raw_line_predictions.append(prediction.line.detach().cpu())
                raw_downside_risk_logits.append(prediction.downside_risk_logit.detach().cpu())
                line_predictions.append(prediction.line.detach().cpu())
            elif isinstance(prediction, LineRegimeOutput):
                raw_line_predictions.append(prediction.line.detach().cpu())
                raw_regime_logits.append(prediction.regime_logits.detach().cpu())
                line_predictions.append(prediction.line.detach().cpu())
            elif isinstance(prediction, BandOutput):
                lower = torch.minimum(prediction.lower_band.detach().cpu(), prediction.upper_band.detach().cpu())
                upper = torch.maximum(prediction.lower_band.detach().cpu(), prediction.upper_band.detach().cpu())
                raw_lower_predictions.append(prediction.lower_band.detach().cpu())
                raw_upper_predictions.append(prediction.upper_band.detach().cpu())
                lower_predictions.append(lower)
                upper_predictions.append(upper)
            line_targets.append(line_target.detach().cpu())
            band_targets.append(band_target.detach().cpu())
            raw_targets.append(raw_future_returns.detach().cpu())

    averaged = {key: value / max(batch_count, 1) for key, value in totals.items()}
    averaged["grad_norm_mean"] = 0.0
    if line_predictions or lower_predictions:
        aggregate_tensors = {
            "raw.line": torch.cat(raw_line_predictions, dim=0) if raw_line_predictions else None,
            "raw.lower_band": torch.cat(raw_lower_predictions, dim=0) if raw_lower_predictions else None,
            "raw.upper_band": torch.cat(raw_upper_predictions, dim=0) if raw_upper_predictions else None,
            "raw.direction_logit": torch.cat(raw_direction_logits, dim=0) if raw_direction_logits else None,
            "raw.downside_risk_logit": torch.cat(raw_downside_risk_logits, dim=0) if raw_downside_risk_logits else None,
            "raw.regime_logits": torch.cat(raw_regime_logits, dim=0) if raw_regime_logits else None,
            "post.line": torch.cat(line_predictions, dim=0) if line_predictions else None,
            "post.lower_band": torch.cat(lower_predictions, dim=0) if lower_predictions else None,
            "post.upper_band": torch.cat(upper_predictions, dim=0) if upper_predictions else None,
            "post.band_width": (
                torch.cat(upper_predictions, dim=0) - torch.cat(lower_predictions, dim=0)
                if lower_predictions and upper_predictions
                else None
            ),
            "target.line": torch.cat(line_targets, dim=0),
            "target.band": torch.cat(band_targets, dim=0),
            "target.raw_future_returns": torch.cat(raw_targets, dim=0),
        }
        first_nonfinite, _ = _find_first_nonfinite_tensor(aggregate_tensors)
        if first_nonfinite is not None:
            tensor_name = _dump_nonfinite_aggregate_diagnostics(
                phase=phase,
                epoch=epoch,
                run_id=run_id,
                named_tensors=aggregate_tensors,
            )
            raise RuntimeError(
                f"[NaN-GATE phase={phase} run_id={run_id} epoch={epoch} metric=tensor:{tensor_name} value=nan]"
            )
    if line_predictions and lower_predictions:
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
                q_low=q_low,
                q_high=q_high,
                severe_downside_threshold=severe_downside_threshold,
                squeeze_breakout_threshold=squeeze_breakout_threshold,
            )
        )
    elif line_predictions and raw_downside_risk_logits:
        averaged.update(
            summarize_line_v2_metrics(
                metadata=metadata,
                line_predictions=torch.cat(line_predictions, dim=0),
                risk_logits=torch.cat(raw_downside_risk_logits, dim=0),
                line_targets=torch.cat(line_targets, dim=0),
                raw_future_returns=torch.cat(raw_targets, dim=0),
                line_target_type=line_target_type,
                severe_downside_threshold=severe_downside_threshold,
                risk_decision_threshold=risk_decision_threshold,
            )
        )
    elif line_predictions and raw_regime_logits:
        averaged.update(
            summarize_line_regime_metrics(
                metadata=metadata,
                line_predictions=torch.cat(line_predictions, dim=0),
                regime_logits=torch.cat(raw_regime_logits, dim=0),
                line_targets=torch.cat(line_targets, dim=0),
                raw_future_returns=torch.cat(raw_targets, dim=0),
                line_target_type=line_target_type,
                regime_thresholds=regime_thresholds,
            )
        )
    elif lower_predictions:
        averaged.update(
            summarize_band_metrics(
                lower_predictions=torch.cat(lower_predictions, dim=0),
                upper_predictions=torch.cat(upper_predictions, dim=0),
                band_targets=torch.cat(band_targets, dim=0),
                raw_future_returns=torch.cat(raw_targets, dim=0),
                q_low=q_low,
                q_high=q_high,
                squeeze_breakout_threshold=squeeze_breakout_threshold,
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
    q_low: float = 0.1,
    q_high: float = 0.9,
    severe_downside_threshold: float | None = None,
    squeeze_breakout_threshold: float | None = None,
    regime_thresholds: tuple[float, float, float, float] | list[float] | None = None,
    amp_dtype: str = "bf16",
    phase: str = "test",
    run_id: str = "",
    epoch: int = -1,
    criterion: ForecastCompositeLoss | LineV2Loss | LineRegimeLoss | BandOnlyLoss | None = None,
    model_role: str | None = None,
) -> dict[str, float]:
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=num_workers)
    resolved_model_role = model_role or resolve_model_output_role(model)
    if criterion is None:
        criterion = build_evaluation_criterion(
            model_role=resolved_model_role,
            q_low=q_low,
            q_high=q_high,
            band_mode=str(getattr(unwrap_model(model), "band_mode", "direct") or "direct"),
            severe_downside_threshold=severe_downside_threshold,
            regime_thresholds=regime_thresholds,
        )
    metrics = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        metadata=bundle.metadata,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        q_low=q_low,
        q_high=q_high,
        severe_downside_threshold=severe_downside_threshold,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        regime_thresholds=regime_thresholds,
        amp_dtype=amp_dtype,
        phase=phase,
        run_id=run_id,
        epoch=epoch,
        model_role=resolved_model_role,
    )
    result = {
        "total_loss": metrics.get("total_loss"),
        "forecast_loss": metrics.get("forecast_loss"),
        "direction_loss": metrics.get("direction_loss"),
        "risk_loss": metrics.get("risk_loss"),
        "regime_loss": metrics.get("regime_loss"),
        "line_loss": metrics.get("line_loss"),
        "band_loss": metrics.get("band_loss"),
        "cross_loss": metrics.get("cross_loss"),
        "coverage": metrics.get("coverage"),
        "lower_breach_rate": metrics.get("lower_breach_rate"),
        "upper_breach_rate": metrics.get("upper_breach_rate"),
        "avg_band_width": metrics.get("avg_band_width"),
        "direction_accuracy": metrics.get("direction_accuracy"),
        "mae": metrics.get("mae"),
        "smape": metrics.get("smape"),
        "mean_signed_error": metrics.get("mean_signed_error"),
        "overprediction_rate": metrics.get("overprediction_rate"),
        "mean_overprediction": metrics.get("mean_overprediction"),
        "underprediction_rate": metrics.get("underprediction_rate"),
        "mean_underprediction": metrics.get("mean_underprediction"),
        "downside_capture_rate": metrics.get("downside_capture_rate"),
        "severe_downside_recall": metrics.get("severe_downside_recall"),
        "false_safe_rate": metrics.get("false_safe_rate"),
        "conservative_bias": metrics.get("conservative_bias"),
        "upside_sacrifice": metrics.get("upside_sacrifice"),
        "spearman_ic": metrics.get("spearman_ic"),
        "top_k_long_spread": metrics.get("top_k_long_spread"),
        "top_k_short_spread": metrics.get("top_k_short_spread"),
        "long_short_spread": metrics.get("long_short_spread"),
        "fee_adjusted_return": metrics.get("fee_adjusted_return"),
        "fee_adjusted_sharpe": metrics.get("fee_adjusted_sharpe"),
        "fee_adjusted_turnover": metrics.get("fee_adjusted_turnover"),
    }
    for key, value in metrics.items():
        if key not in result:
            result[key] = value
    return result


def _load_wandb_api_key_from_dotenv(dotenv_path: Path | None = None) -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return False
    path = dotenv_path if dotenv_path is not None else PROJECT_ROOT / ".env"
    if path is None or not path.exists():
        return False
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "WANDB_API_KEY":
                continue
            secret = value.strip().strip("'\"")
            if secret:
                os.environ["WANDB_API_KEY"] = secret
                return True
    except OSError:
        return False
    return False


def _sanitize_wandb_message(message: Any) -> str:
    text = str(message)
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(r"(WANDB_API_KEY\s*=\s*)[^\s]+", r"\1[redacted]", text)
    return text[:500]


def _wandb_required_error(status: str, message: str | None = None) -> RuntimeError:
    detail = f" ({message})" if message else ""
    return RuntimeError(
        "W&B required for sweep, set WANDB_API_KEY or use --no-wandb"
        f"; status={status}{detail}"
    )


def init_wandb_run(
    *,
    use_wandb: bool,
    project: str,
    config_payload: dict[str, Any],
    group: str | None = None,
    name: str | None = None,
    required: bool = False,
    dotenv_path: Path | None = None,
) -> WandbInitOutcome:
    if not use_wandb:
        return WandbInitOutcome(run=None, status=WANDB_STATUS_DISABLED_BY_CLI, message="--no-wandb")
    if wandb is None:
        outcome = WandbInitOutcome(run=None, status=WANDB_STATUS_PACKAGE_MISSING, message="wandb package is not installed")
        if required:
            raise _wandb_required_error(outcome.status, outcome.message)
        print(f"W&B 초기화 경고: {outcome.status}. 학습은 계속 진행합니다.")
        return outcome
    if os.environ.get("WANDB_MODE") == "disabled":
        outcome = WandbInitOutcome(run=None, status=WANDB_STATUS_DISABLED_BY_ENV, message="WANDB_MODE=disabled")
        if required:
            raise _wandb_required_error(outcome.status, outcome.message)
        return outcome

    _load_wandb_api_key_from_dotenv(dotenv_path)
    if not os.environ.get("WANDB_API_KEY"):
        outcome = WandbInitOutcome(
            run=None,
            status=WANDB_STATUS_DISABLED_MISSING_KEY,
            message="WANDB_API_KEY is not set",
        )
        if required:
            raise _wandb_required_error(outcome.status, outcome.message)
        print(f"W&B 초기화 경고: {outcome.status}. 학습은 계속 진행합니다.")
        return outcome

    try:
        run = wandb.init(
            project=project,
            config=config_payload,
            group=group,
            name=name,
            reinit=True,
        )
    except Exception as exc:
        message = _sanitize_wandb_message(exc)
        outcome = WandbInitOutcome(
            run=None,
            status=WANDB_STATUS_DISABLED_INIT_FAILED,
            message=message,
        )
        if required:
            raise _wandb_required_error(outcome.status, outcome.message) from exc
        print(f"W&B 초기화 경고: {outcome.status}: {message}. 학습은 계속 진행합니다.")
        return outcome

    return WandbInitOutcome(
        run=run,
        status=WANDB_STATUS_ONLINE_OK,
        run_id=getattr(run, "id", None),
        run_url=getattr(run, "url", None),
    )


def init_wandb_for_training(
    config: TrainConfig,
    run_id: str,
    *,
    group: str | None = None,
    name: str | None = None,
    config_override: dict[str, Any] | None = None,
    required: bool = False,
    dotenv_path: Path | None = None,
) -> WandbInitOutcome:
    return init_wandb_run(
        use_wandb=config.use_wandb,
        project=config.wandb_project,
        config_payload=config_override or asdict(config),
        group=group,
        name=name or f"{config.model}-{config.timeframe}-{run_id[:8]}",
        required=required,
        dotenv_path=dotenv_path,
    )


def maybe_init_wandb(
    config: TrainConfig,
    run_id: str,
    *,
    group: str | None = None,
    name: str | None = None,
    config_override: dict[str, Any] | None = None,
):
    return init_wandb_for_training(
        config,
        run_id,
        group=group,
        name=name,
        config_override=config_override,
    ).run


def emit_exit_marker(step: str, *, run_id: str = "", extras: dict[str, Any] | None = None) -> None:
    payload = {"step": step}
    if run_id:
        payload["run_id"] = run_id
    if extras:
        payload.update(extras)
    print(f"[EXIT-MARKER {json.dumps(payload, ensure_ascii=False)}]")


def register_cli_cuda_cleanup_state(**state: Any) -> None:
    global CLI_CUDA_CLEANUP_STATE
    CLI_CUDA_CLEANUP_STATE = state


def run_registered_cuda_cleanup(*, run_id: str = "") -> None:
    global CLI_CUDA_CLEANUP_STATE
    if CLI_CUDA_CLEANUP_STATE is None:
        return
    state = CLI_CUDA_CLEANUP_STATE
    try:
        torch.cuda.synchronize()
        emit_exit_marker("after_cuda_synchronize", run_id=run_id)
    finally:
        try:
            torch.cuda.empty_cache()
            emit_exit_marker("after_cuda_empty_cache", run_id=run_id)
        finally:
            state.clear()
            CLI_CUDA_CLEANUP_STATE = None
            emit_exit_marker("after_reference_release", run_id=run_id)
            gc_collected = gc.collect()
            emit_exit_marker("after_gc_collect", run_id=run_id, extras={"collected": gc_collected})


def _first_nonfinite_in_dataset(bundle: SequenceDatasetBundle | SequenceDataset) -> dict[str, Any] | None:
    if isinstance(bundle, SequenceDatasetBundle):
        mask = ~torch.isfinite(bundle.features)
        if not bool(mask.any().item()):
            return None
        first = mask.nonzero(as_tuple=False)[0]
        sample_idx, seq_idx, feat_idx = [int(value) for value in first.tolist()]
        meta = bundle.metadata.iloc[sample_idx]
        return {
            "ticker": str(meta["ticker"]),
            "asof_date": str(meta["asof_date"]),
            "column_index": feat_idx,
            "seq_index": seq_idx,
            "sample_index": sample_idx,
        }

    for sample_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        window = bundle.ticker_arrays[ticker]["features"][end_idx - bundle.seq_len + 1 : end_idx + 1]
        mask = ~np.isfinite(window)
        if not bool(mask.any()):
            continue
        seq_idx, feat_idx = [int(value) for value in np.argwhere(mask)[0].tolist()]
        meta = bundle.metadata.iloc[sample_idx]
        return {
            "ticker": str(meta["ticker"]),
            "asof_date": str(meta["asof_date"]),
            "column_index": feat_idx,
            "seq_index": seq_idx,
            "sample_index": int(sample_idx),
        }
    return None


def assert_dataset_features_finite(name: str, bundle: SequenceDatasetBundle | SequenceDataset) -> None:
    failure = _first_nonfinite_in_dataset(bundle)
    if failure is None:
        return
    payload = {"feature_preflight": name, **failure}
    print(json.dumps(payload, ensure_ascii=False))
    raise RuntimeError(
        f"{name} feature tensor에 non-finite 값이 남아 있습니다: {json.dumps(payload, ensure_ascii=False)}"
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
    checkpoint_config = {
        **asdict(config),
        "feature_version": FEATURE_CONTRACT_VERSION,
        "output_role": getattr(state_model, "output_role", config.model_role),
        "severe_downside_threshold": metrics.get("severe_downside_threshold"),
        "squeeze_breakout_threshold": metrics.get("squeeze_breakout_threshold"),
        "risk_threshold_source": metrics.get("risk_threshold_source"),
        "regime_thresholds": metrics.get("regime_thresholds"),
        "regime_threshold_q10": metrics.get("regime_threshold_q10"),
        "regime_threshold_q35": metrics.get("regime_threshold_q35"),
        "regime_threshold_q65": metrics.get("regime_threshold_q65"),
        "regime_threshold_q90": metrics.get("regime_threshold_q90"),
        "regime_threshold_source": metrics.get("regime_threshold_source"),
    }
    torch.save(
        {
            "model_state_dict": state_model.state_dict(),
            "config": checkpoint_config,
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
    wandb_required: bool = False,
    local_log: bool = True,
    local_log_dir: str | Path = DEFAULT_LOCAL_LOG_DIR,
) -> dict[str, Any]:
    set_seed(config.seed)
    apply_feature_set_to_config(config)
    device = resolve_device(config.device)
    run_id = f"{config.model}-{config.timeframe}-{uuid4().hex[:12]}"
    local_started_at = time.perf_counter()
    source_data_hash: str | None = None
    local_logger = LocalTrainingProgressLogger(
        run_id=run_id,
        base_dir=resolve_local_log_base_dir(local_log_dir),
        enabled=local_log,
    )
    local_logger.write_config(
        build_local_config_payload(
            run_id=run_id,
            config=config,
            local_log=local_log,
            local_log_dir=local_log_dir,
            run_log_dir=local_logger.run_dir,
            source_data_hash=source_data_hash,
        )
    )
    print(f"학습 디바이스: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(device)
    print(f"amp_dtype={config.amp_dtype} detect_anomaly={config.detect_anomaly}")
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if precomputed_bundles is not None:
        train_bundle, val_bundle, test_bundle, mean, std, plan = precomputed_bundles
    else:
        train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
            timeframe=config.timeframe,
            seq_len=config.seq_len,
            horizon=config.horizon,
            tickers=config.tickers,
            limit_tickers=config.limit_tickers,
            market_data_provider=config.market_data_provider,
            include_future_covariate=config.model == "tide" and config.use_future_covariate,
            line_target_type=config.line_target_type,
            band_target_type=config.band_target_type,
            split_mode=config.split_mode,
        )
    train_bundle, val_bundle, test_bundle, mean, std = apply_feature_columns_to_splits(
        train_bundle,
        val_bundle,
        test_bundle,
        mean,
        std,
        config.feature_columns or list(MODEL_FEATURE_COLUMNS),
    )
    config.num_tickers = plan.num_tickers
    config.ticker_registry_path = plan.ticker_registry_path
    source_data_hash = getattr(plan, "source_data_hash", None)
    local_logger.write_config(
        build_local_config_payload(
            run_id=run_id,
            config=config,
            local_log=local_log,
            local_log_dir=local_log_dir,
            run_log_dir=local_logger.run_dir,
            source_data_hash=source_data_hash,
            dataset_plan=summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle),
        )
    )
    assert_dataset_features_finite("train_bundle", train_bundle)
    assert_dataset_features_finite("val_bundle", val_bundle)
    assert_dataset_features_finite("test_bundle", test_bundle)
    risk_thresholds = estimate_train_risk_thresholds(train_bundle)
    severe_downside_threshold = risk_thresholds["severe_downside_threshold"]
    squeeze_breakout_threshold = risk_thresholds["squeeze_breakout_threshold"]
    regime_threshold_info = estimate_train_regime_thresholds(train_bundle)
    config.regime_thresholds = list(regime_threshold_info["regime_thresholds"])

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
    criterion = build_criterion(config, severe_downside_threshold=severe_downside_threshold)
    early_stopping = EarlyStopping(
        patience=0 if config.early_stop_patience < 0 else config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        mode="min",
    )
    checkpoint_selector = CheckpointSelector(config.checkpoint_selection)

    wandb_outcome = init_wandb_for_training(
        config,
        run_id,
        group=wandb_group,
        name=wandb_name,
        config_override=wandb_config_override,
        required=wandb_required,
    )
    wandb_run = wandb_outcome.run
    wandb_status = wandb_outcome.to_meta()
    wandb_run_id = wandb_status["run_id"] if wandb_status["status"] == WANDB_STATUS_ONLINE_OK else None
    grad_norm_history: list[float] = []
    failure_state: FiniteCheckResult | None = None
    started_at = time.perf_counter()

    def _write_local_summary(
        *,
        status: str,
        checkpoint_path: str | None = None,
        best_metrics: dict[str, Any] | None = None,
        test_metrics: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        local_logger.write_summary(
            build_local_summary_payload(
                run_id=run_id,
                status=status,
                config=config,
                checkpoint_path=checkpoint_path,
                best_metrics=best_metrics,
                test_metrics=test_metrics,
                wandb_status=wandb_status,
                total_elapsed_seconds=time.perf_counter() - local_started_at,
                source_data_hash=source_data_hash,
                error=error,
            )
        )

    def _record_failure(result: FiniteCheckResult) -> None:
        nonlocal failure_state
        failure_state = result
        print(result.format_message())

    def _persist_failed_run(result: FiniteCheckResult) -> None:
        """NaN으로 실패한 run의 메타를 model_runs에 status='failed_nan'로 남긴다.
        결과 테이블 (predictions/prediction_evaluations/backtest_results) 및 checkpoint 저장은 차단."""
        if not save_run:
            return
        try:
            metadata = train_bundle.metadata.sort_values("asof_date") if hasattr(train_bundle, "metadata") else None
            train_start = metadata["asof_date"].min() if metadata is not None else None
            train_end = metadata["asof_date"].max() if metadata is not None else None
        except Exception:
            train_start = None
            train_end = None
        config_hash = build_config_hash(config)
        save_model_run(
            {
                "run_id": run_id,
                "wandb_run_id": wandb_run_id,
                "model_name": config.model,
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "feature_version": FEATURE_CONTRACT_VERSION,
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
                "train_start": train_start,
                "train_end": train_end,
                "val_metrics": {},
                "test_metrics": {},
                "config": {
                    **asdict(config),
                    "config_hash": config_hash,
                    "feature_version": FEATURE_CONTRACT_VERSION,
                    "wandb_status": wandb_status,
                    "failure": result.to_meta(),
                },
                "checkpoint_path": None,
                "status": RUN_STATUS_FAILED_NAN,
            }
        )

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        try:
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
                amp_dtype=config.amp_dtype,
                run_id=run_id,
            )
        except RuntimeError as exc:
            result = FiniteCheckResult(
                ok=False,
                failed_metric="train_epoch",
                failed_value=None,
                phase="train",
                run_id=run_id,
                epoch=epoch,
                batch=-1,
                extras={"exception": str(exc)},
            )
            _record_failure(result)
            if wandb_run is not None:
                wandb_run.finish()
            _persist_failed_run(result)
            _write_local_summary(status=RUN_STATUS_FAILED_NAN, error=result.to_meta())
            raise
        val_summary = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metadata=val_bundle.metadata,
            line_target_type=config.line_target_type,
            band_target_type=config.band_target_type,
            q_low=config.q_low,
            q_high=config.q_high,
            severe_downside_threshold=severe_downside_threshold,
            squeeze_breakout_threshold=squeeze_breakout_threshold,
            risk_decision_threshold=config.risk_decision_threshold,
            regime_thresholds=config.regime_thresholds,
            amp_dtype=config.amp_dtype,
            phase="val",
            run_id=run_id,
            epoch=epoch,
            model_role=config.model_role,
        )

        # CP12 finite gate (validation)
        val_check = check_metrics_finite(val_summary, phase="val", run_id=run_id, epoch=epoch)
        if not val_check.ok:
            _record_failure(val_check)
            if wandb_run is not None:
                wandb_run.finish()
            _persist_failed_run(val_check)
            _write_local_summary(status=RUN_STATUS_FAILED_NAN, error=val_check.to_meta())
            raise RuntimeError(val_check.format_message())
        grad_norm_history.append(float(train_metrics.get("grad_norm_mean", 0.0)))

        val_total = val_summary["total_loss"]
        val_forecast = val_summary["forecast_loss"]
        stop_triggered = early_stopping.step(val_forecast, epoch, unwrap_model(model))
        early_state = early_stopping.snapshot()

        combined_summary = {
            **train_metrics,
            **val_summary,
        }
        checkpoint_selector.update(
            epoch=epoch,
            metrics=combined_summary,
            model=unwrap_model(model),
        )

        elapsed_total = time.perf_counter() - started_at
        epoch_seconds = time.perf_counter() - epoch_started_at
        remaining_epochs = max(config.epochs - epoch, 0)
        estimated_remaining_seconds = epoch_seconds * remaining_epochs
        vram_peak_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            if device.type == "cuda"
            else 0.0
        )
        summary = {
            "epoch": epoch,
            "epoch_seconds": round(epoch_seconds, 4),
            "elapsed_seconds": round(elapsed_total, 4),
            "estimated_remaining_seconds": round(estimated_remaining_seconds, 4),
            "vram_peak_allocated_mb": vram_peak_mb,
            "val_total": val_total,
            "val_forecast": val_forecast,
            "best_so_far": early_state.best_value,
            "epochs_since_improve": early_state.epochs_since_improve,
            "lr": current_lr(optimizer),
            "grad_norm_mean": train_metrics.get("grad_norm_mean", 0.0),
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_summary.items()},
        }
        local_logger.write_epoch(
            build_local_epoch_log_payload(
                run_id=run_id,
                epoch=epoch,
                epochs=config.epochs,
                elapsed_seconds=elapsed_total,
                epoch_seconds=epoch_seconds,
                estimated_remaining_seconds=estimated_remaining_seconds,
                learning_rate=current_lr(optimizer),
                train_metrics=train_metrics,
                val_metrics=val_summary,
                early_state=early_state,
                checkpoint_selector=checkpoint_selector,
                checkpoint_selection=config.checkpoint_selection,
                vram_peak_allocated_mb=vram_peak_mb,
            )
        )
        print(json.dumps(summary, ensure_ascii=False))
        print(
            f"val_total={val_total:.6f} val_forecast={val_forecast:.6f} best_so_far={early_state.best_value:.6f} "
            f"epochs_since_improve={early_state.epochs_since_improve} "
            f"lr={current_lr(optimizer):.8f} grad_norm_mean={train_metrics.get('grad_norm_mean', 0.0):.6f} "
            f"epoch_seconds={epoch_seconds:.2f} elapsed_seconds={elapsed_total:.2f} "
            f"eta_seconds={estimated_remaining_seconds:.2f} vram_peak_allocated_mb={vram_peak_mb:.2f}"
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
                    emit_exit_marker("after_wandb_finish", run_id=run_id, extras={"reason": "trial_pruned"})
                if optuna is None:
                    raise RuntimeError("Optuna가 필요하지만 설치되지 않았습니다.")
                raise optuna.TrialPruned()

        if stop_triggered:
            print(
                "EarlyStopping triggered at epoch "
                f"{epoch} (best=epoch {early_state.best_epoch}, val={early_state.best_value:.6f})"
            )
            break

    selection_result = checkpoint_selector.select()
    unwrap_model(model).load_state_dict(selection_result.candidate.state_dict)
    selected_summary = selection_result.candidate.metrics
    checkpoint_metrics = {
        **selected_summary,
        "best_epoch": selection_result.candidate.epoch,
        "best_val_total": selected_summary.get("forecast_loss"),
        "best_val_forecast_loss": selected_summary.get("forecast_loss"),
        "best_val_total_including_direction": selected_summary.get("total_loss"),
        "early_stopped": bool(early_stopping.should_stop),
        "grad_norm_history": grad_norm_history,
        "best_grad_norm_mean": selected_summary.get("grad_norm_mean", 0.0),
        "checkpoint_selection": selection_result.checkpoint_selection,
        "selected_epoch": selection_result.candidate.epoch,
        "selected_reason": selection_result.selected_reason,
        "gate_type": selection_result.gate_type,
        "gate_failed": selection_result.gate_failed,
        "line_gate_pass": selection_result.line_gate_pass,
        "band_gate_pass": selection_result.band_gate_pass,
        "combined_gate_pass": selection_result.combined_gate_pass,
        "role": checkpoint_role_for_gate(selection_result.gate_type),
        "model_role": config.model_role,
        "output_role": resolve_model_output_role(model),
        "coverage_gate_failed": selection_result.coverage_gate_failed,
        "selected_coverage": selected_summary.get("coverage"),
        "selected_upper_breach_rate": selected_summary.get("upper_breach_rate"),
        "selected_lower_breach_rate": selected_summary.get("lower_breach_rate"),
        "selected_spearman_ic": selected_summary.get("spearman_ic"),
        "selected_long_short_spread": selected_summary.get("long_short_spread"),
        "selected_fee_adjusted_return": selected_summary.get("fee_adjusted_return"),
        "best_val_total_epoch": selection_result.best_val_total_epoch,
        "legacy_best_val_total": early_stopping.best_value,
        "severe_downside_threshold": severe_downside_threshold,
        "squeeze_breakout_threshold": squeeze_breakout_threshold,
        "risk_threshold_source": "train_split_quantile",
        **regime_threshold_info,
        **dataset_plan_split_metadata(plan),
    }
    print(
        json.dumps(
            {
                "checkpoint_selection": checkpoint_metrics["checkpoint_selection"],
                "selected_epoch": checkpoint_metrics["selected_epoch"],
                "selected_reason": checkpoint_metrics["selected_reason"],
                "gate_type": checkpoint_metrics["gate_type"],
                "gate_failed": checkpoint_metrics["gate_failed"],
                "line_gate_pass": checkpoint_metrics["line_gate_pass"],
                "band_gate_pass": checkpoint_metrics["band_gate_pass"],
                "combined_gate_pass": checkpoint_metrics["combined_gate_pass"],
                "coverage_gate_failed": checkpoint_metrics["coverage_gate_failed"],
                "selected_coverage": checkpoint_metrics["selected_coverage"],
                "selected_upper_breach_rate": checkpoint_metrics["selected_upper_breach_rate"],
                "selected_lower_breach_rate": checkpoint_metrics["selected_lower_breach_rate"],
                "selected_spearman_ic": checkpoint_metrics["selected_spearman_ic"],
                "selected_long_short_spread": checkpoint_metrics["selected_long_short_spread"],
                "selected_fee_adjusted_return": checkpoint_metrics["selected_fee_adjusted_return"],
                "best_val_total_epoch": checkpoint_metrics["best_val_total_epoch"],
            },
            ensure_ascii=False,
        )
    )

    # CP12 finite gate (checkpoint metrics 저장 직전)
    cp_check = check_metrics_finite(checkpoint_metrics, phase="checkpoint", run_id=run_id)
    if not cp_check.ok:
        _record_failure(cp_check)
        if wandb_run is not None:
            wandb_run.finish()
        _persist_failed_run(cp_check)
        _write_local_summary(
            status=RUN_STATUS_FAILED_NAN,
            best_metrics=checkpoint_metrics,
            error=cp_check.to_meta(),
        )
        raise RuntimeError(cp_check.format_message())

    checkpoint_path = save_checkpoint(model, config, run_id, checkpoint_metrics, mean, std)

    test_quality = evaluate_bundle(
        model,
        test_bundle,
        device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        line_target_type=config.line_target_type,
        band_target_type=config.band_target_type,
        q_low=config.q_low,
        q_high=config.q_high,
        severe_downside_threshold=severe_downside_threshold,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        regime_thresholds=config.regime_thresholds,
        amp_dtype=config.amp_dtype,
        phase="test",
        run_id=run_id,
        epoch=early_stopping.best_epoch or -1,
        criterion=criterion,
        model_role=config.model_role,
    )

    # CP12 finite gate (test_quality 저장 직전)
    test_check = check_metrics_finite(test_quality, phase="test", run_id=run_id)
    if not test_check.ok:
        _record_failure(test_check)
        if wandb_run is not None:
            wandb_run.finish()
        _persist_failed_run(test_check)
        _write_local_summary(
            status=RUN_STATUS_FAILED_NAN,
            checkpoint_path=str(checkpoint_path),
            best_metrics=checkpoint_metrics,
            error=test_check.to_meta(),
        )
        raise RuntimeError(test_check.format_message())

    persisted_status = resolve_persisted_run_status(gate_failed=selection_result.gate_failed)
    result = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "local_log_dir": str(local_logger.run_dir) if local_log else None,
        "feature_set": config.feature_set,
        "feature_columns": config.feature_columns,
        "n_features": config.n_features,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "best_metrics": checkpoint_metrics,
        "test_metrics": test_quality,
        "dataset_plan": summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle),
        "wandb_status": wandb_status,
        "wandb_run_id": wandb_run_id,
        "wandb_run_url": wandb_status.get("run_url") if wandb_status["status"] == WANDB_STATUS_ONLINE_OK else None,
    }
    _write_local_summary(
        status=persisted_status,
        checkpoint_path=str(checkpoint_path),
        best_metrics=checkpoint_metrics,
        test_metrics=test_quality,
    )

    if wandb_run is not None:
        wandb.log({f"test/{key}": value for key, value in test_quality.items()})
        wandb_run.finish()
        emit_exit_marker("after_wandb_finish", run_id=run_id, extras={"reason": "completed"})

    if save_run:
        metadata = train_bundle.metadata.sort_values("asof_date")
        config_hash = build_config_hash(config)
        save_model_run(
            {
                "run_id": run_id,
                "wandb_run_id": wandb_run_id,
                "model_name": config.model,
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "feature_version": FEATURE_CONTRACT_VERSION,
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
                    "feature_version": FEATURE_CONTRACT_VERSION,
                    "wandb_status": wandb_status,
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                    "grad_norm_history": grad_norm_history,
                    "role": checkpoint_metrics.get("role"),
                    "severe_downside_threshold": severe_downside_threshold,
                    "squeeze_breakout_threshold": squeeze_breakout_threshold,
                    "risk_threshold_source": "train_split_quantile",
                    "dataset_split": dataset_plan_split_metadata(plan),
                    **regime_threshold_info,
                    "best_val_loss": checkpoint_metrics.get("best_val_total"),
                    "best_epoch": checkpoint_metrics.get("best_epoch"),
                    "best_val_total": checkpoint_metrics.get("best_val_total"),
                    "early_stopped": checkpoint_metrics.get("early_stopped"),
                },
                "checkpoint_path": str(checkpoint_path),
                "status": persisted_status,
            }
        )

    if device.type == "cuda" and (config.explicit_cuda_cleanup or config.hard_exit_after_result):
        register_cli_cuda_cleanup_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            train_bundle=train_bundle,
            val_bundle=val_bundle,
            test_bundle=test_bundle,
        )
    emit_exit_marker("result_ready", run_id=run_id)

    return result


def train(
    config: TrainConfig,
    *,
    save_run: bool,
    local_log: bool = True,
    local_log_dir: str | Path = DEFAULT_LOCAL_LOG_DIR,
) -> dict[str, Any]:
    return run_training(config, save_run=save_run, local_log=local_log, local_log_dir=local_log_dir)


def summarize_dataset_plan(
    plan: DatasetPlan,
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
) -> dict[str, Any]:
    summary = {
        "timeframe": plan.timeframe,
        "seq_len": plan.seq_len,
        "horizon": plan.horizon,
        "h_max": plan.h_max,
        "min_fold_samples": plan.min_fold_samples,
        "provider": getattr(plan, "provider", None),
        "source": getattr(plan, "source", None),
        "source_data_hash": getattr(plan, "source_data_hash", None),
        "date_min": getattr(plan, "date_min", None),
        "date_max": getattr(plan, "date_max", None),
        "usable_row_count": getattr(plan, "usable_row_count", None),
        "estimated_usable_sample_count": getattr(plan, "estimated_usable_sample_count", None),
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
    summary.update(dataset_plan_split_metadata(plan))
    return summary


def summarize_plan_only(plan: DatasetPlan) -> dict[str, Any]:
    train_samples = sum(spec.train.count for spec in plan.split_specs.values())
    val_samples = sum(spec.val.count for spec in plan.split_specs.values())
    test_samples = sum(spec.test.count for spec in plan.split_specs.values())
    summary = {
        "timeframe": plan.timeframe,
        "seq_len": plan.seq_len,
        "horizon": plan.horizon,
        "h_max": plan.h_max,
        "min_fold_samples": plan.min_fold_samples,
        "provider": getattr(plan, "provider", None),
        "source": getattr(plan, "source", None),
        "source_data_hash": getattr(plan, "source_data_hash", None),
        "date_min": getattr(plan, "date_min", None),
        "date_max": getattr(plan, "date_max", None),
        "usable_row_count": getattr(plan, "usable_row_count", None),
        "estimated_usable_sample_count": getattr(plan, "estimated_usable_sample_count", None),
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
    summary.update(dataset_plan_split_metadata(plan))
    return summary


def run_dry(config: TrainConfig) -> dict[str, Any]:
    apply_feature_set_to_config(config)
    feature_index_frame = fetch_feature_index_frame(
        timeframe=config.timeframe,
        tickers=config.tickers,
        limit_tickers=config.limit_tickers,
        market_data_provider=config.market_data_provider,
    )
    plan = build_dataset_plan(
        feature_index_frame,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        market_data_provider=config.market_data_provider,
        split_mode=config.split_mode,
        source_data_hash=resolve_data_fingerprint(
            config.timeframe,
            tickers=config.tickers,
            limit_tickers=config.limit_tickers,
            market_data_provider=config.market_data_provider,
        ),
    )
    config.num_tickers = plan.num_tickers
    config.ticker_registry_path = plan.ticker_registry_path
    model = build_model(config)
    criterion = build_criterion(config)
    sample_input = torch.randn(4, config.seq_len, config.n_features)
    sample_ticker_modulo = max(int(config.num_tickers) + 1, 1)
    sample_ticker_ids = torch.arange(4, dtype=torch.long) % sample_ticker_modulo
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
        assert_prediction_matches_criterion(
            output,
            criterion,
            model_role=config.model_role,
            phase="dry_run",
            run_id="dry_run",
        )
        loss = criterion(output, sample_line_target, sample_band_target, sample_raw_future_returns)
        if isinstance(output, ForecastOutput):
            line_pp, lower_pp, upper_pp = apply_band_postprocess(output.line, output.lower_band, output.upper_band)
    summary = summarize_plan_only(plan)
    forward_smoke: dict[str, Any] = {
        "feature_set": config.feature_set,
        "n_features": config.n_features,
        "feature_columns": config.feature_columns,
        "model_role": config.model_role,
        "loss_total": float(loss.total.detach().cpu()),
        "ticker_id_shape": list(sample_ticker_ids.shape),
    }
    if isinstance(output, ForecastOutput):
        forward_smoke.update(
            {
                "output_type": "legacy",
                "line_shape": list(output.line.shape),
                "lower_shape": list(output.lower_band.shape),
                "upper_shape": list(output.upper_band.shape),
                "postprocess_lower_le_upper": bool(torch.all(lower_pp <= upper_pp).item()),
                "line_preserved": bool(torch.equal(line_pp, output.line)),
                "contains_nan_or_inf": bool(
                    not torch.isfinite(output.line).all()
                    or not torch.isfinite(output.lower_band).all()
                    or not torch.isfinite(output.upper_band).all()
                ),
            }
        )
    elif isinstance(output, LineV2Output):
        forward_smoke.update(
            {
                "output_type": "line_v2",
                "line_shape": list(output.line.shape),
                "downside_risk_logit_shape": list(output.downside_risk_logit.shape),
                "contains_band_output": False,
                "contains_nan_or_inf": bool(
                    not torch.isfinite(output.line).all()
                    or not torch.isfinite(output.downside_risk_logit).all()
                ),
            }
        )
    elif isinstance(output, LineRegimeOutput):
        forward_smoke.update(
            {
                "output_type": "line_regime",
                "line_shape": list(output.line.shape),
                "regime_logits_shape": list(output.regime_logits.shape),
                "contains_band_output": False,
                "contains_nan_or_inf": bool(
                    not torch.isfinite(output.line).all()
                    or not torch.isfinite(output.regime_logits).all()
                ),
            }
        )
    elif isinstance(output, BandOutput):
        lower_pp = torch.minimum(output.lower_band, output.upper_band)
        upper_pp = torch.maximum(output.lower_band, output.upper_band)
        forward_smoke.update(
            {
                "output_type": "band",
                "lower_shape": list(output.lower_band.shape),
                "upper_shape": list(output.upper_band.shape),
                "contains_line_output": False,
                "postprocess_lower_le_upper": bool(torch.all(lower_pp <= upper_pp).item()),
                "contains_nan_or_inf": bool(
                    not torch.isfinite(output.lower_band).all()
                    or not torch.isfinite(output.upper_band).all()
                ),
            }
        )
    summary["forward_smoke"] = forward_smoke
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
        lambda_risk=args.lambda_risk,
        lambda_regime=args.lambda_regime,
        lambda_ordinal=args.lambda_ordinal,
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
        fp32_modules=args.fp32_modules,
        use_wandb=bool(args.use_wandb and not args.dry_run),
        wandb_project=args.wandb_project,
        model_ver=args.model_ver,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        checkpoint_selection=args.checkpoint_selection,
        amp_dtype=args.amp_dtype,
        detect_anomaly=args.detect_anomaly,
        explicit_cuda_cleanup=args.explicit_cuda_cleanup,
        hard_exit_after_result=args.hard_exit_after_result,
        use_revin=args.use_revin,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_n_heads=args.patchtst_n_heads,
        patchtst_n_layers=args.patchtst_n_layers,
        feature_set=args.feature_set,
        market_data_provider=args.market_data_provider,
        split_mode=args.split_mode,
        lower_band_loss_weight=args.lower_band_loss_weight,
        upper_band_loss_weight=args.upper_band_loss_weight,
        model_role=args.model_role,
        risk_decision_threshold=args.risk_decision_threshold,
    )
    is_cuda_run = str(config.device).lower().startswith("cuda")
    if args.dry_run:
        result = run_dry(config)
    else:
        result = train(
            config,
            save_run=args.save_run,
            local_log=args.local_log,
            local_log_dir=args.local_log_dir,
        )
    emit_exit_marker("before_result_json", run_id=result.get("run_id", ""))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    emit_exit_marker("after_result_json", run_id=result.get("run_id", ""))
    if not args.dry_run and is_cuda_run and config.hard_exit_after_result:
        if config.explicit_cuda_cleanup:
            run_registered_cuda_cleanup(run_id=result.get("run_id", ""))
            emit_exit_marker("after_explicit_cleanup_return", run_id=result.get("run_id", ""))
        emit_exit_marker("before_hard_exit", run_id=result.get("run_id", ""))
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    emit_exit_marker("after_stdio_flush", run_id=result.get("run_id", ""))
    if not args.dry_run and is_cuda_run and config.explicit_cuda_cleanup:
        run_registered_cuda_cleanup(run_id=result.get("run_id", ""))
        emit_exit_marker("after_explicit_cleanup_return", run_id=result.get("run_id", ""))
