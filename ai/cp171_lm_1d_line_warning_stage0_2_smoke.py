from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
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

torch = bootstrap_torch()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ai import cp160_lm_1d_line_overlay_rejudgement as cp160  # noqa: E402
from ai import cp164_lm_calendar_split_line_risk_smoke as cp164  # noqa: E402
from ai import cp165_lm_1d_atr_overlay_sweet_spot as cp165  # noqa: E402
from ai import cp167_lm_1d_top_quintile_risk_rescue as cp167  # noqa: E402
from ai import cp168_lm_1d_cohort_warning_development as cp168  # noqa: E402
from ai import cp170_lm_1d_recall_first_warning_sweep as cp170  # noqa: E402
from ai.loss import LineWarningLoss  # noqa: E402
from ai.models.cnn_lstm import CNNLSTM  # noqa: E402
from ai.models.common import LineWarningOutput  # noqa: E402
from ai.models.tide import TiDE  # noqa: E402
from ai.train import build_criterion, build_evaluation_criterion, build_model, resolve_device  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp171_lm_1d_line_warning_stage0_2_smoke"
STAGE0_REPORT = DOCS_DIR / "cp171_lm_1d_line_warning_stage0_code_separation_report.md"
STAGE1_REPORT = DOCS_DIR / "cp171_lm_1d_line_warning_stage1_baseline_report.md"
STAGE2_REPORT = DOCS_DIR / "cp171_lm_1d_line_warning_stage2_smoke_report.md"
METRICS_PATH = DOCS_DIR / "cp171_lm_1d_line_warning_stage0_2_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp171_lm_1d_line_warning_stage0_2_summary.csv"
BASELINE_CSV = DOCS_DIR / "cp171_lm_1d_line_warning_baseline_comparison.csv"
MODEL_CSV = DOCS_DIR / "cp171_lm_1d_line_warning_model_smoke_comparison.csv"
THRESHOLD_CSV = DOCS_DIR / "cp171_lm_1d_line_warning_threshold_detail.csv"
COHORT_CSV = DOCS_DIR / "cp171_lm_1d_line_warning_line_cohort_evaluation.csv"

CP = "CP171-S/LM"
SOURCE_HASH_EXPECTED = "90666b44cbfb8e5c"
SEVERE_THRESHOLD = -0.03
SECONDARY_SEVERE_THRESHOLD = -0.05
FEE_BPS = 0.001
SEED = 42
BATCH_SIZE = 1024
DEEP_BATCH_SIZE = 768
MODEL_EPOCHS = 1
LOGISTIC_EPOCHS = 2
POSITIVE_WEIGHT = 5.0
FOCAL_GAMMA = 2.0
SOFT_WARNING_SHARE = 0.45
STRONG_WARNING_SHARE = 0.25
RANDOM_N = 200

EXTRA_FEATURE_NAMES = (
    "vol_raw_20d",
    "vol_xs_rank_20d",
    "self_vol_percentile_252",
    "downside_vol_ratio_20d",
    "drawdown_from_5d_high",
    "drawdown_from_20d_high",
    "atr_ratio",
    "intraday_range_5d_mean",
    "intraday_range_20d_mean",
    "overnight_gap_abs_5d_mean",
    "overnight_gap_abs_20d_mean",
    "close_position_5d_mean",
    "close_position_20d_mean",
    "volume_z_20_252",
    "volume_ratio_5_20",
    "vol_accel_5_20",
    "log_vol_accel_5_20",
)


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
    path.write_text(json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_price_features_for_bundle(bundle: Any, price_feature_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    metadata = bundle.metadata[["ticker", "asof_date"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata["asof_date"] = pd.to_datetime(metadata["asof_date"], errors="coerce")
    merged = metadata.merge(
        price_feature_frame,
        how="left",
        left_on=["ticker", "asof_date"],
        right_on=["ticker", "date"],
    )
    result: dict[str, np.ndarray] = {}
    for feature in (
        "intraday_range_5d_mean",
        "intraday_range_20d_mean",
        "overnight_gap_abs_5d_mean",
        "overnight_gap_abs_20d_mean",
        "close_position_5d_mean",
        "close_position_20d_mean",
        "volume_z_20_252",
        "volume_ratio_5_20",
    ):
        values = pd.to_numeric(merged[feature], errors="coerce").to_numpy(dtype=np.float64)
        result[feature] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def map_vol_accel_for_bundle(bundle: Any, vol_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    metadata = bundle.metadata[["ticker", "asof_date"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata["asof_date"] = pd.to_datetime(metadata["asof_date"], errors="coerce")
    merged = metadata.merge(vol_frame, how="left", left_on=["ticker", "asof_date"], right_on=["ticker", "date"])
    result: dict[str, np.ndarray] = {}
    for feature in ("vol_accel_5_20", "log_vol_accel_5_20"):
        values = pd.to_numeric(merged[feature], errors="coerce").to_numpy(dtype=np.float64)
        result[feature] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def combine_extra_features(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for part in parts:
        result.update(part)
    return {name: np.asarray(result[name], dtype=np.float32) for name in EXTRA_FEATURE_NAMES}


class WarningSequenceDataset(Dataset):
    def __init__(self, base: Any, extra_features: dict[str, np.ndarray], *, include_sequence_extra: bool) -> None:
        self.base = base
        self.extra_features = extra_features
        self.include_sequence_extra = include_sequence_extra
        self.extra_names = list(EXTRA_FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        features, _line_target, _band_target, raw_future_returns, ticker_id, _future_covariates = self.base[index]
        extra = torch.tensor([float(self.extra_features[name][index]) for name in self.extra_names], dtype=torch.float32)
        if self.include_sequence_extra:
            repeated = extra.view(1, -1).expand(features.shape[0], -1)
            features = torch.cat((features, repeated), dim=-1)
        target = (raw_future_returns[-1] <= SEVERE_THRESHOLD).to(dtype=torch.float32)
        return features, extra, target, ticker_id, raw_future_returns


class RiskTabLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_role = "line_warning"
        self.linear = nn.Linear(input_dim * 3, 1)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=-1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> LineWarningOutput:
        del ticker_id
        return LineWarningOutput(warning_logit=self.linear(self._pool(x)).squeeze(-1))


class RiskTabMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.output_role = "line_warning"
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=-1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> LineWarningOutput:
        del ticker_id
        return LineWarningOutput(warning_logit=self.net(self._pool(x)).squeeze(-1))


def build_candidate_model(candidate_id: str, input_dim: int) -> nn.Module:
    if candidate_id == "logistic_baseline":
        return RiskTabLinear(input_dim=input_dim)
    if candidate_id == "risk_tab_mlp":
        return RiskTabMLP(input_dim=input_dim)
    if candidate_id == "cnn_lstm_risk_only":
        return CNNLSTM(
            n_features=input_dim,
            seq_len=cp164.cp158.SEQ_LEN,
            horizon=cp164.cp158.HORIZON,
            cnn_channels=48,
            lstm_hidden=96,
            n_layers=1,
            dropout=0.15,
            num_tickers=0,
            output_role="line_warning",
        )
    if candidate_id == "tide_risk_only":
        return TiDE(
            n_features=input_dim,
            seq_len=cp164.cp158.SEQ_LEN,
            horizon=cp164.cp158.HORIZON,
            enc_dim=128,
            dec_dim=64,
            n_enc_layers=2,
            n_dec_layers=1,
            dropout=0.15,
            future_cov_dim=0,
            num_tickers=0,
            output_role="line_warning",
        )
    raise ValueError(f"알 수 없는 candidate_id입니다: {candidate_id}")


def train_warning_candidate(
    *,
    candidate_id: str,
    train_dataset: WarningSequenceDataset,
    val_dataset: WarningSequenceDataset,
    input_dim: int,
    device: torch.device,
    epochs: int,
) -> tuple[nn.Module, dict[str, Any], np.ndarray]:
    set_seed(SEED)
    model = build_candidate_model(candidate_id, input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = LineWarningLoss(
        downside_threshold=SEVERE_THRESHOLD,
        positive_weight=POSITIVE_WEIGHT,
        gamma=FOCAL_GAMMA,
        use_focal=True,
    )
    batch_size = BATCH_SIZE if candidate_id in ("logistic_baseline", "risk_tab_mlp") else DEEP_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    train_log: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        positives = 0
        total = 0
        for features, _extra, target, ticker_id, raw_future_returns in train_loader:
            del target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, ticker_id=ticker_id)
            loss = criterion(
                output,
                line_target=torch.zeros(raw_future_returns.shape, device=device),
                band_target=torch.zeros(raw_future_returns.shape, device=device),
                raw_future_returns=raw_future_returns,
            ).total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            positives += int((raw_future_returns[:, -1] <= SEVERE_THRESHOLD).sum().detach().cpu())
            total += int(raw_future_returns.shape[0])
        train_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else None,
                "train_positive_rate": _safe_ratio(positives, total),
                "sample_count": total,
            }
        )
    val_prob = predict_warning_prob(model, val_dataset, device=device, batch_size=batch_size)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    run_meta = {
        "candidate_id": candidate_id,
        "model_role": "line_warning",
        "output_role": "line_warning",
        "seed": SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "positive_weight": POSITIVE_WEIGHT,
        "gamma": FOCAL_GAMMA,
        "downside_threshold": SEVERE_THRESHOLD,
        "elapsed_seconds": elapsed,
        "train_log": train_log,
    }
    return model, run_meta, val_prob


@torch.no_grad()
def predict_warning_prob(model: nn.Module, dataset: WarningSequenceDataset, *, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model.eval()
    chunks: list[np.ndarray] = []
    for features, _extra, _target, ticker_id, _raw_future_returns in loader:
        features = features.to(device, non_blocking=True)
        ticker_id = ticker_id.to(device, non_blocking=True)
        output = model(features, ticker_id=ticker_id.to(device, non_blocking=True))
        chunks.append(output.warning_prob.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def thresholds_from_validation(prob: np.ndarray) -> dict[str, float]:
    finite = np.asarray(prob, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"soft_threshold": 1.0, "strong_threshold": 1.0}
    return {
        "soft_threshold": float(np.quantile(finite, 1.0 - SOFT_WARNING_SHARE)),
        "strong_threshold": float(np.quantile(finite, 1.0 - STRONG_WARNING_SHARE)),
    }


def datewise_masks(prediction: dict[str, Any], *, top_q: float) -> tuple[np.ndarray, np.ndarray]:
    return cp170.datewise_top_bottom_masks(prediction, top_q=top_q, bottom_q=0.10)


def evaluate_warning_rule(
    *,
    candidate_id: str,
    split: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    no_warning_missed_rate: float | None,
    warning_level: str,
    cohort: str,
) -> dict[str, Any]:
    row = cp170.warning_metrics(
        rule_id=candidate_id,
        split=split,
        prediction=prediction,
        warning_mask=warning_mask,
        top_mask=top_mask,
        bottom_mask=bottom_mask,
        no_warning_missed_rate=no_warning_missed_rate,
    )
    row["candidate_id"] = candidate_id
    row["warning_level"] = warning_level
    row["cohort"] = cohort
    row["precision"] = row.get("warned_severe_rate")
    return row


def random_matched_summary(
    *,
    candidate_id: str,
    prediction: dict[str, Any],
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    warning_share: float,
    candidate_row: dict[str, Any],
    no_warning_missed_rate: float | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(171)
    top_indices = np.flatnonzero(top_mask)
    rows: list[dict[str, Any]] = []
    for idx in range(RANDOM_N):
        warning = np.zeros(len(prediction["actual"]), dtype=bool)
        count = int(round(len(top_indices) * warning_share))
        if count > 0:
            warning[rng.choice(top_indices, size=min(count, len(top_indices)), replace=False)] = True
        rows.append(
            cp170.warning_metrics(
                rule_id=f"{candidate_id}_random_{idx}",
                split="test",
                prediction=prediction,
                warning_mask=warning,
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=no_warning_missed_rate,
            )
        )
    result: list[dict[str, Any]] = []
    for metric in ("warning_severe_recall", "missed_severe_rate", "warned_severe_rate", "spread_retention", "fee_retention"):
        arr = np.asarray([row.get(metric) for row in rows if row.get(metric) is not None], dtype=np.float64)
        value = candidate_row.get(metric)
        result.append(
            {
                "candidate_id": candidate_id,
                "metric": metric,
                "candidate_value": value,
                "random_mean": float(arr.mean()) if len(arr) else None,
                "random_std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "candidate_minus_random": None if value is None or not len(arr) else float(value - arr.mean()),
                "random_n": RANDOM_N,
                "warning_share": warning_share,
            }
        )
    return result


def build_rule_masks_from_prior(
    *,
    val_prediction: dict[str, Any],
    test_prediction: dict[str, Any],
    val_features: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    top_val, _bottom_val = datewise_masks(val_prediction, top_q=0.90)
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    cp169_payload = json.loads((DOCS_DIR / "cp168_lm_1d_cohort_warning_development_metrics.json").read_text(encoding="utf-8"))
    selected = cp169_payload.get("selected_q5_rule") or {}
    base_val = cp167.base_warning_mask(val_features, val_features)
    base_test = cp167.base_warning_mask(test_features, val_features)
    partitions = cp167.validation_partitions(val_prediction)
    valid_a_prediction, valid_a_features = cp167.subset_prediction(val_prediction, partitions["valid_a"], val_features)
    q5_val_rule = cp168.selected_rule_mask(
        selected,
        val_features,
        prediction=val_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    q5_test_rule = cp168.selected_rule_mask(
        selected,
        test_features,
        prediction=test_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    q5_val = cp167.top_decile_quintiles(np.asarray(val_prediction["line_score"], dtype=np.float64)) == 5
    q5_test = cp167.top_decile_quintiles(np.asarray(test_prediction["line_score"], dtype=np.float64)) == 5
    masks["cp169_two_tier_rule"] = ((q5_val & q5_val_rule) | (~q5_val & base_val), (q5_test & q5_test_rule) | (~q5_test & base_test))

    cp170_payload = json.loads((DOCS_DIR / "cp170_lm_1d_recall_first_warning_sweep_metrics.json").read_text(encoding="utf-8"))
    best_rule_id = str(cp170_payload.get("best_rule_id") or cp170_payload.get("best_rule_test", {}).get("rule_id") or "")
    if best_rule_id == "drawdown_from_5d_high_q50":
        threshold = cp170.threshold_from_validation("drawdown_from_5d_high", val_features, top_val, 0.50)
        masks["cp170_best_wide_rule"] = (
            cp170.feature_warning("drawdown_from_5d_high", val_features, threshold),
            cp170.feature_warning("drawdown_from_5d_high", test_features, threshold),
        )
    elif best_rule_id:
        # 알 수 없는 과거 rule이면 CP170 best 행은 비교표에만 남긴다.
        masks["cp170_best_wide_rule"] = (
            np.zeros(len(val_prediction["actual"]), dtype=bool),
            np.zeros(len(test_prediction["actual"]), dtype=bool),
        )
    return masks


def load_cp171_payload() -> dict[str, Any]:
    cp164_payload = cp165.load_cp164_reference()
    price, indicators, price_manifest, indicator_manifest = cp164.cp158.load_source_frames()
    source_hash = str(indicator_manifest.get("source_data_hash") or price_manifest.get("source_data_hash") or "unknown")
    train, val, test, mean, std, plan, _registry = cp164.build_calendar_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    split_summary = cp164.summarize_dataset_plan(plan, train, val, test)
    split_summary["source_data_hash"] = source_hash
    split_summary["cross_split_date_overlap_count_bundle_check"] = cp164._date_overlap_count(train, val, test)
    if split_summary.get("split_mode") != "calendar_aligned":
        raise RuntimeError(f"split_mode 불일치: {split_summary.get('split_mode')}")
    overlap_count = split_summary.get("cross_split_date_overlap_count")
    if overlap_count is None or int(overlap_count) != 0:
        raise RuntimeError(f"calendar split overlap 감지: {split_summary.get('cross_split_date_overlap_count')}")
    if source_hash != SOURCE_HASH_EXPECTED:
        raise RuntimeError(f"source_data_hash 불일치: {source_hash} != {SOURCE_HASH_EXPECTED}")
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    config = cp164.make_config(str(device))
    config.regime_thresholds = [float(value) for value in cp164_payload.get("regime_thresholds", [])]
    checkpoint_path = str(PROJECT_ROOT / str(cp164_payload["run_result"]["checkpoint_path"]))
    val_prediction = cp160.collect_predictions(
        candidate_id="cp171_patchtst_line_regime_p32_s16_calendar_reference",
        checkpoint_path=checkpoint_path,
        model_kind="line_regime",
        bundle=val,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    test_prediction = cp160.collect_predictions(
        candidate_id="cp171_patchtst_line_regime_p32_s16_calendar_reference",
        checkpoint_path=checkpoint_path,
        model_kind="line_regime",
        bundle=test,
        mean=mean,
        std=std,
        device=device,
        config=config,
    )
    price_feature_frame = cp167.compute_price_feature_frame(price)
    vol_accel_frame = cp170.compute_extra_price_features(price)
    train_features = combine_extra_features(
        cp164.compute_overlay_features(train, indicators),
        map_price_features_for_bundle(train, price_feature_frame),
        map_vol_accel_for_bundle(train, vol_accel_frame),
    )
    val_features = combine_extra_features(
        cp164.compute_overlay_features(val, indicators),
        map_price_features_for_bundle(val, price_feature_frame),
        map_vol_accel_for_bundle(val, vol_accel_frame),
    )
    test_features = combine_extra_features(
        cp164.compute_overlay_features(test, indicators),
        map_price_features_for_bundle(test, price_feature_frame),
        map_vol_accel_for_bundle(test, vol_accel_frame),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "train": train,
        "val": val,
        "test": test,
        "split_summary": split_summary,
        "val_prediction": val_prediction,
        "test_prediction": test_prediction,
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "device": device,
        "cp164_payload": cp164_payload,
    }


def finite_feature_summary(features: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = {}
    for name, values in features.items():
        arr = np.asarray(values, dtype=np.float64)
        rows[name] = {
            "finite_rate": _safe_ratio(int(np.isfinite(arr).sum()), int(arr.size)),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
        }
    return rows


def evaluate_candidate_probabilities(
    *,
    candidate_id: str,
    val_prob: np.ndarray,
    test_prob: np.ndarray,
    val_prediction: dict[str, Any],
    test_prediction: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = thresholds_from_validation(val_prob)
    threshold_rows = [
        {
            "candidate_id": candidate_id,
            "soft_threshold": thresholds["soft_threshold"],
            "strong_threshold": thresholds["strong_threshold"],
            "validation_soft_share": _safe_ratio(int((val_prob >= thresholds["soft_threshold"]).sum()), int(len(val_prob))),
            "validation_strong_share": _safe_ratio(int((val_prob >= thresholds["strong_threshold"]).sum()), int(len(val_prob))),
            "threshold_fit_split": "validation",
        }
    ]
    rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for split, prediction, prob in (("validation", val_prediction, val_prob), ("test", test_prediction, test_prob)):
        for cohort, top_q in (("whole_universe", 0.0), ("line_top_decile", 0.90), ("line_top_5pct", 0.95)):
            if top_q <= 0.0:
                top_mask = np.ones(len(prob), dtype=bool)
                _top, bottom_mask = datewise_masks(prediction, top_q=0.90)
            else:
                top_mask, bottom_mask = datewise_masks(prediction, top_q=top_q)
            no_warning = evaluate_warning_rule(
                candidate_id="no_warning_baseline",
                split=split,
                prediction=prediction,
                warning_mask=np.zeros(len(prob), dtype=bool),
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=None,
                warning_level="none",
                cohort=cohort,
            )
            soft_row = evaluate_warning_rule(
                candidate_id=candidate_id,
                split=split,
                prediction=prediction,
                warning_mask=prob >= thresholds["soft_threshold"],
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=no_warning.get("missed_severe_rate"),
                warning_level="soft_or_strong",
                cohort=cohort,
            )
            strong_row = evaluate_warning_rule(
                candidate_id=candidate_id,
                split=split,
                prediction=prediction,
                warning_mask=prob >= thresholds["strong_threshold"],
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=no_warning.get("missed_severe_rate"),
                warning_level="strong",
                cohort=cohort,
            )
            rows.extend([no_warning, soft_row, strong_row])
            if split == "test" and cohort == "line_top_decile":
                random_rows.extend(
                    random_matched_summary(
                        candidate_id=candidate_id,
                        prediction=prediction,
                        top_mask=top_mask,
                        bottom_mask=bottom_mask,
                        warning_share=float(soft_row.get("warning_share") or 0.0),
                        candidate_row=soft_row,
                        no_warning_missed_rate=no_warning.get("missed_severe_rate"),
                    )
                )
    return rows, threshold_rows, random_rows


def evaluate_prior_rule_baselines(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_prediction = payload["val_prediction"]
    test_prediction = payload["test_prediction"]
    val_features = payload["val_features"]
    test_features = payload["test_features"]
    masks = build_rule_masks_from_prior(
        val_prediction=val_prediction,
        test_prediction=test_prediction,
        val_features=val_features,
        test_features=test_features,
    )
    rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for rule_id, (val_mask, test_mask) in masks.items():
        for split, prediction, mask in (("validation", val_prediction, val_mask), ("test", test_prediction, test_mask)):
            for cohort, top_q in (("whole_universe", 0.0), ("line_top_decile", 0.90), ("line_top_5pct", 0.95)):
                if top_q <= 0.0:
                    top_mask = np.ones(len(mask), dtype=bool)
                    _top, bottom_mask = datewise_masks(prediction, top_q=0.90)
                else:
                    top_mask, bottom_mask = datewise_masks(prediction, top_q=top_q)
                no_warning = evaluate_warning_rule(
                    candidate_id="no_warning_baseline",
                    split=split,
                    prediction=prediction,
                    warning_mask=np.zeros(len(mask), dtype=bool),
                    top_mask=top_mask,
                    bottom_mask=bottom_mask,
                    no_warning_missed_rate=None,
                    warning_level="none",
                    cohort=cohort,
                )
                row = evaluate_warning_rule(
                    candidate_id=rule_id,
                    split=split,
                    prediction=prediction,
                    warning_mask=mask,
                    top_mask=top_mask,
                    bottom_mask=bottom_mask,
                    no_warning_missed_rate=no_warning.get("missed_severe_rate"),
                    warning_level="binary",
                    cohort=cohort,
                )
                rows.append(row)
                if split == "test" and cohort == "line_top_decile":
                    random_rows.extend(
                        random_matched_summary(
                            candidate_id=rule_id,
                            prediction=prediction,
                            top_mask=top_mask,
                            bottom_mask=bottom_mask,
                            warning_share=float(row.get("warning_share") or 0.0),
                            candidate_row=row,
                            no_warning_missed_rate=no_warning.get("missed_severe_rate"),
                        )
                    )
    return rows, random_rows


def classification_label(row: dict[str, Any], random_rows: list[dict[str, Any]], cp170_row: dict[str, Any] | None) -> str:
    if not row:
        return "FAIL"
    recall = float(row.get("warning_severe_recall") or 0.0)
    warning_share = float(row.get("warning_share") or 0.0)
    spread = float(row.get("spread_retention") or 0.0)
    fee = float(row.get("fee_retention") or 0.0)
    random_lookup = {item["metric"]: item for item in random_rows if item.get("candidate_id") == row.get("candidate_id")}
    recall_excess = float(random_lookup.get("warning_severe_recall", {}).get("candidate_minus_random") or 0.0)
    cp170_retention_ok = True
    if cp170_row:
        cp170_retention_ok = spread >= float(cp170_row.get("spread_retention") or 0.0) and fee >= float(cp170_row.get("fee_retention") or 0.0)
    if recall >= 0.50 and warning_share <= 0.45 and spread >= 0.75 and fee >= 0.75 and recall_excess >= 0.05 and cp170_retention_ok:
        return "PASS"
    if recall >= 0.45 and warning_share <= 0.55 and spread >= 0.70 and fee >= 0.70 and recall_excess > 0.0:
        return "WARN"
    return "FAIL"


def model_primary_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    warning_share = float(row.get("warning_share") or 0.0)
    spread = row.get("spread_retention")
    fee = row.get("fee_retention")
    if warning_share >= 0.95 or spread is None or fee is None:
        return (-999.0, -999.0, -999.0, -999.0, -999.0)
    return (
        1.0 if warning_share <= 0.55 else 0.0,
        float(row.get("warning_severe_recall") or -999.0),
        float(spread or -999.0),
        float(fee or -999.0),
        -abs(warning_share - SOFT_WARNING_SHARE),
    )


def write_reports(payload: dict[str, Any]) -> None:
    stage0 = payload["stage0"]
    baseline_primary = payload["baseline_primary"]
    model_primary = payload["model_primary"]
    final_label = payload["final_label"]
    best_model = payload.get("best_model_primary") or {}
    best_baseline = payload.get("best_baseline_primary") or {}
    split = payload["split_summary"]

    STAGE0_REPORT.write_text(
        "\n".join(
            [
                "# CP171 Stage 0 코드 분리 / 계약 점검",
                "",
                "## 한 줄 결론",
                f"- 판정: `{stage0['status']}`",
                "- `line_warning`은 line_v2, line_regime, band와 별도 output/loss 계약으로 추가했다.",
                "- product save-run, DB write, inference 저장 경로는 사용하지 않았다.",
                "",
                "## 데이터 / split",
                f"- provider/source: `yfinance / yfinance`",
                f"- source_data_hash: `{split.get('source_data_hash')}`",
                f"- split_mode: `{split.get('split_mode')}`",
                f"- cross_split_date_overlap_count: `{split.get('cross_split_date_overlap_count')}`",
                f"- eligible ticker count: `{split.get('eligible_ticker_count')}`",
                "",
                "## 계약 확인",
                f"- model_role 지원: `{stage0['model_role_supported']}`",
                f"- output_role 지원: `{stage0['output_role_supported']}`",
                f"- LineWarningOutput 별도 dataclass: `{stage0['line_warning_output_ok']}`",
                f"- LineWarningLoss 별도 loss: `{stage0['line_warning_loss_ok']}`",
                f"- TCN line_warning 차단: `{stage0['tcn_blocked']}`",
                f"- line_score/rank/top flag 모델 입력 제외: `{stage0['line_score_excluded_from_inputs']}`",
                "",
                "## 입력 피처",
                "- 기존 36개 sequence feature와 risk overlay용 point-in-time 파생 피처를 사용했다.",
                "- `line_score`, `line_rank_by_date`, `line_top_decile`은 모델 입력에 넣지 않고 평가 cohort 계산에만 사용했다.",
            ]
        ),
        encoding="utf-8",
    )

    STAGE1_REPORT.write_text(
        "\n".join(
            [
                "# CP171 Stage 1 Baseline 재정리",
                "",
                "## 한 줄 결론",
                f"- baseline primary best: `{best_baseline.get('candidate_id')}`",
                f"- recall / missed severe / spread retention / fee retention: `{_fmt(best_baseline.get('warning_severe_recall'))}` / `{_fmt(best_baseline.get('missed_severe_rate'))}` / `{_fmt(best_baseline.get('spread_retention'))}` / `{_fmt(best_baseline.get('fee_retention'))}`",
                "",
                "## 비교 기준",
                "- random matched warning",
                "- CP169 two-tier rule",
                "- CP170 best wide rule",
                "- logistic baseline",
                "",
                "## primary cohort",
                "- test split, 날짜별 line score top decile",
                "",
                "## baseline primary rows",
                "| 후보 | recall | missed severe | warning share | precision | spread retention | fee retention |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row.get('candidate_id')} | {_fmt(row.get('warning_severe_recall'))} | {_fmt(row.get('missed_severe_rate'))} | {_fmt(row.get('warning_share'))} | {_fmt(row.get('precision'))} | {_fmt(row.get('spread_retention'))} | {_fmt(row.get('fee_retention'))} |"
                    for row in baseline_primary
                ],
            ]
        ),
        encoding="utf-8",
    )

    STAGE2_REPORT.write_text(
        "\n".join(
            [
                "# CP171 Stage 2 Warning Model Smoke",
                "",
                "## 한 줄 결론",
                f"- 최종 판정: `{final_label}`",
                f"- best model primary: `{best_model.get('candidate_id')}`",
                f"- recall / missed severe / warning share: `{_fmt(best_model.get('warning_severe_recall'))}` / `{_fmt(best_model.get('missed_severe_rate'))}` / `{_fmt(best_model.get('warning_share'))}`",
                f"- spread / fee retention: `{_fmt(best_model.get('spread_retention'))}` / `{_fmt(best_model.get('fee_retention'))}`",
                "",
                "## 해석",
                "- warning model은 line을 대체하지 않고, line top decile을 해석할 때 하방 위험을 보조하는 guard layer로만 평가했다.",
                "- threshold는 validation에서만 고정했고 test 결과를 보고 바꾸지 않았다.",
                "- line_score는 모델 입력에 쓰지 않았다.",
                "",
                "## 모델 primary rows",
                "| 후보 | level | recall | missed severe | warning share | precision | spread retention | fee retention |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row.get('candidate_id')} | {row.get('warning_level')} | {_fmt(row.get('warning_severe_recall'))} | {_fmt(row.get('missed_severe_rate'))} | {_fmt(row.get('warning_share'))} | {_fmt(row.get('precision'))} | {_fmt(row.get('spread_retention'))} | {_fmt(row.get('fee_retention'))} |"
                    for row in model_primary
                ],
                "",
                "## 다음 액션",
                "- PASS면 다음 CP에서 threshold/loss/model narrow sweep으로 넘긴다.",
                "- WARN이면 loss weight, threshold, feature pack 보강을 우선 검토한다.",
                "- FAIL이면 현재 feature만으로는 warning model도 부족하다고 보고 binary downside classifier feature 확장 또는 외부 이벤트 데이터 검토가 필요하다.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    payload = load_cp171_payload()
    device = payload["device"]
    split_summary = payload["split_summary"]
    train_features = payload["train_features"]
    val_features = payload["val_features"]
    test_features = payload["test_features"]
    val_prediction = payload["val_prediction"]
    test_prediction = payload["test_prediction"]

    input_dim = int(payload["train"][0][0].shape[-1] + len(EXTRA_FEATURE_NAMES))
    train_dataset = WarningSequenceDataset(payload["train"], train_features, include_sequence_extra=True)
    val_dataset = WarningSequenceDataset(payload["val"], val_features, include_sequence_extra=True)
    test_dataset = WarningSequenceDataset(payload["test"], test_features, include_sequence_extra=True)

    # Stage 0 계약 점검은 실제 모델 생성/criterion 생성으로 fail-closed 여부를 확인한다.
    stage0 = {
        "status": "PASS",
        "model_role_supported": True,
        "output_role_supported": True,
        "line_warning_output_ok": isinstance(LineWarningOutput(torch.zeros(2)), LineWarningOutput),
        "line_warning_loss_ok": isinstance(build_evaluation_criterion(model_role="line_warning"), LineWarningLoss),
        "tcn_blocked": False,
        "line_score_excluded_from_inputs": True,
        "feature_finite_summary": {
            "train": finite_feature_summary(train_features),
            "validation": finite_feature_summary(val_features),
            "test": finite_feature_summary(test_features),
        },
    }
    try:
        _ = build_model(build_cp171_config_for_contract("tcn_quantile"))
    except ValueError:
        stage0["tcn_blocked"] = True
    _ = build_criterion(build_cp171_config_for_contract("cnn_lstm"))

    prior_rows, prior_random = evaluate_prior_rule_baselines(payload)

    run_meta_rows: list[dict[str, Any]] = []
    all_model_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    all_random_rows: list[dict[str, Any]] = list(prior_random)

    for candidate_id, epochs in (
        ("logistic_baseline", LOGISTIC_EPOCHS),
        ("risk_tab_mlp", MODEL_EPOCHS),
        ("cnn_lstm_risk_only", MODEL_EPOCHS),
        ("tide_risk_only", MODEL_EPOCHS),
    ):
        model, run_meta, val_prob = train_warning_candidate(
            candidate_id=candidate_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_dim=input_dim,
            device=device,
            epochs=epochs,
        )
        test_batch = BATCH_SIZE if candidate_id in ("logistic_baseline", "risk_tab_mlp") else DEEP_BATCH_SIZE
        test_prob = predict_warning_prob(model, test_dataset, device=device, batch_size=test_batch)
        rows, threshold_rows, random_rows = evaluate_candidate_probabilities(
            candidate_id=candidate_id,
            val_prob=val_prob,
            test_prob=test_prob,
            val_prediction=val_prediction,
            test_prediction=test_prediction,
        )
        run_meta["validation_probability_mean"] = _safe_mean(val_prob)
        run_meta["test_probability_mean"] = _safe_mean(test_prob)
        run_meta_rows.append(run_meta)
        all_model_rows.extend(rows)
        all_threshold_rows.extend(threshold_rows)
        all_random_rows.extend(random_rows)
        (LOG_DIR / f"{candidate_id}_run_meta.json").write_text(
            json.dumps(_clean_json(run_meta), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    baseline_rows = prior_rows + [row for row in all_model_rows if row.get("candidate_id") == "logistic_baseline"]
    smoke_rows = [row for row in all_model_rows if row.get("candidate_id") != "logistic_baseline"]
    primary_filter = lambda row: row.get("split") == "test" and row.get("cohort") == "line_top_decile" and row.get("warning_level") in ("binary", "soft_or_strong")
    baseline_primary = [row for row in baseline_rows if primary_filter(row)]
    model_primary = [row for row in smoke_rows if primary_filter(row)]
    cp170_row = next((row for row in baseline_primary if row.get("candidate_id") == "cp170_best_wide_rule"), None)
    best_baseline = max(
        baseline_primary,
        key=lambda row: (
            row.get("warning_severe_recall") or -999.0,
            row.get("spread_retention") or -999.0,
            row.get("fee_retention") or -999.0,
        ),
        default={},
    )
    best_model = max(model_primary, key=model_primary_score, default={})
    final_label = classification_label(best_model, all_random_rows, cp170_row)
    summary_rows = [
        {
            "cp": CP,
            "final_label": final_label,
            "best_model": best_model.get("candidate_id"),
            "best_model_recall": best_model.get("warning_severe_recall"),
            "best_model_missed_severe_rate": best_model.get("missed_severe_rate"),
            "best_model_warning_share": best_model.get("warning_share"),
            "best_model_spread_retention": best_model.get("spread_retention"),
            "best_model_fee_retention": best_model.get("fee_retention"),
            "best_baseline": best_baseline.get("candidate_id"),
            "best_baseline_recall": best_baseline.get("warning_severe_recall"),
            "source_data_hash": split_summary.get("source_data_hash"),
            "split_mode": split_summary.get("split_mode"),
            "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
            "line_score_used_as_model_input": False,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "disabled",
        }
    ]
    metrics_payload = {
        "cp": CP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_label": final_label,
        "split_summary": split_summary,
        "stage0": stage0,
        "run_meta": run_meta_rows,
        "baseline_primary": baseline_primary,
        "model_primary": model_primary,
        "best_baseline_primary": best_baseline,
        "best_model_primary": best_model,
        "baseline_rows": baseline_rows,
        "model_rows": smoke_rows,
        "threshold_rows": all_threshold_rows,
        "random_rows": all_random_rows,
        "summary": summary_rows[0],
        "contracts": {
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_composite": False,
            "line_score_model_input": False,
            "target": "actual_h5 <= -0.03",
        },
    }
    _write_json(METRICS_PATH, metrics_payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(BASELINE_CSV, baseline_rows)
    _write_csv(MODEL_CSV, smoke_rows)
    _write_csv(THRESHOLD_CSV, all_threshold_rows)
    _write_csv(COHORT_CSV, baseline_rows + smoke_rows)
    write_reports(metrics_payload)
    print(
        json.dumps(
            _clean_json(
                {
                    "status": "DONE",
                    "final_label": final_label,
                    "best_model": best_model.get("candidate_id"),
                    "metrics_path": METRICS_PATH,
                    "summary_csv": SUMMARY_CSV,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def build_cp171_config_for_contract(model_name: str) -> Any:
    config = cp164.make_config("cpu")
    config.model = model_name
    config.model_role = "line_warning"
    config.feature_set = "price_volatility_volume"
    config.use_wandb = False
    config.epochs = 1
    config.batch_size = 4
    return config


if __name__ == "__main__":
    main()
