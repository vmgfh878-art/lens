from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
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

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.models.common import ForecastOutput  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import build_registry  # noqa: E402
from ai.train import (  # noqa: E402
    TrainConfig,
    apply_feature_columns_to_splits,
    assert_dataset_features_finite,
    build_model,
    estimate_train_regime_thresholds,
    evaluate_bundle,
    forward_model,
    line_gate_eligible,
    make_loader,
    resolve_device,
    resolve_feature_columns,
    run_dry,
    run_training,
    summarize_dataset_plan,
)


TIMEFRAME = "1D"
HORIZON = 5
SEQ_LEN = 252
TARGET_TYPE = "raw_future_return"
PROVIDER = "yfinance"
FEATURE_SET = "price_volatility_volume"
MODEL_ROLE = "line_regime"
OUTPUT_ROLE = "line_regime"

PRICE_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.parquet"
PRICE_MANIFEST_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.manifest.json"
INDICATOR_PATH = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.parquet"
INDICATOR_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.manifest.json"
)

LOG_DIR = PROJECT_ROOT / "logs" / "cp158_lm_1d_line_regime_stage0_2"
DOCS_DIR = PROJECT_ROOT / "docs"
STAGE0_REPORT = DOCS_DIR / "cp158_lm_1d_line_regime_stage0_preflight_report.md"
STAGE1_REPORT = DOCS_DIR / "cp158_lm_1d_line_regime_stage1_baseline_report.md"
STAGE2_REPORT = DOCS_DIR / "cp158_lm_1d_line_regime_stage2_smoke_report.md"
METRICS_PATH = DOCS_DIR / "cp158_lm_1d_line_regime_metrics.json"
STAGE2_SUMMARY = DOCS_DIR / "cp158_lm_1d_line_regime_stage2_summary.csv"

CP153_GUARD_PATHS = [
    DOCS_DIR / "cp153_bm_1d_band_500_stage0_1_baseline_metrics.json",
    DOCS_DIR / "cp153_bm_1d_band_500_stage0_1_baseline_report.md",
]
CP154_SINGLE_HEAD_STAGE1 = DOCS_DIR / "cp154_lm_1d_line_v2_stage1_baseline_metrics.json"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model: str
    patch_len: int = 16
    patch_stride: int = 8
    lr: float = 7.362816234925851e-4
    weight_decay: float = 8.143270337695065e-5
    dropout: float = 0.10
    epochs: int = 1
    use_future_covariate: bool = False
    fp32_modules: str = "none"
    use_revin: bool = True
    note: str = ""


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
    path.write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    return prepared.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    price_manifest = _read_json(PRICE_MANIFEST_PATH) if PRICE_MANIFEST_PATH.exists() else {}
    indicator_manifest = (
        _read_json(INDICATOR_MANIFEST_PATH) if INDICATOR_MANIFEST_PATH.exists() else {}
    )
    price = _prepare_frame(pd.read_parquet(PRICE_PATH))
    indicators = _prepare_frame(pd.read_parquet(INDICATOR_PATH))
    indicators = indicators[indicators["timeframe"].astype(str).str.upper() == TIMEFRAME].copy()
    return price, indicators, price_manifest, indicator_manifest


def prepare_snapshot_overlay() -> dict[str, Any]:
    overlay_dir = LOG_DIR / "snapshot_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    links = [
        (PRICE_PATH, overlay_dir / "price_data_yfinance_1D.parquet"),
        (PRICE_MANIFEST_PATH, overlay_dir / "price_data_yfinance_1D.manifest.json"),
        (PRICE_PATH, overlay_dir / "price_data_yfinance.parquet"),
        (PRICE_MANIFEST_PATH, overlay_dir / "price_data_yfinance.manifest.json"),
        (PRICE_PATH, overlay_dir / "price_data.parquet"),
        (PRICE_MANIFEST_PATH, overlay_dir / "price_data.manifest.json"),
        (INDICATOR_PATH, overlay_dir / "indicators_yfinance_1D.parquet"),
        (INDICATOR_MANIFEST_PATH, overlay_dir / "indicators_yfinance_1D.manifest.json"),
        (INDICATOR_PATH, overlay_dir / "indicators.parquet"),
        (INDICATOR_MANIFEST_PATH, overlay_dir / "indicators.manifest.json"),
    ]
    entries: list[dict[str, Any]] = []
    for source, target in links:
        if not source.exists():
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            mode = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        entries.append({"source": source, "target": target, "mode": mode})
    os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(overlay_dir)
    return {"overlay_dir": overlay_dir, "entries": entries}


def build_split_payload(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    source_data_hash: str,
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    all_tickers = sorted({str(ticker).upper() for ticker in indicators["ticker"].dropna().unique()})
    registry = build_registry(all_tickers, TIMEFRAME)
    registry_path = LOG_DIR / "cp158_ticker_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        ticker_registry=registry,
        ticker_registry_path=str(registry_path),
        market_data_provider=PROVIDER,
        source_data_hash=source_data_hash,
    )
    eligible = set(plan.eligible_tickers)
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(eligible)].copy(),
        price_df=price[price["ticker"].isin(eligible)].copy(),
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    train, val, test, mean, std = normalize_sequence_splits(train, val, test)
    return train, val, test, mean, std, plan, registry


def collect_targets(bundle: Any) -> np.ndarray:
    targets = np.empty((len(bundle.sample_refs), bundle.horizon), dtype=np.float32)
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        arrays = bundle.ticker_arrays[str(ticker)]
        closes = np.asarray(arrays["closes"], dtype=np.float64)
        anchor = float(closes[int(end_idx)])
        future = closes[int(end_idx) + 1 : int(end_idx) + 1 + bundle.horizon]
        targets[row_idx] = ((future / anchor) - 1.0).astype(np.float32)
    return targets


def collect_recent_return_score(bundle: Any, lookback: int = 20) -> np.ndarray:
    scores = np.empty(len(bundle.sample_refs), dtype=np.float32)
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        arrays = bundle.ticker_arrays[str(ticker)]
        closes = np.asarray(arrays["closes"], dtype=np.float64)
        end_idx = int(end_idx)
        if end_idx - lookback < 0:
            scores[row_idx] = 0.0
            continue
        base = float(closes[end_idx - lookback])
        now = float(closes[end_idx])
        scores[row_idx] = 0.0 if base == 0.0 else float((now / base) - 1.0)
    return scores


def split_overlap_summary(train: Any, val: Any, test: Any) -> dict[str, Any]:
    def keys(bundle: Any) -> set[tuple[str, int]]:
        return {(str(ticker), int(end_idx)) for ticker, end_idx in bundle.sample_refs}

    return {
        "train_val_sample_overlap": len(keys(train) & keys(val)),
        "train_test_sample_overlap": len(keys(train) & keys(test)),
        "val_test_sample_overlap": len(keys(val) & keys(test)),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_min": str(train.metadata["asof_date"].min()),
        "train_date_max": str(train.metadata["asof_date"].max()),
        "val_date_min": str(val.metadata["asof_date"].min()),
        "val_date_max": str(val.metadata["asof_date"].max()),
        "test_date_min": str(test.metadata["asof_date"].min()),
        "test_date_max": str(test.metadata["asof_date"].max()),
    }


def finite_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(array)
    return {
        "name": name,
        "shape": list(array.shape),
        "finite_count": int(finite.sum()),
        "total_count": int(array.size),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "finite_ok": bool(finite.all()),
    }


def cp153_artifact_state() -> dict[str, Any]:
    return {
        str(path): {
            "exists": path.exists(),
            "mtime": path.stat().st_mtime if path.exists() else None,
            "size": path.stat().st_size if path.exists() else None,
        }
        for path in CP153_GUARD_PATHS
    }


def class_targets(
    h5_return: np.ndarray, thresholds: list[float] | tuple[float, float, float, float]
) -> np.ndarray:
    target = np.zeros(len(h5_return), dtype=np.int64)
    for threshold in thresholds:
        target += (h5_return > float(threshold)).astype(np.int64)
    return target


def class_distribution(classes: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(classes.astype(np.int64), minlength=5)
    total = int(counts.sum())
    return {
        "counts": {str(idx): int(counts[idx]) for idx in range(5)},
        "rates": {str(idx): float(counts[idx] / total) if total else None for idx in range(5)},
        "total": total,
    }


def _macro_f1(pred: np.ndarray, target: np.ndarray) -> float | None:
    values: list[float] = []
    for cls in range(5):
        pred_pos = pred == cls
        target_pos = target == cls
        tp = float(np.logical_and(pred_pos, target_pos).sum())
        fp = float(np.logical_and(pred_pos, ~target_pos).sum())
        fn = float(np.logical_and(~pred_pos, target_pos).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        values.append(
            (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        )
    return float(np.mean(values)) if values else None


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    value = frame["x"].corr(frame["y"], method="spearman")
    return float(value) if pd.notna(value) else None


def regime_class_metrics(
    pred: np.ndarray, target: np.ndarray, actual: np.ndarray
) -> dict[str, Any]:
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    error = np.abs(pred - target).astype(np.float64)
    result: dict[str, Any] = {
        "regime_accuracy": float((pred == target).mean()),
        "regime_adjacent_accuracy": float((error <= 1.0).mean()),
        "regime_ordinal_mae": float(error.mean()),
        "regime_ordinal_mse": float((error**2).mean()),
        "regime_macro_f1": _macro_f1(pred, target),
    }
    for cls in range(5):
        pred_pos = pred == cls
        target_pos = target == cls
        tp = float(np.logical_and(pred_pos, target_pos).sum())
        fp = float(np.logical_and(pred_pos, ~target_pos).sum())
        fn = float(np.logical_and(~pred_pos, target_pos).sum())
        result[f"class_{cls}_precision"] = tp / (tp + fp) if tp + fp > 0 else None
        result[f"class_{cls}_recall"] = tp / (tp + fn) if tp + fn > 0 else None
        result[f"predicted_class_{cls}_mean_return"] = (
            float(actual[pred_pos].mean()) if pred_pos.any() else None
        )
    result["strong_down_recall"] = result["class_0_recall"]
    result["strong_down_precision"] = result["class_0_precision"]
    result["strong_up_precision"] = result["class_4_precision"]
    means = [result[f"predicted_class_{idx}_mean_return"] for idx in range(5)]
    valid_idx = np.asarray(
        [idx for idx, value in enumerate(means) if value is not None], dtype=np.float64
    )
    valid_ret = np.asarray([value for value in means if value is not None], dtype=np.float64)
    result["regime_return_monotonicity"] = (
        _spearman(valid_idx, valid_ret) if len(valid_idx) >= 3 else None
    )
    result["regime_return_monotonic_non_decreasing"] = bool(
        all(
            means[idx] is None or means[idx + 1] is None or means[idx] <= means[idx + 1]
            for idx in range(4)
        )
    )
    return result


def fit_score_thresholds(score: np.ndarray, target_class: np.ndarray) -> list[float]:
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    score = score[finite]
    target_class = target_class[finite]
    if len(score) == 0:
        return [-0.01, 0.0, 0.01, 0.02]
    proportions = [float((target_class <= idx).mean()) for idx in range(4)]
    return [float(np.quantile(score, min(max(prop, 0.001), 0.999))) for prop in proportions]


def predict_class_from_score(score: np.ndarray, thresholds: list[float]) -> np.ndarray:
    pred = np.zeros(len(score), dtype=np.int64)
    for threshold in thresholds:
        pred += (score > float(threshold)).astype(np.int64)
    return pred


def build_config(spec: CandidateSpec, *, seed: int, device: str) -> TrainConfig:
    return TrainConfig(
        model=spec.model,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        seq_len=SEQ_LEN,
        epochs=spec.epochs,
        batch_size=256,
        lr=spec.lr,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=spec.weight_decay,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        lambda_regime=0.5,
        lambda_ordinal=0.1,
        dropout=spec.dropout,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=16,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=bool(spec.use_future_covariate),
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=seed,
        device=device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules=spec.fp32_modules,
        use_wandb=False,
        wandb_project="lens-cp158-line-regime",
        model_ver=f"cp158_{spec.candidate_id}",
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        checkpoint_selection="line_gate",
        amp_dtype="bf16",
        detect_anomaly=False,
        explicit_cuda_cleanup=True,
        hard_exit_after_result=False,
        use_revin=spec.use_revin,
        patch_len=spec.patch_len,
        patch_stride=spec.patch_stride,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=FEATURE_SET,
        market_data_provider=PROVIDER,
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
        model_role=MODEL_ROLE,
        risk_decision_threshold=0.5,
    )


def build_single_head_reference_config(*, device: str) -> TrainConfig:
    return TrainConfig(
        model="patchtst",
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        seq_len=SEQ_LEN,
        epochs=1,
        batch_size=512,
        lr=7.362816234925851e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=8.143270337695065e-5,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.10,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=16,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=False,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=42,
        device=device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-cp158-line-regime",
        model_ver="cp158_single_head_reference",
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        checkpoint_selection="line_gate",
        amp_dtype="bf16",
        detect_anomaly=False,
        explicit_cuda_cleanup=True,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=32,
        patch_stride=16,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=FEATURE_SET,
        market_data_provider=PROVIDER,
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
        model_role="legacy",
    )


def find_cp154_single_head_checkpoint() -> str | None:
    if not CP154_SINGLE_HEAD_STAGE1.exists():
        return None
    payload = _read_json(CP154_SINGLE_HEAD_STAGE1)
    run = payload.get("single_head_reference_run")
    if isinstance(run, dict):
        checkpoint = run.get("checkpoint_path")
        if checkpoint:
            path = PROJECT_ROOT / str(checkpoint)
            if path.exists():
                return str(path)
    text = json.dumps(payload, ensure_ascii=False)
    marker = "patchtst_1D_patchtst-1D-59d659036b64.pt"
    if marker in text:
        path = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / marker
        if path.exists():
            return str(path)
    return None


def load_model_for_prediction(config: TrainConfig, checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config") or {}
    config.num_tickers = int(checkpoint_config.get("num_tickers") or config.num_tickers)
    config.ticker_registry_path = (
        checkpoint_config.get("ticker_registry_path") or config.ticker_registry_path
    )
    config.feature_columns = list(
        checkpoint_config.get("feature_columns")
        or config.feature_columns
        or resolve_feature_columns(config.feature_set)
    )
    config.n_features = int(checkpoint_config.get("n_features") or len(config.feature_columns))
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def collect_line_scores_from_checkpoint(
    *,
    config: TrainConfig,
    checkpoint_path: str,
    bundle: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, selected_bundle, _, _, _ = apply_feature_columns_to_splits(
        bundle,
        bundle,
        bundle,
        mean,
        std,
        config.feature_columns or resolve_feature_columns(config.feature_set),
    )
    model, payload = load_model_for_prediction(config, checkpoint_path, device)
    loader = make_loader(
        selected_bundle, batch_size=512, shuffle=False, device=device, num_workers=0
    )
    line_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for features, _line_target, _band_target, raw_returns, ticker_id, future_cov in loader:
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_cov = future_cov.to(device, non_blocking=True)
            output = forward_model(model, features, ticker_id, future_cov)
            if not isinstance(output, ForecastOutput):
                raise TypeError(
                    f"single-head reference는 ForecastOutput이어야 합니다: {type(output).__name__}"
                )
            line_chunks.append(output.line.detach().cpu())
            raw_chunks.append(raw_returns.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    line = torch.cat(line_chunks, dim=0).to(torch.float32).numpy()
    return line[:, -1], {
        "checkpoint_config": payload.get("config", {}),
        "raw_shape": list(torch.cat(raw_chunks, dim=0).shape),
    }


def stage1_baselines(
    *,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    thresholds: list[float],
    val_bundle: Any,
    test_bundle: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    train_h5 = train_targets[:, -1]
    val_h5 = val_targets[:, -1]
    test_h5 = test_targets[:, -1]
    train_class = class_targets(train_h5, thresholds)
    val_class = class_targets(val_h5, thresholds)
    test_class = class_targets(test_h5, thresholds)
    majority_class = int(np.bincount(train_class, minlength=5).argmax())
    rng = np.random.default_rng(42)
    train_dist = np.bincount(train_class, minlength=5).astype(np.float64)
    train_dist = train_dist / train_dist.sum()

    baselines: dict[str, Any] = {}
    for name, val_pred, test_pred in [
        (
            "majority_baseline",
            np.full_like(val_class, majority_class),
            np.full_like(test_class, majority_class),
        ),
        (
            "stratified_random_baseline",
            rng.choice(5, size=len(val_class), p=train_dist),
            rng.choice(5, size=len(test_class), p=train_dist),
        ),
    ]:
        baselines[name] = {
            "validation": regime_class_metrics(np.asarray(val_pred), val_class, val_h5),
            "test": regime_class_metrics(np.asarray(test_pred), test_class, test_h5),
        }

    val_roll = collect_recent_return_score(val_bundle, lookback=20)
    test_roll = collect_recent_return_score(test_bundle, lookback=20)
    roll_thresholds = fit_score_thresholds(val_roll, val_class)
    baselines["rolling_return_quantile_baseline"] = {
        "score": "20일 누적수익률",
        "validation_score_thresholds": roll_thresholds,
        "validation": regime_class_metrics(
            predict_class_from_score(val_roll, roll_thresholds), val_class, val_h5
        ),
        "test": regime_class_metrics(
            predict_class_from_score(test_roll, roll_thresholds), test_class, test_h5
        ),
    }

    checkpoint_path = find_cp154_single_head_checkpoint()
    if checkpoint_path:
        config = build_single_head_reference_config(device=str(device))
        val_score, val_meta = collect_line_scores_from_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            bundle=val_bundle,
            mean=mean,
            std=std,
            device=device,
        )
        test_score, test_meta = collect_line_scores_from_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            bundle=test_bundle,
            mean=mean,
            std=std,
            device=device,
        )
        score_thresholds = fit_score_thresholds(val_score, val_class)
        baselines["single_head_line_score_posthoc_5class_binning"] = {
            "checkpoint_path": checkpoint_path,
            "validation_score_thresholds": score_thresholds,
            "validation_meta": val_meta,
            "test_meta": test_meta,
            "validation": regime_class_metrics(
                predict_class_from_score(val_score, score_thresholds), val_class, val_h5
            ),
            "test": regime_class_metrics(
                predict_class_from_score(test_score, score_thresholds), test_class, test_h5
            ),
        }
    else:
        baselines["single_head_line_score_posthoc_5class_binning"] = {
            "status": "unavailable",
            "reason": "CP154 single-head checkpoint를 찾지 못했다.",
        }

    return {
        "train_class_distribution": class_distribution(train_class),
        "validation_class_distribution": class_distribution(val_class),
        "test_class_distribution": class_distribution(test_class),
        "baselines": baselines,
    }


def run_candidate(
    spec: CandidateSpec,
    *,
    seed: int,
    device: str,
    precomputed_bundles: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[TrainConfig, dict[str, Any]]:
    config = build_config(spec, seed=seed, device=device)
    print(
        json.dumps(
            {
                "stage": "cp158_train_start",
                "candidate_id": spec.candidate_id,
                "model": spec.model,
                "model_role": MODEL_ROLE,
                "output_role": OUTPUT_ROLE,
                "epochs": spec.epochs,
                "device": device,
                "feature_set": FEATURE_SET,
                "save_run": False,
                "wandb": "disabled",
            },
            ensure_ascii=False,
        )
    )
    result = run_training(
        config,
        save_run=False,
        precomputed_bundles=precomputed_bundles,
        enable_compile=False,
        local_log=True,
        local_log_dir=LOG_DIR / "train_local_logs" / spec.candidate_id,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return config, result


def row_from_result(spec: CandidateSpec, result: dict[str, Any]) -> dict[str, Any]:
    best = result.get("best_metrics") or {}
    test = result.get("test_metrics") or {}
    return {
        "candidate_id": spec.candidate_id,
        "model": spec.model,
        "patch_len": spec.patch_len,
        "patch_stride": spec.patch_stride,
        "epochs": spec.epochs,
        "line_gate_pass": bool(best.get("line_gate_pass"))
        if best.get("line_gate_pass") is not None
        else line_gate_eligible(best),
        "val_ic": best.get("ic_mean"),
        "val_spread": best.get("long_short_spread"),
        "val_fee": best.get("fee_adjusted_return"),
        "val_regime_accuracy": best.get("regime_accuracy"),
        "val_adjacent_accuracy": best.get("regime_adjacent_accuracy"),
        "val_ordinal_mae": best.get("regime_ordinal_mae"),
        "val_macro_f1": best.get("regime_macro_f1"),
        "val_strong_down_recall": best.get("strong_down_recall"),
        "val_monotonicity": best.get("regime_return_monotonicity"),
        "val_filter_false_safe_reduction": best.get("regime_filtered_false_safe_reduction"),
        "val_spread_retention": best.get("spread_retention"),
        "val_regime_risk_line_corr": best.get("regime_risk_line_corr_spearman"),
        "val_regime_risk_residual_auc": best.get("regime_risk_residual_severe_auc"),
        "val_top_false_safe_date_max_share": best.get("top_false_safe_date_max_share"),
        "val_top_false_safe_ticker_max_share": best.get("top_false_safe_ticker_max_share"),
        "test_ic": test.get("ic_mean"),
        "test_spread": test.get("long_short_spread"),
        "test_fee": test.get("fee_adjusted_return"),
        "test_regime_accuracy": test.get("regime_accuracy"),
        "test_adjacent_accuracy": test.get("regime_adjacent_accuracy"),
        "test_ordinal_mae": test.get("regime_ordinal_mae"),
        "test_macro_f1": test.get("regime_macro_f1"),
        "test_strong_down_recall": test.get("strong_down_recall"),
        "test_monotonicity": test.get("regime_return_monotonicity"),
        "test_filter_false_safe_reduction": test.get("regime_filtered_false_safe_reduction"),
        "test_spread_retention": test.get("spread_retention"),
        "test_regime_risk_line_corr": test.get("regime_risk_line_corr_spearman"),
        "test_regime_risk_residual_auc": test.get("regime_risk_residual_severe_auc"),
        "test_top_false_safe_date_max_share": test.get("top_false_safe_date_max_share"),
        "test_top_false_safe_ticker_max_share": test.get("top_false_safe_ticker_max_share"),
        "checkpoint_path": result.get("checkpoint_path"),
        "run_id": result.get("run_id"),
        "local_log_dir": result.get("local_log_dir"),
    }


def judgement_from_rows(
    rows: list[dict[str, Any]], baseline_sota: dict[str, Any]
) -> dict[str, Any]:
    sota_mae = baseline_sota.get("validation_ordinal_mae")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        line_alive = (
            bool(row.get("line_gate_pass"))
            and (row.get("val_ic") or 0) > 0
            and (row.get("val_spread") or 0) > 0
            and (row.get("val_fee") or 0) > 0
        )
        regime_better = False
        if sota_mae is not None and row.get("val_ordinal_mae") is not None:
            regime_better = float(row["val_ordinal_mae"]) < float(sota_mae)
        filter_help = (row.get("val_filter_false_safe_reduction") or 0) > 0
        monotonic = row.get("val_monotonicity") is not None and float(row["val_monotonicity"]) > 0
        if line_alive and regime_better and filter_help and monotonic:
            label = "stage3_candidate"
        elif regime_better or filter_help:
            label = "research_reserve"
        else:
            label = "not_enough_signal"
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "label": label,
                "line_alive": line_alive,
                "regime_better_than_baseline_sota": regime_better,
                "regime_filter_helped": filter_help,
                "monotonic_positive": monotonic,
            }
        )
    stage3 = [item for item in candidates if item["label"] == "stage3_candidate"]
    reserve = [item for item in candidates if item["label"] == "research_reserve"]
    return {
        "stage3_execute_now": False,
        "stage3_candidate_count": len(stage3),
        "research_reserve_count": len(reserve),
        "candidate_judgements": candidates,
        "overall": "stage3_candidates_need_user_review"
        if stage3
        else ("research_reserve_only" if reserve else "no_clear_dual_signal"),
    }


def write_stage0_report(metrics: dict[str, Any]) -> None:
    thresholds = metrics["regime_thresholds"]
    lines = [
        "# CP158-LM 1D Line V2.1 Regime Head Stage 0 Preflight",
        "",
        f"- provider/source: {metrics['provider']} / {metrics['source']}",
        f"- source_data_hash: {metrics['source_data_hash']}",
        f"- eligible ticker 수: {metrics['dataset_plan']['eligible_ticker_count']}",
        f"- feature_version: {metrics['feature_version']}",
        f"- feature/target finite: {metrics['feature_finite_pass']} / {metrics['target_finite_pass']}",
        f"- split overlap 0: {metrics['split_overlap_zero']}",
        f"- regime 경계값: q10={thresholds['regime_threshold_q10']:.6f}, q35={thresholds['regime_threshold_q35']:.6f}, q65={thresholds['regime_threshold_q65']:.6f}, q90={thresholds['regime_threshold_q90']:.6f}",
        f"- train class distribution: {metrics['class_distribution']['train']['counts']}",
        f"- validation class distribution: {metrics['class_distribution']['validation']['counts']}",
        f"- test class distribution: {metrics['class_distribution']['test']['counts']}",
        f"- line_regime dry-run: {metrics['dry_run']['forward_smoke']}",
        f"- TCN line_regime 차단: {metrics['tcn_line_regime_blocked']}",
        f"- CP153 band artifact 변경: {metrics['cp153_artifact_guard']['changed']}",
        "",
        "제품 save-run, DB write, inference 저장, band/composite 실행은 하지 않았다.",
    ]
    STAGE0_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage1_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# CP158-LM 1D Line V2.1 Regime Head Stage 1 Baseline",
        "",
        "Stage 1은 line_regime head가 기존 line score의 사후 binning보다 의미 있는지 보기 위한 기준선이다.",
        "",
        "| baseline | val ordinal_mae | test ordinal_mae | val adjacent_acc | test adjacent_acc | val strong_down_recall | test strong_down_recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in metrics["baselines"].items():
        if payload.get("status") == "unavailable":
            lines.append(
                f"| {name} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |"
            )
            continue
        val = payload.get("validation") or {}
        test = payload.get("test") or {}
        lines.append(
            "| {name} | {v_mae} | {t_mae} | {v_adj} | {t_adj} | {v_sd} | {t_sd} |".format(
                name=name,
                v_mae=_fmt(val.get("regime_ordinal_mae")),
                t_mae=_fmt(test.get("regime_ordinal_mae")),
                v_adj=_fmt(val.get("regime_adjacent_accuracy")),
                t_adj=_fmt(test.get("regime_adjacent_accuracy")),
                v_sd=_fmt(val.get("strong_down_recall")),
                t_sd=_fmt(test.get("strong_down_recall")),
            )
        )
    lines.extend(
        [
            "",
            f"- validation baseline SOTA ordinal_mae: {metrics['baseline_sota']['validation_ordinal_mae_baseline']} = {_fmt(metrics['baseline_sota']['validation_ordinal_mae'])}",
            "- single-head line score post-hoc 5-class binning은 validation score threshold를 고정하고 test에는 재조정 없이 적용했다.",
            "- 제품 save-run, DB write, inference 저장은 하지 않았다.",
        ]
    )
    STAGE1_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else ""


def write_stage2_outputs(
    rows: list[dict[str, Any]], metrics: dict[str, Any], judgement: dict[str, Any]
) -> None:
    fieldnames = [
        "candidate_id",
        "model",
        "patch_len",
        "patch_stride",
        "epochs",
        "line_gate_pass",
        "val_ic",
        "val_spread",
        "val_fee",
        "val_regime_accuracy",
        "val_adjacent_accuracy",
        "val_ordinal_mae",
        "val_macro_f1",
        "val_strong_down_recall",
        "val_monotonicity",
        "val_filter_false_safe_reduction",
        "val_spread_retention",
        "val_regime_risk_line_corr",
        "val_regime_risk_residual_auc",
        "val_top_false_safe_date_max_share",
        "val_top_false_safe_ticker_max_share",
        "test_ic",
        "test_spread",
        "test_fee",
        "test_regime_accuracy",
        "test_adjacent_accuracy",
        "test_ordinal_mae",
        "test_macro_f1",
        "test_strong_down_recall",
        "test_monotonicity",
        "test_filter_false_safe_reduction",
        "test_spread_retention",
        "test_regime_risk_line_corr",
        "test_regime_risk_residual_auc",
        "test_top_false_safe_date_max_share",
        "test_top_false_safe_ticker_max_share",
        "checkpoint_path",
        "run_id",
        "local_log_dir",
    ]
    with STAGE2_SUMMARY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# CP158-LM 1D Line V2.1 Regime Head Stage 2 Smoke",
        "",
        "본 실험은 regime head가 line head의 원시 수익 예측력을 직접 개선한다고 가정하지 않는다.",
        "대신 단일 line head가 놓친 high-confidence false-safe 문제를, regime head가 사후 해석 및 후보 필터링 단계에서 완화할 수 있는지 검증한다.",
        "",
        "기존 asymmetric downside loss는 리스크를 직접 분류하는 장치가 아니라, 과대예측을 피하도록 만든 보수적 점추정 장치였다.",
        "CP148/CP154 결과상 이 장치만으로는 tail risk를 분리하기 부족했으므로, line head와 regime head를 분리한다.",
        "",
        "| candidate | line_gate | val IC | val spread | val fee | val ordinal_mae | test ordinal_mae | val filter FS reduction | test filter FS reduction | test risk-line corr | test residual AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {gate} | {ic} | {spread} | {fee} | {vmae} | {tmae} | {vfs} | {tfs} | {corr} | {auc} |".format(
                candidate=row["candidate_id"],
                gate=row.get("line_gate_pass"),
                ic=_fmt(row.get("val_ic")),
                spread=_fmt(row.get("val_spread")),
                fee=_fmt(row.get("val_fee")),
                vmae=_fmt(row.get("val_ordinal_mae")),
                tmae=_fmt(row.get("test_ordinal_mae")),
                vfs=_fmt(row.get("val_filter_false_safe_reduction")),
                tfs=_fmt(row.get("test_filter_false_safe_reduction")),
                corr=_fmt(row.get("test_regime_risk_line_corr")),
                auc=_fmt(row.get("test_regime_risk_residual_auc")),
            )
        )
    lines.extend(
        [
            "",
            f"- Stage 3 즉시 실행 여부: {judgement['stage3_execute_now']}",
            f"- 전체 판정: {judgement['overall']}",
            "- Stage 3 후보 제안은 결과 해석용이며 이번 CP에서는 Stage 3를 실행하지 않았다.",
            "- 제품 save-run, DB write, inference 저장, band/composite 실행은 하지 않았다.",
            f"- CP153 band artifact 변경: {metrics['cp153_artifact_guard']['changed']}",
            "",
            "## 후보별 판정",
        ]
    )
    for item in judgement["candidate_judgements"]:
        lines.append(
            f"- {item['candidate_id']}: {item['label']} "
            f"(line_alive={item['line_alive']}, regime_better={item['regime_better_than_baseline_sota']}, "
            f"filter_helped={item['regime_filter_helped']})"
        )
    STAGE2_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage0(
    precomputed: tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]],
    manifests: dict[str, Any],
    cp153_before: dict[str, Any],
) -> dict[str, Any]:
    train, val, test, mean, std, plan, _registry = precomputed
    del mean, std
    train_targets = collect_targets(train)
    val_targets = collect_targets(val)
    test_targets = collect_targets(test)
    thresholds = estimate_train_regime_thresholds(train)
    train_class = class_targets(train_targets[:, -1], thresholds["regime_thresholds"])
    val_class = class_targets(val_targets[:, -1], thresholds["regime_thresholds"])
    test_class = class_targets(test_targets[:, -1], thresholds["regime_thresholds"])
    assert_dataset_features_finite("train", train)
    assert_dataset_features_finite("val", val)
    assert_dataset_features_finite("test", test)
    split_overlap = split_overlap_summary(train, val, test)
    dry_config = build_config(
        CandidateSpec(
            "dry_run_patchtst_line_regime",
            model="patchtst",
            patch_len=32,
            patch_stride=16,
            epochs=1,
        ),
        seed=42,
        device="cpu",
    )
    dry_config.limit_tickers = 5
    dry = run_dry(dry_config)
    tcn_blocked = False
    try:
        build_config(
            CandidateSpec("dry_run_tcn_block", model="tcn_quantile"), seed=42, device="cpu"
        )
        tcn_config = TrainConfig(**{**dry_config.__dict__, "model": "tcn_quantile"})
        build_model(tcn_config)
    except ValueError:
        tcn_blocked = True
    cp153_after = cp153_artifact_state()
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "source": PROVIDER,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "source_data_hash": manifests["indicator"].get("source_data_hash"),
        "price_manifest": manifests["price"],
        "indicator_manifest": manifests["indicator"],
        "dataset_plan": summarize_dataset_plan(plan, train, val, test),
        "split_overlap": split_overlap,
        "split_overlap_zero": bool(
            split_overlap["train_val_sample_overlap"] == 0
            and split_overlap["train_test_sample_overlap"] == 0
            and split_overlap["val_test_sample_overlap"] == 0
        ),
        "feature_finite_pass": True,
        "target_finite_pass": bool(
            np.isfinite(train_targets).all()
            and np.isfinite(val_targets).all()
            and np.isfinite(test_targets).all()
        ),
        "target_finite_summary": {
            "train": finite_summary("train_targets", train_targets),
            "validation": finite_summary("val_targets", val_targets),
            "test": finite_summary("test_targets", test_targets),
        },
        "regime_thresholds": thresholds,
        "class_distribution": {
            "train": class_distribution(train_class),
            "validation": class_distribution(val_class),
            "test": class_distribution(test_class),
        },
        "dry_run": dry,
        "line_regime_output_only": dry.get("forward_smoke", {}).get("output_type") == "line_regime",
        "band_output_used": False,
        "composite_used": False,
        "tcn_line_regime_blocked": tcn_blocked,
        "cp153_artifact_guard": {
            "before": cp153_before,
            "after": cp153_after,
            "changed": cp153_before != cp153_after,
        },
        "db_write": False,
        "save_run": False,
        "inference_saved": False,
    }
    write_stage0_report(metrics)
    return metrics


def baseline_sota(stage1: dict[str, Any]) -> dict[str, Any]:
    best_name = None
    best_value = None
    for name, payload in stage1["baselines"].items():
        val = payload.get("validation") if isinstance(payload, dict) else None
        if not val:
            continue
        value = val.get("regime_ordinal_mae")
        if value is None:
            continue
        if best_value is None or float(value) < float(best_value):
            best_name = name
            best_value = float(value)
    return {
        "validation_ordinal_mae_baseline": best_name,
        "validation_ordinal_mae": best_value,
    }


def spec_from_row(row: dict[str, Any]) -> CandidateSpec:
    candidate_id = str(row["candidate_id"])
    model = str(row["model"])
    return CandidateSpec(
        candidate_id=candidate_id,
        model=model,
        patch_len=int(row.get("patch_len") or 16),
        patch_stride=int(row.get("patch_stride") or 8),
        epochs=int(row.get("epochs") or 1),
        use_future_covariate=model == "tide",
        fp32_modules="lstm,heads" if model == "cnn_lstm" else "none",
        use_revin=model == "patchtst",
    )


def refresh_existing_outputs() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"기존 metrics 파일이 없습니다: {METRICS_PATH}")
    metrics = _read_json(METRICS_PATH)
    price, indicators, price_manifest, indicator_manifest = load_source_frames()
    source_hash = str(
        indicator_manifest.get("source_data_hash")
        or price_manifest.get("source_data_hash")
        or "unknown"
    )
    train, val, test, mean, std, plan, _registry = build_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    refreshed_rows: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for row in metrics["stage2"]["candidate_rows"]:
        spec = spec_from_row(row)
        checkpoint_path = str(row.get("checkpoint_path") or "")
        config = build_config(spec, seed=42, device=str(device))
        model, payload = load_model_for_prediction(config, checkpoint_path, device)
        feature_columns = config.feature_columns or resolve_feature_columns(config.feature_set)
        _, val_selected, test_selected, _, _ = apply_feature_columns_to_splits(
            val, val, test, mean, std, feature_columns
        )
        thresholds = (
            payload.get("config", {}).get("regime_thresholds")
            or metrics["stage0"]["regime_thresholds"]["regime_thresholds"]
        )
        val_metrics = evaluate_bundle(
            model,
            val_selected,
            device,
            batch_size=512,
            num_workers=0,
            line_target_type=TARGET_TYPE,
            band_target_type=TARGET_TYPE,
            regime_thresholds=thresholds,
            amp_dtype="bf16",
            phase="val_refresh",
            run_id=str(row.get("run_id") or spec.candidate_id),
            model_role=MODEL_ROLE,
        )
        test_metrics = evaluate_bundle(
            model,
            test_selected,
            device,
            batch_size=512,
            num_workers=0,
            line_target_type=TARGET_TYPE,
            band_target_type=TARGET_TYPE,
            regime_thresholds=thresholds,
            amp_dtype="bf16",
            phase="test_refresh",
            run_id=str(row.get("run_id") or spec.candidate_id),
            model_role=MODEL_ROLE,
        )
        refreshed_result = {
            "run_id": row.get("run_id"),
            "checkpoint_path": checkpoint_path,
            "local_log_dir": row.get("local_log_dir"),
            "best_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        refreshed_rows.append(row_from_result(spec, refreshed_result))
        runs[spec.candidate_id] = {
            "spec": spec.__dict__,
            "checkpoint_config": payload.get("config", {}),
            "refreshed_validation_metrics": val_metrics,
            "refreshed_test_metrics": test_metrics,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    metrics["stage2"]["candidate_rows"] = refreshed_rows
    metrics["stage2"]["runs"] = runs
    metrics["stage2"]["refreshed_with_regime_diagnostics"] = True
    metrics["stage2"]["refresh_created_at"] = datetime.now(timezone.utc).isoformat()
    judgement = judgement_from_rows(refreshed_rows, metrics["stage1"]["baseline_sota"])
    metrics["stage2"]["judgement"] = judgement
    metrics["cp153_artifact_guard"]["after_refresh"] = cp153_artifact_state()
    metrics["cp153_artifact_guard"]["changed_after_refresh"] = (
        metrics["cp153_artifact_guard"].get("before")
        != metrics["cp153_artifact_guard"]["after_refresh"]
    )
    _write_json(METRICS_PATH, metrics)
    write_stage2_outputs(refreshed_rows, metrics, judgement)
    print(
        json.dumps(
            {
                "status": "CP158_REFRESH_DONE",
                "metrics": str(METRICS_PATH),
                "summary_csv": str(STAGE2_SUMMARY),
                "judgement": judgement["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cp153_before = cp153_artifact_state()
    overlay = prepare_snapshot_overlay()
    price, indicators, price_manifest, indicator_manifest = load_source_frames()
    source_hash = str(
        indicator_manifest.get("source_data_hash")
        or price_manifest.get("source_data_hash")
        or "unknown"
    )
    precomputed = build_split_payload(
        price=price, indicators=indicators, source_data_hash=source_hash
    )
    train, val, test, mean, std, plan, _registry = precomputed
    precomputed_bundles = (train, val, test, mean, std, plan)
    manifests = {"price": price_manifest, "indicator": indicator_manifest, "overlay": overlay}

    stage0 = run_stage0(precomputed, manifests, cp153_before)
    train_targets = collect_targets(train)
    val_targets = collect_targets(val)
    test_targets = collect_targets(test)
    stage1 = stage1_baselines(
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        thresholds=stage0["regime_thresholds"]["regime_thresholds"],
        val_bundle=val,
        test_bundle=test,
        mean=mean,
        std=std,
        device=resolve_device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    stage1["baseline_sota"] = baseline_sota(stage1)
    stage1["created_at"] = datetime.now(timezone.utc).isoformat()
    stage1["provider"] = PROVIDER
    stage1["source_data_hash"] = source_hash
    stage1["db_write"] = False
    stage1["save_run"] = False
    stage1["inference_saved"] = False
    write_stage1_report(stage1)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = [
        CandidateSpec(
            candidate_id="patchtst_line_regime_p32_s16",
            model="patchtst",
            patch_len=32,
            patch_stride=16,
            lr=7.362816234925851e-4,
            weight_decay=8.143270337695065e-5,
            dropout=0.10,
            epochs=1,
            use_revin=True,
            note="CP148/154 p32/s16에 가장 가까운 PatchTST line_regime",
        ),
        CandidateSpec(
            candidate_id="patchtst_line_regime_p16_s8",
            model="patchtst",
            patch_len=16,
            patch_stride=8,
            lr=1.0e-3,
            weight_decay=1.9e-4,
            dropout=0.18,
            epochs=1,
            use_revin=True,
            note="단기 변화 해상도를 높인 PatchTST line_regime",
        ),
        CandidateSpec(
            candidate_id="tide_line_regime_light_pvv",
            model="tide",
            patch_len=16,
            patch_stride=8,
            lr=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.10,
            epochs=1,
            use_future_covariate=True,
            use_revin=False,
            note="TiDE light line_regime reference",
        ),
        CandidateSpec(
            candidate_id="cnn_lstm_line_regime_reference",
            model="cnn_lstm",
            patch_len=16,
            patch_stride=8,
            lr=5.0e-4,
            weight_decay=1.0e-4,
            dropout=0.10,
            epochs=1,
            fp32_modules="lstm,heads",
            use_revin=False,
            note="CNN-LSTM line_regime reference",
        ),
    ]
    rows: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for spec in candidates:
        config, result = run_candidate(
            spec, seed=42, device=device_name, precomputed_bundles=precomputed_bundles
        )
        rows.append(row_from_result(spec, result))
        runs[spec.candidate_id] = {
            "spec": spec.__dict__,
            "config": config.__dict__,
            "result": result,
        }

    cp153_after = cp153_artifact_state()
    judgement = judgement_from_rows(rows, stage1["baseline_sota"])
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage0": stage0,
        "stage1": stage1,
        "stage2": {
            "candidate_rows": rows,
            "runs": runs,
            "judgement": judgement,
            "stage3_executed": False,
            "stage4_executed": False,
            "stage5_executed": False,
            "tcn_excluded": True,
        },
        "provider": PROVIDER,
        "source": PROVIDER,
        "source_data_hash": source_hash,
        "feature_set": FEATURE_SET,
        "model_role": MODEL_ROLE,
        "output_role": OUTPUT_ROLE,
        "band_output_used": False,
        "composite_used": False,
        "db_write": False,
        "save_run": False,
        "inference_saved": False,
        "cp153_artifact_guard": {
            "before": cp153_before,
            "after": cp153_after,
            "changed": cp153_before != cp153_after,
        },
    }
    _write_json(METRICS_PATH, metrics)
    write_stage2_outputs(rows, metrics, judgement)
    print(
        json.dumps(
            {
                "status": "CP158_STAGE0_2_DONE",
                "stage0_report": str(STAGE0_REPORT),
                "stage1_report": str(STAGE1_REPORT),
                "stage2_report": str(STAGE2_REPORT),
                "metrics": str(METRICS_PATH),
                "summary_csv": str(STAGE2_SUMMARY),
                "judgement": judgement["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP158 line_regime Stage 0~2 runner")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="기존 체크포인트를 재학습 없이 재평가해 보강 지표를 갱신합니다.",
    )
    args = parser.parse_args()
    if args.refresh_existing:
        refresh_existing_outputs()
    else:
        main()
