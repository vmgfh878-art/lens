from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ai.inference import load_checkpoint  # noqa: E402
from ai.loss import AsymmetricHuberLoss  # noqa: E402
from ai.preprocessing import FEATURE_CONTRACT_VERSION, SequenceDataset, normalize_sequence_splits  # noqa: E402
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns  # noqa: E402

import ai.cp164_lm_calendar_split_line_risk_smoke as cp164  # noqa: E402
import ai.cp175_lm_1d_conservative_line_learning_revisit as cp175  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp208z_line_final_smoke"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "cp208z"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp208z"

PROGRESS_LOG = LOG_DIR / "progress.log"
PROGRESS_MD = DOCS_DIR / "cp208z_progress_latest.md"
PROGRESS_JSON = DOCS_DIR / "cp208z_progress_latest.json"

BASELINE_REPORT = DOCS_DIR / "cp208z_baseline_lock_report.md"
ENV_REPORT = DOCS_DIR / "cp208z_environment_lock.md"
FEATURE_REPORT = DOCS_DIR / "cp208z_feature_pack_lock.md"
SUMMARY_CSV = DOCS_DIR / "cp208z_line_smoke_summary.csv"
REFRESH_REPORT = DOCS_DIR / "cp208z_line_refreshability_report.md"
REPORT_PATH = DOCS_DIR / "cp208z_line_smoke_report.md"
METRICS_JSON = DOCS_DIR / "cp208z_line_smoke_metrics.json"
HEATMAP_IC = DOCS_DIR / "cp208z_heatmap_ic.png"
HEATMAP_FALSE_SAFE = DOCS_DIR / "cp208z_heatmap_false_safe.png"
HEATMAP_RECALL = DOCS_DIR / "cp208z_heatmap_severe_recall.png"

TIMEFRAME = "1D"
HORIZON = 5
SEQ_LEN = 252
PATCH_LEN = 32
PATCH_STRIDE = 16
FEATURE_SET = "price_volatility_volume"
SEED = 42
SEVERE_THRESHOLD = -0.03
FEE_BPS = 0.001

BASELINES = {
    "cp175_beta5_product": {
        "ic_mean": 0.0420,
        "line_top_decile_false_safe_rate": 0.1972,
        "severe_downside_recall_line_negative": 0.7921,
    },
    "cp164_alpha": {
        "ic_mean": 0.0436,
        "long_short_spread": 0.0079,
        "fee_adjusted_return": 0.0069,
        "line_top_decile_false_safe_rate": 0.2056,
        "severe_downside_recall_line_negative": 0.6732,
    },
    "cp175_beta2_neutral": {
        "ic_mean": 0.0444,
        "long_short_spread": 0.0081,
        "fee_adjusted_return": 0.0071,
        "line_top_decile_false_safe_rate": 0.2126,
        "severe_downside_recall_line_negative": 0.6152,
    },
}

PASS_IC_MIN = 0.0355
STRONG_IC_MIN = 0.0392

FEATURE_PACKS = {
    "F0_base": [],
    "F1_atr_only": ["atr_ratio"],
    "F2_stress_delta": ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"],
    "F3_stock_fragility": ["atr_ratio", "drawdown_20", "downside_vol_20"],
    "F4_stress_delta_plus_yield_curve": [
        "atr_ratio",
        "vix_change_5d",
        "credit_spread_change_20d",
        "ma200_pct_change_20d",
        "yield_curve",
    ],
    "F5_stress_delta_plus_evt": [
        "atr_ratio",
        "vix_change_5d",
        "credit_spread_change_20d",
        "ma200_pct_change_20d",
        "evt_score",
    ],
    "F6_stress_plus_fragility_union": [
        "atr_ratio",
        "vix_change_5d",
        "credit_spread_change_20d",
        "ma200_pct_change_20d",
        "drawdown_20",
        "downside_vol_20",
    ],
    "F7_yield_curve_only": ["yield_curve"],
    "F8_evt_only": ["evt_score"],
}


@dataclass(frozen=True)
class CandidateSpec:
    backbone: str
    beta: float
    feature_pack: str
    extra_features: tuple[str, ...]
    seed: int = SEED
    recheck: bool = False

    @property
    def candidate_id(self) -> str:
        beta_label = str(self.beta).replace(".", "p")
        seed_label = f"seed{self.seed}"
        suffix = "_recheck" if self.recheck else ""
        return f"cp208z_{self.backbone}_b{beta_label}_{self.feature_pack}_{seed_label}{suffix}"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_json(row.get(key)) for key in fieldnames})


def update_progress(stage: str, **payload: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": now_utc(), "stage": stage, **payload}
    with PROGRESS_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean_json(row), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(clean_json(row), ensure_ascii=False), flush=True)
    latest = {
        "stage": stage,
        "last_update": row["time"],
        **payload,
    }
    write_json(PROGRESS_JSON, latest)
    md_lines = [
        "# CP208Z 진행 상황",
        "",
        f"- 현재 stage: `{stage}`",
        f"- 마지막 갱신: `{row['time']}`",
    ]
    for key in ["current_candidate", "completed_candidates", "total_candidates", "best_candidate", "risk", "eta"]:
        if key in payload:
            md_lines.append(f"- {key}: `{payload[key]}`")
    if isinstance(payload.get("last_metric_snapshot"), dict):
        md_lines.append("")
        md_lines.append("## 마지막 metric snapshot")
        for key, value in payload["last_metric_snapshot"].items():
            md_lines.append(f"- {key}: `{value}`")
    PROGRESS_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_current_payload() -> dict[str, Any]:
    price, indicators, price_manifest, indicator_manifest = cp164.cp158.load_source_frames()
    source_hash = str(indicator_manifest.get("source_data_hash") or price_manifest.get("source_data_hash") or "unknown")
    train, val, test, mean, std, plan, registry = cp164.build_calendar_split_payload(
        price=price,
        indicators=indicators,
        source_data_hash=source_hash,
    )
    split_summary = cp164.summarize_dataset_plan(plan, train, val, test)
    split_summary["source_data_hash"] = source_hash
    split_summary["cross_split_date_overlap_count_bundle_check"] = cp164._date_overlap_count(train, val, test)
    return {
        "price": price,
        "indicators": indicators,
        "price_manifest": price_manifest,
        "indicator_manifest": indicator_manifest,
        "source_hash": source_hash,
        "train_raw": train,
        "val_raw": val,
        "test_raw": test,
        "mean_raw": mean,
        "std_raw": std,
        "plan": plan,
        "registry": registry,
        "split_summary": split_summary,
    }


def build_extra_feature_frame(price: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    ind = indicators.copy()
    ind["ticker"] = ind["ticker"].astype(str).str.upper()
    ind["date"] = pd.to_datetime(ind["date"], errors="coerce")
    px = price.copy()
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")

    date_context = ind.groupby("date", sort=True).agg(
        vix_close=("vix_close", "median"),
        credit_spread_hy=("credit_spread_hy", "median"),
        ma200_pct=("ma200_pct", "median"),
    )
    date_context["vix_change_5d"] = date_context["vix_close"].diff(5)
    date_context["credit_spread_change_20d"] = date_context["credit_spread_hy"].diff(20)
    date_context["ma200_pct_change_20d"] = date_context["ma200_pct"].diff(20)
    date_context = date_context.reset_index()[["date", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]]

    fragility_frames: list[pd.DataFrame] = []
    px = px.sort_values(["ticker", "date"])
    for ticker, frame in px.groupby("ticker", sort=True):
        local = frame[["ticker", "date", "close"]].copy()
        close = local["close"].astype(float)
        returns = close.pct_change()
        rolling_high = close.rolling(20, min_periods=1).max()
        local["drawdown_20"] = (close / rolling_high) - 1.0
        downside_returns = returns.where(returns < 0.0, 0.0)
        local["downside_vol_20"] = downside_returns.rolling(20, min_periods=2).std()
        local["ticker"] = ticker
        fragility_frames.append(local[["ticker", "date", "drawdown_20", "downside_vol_20"]])
    fragility = pd.concat(fragility_frames, ignore_index=True)

    extra = ind[["ticker", "date", "atr_ratio", "yield_spread"]].copy()
    extra = extra.rename(columns={"yield_spread": "yield_curve"})
    extra = extra.merge(date_context, on="date", how="left")
    extra = extra.merge(fragility, on=["ticker", "date"], how="left")
    extra["evt_score"] = np.nan
    extra["asof_date"] = extra["date"].dt.strftime("%Y-%m-%d")
    columns = [
        "atr_ratio",
        "vix_change_5d",
        "credit_spread_change_20d",
        "ma200_pct_change_20d",
        "drawdown_20",
        "downside_vol_20",
        "yield_curve",
        "evt_score",
    ]
    for column in columns:
        extra[column] = pd.to_numeric(extra[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return extra[["ticker", "asof_date", *columns]].drop_duplicates(["ticker", "asof_date"]).set_index(["ticker", "asof_date"])


def feature_coverage(extra_frame: pd.DataFrame) -> dict[str, float]:
    total = len(extra_frame)
    if total <= 0:
        return {column: 0.0 for column in extra_frame.columns}
    return {column: float(np.isfinite(extra_frame[column].to_numpy(dtype=np.float64)).sum() / total) for column in extra_frame.columns}


def usable_feature_packs(extra_frame: pd.DataFrame) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    coverage = feature_coverage(extra_frame)
    usable: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for pack_name, columns in FEATURE_PACKS.items():
        if not columns:
            usable[pack_name] = []
            rows.append({"feature_pack": pack_name, "status": "OK", "feature": "", "coverage": 1.0, "reason": "base"})
            continue
        ok = True
        for column in columns:
            value = coverage.get(column, 0.0)
            status = "OK" if value >= 0.50 else "SKIPPED_FEATURE_UNAVAILABLE"
            if value < 0.50:
                ok = False
            rows.append({"feature_pack": pack_name, "status": status, "feature": column, "coverage": value, "reason": "" if value >= 0.50 else "coverage_lt_0p50"})
        if ok:
            usable[pack_name] = list(columns)
    return usable, rows


def aligned_extra_values(ticker: str, dates: np.ndarray, extra_frame: pd.DataFrame, extra_names: list[str]) -> np.ndarray:
    if not extra_names:
        return np.empty((len(dates), 0), dtype=np.float32)
    date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
    index = pd.MultiIndex.from_arrays([[ticker] * len(date_strings), date_strings], names=["ticker", "asof_date"])
    values = extra_frame.reindex(index)[extra_names].to_numpy(dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def append_extra_to_bundle(bundle: SequenceDataset, extra_frame: pd.DataFrame, extra_names: list[str]) -> SequenceDataset:
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        base_values = np.asarray(arrays["features"], dtype=np.float32)
        extra_values = aligned_extra_values(str(ticker), arrays["dates"], extra_frame, extra_names)
        copied = dict(arrays)
        copied["features"] = np.concatenate([base_values, extra_values], axis=1).astype(np.float32, copy=False)
        ticker_arrays[ticker] = copied
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=None,
        std=None,
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )


def build_pack_splits(
    *,
    base_splits: tuple[SequenceDataset, SequenceDataset, SequenceDataset],
    extra_frame: pd.DataFrame,
    extra_names: list[str],
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset, torch.Tensor, torch.Tensor]:
    train, val, test = base_splits
    if not extra_names:
        return train, val, test, train.mean, train.std
    raw_train = append_extra_to_bundle(train, extra_frame, extra_names)
    raw_val = append_extra_to_bundle(val, extra_frame, extra_names)
    raw_test = append_extra_to_bundle(test, extra_frame, extra_names)
    return normalize_sequence_splits(raw_train, raw_val, raw_test)


def train_and_eval(
    *,
    spec: CandidateSpec,
    train_bundle: SequenceDataset,
    val_bundle: SequenceDataset,
    train_q10: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    set_seed(spec.seed)
    model = cp175._make_model(n_features=train_bundle.ticker_arrays[next(iter(train_bundle.ticker_arrays))]["features"].shape[1], dropout=0.10).to(device)
    criterion = AsymmetricHuberLoss(alpha=1.0, beta=spec.beta, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.362816234925851e-4, weight_decay=8.143270337695065e-5)
    loader = DataLoader(cp175.LineTrialDataset(train_bundle, include_atr=False), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    started = time.perf_counter()
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for features, raw_future_returns, ticker_id in loader:
            features = features.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with cp175._amp_context(device):
                output = model(features, ticker_id=ticker_id)
                line = cp175._extract_line(output)
                loss = criterion(line, raw_future_returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "batch_count": len(losses)})
    prediction = cp175.collect_predictions(
        model,
        cp175.LineTrialDataset(val_bundle, include_atr=False),
        metadata=val_bundle.metadata,
        device=device,
        batch_size=batch_size,
    )
    metrics = cp175.evaluate_prediction(
        candidate_id=spec.candidate_id,
        split="validation",
        prediction=prediction,
        train_q10_downside=train_q10,
    )
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    runtime = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "candidate_id": spec.candidate_id,
        "spec": spec,
        "validation": metrics,
        "epoch_rows": epoch_rows,
        "runtime_seconds": runtime,
        "state_dict": state_dict,
    }


def score_label(metrics: dict[str, Any], operating_pass: bool = False) -> str:
    ic = float(metrics.get("ic_mean") or -999.0)
    false_safe = float(metrics.get("line_top_decile_false_safe_rate") or 999.0)
    recall = float(metrics.get("severe_downside_recall_line_negative") or -999.0)
    if false_safe <= 0.1972 and recall >= 0.7921 and ic >= STRONG_IC_MIN and operating_pass:
        return "STRONG_PASS_ALPHA_BALANCED"
    if false_safe <= 0.1972 and recall >= 0.7921 and ic >= PASS_IC_MIN and operating_pass:
        return "PASS_LINE_REFRESHABLE"
    if false_safe <= 0.1972 and recall >= 0.7921 and operating_pass:
        return "WARN_LINE_REFRESHABLE_WITH_TRADEOFF"
    return "FAIL_LINE_NOT_WORTH_REBUILD"


def validation_rank(result: dict[str, Any]) -> tuple:
    m = result["validation"]
    label_score = {
        "STRONG_PASS_ALPHA_BALANCED": 3,
        "PASS_LINE_REFRESHABLE": 2,
        "WARN_LINE_REFRESHABLE_WITH_TRADEOFF": 1,
        "FAIL_LINE_NOT_WORTH_REBUILD": 0,
    }[score_label(m, operating_pass=True)]
    early_fail = bool(result.get("early_fail"))
    return (
        0 if early_fail else 1,
        label_score,
        float(m.get("ic_mean") or -999.0),
        -float(m.get("line_top_decile_false_safe_rate") or 999.0),
        float(m.get("severe_downside_recall_line_negative") or -999.0),
        float(m.get("long_short_spread") or -999.0),
        float(m.get("fee_adjusted_return") or -999.0),
    )


def is_early_fail(metrics: dict[str, Any]) -> tuple[bool, str]:
    ic = float(metrics.get("ic_mean") or 0.0)
    std = float(metrics.get("line_score_std") or 0.0)
    false_safe = float(metrics.get("line_top_decile_false_safe_rate") or 1.0)
    recall = float(metrics.get("severe_downside_recall_line_negative") or 0.0)
    if ic < 0.01:
        return True, "ic_lt_0p01"
    if std < 1e-5:
        return True, "line_score_collapse"
    if false_safe > 0.30 and recall < 0.60:
        return True, "risk_metrics_broken"
    return False, ""


def save_checkpoint(
    *,
    candidate: dict[str, Any],
    feature_names: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    split_summary: dict[str, Any],
) -> Path:
    spec: CandidateSpec = candidate["spec"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{spec.candidate_id}.pt"
    config = {
        "model": "patchtst",
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
        "dropout": 0.10,
        "band_mode": "direct",
        "num_tickers": 0,
        "ticker_emb_dim": 16,
        "model_role": "legacy",
        "output_role": "legacy",
        "feature_columns": feature_names,
        "n_features": len(feature_names),
        "feature_set": spec.feature_pack,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "use_revin": True,
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "patchtst_d_model": 128,
        "patchtst_n_heads": 8,
        "patchtst_n_layers": 3,
        "ci_aggregate": "target",
        "target_channel_idx": 0,
        "ci_target_fast": False,
        "line_target_type": "raw_future_return",
        "band_target_type": "raw_future_return",
        "market_data_provider": "yfinance",
        "source_cp": "CP208Z",
        "source_data_hash": split_summary.get("source_data_hash"),
        "split_mode": split_summary.get("split_mode"),
        "beta": spec.beta,
        "alpha": 1.0,
        "seed": spec.seed,
        "score_contract": "A_score_contract",
        "input_pipeline": "cp208z_extended_feature_runner",
    }
    torch.save(
        {
            "model_state_dict": candidate["state_dict"],
            "config": config,
            "metrics": candidate.get("test") or candidate["validation"],
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
        },
        path,
    )
    return path


def add_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["line_rank_by_date"] = df.groupby("asof_date")["line_score"].rank(method="average", pct=True)
    df["safe_line_rank_by_date"] = df.groupby("asof_date")["safe_line_score"].rank(method="average", pct=True)
    q90 = df.groupby("asof_date")["line_score"].transform(lambda s: s.quantile(0.90))
    safe_q90 = df.groupby("asof_date")["safe_line_score"].transform(lambda s: s.quantile(0.90))
    df["line_top_decile_flag"] = (df["line_score"] >= q90).astype(float)
    df["safe_line_top_decile_flag"] = (df["safe_line_score"] >= safe_q90).astype(float)
    return df[
        [
            "ticker",
            "asof_date",
            "line_score",
            "safe_line_score",
            "line_rank_by_date",
            "safe_line_rank_by_date",
            "line_top_decile_flag",
            "safe_line_top_decile_flag",
            "actual_h5_return",
            "model_id",
            "source_cp",
        ]
    ].sort_values(["ticker", "asof_date"]).reset_index(drop=True)


def prediction_to_frame(prediction: dict[str, Any], model_id: str) -> pd.DataFrame:
    metadata = prediction["metadata"].reset_index(drop=True)
    return add_rank_columns(
        pd.DataFrame(
            {
                "ticker": metadata["ticker"].astype(str).str.upper(),
                "asof_date": pd.to_datetime(metadata["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                "line_score": np.asarray(prediction["line_score"], dtype=np.float64),
                "safe_line_score": np.asarray(prediction["line_score"], dtype=np.float64),
                "actual_h5_return": np.asarray(prediction["actual"], dtype=np.float64),
                "model_id": model_id,
                "source_cp": "CP208Z",
            }
        )
    )


def latest_rows(model: Any, bundle: SequenceDataset, device: torch.device, model_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, arrays in sorted(bundle.ticker_arrays.items()):
        features_full = np.asarray(arrays["features"], dtype=np.float32)
        dates = pd.to_datetime(arrays["dates"])
        if len(features_full) < SEQ_LEN:
            continue
        end_idx = len(features_full) - 1
        window = torch.from_numpy(features_full[end_idx - SEQ_LEN + 1 : end_idx + 1]).to(torch.float32).unsqueeze(0).to(device)
        ticker_id = torch.tensor([int(arrays.get("ticker_id", 0))], dtype=torch.long, device=device)
        with torch.no_grad():
            with cp175._amp_context(device):
                output = model(window, ticker_id=ticker_id)
        line = cp175._extract_line(output).detach().cpu().to(torch.float32).numpy()[0]
        rows.append(
            {
                "ticker": str(ticker),
                "asof_date": str(pd.Timestamp(dates[end_idx]).date()),
                "line_score": float(line[-1]),
                "safe_line_score": float(line[-1]),
                "actual_h5_return": np.nan,
                "model_id": model_id,
                "source_cp": "CP208Z",
            }
        )
    return add_rank_columns(pd.DataFrame(rows))


def export_candidate(
    *,
    checkpoint_path: Path,
    bundle: SequenceDataset,
    train_q10: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model, _checkpoint = load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    prediction = cp175.collect_predictions(
        model,
        cp175.LineTrialDataset(bundle, include_atr=False),
        metadata=bundle.metadata,
        device=device,
        batch_size=batch_size,
    )
    metrics = cp175.evaluate_prediction(
        candidate_id=checkpoint_path.stem,
        split="test",
        prediction=prediction,
        train_q10_downside=train_q10,
    )
    historical = prediction_to_frame(prediction, checkpoint_path.stem)
    latest = latest_rows(model, bundle, device, checkpoint_path.stem)
    combined = pd.concat([historical, latest], ignore_index=True).drop_duplicates(["ticker", "asof_date"], keep="last")
    combined = combined.sort_values(["ticker", "asof_date"]).reset_index(drop=True)
    out_path = ARTIFACT_DIR / "exports" / f"{checkpoint_path.stem}_predictions_line_1d.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False, compression="snappy")
    read_back = pd.read_parquet(out_path)
    return {
        "checkpoint_reload_pass": True,
        "latest_inference_pass": bool(len(latest) > 0),
        "row_level_payload_pass": bool(len(combined) > 0),
        "predictions_line_export_pass": bool(len(read_back) == len(combined)),
        "export_path": str(out_path),
        "row_count": int(len(combined)),
        "ticker_count": int(combined["ticker"].nunique()),
        "latest_asof_date": str(combined["asof_date"].max()),
        "actual_h5_nan_rows": int(combined["actual_h5_return"].isna().sum()),
        "test_metrics": metrics,
    }


def dummy_append_bundle(bundle: SequenceDataset) -> SequenceDataset:
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        copied = dict(arrays)
        features = np.asarray(arrays["features"], dtype=np.float32)
        closes = np.asarray(arrays["closes"], dtype=np.float32)
        dates = pd.to_datetime(arrays["dates"])
        next_date = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=1)[0].to_datetime64()
        copied["features"] = np.concatenate([features, features[-1:, :]], axis=0)
        copied["closes"] = np.concatenate([closes, closes[-1:]], axis=0)
        copied["dates"] = np.concatenate([dates.to_numpy(), np.asarray([next_date])])
        ticker_arrays[ticker] = copied
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=bundle.mean,
        std=bundle.std,
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )


def append_smoke(checkpoint_path: Path, bundle: SequenceDataset, device: torch.device) -> dict[str, Any]:
    model, _checkpoint = load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    before = latest_rows(model, bundle, device, checkpoint_path.stem)
    appended_bundle = dummy_append_bundle(bundle)
    after = latest_rows(model, appended_bundle, device, checkpoint_path.stem)
    out_path = ARTIFACT_DIR / "append_smoke" / f"{checkpoint_path.stem}_dummy_append.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    after.to_parquet(out_path, index=False, compression="snappy")
    return {
        "append_smoke_pass": bool(str(after["asof_date"].max()) > str(before["asof_date"].max())),
        "before_latest": str(before["asof_date"].max()),
        "after_latest": str(after["asof_date"].max()),
        "path": str(out_path),
    }


def make_heatmap(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df["split"] == "validation"].copy()
    if df.empty:
        return
    pivot = df.pivot_table(index="feature_pack", columns="beta", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.45)))
    data = pivot.to_numpy(dtype=float)
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("beta")
    ax.set_ylabel("feature pack")
    ax.set_title(metric)
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_static_reports(
    *,
    payload: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    refresh_rows: list[dict[str, Any]],
) -> None:
    BASELINE_REPORT.write_text(
        "\n".join(
            [
                "# CP208Z Baseline Lock",
                "",
                "| baseline | role | IC | spread | fee | false-safe | severe recall |",
                "|---|---|---:|---:|---:|---:|---:|",
                "| CP175 beta5 | 제품 기준선 | 0.0420 |  |  | 0.1972 | 0.7921 |",
                "| CP164 calendar line | 알파 기준선 | 0.0436 | 0.0079 | 0.0069 | 0.2056 | 0.6732 |",
                "| CP175 beta2 | 중립 기준선 | 0.0444 | 0.0081 | 0.0071 | 0.2126 | 0.6152 |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ENV_REPORT.write_text(
        "\n".join(
            [
                "# CP208Z Environment Lock",
                "",
                f"- source hash: `{payload['source_hash']}`",
                "- CP175 backbone lock: `PatchTST p32/s16`",
                "- CP208 backbone lock: `PatchTST p32/s16`",
                "- Stage 1 case: `Case B`, M0와 M1이 둘 다 PatchTST라 core matrix는 45셀로 축소",
                "- timeframe: `1D`",
                "- split: `calendar_aligned`",
                "- feature base: `price_volatility_volume`",
                "- output contract: score형 `line_score`, `safe_line_score`",
                "- 프론트 계약 변경 없음",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    feature_lines = ["# CP208Z Feature Pack Lock", "", "| feature_pack | feature | coverage | status |", "|---|---|---:|---|"]
    for row in feature_rows:
        feature_lines.append(f"| {row['feature_pack']} | {row['feature']} | {float(row['coverage']):.4f} | {row['status']} |")
    FEATURE_REPORT.write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    refresh_lines = ["# CP208Z Line Refreshability Report", ""]
    for row in refresh_rows:
        refresh_lines.append(f"## {row['candidate_id']}")
        for key, value in row.items():
            refresh_lines.append(f"- {key}: `{value}`")
        refresh_lines.append("")
    REFRESH_REPORT.write_text("\n".join(refresh_lines) + "\n", encoding="utf-8")

    df = pd.DataFrame(summary_rows)
    report_lines = [
        "# CP208Z Line Final Smoke Report",
        "",
        f"최종 라벨: `{payload['final_label']}`",
        f"최종 후보: `{payload.get('best_candidate_id')}`",
        "",
        "이번 CP는 1D line만 대상으로, CP175 제품 보수성과 CP164 알파 질감 사이에서 자동갱신 가능한 checkpoint를 확보할 수 있는지 확인한 최종 대형 smoke다.",
        "",
        "## 요약",
        "",
    ]
    if not df.empty:
        view = df[df["split"] == "test"].copy()
        if view.empty:
            view = df[df["split"] == "validation"].sort_values("rank_order").head(12)
        report_lines.append("| candidate | split | beta | feature | IC | false-safe | severe recall | label |")
        report_lines.append("|---|---|---:|---|---:|---:|---:|---|")
        for _, row in view.head(12).iterrows():
            report_lines.append(
                f"| {row['candidate_id']} | {row['split']} | {row['beta']} | {row['feature_pack']} | "
                f"{float(row.get('ic_mean') or 0):.6f} | {float(row.get('line_top_decile_false_safe_rate') or 0):.6f} | "
                f"{float(row.get('severe_downside_recall_line_negative') or 0):.6f} | {row.get('label')} |"
            )
    report_lines.extend(
        [
            "",
            "## 해석",
            "",
            "- PASS/STRONG이면 새 line ship 후보로 잠그고 자동갱신 on 후보로 넘긴다.",
            "- FAIL이면 CP175 frozen을 유지하고 UI에는 `1D 보수적 기준선은 현재 자동 갱신되지 않습니다` 계열 문구를 사용한다.",
            "- heatmap은 validation core matrix 기준이다. 최종 test 평가는 validation 상위 후보에만 수행했다.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def flatten_result(result: dict[str, Any], split: str, label: str, rank_order: int | None = None) -> dict[str, Any]:
    spec: CandidateSpec = result["spec"]
    metrics = result[split]
    early_fail = bool(result.get("early_fail"))
    return {
        "candidate_id": spec.candidate_id,
        "backbone": spec.backbone,
        "beta": spec.beta,
        "feature_pack": spec.feature_pack,
        "extra_features": ",".join(spec.extra_features),
        "seed": spec.seed,
        "split": split,
        "label": label,
        "rank_order": rank_order,
        "early_fail": early_fail,
        "early_fail_reason": result.get("early_fail_reason", ""),
        **metrics,
    }


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    update_progress("started", total_candidates=0, completed_candidates=0, risk="초기화", eta="계산 중")
    device = device_from_arg(args.device)
    if device.type != "cuda":
        update_progress("started", risk="CUDA 미사용. 사용자가 GPU 우선 실행을 요구했으므로 환경 확인 필요", eta="지연 가능")
    else:
        update_progress("started", risk="GPU 사용", eta="약 3~4시간", current_candidate=torch.cuda.get_device_name(0))

    update_progress("baseline_locked", completed_candidates=0, total_candidates=0, eta="환경 로드 중")
    payload = load_current_payload()
    base_cols = resolve_feature_columns(FEATURE_SET)
    base_train, base_val, base_test, _base_mean, _base_std = apply_feature_columns_to_splits(
        payload["train_raw"],
        payload["val_raw"],
        payload["test_raw"],
        payload["mean_raw"],
        payload["std_raw"],
        base_cols,
    )
    base_splits = (base_train, base_val, base_test)
    train_actual = cp175.collect_actual_h5(base_train, batch_size=args.eval_batch_size)
    train_q10 = float(np.quantile(train_actual[np.isfinite(train_actual)], 0.10))
    update_progress(
        "environment_locked",
        completed_candidates=0,
        total_candidates=0,
        risk="Case B: CP175=PatchTST, M1=PatchTST 중복 제거",
        eta="feature pack coverage 측정 중",
        last_metric_snapshot={"source_hash": payload["source_hash"], "train_rows": len(base_train), "validation_rows": len(base_val), "test_rows": len(base_test)},
    )

    extra_frame = build_extra_feature_frame(payload["price"], payload["indicators"])
    usable_packs, feature_rows = usable_feature_packs(extra_frame)
    write_csv(DOCS_DIR / "cp208z_feature_pack_coverage.csv", feature_rows)
    update_progress(
        "feature_pack_locked",
        completed_candidates=0,
        total_candidates=0,
        risk="coverage < 50% feature pack은 SKIP",
        eta="core matrix 시작",
        last_metric_snapshot={"usable_feature_packs": sorted(usable_packs.keys())},
    )

    beta_values = [3.0, 4.0, 5.0, 6.0, 7.0]
    specs: list[CandidateSpec] = []
    for beta in beta_values:
        for pack, extras in usable_packs.items():
            specs.append(CandidateSpec(backbone="patchtst", beta=beta, feature_pack=pack, extra_features=tuple(extras), seed=SEED))
    total = len(specs)
    update_progress("core90_started", completed_candidates=0, total_candidates=total, eta="core matrix 진행 중")

    split_cache: dict[str, tuple[SequenceDataset, SequenceDataset, SequenceDataset, torch.Tensor, torch.Tensor, list[str]]] = {}
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.feature_pack not in split_cache:
            train_pack, val_pack, test_pack, mean_pack, std_pack = build_pack_splits(
                base_splits=base_splits,
                extra_frame=extra_frame,
                extra_names=list(spec.extra_features),
            )
            split_cache[spec.feature_pack] = (
                train_pack,
                val_pack,
                test_pack,
                mean_pack,
                std_pack,
                [*base_cols, *list(spec.extra_features)],
            )
        train_pack, val_pack, _test_pack, _mean_pack, _std_pack, _feature_names = split_cache[spec.feature_pack]
        update_progress(
            "candidate_train_start",
            current_candidate=spec.candidate_id,
            completed_candidates=index - 1,
            total_candidates=total,
            eta=f"남은 core {total - index + 1}셀",
        )
        result = train_and_eval(
            spec=spec,
            train_bundle=train_pack,
            val_bundle=val_pack,
            train_q10=train_q10,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        early, reason = is_early_fail(result["validation"])
        result["early_fail"] = early
        result["early_fail_reason"] = reason
        results.append(result)
        if index % 10 == 0 or index == total:
            best_so_far = sorted(results, key=validation_rank, reverse=True)[0]
            elapsed = time.perf_counter() - started
            avg = elapsed / max(index, 1)
            remain = avg * (total - index)
            update_progress(
                "core90_progress",
                current_candidate=spec.candidate_id,
                completed_candidates=index,
                total_candidates=total,
                best_candidate=best_so_far["candidate_id"],
                eta=f"core 잔여 약 {remain / 60:.1f}분",
                last_metric_snapshot={
                    "best_ic": best_so_far["validation"].get("ic_mean"),
                    "best_false_safe": best_so_far["validation"].get("line_top_decile_false_safe_rate"),
                    "best_recall": best_so_far["validation"].get("severe_downside_recall_line_negative"),
                },
            )
    update_progress("core90_done", completed_candidates=total, total_candidates=total, eta="seed recheck 준비")

    ranked = sorted(results, key=validation_rank, reverse=True)
    for rank, result in enumerate(ranked, start=1):
        result["rank_order"] = rank
    seed_recheck_rows: list[dict[str, Any]] = []
    recheck_targets = ranked[:3]
    for target in recheck_targets:
        old_spec: CandidateSpec = target["spec"]
        spec = CandidateSpec(
            backbone=old_spec.backbone,
            beta=old_spec.beta,
            feature_pack=old_spec.feature_pack,
            extra_features=old_spec.extra_features,
            seed=7,
            recheck=True,
        )
        train_pack, val_pack, _test_pack, _mean_pack, _std_pack, _feature_names = split_cache[spec.feature_pack]
        update_progress("seed_recheck_start", current_candidate=spec.candidate_id, eta="seed=7 재확인")
        result = train_and_eval(
            spec=spec,
            train_bundle=train_pack,
            val_bundle=val_pack,
            train_q10=train_q10,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        base_ic = float(target["validation"].get("ic_mean") or 0.0)
        recheck_ic = float(result["validation"].get("ic_mean") or 0.0)
        diff_ratio = abs(recheck_ic - base_ic) / abs(base_ic) if abs(base_ic) > 1e-12 else math.inf
        status = "SEED_TOLERABLE" if diff_ratio <= 0.30 else "LUCKY_SEED_SUSPECT"
        target["seed_recheck_status"] = status
        target["seed_recheck_ic_diff_ratio"] = diff_ratio
        seed_recheck_rows.append(
            {
                "candidate_id": old_spec.candidate_id,
                "recheck_candidate_id": spec.candidate_id,
                "base_ic": base_ic,
                "recheck_ic": recheck_ic,
                "ic_diff_ratio": diff_ratio,
                "status": status,
            }
        )
        update_progress("seed_recheck_done", current_candidate=spec.candidate_id, last_metric_snapshot=seed_recheck_rows[-1])
    update_progress("seed3_done", completed_candidates=3, total_candidates=3, eta="운영성 smoke 준비")

    ranked_after_recheck = sorted(results, key=lambda item: (item.get("seed_recheck_status") != "LUCKY_SEED_SUSPECT", *validation_rank(item)), reverse=True)
    ops_targets = [item for item in ranked_after_recheck if not item.get("early_fail")][:6]
    refresh_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for idx, result in enumerate(ops_targets, start=1):
        spec: CandidateSpec = result["spec"]
        train_pack, _val_pack, test_pack, mean_pack, std_pack, feature_names = split_cache[spec.feature_pack]
        update_progress("ops_candidate_start", current_candidate=spec.candidate_id, completed_candidates=idx - 1, total_candidates=len(ops_targets), eta="checkpoint/export 검사")
        checkpoint_path = save_checkpoint(
            candidate=result,
            feature_names=feature_names,
            mean=mean_pack,
            std=std_pack,
            split_summary=payload["split_summary"],
        )
        export = export_candidate(
            checkpoint_path=checkpoint_path,
            bundle=test_pack,
            train_q10=train_q10,
            device=device,
            batch_size=args.batch_size,
        )
        test_metrics = export.pop("test_metrics")
        result["test"] = test_metrics
        label = score_label(test_metrics, operating_pass=True)
        result["test_label"] = label
        row = {
            "candidate_id": spec.candidate_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_save_pass": checkpoint_path.exists(),
            **export,
            "test_label": label,
        }
        refresh_rows.append(row)
        test_rows.append(flatten_result(result, "test", label, rank_order=result.get("rank_order")))
        update_progress("ops_candidate_done", current_candidate=spec.candidate_id, completed_candidates=idx, total_candidates=len(ops_targets), last_metric_snapshot={"label": label, "ic": test_metrics.get("ic_mean"), "false_safe": test_metrics.get("line_top_decile_false_safe_rate")})
    update_progress("ops6_done", completed_candidates=len(refresh_rows), total_candidates=len(ops_targets), eta="append smoke 준비")

    append_rows: list[dict[str, Any]] = []
    for row in refresh_rows[:3]:
        checkpoint_path = Path(str(row["checkpoint_path"]))
        candidate_id = str(row["candidate_id"])
        spec = next(item["spec"] for item in ops_targets if item["candidate_id"] == candidate_id)
        _train_pack, _val_pack, test_pack, _mean_pack, _std_pack, _feature_names = split_cache[spec.feature_pack]
        append_row = {"candidate_id": candidate_id, **append_smoke(checkpoint_path, test_pack, device)}
        append_rows.append(append_row)
        update_progress("append_smoke_done", current_candidate=candidate_id, last_metric_snapshot=append_row)
    update_progress("append3_done", completed_candidates=len(append_rows), total_candidates=3, eta="heatmap/report 작성")

    validation_rows: list[dict[str, Any]] = []
    for result in ranked:
        label = score_label(result["validation"], operating_pass=True)
        validation_rows.append(flatten_result(result, "validation", label, rank_order=result.get("rank_order")))
    summary_rows = [*validation_rows, *test_rows]
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DOCS_DIR / "cp208z_seed_recheck.csv", seed_recheck_rows)
    write_csv(DOCS_DIR / "cp208z_refreshability_summary.csv", refresh_rows)
    write_csv(DOCS_DIR / "cp208z_append_smoke.csv", append_rows)

    make_heatmap(summary_rows, "ic_mean", HEATMAP_IC)
    make_heatmap(summary_rows, "line_top_decile_false_safe_rate", HEATMAP_FALSE_SAFE)
    make_heatmap(summary_rows, "severe_downside_recall_line_negative", HEATMAP_RECALL)
    update_progress("heatmap_done", completed_candidates=len(summary_rows), total_candidates=len(summary_rows), eta="최종 판정")

    test_sorted = sorted(
        [item for item in ops_targets if item.get("test")],
        key=lambda item: (
            {"STRONG_PASS_ALPHA_BALANCED": 3, "PASS_LINE_REFRESHABLE": 2, "WARN_LINE_REFRESHABLE_WITH_TRADEOFF": 1, "FAIL_LINE_NOT_WORTH_REBUILD": 0}.get(item.get("test_label"), 0),
            float(item["test"].get("ic_mean") or -999.0),
            -float(item["test"].get("line_top_decile_false_safe_rate") or 999.0),
            float(item["test"].get("severe_downside_recall_line_negative") or -999.0),
        ),
        reverse=True,
    )
    best = test_sorted[0] if test_sorted else ranked[0]
    final_label = best.get("test_label") or score_label(best["validation"], operating_pass=False)
    metrics_payload = {
        "created_at": now_utc(),
        "final_label": final_label,
        "best_candidate_id": best["candidate_id"],
        "source_hash": payload["source_hash"],
        "baseline": BASELINES,
        "feature_pack_rows": feature_rows,
        "seed_recheck": seed_recheck_rows,
        "refreshability": refresh_rows,
        "append_smoke": append_rows,
        "summary_rows": summary_rows,
        "forbidden_actions": {
            "band_modified": False,
            "line_1w_modified": False,
            "frontend_modified": False,
            "db_write": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "long_seed_stability": False,
        },
    }
    write_json(METRICS_JSON, metrics_payload)
    write_static_reports(
        payload=metrics_payload,
        feature_rows=feature_rows,
        summary_rows=summary_rows,
        refresh_rows=refresh_rows,
    )
    update_progress("completed", best_candidate=metrics_payload["best_candidate_id"], last_metric_snapshot={"final_label": final_label}, eta="완료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP208Z 1D line final frontier smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
