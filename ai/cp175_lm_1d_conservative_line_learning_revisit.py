from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
import os
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ai.loss import AsymmetricHuberLoss  # noqa: E402
from ai.models.common import ForecastOutput, LineRegimeOutput, LineV2Output  # noqa: E402
from ai.models.patchtst import PatchTST  # noqa: E402
from ai.preprocessing import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS  # noqa: E402
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns  # noqa: E402

import ai.cp160_lm_1d_line_overlay_rejudgement as cp160  # noqa: E402
import ai.cp164_lm_calendar_split_line_risk_smoke as cp164  # noqa: E402
import ai.cp171_lm_1d_line_warning_stage0_2_smoke as cp171  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp175_lm_1d_conservative_line_learning_revisit"

REPORT_PATH = DOCS_DIR / "cp175_lm_1d_conservative_line_learning_revisit_report.md"
METRICS_PATH = DOCS_DIR / "cp175_lm_1d_conservative_line_learning_revisit_metrics.json"
SUMMARY_PATH = DOCS_DIR / "cp175_lm_1d_conservative_line_learning_revisit_summary.csv"
BETA_ONLY_PATH = DOCS_DIR / "cp175_lm_1d_beta_only_sweep.csv"
ATR_ONLY_PATH = DOCS_DIR / "cp175_lm_1d_atr_only_result.csv"
BETA_ATR_PATH = DOCS_DIR / "cp175_lm_1d_beta_atr_combo.csv"
PARETO_PATH = DOCS_DIR / "cp175_lm_1d_pareto_frontier.csv"
COLLAPSE_PATH = DOCS_DIR / "cp175_lm_1d_line_collapse_diagnostic.csv"

SOURCE_HASH_EXPECTED = "90666b44cbfb8e5c"
TIMEFRAME = "1D"
HORIZON = 5
SEQ_LEN = 252
PATCH_LEN = 32
PATCH_STRIDE = 16
FEATURE_SET = "price_volatility_volume"
SEVERE_THRESHOLD = -0.03
FEE_BPS = 0.001
SEED = 42


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    beta: float
    include_atr: bool
    stage: str


TRIALS = [
    TrialSpec("beta_2_baseline", 2.0, False, "beta_only"),
    TrialSpec("beta_3", 3.0, False, "beta_only"),
    TrialSpec("beta_5", 5.0, False, "beta_only"),
    TrialSpec("beta_7", 7.0, False, "beta_only"),
    TrialSpec("beta_2_plus_atr_ratio", 2.0, True, "atr_only"),
    TrialSpec("beta_3_plus_atr_ratio", 3.0, True, "beta_atr_combo"),
    TrialSpec("beta_5_plus_atr_ratio", 5.0, True, "beta_atr_combo"),
    TrialSpec("beta_7_plus_atr_ratio", 7.0, True, "beta_atr_combo"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_json(row.get(key)) for key in fieldnames})


def _log(stage: str, **payload: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": _now(), "stage": stage, **payload}
    print(json.dumps(_clean_json(row), ensure_ascii=False), flush=True)
    with (LOG_DIR / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean_json(row), ensure_ascii=False) + "\n")


def _safe_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else None


def _safe_std(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=1)) if len(arr) > 1 else None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LineTrialDataset(Dataset):
    def __init__(
        self,
        base: Any,
        *,
        include_atr: bool,
        atr_values: np.ndarray | None = None,
        atr_mean: float = 0.0,
        atr_std: float = 1.0,
    ) -> None:
        self.base = base
        self.include_atr = include_atr
        self.atr_values = atr_values
        self.atr_mean = float(atr_mean)
        self.atr_std = float(atr_std if abs(atr_std) > 1e-12 else 1.0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        features, _line_target, _band_target, raw_future_returns, ticker_id, _future_covariate = self.base[index]
        if self.include_atr:
            if self.atr_values is None:
                raise RuntimeError("ATR feature가 요청됐지만 atr_values가 없습니다.")
            atr_value = (float(self.atr_values[index]) - self.atr_mean) / self.atr_std
            atr_column = torch.full((features.shape[0], 1), atr_value, dtype=features.dtype)
            features = torch.cat((features, atr_column), dim=1)
        return features, raw_future_returns, ticker_id


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _extract_line(output: Any) -> torch.Tensor:
    if isinstance(output, (ForecastOutput, LineRegimeOutput, LineV2Output)):
        return output.line
    if hasattr(output, "line"):
        return output.line
    raise TypeError(f"line 출력을 찾을 수 없습니다: {type(output).__name__}")


def _make_model(*, n_features: int, dropout: float) -> PatchTST:
    return PatchTST(
        n_features=n_features,
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        stride=PATCH_STRIDE,
        d_model=128,
        n_heads=8,
        n_layers=3,
        horizon=HORIZON,
        dropout=dropout,
        band_mode="direct",
        use_revin=True,
        channel_independent=True,
        target_channel_idx=0,
        ci_aggregate="target",
        ci_target_fast=False,
        num_tickers=0,
        ticker_emb_dim=16,
        output_role="legacy",
    )


def collect_actual_h5(bundle: Any, *, batch_size: int = 2048) -> np.ndarray:
    loader = DataLoader(LineTrialDataset(bundle, include_atr=False), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[torch.Tensor] = []
    for _features, raw_future_returns, _ticker_id in loader:
        chunks.append(raw_future_returns[:, -1].detach().cpu())
    return torch.cat(chunks, dim=0).to(torch.float32).numpy()


def collect_predictions(
    model: PatchTST,
    dataset: LineTrialDataset,
    *,
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    line_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for features, raw_future_returns, ticker_id in loader:
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            with _amp_context(device):
                output = model(features, ticker_id=ticker_id)
            line_chunks.append(_extract_line(output).detach().cpu().to(torch.float32))
            raw_chunks.append(raw_future_returns.detach().cpu().to(torch.float32))
    line = torch.cat(line_chunks, dim=0).numpy()
    raw = torch.cat(raw_chunks, dim=0).numpy()
    return {
        "line_score": line[:, -1],
        "actual": raw[:, -1],
        "line_path": line,
        "actual_path": raw,
        "metadata": metadata.reset_index(drop=True).copy(),
    }


def datewise_top_bottom_masks(metadata: pd.DataFrame, line_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = metadata[["asof_date"]].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["line_score"] = np.asarray(line_score, dtype=np.float64)
    top = np.zeros(len(frame), dtype=bool)
    bottom = np.zeros(len(frame), dtype=bool)
    for _date, index in frame.groupby("date", sort=False).groups.items():
        values = frame.loc[index, "line_score"].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if int(finite.sum()) < 10:
            continue
        q90 = float(np.quantile(values[finite], 0.90))
        q10 = float(np.quantile(values[finite], 0.10))
        idx = np.asarray(list(index), dtype=np.int64)
        top[idx] = values >= q90
        bottom[idx] = values <= q10
    return top, bottom


def spearman_ic_by_date(line_score: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    frame = metadata[["asof_date"]].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["line_score"] = np.asarray(line_score, dtype=np.float64)
    frame["actual"] = np.asarray(actual, dtype=np.float64)
    values: list[float] = []
    for _date, group in frame.groupby("date", sort=False):
        group = group[np.isfinite(group["line_score"]) & np.isfinite(group["actual"])]
        if len(group) < 10:
            continue
        corr = group["line_score"].rank(method="average").corr(group["actual"].rank(method="average"))
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    arr = np.asarray(values, dtype=np.float64)
    ic_mean = float(arr.mean()) if len(arr) else None
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else None
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": None if ic_mean is None or ic_std in (None, 0.0) else float(ic_mean / ic_std),
        "ic_t_stat": None if ic_mean is None or ic_std in (None, 0.0) else float(ic_mean / (ic_std / math.sqrt(len(arr)))),
        "ic_observation_count": int(len(arr)),
    }


def datewise_long_short_spread(line_score: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    top, bottom = datewise_top_bottom_masks(metadata, line_score)
    frame = metadata[["asof_date"]].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["top"] = top
    frame["bottom"] = bottom
    frame["actual"] = np.asarray(actual, dtype=np.float64)
    spreads: list[float] = []
    for _date, group in frame.groupby("date", sort=False):
        top_mean = group.loc[group["top"], "actual"].mean()
        bottom_mean = group.loc[group["bottom"], "actual"].mean()
        if pd.notna(top_mean) and pd.notna(bottom_mean):
            spreads.append(float(top_mean - bottom_mean))
    spread_arr = np.asarray(spreads, dtype=np.float64)
    return {
        "long_short_spread": float(spread_arr.mean()) if len(spread_arr) else None,
        "spread_std": float(spread_arr.std(ddof=1)) if len(spread_arr) > 1 else None,
        "spread_observation_count": int(len(spread_arr)),
    }


def evaluate_prediction(
    *,
    candidate_id: str,
    split: str,
    prediction: dict[str, Any],
    train_q10_downside: float,
) -> dict[str, Any]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    metadata = prediction["metadata"].reset_index(drop=True)
    top, bottom = datewise_top_bottom_masks(metadata, line_score)
    positive = line_score >= 0.0
    high_confidence = line_score >= 0.005
    severe = actual <= SEVERE_THRESHOLD
    q10_downside = actual <= train_q10_downside
    top_mean = _safe_mean(actual[top])
    bottom_mean = _safe_mean(actual[bottom])
    global_spread = None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
    spread_payload = datewise_long_short_spread(line_score, actual, metadata)
    spread = spread_payload["long_short_spread"]
    fee = None if spread is None else float(spread - FEE_BPS)
    severe_top = top & severe
    line_score_std = _safe_std(line_score)
    return {
        "candidate_id": candidate_id,
        "split": split,
        **spearman_ic_by_date(line_score, actual, metadata),
        **spread_payload,
        "fee_adjusted_return": fee,
        "top_decile_actual_return": top_mean,
        "bottom_decile_actual_return": bottom_mean,
        "global_top_bottom_spread": global_spread,
        "top_decile_positive_return_rate": _safe_ratio(int((top & (actual > 0)).sum()), int(top.sum())),
        "line_top_decile_false_safe_rate": _safe_ratio(int(severe_top.sum()), int(top.sum())),
        "line_top_decile_severe_rate": _safe_ratio(int(severe_top.sum()), int(top.sum())),
        "line_positive_false_safe_rate": _safe_ratio(int((positive & severe).sum()), int(positive.sum())),
        "high_confidence_false_safe_rate": _safe_ratio(int((high_confidence & severe).sum()), int(high_confidence.sum())),
        "high_confidence_false_safe_share_of_severe": _safe_ratio(int((high_confidence & severe).sum()), int(severe.sum())),
        "severe_top_decile_capture_rate": _safe_ratio(int(severe_top.sum()), int(severe.sum())),
        "false_safe_expected_damage": _safe_mean(actual[severe_top]),
        "severe_downside_recall_line_negative": _safe_ratio(int((severe & (line_score < 0)).sum()), int(severe.sum())),
        "q10_downside_recall_line_negative": _safe_ratio(int((q10_downside & (line_score < 0)).sum()), int(q10_downside.sum())),
        "line_score_mean": _safe_mean(line_score),
        "line_score_std": line_score_std,
        "line_score_q10": float(np.quantile(line_score[np.isfinite(line_score)], 0.10)) if np.isfinite(line_score).any() else None,
        "line_score_q50": float(np.quantile(line_score[np.isfinite(line_score)], 0.50)) if np.isfinite(line_score).any() else None,
        "line_score_q90": float(np.quantile(line_score[np.isfinite(line_score)], 0.90)) if np.isfinite(line_score).any() else None,
        "prediction_bias": _safe_mean(line_score - actual),
        "overprediction_rate": _safe_ratio(int((line_score > actual).sum()), int(len(actual))),
        "underprediction_rate": _safe_ratio(int((line_score < actual).sum()), int(len(actual))),
        "line_gate_pass": bool((spread or 0.0) > 0.0 and (fee or -1.0) > 0.0 and ((spearman_ic_by_date(line_score, actual, metadata).get("ic_mean") or 0.0) > 0.0)),
        "top_count": int(top.sum()),
        "bottom_count": int(bottom.sum()),
        "severe_count": int(severe.sum()),
        "high_confidence_count": int(high_confidence.sum()),
    }


def train_one_trial(
    *,
    spec: TrialSpec,
    train_dataset: LineTrialDataset,
    val_dataset: LineTrialDataset,
    test_dataset: LineTrialDataset,
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    train_q10_downside: float,
    n_features: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    set_seed(SEED)
    model = _make_model(n_features=n_features, dropout=0.10).to(device)
    criterion = AsymmetricHuberLoss(alpha=1.0, beta=spec.beta, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.362816234925851e-4, weight_decay=8.143270337695065e-5)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    start = time.perf_counter()
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for features, raw_future_returns, ticker_id in loader:
            features = features.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device):
                output = model(features, ticker_id=ticker_id)
                line = _extract_line(output)
                loss = criterion(line, raw_future_returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_rows.append(
            {
                "trial_id": spec.trial_id,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "batch_count": int(len(losses)),
            }
        )
        _log("trial_epoch_done", trial_id=spec.trial_id, epoch=epoch, train_loss=epoch_rows[-1]["train_loss"])
    val_prediction = collect_predictions(model, val_dataset, metadata=val_metadata, device=device, batch_size=batch_size)
    test_prediction = collect_predictions(model, test_dataset, metadata=test_metadata, device=device, batch_size=batch_size)
    val_metrics = evaluate_prediction(
        candidate_id=spec.trial_id,
        split="validation",
        prediction=val_prediction,
        train_q10_downside=train_q10_downside,
    )
    test_metrics = evaluate_prediction(
        candidate_id=spec.trial_id,
        split="test",
        prediction=test_prediction,
        train_q10_downside=train_q10_downside,
    )
    runtime = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "trial_id": spec.trial_id,
        "stage": spec.stage,
        "beta": spec.beta,
        "include_atr": spec.include_atr,
        "model_role": "line_model",
        "output_role_runtime": "legacy_line_head_only",
        "seed": SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "n_features": n_features,
        "runtime_seconds": runtime,
        "epoch_rows": epoch_rows,
        "validation": val_metrics,
        "test": test_metrics,
    }


def add_retention_and_decision(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    base_false_safe = baseline.get("line_top_decile_false_safe_rate")
    base_ic = baseline.get("ic_mean")
    base_spread = baseline.get("long_short_spread")
    base_fee = baseline.get("fee_adjusted_return")
    base_std = baseline.get("line_score_std")
    for row in rows:
        item = dict(row)
        false_safe = item.get("line_top_decile_false_safe_rate")
        ic = item.get("ic_mean")
        spread = item.get("long_short_spread")
        fee = item.get("fee_adjusted_return")
        std = item.get("line_score_std")
        item["false_safe_absolute_reduction_vs_beta2"] = None if base_false_safe is None or false_safe is None else float(base_false_safe - false_safe)
        item["ic_retention_vs_beta2"] = None if not base_ic or ic is None else float(ic / base_ic)
        item["spread_retention_vs_beta2"] = None if not base_spread or spread is None else float(spread / base_spread)
        item["fee_retention_vs_beta2"] = None if not base_fee or fee is None else float(fee / base_fee)
        item["line_score_std_retention_vs_beta2"] = None if not base_std or std is None else float(std / base_std)
        collapse_warn = bool((item["line_score_std_retention_vs_beta2"] is not None and item["line_score_std_retention_vs_beta2"] < 0.50))
        reject = bool(
            (spread is None or spread <= 0)
            or (item["ic_retention_vs_beta2"] is not None and item["ic_retention_vs_beta2"] < 0.70)
            or collapse_warn
            or ((item.get("top_decile_actual_return") or 0.0) < -0.01)
        )
        if reject:
            decision = "FAIL"
        elif (
            (item["false_safe_absolute_reduction_vs_beta2"] or 0.0) >= 0.03
            and (item["ic_retention_vs_beta2"] or 0.0) >= 0.85
            and (item["spread_retention_vs_beta2"] or 0.0) >= 0.80
            and (item["fee_retention_vs_beta2"] or 0.0) >= 0.80
            and (item["line_score_std_retention_vs_beta2"] or 0.0) >= 0.50
        ):
            decision = "PASS_CANDIDATE"
        elif (
            (item["false_safe_absolute_reduction_vs_beta2"] or 0.0) >= 0.015
            and (item["ic_retention_vs_beta2"] or 0.0) >= 0.75
            and (item["spread_retention_vs_beta2"] or 0.0) >= 0.75
            and not collapse_warn
        ):
            decision = "WARN_PARETO"
        else:
            decision = "FAIL"
        item["line_collapse_warn"] = collapse_warn
        item["cp175_decision"] = decision
        enriched.append(item)
    return enriched


def pareto_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in summary_rows if row.get("split") == "test"]
    rows: list[dict[str, Any]] = []
    for row in candidates:
        dominated = False
        fs = row.get("false_safe_absolute_reduction_vs_beta2")
        spread = row.get("spread_retention_vs_beta2")
        fee = row.get("fee_retention_vs_beta2")
        if fs is None or spread is None:
            dominated = True
        else:
            for other in candidates:
                if other is row:
                    continue
                ofs = other.get("false_safe_absolute_reduction_vs_beta2")
                ospread = other.get("spread_retention_vs_beta2")
                ofee = other.get("fee_retention_vs_beta2")
                if ofs is None or ospread is None:
                    continue
                if ofs >= fs and ospread >= spread and (ofee or -999) >= (fee or -999) and (ofs > fs or ospread > spread or (ofee or -999) > (fee or -999)):
                    dominated = True
                    break
        rows.append(
            {
                "trial_id": row.get("trial_id"),
                "beta": row.get("beta"),
                "include_atr": row.get("include_atr"),
                "spread_retention": row.get("spread_retention_vs_beta2"),
                "ic_retention": row.get("ic_retention_vs_beta2"),
                "fee_retention": row.get("fee_retention_vs_beta2"),
                "false_safe_absolute_reduction": row.get("false_safe_absolute_reduction_vs_beta2"),
                "line_score_std_retention": row.get("line_score_std_retention_vs_beta2"),
                "pareto_frontier": not dominated,
                "decision": row.get("cp175_decision"),
            }
        )
    return rows


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "finite_count": int(np.isfinite(arr).sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "finite_rate": _safe_ratio(int(np.isfinite(arr).sum()), int(arr.size)),
        "mean": _safe_mean(arr),
        "std": _safe_std(arr),
        "q01": float(np.quantile(arr[np.isfinite(arr)], 0.01)) if np.isfinite(arr).any() else None,
        "q50": float(np.quantile(arr[np.isfinite(arr)], 0.50)) if np.isfinite(arr).any() else None,
        "q99": float(np.quantile(arr[np.isfinite(arr)], 0.99)) if np.isfinite(arr).any() else None,
    }


def build_report(payload: dict[str, Any]) -> str:
    summary_rows = payload["summary_rows"]
    test_rows = [row for row in summary_rows if row["split"] == "test"]
    best_warn = next((row for row in test_rows if row.get("cp175_decision") == "PASS_CANDIDATE"), None)
    if best_warn is None:
        best_warn = next((row for row in test_rows if row.get("cp175_decision") == "WARN_PARETO"), None)
    final = "FAIL"
    if any(row.get("cp175_decision") == "PASS_CANDIDATE" for row in test_rows):
        final = "PASS"
    elif any(row.get("cp175_decision") == "WARN_PARETO" for row in test_rows):
        final = "WARN"
    lines = [
        "# CP175-LM 1D Conservative Line Learning Revisit",
        "",
        f"한 줄 결론: `{final}`. β/ATR 학습 fix smoke 결과는 아래 표 기준으로 판정했다.",
        "",
        "## 범위 준수",
        "",
        "- 새 product save-run, DB write, inference 저장, live fetch, EODHD fallback, band/composite 실험은 수행하지 않았다.",
        "- 데이터는 yfinance 500 local parquet와 calendar_aligned split만 사용했다.",
        "- `atr_ratio`는 전역 feature contract에 넣지 않고 CP175 runner 내부 실험용 확장 컬럼으로만 붙였다.",
        "- runtime output은 기존 PatchTST legacy head를 사용했지만 loss/evaluation은 line head만 사용했다. lower/upper band는 학습/평가하지 않았다.",
        "",
        "## Stage 0 Preflight",
        "",
        f"- source_data_hash: `{payload['preflight'].get('source_data_hash')}`",
        f"- split_mode: `{payload['preflight'].get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{payload['preflight'].get('cross_split_date_overlap_count')}`",
        f"- feature_version: `{payload['preflight'].get('feature_version')}`",
        f"- feature_set: `{payload['preflight'].get('feature_set')}`",
        f"- base feature count: `{payload['preflight'].get('base_feature_count')}`",
        f"- `atr_ratio` in indicator parquet: `{payload['preflight'].get('atr_ratio_in_indicator')}`",
        f"- `atr_ratio` in MODEL_FEATURE_COLUMNS: `{payload['preflight'].get('atr_ratio_in_model_features')}`",
        "",
        "## Test 결과 요약",
        "",
        "| trial | beta | ATR | IC | spread | fee | false-safe | severe recall(line<0) | FS 감소 | IC 유지 | spread 유지 | decision |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in test_rows:
        lines.append(
            "| {trial} | {beta} | {atr} | {ic} | {spread} | {fee} | {fs} | {recall} | {fsred} | {icret} | {spreadret} | {decision} |".format(
                trial=row.get("trial_id"),
                beta=row.get("beta"),
                atr="yes" if row.get("include_atr") else "no",
                ic=_fmt(row.get("ic_mean")),
                spread=_fmt(row.get("long_short_spread")),
                fee=_fmt(row.get("fee_adjusted_return")),
                fs=_fmt(row.get("line_top_decile_false_safe_rate")),
                recall=_fmt(row.get("severe_downside_recall_line_negative")),
                fsred=_fmt(row.get("false_safe_absolute_reduction_vs_beta2")),
                icret=_fmt(row.get("ic_retention_vs_beta2")),
                spreadret=_fmt(row.get("spread_retention_vs_beta2")),
                decision=row.get("cp175_decision"),
            )
        )
    lines.extend(
        [
            "",
            "## 핵심 해석",
            "",
            "- β를 키우면 예측 bias와 score 분포가 어떻게 움직이는지, false-safe 감소가 실제 top decile에서 생기는지를 봤다.",
            "- ATR 재포함은 `MODEL_FEATURE_COLUMNS` 변경 없이 실험용으로만 수행했으므로, 효과가 있더라도 정식 feature contract 변경 후보로 별도 CP가 필요하다.",
            "- `line_top_decile_false_safe_rate`는 line 단독 탈락 게이트가 아니라 이번 CP에서는 보수 학습 fix의 효과 측정 지표로 사용했다.",
        ]
    )
    if best_warn:
        lines.extend(
            [
                "",
                "## 다음 후보",
                "",
                f"- 가장 좋은 Pareto 후보: `{best_warn.get('trial_id')}`",
                f"- false-safe 감소: `{_fmt(best_warn.get('false_safe_absolute_reduction_vs_beta2'))}`",
                f"- IC/spread/fee 유지율: `{_fmt(best_warn.get('ic_retention_vs_beta2'))}` / `{_fmt(best_warn.get('spread_retention_vs_beta2'))}` / `{_fmt(best_warn.get('fee_retention_vs_beta2'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            f"- metrics JSON: `{METRICS_PATH.name}`",
            f"- summary CSV: `{SUMMARY_PATH.name}`",
            f"- beta-only CSV: `{BETA_ONLY_PATH.name}`",
            f"- ATR-only CSV: `{ATR_ONLY_PATH.name}`",
            f"- beta+ATR CSV: `{BETA_ATR_PATH.name}`",
            f"- Pareto CSV: `{PARETO_PATH.name}`",
            f"- collapse diagnostic CSV: `{COLLAPSE_PATH.name}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6f}"


def run(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("runtime_check", sys_executable=sys.executable, torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    device = _device_from_arg(args.device)
    if device.type == "cuda":
        _log("cuda_device", name=torch.cuda.get_device_name(0))
    payload = cp171.load_cp171_payload()
    split_summary = payload["split_summary"]
    if str(split_summary.get("source_data_hash")) != SOURCE_HASH_EXPECTED:
        raise RuntimeError(f"source_data_hash 불일치: {split_summary.get('source_data_hash')} != {SOURCE_HASH_EXPECTED}")
    if str(split_summary.get("split_mode")) != "calendar_aligned":
        raise RuntimeError(f"split_mode 불일치: {split_summary.get('split_mode')}")
    if int(split_summary.get("cross_split_date_overlap_count") or 0) != 0:
        raise RuntimeError(f"cross_split_date_overlap_count 불일치: {split_summary.get('cross_split_date_overlap_count')}")

    train_raw = payload["train"]
    val_raw = payload["val"]
    test_raw = payload["test"]
    feature_columns = resolve_feature_columns(FEATURE_SET)
    train, val, test, _mean, _std = apply_feature_columns_to_splits(
        train_raw,
        val_raw,
        test_raw,
        train_raw.mean,
        train_raw.std,
        feature_columns,
    )
    train_actual = collect_actual_h5(train, batch_size=args.eval_batch_size)
    train_q10 = float(np.quantile(train_actual[np.isfinite(train_actual)], 0.10))
    atr_train = np.asarray(payload["train_features"]["atr_ratio"], dtype=np.float64)
    atr_val = np.asarray(payload["val_features"]["atr_ratio"], dtype=np.float64)
    atr_test = np.asarray(payload["test_features"]["atr_ratio"], dtype=np.float64)
    atr_train_clean = np.nan_to_num(atr_train, nan=0.0, posinf=0.0, neginf=0.0)
    atr_mean = float(np.mean(atr_train_clean))
    atr_std = float(np.std(atr_train_clean, ddof=1)) if len(atr_train_clean) > 1 else 1.0
    if not math.isfinite(atr_std) or abs(atr_std) < 1e-12:
        atr_std = 1.0

    base_train_dataset = LineTrialDataset(train, include_atr=False)
    base_val_dataset = LineTrialDataset(val, include_atr=False)
    base_test_dataset = LineTrialDataset(test, include_atr=False)

    cp164_test = evaluate_prediction(
        candidate_id="cp164_calendar_line_regime_reference",
        split="test",
        prediction=payload["test_prediction"],
        train_q10_downside=train_q10,
    )
    cp164_val = evaluate_prediction(
        candidate_id="cp164_calendar_line_regime_reference",
        split="validation",
        prediction=payload["val_prediction"],
        train_q10_downside=train_q10,
    )

    preflight = {
        "source_data_hash": split_summary.get("source_data_hash"),
        "split_mode": split_summary.get("split_mode"),
        "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
        "cross_split_date_overlap_count_bundle_check": split_summary.get("cross_split_date_overlap_count_bundle_check"),
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": FEATURE_SET,
        "base_feature_count": len(feature_columns),
        "base_feature_columns": feature_columns,
        "atr_ratio_in_indicator": True,
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "atr_ratio_experiment_extension_only": True,
        "atr_ratio_train_summary": finite_summary(atr_train),
        "atr_ratio_validation_summary": finite_summary(atr_val),
        "atr_ratio_test_summary": finite_summary(atr_test),
        "train_q10_downside": train_q10,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
    }
    _log("preflight_done", **{k: v for k, v in preflight.items() if k not in {"base_feature_columns"}})

    trials_to_run = TRIALS
    if args.trials:
        wanted = {name.strip() for name in args.trials.split(",") if name.strip()}
        trials_to_run = [spec for spec in TRIALS if spec.trial_id in wanted]
        missing = wanted - {spec.trial_id for spec in trials_to_run}
        if missing:
            raise ValueError(f"알 수 없는 trial_id: {sorted(missing)}")

    raw_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in trials_to_run:
        include_atr = bool(spec.include_atr)
        n_features = len(feature_columns) + (1 if include_atr else 0)
        train_dataset = (
            LineTrialDataset(train, include_atr=True, atr_values=atr_train_clean, atr_mean=atr_mean, atr_std=atr_std)
            if include_atr
            else base_train_dataset
        )
        val_dataset = (
            LineTrialDataset(val, include_atr=True, atr_values=np.nan_to_num(atr_val, nan=0.0, posinf=0.0, neginf=0.0), atr_mean=atr_mean, atr_std=atr_std)
            if include_atr
            else base_val_dataset
        )
        test_dataset = (
            LineTrialDataset(test, include_atr=True, atr_values=np.nan_to_num(atr_test, nan=0.0, posinf=0.0, neginf=0.0), atr_mean=atr_mean, atr_std=atr_std)
            if include_atr
            else base_test_dataset
        )
        _log("trial_start", trial_id=spec.trial_id, beta=spec.beta, include_atr=spec.include_atr, n_features=n_features)
        result = train_one_trial(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            val_metadata=val.metadata,
            test_metadata=test.metadata,
            train_q10_downside=train_q10,
            n_features=n_features,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        raw_results.append(result)
        for split in ("validation", "test"):
            row = {
                "trial_id": spec.trial_id,
                "stage": spec.stage,
                "beta": spec.beta,
                "include_atr": spec.include_atr,
                "split": split,
                **result[split],
                "runtime_seconds": result["runtime_seconds"],
            }
            summary_rows.append(row)
        _log("trial_done", trial_id=spec.trial_id, runtime_seconds=result["runtime_seconds"], test_false_safe=result["test"].get("line_top_decile_false_safe_rate"))

    beta2_test = next((row for row in summary_rows if row.get("trial_id") == "beta_2_baseline" and row.get("split") == "test"), None)
    if beta2_test is None:
        raise RuntimeError("beta_2_baseline test 결과가 없어 retention 기준을 만들 수 없습니다.")
    summary_rows = add_retention_and_decision(summary_rows, beta2_test)
    pareto = pareto_rows(summary_rows)
    collapse_rows = [
        {
            "trial_id": row.get("trial_id"),
            "split": row.get("split"),
            "line_score_std": row.get("line_score_std"),
            "line_score_std_retention_vs_beta2": row.get("line_score_std_retention_vs_beta2"),
            "prediction_bias": row.get("prediction_bias"),
            "overprediction_rate": row.get("overprediction_rate"),
            "underprediction_rate": row.get("underprediction_rate"),
            "line_collapse_warn": row.get("line_collapse_warn"),
            "cp175_decision": row.get("cp175_decision"),
        }
        for row in summary_rows
    ]

    _write_csv(SUMMARY_PATH, summary_rows)
    _write_csv(BETA_ONLY_PATH, [row for row in summary_rows if row.get("stage") == "beta_only"])
    _write_csv(ATR_ONLY_PATH, [row for row in summary_rows if row.get("stage") == "atr_only"])
    _write_csv(BETA_ATR_PATH, [row for row in summary_rows if row.get("stage") == "beta_atr_combo"])
    _write_csv(PARETO_PATH, pareto)
    _write_csv(COLLAPSE_PATH, collapse_rows)

    metrics_payload = {
        "created_at": _now(),
        "preflight": preflight,
        "cp164_reference": {
            "validation": cp164_val,
            "test": cp164_test,
        },
        "trial_results": raw_results,
        "summary_rows": summary_rows,
        "pareto_frontier": pareto,
        "paths": {
            "report": REPORT_PATH,
            "metrics": METRICS_PATH,
            "summary": SUMMARY_PATH,
            "beta_only": BETA_ONLY_PATH,
            "atr_only": ATR_ONLY_PATH,
            "beta_atr_combo": BETA_ATR_PATH,
            "pareto": PARETO_PATH,
            "collapse": COLLAPSE_PATH,
        },
        "forbidden_actions": {
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "band_experiment": False,
            "composite_experiment": False,
        },
    }
    _write_json(METRICS_PATH, metrics_payload)
    REPORT_PATH.write_text(build_report(metrics_payload), encoding="utf-8")
    _log("done", report_path=str(REPORT_PATH), metrics_path=str(METRICS_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP175 1D conservative line beta/ATR smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--trials", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
