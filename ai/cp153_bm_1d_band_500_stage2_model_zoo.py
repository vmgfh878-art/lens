from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_PYTHON_PATH = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# CP153 Stage 2는 yfinance 500 로컬 parquet만 사용한다.
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SequenceDataset,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import load_registry  # noqa: E402
from ai.train import MODEL_REGISTRY, autocast_context, forward_model, make_loader, resolve_device, resolve_feature_columns  # noqa: E402

from ai.cp153_bm_1d_band_500_stage0_1_baseline import (  # noqa: E402
    BACKFILL_STATE_PATH,
    INDICATOR_MANIFEST_PATH,
    INDICATOR_PATH,
    PRICE_MANIFEST_PATH,
    PRICE_PATH,
    SUMMARY_CSV_PATH as STAGE0_SUMMARY_CSV_PATH,
    _clean_json,
    _date_cross_sectional_width_ic,
    _json_default,
    _read_json,
    _safe_float,
    ctra_hubb_status,
    duplicate_summary,
    ensure_feature_set_plan_available,
    feature_quality_summary,
    provider_spot_check,
    split_overlap_summary,
)


TIMEFRAME = "1D"
HORIZON = 5
TARGET_TYPE = "raw_future_return"
PROVIDER = "yfinance"
FEATURE_SET = "price_volatility_volume"
SOURCE_DATA_HASH = "90666b44cbfb8e5c"
SEED = 42
DEFAULT_EPOCHS = 3

STAGE0_METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_metrics.json"
OHLC_AUDIT_METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_ohlc_ratio_audit_metrics.json"

REPORT_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_model_zoo_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_model_zoo_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_model_zoo_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_model_zoo_logs"
OVERLAY_DIR = LOG_DIR / "snapshot_overlay"
TRAIN_LOG_BASE_DIR = LOG_DIR / "ai_train_local_logs"

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "lower_breach_abs_error",
    "upper_breach_abs_error",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "asymmetric_interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
]

SCORE_WEIGHTS = {
    "asymmetric_interval_score": 0.30,
    "coverage_abs_error": 0.20,
    "lower_breach_abs_error": 0.15,
    "band_width_ic": 0.15,
    "downside_width_ic": 0.10,
    "p90_band_width": 0.05,
    "squeeze_breakout_rate": 0.05,
}

LOWER_IS_BETTER_FOR_SCORE = {
    "asymmetric_interval_score",
    "coverage_abs_error",
    "lower_breach_abs_error",
    "p90_band_width",
    "squeeze_breakout_rate",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    seq_len: int
    q_low: float
    q_high: float
    band_mode: str
    family: str
    note: str
    batch_size: int = 256
    epochs: int = DEFAULT_EPOCHS
    lower_band_loss_weight: float = 1.0
    fp32_modules: str | None = None

    @property
    def q_label(self) -> str:
        if abs(self.q_low - 0.15) < 1e-9 and abs(self.q_high - 0.85) < 1e-9:
            return "q15_q85"
        if abs(self.q_low - 0.10) < 1e-9 and abs(self.q_high - 0.90) < 1e-9:
            return "q10_q90"
        return f"q{int(self.q_low * 100):02d}_q{int(self.q_high * 100):02d}"


CANDIDATES = [
    Candidate(
        candidate_id="cnn_s60_q15_direct",
        model="cnn_lstm",
        family="cnn_lstm",
        seq_len=60,
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        fp32_modules="lstm,heads",
        note="CP72 계열을 yfinance 500 기준으로 재확인",
    ),
    Candidate(
        candidate_id="cnn_s60_q10_direct",
        model="cnn_lstm",
        family="cnn_lstm",
        seq_len=60,
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        fp32_modules="lstm,heads",
        note="보수 coverage 후보",
    ),
    Candidate(
        candidate_id="cnn_s60_q15_lower_guard",
        model="cnn_lstm",
        family="cnn_lstm",
        seq_len=60,
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        fp32_modules="lstm,heads",
        lower_band_loss_weight=1.5,
        note="lower breach guard 실험",
    ),
    Candidate(
        candidate_id="tcn_s60_q15_direct",
        model="tcn_quantile",
        family="tcn_quantile",
        seq_len=60,
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        note="band 전용 신규 family",
    ),
    Candidate(
        candidate_id="tcn_s120_q15_direct",
        model="tcn_quantile",
        family="tcn_quantile",
        seq_len=120,
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        note="긴 receptive field 비교",
    ),
    Candidate(
        candidate_id="tcn_s60_q10_direct",
        model="tcn_quantile",
        family="tcn_quantile",
        seq_len=60,
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        note="보수 TCN 후보",
    ),
    Candidate(
        candidate_id="tide_s104_q15_param",
        model="tide",
        family="tide",
        seq_len=104,
        q_low=0.15,
        q_high=0.85,
        band_mode="param",
        note="계획서상 TiDE 대안 후보",
    ),
    Candidate(
        candidate_id="patch_s252_q15_direct",
        model="patchtst",
        family="patchtst",
        seq_len=252,
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        batch_size=128,
        note="reference 후보",
    ),
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_parquet_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(PRICE_PATH)
    indicators = pd.read_parquet(INDICATOR_PATH)
    price["ticker"] = price["ticker"].astype(str).str.upper()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"].astype(str).str.upper() == TIMEFRAME].copy()
    return price, indicators


def prepare_stage2_snapshot_overlay() -> dict[str, Any]:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    links = [
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance_1D.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance_1D.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators_yfinance_1D.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators_yfinance_1D.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators.manifest.json"),
    ]
    entries: list[dict[str, Any]] = []
    for source, target in links:
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            mode = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        entries.append({"source": str(source), "target": str(target), "mode": mode})
    return {"overlay_dir": str(OVERLAY_DIR), "entries": entries}


def build_contract_payload(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    stage0_metrics: dict[str, Any],
) -> dict[str, Any]:
    feature_set_plan = ensure_feature_set_plan_available()
    feature_columns = resolve_feature_columns(FEATURE_SET)
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=60,
        horizon=HORIZON,
        market_data_provider=PROVIDER,
        source_data_hash=SOURCE_DATA_HASH,
    )
    registry = load_registry(TIMEFRAME, Path(plan.ticker_registry_path))
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price[price["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=60,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    targets = {
        "train": _collect_targets(train),
        "val": _collect_targets(val),
        "test": _collect_targets(test),
    }
    ohlc_audit = _read_json(OHLC_AUDIT_METRICS_PATH) if OHLC_AUDIT_METRICS_PATH.exists() else {"status": "missing"}
    stale_reasons = stage0_metrics.get("stage0", {}).get("contract_fail_reasons")
    return {
        "source_data_hash": SOURCE_DATA_HASH,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": {
            "name": FEATURE_SET,
            "columns": feature_columns,
            "column_count": len(feature_columns),
            "plan_file": feature_set_plan,
        },
        "stage0_reference": {
            "metrics_path": str(STAGE0_METRICS_PATH),
            "summary_csv_path": str(STAGE0_SUMMARY_CSV_PATH),
            "final_status": stage0_metrics.get("final_status"),
            "timing_smoke_status": stage0_metrics.get("stage0", {}).get("timing_smoke", {}).get("status"),
            "stale_contract_fail_reasons": stale_reasons,
            "stale_contract_correction": (
                "Stage 0/1 stale field이며 timing smoke 실제 status는 PASS"
                if stale_reasons and "timing_smoke_failed" in stale_reasons
                else None
            ),
        },
        "ohlc_ratio_audit": {
            "path": str(OHLC_AUDIT_METRICS_PATH),
            "status": ohlc_audit.get("status"),
            "open_ratio": ohlc_audit.get("feature_contract", {}).get("open_ratio") or ohlc_audit.get("open_ratio"),
            "high_ratio": ohlc_audit.get("feature_contract", {}).get("high_ratio") or ohlc_audit.get("high_ratio"),
            "low_ratio": ohlc_audit.get("feature_contract", {}).get("low_ratio") or ohlc_audit.get("low_ratio"),
        },
        "dataset_contract": {
            "input_ticker_count": int(plan.input_ticker_count),
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
            "excluded_reasons": plan.excluded_reasons,
            "split_overlap": split_overlap_summary(train, val, test),
            "duplicates": duplicate_summary(price, indicators),
            "feature_quality": feature_quality_summary(indicators, targets),
            "ctra_hubb_status": ctra_hubb_status(price=price, indicators=indicators, plan=plan),
            "provider_spot_check_status_from_stage0": stage0_metrics.get("stage0", {})
            .get("provider_spot_check", {})
            .get("status"),
        },
        "forbidden_actions": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
            "line_metric_for_rejection": False,
        },
    }


def _collect_targets(bundle: SequenceDataset) -> np.ndarray:
    targets = np.empty((len(bundle.sample_refs), bundle.horizon), dtype=np.float32)
    refs_by_ticker: dict[str, list[tuple[int, int]]] = {}
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        refs_by_ticker.setdefault(str(ticker), []).append((row_idx, int(end_idx)))
    for ticker, refs in refs_by_ticker.items():
        closes = np.asarray(bundle.ticker_arrays[ticker]["closes"], dtype=np.float64)
        for row_idx, end_idx in refs:
            anchor = float(closes[end_idx])
            future = closes[end_idx + 1 : end_idx + 1 + bundle.horizon]
            targets[row_idx] = ((future / anchor) - 1.0).astype(np.float32)
    return targets


def command_for_candidate(candidate: Candidate, *, device: str) -> list[str]:
    cmd = [
        str(VENV_PYTHON_PATH),
        "-m",
        "ai.train",
        "--model",
        candidate.model,
        "--model-role",
        "band",
        "--timeframe",
        TIMEFRAME,
        "--horizon",
        str(HORIZON),
        "--seq-len",
        str(candidate.seq_len),
        "--feature-set",
        FEATURE_SET,
        "--line-target-type",
        TARGET_TYPE,
        "--band-target-type",
        TARGET_TYPE,
        "--q-low",
        str(candidate.q_low),
        "--q-high",
        str(candidate.q_high),
        "--lambda-band",
        "2.0",
        "--band-mode",
        candidate.band_mode,
        "--checkpoint-selection",
        "band_gate",
        "--epochs",
        str(candidate.epochs),
        "--batch-size",
        str(candidate.batch_size),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--amp-dtype",
        "bf16",
        "--no-compile",
        "--no-wandb",
        "--num-workers",
        "0",
        "--local-log",
        "--local-log-dir",
        str(TRAIN_LOG_BASE_DIR),
        "--explicit-cuda-cleanup",
    ]
    if candidate.fp32_modules:
        cmd.extend(["--fp32-modules", candidate.fp32_modules])
    if candidate.lower_band_loss_weight != 1.0:
        cmd.extend(["--lower-band-loss-weight", str(candidate.lower_band_loss_weight)])
    return cmd


def run_candidate(candidate: Candidate, *, device: str, force: bool = False) -> dict[str, Any]:
    candidate_dir = LOG_DIR / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    process_path = candidate_dir / "train_process.json"
    stdout_path = candidate_dir / "train_stdout.log"
    evaluation_path = candidate_dir / "evaluation.json"
    if not force and process_path.exists():
        existing = _read_json(process_path)
        if existing.get("status") == "PASS" and existing.get("checkpoint_path") and Path(existing["checkpoint_path"]).exists():
            return existing

    cmd = command_for_candidate(candidate, device=device)
    env = os.environ.copy()
    env.update(
        {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(OVERLAY_DIR),
            "WANDB_MODE": "disabled",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    start_ts = _now_utc()
    start = time.perf_counter()
    output_lines: list[str] = []
    run_id: str | None = None
    print(json.dumps({"candidate": candidate.candidate_id, "event": "start", "time": start_ts}, ensure_ascii=False))
    with stdout_path.open("w", encoding="utf-8", newline="") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line.rstrip("\n"))
            log_handle.write(line)
            log_handle.flush()
            maybe_run_id = _extract_run_id(line)
            if maybe_run_id:
                run_id = maybe_run_id
            if _should_echo_training_line(line):
                print(f"[{candidate.candidate_id}] {line}", end="")
        exit_code = proc.wait()
    elapsed = time.perf_counter() - start
    end_ts = _now_utc()
    run_id = run_id or _extract_run_id("\n".join(output_lines))
    local_summary = _read_local_summary(run_id) if run_id else None
    checkpoint_path = _resolve_checkpoint_path(local_summary)
    metrics_jsonl_path = str(TRAIN_LOG_BASE_DIR / run_id / "metrics.jsonl") if run_id else None
    epoch_logs = _read_epoch_logs(Path(metrics_jsonl_path)) if metrics_jsonl_path else []
    status = "PASS" if exit_code == 0 and checkpoint_path and Path(checkpoint_path).exists() else "FAIL"
    failure_reason = None
    if status != "PASS":
        failure_reason = classify_runtime_failure(output_lines)
    process = {
        "candidate": asdict(candidate),
        "status": status,
        "failure_reason": failure_reason,
        "failure_category": "runtime" if status != "PASS" else None,
        "start_time_utc": start_ts,
        "end_time_utc": end_ts,
        "elapsed_seconds": round(elapsed, 3),
        "pid": None,
        "exit_code": int(exit_code),
        "command": cmd,
        "stdout_path": str(stdout_path),
        "train_process_path": str(process_path),
        "evaluation_path": str(evaluation_path),
        "run_id": run_id,
        "local_summary": local_summary,
        "checkpoint_path": checkpoint_path,
        "metrics_jsonl_path": metrics_jsonl_path,
        "epoch_logs": epoch_logs,
        "epoch_seconds": _epoch_seconds(epoch_logs),
        "vram_peak_allocated_mb": _vram_peak(epoch_logs),
        "full_universe": True,
        "limit_tickers": None,
        "save_run": False,
        "wandb": False,
    }
    process_path.write_text(json.dumps(_clean_json(process), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate": candidate.candidate_id,
                "event": "end",
                "status": status,
                "exit_code": exit_code,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return process


def _should_echo_training_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if "epoch_seconds" in text or "checkpoint_selection" in text or "[EXIT-MARKER" in text:
        return True
    if text.startswith("학습 디바이스") or text.startswith("GPU:") or text.startswith("amp_dtype"):
        return True
    if text.startswith("val_total="):
        return True
    return False


def _extract_run_id(text: str) -> str | None:
    patterns = [
        r'"run_id"\s*:\s*"([^"]+)"',
        r"run_id=([A-Za-z0-9_.:-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _read_local_summary(run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    summary_path = TRAIN_LOG_BASE_DIR / run_id / "summary.json"
    if not summary_path.exists():
        return None
    return _read_json(summary_path)


def _resolve_checkpoint_path(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    raw = summary.get("checkpoint_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _read_epoch_logs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _epoch_seconds(epoch_logs: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["epoch_seconds"])
        for row in epoch_logs
        if _safe_float(row.get("epoch_seconds")) is not None
    ]


def _vram_peak(epoch_logs: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["vram_peak_allocated_mb"])
        for row in epoch_logs
        if _safe_float(row.get("vram_peak_allocated_mb")) is not None
    ]
    return max(values) if values else None


def classify_runtime_failure(lines: list[str]) -> dict[str, Any]:
    tail = "\n".join(lines[-80:])
    category = "runtime"
    if "NaN" in tail or "non-finite" in tail or "nonfinite" in tail:
        category = "contract"
    if "out of memory" in tail.lower() or "cuda" in tail.lower():
        category = "runtime"
    if "unsupported" in tail.lower() or "지원하지" in tail:
        category = "model_family"
    return {"category": category, "tail": tail[-4000:]}


def build_split_for_checkpoint(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    checkpoint_config: dict[str, Any],
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset, Any]:
    registry_path = checkpoint_config.get("ticker_registry_path")
    if not registry_path:
        raise ValueError("checkpoint에 ticker_registry_path가 없습니다.")
    registry = load_registry(TIMEFRAME, Path(str(registry_path)))
    mapping = registry.get("mapping") or {}
    tickers = sorted(set(mapping).intersection(set(indicators["ticker"].unique())))
    feature_frame = indicators[indicators["ticker"].isin(tickers)].copy()
    price_frame = price[price["ticker"].isin(tickers)].copy()
    plan = build_dataset_plan(
        feature_frame,
        timeframe=TIMEFRAME,
        seq_len=int(checkpoint_config["seq_len"]),
        horizon=int(checkpoint_config["horizon"]),
        ticker_registry=registry,
        ticker_registry_path=str(registry_path),
        market_data_provider=PROVIDER,
        source_data_hash=SOURCE_DATA_HASH,
    )
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_frame[feature_frame["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price_frame[price_frame["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=int(checkpoint_config["seq_len"]),
        horizon=int(checkpoint_config["horizon"]),
        ticker_registry=registry,
        include_future_covariate=bool(checkpoint_config.get("use_future_covariate", True)),
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    return train, val, test, plan


def select_bundle_features_with_checkpoint_stats(
    bundle: SequenceDataset,
    *,
    feature_columns: list[str],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> SequenceDataset:
    indices = [MODEL_FEATURE_COLUMNS.index(column) for column in feature_columns]
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        copied = dict(arrays)
        copied["features"] = arrays["features"][:, indices].copy()
        ticker_arrays[ticker] = copied
    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=feature_mean.to(dtype=torch.float32),
        std=feature_std.to(dtype=torch.float32),
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )


def load_model_from_checkpoint(checkpoint_path: str | Path) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config") or {})
    model_cls = MODEL_REGISTRY[config["model"]]
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    n_features = int(config.get("n_features") or len(feature_columns) or MODEL_N_FEATURES)
    model_role = str(config.get("model_role") or config.get("output_role") or "legacy").lower()
    model_kwargs = {
        "n_features": n_features,
        "seq_len": int(config["seq_len"]),
        "horizon": int(config["horizon"]),
        "dropout": float(config.get("dropout", 0.2)),
        "band_mode": config.get("band_mode", "direct"),
        "num_tickers": int(config.get("num_tickers", 0)),
        "ticker_emb_dim": int(config.get("ticker_emb_dim", 32)),
        "output_role": model_role,
    }
    if config["model"] == "cnn_lstm":
        model_kwargs["use_direction_head"] = bool(config.get("use_direction_head", False))
        model_kwargs["fp32_modules"] = str(config.get("fp32_modules", "none"))
    if config["model"] == "tide":
        use_future_covariate = bool(config.get("use_future_covariate", True))
        model_kwargs["future_cov_dim"] = config.get("future_cov_dim", FUTURE_COVARIATE_DIM) if use_future_covariate else 0
    if config["model"] == "patchtst":
        model_kwargs["use_revin"] = bool(config.get("use_revin", True))
        model_kwargs["ci_aggregate"] = config.get("ci_aggregate", "target")
        model_kwargs["target_channel_idx"] = int(config.get("target_channel_idx", 0))
        model_kwargs["ci_target_fast"] = bool(config.get("ci_target_fast", False))
        model_kwargs["patch_len"] = int(config.get("patch_len", 16))
        model_kwargs["stride"] = int(config.get("patch_stride", config.get("stride", 8)))
        model_kwargs["d_model"] = int(config.get("patchtst_d_model", 128))
        model_kwargs["n_heads"] = int(config.get("patchtst_n_heads", 8))
        model_kwargs["n_layers"] = int(config.get("patchtst_n_layers", 3))
    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint["feature_mean"], checkpoint["feature_std"]


def evaluate_candidate_checkpoint(
    *,
    candidate: Candidate,
    process: dict[str, Any],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    if process.get("status") != "PASS":
        return {
            "candidate_id": candidate.candidate_id,
            "status": "skipped",
            "reason": "training_failed",
        }
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path:
        return {"candidate_id": candidate.candidate_id, "status": "failed", "reason": "checkpoint_path_missing"}
    model, config, feature_mean, feature_std = load_model_from_checkpoint(checkpoint_path)
    train, val, test, plan = build_split_for_checkpoint(price=price, indicators=indicators, checkpoint_config=config)
    feature_columns = list(config.get("feature_columns") or resolve_feature_columns(FEATURE_SET))
    val_selected = select_bundle_features_with_checkpoint_stats(
        val,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_selected = select_bundle_features_with_checkpoint_stats(
        test,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    eval_device = resolve_device(device)
    model = model.to(eval_device)
    try:
        val_metrics = _evaluate_band_bundle(
            model=model,
            bundle=val_selected,
            device=eval_device,
            config=config,
        )
        test_metrics = _evaluate_band_bundle(
            model=model,
            bundle=test_selected,
            device=eval_device,
            config=config,
        )
    finally:
        if eval_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "candidate_id": candidate.candidate_id,
        "status": "completed",
        "split_rows": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "eligible_ticker_count": len(plan.eligible_tickers),
        "input_ticker_count": int(plan.input_ticker_count),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "checkpoint_config": {
            "model": config.get("model"),
            "model_role": config.get("model_role"),
            "feature_set": config.get("feature_set"),
            "q_low": config.get("q_low"),
            "q_high": config.get("q_high"),
            "band_mode": config.get("band_mode"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "checkpoint_selection": config.get("checkpoint_selection"),
        },
        "val_metrics": val_metrics,
        "test_metrics_readonly": test_metrics,
    }


def _evaluate_band_bundle(
    *,
    model: Any,
    bundle: SequenceDataset,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    loader = make_loader(bundle, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    actual_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_id, future_covariates in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = forward_model(model, features, ticker_id, future_covariates)
            if isinstance(output, ForecastOutput):
                _, lower, upper = apply_band_postprocess(
                    output.line.detach().cpu().to(torch.float32),
                    output.lower_band.detach().cpu().to(torch.float32),
                    output.upper_band.detach().cpu().to(torch.float32),
                )
            elif isinstance(output, BandOutput):
                raw_lower = output.lower_band.detach().cpu().to(torch.float32)
                raw_upper = output.upper_band.detach().cpu().to(torch.float32)
                lower = torch.minimum(raw_lower, raw_upper)
                upper = torch.maximum(raw_lower, raw_upper)
            else:
                raise TypeError(f"band_model 평가에서 허용하지 않는 출력입니다: {type(output).__name__}")
            lower_predictions.append(lower)
            upper_predictions.append(upper)
            actual_targets.append(raw_future_returns.detach().cpu().to(torch.float32))
    lower_t = torch.cat(lower_predictions, dim=0)
    upper_t = torch.cat(upper_predictions, dim=0)
    actual_t = torch.cat(actual_targets, dim=0)
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=bundle.metadata,
        line_predictions=line_t,
        lower_predictions=lower_t,
        upper_predictions=upper_t,
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=float(config.get("q_low", 0.15)),
        q_high=float(config.get("q_high", 0.85)),
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        include_legacy_overlay_diagnostics=False,
    )
    lower_np = lower_t.numpy()
    upper_np = upper_t.numpy()
    actual_np = actual_t.numpy()
    date_cs = _date_cross_sectional_width_ic(
        metadata=bundle.metadata,
        lower=lower_np,
        upper=upper_np,
        actual=actual_np,
    )
    metrics = {key: _safe_float(summary.get(key)) for key in BAND_METRIC_KEYS if key in summary}
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    return metrics


def score_and_classify(
    *,
    candidates: list[Candidate],
    processes: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    stage0_metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    stage1_gates = stage0_metrics["stage1"]["gates"]["by_q"]
    for candidate in candidates:
        process = processes.get(candidate.candidate_id) or {}
        evaluation = evaluations.get(candidate.candidate_id) or {}
        val_metrics = evaluation.get("val_metrics") or {}
        gate = stage1_gates.get(candidate.q_label, {}).get("hard_gate_lock") or {}
        gate_result = _gate_result(candidate, val_metrics, gate, process, evaluation)
        row = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "model": candidate.model,
            "seq_len": candidate.seq_len,
            "q_label": candidate.q_label,
            "q_low": candidate.q_low,
            "q_high": candidate.q_high,
            "band_mode": candidate.band_mode,
            "lower_band_loss_weight": candidate.lower_band_loss_weight,
            "status": process.get("status"),
            "exit_code": process.get("exit_code"),
            "elapsed_seconds": process.get("elapsed_seconds"),
            "epoch_seconds_mean": _mean(process.get("epoch_seconds") or []),
            "vram_peak_allocated_mb": process.get("vram_peak_allocated_mb"),
            "run_id": process.get("run_id"),
            "checkpoint_path": process.get("checkpoint_path"),
            "eligible_ticker_count": evaluation.get("eligible_ticker_count"),
            "input_ticker_count": evaluation.get("input_ticker_count"),
            "train_rows": (evaluation.get("split_rows") or {}).get("train"),
            "val_rows": (evaluation.get("split_rows") or {}).get("val"),
            "test_rows": (evaluation.get("split_rows") or {}).get("test"),
            "failure_category": gate_result["failure_category"],
            "failure_reasons": gate_result["failure_reasons"],
            "decision": gate_result["decision"],
            "near_pass": gate_result["near_pass"],
            "hard_gate_pass": gate_result["hard_gate_pass"],
        }
        for key in BAND_METRIC_KEYS:
            row[key] = val_metrics.get(key)
        row["band_selection_score"] = None
        rows.append(row)

    _attach_selection_scores(rows)
    survivors = [row for row in rows if row["decision"] in {"hard_pass", "near_pass"}]
    if not survivors:
        scored = sorted(
            [row for row in rows if _safe_float(row.get("band_selection_score")) is not None],
            key=lambda row: float(row["band_selection_score"]),
            reverse=True,
        )
        reserve_ids: list[str] = []
        if scored:
            reserve_ids.append(scored[0]["candidate_id"])
            first_family = scored[0]["family"]
            diverse = next((row for row in scored[1:] if row["family"] != first_family), None)
            if diverse:
                reserve_ids.append(diverse["candidate_id"])
        for row in rows:
            if row["candidate_id"] in reserve_ids:
                row["decision"] = "research_reserve"
                row["failure_category"] = row["failure_category"] or "metric"
        stage3_ids = reserve_ids[:2]
    else:
        stage3_rows = sorted(
            survivors,
            key=lambda row: (
                1 if row["decision"] == "hard_pass" else 0,
                float(row.get("band_selection_score") or -1),
            ),
            reverse=True,
        )
        stage3_ids = [row["candidate_id"] for row in stage3_rows[:3]]
    return rows, stage3_ids


def _gate_result(
    candidate: Candidate,
    metrics: dict[str, Any],
    gate: dict[str, Any],
    process: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    if process.get("status") != "PASS":
        return {
            "decision": "fail",
            "hard_gate_pass": False,
            "near_pass": False,
            "failure_category": process.get("failure_category") or "runtime",
            "failure_reasons": [process.get("failure_reason") or "training_failed"],
        }
    if evaluation.get("status") != "completed":
        return {
            "decision": "fail",
            "hard_gate_pass": False,
            "near_pass": False,
            "failure_category": "runtime",
            "failure_reasons": [evaluation.get("reason") or "evaluation_failed"],
        }
    if any(_safe_float(metrics.get(key)) is None for key in ["coverage_abs_error", "lower_breach_rate", "asymmetric_interval_score"]):
        return {
            "decision": "fail",
            "hard_gate_pass": False,
            "near_pass": False,
            "failure_category": "contract",
            "failure_reasons": ["metric_nan_or_missing"],
        }

    checks = [
        ("coverage_abs_error", "max", gate.get("coverage_abs_error_max")),
        ("lower_breach_rate", "max", gate.get("lower_breach_rate_max")),
        ("lower_breach_abs_error", "max", gate.get("lower_breach_abs_error_max")),
        ("upper_breach_abs_error", "max", gate.get("upper_breach_abs_error_max")),
        ("asymmetric_interval_score", "max", gate.get("asymmetric_interval_score_max")),
        ("band_width_ic", "min", gate.get("band_width_ic_min")),
        ("downside_width_ic", "min", gate.get("downside_width_ic_min")),
        ("p90_band_width", "max", gate.get("p90_band_width_overwide_threshold")),
        ("squeeze_breakout_rate", "max", gate.get("squeeze_breakout_rate_max")),
    ]
    failed: list[dict[str, Any]] = []
    near_failed: list[dict[str, Any]] = []
    for metric, direction, threshold in checks:
        value = _safe_float(metrics.get(metric))
        threshold_value = _safe_float(threshold)
        if value is None or threshold_value is None:
            failed.append({"metric": metric, "value": value, "threshold": threshold_value, "reason": "missing"})
            continue
        ok = value <= threshold_value if direction == "max" else value >= threshold_value
        if ok:
            continue
        fail_payload = {"metric": metric, "value": value, "threshold": threshold_value, "direction": direction}
        failed.append(fail_payload)
        if _is_near_miss(metric, value, threshold_value, direction):
            near_failed.append(fail_payload)

    fatal_reasons: list[str] = []
    coverage_abs_error = float(metrics.get("coverage_abs_error"))
    lower_breach_rate = float(metrics.get("lower_breach_rate"))
    p90_width = float(metrics.get("p90_band_width")) if _safe_float(metrics.get("p90_band_width")) is not None else math.inf
    p90_gate = _safe_float(gate.get("p90_band_width_overwide_threshold"))
    if coverage_abs_error > 0.15:
        fatal_reasons.append("coverage_collapse")
    if lower_breach_rate > candidate.q_low + 0.15:
        fatal_reasons.append("lower_breach_collapse")
    if p90_gate is not None and p90_width > p90_gate * 2.0:
        fatal_reasons.append("band_width_overwide_collapse")

    if not failed:
        return {
            "decision": "hard_pass",
            "hard_gate_pass": True,
            "near_pass": False,
            "failure_category": None,
            "failure_reasons": [],
        }
    if len(failed) == 1 and len(near_failed) == 1 and not fatal_reasons:
        return {
            "decision": "near_pass",
            "hard_gate_pass": False,
            "near_pass": True,
            "failure_category": "metric",
            "failure_reasons": failed,
        }
    return {
        "decision": "fail",
        "hard_gate_pass": False,
        "near_pass": False,
        "failure_category": "metric",
        "failure_reasons": [*failed, *fatal_reasons],
    }


def _is_near_miss(metric: str, value: float, threshold: float, direction: str) -> bool:
    if direction == "max":
        return value <= threshold * 1.10 or (value - threshold) <= 0.005
    if metric in {"band_width_ic", "downside_width_ic"}:
        return value >= threshold * 0.90 or (threshold - value) <= 0.02
    return value >= threshold * 0.90


def _attach_selection_scores(rows: list[dict[str, Any]]) -> None:
    for metric, weight in SCORE_WEIGHTS.items():
        values = [
            (idx, float(row[metric]))
            for idx, row in enumerate(rows)
            if _safe_float(row.get(metric)) is not None and row.get("status") == "PASS"
        ]
        if not values:
            continue
        reverse = metric not in LOWER_IS_BETTER_FOR_SCORE
        sorted_values = sorted(values, key=lambda item: item[1], reverse=reverse)
        count = len(sorted_values)
        for rank, (idx, _) in enumerate(sorted_values):
            pct = 1.0 if count == 1 else 1.0 - (rank / (count - 1))
            rows[idx].setdefault("_score_components", {})[metric] = pct * weight
    for row in rows:
        components = row.pop("_score_components", {})
        row["band_selection_score"] = sum(components.values()) if components else None
        row["band_selection_score_components"] = components


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if _safe_float(value) is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "candidate_id",
        "family",
        "model",
        "seq_len",
        "q_label",
        "band_mode",
        "lower_band_loss_weight",
        "decision",
        "band_selection_score",
        "eligible_ticker_count",
        "exit_code",
        "elapsed_seconds",
        "epoch_seconds_mean",
        "vram_peak_allocated_mb",
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "lower_breach_abs_error",
        "asymmetric_interval_score",
        "band_width_ic",
        "downside_width_ic",
        "p90_band_width",
        "squeeze_breakout_rate",
        "failure_category",
        "failure_reasons",
        "run_id",
        "checkpoint_path",
    ]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_report(metrics: dict[str, Any]) -> None:
    rows = metrics["summary_rows"]
    contract = metrics["contract"]
    stage3 = metrics["stage3_recommendations"]
    lines = [
        "# CP153-BM 1D Band 500 Stage 2 Model Zoo Report",
        "",
        f"- 생성 시각 UTC: `{metrics['created_at_utc']}`",
        f"- source_data_hash: `{contract['source_data_hash']}`",
        f"- provider/source: `{PROVIDER}`",
        f"- timeframe/horizon: `{TIMEFRAME}` / `{HORIZON}`",
        f"- feature_version: `{contract['feature_version']}`",
        f"- feature_set: `{FEATURE_SET}`",
        f"- feature columns ({contract['feature_set']['column_count']}): `{', '.join(contract['feature_set']['columns'])}`",
        f"- 500티커 full 실행 여부: `{metrics['full_universe_execution']}`",
        f"- save-run/DB/inference 저장: `false/false/false`",
        f"- Stage 0/1 stale 정정: `{contract['stage0_reference']['stale_contract_correction']}`",
        f"- OHLC ratio audit: `{contract['ohlc_ratio_audit']['status']}`",
        "",
        "## 실행 결과",
        "",
        _summary_table(rows),
        "",
        "## q별 주요 metric",
        "",
        _metric_table(rows),
        "",
        "## Stage 3 후보",
        "",
        f"- hard_pass 수: `{sum(1 for row in rows if row.get('decision') == 'hard_pass')}`",
        f"- near_pass 수: `{sum(1 for row in rows if row.get('decision') == 'near_pass')}`",
        f"- research_reserve 수: `{sum(1 for row in rows if row.get('decision') == 'research_reserve')}`",
        "",
        *[f"- {candidate_id}" for candidate_id in stage3],
        "",
        "## 실패 분류",
        "",
        _failure_table(rows),
        "",
        "## 판정 원칙",
        "",
        "- Stage 1 q별 baseline SOTA hard gate를 기준으로 판정했다.",
        "- band_width_ic/downside_width_ic는 date_cross_sectional_mean으로 집계했다.",
        "- line metric으로 후보를 탈락시키지 않았다.",
        "- CP72 대비 개선 표현은 사용하지 않고, yfinance 500 baseline SOTA 대비로만 해석했다.",
        "- Stage 2는 제품 후보 확정이 아니라 Stage 3/4 후보를 넓게 살리는 단계다.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| 후보 | family | q | eligible | epoch_mean_s | VRAM_MB | score | 판정 |\n"
        "|---|---|---|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for row in sorted(rows, key=lambda item: float(item.get("band_selection_score") or -1), reverse=True):
        lines.append(
            "| {candidate_id} | {family} | {q_label} | {eligible} | {epoch} | {vram} | {score} | {decision} |".format(
                candidate_id=row.get("candidate_id"),
                family=row.get("family"),
                q_label=row.get("q_label"),
                eligible=row.get("eligible_ticker_count") or "",
                epoch=_fmt(row.get("epoch_seconds_mean")),
                vram=_fmt(row.get("vram_peak_allocated_mb")),
                score=_fmt(row.get("band_selection_score")),
                decision=row.get("decision"),
            )
        )
    return "\n".join(lines)


def _failure_table(rows: list[dict[str, Any]]) -> str:
    header = "| 후보 | 판정 | failure_category | 주요 사유 |\n|---|---|---|---|"
    lines = [header]
    for row in rows:
        reasons = row.get("failure_reasons") or []
        reason_text = _compact_failure_reasons(reasons)
        lines.append(
            "| {candidate_id} | {decision} | {category} | {reason} |".format(
                candidate_id=row.get("candidate_id"),
                decision=row.get("decision"),
                category=row.get("failure_category") or "",
                reason=reason_text,
            )
        )
    return "\n".join(lines)


def _compact_failure_reasons(reasons: Any) -> str:
    if not reasons:
        return ""
    if not isinstance(reasons, list):
        return str(reasons)
    labels: list[str] = []
    for item in reasons:
        if isinstance(item, dict):
            metric = item.get("metric")
            if metric:
                labels.append(str(metric))
            else:
                labels.append(str(item.get("category") or item.get("reason") or item)[:60])
        else:
            labels.append(str(item))
    return ", ".join(labels[:8])


def _metric_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| 후보 | cov_abs | lower | upper | lower_abs | interval | width_ic | downside_ic | p90 | squeeze |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in sorted(rows, key=lambda item: (str(item.get("q_label")), str(item.get("candidate_id")))):
        lines.append(
            "| {candidate_id} | {coverage_abs_error} | {lower_breach_rate} | {upper_breach_rate} | "
            "{lower_breach_abs_error} | {asymmetric_interval_score} | {band_width_ic} | {downside_width_ic} | "
            "{p90_band_width} | {squeeze_breakout_rate} |".format(
                candidate_id=row.get("candidate_id"),
                coverage_abs_error=_fmt(row.get("coverage_abs_error")),
                lower_breach_rate=_fmt(row.get("lower_breach_rate")),
                upper_breach_rate=_fmt(row.get("upper_breach_rate")),
                lower_breach_abs_error=_fmt(row.get("lower_breach_abs_error")),
                asymmetric_interval_score=_fmt(row.get("asymmetric_interval_score")),
                band_width_ic=_fmt(row.get("band_width_ic")),
                downside_width_ic=_fmt(row.get("downside_width_ic")),
                p90_band_width=_fmt(row.get("p90_band_width")),
                squeeze_breakout_rate=_fmt(row.get("squeeze_breakout_rate")),
            )
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def write_metrics(metrics: dict[str, Any]) -> None:
    METRICS_PATH.write_text(
        json.dumps(_clean_json(metrics), ensure_ascii=False, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CP153 1D band 500 Stage 2 model zoo")
    parser.add_argument("--force", action="store_true", help="기존 PASS 후보도 재실행한다.")
    parser.add_argument("--candidates", nargs="*", default=None, help="특정 후보 id만 실행한다.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    stage0_metrics = _read_json(STAGE0_METRICS_PATH)
    price, indicators = _read_parquet_frames()
    overlay = prepare_stage2_snapshot_overlay()
    contract = build_contract_payload(price=price, indicators=indicators, stage0_metrics=stage0_metrics)
    selected_candidates = [
        candidate
        for candidate in CANDIDATES
        if args.candidates is None or candidate.candidate_id in set(args.candidates)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processes: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for candidate in selected_candidates:
        process = run_candidate(candidate, device=device, force=args.force)
        processes[candidate.candidate_id] = process
        evaluation_path = LOG_DIR / candidate.candidate_id / "evaluation.json"
        existing_evaluation = _read_json(evaluation_path) if evaluation_path.exists() and not args.force else None
        if (
            process.get("status") == "PASS"
            and existing_evaluation is not None
            and existing_evaluation.get("status") == "completed"
        ):
            evaluation = existing_evaluation
        else:
            try:
                evaluation = evaluate_candidate_checkpoint(
                    candidate=candidate,
                    process=process,
                    price=price,
                    indicators=indicators,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                evaluation = {
                    "candidate_id": candidate.candidate_id,
                    "status": "failed",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            evaluation_path.write_text(
                json.dumps(_clean_json(evaluation), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        evaluations[candidate.candidate_id] = evaluation

    rows, stage3_ids = score_and_classify(
        candidates=selected_candidates,
        processes=processes,
        evaluations=evaluations,
        stage0_metrics=stage0_metrics,
    )
    metrics = {
        "cp": "CP153-BM",
        "stage": "Stage 2 model zoo",
        "created_at_utc": _now_utc(),
        "contract": contract,
        "overlay": overlay,
        "candidates": [asdict(candidate) for candidate in selected_candidates],
        "processes": processes,
        "evaluations": evaluations,
        "summary_rows": rows,
        "stage3_recommendations": stage3_ids,
        "full_universe_execution": True,
        "selection_note": "제품 후보 확정이 아니라 Stage 3/4 후보를 넓게 살리는 단계",
        "outputs": {
            "report": str(REPORT_PATH),
            "metrics": str(METRICS_PATH),
            "summary_csv": str(SUMMARY_CSV_PATH),
            "logs": str(LOG_DIR),
        },
    }
    write_summary_csv(rows)
    write_metrics(metrics)
    write_report(metrics)
    print(
        json.dumps(
            {
                "status": "completed",
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
                "summary_csv": str(SUMMARY_CSV_PATH),
                "stage3_recommendations": stage3_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
