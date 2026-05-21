from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.band_calibration import PredictionSet, summarize_predictions  # noqa: E402
from ai.inference import _select_bundle_features, load_checkpoint, resolve_bundle, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from ai.train import (  # noqa: E402
    apply_feature_columns_to_splits,
    autocast_context,
    forward_model,
    make_loader,
    resolve_device,
    resolve_feature_columns,
    summarize_dataset_plan,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "cp138_bm_1w_band_context_backfill_revalidation_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp138_bm_1w_band_context_backfill_revalidation_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp138_bm_1w_band_candidate_registry.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp138_bm_1w_band_revalidation_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp138_bm_1w_band_context_backfill_revalidation_logs"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"
CP133_METRICS_PATH = PROJECT_ROOT / "docs" / "cp133_local_full_features_context_backfill_metrics.json"

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
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

DECISION_KEYS = [
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "asymmetric_interval_score",
    "interval_lower_penalty",
    "p90_band_width",
    "band_width_ic",
    "downside_width_ic",
]

REGIME_NAMES = [
    "rising_market",
    "falling_market",
    "high_volatility",
    "low_volatility",
    "high_atr",
    "low_atr",
    "wide_band",
    "narrow_band",
]

CONTEXT_COLUMNS = [
    "us10y",
    "yield_spread",
    "vix_close",
    "credit_spread_hy",
    "nh_nl_index",
    "ma200_pct",
    "regime_calm",
    "regime_neutral",
    "regime_stress",
    "revenue",
    "net_income",
    "equity",
    "eps",
    "roe",
    "debt_ratio",
    "has_macro",
    "has_breadth",
    "has_fundamentals",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    display_name: str
    model: str
    feature_set: str
    q_low: float
    q_high: float
    band_mode: str
    lower_band_loss_weight: float = 1.0
    upper_band_loss_weight: float = 1.0
    optional: bool = False
    note: str = ""


CANDIDATES = [
    Candidate(
        name="cnn_pvv_q10_direct",
        display_name="1W CNN-LSTM PVV q10 direct",
        model="cnn_lstm",
        feature_set="price_volatility_volume",
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        note="context backfill 이후 clean PVV 기준선",
    ),
    Candidate(
        name="cnn_pvv_q15_direct",
        display_name="1W CNN-LSTM PVV q15 direct",
        model="cnn_lstm",
        feature_set="price_volatility_volume",
        q_low=0.15,
        q_high=0.85,
        band_mode="direct",
        note="q10 대비 좁은 raw band 비교",
    ),
    Candidate(
        name="cnn_no_fundamentals_q10_direct",
        display_name="1W CNN-LSTM no fundamentals q10 direct",
        model="cnn_lstm",
        feature_set="no_fundamentals",
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        note="fundamentals 제외 안정성 확인",
    ),
    Candidate(
        name="cnn_full_q10_direct",
        display_name="1W CNN-LSTM full q10 direct",
        model="cnn_lstm",
        feature_set="full_features",
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        note="CP133 context가 채워진 full_features 효과 확인",
    ),
    Candidate(
        name="cnn_full_q10_direct_lower_guard_w1p5",
        display_name="1W CNN-LSTM full q10 direct lower guard 1.5",
        model="cnn_lstm",
        feature_set="full_features",
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        lower_band_loss_weight=1.5,
        note="CP125 하방 방어형 후보 재검증",
    ),
    Candidate(
        name="tide_pvv_q15_param",
        display_name="1W TiDE PVV q15 param",
        model="tide",
        feature_set="price_volatility_volume",
        q_low=0.15,
        q_high=0.85,
        band_mode="param",
        note="이전 TiDE 대안 후보의 하락장 lower breach 재확인",
    ),
    Candidate(
        name="context_light_q10_direct",
        display_name="1W CNN-LSTM context light q10 direct",
        model="cnn_lstm",
        feature_set="context_light",
        q_low=0.10,
        q_high=0.90,
        band_mode="direct",
        optional=True,
        note="계약에 정의되어 있을 때만 실행하는 선택 후보",
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any, digits: int = 6) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _sha256_file(path: Path, *, short: int = 16) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:short]


def _sha256_text(text: str, *, short: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:short]


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values: list[float] = []
    for record in records:
        source = record.get("metrics") if isinstance(record.get("metrics"), dict) else record
        value = _safe_float(source.get(key))
        if value is not None:
            values.append(value)
    return _mean_std(values)


def _band_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    source = summary or {}
    if isinstance(source.get("band_metrics"), dict):
        source = source["band_metrics"]
    return {key: source.get(key) for key in BAND_METRIC_KEYS}


def _snapshot_meta() -> dict[str, Any]:
    base = Path(os.environ.get("LENS_LOCAL_SNAPSHOT_DIR") or PROJECT_ROOT / "data" / "parquet")
    price = base / "price_data_yfinance_1W.parquet"
    indicators = base / "indicators_yfinance_1W.parquet"
    return {
        "base_dir": str(base),
        "price_path": str(price),
        "price_exists": price.exists(),
        "price_size_bytes": price.stat().st_size if price.exists() else None,
        "price_mtime": datetime.fromtimestamp(price.stat().st_mtime).isoformat(timespec="seconds") if price.exists() else None,
        "price_checksum": _sha256_file(price),
        "indicator_path": str(indicators),
        "indicator_exists": indicators.exists(),
        "indicator_size_bytes": indicators.stat().st_size if indicators.exists() else None,
        "indicator_mtime": datetime.fromtimestamp(indicators.stat().st_mtime).isoformat(timespec="seconds") if indicators.exists() else None,
        "indicator_file_checksum": _sha256_file(indicators),
    }


def _context_column_checksum() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W.parquet"
    if not path.exists():
        return {"available": False, "reason": "indicator parquet missing"}
    frame = pd.read_parquet(path)
    columns = [column for column in ["ticker", "date", *CONTEXT_COLUMNS] if column in frame.columns]
    if not {"ticker", "date"}.issubset(columns):
        return {"available": False, "reason": "ticker/date columns missing", "columns": columns}
    subset = frame[columns].copy()
    subset["ticker"] = subset["ticker"].astype(str).str.upper()
    subset["date"] = pd.to_datetime(subset["date"]).dt.strftime("%Y-%m-%d")
    subset = subset.sort_values(["ticker", "date"]).reset_index(drop=True)
    numeric_columns = [column for column in columns if column not in {"ticker", "date"}]
    nonfinite_count = 0
    for column in numeric_columns:
        values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype="float64", copy=False)
        nonfinite_count += int((~np.isfinite(values)).sum())
    digest_source = pd.util.hash_pandas_object(subset, index=False).values.tobytes()
    checksum = hashlib.sha256(digest_source).hexdigest()[:16]
    return {
        "available": True,
        "checksum": checksum,
        "row_count": int(len(subset)),
        "columns": columns,
        "context_columns_present": numeric_columns,
        "context_feature_nonfinite_count": nonfinite_count,
    }


def _cp133_reference() -> dict[str, Any]:
    payload = _read_json(CP133_METRICS_PATH)
    after_1w = ((payload.get("after_indicator_stats") or {}).get("1W") or {})
    before_1w = ((payload.get("before_indicator_stats") or {}).get("1W") or {})
    return {
        "source": str(CP133_METRICS_PATH.relative_to(PROJECT_ROOT)) if CP133_METRICS_PATH.exists() else None,
        "after_indicator_value_checksum_1w": after_1w.get("indicator_value_checksum"),
        "before_indicator_value_checksum_1w": before_1w.get("indicator_value_checksum"),
        "after_feature_non_finite_count_1w": after_1w.get("feature_non_finite_count"),
        "after_flag_true_rate_1w": after_1w.get("flag_true_rate"),
        "after_non_zero_rate_1w": after_1w.get("non_zero_rate"),
        "after_rows_1w": after_1w.get("rows"),
        "after_ticker_count_1w": after_1w.get("ticker_count"),
    }


def _feature_set_status(name: str) -> dict[str, Any]:
    try:
        columns = resolve_feature_columns(name)
        return {
            "feature_set": name,
            "exists": True,
            "column_count": len(columns),
            "columns": columns,
            "atr_ratio_in_columns": "atr_ratio" in columns,
            "intraday_range_ratio_in_columns": "intraday_range_ratio" in columns,
        }
    except Exception as exc:
        return {"feature_set": name, "exists": False, "error": str(exc)}


def _feature_finite_summary(bundle: Any) -> dict[str, Any]:
    total = 0
    nonfinite = 0
    first_failure: dict[str, Any] | None = None
    mean = getattr(bundle, "mean", None)
    std = getattr(bundle, "std", None)
    if hasattr(bundle, "features"):
        features = bundle.features
        total = int(features.numel())
        mask = ~torch.isfinite(features)
        nonfinite = int(mask.sum().item())
        if nonfinite:
            first_failure = {"index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    else:
        for ticker in sorted({str(ticker) for ticker, _ in bundle.sample_refs}):
            array = torch.from_numpy(bundle.ticker_arrays[str(ticker)]["features"]).to(dtype=torch.float32)
            if mean is not None and std is not None:
                array = (array - mean.view(1, -1)) / std.view(1, -1)
            total += int(array.numel())
            mask = ~torch.isfinite(array)
            count = int(mask.sum().item())
            nonfinite += count
            if count and first_failure is None:
                first_failure = {"ticker": ticker, "index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _target_finite_summary(bundle: Any, horizon: int) -> dict[str, Any]:
    total = 0
    nonfinite = 0
    first_failure: dict[str, Any] | None = None
    if hasattr(bundle, "raw_future_returns"):
        target = bundle.raw_future_returns
        total = int(target.numel())
        mask = ~torch.isfinite(target)
        nonfinite = int(mask.sum().item())
        if nonfinite:
            first_failure = {"index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    else:
        for sample_index, (ticker, end_idx) in enumerate(bundle.sample_refs):
            closes = bundle.ticker_arrays[str(ticker)]["closes"]
            anchor = float(closes[int(end_idx)])
            future = closes[int(end_idx) + 1 : int(end_idx) + 1 + horizon]
            target = torch.tensor((future / anchor) - 1.0, dtype=torch.float32)
            total += int(target.numel())
            mask = ~torch.isfinite(target)
            count = int(mask.sum().item())
            nonfinite += count
            if count and first_failure is None:
                first_failure = {"sample_index": int(sample_index), "ticker": str(ticker), "end_idx": int(end_idx)}
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _finite_for_feature_set(
    feature_set: str,
    train_bundle: Any,
    val_bundle: Any,
    test_bundle: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    horizon: int,
) -> dict[str, Any]:
    columns = resolve_feature_columns(feature_set)
    train_selected, val_selected, test_selected, _, _ = apply_feature_columns_to_splits(
        train_bundle,
        val_bundle,
        test_bundle,
        mean,
        std,
        columns,
    )
    finite = {
        "train": {
            "features": _feature_finite_summary(train_selected),
            "targets": _target_finite_summary(train_selected, horizon),
        },
        "val": {
            "features": _feature_finite_summary(val_selected),
            "targets": _target_finite_summary(val_selected, horizon),
        },
        "test": {
            "features": _feature_finite_summary(test_selected),
            "targets": _target_finite_summary(test_selected, horizon),
        },
    }
    feature_nonfinite = sum(int(split["features"]["nonfinite_count"]) for split in finite.values())
    target_nonfinite = sum(int(split["targets"]["nonfinite_count"]) for split in finite.values())
    return {
        "feature_set": feature_set,
        "feature_columns": columns,
        "feature_column_count": len(columns),
        "finite": finite,
        "feature_nonfinite_count": feature_nonfinite,
        "target_nonfinite_count": target_nonfinite,
    }


def build_preflight_payload() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = resolve_data_fingerprint("1W", market_data_provider="yfinance")
    snapshot = _snapshot_meta()
    context_checksum = _context_column_checksum()
    cp133 = _cp133_reference()
    feature_set_names = sorted({candidate.feature_set for candidate in CANDIDATES} | {"price_volatility_volume", "no_fundamentals", "full_features"})
    feature_sets = {name: _feature_set_status(name) for name in feature_set_names}
    train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
        timeframe="1W",
        seq_len=104,
        horizon=4,
        include_future_covariate=True,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        market_data_provider="yfinance",
    )
    finite_by_feature_set: dict[str, Any] = {}
    feature_nonfinite_total = 0
    target_nonfinite_total = 0
    for feature_set, status in feature_sets.items():
        if not status.get("exists"):
            continue
        finite_payload = _finite_for_feature_set(feature_set, train_bundle, val_bundle, test_bundle, mean, std, 4)
        finite_by_feature_set[feature_set] = finite_payload
        feature_nonfinite_total += int(finite_payload["feature_nonfinite_count"])
        target_nonfinite_total += int(finite_payload["target_nonfinite_count"])
    plan_summary = summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle)
    expected_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    pass_gate = bool(
        snapshot.get("price_exists")
        and snapshot.get("indicator_exists")
        and source_hash
        and context_checksum.get("available")
        and int(context_checksum.get("context_feature_nonfinite_count") or 0) == 0
        and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and len(MODEL_FEATURE_COLUMNS) == 36
        and "atr_ratio" not in MODEL_FEATURE_COLUMNS
        and "intraday_range_ratio" not in MODEL_FEATURE_COLUMNS
        and feature_sets["price_volatility_volume"]["exists"]
        and feature_sets["no_fundamentals"]["exists"]
        and feature_sets["full_features"]["exists"]
        and feature_nonfinite_total == 0
        and target_nonfinite_total == 0
    )
    payload = {
        "cp": "CP138-BM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PREFLIGHT_PASS" if pass_gate else "PREFLIGHT_FAIL",
        "preflight_gate_pass": pass_gate,
        "environment": {
            "python_executable": sys.executable,
            "expected_venv_python": str(expected_python),
            "venv_python_required": True,
            "venv_python_match": Path(sys.executable).resolve() == expected_python.resolve() if expected_python.exists() else None,
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "LENS_USE_LOCAL_SNAPSHOTS": os.environ.get("LENS_USE_LOCAL_SNAPSHOTS"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
        },
        "scope_guard": {
            "role": "band_model",
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "off",
            "composite": False,
            "frontend_modified": False,
            "supabase_bulk_read": False,
            "yfinance_live_fetch": False,
            "eodhd_call": False,
            "new_target_implemented": False,
            "calibration_used_for_decision": False,
        },
        "data": {
            "provider": "yfinance",
            "source": "yfinance",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "source_data_hash": source_hash,
            "snapshot": snapshot,
            "context_column_checksum": context_checksum,
            "cp133_reference": cp133,
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
            "intraday_range_ratio_in_model_features": "intraday_range_ratio" in MODEL_FEATURE_COLUMNS,
            "feature_sets": feature_sets,
            "finite_by_feature_set": finite_by_feature_set,
            "feature_nonfinite_count_total": feature_nonfinite_total,
            "target_nonfinite_count_total": target_nonfinite_total,
            "dataset_plan": plan_summary,
            "test_target_points_per_run": int(plan_summary.get("test_samples") or 0) * 4,
        },
    }
    _write_json(PREFLIGHT_PATH, payload)
    return payload


def _extract_train_json(stdout: str) -> dict[str, Any]:
    marker = "[EXIT-MARKER {\"step\": \"before_result_json\""
    marker_index = stdout.rfind(marker)
    search_start = stdout.find("\n", marker_index) + 1 if marker_index >= 0 else 0
    decoder = json.JSONDecoder()
    index = search_start
    while index < len(stdout):
        if stdout[index] != "{":
            index += 1
            continue
        try:
            parsed, end_index = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(parsed, dict) and "run_id" in parsed and "best_metrics" in parsed:
            return parsed
        index += max(end_index, 1)
    raise ValueError("ai.train stdout에서 결과 JSON을 찾지 못했습니다.")


def _epoch_summary(stdout: str) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    remaining_seconds: list[float] = []
    vram_values: list[float] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        epoch_value = _safe_float(payload.get("epoch_seconds"))
        if epoch_value is not None:
            epoch_seconds.append(epoch_value)
        remaining_value = _safe_float(payload.get("estimated_remaining_seconds"))
        if remaining_value is not None:
            remaining_seconds.append(remaining_value)
        vram_value = _safe_float(payload.get("vram_peak_allocated_mb"))
        if vram_value is not None:
            vram_values.append(vram_value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "estimated_remaining_seconds_last": remaining_seconds[-1] if remaining_seconds else None,
        "vram_peak_allocated_mb": max(vram_values) if vram_values else None,
    }


def _command_for_run(candidate: Candidate, seed: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        candidate.model,
        "--timeframe",
        "1W",
        "--horizon",
        "4",
        "--seq-len",
        "104",
        "--feature-set",
        candidate.feature_set,
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--q-low",
        str(candidate.q_low),
        "--q-high",
        str(candidate.q_high),
        "--lambda-band",
        "2.0",
        "--lower-band-loss-weight",
        str(candidate.lower_band_loss_weight),
        "--upper-band-loss-weight",
        str(candidate.upper_band_loss_weight),
        "--band-mode",
        candidate.band_mode,
        "--checkpoint-selection",
        "band_gate",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--amp-dtype",
        args.amp_dtype,
        "--no-compile",
        "--no-wandb",
        "--num-workers",
        "0",
        "--explicit-cuda-cleanup",
        "--market-data-provider",
        "yfinance",
        "--seed",
        str(seed),
        "--local-log-dir",
        str(LOG_DIR / "local_runs" / candidate.name / f"seed_{seed}"),
    ]
    if candidate.model == "cnn_lstm":
        command.extend(["--fp32-modules", "lstm,heads"])
    if candidate.model == "tide":
        command.append("--use-future-covariate")
    return command


def _design_needed_record(candidate: Candidate, reason: str) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "seed": None,
        "candidate_config": asdict(candidate),
        "execution_status": "DESIGN_NEEDED",
        "exit_code": None,
        "failed_with_reason": reason,
        "validation_band_metrics": {},
        "test_band_metrics_read_only": {},
        "test_metrics_present": False,
        "gate": {},
        "validation_regime_metrics": {"regimes": {}},
    }


def _completed_existing_run(existing: dict[str, Any], candidate: Candidate, seed: int, source_hash: str | None) -> dict[str, Any] | None:
    for summary in existing.get("candidate_summaries", []):
        if summary.get("candidate") != candidate.name:
            continue
        for run in summary.get("runs", []):
            run_source_hash = run.get("source_data_hash") or ((run.get("dataset_plan") or {}).get("source_data_hash"))
            if (
                run.get("seed") == seed
                and run.get("execution_status") in {"PASS", "REUSED_CP138"}
                and run_source_hash == source_hash
                and (run.get("candidate_config") or {}).get("feature_set") == candidate.feature_set
                and _safe_float((run.get("candidate_config") or {}).get("q_low")) == candidate.q_low
                and _safe_float((run.get("candidate_config") or {}).get("q_high")) == candidate.q_high
            ):
                reused = dict(run)
                reused["execution_status"] = "PASS"
                reused["reused_from_existing_metrics"] = True
                return reused
    return None


def _run_training(candidate: Candidate, seed: int, args: argparse.Namespace, preflight: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = ((preflight.get("data") or {}).get("source_data_hash"))
    reused = _completed_existing_run(existing, candidate, seed, source_hash)
    if reused is not None and not args.force:
        print(f"[CP138] 재사용: {candidate.name} seed={seed}", flush=True)
        return reused

    feature_status = (((preflight.get("data") or {}).get("feature_sets") or {}).get(candidate.feature_set) or {})
    if not feature_status.get("exists"):
        reason = f"feature_set 미구현: {feature_status.get('error') or candidate.feature_set}"
        print(f"[CP138] design_needed: {candidate.name} - {reason}", flush=True)
        return _design_needed_record(candidate, reason)

    log_path = LOG_DIR / f"{candidate.name}_seed{seed}.stdout.log"
    command = _command_for_run(candidate, seed, args)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCHDYNAMO_DISABLE": "1",
            "WANDB_MODE": "disabled",
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    print(f"[CP138] 시작: {candidate.name} seed={seed}", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        execution_status = "PASS" if exit_code == 0 else "FAIL"
        timeout_expired = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        exit_code = None
        execution_status = "TIMEOUT"
        timeout_expired = True
    elapsed = time.perf_counter() - started
    ended_at = datetime.now().isoformat(timespec="seconds")
    log_path.write_text(
        "\n".join(
            [
                "# command: " + " ".join(command),
                "# environment: yfinance local snapshots, WANDB_MODE=disabled",
                "# stdout",
                stdout,
                "# stderr",
                stderr,
                f"# exit_code: {exit_code}",
                f"# elapsed_seconds: {elapsed:.4f}",
                f"# timeout_expired: {timeout_expired}",
            ]
        ),
        encoding="utf-8",
    )
    record: dict[str, Any] = {
        "candidate": candidate.name,
        "seed": seed,
        "candidate_config": asdict(candidate),
        "command": command,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "execution_status": execution_status,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "timeout_expired": timeout_expired,
        "epoch_summary": _epoch_summary(stdout),
    }
    if execution_status != "PASS":
        record["failed_with_reason"] = "timeout" if timeout_expired else "nonzero_exit"
        record["stderr_tail"] = stderr[-4000:]
        print(f"[CP138] 실패: {candidate.name} seed={seed} status={execution_status}", flush=True)
        return record
    try:
        result = _extract_train_json(stdout)
    except Exception as exc:
        record["execution_status"] = "FAIL"
        record["failed_with_reason"] = f"result_json_parse_failed: {exc}"
        record["stderr_tail"] = stderr[-4000:]
        print(f"[CP138] 결과 JSON 파싱 실패: {candidate.name} seed={seed}", flush=True)
        return record
    best_metrics = result.get("best_metrics") if isinstance(result.get("best_metrics"), dict) else {}
    test_metrics = result.get("test_metrics") if isinstance(result.get("test_metrics"), dict) else {}
    dataset_plan = result.get("dataset_plan") if isinstance(result.get("dataset_plan"), dict) else {}
    record.update(
        {
            "run_id": result.get("run_id"),
            "checkpoint_path": result.get("checkpoint_path"),
            "source_data_hash": result.get("source_data_hash") or dataset_plan.get("source_data_hash"),
            "local_log_dir": result.get("local_log_dir"),
            "feature_version": result.get("feature_version") or FEATURE_CONTRACT_VERSION,
            "n_features": result.get("n_features"),
            "feature_columns": result.get("feature_columns"),
            "validation_band_metrics": _band_metrics(best_metrics),
            "test_band_metrics_read_only": _band_metrics(test_metrics),
            "test_metrics_present": bool(test_metrics),
            "gate": {
                "checkpoint_selection": best_metrics.get("checkpoint_selection"),
                "gate_type": best_metrics.get("gate_type"),
                "gate_failed": best_metrics.get("gate_failed"),
                "band_gate_pass": best_metrics.get("band_gate_pass"),
                "role": best_metrics.get("role"),
                "best_epoch": best_metrics.get("best_epoch"),
                "selected_reason": best_metrics.get("selected_reason"),
            },
            "wandb_status": result.get("wandb_status"),
            "dataset_plan": dataset_plan,
        }
    )
    print(f"[CP138] 완료: {candidate.name} seed={seed} run_id={record.get('run_id')}", flush=True)
    return record


def _collect_predictions(
    *,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
) -> tuple[PredictionSet, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    registry = resolve_checkpoint_ticker_registry(config, str(config["timeframe"]))
    bundle = resolve_bundle(
        split_name=split,
        timeframe=str(config["timeframe"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        tickers=config.get("tickers"),
        limit_tickers=config.get("limit_tickers"),
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    bundle = _select_bundle_features(bundle, list(config.get("feature_columns") or []))
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    line_chunks: list[torch.Tensor] = []
    lower_chunks: list[torch.Tensor] = []
    upper_chunks: list[torch.Tensor] = []
    line_target_chunks: list[torch.Tensor] = []
    band_target_chunks: list[torch.Tensor] = []
    raw_target_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_ids, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype=amp_dtype):
                output = forward_model(model, features, ticker_ids, future_covariates)
            line, lower, upper = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            line_chunks.append(line)
            lower_chunks.append(lower)
            upper_chunks.append(upper)
            line_target_chunks.append(line_target.detach().cpu())
            band_target_chunks.append(band_target.detach().cpu())
            raw_target_chunks.append(raw_future_returns.detach().cpu())

    predictions = PredictionSet(
        line=torch.cat(line_chunks, dim=0),
        lower=torch.cat(lower_chunks, dim=0),
        upper=torch.cat(upper_chunks, dim=0),
        line_target=torch.cat(line_target_chunks, dim=0),
        band_target=torch.cat(band_target_chunks, dim=0),
        raw_future_returns=torch.cat(raw_target_chunks, dim=0),
        metadata=bundle.metadata.reset_index(drop=True),
    )
    return predictions, {"config": config}


def _slice_predictions(predictions: PredictionSet, mask_or_indices: torch.Tensor) -> PredictionSet:
    if mask_or_indices.dtype == torch.bool:
        indices = mask_or_indices.nonzero(as_tuple=False).flatten()
    else:
        indices = mask_or_indices.flatten().to(dtype=torch.long)
    metadata = predictions.metadata.iloc[indices.detach().cpu().numpy()].reset_index(drop=True)
    return PredictionSet(
        line=predictions.line.index_select(0, indices),
        lower=predictions.lower.index_select(0, indices),
        upper=predictions.upper.index_select(0, indices),
        line_target=predictions.line_target.index_select(0, indices),
        band_target=predictions.band_target.index_select(0, indices),
        raw_future_returns=predictions.raw_future_returns.index_select(0, indices),
        metadata=metadata,
    )


def _summarize(predictions: PredictionSet, line: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    return _band_metrics(
        summarize_predictions(
            predictions,
            line=line,
            lower=lower,
            upper=upper,
            q_low=float(config.get("q_low", 0.1)),
            q_high=float(config.get("q_high", 0.9)),
            line_target_type=str(config.get("line_target_type", "raw_future_return")),
            band_target_type=str(config.get("band_target_type", "raw_future_return")),
        )
    )


def _atr_values(metadata: pd.DataFrame) -> torch.Tensor:
    path = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W.parquet"
    if not path.exists():
        return torch.full((len(metadata),), float("nan"), dtype=torch.float32)
    indicators = pd.read_parquet(path, columns=["ticker", "date", "atr_ratio", "source", "provider"])
    if "provider" in indicators.columns:
        indicators = indicators[indicators["provider"] == "yfinance"].copy()
    if "source" in indicators.columns:
        indicators = indicators[indicators["source"] == "yfinance"].copy()
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    lookup = {
        (str(row.ticker).upper(), row.date): float(row.atr_ratio)
        for row in indicators.itertuples(index=False)
        if pd.notna(row.atr_ratio)
    }
    values = []
    for row in metadata.itertuples(index=False):
        ticker = str(getattr(row, "ticker")).upper()
        date = pd.to_datetime(getattr(row, "asof_date")).date()
        values.append(lookup.get((ticker, date), float("nan")))
    return torch.tensor(values, dtype=torch.float32)


def _regime_masks(predictions: PredictionSet, *, lower: torch.Tensor, upper: torch.Tensor) -> dict[str, torch.Tensor]:
    width = (upper - lower).mean(dim=1)
    realized_volatility = predictions.raw_future_returns.std(dim=1, unbiased=False)
    realized_h4_return = predictions.raw_future_returns[:, -1]
    atr = _atr_values(predictions.metadata)
    high_vol_threshold = torch.median(realized_volatility)
    width_threshold = torch.median(width)
    finite_atr = atr[torch.isfinite(atr)]
    if finite_atr.numel() > 0:
        atr_threshold = torch.median(finite_atr)
        high_atr = torch.isfinite(atr) & (atr >= atr_threshold)
        low_atr = torch.isfinite(atr) & (atr < atr_threshold)
    else:
        high_atr = torch.zeros_like(width, dtype=torch.bool)
        low_atr = torch.zeros_like(width, dtype=torch.bool)
    return {
        "rising_market": realized_h4_return >= 0.0,
        "falling_market": realized_h4_return < 0.0,
        "high_volatility": realized_volatility >= high_vol_threshold,
        "low_volatility": realized_volatility < high_vol_threshold,
        "high_atr": high_atr,
        "low_atr": low_atr,
        "wide_band": width >= width_threshold,
        "narrow_band": width < width_threshold,
    }


def _regime_metrics(
    predictions: PredictionSet,
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    masks = _regime_masks(predictions, lower=lower, upper=upper)
    for name in REGIME_NAMES:
        mask = masks[name]
        count = int(mask.sum().item())
        if count < 10:
            regimes[name] = {"sample_count": count, "metrics": {}}
            continue
        sliced = _slice_predictions(predictions, mask)
        indices = mask.nonzero(as_tuple=False).flatten()
        regimes[name] = {
            "sample_count": count,
            "metrics": _summarize(
                sliced,
                line.index_select(0, indices),
                lower.index_select(0, indices),
                upper.index_select(0, indices),
                config,
            ),
        }
    return regimes


def _attach_validation_regimes(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if record.get("execution_status") != "PASS":
        record.setdefault("validation_regime_metrics", {"regimes": {}})
        return record
    checkpoint_path = record.get("checkpoint_path")
    if not checkpoint_path:
        record["validation_regime_metrics"] = {"error": "checkpoint_missing", "regimes": {}}
        return record
    try:
        predictions, context = _collect_predictions(
            checkpoint_path=PROJECT_ROOT / str(checkpoint_path),
            split="val",
            device_name=args.device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        )
        config = context["config"]
        regimes = _regime_metrics(
            predictions,
            line=predictions.line,
            lower=predictions.lower,
            upper=predictions.upper,
            config=config,
        )
        record["validation_regime_metrics"] = {"regimes": regimes}
    except Exception as exc:
        record["validation_regime_metrics"] = {"error": str(exc), "regimes": {}}
    return record


def _aggregate_regime_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime_name in REGIME_NAMES:
        regime_result: dict[str, Any] = {}
        sample_counts: list[float] = []
        for key in BAND_METRIC_KEYS:
            values: list[float] = []
            for run in runs:
                regime_payload = (((run.get("validation_regime_metrics") or {}).get("regimes") or {}).get(regime_name) or {})
                sample_count = _safe_float(regime_payload.get("sample_count"))
                if sample_count is not None:
                    sample_counts.append(sample_count)
                value = _safe_float((regime_payload.get("metrics") or {}).get(key))
                if value is not None:
                    values.append(value)
            regime_result[key] = _mean_std(values)
        regime_result["sample_count"] = _mean_std(sample_counts)
        result[regime_name] = regime_result
    return result


def _regime_mean(summary: dict[str, Any], regime: str, key: str) -> float | None:
    return _safe_float(((((summary.get("validation_regime_metric_stats") or {}).get(regime) or {}).get(key) or {}).get("mean")))


def _metric_mean(summary: dict[str, Any], key: str) -> float | None:
    return _safe_float((((summary.get("validation_metric_stats") or {}).get(key) or {}).get("mean")))


def _aggregate_candidate(candidate: Candidate, runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("execution_status") == "PASS"]
    design_needed = [run for run in runs if run.get("execution_status") == "DESIGN_NEEDED"]
    metric_stats = {key: _metric_stats([run.get("validation_band_metrics") or {} for run in completed], key) for key in BAND_METRIC_KEYS}
    gate_pass_count = sum(
        1
        for run in completed
        if (run.get("gate") or {}).get("band_gate_pass") and not (run.get("gate") or {}).get("gate_failed")
    )
    test_target_points = [
        int(((run.get("dataset_plan") or {}).get("test_samples") or 0)) * 4
        for run in completed
        if isinstance(run.get("dataset_plan"), dict)
    ]
    return {
        "candidate": candidate.name,
        "candidate_config": asdict(candidate),
        "runs": runs,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "design_needed_count": len(design_needed),
        "band_gate_pass_count": gate_pass_count,
        "band_gate_pass_rate": gate_pass_count / len(completed) if completed else 0.0,
        "validation_metric_stats": metric_stats,
        "validation_regime_metric_stats": _aggregate_regime_metrics(completed),
        "test_metric_opened_run_count": sum(1 for run in completed if run.get("test_metrics_present")),
        "test_target_points_per_run": test_target_points[0] if test_target_points else None,
        "test_exposure_count": sum(test_target_points),
        "test_metric_policy": "ai.train 결과 JSON에 test_metrics가 포함되어 read-only로 기록했으며 후보 선택에는 사용하지 않음",
    }


def _criteria() -> dict[str, Any]:
    return {
        "band_gate_pass_rate_min": 1.0,
        "coverage_abs_error_max": 0.05,
        "lower_breach_rate_max": 0.18,
        "falling_lower_breach_rate_max": 0.20,
        "high_vol_lower_breach_rate_max": 0.20,
        "high_atr_lower_breach_rate_max": 0.20,
        "band_width_ic_min": 0.15,
        "downside_width_ic_min": 0.0,
        "p90_band_width_max": 0.35,
        "interval_relative_to_best_max": 1.10,
    }


def _classify_candidate(summary: dict[str, Any], criteria: dict[str, Any], best_interval: float | None) -> dict[str, Any]:
    if summary.get("design_needed_count"):
        return {
            "category": "design_needed",
            "checks": {"feature_set_available": False},
            "failures": ["feature_set 미구현"],
        }
    if summary.get("completed_run_count", 0) <= 0:
        return {
            "category": "rejected",
            "checks": {"completed_run": False},
            "failures": ["완료된 실행 없음"],
        }
    interval = _metric_mean(summary, "asymmetric_interval_score")
    interval_limit = best_interval * criteria["interval_relative_to_best_max"] if best_interval is not None else None
    checks = {
        "band_gate_pass": float(summary.get("band_gate_pass_rate") or 0.0) >= criteria["band_gate_pass_rate_min"],
        "coverage_abs_error": (_metric_mean(summary, "coverage_abs_error") or float("inf")) <= criteria["coverage_abs_error_max"],
        "lower_breach_rate": (_metric_mean(summary, "lower_breach_rate") or float("inf")) <= criteria["lower_breach_rate_max"],
        "falling_regime_lower_breach_rate": (
            _regime_mean(summary, "falling_market", "lower_breach_rate") or float("inf")
        )
        <= criteria["falling_lower_breach_rate_max"],
        "high_volatility_lower_breach_rate": (
            _regime_mean(summary, "high_volatility", "lower_breach_rate") or float("inf")
        )
        <= criteria["high_vol_lower_breach_rate_max"],
        "high_atr_lower_breach_rate": (
            _regime_mean(summary, "high_atr", "lower_breach_rate") or float("inf")
        )
        <= criteria["high_atr_lower_breach_rate_max"],
        "band_width_ic": (_metric_mean(summary, "band_width_ic") or -float("inf")) > criteria["band_width_ic_min"],
        "downside_width_ic": (_metric_mean(summary, "downside_width_ic") or -float("inf")) >= criteria["downside_width_ic_min"],
        "p90_band_width": (_metric_mean(summary, "p90_band_width") or float("inf")) <= criteria["p90_band_width_max"],
        "asymmetric_interval_score_competitive": (
            interval is not None and interval_limit is not None and interval <= interval_limit
        ),
    }
    failures = [key for key, ok in checks.items() if not ok]
    if not failures:
        category = "selectable_verified"
    elif (
        not checks["band_gate_pass"]
        or (_metric_mean(summary, "coverage_abs_error") or float("inf")) > 0.08
        or (_metric_mean(summary, "lower_breach_rate") or float("inf")) > 0.25
        or (_metric_mean(summary, "band_width_ic") or -float("inf")) <= 0.0
    ):
        category = "rejected"
    else:
        category = "experiment_record"
    return {
        "category": category,
        "checks": checks,
        "failures": failures,
        "best_interval_reference": best_interval,
        "interval_limit": interval_limit,
    }


def _select_default(candidate_summaries: list[dict[str, Any]]) -> str | None:
    verified = [item for item in candidate_summaries if (item.get("decision") or {}).get("category") == "selectable_verified"]
    if not verified:
        return None

    def priority(summary: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
        name = str(summary.get("candidate") or "")
        minimal_rank = {
            "cnn_pvv_q10_direct": 0,
            "cnn_no_fundamentals_q10_direct": 1,
            "cnn_full_q10_direct_lower_guard_w1p5": 2,
            "cnn_full_q10_direct": 3,
            "tide_pvv_q15_param": 4,
            "cnn_pvv_q15_direct": 5,
        }.get(name, 9)
        return (
            _metric_mean(summary, "coverage_abs_error") or float("inf"),
            _metric_mean(summary, "lower_breach_rate") or float("inf"),
            _regime_mean(summary, "falling_market", "lower_breach_rate") or float("inf"),
            _metric_mean(summary, "asymmetric_interval_score") or float("inf"),
            _metric_mean(summary, "p90_band_width") or float("inf"),
            minimal_rank,
        )

    return sorted(verified, key=priority)[0].get("candidate")


def _apply_decisions(candidate_summaries: list[dict[str, Any]]) -> None:
    criteria = _criteria()
    intervals = [
        _metric_mean(summary, "asymmetric_interval_score")
        for summary in candidate_summaries
        if summary.get("completed_run_count", 0) > 0 and _metric_mean(summary, "asymmetric_interval_score") is not None
    ]
    best_interval = min(intervals) if intervals else None
    for summary in candidate_summaries:
        summary["decision"] = _classify_candidate(summary, criteria, best_interval)
    selected = _select_default(candidate_summaries)
    for summary in candidate_summaries:
        decision = summary.setdefault("decision", {})
        if decision.get("category") == "selectable_verified" and summary.get("candidate") == selected:
            decision["category"] = "recommended_default"


def _why_not_default(summary: dict[str, Any]) -> str:
    decision = summary.get("decision") or {}
    if decision.get("category") == "recommended_default":
        return "현재 기본 후보"
    failures = decision.get("failures") or []
    if failures:
        return ", ".join(failures)
    return "기본 후보보다 validation 균형이 약함"


def _registry_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("candidate_config") or {}
    stats = summary.get("validation_metric_stats") or {}
    decision = summary.get("decision") or {}
    category = decision.get("category") or "experiment_record"
    strength = (
        f"val cov_abs={_fmt((stats.get('coverage_abs_error') or {}).get('mean'))}, "
        f"lower={_fmt((stats.get('lower_breach_rate') or {}).get('mean'))}, "
        f"interval={_fmt((stats.get('asymmetric_interval_score') or {}).get('mean'))}, "
        f"bw_ic={_fmt((stats.get('band_width_ic') or {}).get('mean'))}"
    )
    weakness = "검증 기준 통과" if category in {"recommended_default", "selectable_verified"} else _why_not_default(summary)
    return {
        "category": category,
        "display_name": config.get("display_name"),
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": config.get("model"),
        "feature_set": config.get("feature_set"),
        "target_type": "raw_future_return",
        "band_mode": config.get("band_mode"),
        "q_low": config.get("q_low"),
        "q_high": config.get("q_high"),
        "lower_band_loss_weight": config.get("lower_band_loss_weight"),
        "strength_summary": strength,
        "weakness_summary": weakness,
        "best_use_case": _best_use_case(summary),
        "why_not_default": _why_not_default(summary),
        "key_metrics": {
            key: {
                "validation_mean": (stats.get(key) or {}).get("mean"),
                "validation_std": (stats.get(key) or {}).get("std"),
            }
            for key in DECISION_KEYS
        },
        "regime_key_metrics": {
            "falling_lower_breach_rate": _regime_mean(summary, "falling_market", "lower_breach_rate"),
            "high_volatility_lower_breach_rate": _regime_mean(summary, "high_volatility", "lower_breach_rate"),
            "high_atr_lower_breach_rate": _regime_mean(summary, "high_atr", "lower_breach_rate"),
        },
        "seed_count": summary.get("run_count"),
        "completed_seed_count": summary.get("completed_run_count"),
        "band_gate_pass_rate": summary.get("band_gate_pass_rate"),
        "test_exposure_count": summary.get("test_exposure_count"),
        "test_metric_policy": summary.get("test_metric_policy"),
        "raw_run_ids": [run.get("run_id") for run in summary.get("runs", []) if run.get("run_id")],
        "source_experiment_id": summary.get("candidate"),
    }


def _best_use_case(summary: dict[str, Any]) -> str:
    name = str(summary.get("candidate") or "")
    if name == "cnn_pvv_q10_direct":
        return "해석 가능한 최소 피처 기반 1W AI band 기본 후보"
    if name == "cnn_no_fundamentals_q10_direct":
        return "fundamentals 희소성이 노이즈일 때의 context-light 대안"
    if name == "cnn_full_q10_direct_lower_guard_w1p5":
        return "하방 breach 방어를 우선하는 1W AI band 후보"
    if name == "cnn_full_q10_direct":
        return "macro/breadth/fundamentals context를 모두 반영하는 1W AI band 후보"
    if name == "tide_pvv_q15_param":
        return "TiDE param 방식의 변동성 민감 대안 검토"
    if name == "context_light_q10_direct":
        return "별도 feature_set 설계가 필요한 context 경량 후보"
    return "1W AI band 재검증 후보"


def _summary_rows(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidate_summaries:
        stats = item.get("validation_metric_stats") or {}
        decision = item.get("decision") or {}
        config = item.get("candidate_config") or {}
        rows.append(
            {
                "candidate": item.get("candidate"),
                "category": decision.get("category"),
                "model": config.get("model"),
                "feature_set": config.get("feature_set"),
                "q_low": config.get("q_low"),
                "q_high": config.get("q_high"),
                "band_mode": config.get("band_mode"),
                "lower_band_loss_weight": config.get("lower_band_loss_weight"),
                "run_count": item.get("run_count"),
                "completed_run_count": item.get("completed_run_count"),
                "band_gate_pass_rate": item.get("band_gate_pass_rate"),
                "test_exposure_count": item.get("test_exposure_count"),
                "coverage_abs_error_mean": (stats.get("coverage_abs_error") or {}).get("mean"),
                "coverage_abs_error_std": (stats.get("coverage_abs_error") or {}).get("std"),
                "lower_breach_rate_mean": (stats.get("lower_breach_rate") or {}).get("mean"),
                "upper_breach_rate_mean": (stats.get("upper_breach_rate") or {}).get("mean"),
                "asymmetric_interval_score_mean": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                "interval_lower_penalty_mean": (stats.get("interval_lower_penalty") or {}).get("mean"),
                "p90_band_width_mean": (stats.get("p90_band_width") or {}).get("mean"),
                "band_width_ic_mean": (stats.get("band_width_ic") or {}).get("mean"),
                "downside_width_ic_mean": (stats.get("downside_width_ic") or {}).get("mean"),
                "falling_lower_breach_rate_mean": _regime_mean(item, "falling_market", "lower_breach_rate"),
                "high_volatility_lower_breach_rate_mean": _regime_mean(item, "high_volatility", "lower_breach_rate"),
                "high_atr_lower_breach_rate_mean": _regime_mean(item, "high_atr", "lower_breach_rate"),
                "failures": ",".join(decision.get("failures") or []),
                "run_ids": ",".join(run.get("run_id") for run in item.get("runs", []) if run.get("run_id")),
            }
        )
    return rows


def _write_summary_csv(candidate_summaries: list[dict[str, Any]]) -> None:
    rows = _summary_rows(candidate_summaries)
    fieldnames = [
        "candidate",
        "category",
        "model",
        "feature_set",
        "q_low",
        "q_high",
        "band_mode",
        "lower_band_loss_weight",
        "run_count",
        "completed_run_count",
        "band_gate_pass_rate",
        "test_exposure_count",
        "coverage_abs_error_mean",
        "coverage_abs_error_std",
        "lower_breach_rate_mean",
        "upper_breach_rate_mean",
        "asymmetric_interval_score_mean",
        "interval_lower_penalty_mean",
        "p90_band_width_mean",
        "band_width_ic_mean",
        "downside_width_ic_mean",
        "falling_lower_breach_rate_mean",
        "high_volatility_lower_breach_rate_mean",
        "high_atr_lower_breach_rate_mean",
        "failures",
        "run_ids",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_fmt(value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _feature_set_rows(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, status in sorted((((preflight.get("data") or {}).get("feature_sets") or {}).items())):
        rows.append(
            {
                "feature_set": name,
                "exists": status.get("exists"),
                "column_count": status.get("column_count"),
                "atr_ratio": status.get("atr_ratio_in_columns"),
                "intraday_range": status.get("intraday_range_ratio_in_columns"),
                "error": status.get("error"),
            }
        )
    return rows


def _report(payload: dict[str, Any], registry: list[dict[str, Any]]) -> str:
    preflight = payload.get("preflight") or {}
    data = preflight.get("data") or {}
    snapshot = data.get("snapshot") or {}
    context = data.get("context_column_checksum") or {}
    summaries = payload.get("candidate_summaries") or []
    summary_rows = _summary_rows(summaries)
    default = next((entry for entry in registry if entry.get("category") == "recommended_default"), None)
    selectable = [entry for entry in registry if entry.get("category") == "selectable_verified"]
    rejected = [entry for entry in registry if entry.get("category") == "rejected"]
    design_needed = [entry for entry in registry if entry.get("category") == "design_needed"]
    default_label = default.get("source_experiment_id") if default else "없음 - raw validation 제품 기준 통과 후보 없음"
    lines = [
        "# CP138-BM: 1W Band Context Backfill 이후 후보 재검증",
        "",
        "## 1. 결론",
        f"- 기본 후보: `{default_label}`",
        f"- selectable_verified: `{', '.join(entry.get('source_experiment_id') for entry in selectable) if selectable else '없음'}`",
        f"- rejected: `{', '.join(entry.get('source_experiment_id') for entry in rejected) if rejected else '없음'}`",
        f"- design_needed: `{', '.join(entry.get('source_experiment_id') for entry in design_needed) if design_needed else '없음'}`",
        "- 후보 선택은 validation과 raw band 기준으로만 수행했다. test metric은 read-only 기록으로만 남겼다.",
        "",
        "## 2. 데이터와 금지 조건",
        f"- provider/source: `{data.get('provider')}` / `{data.get('source')}`",
        f"- source_data_hash: `{data.get('source_data_hash')}`",
        f"- context_column_checksum: `{context.get('checksum')}`",
        f"- CP133 1W indicator_value_checksum: `{((data.get('cp133_reference') or {}).get('after_indicator_value_checksum_1w'))}`",
        f"- price parquet: `{snapshot.get('price_path')}`, mtime `{snapshot.get('price_mtime')}`",
        f"- indicator parquet: `{snapshot.get('indicator_path')}`, mtime `{snapshot.get('indicator_mtime')}`",
        f"- feature_version: `{data.get('feature_version')}`",
        f"- MODEL_FEATURE_COLUMNS: `{data.get('model_feature_columns_count')}`",
        f"- atr_ratio 모델 feature 포함: `{data.get('atr_ratio_in_model_features')}`",
        f"- feature NaN/Inf total: `{data.get('feature_nonfinite_count_total')}`",
        f"- target NaN/Inf total: `{data.get('target_nonfinite_count_total')}`",
        "- save-run/DB write/inference 저장/W&B/composite/live fetch/EODHD 호출은 수행하지 않았다.",
        "",
        "## 3. Feature Set 확인",
        _table(
            _feature_set_rows(preflight),
            [
                ("feature_set", "feature_set"),
                ("exists", "exists"),
                ("columns", "column_count"),
                ("atr", "atr_ratio"),
                ("range", "intraday_range"),
                ("error", "error"),
            ],
        ),
        "",
        "## 4. 후보 결과표",
        _table(
            summary_rows,
            [
                ("candidate", "candidate"),
                ("category", "category"),
                ("model", "model"),
                ("feature_set", "feature_set"),
                ("q_low", "q_low"),
                ("gate", "band_gate_pass_rate"),
                ("cov_abs", "coverage_abs_error_mean"),
                ("lower", "lower_breach_rate_mean"),
                ("upper", "upper_breach_rate_mean"),
                ("interval", "asymmetric_interval_score_mean"),
                ("p90_w", "p90_band_width_mean"),
                ("bw_ic", "band_width_ic_mean"),
                ("down_ic", "downside_width_ic_mean"),
            ],
        ),
        "",
        "## 5. Regime 하방 이탈",
        _table(
            summary_rows,
            [
                ("candidate", "candidate"),
                ("falling lower", "falling_lower_breach_rate_mean"),
                ("high vol lower", "high_volatility_lower_breach_rate_mean"),
                ("high ATR lower", "high_atr_lower_breach_rate_mean"),
                ("failures", "failures"),
            ],
        ),
        "",
        "## 6. 해석",
        "- `full_features`가 우위로 남으면 CP133 context backfill이 macro/breadth/fundamentals 정보를 실제 후보 판정에 반영한 것으로 해석한다.",
        "- `price_volatility_volume`이 우위면 해석 가능한 최소 피처 우선 원칙에 따라 PVV를 기본 후보로 둔다.",
        "- `no_fundamentals`가 우위면 fundamentals의 희소성 또는 0 대체 노이즈를 기록하고 full 기본화를 보류한다.",
        "- `tide_pvv_q15_param`은 falling/high-vol/high-ATR lower breach가 유지되는지로 대안 후보 여부를 판단했다.",
        "- `context_light_q10_direct`는 현재 `docs/cp63_bm_feature_set_plan.json` 계약에 없으면 이번 CP에서 새로 구현하지 않고 design_needed로 남겼다.",
        "",
        "## 7. 산출물",
        f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- registry: `{REGISTRY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- summary csv: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
        f"- logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def _build_registry(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_registry_entry(summary) for summary in candidate_summaries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP138-BM 1W band context backfill 이후 후보 재검증")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_json(METRICS_PATH)
    preflight = build_preflight_payload()
    if args.preflight_only:
        print(json.dumps({"cp": "CP138-BM", "preflight_gate_pass": preflight.get("preflight_gate_pass"), "preflight_path": str(PREFLIGHT_PATH)}, ensure_ascii=False), flush=True)
        return
    if not preflight.get("preflight_gate_pass"):
        print("[CP138] preflight 경고: 게이트가 통과하지 않았지만 가능한 후보는 실행 상태를 기록합니다.", flush=True)
    runs_by_candidate: dict[str, list[dict[str, Any]]] = {candidate.name: [] for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        if candidate.optional:
            status = (((preflight.get("data") or {}).get("feature_sets") or {}).get(candidate.feature_set) or {})
            if not status.get("exists"):
                runs_by_candidate[candidate.name].append(_design_needed_record(candidate, status.get("error") or "feature_set 미구현"))
                continue
        for seed in args.seeds:
            run = _run_training(candidate, seed, args, preflight, existing)
            run = _attach_validation_regimes(run, args)
            runs_by_candidate[candidate.name].append(run)
            if run.get("execution_status") == "DESIGN_NEEDED":
                break
    candidate_summaries = [_aggregate_candidate(candidate, runs_by_candidate[candidate.name]) for candidate in CANDIDATES]
    _apply_decisions(candidate_summaries)
    registry = _build_registry(candidate_summaries)
    payload = {
        "cp": "CP138-BM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preflight": preflight,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "candidate_summaries": candidate_summaries,
        "registry_path": str(REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        "summary_csv_path": str(SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "decision_policy": {
            "selection_split": "validation",
            "test_metric_usage": "read_only_count_only",
            "calibration": "not_used_for_product_performance",
            "criteria": _criteria(),
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, registry)
    _write_summary_csv(candidate_summaries)
    REPORT_PATH.write_text(_report(payload, registry), encoding="utf-8")
    default = next((entry.get("source_experiment_id") for entry in registry if entry.get("category") == "recommended_default"), None)
    print(json.dumps({"cp": "CP138-BM", "recommended_default": default, "metrics_path": str(METRICS_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
