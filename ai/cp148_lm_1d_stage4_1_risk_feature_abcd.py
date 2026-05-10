from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.train as train_mod  # noqa: E402
from ai import cp148_lm_1d_stage0_2 as s2  # noqa: E402
from ai import cp148_lm_1d_stage4_0_risk_aware_rerank as r0  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402
from ai.preprocessing import MODEL_FEATURE_COLUMNS, SequenceDataset, normalize_sequence_splits  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_1_risk_feature_abcd_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_1_risk_feature_abcd_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_1_risk_feature_abcd_summary.csv"
DESIGN_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_revised_experiment_design.md"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_1_risk_feature_abcd_logs"

NO_FUND_FEATURES = [
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "vol_change",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "rsi",
    "macd_ratio",
    "bb_position",
    "us10y",
    "yield_spread",
    "vix_close",
    "credit_spread_hy",
    "nh_nl_index",
    "ma200_pct",
    "regime_calm",
    "regime_neutral",
    "regime_stress",
    "has_macro",
    "has_breadth",
    "has_fundamentals",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_month_end",
    "is_quarter_end",
    "is_opex_friday",
]

EXPERIMENTS = [
    {
        "experiment_id": "stage4_1_a_selector_only",
        "label": "A_selector_only",
        "extra_features": [],
        "question": "checkpoint 선택 기준만 바꿔도 false_safe와 severe가 개선되는가?",
    },
    {
        "experiment_id": "stage4_1_b_atr_only",
        "label": "B_atr_only",
        "extra_features": ["atr_ratio"],
        "question": "종목의 단기 불안정성, 갭, 장중 진폭을 넣으면 h1 약점과 false_safe가 개선되는가?",
    },
    {
        "experiment_id": "stage4_1_c_stress_delta",
        "label": "C_stress_delta",
        "extra_features": ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"],
        "question": "시장 stress 변화와 시장 내부 붕괴 신호를 넣으면 stress 구간 하방 안정성이 개선되는가?",
    },
    {
        "experiment_id": "stage4_1_d_stock_fragility",
        "label": "D_stock_fragility",
        "extra_features": ["atr_ratio", "drawdown_20", "downside_vol_20"],
        "question": "개별 종목의 추세 훼손과 하락 변동성을 넣으면 false_safe가 개선되는가?",
    },
]

BASELINES = {
    "stage2_best_false_safe": 0.30866056266521946,
    "stage2_best_severe_recall": 0.6850190785352241,
    "primary_h1_false_safe": 0.321443,
    "primary_h1_severe_recall": 0.641115,
    "primary_stress_false_safe": 0.20648967551622419,
    "primary_stress_severe_recall": 0.6515151515151515,
}

LINE_KEYS = [
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "direction_accuracy",
    "mae",
    "smape",
]

_BASE_DATASET_CACHE: tuple | None = None
_EXTRA_FRAME_CACHE: pd.DataFrame | None = None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _configure_environment() -> Path:
    alias = r0._configure_cp146_snapshot_alias()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return alias


def _risk_aware_line_gate_sort_key(candidate: Any) -> tuple[float, float, float, float, float, float, float, float]:
    metrics = candidate.metrics

    def finite(name: str, default: float, *, layer: str | None = "line_metrics") -> float:
        return train_mod._finite_metric(metrics, name, default, layer=layer)

    return (
        finite("false_safe_tail_rate", float("inf")),
        -finite("severe_downside_recall", float("-inf")),
        -finite("fee_adjusted_return", float("-inf")),
        -finite("long_short_spread", float("-inf")),
        -finite("spearman_ic", float("-inf")),
        finite("upside_sacrifice", float("inf")),
        finite("mae", float("inf")),
        train_mod._finite_metric(metrics, "forecast_loss", train_mod._finite_metric(metrics, "total_loss", float("inf")), layer=None),
    )


def _patch_training_for_experiment() -> None:
    # line_gate_eligible의 생존 조건은 그대로 두고, line_gate 통과 후보의 정렬 기준만 risk-aware로 바꾼다.
    train_mod.line_gate_sort_key = _risk_aware_line_gate_sort_key

    def _keep_preselected_splits(train_bundle, val_bundle, test_bundle, mean, std, feature_columns):
        del feature_columns
        return train_bundle, val_bundle, test_bundle, mean, std

    # Stage 4-1은 제품 feature contract를 바꾸지 않고 실험용 텐서를 이미 선택/확장해서 넘긴다.
    train_mod.apply_feature_columns_to_splits = _keep_preselected_splits


def _base_dataset(*, device: str, batch_size: int) -> tuple:
    global _BASE_DATASET_CACHE
    if _BASE_DATASET_CACHE is not None:
        return _BASE_DATASET_CACHE
    candidate = {
        "candidate": "stage4_1_base_dataset",
        "model": "patchtst",
        "feature_set": "no_fundamentals",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
    }
    config = s2._make_config(
        candidate=candidate,
        epochs=0,
        seed=42,
        limit_tickers=None,
        device=device,
        batch_size=batch_size,
    )
    config.market_data_provider = "eodhd"
    _BASE_DATASET_CACHE = s2._prepare_dataset_splits_with_progress(config, LOG_DIR / "dataset_base_seq252")
    return _BASE_DATASET_CACHE


def _load_extra_feature_frame() -> pd.DataFrame:
    global _EXTRA_FRAME_CACHE
    if _EXTRA_FRAME_CACHE is not None:
        return _EXTRA_FRAME_CACHE

    alias = r0._snapshot_alias_dir()
    indicator_path = alias / "indicators_eodhd_1D.parquet"
    price_path = alias / "price_data_eodhd.parquet"
    indicators = pd.read_parquet(
        indicator_path,
        columns=[
            "ticker",
            "date",
            "atr_ratio",
            "vix_close",
            "credit_spread_hy",
            "ma200_pct",
        ],
    )
    prices = pd.read_parquet(price_path, columns=["ticker", "date", "close", "high", "low"])
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    date_context = indicators.groupby("date", sort=True).agg(
        vix_close=("vix_close", "median"),
        credit_spread_hy=("credit_spread_hy", "median"),
        ma200_pct=("ma200_pct", "median"),
    )
    date_context["vix_change_5d"] = date_context["vix_close"].diff(5)
    date_context["credit_spread_change_20d"] = date_context["credit_spread_hy"].diff(20)
    date_context["ma200_pct_change_20d"] = date_context["ma200_pct"].diff(20)
    date_context = date_context.reset_index()[
        ["date", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]
    ]

    fragility_frames: list[pd.DataFrame] = []
    for ticker, frame in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        local = frame.copy()
        close = local["close"].astype(float)
        returns = close.pct_change()
        rolling_high = close.rolling(20, min_periods=1).max()
        local["drawdown_20"] = (close / rolling_high) - 1.0
        downside_returns = returns.where(returns < 0.0, 0.0)
        local["downside_vol_20"] = downside_returns.rolling(20, min_periods=2).std()
        local["ticker"] = ticker
        fragility_frames.append(local[["ticker", "date", "drawdown_20", "downside_vol_20"]])
    fragility = pd.concat(fragility_frames, ignore_index=True)

    extra = indicators[["ticker", "date", "atr_ratio"]].merge(date_context, on="date", how="left")
    extra = extra.merge(fragility, on=["ticker", "date"], how="left")
    for column in ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d", "drawdown_20", "downside_vol_20"]:
        extra[column] = extra[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    extra["asof_date"] = extra["date"].dt.strftime("%Y-%m-%d")
    _EXTRA_FRAME_CACHE = extra.drop_duplicates(["ticker", "asof_date"]).set_index(["ticker", "asof_date"])
    return _EXTRA_FRAME_CACHE


def _aligned_extra_values(ticker: str, dates: np.ndarray, extra_names: list[str]) -> np.ndarray:
    if not extra_names:
        return np.empty((len(dates), 0), dtype="float32")
    extra = _load_extra_feature_frame()
    date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
    index = pd.MultiIndex.from_arrays([[ticker] * len(date_strings), date_strings], names=["ticker", "asof_date"])
    values = extra.reindex(index)[extra_names].fillna(0.0).to_numpy(dtype="float32")
    return values


def _build_exact_feature_splits(
    *,
    extra_names: list[str],
    device: str,
    batch_size: int,
) -> tuple[tuple, list[str]]:
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = _base_dataset(device=device, batch_size=batch_size)
    feature_names = [*NO_FUND_FEATURES, *extra_names]
    base_indices = [MODEL_FEATURE_COLUMNS.index(column) for column in NO_FUND_FEATURES]
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in train_bundle.ticker_arrays.items():
        base_values = arrays["features"][:, base_indices].astype("float32", copy=False)
        extra_values = _aligned_extra_values(ticker, arrays["dates"], extra_names)
        feature_values = np.concatenate([base_values, extra_values], axis=1).astype("float32", copy=False)
        copied = dict(arrays)
        copied["features"] = feature_values
        ticker_arrays[ticker] = copied

    def remake(bundle: SequenceDataset) -> SequenceDataset:
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

    exact_train, exact_val, exact_test, mean, std = normalize_sequence_splits(
        remake(train_bundle),
        remake(val_bundle),
        remake(test_bundle),
    )
    return (exact_train, exact_val, exact_test, mean, std, plan), feature_names


def _line_metrics(source: dict[str, Any]) -> dict[str, Any]:
    line = source.get("line_metrics") if isinstance(source.get("line_metrics"), dict) else source
    return {key: line.get(key) for key in LINE_KEYS if line.get(key) is not None}


def _regime_and_bucket_metrics(
    *,
    checkpoint_path: str | Path,
    val_bundle: SequenceDataset,
    batch_size: int,
    device: str,
    amp_dtype: str,
    severe_threshold: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, _checkpoint = load_checkpoint(checkpoint_path)
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        line, raw = r0._predict_bundle(
            model=model,
            bundle=val_bundle,
            device=resolved_device,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
    except (RuntimeError, TypeError) as exc:
        if "BFloat16" not in str(exc):
            raise
        line, raw = r0._predict_bundle(
            model=model,
            bundle=val_bundle,
            device=resolved_device,
            batch_size=batch_size,
            amp_dtype="fp32",
        )
    context = r0._load_context_frame()
    metadata = r0._decorate_metadata(val_bundle.metadata, context)
    masks = {
        "calm": (metadata.get("regime_calm", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
        "neutral": (metadata.get("regime_neutral", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
        "stress": (metadata.get("regime_stress", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
        "vix_rising": (metadata.get("vix_change_5d", pd.Series(np.nan, index=metadata.index)).astype(float) > 0.0).fillna(False).to_numpy(),
        "breadth_worsening": (metadata.get("ma200_pct_change_20d", pd.Series(np.nan, index=metadata.index)).astype(float) < 0.0).fillna(False).to_numpy(),
    }
    regimes = {
        name: r0._metric_for_segment(
            metadata=metadata,
            line=line,
            raw=raw,
            mask=mask,
            start=0,
            end=int(raw.shape[1]),
            severe_threshold=severe_threshold,
        )
        for name, mask in masks.items()
    }
    all_mask = np.ones(len(metadata), dtype=bool)
    buckets = {
        name: r0._metric_for_segment(
            metadata=metadata,
            line=line,
            raw=raw,
            mask=all_mask,
            start=start,
            end=end,
            severe_threshold=severe_threshold,
        )
        for name, (start, end) in r0.HORIZON_BUCKETS.items()
    }
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return regimes, buckets


def _make_config(experiment: dict[str, Any], *, epochs: int, seed: int, batch_size: int, device: str, feature_names: list[str]) -> train_mod.TrainConfig:
    candidate = {
        "candidate": experiment["experiment_id"],
        "model": "patchtst",
        "feature_set": experiment["label"],
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
    }
    config = s2._make_config(
        candidate=candidate,
        epochs=epochs,
        seed=seed,
        limit_tickers=None,
        device=device,
        batch_size=batch_size,
    )
    config.feature_set = experiment["label"]
    config.feature_columns = list(feature_names)
    config.n_features = len(feature_names)
    config.checkpoint_selection = "line_gate"
    config.compile_model = False
    config.use_wandb = False
    config.market_data_provider = "eodhd"
    config.explicit_cuda_cleanup = True
    return config


def _classify(record: dict[str, Any]) -> str:
    metrics = record.get("validation") or {}
    stress = (record.get("regime_metrics") or {}).get("stress") or {}
    h1 = (record.get("horizon_bucket_metrics") or {}).get("h1") or {}
    line_gate = bool(record.get("line_gate_pass"))
    fee = _safe_float(metrics.get("fee_adjusted_return")) or -999.0
    spread = _safe_float(metrics.get("long_short_spread")) or -999.0
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    if (
        line_gate
        and fee > 0.0
        and false_safe is not None
        and severe is not None
        and false_safe < BASELINES["stage2_best_false_safe"]
        and severe >= BASELINES["stage2_best_severe_recall"]
        and spread >= 0.008
    ):
        return "strong_stage4_candidate"
    stress_fs = _safe_float(stress.get("false_safe_tail_rate"))
    stress_severe = _safe_float(stress.get("severe_downside_recall"))
    h1_fs = _safe_float(h1.get("false_safe_tail_rate"))
    h1_severe = _safe_float(h1.get("severe_downside_recall"))
    if (
        (stress_fs is not None and stress_fs < BASELINES["primary_stress_false_safe"])
        or (stress_severe is not None and stress_severe > BASELINES["primary_stress_severe_recall"])
    ):
        return "stress_risk_improvement"
    if (
        h1_fs is not None
        and h1_severe is not None
        and h1_fs < BASELINES["primary_h1_false_safe"]
        and h1_severe > BASELINES["primary_h1_severe_recall"]
    ):
        return "h1_improvement"
    if line_gate and fee > 0.0:
        return "experiment_record"
    return "rejected"


def _run_experiment(experiment: dict[str, Any], *, epochs: int, seed: int, batch_size: int, device: str, amp_dtype: str, force: bool) -> dict[str, Any]:
    candidate_dir = LOG_DIR / experiment["experiment_id"]
    metrics_path = candidate_dir / "trial_metrics.json"
    if metrics_path.exists() and not force:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    started = time.perf_counter()
    precomputed, feature_names = _build_exact_feature_splits(
        extra_names=list(experiment["extra_features"]),
        device=device,
        batch_size=batch_size,
    )
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    config = _make_config(
        experiment,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        device=device,
        feature_names=feature_names,
    )
    result = train_mod.run_training(
        config,
        save_run=False,
        precomputed_bundles=precomputed,
        enable_compile=False,
        local_log=True,
        local_log_dir=candidate_dir / "train_local_logs",
        wandb_group="cp148_stage4_1",
        wandb_name=experiment["experiment_id"],
        wandb_required=False,
    )
    validation = _line_metrics(result.get("best_metrics") or {})
    test = _line_metrics(result.get("test_metrics") or {})
    severe_threshold = _safe_float((result.get("best_metrics") or {}).get("severe_downside_threshold"))
    regimes, buckets = _regime_and_bucket_metrics(
        checkpoint_path=result["checkpoint_path"],
        val_bundle=val_bundle,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        severe_threshold=severe_threshold,
    )
    record = {
        "experiment_id": experiment["experiment_id"],
        "label": experiment["label"],
        "question": experiment["question"],
        "base": "cp148_s2_patchtst_no_fund_p32_s16",
        "model": "patchtst",
        "feature_columns": feature_names,
        "extra_features": list(experiment["extra_features"]),
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "checkpoint_selection": "line_gate",
        "selector_policy": "risk_aware_line_gate_sort",
        "line_gate_pass": bool((result.get("best_metrics") or {}).get("line_gate_pass")),
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "elapsed_seconds": time.perf_counter() - started,
        "source_data_hash": getattr(plan, "source_data_hash", None),
        "eligible_ticker_count": len(getattr(plan, "eligible_tickers", [])),
        "validation": validation,
        "test": test,
        "regime_metrics": regimes,
        "horizon_bucket_metrics": buckets,
        "scope_compliance": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "product_promotion": False,
            "band_or_composite_experiment": False,
            "beta": 2.0,
        },
    }
    record["classification"] = _classify(record)
    _write_json(metrics_path, record)
    return record


def _write_summary_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "label",
        "classification",
        "line_gate_pass",
        "extra_features",
        "ic_mean",
        "long_short_spread",
        "fee_adjusted_return",
        "false_safe_tail_rate",
        "severe_downside_recall",
        "stress_false_safe_tail_rate",
        "stress_severe_downside_recall",
        "h1_false_safe_tail_rate",
        "h1_severe_downside_recall",
        "h2_h3_false_safe_tail_rate",
        "h4_h5_false_safe_tail_rate",
        "checkpoint_path",
    ]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get("validation") or {}
            stress = (record.get("regime_metrics") or {}).get("stress") or {}
            buckets = record.get("horizon_bucket_metrics") or {}
            writer.writerow(
                {
                    "experiment_id": record.get("experiment_id"),
                    "label": record.get("label"),
                    "classification": record.get("classification"),
                    "line_gate_pass": record.get("line_gate_pass"),
                    "extra_features": ",".join(record.get("extra_features") or []),
                    "ic_mean": metrics.get("ic_mean"),
                    "long_short_spread": metrics.get("long_short_spread"),
                    "fee_adjusted_return": metrics.get("fee_adjusted_return"),
                    "false_safe_tail_rate": metrics.get("false_safe_tail_rate"),
                    "severe_downside_recall": metrics.get("severe_downside_recall"),
                    "stress_false_safe_tail_rate": stress.get("false_safe_tail_rate"),
                    "stress_severe_downside_recall": stress.get("severe_downside_recall"),
                    "h1_false_safe_tail_rate": (buckets.get("h1") or {}).get("false_safe_tail_rate"),
                    "h1_severe_downside_recall": (buckets.get("h1") or {}).get("severe_downside_recall"),
                    "h2_h3_false_safe_tail_rate": (buckets.get("h2_h3") or {}).get("false_safe_tail_rate"),
                    "h4_h5_false_safe_tail_rate": (buckets.get("h4_h5") or {}).get("false_safe_tail_rate"),
                    "checkpoint_path": record.get("checkpoint_path"),
                }
            )


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _write_design() -> None:
    lines = [
        "# CP148-LM-1D Stage 4 Revised Experiment Design",
        "",
        "## 목적",
        "",
        "Stage 4-0 risk-aware rerank 결과를 반영해 `cp148_s2_patchtst_no_fund_p32_s16`을 주 베이스로 두고 하방 안정성 개선 원인을 분해한다.",
        "",
        "## 고정 조건",
        "",
        "- beta=2.0, alpha/beta/delta=1.0/2.0/1.0 유지",
        "- tail 전용 loss 추가 없음",
        "- 기존 line_gate 생존 의미 변경 없음",
        "- product save, DB write, inference 저장, live fetch 없음",
        "- band/composite 실험 없음",
        "- EODHD 500 local parquet 기준",
        "",
        "## 후보 역할",
        "",
        "- primary_stage4_base: `cp148_s2_patchtst_no_fund_p32_s16`",
        "- secondary_stage4_base: `cp148_s2_patchtst_pvv_p32_s16`",
        "- alpha/stress 참고선: `cp148_s2_patchtst_pvv_p16_s8`",
        "- CNN-LSTM: 이번 LM 실험 제외, risk_only_reference로 보관",
        "",
        "## A/B/C/D",
        "",
    ]
    lines.extend(
        _table(
            ["실험", "추가 피처", "질문"],
            [[item["label"], ", ".join(item["extra_features"]) or "없음", item["question"]] for item in EXPERIMENTS],
        )
    )
    lines.extend(
        [
            "",
            "## 평가 분해",
            "",
            "전체 validation, calm/neutral/stress, vix_rising, breadth_worsening, h1, h2_h3, h4_h5를 기록한다.",
            "",
            "## 성공 기준",
            "",
            "- line_gate_pass=True",
            "- fee_adjusted_return > 0",
            "- false_safe_tail_rate < 0.30866056266521946",
            "- severe_downside_recall >= 0.6850190785352241",
            "- spread >= 0.008 권장",
            "- upside_sacrifice 과도 증가 없음",
        ]
    )
    DESIGN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(records: list[dict[str, Any]]) -> None:
    lines = [
        "# CP148-LM-1D Stage 4-1 risk feature A/B/C/D 보고서",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 범위: PatchTST no_fund p32/s16 기반 risk-aware selector 및 실험용 피처 확장",
        "- 금지 준수: save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, live fetch 없음, band/composite 없음",
        "- CNN-LSTM은 이번 LM 실험에서 제외하고 risk_only_reference로만 보관",
        "",
        "## 1. 전체 validation 결과",
        "",
    ]
    lines.extend(
        _table(
            ["실험", "분류", "line_gate", "IC", "spread", "fee", "false_safe", "severe", "bias", "sacrifice"],
            [
                [
                    record.get("label"),
                    record.get("classification"),
                    record.get("line_gate_pass"),
                    _fmt((record.get("validation") or {}).get("ic_mean")),
                    _fmt((record.get("validation") or {}).get("long_short_spread")),
                    _fmt((record.get("validation") or {}).get("fee_adjusted_return")),
                    _fmt((record.get("validation") or {}).get("false_safe_tail_rate")),
                    _fmt((record.get("validation") or {}).get("severe_downside_recall")),
                    _fmt((record.get("validation") or {}).get("conservative_bias")),
                    _fmt((record.get("validation") or {}).get("upside_sacrifice")),
                ]
                for record in records
            ],
        )
    )
    lines.extend(["", "## 2. stress regime 결과", ""])
    stress_rows: list[list[Any]] = []
    for record in records:
        for regime in ["calm", "neutral", "stress", "vix_rising", "breadth_worsening"]:
            metrics = (record.get("regime_metrics") or {}).get(regime) or {}
            stress_rows.append(
                [
                    record.get("label"),
                    regime,
                    metrics.get("sample_count"),
                    metrics.get("date_count"),
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                ]
            )
    lines.extend(_table(["실험", "구간", "samples", "dates", "FS", "severe", "spread", "fee"], stress_rows))
    lines.extend(["", "## 3. horizon bucket 결과", ""])
    bucket_rows: list[list[Any]] = []
    for record in records:
        for bucket in ["h1", "h2_h3", "h4_h5"]:
            metrics = (record.get("horizon_bucket_metrics") or {}).get(bucket) or {}
            bucket_rows.append(
                [
                    record.get("label"),
                    bucket,
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                ]
            )
    lines.extend(_table(["실험", "bucket", "FS", "severe", "spread", "fee"], bucket_rows))
    best = sorted(records, key=lambda row: (
        _safe_float((row.get("validation") or {}).get("false_safe_tail_rate")) or 999.0,
        -(_safe_float((row.get("validation") or {}).get("severe_downside_recall")) or -999.0),
    ))[0] if records else None
    lines.extend(
        [
            "",
            "## 4. 결론",
            "",
            f"- 가장 낮은 전체 false_safe 후보: `{best.get('label') if best else ''}`",
            "- A_selector_only는 selector만 바꿨을 때의 기준선이다. false_safe와 severe가 Stage 2 primary보다 소폭 좋아져도 spread가 0.008 권장선 아래면 충분한 해결책으로 보지 않는다.",
            "- B_atr_only는 ATR이 h1과 전체 false_safe 개선에 기여하는지 확인한다. spread 희생이 크면 stress-risk 개선 후보로만 둔다.",
            "- C_stress_delta는 ranking/fee를 가장 잘 살리는지와 stress regime false_safe를 실제로 낮추는지를 분리해 본다.",
            "- D_stock_fragility는 drawdown/downside volatility가 전체 false_safe와 horizon bucket 하방 안정성을 낮추는지 확인한다.",
            "- strong_stage4_candidate가 나오면 primary_stage4_base는 해당 변형으로 옮겨 seed 3개 재평가 후보로 둔다.",
            "- secondary_stage4_base인 pvv p32/s16 확장은 no_fund 기반 strong 또는 stress-risk 개선이 seed에서 유지될 때 다음 CP에서 검토한다.",
            "- CNN-LSTM은 이번 LM 실험에서 제외했고 risk_only_reference로만 보관한다.",
            "- 터미널 실행 주의: PowerShell Start-Process가 Path/PATH 환경변수 중복 때문에 실패할 수 있으므로, 장시간 실행은 venv python 직접 실행 또는 환경변수 정리 후 로그 파일을 붙여 실행한다.",
            "",
            "## 5. 산출물",
            "",
            f"- design: `{DESIGN_PATH.relative_to(PROJECT_ROOT)}`",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- script: `ai/cp148_lm_1d_stage4_1_risk_feature_abcd.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_experiments(names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return EXPERIMENTS
    wanted = {name.lower() for name in names}
    return [
        item
        for item in EXPERIMENTS
        if item["experiment_id"].lower() in wanted or item["label"].lower() in wanted or item["label"].split("_", 1)[0].lower() in wanted
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    alias = _configure_environment()
    _patch_training_for_experiment()
    _write_design()
    records: list[dict[str, Any]] = []
    for experiment in _selected_experiments(args.experiment):
        record = _run_experiment(
            experiment,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            force=args.force,
        )
        records.append(record)
        print(json.dumps({"experiment_id": record["experiment_id"], "classification": record["classification"]}, ensure_ascii=False), flush=True)
    payload = {
        "cp": "CP148-LM-1D-Stage4-1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_alias_dir": str(alias),
        "scope_compliance": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "product_promotion": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "beta": 2.0,
            "line_gate_meaning_changed": False,
            "selector_sort_changed": True,
        },
        "baselines": BASELINES,
        "records": records,
        "process_check": {
            "status": "deferred_external_check",
            "note": "실행 중 자기 자신이 잡힐 수 있어 최종 외부 검증에서 확인한다.",
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records)
    _write_report(records)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-1 risk feature A/B/C/D")
    parser.add_argument("--experiment", action="append", help="A, B, C, D 또는 experiment_id. 생략하면 전체 실행")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
