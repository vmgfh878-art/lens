from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402

from ai.preprocessing import (
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    resolve_feature_cache_path,
    resolve_feature_index_cache_path,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import build_registry, registry_path_for_tickers
from ai.train import TrainConfig, resolve_feature_columns, run_training


REPORT_PATH = PROJECT_ROOT / "docs" / "cp65_lm_feature_h20_smoke_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp65_lm_feature_h20_smoke_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp65_lm_feature_h20_smoke_logs"
CP62_METRICS_PATH = PROJECT_ROOT / "docs" / "cp62_cp61_schema_candidate_regrade_metrics.json"

LINE_METRIC_KEYS = [
    "spearman_ic",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_mean",
    "spread_std",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "mae",
    "smape",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
]
BUCKET_PREFIXES = ["h1_h5", "h6_h10", "h11_h20"]
LINE_BASELINE_NAMES = [
    "historical_mean_line_w60",
    "reversal_line_horizon",
    "random_or_shuffled_score",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    branch: str
    horizon: int
    feature_set: str
    seq_len: int = 252
    patch_len: int = 32
    patch_stride: int = 16
    epochs: int = 3
    limit_tickers: int = 50
    batch_size: int = 256


EXPERIMENTS = [
    Experiment("h5_technical_only_seq252_p32_s16", "h5_product_line", 5, "technical_only"),
    Experiment("h5_no_fundamentals_seq252_p32_s16", "h5_product_line", 5, "no_fundamentals"),
    Experiment("h5_price_volatility_volume_seq252_p32_s16", "h5_product_line", 5, "price_volatility_volume"),
    Experiment("h20_technical_only_seq252_p32_s16", "h20_visual_risk", 20, "technical_only"),
    Experiment("h20_no_fundamentals_seq252_p32_s16", "h20_visual_risk", 20, "no_fundamentals"),
    Experiment("h20_price_volatility_volume_seq252_p32_s16", "h20_visual_risk", 20, "price_volatility_volume"),
]

FULL_FEATURE_REFERENCE = {
    5: "h5_longer_context_seq252_p32_s16",
    20: "h20_longer_context_seq252_p32_s16",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _line_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    return {key: source.get(key) for key in LINE_METRIC_KEYS}


def _bucket_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    buckets: dict[str, dict[str, Any]] = {}
    for prefix in BUCKET_PREFIXES:
        bucket = {}
        for key in LINE_METRIC_KEYS:
            value = source.get(f"{prefix}_{key}")
            if value is not None:
                bucket[key] = value
        if bucket:
            buckets[prefix] = bucket
    return buckets


def _epoch_summary_from_log(path: Path) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    vram_peak_mb: list[float] = []
    if not path.exists():
        return {"epoch_seconds": [], "epoch_seconds_mean": None, "vram_peak_allocated_mb": None}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        epoch_value = _safe_float(payload.get("epoch_seconds"))
        vram_value = _safe_float(payload.get("vram_peak_allocated_mb"))
        if epoch_value is not None:
            epoch_seconds.append(epoch_value)
        if vram_value is not None:
            vram_peak_mb.append(vram_value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_allocated_mb": max(vram_peak_mb) if vram_peak_mb else None,
    }


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _load_references() -> dict[str, Any]:
    cp62 = _read_json(CP62_METRICS_PATH)
    line_candidates = {}
    for row in cp62.get("line_candidates", []):
        name = row.get("candidate_name")
        if not name:
            continue
        line_candidates[str(name)] = {
            "candidate_name": name,
            "horizon": row.get("horizon"),
            "feature_set": "full_features",
            "source": row.get("source"),
            "checkpoint_path": row.get("checkpoint_path"),
            "line_metrics": _line_metrics(row.get("line_metrics", {})),
            "bucket_line_metrics": _bucket_metrics(row.get("line_metrics", {})),
            "validation_line_metrics": _line_metrics(row.get("validation_line_metrics", {})),
            "verdict": row.get("verdict"),
        }

    line_baselines = {}
    for row in cp62.get("line_baselines", []):
        name = row.get("name")
        if name in LINE_BASELINE_NAMES:
            line_baselines[str(name)] = {
                "name": name,
                "category": row.get("category"),
                "line_metrics": _line_metrics(row.get("line_metrics", {})),
            }
    return {
        "source": str(CP62_METRICS_PATH.relative_to(PROJECT_ROOT)),
        "full_features": {
            name: line_candidates.get(name)
            for name in FULL_FEATURE_REFERENCE.values()
            if line_candidates.get(name)
        },
        "line_baselines": line_baselines,
    }


def _metric_delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _risk_baseline(references: dict[str, Any]) -> dict[str, Any]:
    baselines = list(references.get("line_baselines", {}).values())
    if not baselines:
        return {}
    return min(
        (row["line_metrics"] for row in baselines),
        key=lambda metrics: (
            _safe_float(metrics.get("false_safe_tail_rate")) or float("inf"),
            -(_safe_float(metrics.get("severe_downside_recall")) or float("-inf")),
        ),
    )


def _classify(record: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("line_metrics", {})
    risk_baseline = _risk_baseline(references)
    full_name = FULL_FEATURE_REFERENCE.get(int(record["experiment"]["horizon"]))
    full_ref = references.get("full_features", {}).get(full_name, {}) if full_name else {}
    full_metrics = full_ref.get("line_metrics", {})

    ic = _safe_float(metrics.get("spearman_ic"))
    spread = _safe_float(metrics.get("long_short_spread"))
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe_recall = _safe_float(metrics.get("severe_downside_recall"))
    baseline_false_safe = _safe_float(risk_baseline.get("false_safe_tail_rate"))
    baseline_recall = _safe_float(risk_baseline.get("severe_downside_recall"))
    full_false_safe = _safe_float(full_metrics.get("false_safe_tail_rate"))
    full_recall = _safe_float(full_metrics.get("severe_downside_recall"))

    risk_improved_vs_baseline = (
        false_safe is not None
        and baseline_false_safe is not None
        and false_safe < baseline_false_safe
    ) or (
        severe_recall is not None
        and baseline_recall is not None
        and severe_recall > baseline_recall
    )
    full_features_improved = (
        false_safe is not None
        and full_false_safe is not None
        and false_safe < full_false_safe
    ) or (
        severe_recall is not None
        and full_recall is not None
        and severe_recall > full_recall
    )

    if record["experiment"]["branch"] == "h20_visual_risk":
        h20_visual_minimum = (
            ((ic is not None and ic > 0.0) or (spread is not None and spread > 0.0))
            and severe_recall is not None
            and baseline_recall is not None
            and severe_recall > baseline_recall
            and false_safe is not None
            and full_false_safe is not None
            and false_safe < full_false_safe
        )
        if h20_visual_minimum:
            verdict = "visual_risk_survive"
        elif (ic is not None and ic > 0.0) or (spread is not None and spread > 0.0) or full_features_improved:
            verdict = "phase_1_5_visual_risk_watch"
        else:
            verdict = "line_fail"
    elif ic is not None and spread is not None and ic > 0.0 and spread > 0.0 and risk_improved_vs_baseline:
        verdict = "line_survive"
    elif (ic is not None and ic > 0.0) or (spread is not None and spread > 0.0) or risk_improved_vs_baseline:
        verdict = "line_watch"
    else:
        verdict = "line_fail"

    return {
        "full_feature_reference": full_name,
        "spearman_ic_delta_vs_full": _metric_delta(metrics, full_metrics, "spearman_ic"),
        "long_short_spread_delta_vs_full": _metric_delta(metrics, full_metrics, "long_short_spread"),
        "false_safe_tail_rate_delta_vs_full": _metric_delta(metrics, full_metrics, "false_safe_tail_rate"),
        "severe_downside_recall_delta_vs_full": _metric_delta(metrics, full_metrics, "severe_downside_recall"),
        "false_safe_tail_rate_delta_vs_risk_baseline": _metric_delta(metrics, risk_baseline, "false_safe_tail_rate"),
        "severe_downside_recall_delta_vs_risk_baseline": _metric_delta(metrics, risk_baseline, "severe_downside_recall"),
        "risk_improved_vs_baseline": risk_improved_vs_baseline,
        "full_features_improved": full_features_improved,
        "verdict": verdict,
    }


def _build_readonly_bundles(
    *,
    data_hash: str,
    timeframe: str,
    seq_len: int,
    horizon: int,
    limit_tickers: int,
) -> tuple[Any, Any, Any, torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
    index_path = resolve_feature_index_cache_path(
        timeframe=timeframe,
        data_hash=data_hash,
        tickers=None,
        limit_tickers=limit_tickers,
    )
    if not index_path.exists():
        raise FileNotFoundError(f"기존 feature index cache가 없습니다: {index_path}")
    index_frame = torch.load(index_path, map_location="cpu", weights_only=False)

    input_registry = build_registry(sorted(index_frame["ticker"].astype(str).str.upper().unique()), timeframe)
    preliminary_plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=input_registry,
        ticker_registry_path="memory",
    )
    ticker_registry = build_registry(preliminary_plan.eligible_tickers, timeframe)
    registry_path = registry_path_for_tickers(timeframe, preliminary_plan.eligible_tickers)
    plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=ticker_registry,
        ticker_registry_path=str(registry_path),
    )

    feature_path = resolve_feature_cache_path(
        timeframe=timeframe,
        data_hash=data_hash,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
    )
    if not feature_path.exists():
        raise FileNotFoundError(f"기존 feature cache가 없습니다: {feature_path}")
    cached = torch.load(feature_path, map_location="cpu", weights_only=False)
    feature_df = cached["feature_df"].copy()
    price_df = cached["price_df"].copy()
    feature_df = feature_df[feature_df["ticker"].isin(plan.eligible_tickers)].copy()
    price_df = price_df[price_df["ticker"].isin(plan.eligible_tickers)].copy()

    dataset = build_lazy_sequence_dataset(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=ticker_registry,
        include_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
        dataset,
        split_specs=plan.split_specs,
    )
    train_bundle, val_bundle, test_bundle, mean, std = normalize_sequence_splits(
        train_bundle,
        val_bundle,
        test_bundle,
    )
    cache_meta = {
        "feature_index_cache": str(index_path.relative_to(PROJECT_ROOT)),
        "feature_cache": str(feature_path.relative_to(PROJECT_ROOT)),
        "ticker_registry_path": str(registry_path),
        "ticker_registry_preexisted": registry_path.exists(),
        "eligible_ticker_count": len(plan.eligible_tickers),
        "input_ticker_count": plan.input_ticker_count,
    }
    return train_bundle, val_bundle, test_bundle, mean, std, plan, cache_meta


def _base_config(experiment: Experiment, args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model="patchtst",
        timeframe="1D",
        horizon=experiment.horizon,
        seq_len=experiment.seq_len,
        epochs=experiment.epochs,
        batch_size=experiment.batch_size,
        lr=args.lr,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=0.25,
        q_high=0.75,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.2,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=experiment.limit_tickers,
        seed=args.seed,
        device=args.device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-cp65",
        model_ver="v2-multihead",
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=1e-4,
        checkpoint_selection="line_gate",
        amp_dtype=args.amp_dtype,
        detect_anomaly=False,
        explicit_cuda_cleanup=False,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=experiment.patch_len,
        patch_stride=experiment.patch_stride,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=experiment.feature_set,
    )


def _run_experiment(
    experiment: Experiment,
    args: argparse.Namespace,
    bundles_by_horizon: dict[int, tuple[Any, Any, Any, torch.Tensor, torch.Tensor, Any, dict[str, Any]]],
) -> dict[str, Any]:
    config = _base_config(experiment, args)
    feature_columns = resolve_feature_columns(experiment.feature_set)
    same_column_key = (experiment.horizon, tuple(feature_columns))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}.log"
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("# cp65 line model smoke\n")
            log_file.write("# experiment: " + experiment.name + "\n")
            log_file.write("# feature_columns: " + ", ".join(feature_columns) + "\n")
            log_file.flush()
            with redirect_stdout(log_file), redirect_stderr(log_file):
                result = run_training(
                    config,
                    save_run=False,
                    precomputed_bundles=bundles_by_horizon[experiment.horizon][:6],
                    enable_compile=False,
                )
        elapsed = time.perf_counter() - started
        status = "completed"
        failed_reason = None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result = {}
        status = "failed"
        failed_reason = str(exc)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n# failed_reason: " + failed_reason + "\n")

    test_metrics = result.get("test_metrics", {}) if isinstance(result, dict) else {}
    best_metrics = result.get("best_metrics", {}) if isinstance(result, dict) else {}
    record: dict[str, Any] = {
        "name": experiment.name,
        "experiment": asdict(experiment),
        "same_column_key": [same_column_key[0], list(same_column_key[1])],
        "status": status,
        "failed_reason": failed_reason,
        "elapsed_seconds": elapsed,
        "epoch_summary": _epoch_summary_from_log(log_path),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "run_id": result.get("run_id") if isinstance(result, dict) else None,
        "checkpoint_path": result.get("checkpoint_path") if isinstance(result, dict) else None,
        "feature_set": experiment.feature_set,
        "n_features": result.get("n_features") if isinstance(result, dict) else len(feature_columns),
        "feature_columns": result.get("feature_columns") if isinstance(result, dict) else feature_columns,
        "line_metrics": _line_metrics(test_metrics),
        "validation_line_metrics": _line_metrics(best_metrics),
        "bucket_line_metrics": _bucket_metrics(test_metrics),
    }
    return record


def _reuse_record(source: dict[str, Any], experiment: Experiment) -> dict[str, Any]:
    record = json.loads(json.dumps(source, ensure_ascii=False))
    record["name"] = experiment.name
    record["experiment"] = asdict(experiment)
    record["status"] = "reused_same_columns"
    record["reused_from"] = source["name"]
    record["feature_set"] = experiment.feature_set
    record["log_path"] = source.get("log_path")
    return record


def _build_report(payload: dict[str, Any]) -> str:
    h5_rows = []
    h20_rows = []
    feature_rows = []
    execution_rows = []

    for record in payload["experiments"]:
        metrics = record.get("line_metrics", {})
        comparison = record.get("comparison", {})
        row = {
            "name": record["name"],
            "feature_set": record["feature_set"],
            "n": record.get("n_features", ""),
            "ic": _fmt(metrics.get("spearman_ic")),
            "spread": _fmt(metrics.get("long_short_spread")),
            "dir": _fmt(metrics.get("direction_accuracy")),
            "fstail": _fmt(metrics.get("false_safe_tail_rate")),
            "fssev": _fmt(metrics.get("false_safe_severe_rate")),
            "recall": _fmt(metrics.get("severe_downside_recall")),
            "bias": _fmt(metrics.get("conservative_bias")),
            "mae": _fmt(metrics.get("mae")),
            "smape": _fmt(metrics.get("smape")),
            "delta_ic": _fmt(comparison.get("spearman_ic_delta_vs_full")),
            "delta_fstail": _fmt(comparison.get("false_safe_tail_rate_delta_vs_full")),
            "delta_recall": _fmt(comparison.get("severe_downside_recall_delta_vs_full")),
            "verdict": comparison.get("verdict", record.get("status")),
        }
        if record["experiment"]["horizon"] == 5:
            h5_rows.append(row)
        else:
            h20_rows.append(row)
        feature_rows.append(
            {
                "branch": record["experiment"]["branch"],
                "feature_set": record["feature_set"],
                "improved": "YES" if comparison.get("full_features_improved") else "NO",
                "reason": comparison.get("verdict", ""),
            }
        )
        execution_rows.append(
            {
                "name": record["name"],
                "status": record.get("status"),
                "seconds": _fmt(record.get("elapsed_seconds"), 1),
                "epoch_mean": _fmt(record.get("epoch_summary", {}).get("epoch_seconds_mean"), 2),
                "reuse": record.get("reused_from", ""),
                "log": record.get("log_path", ""),
            }
        )

    references = payload["references"]
    baseline_rows = []
    for name, row in references.get("line_baselines", {}).items():
        metrics = row.get("line_metrics", {})
        baseline_rows.append(
            {
                "name": name,
                "ic": _fmt(metrics.get("spearman_ic")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )

    full_rows = []
    for name, row in references.get("full_features", {}).items():
        metrics = row.get("line_metrics", {})
        full_rows.append(
            {
                "name": name,
                "horizon": row.get("horizon"),
                "ic": _fmt(metrics.get("spearman_ic")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "fssev": _fmt(metrics.get("false_safe_severe_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
                "bias": _fmt(metrics.get("conservative_bias")),
                "verdict": row.get("verdict", ""),
            }
        )

    h5_completed = [row for row in payload["experiments"] if row["experiment"]["horizon"] == 5 and row.get("line_metrics")]
    h20_completed = [row for row in payload["experiments"] if row["experiment"]["horizon"] == 20 and row.get("line_metrics")]
    h5_survivors = [
        row
        for row in h5_completed
        if row.get("comparison", {}).get("verdict") == "line_survive"
        and row.get("comparison", {}).get("full_features_improved")
    ]
    h5_primary = max(
        h5_survivors,
        key=lambda row: (
            _safe_float(row["line_metrics"].get("spearman_ic")) or float("-inf"),
            _safe_float(row["line_metrics"].get("long_short_spread")) or float("-inf"),
            -(_safe_float(row["line_metrics"].get("false_safe_tail_rate")) or float("inf")),
        ),
    )["name"] if h5_survivors else "기존 h5_longer_context_seq252_p32_s16 유지"

    h20_survivors = [row for row in h20_completed if row.get("comparison", {}).get("verdict") == "visual_risk_survive"]
    h20_judgement = "생존" if h20_survivors else "Phase 1.5 visual/risk branch 보류"

    return f"""# CP65-LM PatchTST line feature set smoke + h20 visual branch

## 1. 실행 계약 확인

이번 CP는 PatchTST `role=line_model`만 다뤘고 제품 산출물 기준은 `line_series`다. 평가는 `line_metrics`만 사용했으며 band coverage, composite/overlay 지표, lower/upper breach로 line 후보를 탈락시키지 않았다.

기존 1D cache만 읽었다. 사용 cache는 `{payload["cache"]["feature_index_cache"]}`와 `{payload["cache"]["feature_cache"]}`이며, DB 백필 데이터 읽기, DB 쓰기, save-run, full 473티커 실행, cache fingerprint 생성/삭제, UI/backend 수정은 하지 않았다. ticker registry는 `{payload["cache"]["ticker_registry_path"]}`가 이미 존재하는지 확인만 했다.

공통 조건은 `feature_version={FEATURE_CONTRACT_VERSION}`, `target=raw_future_return`, `line_target_type=raw_future_return`, `checkpoint_selection=line_gate`, `ci_aggregate=target`, `limit_tickers=50`, `epochs=3`, `batch_size=256`, `wandb=off`, `no-compile`, `save-run=false`다.

## 2. 실행 상태

{_table(execution_rows, [("실험", "name"), ("상태", "status"), ("초", "seconds"), ("epoch 평균", "epoch_mean"), ("재사용", "reuse"), ("로그", "log")])}

`technical_only`와 `price_volatility_volume`은 현재 CP63 feature set 정의상 같은 11개 column이다. 그래서 같은 horizon 안에서는 중복 학습을 피하고 선행 결과를 재사용했다.

## 3. 기존 full_features 기준

{_table(full_rows, [("기준 후보", "name"), ("h", "horizon"), ("IC", "ic"), ("spread", "spread"), ("false_safe_tail", "fstail"), ("false_safe_severe", "fssev"), ("severe_recall", "recall"), ("bias", "bias"), ("판정", "verdict")])}

## 4. line statistical baseline

{_table(baseline_rows, [("baseline", "name"), ("IC", "ic"), ("spread", "spread"), ("false_safe_tail", "fstail"), ("severe_recall", "recall")])}

## 5. h5 product line 결과

{_table(h5_rows, [("실험", "name"), ("feature_set", "feature_set"), ("n", "n"), ("IC", "ic"), ("spread", "spread"), ("dir", "dir"), ("false_tail", "fstail"), ("false_severe", "fssev"), ("severe_recall", "recall"), ("bias", "bias"), ("MAE", "mae"), ("SMAPE", "smape"), ("IC Δfull", "delta_ic"), ("tail Δfull", "delta_fstail"), ("recall Δfull", "delta_recall"), ("판정", "verdict")])}

## 6. h20 visual/risk branch 결과

{_table(h20_rows, [("실험", "name"), ("feature_set", "feature_set"), ("n", "n"), ("IC", "ic"), ("spread", "spread"), ("dir", "dir"), ("false_tail", "fstail"), ("false_severe", "fssev"), ("severe_recall", "recall"), ("bias", "bias"), ("MAE", "mae"), ("SMAPE", "smape"), ("IC Δfull", "delta_ic"), ("tail Δfull", "delta_fstail"), ("recall Δfull", "delta_recall"), ("판정", "verdict")])}

## 7. feature_set별 개선 여부

{_table(feature_rows, [("branch", "branch"), ("feature_set", "feature_set"), ("full 대비 risk 개선", "improved"), ("판정", "reason")])}

## 8. 후보 판단

h5 주력 후보는 `{h5_primary}`다. feature pruning smoke가 기존 h5 full_features보다 line 종합 우위를 안정적으로 보였을 때만 교체한다.

h20 판단은 `{h20_judgement}`이다. h20은 제품 본류 확정 후보가 아니라 Phase 1.5 visual/risk branch로 분리 기록한다. 제품에는 h5 line을 기본선으로 두고, h20은 중기 위험 방향 보조선 또는 별도 horizon 토글로 표시해야 한다. 같은 차트에서 h5와 h20을 한 후보군처럼 랭킹하거나 평균 내면 안 되고, label에는 horizon과 branch provenance를 명시해야 한다.

## 9. 다음 LM 추천

- h5는 `h5_longer_context_seq252_p32_s16`을 주력 후보로 유지하고, pruned 후보가 IC/spread와 false-safe를 동시에 이길 때만 승격한다.
- h20은 Phase 1.5 visual/risk branch로 남기고, 최소 생존 조건을 만족한 feature_set만 제품 시각성 후보로 둔다.
- `technical_only`와 `price_volatility_volume` 정의가 현재 동일하므로 다음 LM에서는 둘 중 하나를 정리하거나, volume 포함 여부를 실제로 갈라서 재정의한 뒤 비교한다.
- `no_market_context`는 이번 6개 제한 밖이라 보류한다. 추가 지시가 있으면 같은 cache-only 경로로 h5/h20에 붙이면 된다.

## 10. 검증

- `python -m py_compile ai\\cp65_lm_feature_h20_smoke.py ai\\train.py`: 통과
- `python -m unittest ai.tests.test_feature_set_selection`: 5개 통과
- `python -m json.tool docs\\cp65_lm_feature_h20_smoke_metrics.json`: 통과
- 마지막 확인 기준 남은 `python/pythonw` 학습 프로세스 없음
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    references = _load_references()
    existing_records: dict[str, dict[str, Any]] = {}
    if METRICS_PATH.exists():
        existing_payload = _read_json(METRICS_PATH)
        existing_records = {
            str(record.get("name")): record
            for record in existing_payload.get("experiments", [])
            if record.get("name")
        }

    requested_names = set(args.only or [experiment.name for experiment in EXPERIMENTS])
    selected_experiments = [experiment for experiment in EXPERIMENTS if experiment.name in requested_names]

    bundles_by_horizon = {}
    cache_meta_by_horizon = {}
    for horizon in sorted({experiment.horizon for experiment in selected_experiments}):
        bundles = _build_readonly_bundles(
            data_hash=args.data_hash,
            timeframe="1D",
            seq_len=252,
            horizon=horizon,
            limit_tickers=50,
        )
        bundles_by_horizon[horizon] = bundles
        cache_meta_by_horizon[horizon] = bundles[6]

    records = []
    completed_by_columns: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for experiment in selected_experiments:
        if experiment.name in existing_records and not args.force:
            record = existing_records[experiment.name]
            record["status"] = record.get("status", "completed")
            record["skip_reason"] = "existing_result_reused"
            records.append(record)
            completed_by_columns[
                (experiment.horizon, tuple(record.get("feature_columns") or resolve_feature_columns(experiment.feature_set)))
            ] = record
            print(json.dumps({"cp65": "skip_existing", "experiment": experiment.name}, ensure_ascii=False), flush=True)
            continue

        columns = tuple(resolve_feature_columns(experiment.feature_set))
        column_key = (experiment.horizon, columns)
        if column_key in completed_by_columns:
            record = _reuse_record(completed_by_columns[column_key], experiment)
            records.append(record)
            print(json.dumps({"cp65": "reuse_same_columns", "experiment": experiment.name, "source": record["reused_from"]}, ensure_ascii=False), flush=True)
            continue

        print(json.dumps({"cp65": "start", "experiment": experiment.name}, ensure_ascii=False), flush=True)
        record = _run_experiment(experiment, args, bundles_by_horizon)
        records.append(record)
        if record.get("status") == "completed":
            completed_by_columns[column_key] = record
        print(
            json.dumps(
                {
                    "cp65": "done",
                    "experiment": experiment.name,
                    "status": record.get("status"),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for record in records:
        if record.get("line_metrics"):
            record["comparison"] = _classify(record, references)

    primary_cache = cache_meta_by_horizon.get(5) or next(iter(cache_meta_by_horizon.values()), {})
    payload = {
        "cp": "CP65-LM",
        "rules": {
            "role": "line_model",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "target": "raw_future_return",
            "line_target_type": "raw_future_return",
            "band_target_type_used_for_line_eval": False,
            "checkpoint_selection": "line_gate",
            "coverage_gate_used": False,
            "combined_gate_used": False,
            "band_model_experiment": False,
            "composite_overlay_metrics_used": False,
            "db_write": False,
            "db_read_sync": False,
            "existing_1d_cache_only": True,
            "cache_fingerprint_created_or_deleted": False,
            "save_run": False,
            "full_473_training": False,
            "wandb": "off",
            "compile": False,
            "atr_intraday_range_as_model_feature": False,
        },
        "cache": primary_cache,
        "cache_by_horizon": cache_meta_by_horizon,
        "feature_set_cli": {
            "source_plan": "docs/cp63_bm_feature_set_plan.json",
            "feature_sets": sorted({experiment.feature_set for experiment in selected_experiments}),
        },
        "experiments": records,
        "references": references,
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP65 LM PatchTST feature smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-hash", default="0c1d7f52")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", choices=[experiment.name for experiment in EXPERIMENTS], default=None)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "cp": payload["cp"],
                "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
                "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
                "experiment_count": len(payload["experiments"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
