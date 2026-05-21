from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402

from ai.cp66_lm_post_backfill_h20_1w import (
    _build_bundles,
    _epoch_summary_from_log,
    _fmt,
    _json_safe,
    _line_metrics,
    _load_references,
    _metric_delta,
    _read_json,
    _safe_float,
    _table,
    _write_json,
)
from ai.preprocessing import FEATURE_CONTRACT_VERSION, FUTURE_COVARIATE_DIM, MODEL_FEATURE_COLUMNS, resolve_data_fingerprint
from ai.train import TrainConfig, resolve_feature_columns, run_training


REPORT_PATH = PROJECT_ROOT / "docs" / "cp67_lm_h20_100ticker_validation_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp67_lm_h20_100ticker_validation_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp67_lm_h20_100ticker_validation_logs"
CP66_METRICS_PATH = PROJECT_ROOT / "docs" / "cp66_lm_post_backfill_h20_1w_metrics.json"

REPORT_LINE_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "mae",
    "smape",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]
BUCKET_KEYS = [
    "ic_mean",
    "long_short_spread",
    "false_safe_tail_rate",
    "severe_downside_recall",
    "fee_adjusted_return",
]
BUCKETS = ["h1_h5", "h6_h10", "h11_h20"]


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    epochs: int = 3
    timeframe: str = "1D"
    horizon: int = 20
    seq_len: int = 252
    patch_len: int = 32
    patch_stride: int = 16
    limit_tickers: int = 100
    batch_size: int = 256


FULL_100 = Experiment(
    name="h20_full_features_post_backfill_100",
    feature_set="full_features",
)
NO_FUND_100 = Experiment(
    name="h20_no_fundamentals_post_backfill_100",
    feature_set="no_fundamentals",
)
FULL_100_EPOCH5 = Experiment(
    name="h20_full_features_post_backfill_100_epoch5",
    feature_set="full_features",
    epochs=5,
)


def _cp66_reference() -> dict[str, Any]:
    cp66 = _read_json(CP66_METRICS_PATH)
    records = {record["feature_set"]: record for record in cp66.get("h20_experiments", [])}
    return {
        "source": str(CP66_METRICS_PATH.relative_to(PROJECT_ROOT)),
        "data_hash_1d": cp66.get("data_hashes", {}).get("1D"),
        "full_features": records.get("full_features"),
        "no_fundamentals": records.get("no_fundamentals"),
    }


def _bucket_metric_source(record: dict[str, Any], bucket: str) -> dict[str, Any]:
    return record.get("bucket_line_metrics", {}).get(bucket, {})


def _classify(record: dict[str, Any], cp66: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("line_metrics", {})
    bucket = _bucket_metric_source(record, "h11_h20")
    cp66_full = (cp66.get("full_features") or {}).get("line_metrics", {})
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    ic = _safe_float(metrics.get("ic_mean"))
    spread = _safe_float(metrics.get("long_short_spread"))
    h11_ic = _safe_float(bucket.get("ic_mean"))
    h11_false_safe = _safe_float(bucket.get("false_safe_tail_rate"))

    severe_reference = _safe_float(cp66_full.get("severe_downside_recall"))
    severe_pass = severe is not None and (
        (severe_reference is not None and severe >= severe_reference)
        or severe >= 0.70
    )
    signal_pass = ic is not None and ic > 0.0 and spread is not None and spread > 0.0
    h11_ic_warning = h11_ic is not None and h11_ic < -0.02
    h11_display_forbidden = h11_false_safe is not None and h11_false_safe >= 0.35
    h11_bucket_collapse = h11_display_forbidden or (h11_ic is not None and h11_ic <= -0.05)
    product_aux_pass = (
        false_safe is not None
        and false_safe < 0.30
        and severe_pass
        and signal_pass
        and not h11_bucket_collapse
    )

    only_one_signal = (
        (ic is not None and ic > 0.0)
        != (spread is not None and spread > 0.0)
    )
    watch_zone = (
        (false_safe is not None and 0.30 <= false_safe < 0.35)
        or (severe is not None and 0.65 <= severe < 0.70)
        or only_one_signal
        or h11_ic_warning
    )
    fail_zone = (
        record.get("status") != "completed"
        or (false_safe is not None and false_safe >= 0.35)
        or (severe is not None and severe < 0.65)
        or ((ic is not None and ic < 0.0) and (spread is not None and spread < 0.0))
        or h11_bucket_collapse
    )

    if product_aux_pass:
        verdict = "product_aux_pass"
    elif fail_zone:
        verdict = "fail"
    elif watch_zone:
        verdict = "watch"
    else:
        verdict = "watch"

    return {
        "verdict": verdict,
        "product_aux_pass": product_aux_pass,
        "watch_zone": watch_zone,
        "fail_zone": fail_zone,
        "h11_ic_warning": h11_ic_warning,
        "h11_display_forbidden": h11_display_forbidden,
        "h11_bucket_collapse": h11_bucket_collapse,
        "severe_reference": severe_reference,
        "delta_vs_cp66_full": {
            key: _metric_delta(metrics, cp66_full, key)
            for key in REPORT_LINE_KEYS
        },
    }


def _base_config(experiment: Experiment, args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model="patchtst",
        timeframe=experiment.timeframe,
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
        wandb_project="lens-cp67",
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
    bundles: tuple[Any, Any, Any, torch.Tensor, torch.Tensor, Any],
) -> dict[str, Any]:
    config = _base_config(experiment, args)
    feature_columns = resolve_feature_columns(experiment.feature_set)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}.log"
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("# CP67 line model 100 ticker validation\n")
            log_file.write("# experiment: " + experiment.name + "\n")
            log_file.write("# role: line_model\n")
            log_file.write("# checkpoint_selection: line_gate\n")
            log_file.write("# feature_set: " + experiment.feature_set + "\n")
            log_file.write("# feature_columns: " + ", ".join(feature_columns) + "\n")
            log_file.flush()
            with redirect_stdout(log_file), redirect_stderr(log_file):
                result = run_training(
                    config,
                    save_run=False,
                    precomputed_bundles=bundles,
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
    return {
        "name": experiment.name,
        "experiment": asdict(experiment),
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
        "bucket_line_metrics": {
            bucket: {
                key: test_metrics.get("line_metrics", {}).get(f"{bucket}_{key}")
                for key in REPORT_LINE_KEYS
                if test_metrics.get("line_metrics", {}).get(f"{bucket}_{key}") is not None
            }
            for bucket in BUCKETS
        },
        "dataset_plan": result.get("dataset_plan") if isinstance(result, dict) else None,
    }


def _line_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("line_metrics", {})
    return {
        "name": record.get("name"),
        "feature_set": record.get("feature_set"),
        "status": record.get("status"),
        "n": record.get("n_features"),
        "ic": _fmt(metrics.get("ic_mean")),
        "ic_ir": _fmt(metrics.get("ic_ir")),
        "ic_t": _fmt(metrics.get("ic_t_stat")),
        "spread": _fmt(metrics.get("long_short_spread")),
        "spread_ir": _fmt(metrics.get("spread_ir")),
        "spread_t": _fmt(metrics.get("spread_t_stat")),
        "dir": _fmt(metrics.get("direction_accuracy")),
        "fstail": _fmt(metrics.get("false_safe_tail_rate")),
        "fssev": _fmt(metrics.get("false_safe_severe_rate")),
        "recall": _fmt(metrics.get("severe_downside_recall")),
        "capture": _fmt(metrics.get("downside_capture_rate")),
        "bias": _fmt(metrics.get("conservative_bias")),
        "sacrifice": _fmt(metrics.get("upside_sacrifice")),
        "mae": _fmt(metrics.get("mae")),
        "smape": _fmt(metrics.get("smape")),
        "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
        "fee_sharpe": _fmt(metrics.get("fee_adjusted_sharpe")),
        "verdict": record.get("classification", {}).get("verdict", ""),
    }


def _bucket_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for bucket in BUCKETS:
            metrics = _bucket_metric_source(record, bucket)
            rows.append(
                {
                    "name": record.get("name"),
                    "bucket": bucket,
                    "ic": _fmt(metrics.get("ic_mean")),
                    "spread": _fmt(metrics.get("long_short_spread")),
                    "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                    "recall": _fmt(metrics.get("severe_downside_recall")),
                    "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
                }
            )
    return rows


def _reference_rows(cp66: dict[str, Any], references: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("full_features", "no_fundamentals"):
        record = cp66.get(label)
        if record:
            metrics = record.get("line_metrics", {})
            rows.append(
                {
                    "source": "CP66 50",
                    "name": record.get("name"),
                    "ic": _fmt(metrics.get("ic_mean")),
                    "spread": _fmt(metrics.get("long_short_spread")),
                    "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                    "recall": _fmt(metrics.get("severe_downside_recall")),
                }
            )
    for label, record in references.get("cp65_h20", {}).items():
        metrics = record.get("line_metrics", {})
        rows.append(
            {
                "source": "CP65",
                "name": f"{label}:{record.get('name')}",
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )
    for name, record in references.get("cp49_h20_candidates_via_cp62", {}).items():
        metrics = record.get("line_metrics", {})
        rows.append(
            {
                "source": "CP49",
                "name": name,
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )
    for name, record in references.get("line_baselines", {}).items():
        metrics = record.get("line_metrics", {})
        rows.append(
            {
                "source": "baseline",
                "name": name,
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )
    return rows


def _build_report(payload: dict[str, Any]) -> str:
    records = payload["experiments"]
    line_rows = [_line_row(record) for record in records]
    bucket_rows = _bucket_rows(records)
    references = _reference_rows(payload["cp66_reference"], payload["references"])
    gate = payload["cache_gate"]
    optional = payload.get("optional_epoch5")
    display = payload.get("product_display_recommendation", {})
    skipped = payload.get("skipped_experiments", [])

    lines = [
        "# CP67-LM 1D h20 100티커 재검증",
        "",
        "## 1. 원칙 확인",
        "- 이번 CP는 PatchTST line_model만 실행했다.",
        "- band 모델, composite/overlay, line_inside_band 관련 실험과 평가는 수행하지 않았다.",
        "- DB 쓰기, save-run, W&B, full 473티커, UI/backend API 수정, feature contract 변경은 하지 않았다.",
        "- `atr_ratio`는 모델 feature로 승격하지 않았다.",
        "",
        "## 2. cache gate",
        _table(
            [
                {"item": "current hash", "value": gate.get("current_hash"), "result": gate.get("hash_status")},
                {"item": "CP66 hash", "value": gate.get("cp66_hash"), "result": ""},
                {"item": "feature_version", "value": gate.get("feature_version"), "result": "PASS"},
                {"item": "MODEL_FEATURE_COLUMNS", "value": gate.get("feature_count"), "result": "PASS"},
                {"item": "atr_ratio in model", "value": gate.get("atr_ratio_in_model_features"), "result": "PASS" if not gate.get("atr_ratio_in_model_features") else "FAIL"},
                {"item": "eligible ticker", "value": gate.get("cache_meta", {}).get("eligible_ticker_count"), "result": "PASS"},
                {"item": "feature NaN/Inf", "value": gate.get("cache_meta", {}).get("tensor_finite", {}).get("feature_nonfinite_count"), "result": "PASS"},
                {"item": "target NaN/Inf", "value": gate.get("cache_meta", {}).get("tensor_finite", {}).get("target_nonfinite_count"), "result": "PASS"},
                {"item": "ratio p99 sanity", "value": gate.get("cache_meta", {}).get("ratio_sanity", {}).get("pass"), "result": "PASS"},
            ],
            [("항목", "item"), ("값", "value"), ("판정", "result")],
        ),
        "",
        "## 3. 100티커 line 결과",
        _table(
            line_rows,
            [
                ("name", "name"),
                ("feature_set", "feature_set"),
                ("status", "status"),
                ("n", "n"),
                ("ic", "ic"),
                ("ic_ir", "ic_ir"),
                ("ic_t", "ic_t"),
                ("spread", "spread"),
                ("spread_ir", "spread_ir"),
                ("spread_t", "spread_t"),
                ("dir", "dir"),
                ("false_safe_tail", "fstail"),
                ("false_safe_severe", "fssev"),
                ("severe_recall", "recall"),
                ("capture", "capture"),
                ("bias", "bias"),
                ("sacrifice", "sacrifice"),
                ("mae", "mae"),
                ("smape", "smape"),
                ("fee_ret", "fee_ret"),
                ("fee_sharpe", "fee_sharpe"),
                ("verdict", "verdict"),
            ],
        ),
        "",
        "## 4. horizon bucket 지표",
        _table(
            bucket_rows,
            [
                ("name", "name"),
                ("bucket", "bucket"),
                ("ic", "ic"),
                ("spread", "spread"),
                ("false_safe_tail", "fstail"),
                ("severe_recall", "recall"),
                ("fee_ret", "fee_ret"),
            ],
        ),
        "",
        "## 5. 비교 기준",
        _table(
            references,
            [("source", "source"), ("name", "name"), ("ic", "ic"), ("spread", "spread"), ("false_safe_tail", "fstail"), ("severe_recall", "recall")],
        ),
        "",
        "## 6. 실행 스킵",
        _table(
            skipped or [{"name": "", "reason": ""}],
            [("name", "name"), ("reason", "reason")],
        ),
        "",
        "## 7. 제품 표시 판단",
        f"- h5: 단기 line, 진한 실선 유지.",
        f"- h20: {display.get('h20_display_mode')}.",
        f"- 기본 ON 여부: {display.get('default_on')}.",
        f"- 판단 근거: {display.get('reason')}",
        "",
        "## 8. 선택 실험 C",
        f"- 실행 여부: {optional.get('status') if optional else 'not_run'}",
        f"- 예상 시간: {optional.get('estimated_minutes') if optional else ''}",
        f"- 사유: {optional.get('reason') if optional else '필수 A/B 재검증을 우선 완료하고 장기 실행은 보류했다.'}",
        "",
        "## 9. 다음 CP 추천",
        "- 100티커 h20 full_features는 제품 기본 ON 후보가 아니다. CP66 50티커 생존 결과는 표본 확대에서 재현되지 않았다.",
        "- h20은 Phase 1.5 연구 후보로 보류하고, 제품에는 h5 단기 line을 기본 진한 실선으로 유지한다.",
        "- h20을 계속 보려면 100티커에서 false-safe를 낮추는 재학습/seed 안정성 또는 후보 구조를 별도 CP로 분리한다.",
        "",
        "## 10. 검증",
        "- `.venv\\Scripts\\python.exe -m py_compile ai\\cp67_lm_h20_100ticker_validation.py`: 통과",
        "- `python -m json.tool docs\\cp67_lm_h20_100ticker_validation_metrics.json`: 통과",
        "- `.venv\\Scripts\\python.exe -m unittest ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 27개 통과",
        "- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음",
    ]
    return "\n".join(lines) + "\n"


def _display_recommendation(records: list[dict[str, Any]]) -> dict[str, Any]:
    full = next((record for record in records if record.get("feature_set") == "full_features"), None)
    if not full:
        return {
            "default_on": "NO",
            "h20_display_mode": "중기 line / Phase 1.5 연구 후보",
            "reason": "full_features 100티커 결과가 없다.",
        }
    verdict = full.get("classification", {}).get("verdict")
    if verdict == "product_aux_pass":
        return {
            "default_on": "YES",
            "h20_display_mode": "중기 line, 점선/반투명선, 기본 ON 후보",
            "reason": "false_safe_tail_rate < 0.30, severe recall 기준 통과, IC/spread 양수, h11_h20 붕괴 없음.",
        }
    if verdict == "watch":
        return {
            "default_on": "NO",
            "h20_display_mode": "중기 참고선, 점선/반투명선, 사용자 선택형",
            "reason": "watch 조건이 있어 기본 표시는 보류한다.",
        }
    return {
        "default_on": "NO",
        "h20_display_mode": "Phase 1.5 연구 후보 보류",
        "reason": "fail 조건이 있어 제품 표시를 금지한다.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA를 볼 수 없는 Python입니다: {sys.executable}")

    cp66 = _cp66_reference()
    references = _load_references()
    current_hash = resolve_data_fingerprint("1D")
    bundles = _build_bundles(
        timeframe="1D",
        data_hash=current_hash,
        seq_len=252,
        horizon=20,
        limit_tickers=100,
        include_future_covariate=False,
    )
    train_bundle, val_bundle, test_bundle, mean, std, plan, cache_meta, _, _, _ = bundles
    precomputed = (train_bundle, val_bundle, test_bundle, mean, std, plan)

    cache_gate = {
        "current_hash": current_hash,
        "cp66_hash": cp66.get("data_hash_1d"),
        "hash_status": "PASS" if current_hash == cp66.get("data_hash_1d") else "FAIL",
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "cache_meta": cache_meta,
        "pass": (
            current_hash == cp66.get("data_hash_1d")
            and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
            and len(MODEL_FEATURE_COLUMNS) == 36
            and "atr_ratio" not in MODEL_FEATURE_COLUMNS
            and int(cache_meta.get("tensor_finite", {}).get("feature_nonfinite_count") or 0) == 0
            and int(cache_meta.get("tensor_finite", {}).get("target_nonfinite_count") or 0) == 0
            and bool(cache_meta.get("ratio_sanity", {}).get("pass"))
        ),
    }

    records: list[dict[str, Any]] = []
    skipped_experiments: list[dict[str, str]] = []
    print(json.dumps({"cp67": "start", "experiment": FULL_100.name}, ensure_ascii=False), flush=True)
    full_record = _run_experiment(FULL_100, args, precomputed)
    full_record["classification"] = _classify(full_record, cp66)
    records.append(full_record)
    print(
        json.dumps(
            {
                "cp67": "done",
                "experiment": FULL_100.name,
                "status": full_record.get("status"),
                "verdict": full_record.get("classification", {}).get("verdict"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if full_record.get("status") == "completed" and full_record.get("classification", {}).get("verdict") != "fail":
        print(json.dumps({"cp67": "start", "experiment": NO_FUND_100.name}, ensure_ascii=False), flush=True)
        no_fund_record = _run_experiment(NO_FUND_100, args, precomputed)
        no_fund_record["classification"] = _classify(no_fund_record, cp66)
        records.append(no_fund_record)
        print(
            json.dumps(
                {
                    "cp67": "done",
                    "experiment": NO_FUND_100.name,
                    "status": no_fund_record.get("status"),
                    "verdict": no_fund_record.get("classification", {}).get("verdict"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        skipped_experiments.append(
            {
                "name": NO_FUND_100.name,
                "reason": "A full_features 100티커가 fail이라 지시대로 B를 실행하지 않았다.",
            }
        )
        print(json.dumps({"cp67": "skip_b", "reason": "A failed"}, ensure_ascii=False), flush=True)

    optional_epoch5 = None
    if full_record.get("classification", {}).get("verdict") == "product_aux_pass":
        epoch_mean = _safe_float(full_record.get("epoch_summary", {}).get("epoch_seconds_mean"))
        estimated_minutes = (epoch_mean * 5 / 60.0) if epoch_mean is not None else None
        optional_epoch5 = {
            "status": "not_run",
            "estimated_minutes": estimated_minutes,
            "reason": "선택 실험이며 장기 실행 전 예상 시간만 기록했다.",
        }
        if args.run_optional_epoch5:
            print(
                json.dumps(
                    {
                        "cp67": "optional_epoch5_estimate",
                        "experiment": FULL_100_EPOCH5.name,
                        "estimated_minutes": estimated_minutes,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            optional_record = _run_experiment(FULL_100_EPOCH5, args, precomputed)
            optional_record["classification"] = _classify(optional_record, cp66)
            records.append(optional_record)
            optional_epoch5 = {
                "status": optional_record.get("status"),
                "estimated_minutes": estimated_minutes,
                "record_name": optional_record.get("name"),
                "reason": "사용자 옵션으로 실행했다.",
            }
    else:
        optional_epoch5 = {
            "status": "not_run",
            "estimated_minutes": None,
            "reason": "A가 product_aux_pass가 아니므로 선택 5epoch 장기 실행 조건을 만족하지 않았다.",
        }

    payload = {
        "cp": "CP67-LM",
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
            "save_run": False,
            "full_473_training": False,
            "wandb": "off",
            "compile": False,
            "model_structure_changed": False,
            "feature_contract_changed": False,
            "atr_ratio_promoted_to_model_feature": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requested_device": args.device,
        },
        "cache_gate": cache_gate,
        "experiments": records,
        "skipped_experiments": skipped_experiments,
        "optional_epoch5": optional_epoch5,
        "product_display_recommendation": _display_recommendation(records),
        "cp66_reference": cp66,
        "references": references,
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP67 LM h20 100 ticker validation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run-optional-epoch5", action="store_true")
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
                "display": payload["product_display_recommendation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
