from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
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

import ai.cp209_lm_f4_f6_pre_ship_verification as cp209  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402

cp175 = cp209.cp175
cp208z = cp209.cp208z

DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp210_ensemble_ship_verification"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "cp210"
EXPORT_DIR = ARTIFACT_DIR / "exports"

PROGRESS_LOG = LOG_DIR / "progress.log"
PROGRESS_MD = DOCS_DIR / "cp210_progress_latest.md"
PROGRESS_JSON = DOCS_DIR / "cp210_progress_latest.json"
REPORT_MD = DOCS_DIR / "cp210_ensemble_report.md"
METRICS_JSON = DOCS_DIR / "cp210_ensemble_metrics.json"
FOLD_CSV = DOCS_DIR / "cp210_ensemble_fold_metrics.csv"
SUMMARY_CSV = DOCS_DIR / "cp210_ensemble_summary.csv"

SEEDS = [7, 13, 23, 42, 71]
SHIP_CUTS = {
    "ic_min": 0.030,
    "false_safe_max": 0.210,
    "severe_recall_min": 0.750,
    "wf_nonnegative_ic_fold_min": 3,
    "wf_ic_range_max": 0.040,
}
RUN_START = time.perf_counter()


def now_utc() -> str:
    from datetime import datetime, timezone

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


def update_progress(
    stage: str,
    *,
    current_candidate: str = "",
    completed_candidates: int = 0,
    total_candidates: int = 0,
    last_metric_snapshot: dict[str, Any] | None = None,
    next_stage: str = "",
    risk: str = "",
    eta: str = "",
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "time": now_utc(),
        "stage": stage,
        "elapsed_seconds": round(time.perf_counter() - RUN_START, 3),
        "current_candidate": current_candidate,
        "completed_candidates": int(completed_candidates),
        "total_candidates": int(total_candidates),
        "last_metric_snapshot": last_metric_snapshot or {},
        "next_stage": next_stage,
        "risk": risk,
        "eta": eta,
    }
    with PROGRESS_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean_json(row), ensure_ascii=False, sort_keys=True) + "\n")
    write_json(PROGRESS_JSON, row)
    done = f"{completed_candidates}/{total_candidates}" if total_candidates else "0/0"
    PROGRESS_MD.write_text(
        "\n".join(
            [
                "# CP210 진행 현황",
                "",
                f"- 현재 stage: `{stage}`",
                f"- 현재 후보: `{current_candidate or '-'}`",
                f"- 완료/전체: `{done}`",
                f"- 마지막 metric: `{json.dumps(clean_json(last_metric_snapshot or {}), ensure_ascii=False)}`",
                f"- 위험/문제: `{risk or '-'}`",
                f"- 다음 stage: `{next_stage or '-'}`",
                f"- ETA: `{eta or '-'}`",
                f"- 마지막 갱신: `{row['time']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def checkpoint_paths(candidate: cp209.VerificationCandidate) -> dict[int, Path]:
    cp208z_dir = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp208z"
    cp209_dir = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp209" / "seed_stability"
    if candidate.key == "F4_b4":
        seed42 = cp208z_dir / "cp208z_patchtst_b4p0_F4_stress_delta_plus_yield_curve_seed42.pt"
    elif candidate.key == "F6_b7":
        seed42 = cp208z_dir / "cp208z_patchtst_b7p0_F6_stress_plus_fragility_union_seed42.pt"
    else:
        raise ValueError(f"알 수 없는 후보입니다: {candidate.key}")
    return {
        7: cp209_dir / f"cp209_seed_{candidate.key}_seed7.pt",
        13: cp209_dir / f"cp209_seed_{candidate.key}_seed13.pt",
        23: cp209_dir / f"cp209_seed_{candidate.key}_seed23.pt",
        42: seed42,
        71: cp209_dir / f"cp209_seed_{candidate.key}_seed71.pt",
    }


def verify_checkpoints() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in cp209.CANDIDATES:
        for seed, path in checkpoint_paths(candidate).items():
            rows.append(
                {
                    "candidate_key": candidate.key,
                    "seed": seed,
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else None,
                }
            )
    missing = [row for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"CP210 ensemble checkpoint 누락: {missing}")
    return rows


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ic": metrics.get("ic_mean"),
        "false_safe": metrics.get("line_top_decile_false_safe_rate"),
        "severe_recall": metrics.get("severe_downside_recall_line_negative"),
        "spread": metrics.get("long_short_spread"),
        "fee": metrics.get("fee_adjusted_return"),
    }


def collect_seed_predictions(
    *,
    candidate: cp209.VerificationCandidate,
    bundle: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    line_scores: list[np.ndarray] = []
    first_prediction: dict[str, Any] | None = None
    for seed, path in checkpoint_paths(candidate).items():
        model, _checkpoint = load_checkpoint(path)
        model = model.to(device)
        model.eval()
        prediction = cp175.collect_predictions(
            model,
            cp175.LineTrialDataset(bundle, include_atr=False),
            metadata=bundle.metadata,
            device=device,
            batch_size=batch_size,
        )
        line_scores.append(np.asarray(prediction["line_score"], dtype=np.float64))
        if first_prediction is None:
            first_prediction = prediction
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    assert first_prediction is not None
    return np.vstack(line_scores), first_prediction


def ensemble_prediction(seed_scores: np.ndarray, base_prediction: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "mean":
        score = np.mean(seed_scores, axis=0)
    elif mode == "median":
        score = np.median(seed_scores, axis=0)
    else:
        raise ValueError(f"지원하지 않는 ensemble mode입니다: {mode}")
    return {
        "line_score": score,
        "actual": np.asarray(base_prediction["actual"], dtype=np.float64),
        "metadata": base_prediction["metadata"].reset_index(drop=True).copy(),
    }


def evaluate_ensemble(
    *,
    candidate: cp209.VerificationCandidate,
    bundle: Any,
    split: str,
    train_q10: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    seed_scores, base_prediction = collect_seed_predictions(candidate=candidate, bundle=bundle, device=device, batch_size=batch_size)
    mean_prediction = ensemble_prediction(seed_scores, base_prediction, "mean")
    median_prediction = ensemble_prediction(seed_scores, base_prediction, "median")
    mean_metrics = cp175.evaluate_prediction(
        candidate_id=f"cp210_{candidate.key}_ensemble_mean",
        split=split,
        prediction=mean_prediction,
        train_q10_downside=train_q10,
    )
    median_metrics = cp175.evaluate_prediction(
        candidate_id=f"cp210_{candidate.key}_ensemble_median",
        split=split,
        prediction=median_prediction,
        train_q10_downside=train_q10,
    )
    return {
        "seed_scores": seed_scores,
        "base_prediction": base_prediction,
        "mean_prediction": mean_prediction,
        "median_prediction": median_prediction,
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
    }


def latest_ensemble_rows(
    *,
    candidate: cp209.VerificationCandidate,
    bundle: Any,
    device: torch.device,
    recent_rows_per_ticker: int = 1,
) -> pd.DataFrame:
    if recent_rows_per_ticker < 1:
        raise ValueError("recent_rows_per_ticker는 1 이상이어야 합니다.")

    def _recent_rows(model: Any, model_id: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for ticker, arrays in sorted(bundle.ticker_arrays.items()):
            features_full = np.asarray(arrays["features"], dtype=np.float32)
            dates = pd.to_datetime(arrays["dates"])
            if len(features_full) < cp208z.SEQ_LEN:
                continue
            start_idx = max(cp208z.SEQ_LEN - 1, len(features_full) - recent_rows_per_ticker)
            for end_idx in range(start_idx, len(features_full)):
                window = (
                    torch.from_numpy(features_full[end_idx - cp208z.SEQ_LEN + 1 : end_idx + 1])
                    .to(torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
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
        return cp208z.add_rank_columns(pd.DataFrame(rows))

    frames: list[pd.DataFrame] = []
    for seed, path in checkpoint_paths(candidate).items():
        model, _checkpoint = load_checkpoint(path)
        model = model.to(device)
        model.eval()
        frame = _recent_rows(model, f"cp210_{candidate.key}_seed{seed}")
        frames.append(frame[["ticker", "asof_date", "line_score"]].rename(columns={"line_score": f"seed_{seed}"}))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["ticker", "asof_date"], how="inner")
    seed_cols = [col for col in merged.columns if col.startswith("seed_")]
    merged["line_score"] = merged[seed_cols].mean(axis=1)
    merged["safe_line_score"] = merged["line_score"]
    merged["actual_h5_return"] = np.nan
    merged["model_id"] = f"cp210_{candidate.key}_ensemble_mean"
    merged["source_cp"] = "CP210"
    return cp208z.add_rank_columns(merged[["ticker", "asof_date", "line_score", "safe_line_score", "actual_h5_return", "model_id", "source_cp"]])


def export_ensemble(
    *,
    candidate: cp209.VerificationCandidate,
    test_result: dict[str, Any],
    bundle: Any,
    device: torch.device,
) -> dict[str, Any]:
    historical = cp208z.prediction_to_frame(test_result["mean_prediction"], f"cp210_{candidate.key}_ensemble_mean")
    latest = latest_ensemble_rows(candidate=candidate, bundle=bundle, device=device)
    combined = pd.concat([historical, latest], ignore_index=True).drop_duplicates(["ticker", "asof_date"], keep="last")
    combined = combined.sort_values(["ticker", "asof_date"]).reset_index(drop=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    label = "F4b4" if candidate.key == "F4_b4" else "F6b7"
    path = EXPORT_DIR / f"cp210_{label}_ensemble_predictions_line_1d.parquet"
    combined.to_parquet(path, index=False, compression="snappy")
    read_back = pd.read_parquet(path)
    return {
        "candidate_key": candidate.key,
        "path": str(path),
        "row_count": int(len(read_back)),
        "ticker_count": int(read_back["ticker"].nunique()),
        "latest_asof_date": str(read_back["asof_date"].max()),
        "actual_h5_nan_rows": int(read_back["actual_h5_return"].isna().sum()),
        "parquet_read_pass": bool(len(read_back) == len(combined)),
    }


def pass_status(test_metrics: dict[str, Any], wf_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ic = float(test_metrics.get("ic_mean") or -999)
    false_safe = float(test_metrics.get("line_top_decile_false_safe_rate") or 999)
    recall = float(test_metrics.get("severe_downside_recall_line_negative") or -999)
    wf_ic = [float(row["ic"]) for row in wf_rows]
    nonnegative = sum(value >= 0 for value in wf_ic)
    wf_range = max(wf_ic) - min(wf_ic) if wf_ic else math.inf
    checks = {
        "test_ic_pass": ic >= SHIP_CUTS["ic_min"],
        "test_false_safe_pass": false_safe <= SHIP_CUTS["false_safe_max"],
        "test_severe_recall_pass": recall >= SHIP_CUTS["severe_recall_min"],
        "wf_nonnegative_ic_pass": nonnegative >= SHIP_CUTS["wf_nonnegative_ic_fold_min"],
        "wf_ic_range_pass": wf_range <= SHIP_CUTS["wf_ic_range_max"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "wf_nonnegative_ic_folds": nonnegative,
        "wf_ic_range": wf_range,
    }


def final_label(candidate_eval: dict[str, dict[str, Any]]) -> str:
    f4 = bool(candidate_eval["F4_b4"]["ship"]["pass"])
    f6 = bool(candidate_eval["F6_b7"]["ship"]["pass"])
    if f4 and f6:
        return "SHIP_BOTH"
    if f4:
        return "SHIP_F4"
    if f6:
        return "SHIP_F6"
    return "NO_SHIP"


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# CP210 Ensemble Ship Verification",
        "",
        "- 목적: CP209가 학습한 F4 beta=4, F6 beta=7의 5 seed checkpoint를 새 학습 없이 ensemble forward로 ship 판정한다.",
        "- Ensemble: mean을 ship 판정 기준으로 사용하고, median은 부산물로 기록했다.",
        "- 금지 준수: 새 학습, 새 seed, backbone/beta/feature 변경, pinball/curriculum/hybrid 없음.",
        f"- 진행 로그: `{PROGRESS_LOG}`",
        "",
        "## Ship 기준",
        "",
        "| metric | 기준 |",
        "|---|---:|",
        "| test IC | >= 0.030 |",
        "| test false-safe | <= 0.210 |",
        "| test severe recall | >= 0.750 |",
        "| WF fold IC | 4 fold 중 3 fold >= 0 |",
        "| WF IC range | <= 0.040 |",
        "",
        "## Test Ensemble Mean",
        "",
        "| 후보 | IC | false-safe | severe recall | spread | fee | ship pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, result in payload["candidate_eval"].items():
        metrics = result["test_mean"]
        lines.append(
            f"| {key} | {metrics['ic_mean']:.6f} | {metrics['line_top_decile_false_safe_rate']:.6f} | "
            f"{metrics['severe_downside_recall_line_negative']:.6f} | {metrics['long_short_spread']:.6f} | "
            f"{metrics['fee_adjusted_return']:.6f} | {result['ship']['pass']} |"
        )
    lines.extend(["", "## Walk-Forward Ensemble Mean", "", "| 후보 | fold | IC | false-safe | severe recall | spread |", "|---|---|---:|---:|---:|---:|"])
    for row in payload["fold_rows"]:
        if row["mode"] != "mean":
            continue
        lines.append(
            f"| {row['candidate_key']} | {row['fold']} | {row['ic']:.6f} | {row['false_safe']:.6f} | "
            f"{row['severe_recall']:.6f} | {row['spread']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 최종 판정",
            "",
            f"- 라벨: `{payload['final_label']}`",
            f"- 권장: {payload['recommendation']}",
            "",
            "판단은 사용자 몫이다. 이 보고서는 ensemble이 완화된 ship 기준을 만족하는지 보조한다.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    update_progress("started", total_candidates=10, next_stage="checkpoint_verify", eta="1~2시간 예상")
    checkpoint_rows = verify_checkpoints()
    write_csv(DOCS_DIR / "cp210_checkpoint_audit.csv", checkpoint_rows)
    update_progress("checkpoint_verified", total_candidates=10, next_stage="payload_load", eta="데이터 로딩")

    device = cp209.cp208z.device_from_arg(args.device)
    update_progress("runtime_locked", current_candidate=str(device), next_stage="payload_load")
    prepared = cp209.build_prepared_splits()
    split_cache = prepared["split_cache"]
    train_q10 = prepared["train_q10"]
    update_progress(
        "payload_loaded",
        total_candidates=10,
        next_stage="test_ensemble",
        last_metric_snapshot={"source_hash": prepared["payload"].get("source_hash"), **prepared["base_counts"]},
    )

    candidate_eval: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    completed = 0

    for candidate in cp209.CANDIDATES:
        cache = split_cache[candidate.key]
        update_progress("test_ensemble_start", current_candidate=candidate.key, completed_candidates=completed, total_candidates=10, eta="test ensemble forward")
        test_result = evaluate_ensemble(
            candidate=candidate,
            bundle=cache["test"],
            split="test",
            train_q10=train_q10,
            device=device,
            batch_size=args.batch_size,
        )
        completed += 5
        export_rows.append(export_ensemble(candidate=candidate, test_result=test_result, bundle=cache["test"], device=device))
        summary_rows.append({"candidate_key": candidate.key, "mode": "mean", **metric_snapshot(test_result["mean_metrics"])})
        summary_rows.append({"candidate_key": candidate.key, "mode": "median", **metric_snapshot(test_result["median_metrics"])})
        update_progress(
            "test_ensemble_done",
            current_candidate=candidate.key,
            completed_candidates=completed,
            total_candidates=10,
            last_metric_snapshot=metric_snapshot(test_result["mean_metrics"]),
            next_stage="wf_ensemble",
            eta="fold 4개 forward",
        )

        full = cache["full"]
        candidate_fold_rows: list[dict[str, Any]] = []
        for fold, _train_end, test_start, test_end in cp209.FOLDS:
            fold_bundle = cp209.subset_by_date(full, start=test_start, end=test_end)
            fold_result = evaluate_ensemble(
                candidate=candidate,
                bundle=fold_bundle,
                split=fold,
                train_q10=train_q10,
                device=device,
                batch_size=args.batch_size,
            )
            for mode, metrics_key in [("mean", "mean_metrics"), ("median", "median_metrics")]:
                metrics = fold_result[metrics_key]
                row = {
                    "candidate_key": candidate.key,
                    "fold": fold,
                    "mode": mode,
                    "row_count": len(fold_bundle),
                    "ic": metrics.get("ic_mean"),
                    "false_safe": metrics.get("line_top_decile_false_safe_rate"),
                    "severe_recall": metrics.get("severe_downside_recall_line_negative"),
                    "spread": metrics.get("long_short_spread"),
                    "fee": metrics.get("fee_adjusted_return"),
                }
                fold_rows.append(row)
                if mode == "mean":
                    candidate_fold_rows.append(row)
            write_csv(FOLD_CSV, fold_rows)
            update_progress(
                "wf_fold_done",
                current_candidate=f"{candidate.key}_{fold}",
                completed_candidates=len([row for row in fold_rows if row["mode"] == "mean"]),
                total_candidates=8,
                last_metric_snapshot={key: candidate_fold_rows[-1][key] for key in ["ic", "false_safe", "severe_recall", "spread"]},
                next_stage="wf_or_next_candidate",
            )
        candidate_eval[candidate.key] = {
            "test_mean": test_result["mean_metrics"],
            "test_median": test_result["median_metrics"],
            "ship": pass_status(test_result["mean_metrics"], candidate_fold_rows),
        }

    label = final_label(candidate_eval)
    if label == "SHIP_BOTH":
        recommendation = "둘 다 완화 기준 통과. IC 우선이면 F4 beta=4 권장."
    elif label == "SHIP_F4":
        recommendation = "F4 beta=4 ensemble ship 후보."
    elif label == "SHIP_F6":
        recommendation = "F6 beta=7 ensemble ship 후보."
    else:
        recommendation = "둘 다 완화 기준 미달. CP175 frozen 유지 또는 별도 hybrid/v2 재고."
    payload = {
        "created_at": now_utc(),
        "final_label": label,
        "recommendation": recommendation,
        "ship_cuts": SHIP_CUTS,
        "checkpoint_rows": checkpoint_rows,
        "candidate_eval": candidate_eval,
        "fold_rows": fold_rows,
        "summary_rows": summary_rows,
        "export_rows": export_rows,
        "forbidden_actions": {
            "new_training": False,
            "new_seed": False,
            "backbone_beta_feature_change": False,
            "pinball_curriculum_hybrid": False,
            "db_write": False,
            "live_fetch": False,
            "eodhd_fallback": False,
        },
    }
    write_json(METRICS_JSON, payload)
    write_csv(SUMMARY_CSV, summary_rows)
    write_report(payload)
    update_progress("completed", completed_candidates=10, total_candidates=10, last_metric_snapshot={"final_label": label}, next_stage="done", eta="완료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP210 5-seed ensemble ship verification")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        update_progress("failed", risk=f"{type(exc).__name__}: {exc}", next_stage="사용자 확인 필요")
        raise


if __name__ == "__main__":
    main()
