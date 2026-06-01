from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
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

import ai.cp208z_lm_1d_line_final_smoke as cp208z  # noqa: E402
from ai.loss import AsymmetricHuberLoss  # noqa: E402
from ai.preprocessing import SequenceDataset  # noqa: E402
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns  # noqa: E402

torch = cp208z.torch
cp175 = cp208z.cp175
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs" / "cp209_pre_ship_verification"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "cp209"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp209"

PROGRESS_LOG = LOG_DIR / "progress.log"
PROGRESS_MD = DOCS_DIR / "cp209_progress_latest.md"
PROGRESS_JSON = DOCS_DIR / "cp209_progress_latest.json"

SEED_CSV = DOCS_DIR / "cp209_seed_stability.csv"
WF_CSV = DOCS_DIR / "cp209_walk_forward.csv"
METRICS_JSON = DOCS_DIR / "cp209_verification_metrics.json"
REPORT_MD = DOCS_DIR / "cp209_verification_report.md"

CP208Z_SUMMARY = DOCS_DIR / "cp208z_line_smoke_summary.csv"
CP208Z_SEED_RECHECK = DOCS_DIR / "cp208z_seed_recheck.csv"

BASELINE_CP175 = {
    "ic_mean": 0.0420,
    "false_safe": 0.1972,
    "severe_recall": 0.7921,
}
FALSE_SAFE_FOLD_MAX = BASELINE_CP175["false_safe"] + 0.02
SEVERE_RECALL_FOLD_MIN = BASELINE_CP175["severe_recall"] - 0.02

SEEDS = [7, 13, 23, 42, 71]
FOLDS = [
    ("W1", "2024-10-29", "2024-10-30", "2025-02-28"),
    ("W2", "2025-02-28", "2025-03-01", "2025-06-30"),
    ("W3", "2025-06-30", "2025-07-01", "2025-10-31"),
    ("W4", "2025-10-31", "2025-11-01", "2026-05-01"),
]


@dataclass(frozen=True)
class VerificationCandidate:
    key: str
    feature_pack: str
    beta: float
    role: str
    cp208z_candidate_id: str


CANDIDATES = [
    VerificationCandidate(
        key="F4_b4",
        feature_pack="F4_stress_delta_plus_yield_curve",
        beta=4.0,
        role="균형형",
        cp208z_candidate_id="cp208z_patchtst_b4p0_F4_stress_delta_plus_yield_curve_seed42",
    ),
    VerificationCandidate(
        key="F6_b7",
        feature_pack="F6_stress_plus_fragility_union",
        beta=7.0,
        role="보수형",
        cp208z_candidate_id="cp208z_patchtst_b7p0_F6_stress_plus_fragility_union_seed42",
    ),
]

RUN_START = time.perf_counter()


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
                "# CP209 진행 현황",
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


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ic": metrics.get("ic_mean"),
        "false_safe": metrics.get("line_top_decile_false_safe_rate"),
        "severe_recall": metrics.get("severe_downside_recall_line_negative"),
        "spread": metrics.get("long_short_spread"),
        "fee": metrics.get("fee_adjusted_return"),
    }


def row_from_metrics(
    *,
    phase: str,
    candidate: VerificationCandidate,
    seed: int,
    split: str,
    metrics: dict[str, Any],
    source: str,
    run_id: str,
    fold: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "run_id": run_id,
        "candidate_key": candidate.key,
        "candidate_id": candidate.cp208z_candidate_id,
        "role": candidate.role,
        "feature_pack": candidate.feature_pack,
        "beta": candidate.beta,
        "seed": seed,
        "split": split,
        "fold": fold,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "row_count": row_count,
        "source": source,
        **metric_snapshot(metrics),
        "top_count": metrics.get("top_count"),
        "severe_count": metrics.get("severe_count"),
        "line_score_std": metrics.get("line_score_std"),
    }


def load_cp208z_seed42_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not CP208Z_SUMMARY.exists():
        return rows
    frame = pd.read_csv(CP208Z_SUMMARY)
    for candidate in CANDIDATES:
        part = frame[(frame["candidate_id"] == candidate.cp208z_candidate_id) & (frame["split"] == "validation")]
        if part.empty:
            continue
        row = part.iloc[0].to_dict()
        metrics = {
            "ic_mean": float(row["ic_mean"]),
            "line_top_decile_false_safe_rate": float(row["line_top_decile_false_safe_rate"]),
            "severe_downside_recall_line_negative": float(row["severe_downside_recall_line_negative"]),
            "long_short_spread": float(row["long_short_spread"]),
            "fee_adjusted_return": float(row["fee_adjusted_return"]),
            "top_count": int(float(row["top_count"])),
            "severe_count": int(float(row["severe_count"])),
            "line_score_std": float(row["line_score_std"]),
        }
        rows[candidate.key] = row_from_metrics(
            phase="seed_stability",
            candidate=candidate,
            seed=42,
            split="validation",
            metrics=metrics,
            source="cp208z_reuse_full_metric",
            run_id=f"seed_{candidate.key}_42_reuse",
            row_count=int(float(row.get("top_count") or 0)),
        )
    return rows


def seed7_reuse_note() -> str:
    if not CP208Z_SEED_RECHECK.exists():
        return "CP208Z seed_recheck 파일 없음. seed=7은 재학습합니다."
    frame = pd.read_csv(CP208Z_SEED_RECHECK)
    f4 = frame[frame["candidate_id"] == CANDIDATES[0].cp208z_candidate_id]
    if f4.empty:
        return "CP208Z seed=7 재점검에 F4 β=4가 없어서 재학습합니다."
    return "CP208Z F4 β=4 seed=7은 IC만 저장되어 false-safe/recall 안정성 판단에 부족합니다. CP209에서 full metric 기준으로 재학습합니다."


def subset_by_date(bundle: SequenceDataset, start: str | None = None, end: str | None = None) -> SequenceDataset:
    dates = pd.to_datetime(bundle.metadata["asof_date"], errors="coerce")
    mask = pd.Series(True, index=bundle.metadata.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    indices = np.flatnonzero(mask.to_numpy())
    return bundle.subset(indices.tolist())


def combine_datasets(*bundles: SequenceDataset) -> SequenceDataset:
    first = bundles[0]
    refs: list[tuple[str, int]] = []
    metas: list[pd.DataFrame] = []
    for bundle in bundles:
        refs.extend(list(bundle.sample_refs))
        metas.append(bundle.metadata.copy())
    metadata = pd.concat(metas, ignore_index=True)
    return SequenceDataset(
        ticker_arrays=first.ticker_arrays,
        sample_refs=refs,
        metadata=metadata,
        seq_len=first.seq_len,
        horizon=first.horizon,
        mean=first.mean,
        std=first.std,
        include_future_covariate=first.include_future_covariate,
        line_target_type=first.line_target_type,
        band_target_type=first.band_target_type,
    )


def train_eval_checkpoint(
    *,
    run_id: str,
    candidate: VerificationCandidate,
    seed: int,
    train_bundle: SequenceDataset,
    eval_bundle: SequenceDataset,
    split: str,
    train_q10: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    feature_names: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    phase: str,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    cp208z.set_seed(seed)
    n_features = train_bundle.ticker_arrays[next(iter(train_bundle.ticker_arrays))]["features"].shape[1]
    model = cp175._make_model(n_features=n_features, dropout=0.10).to(device)
    criterion = AsymmetricHuberLoss(alpha=1.0, beta=candidate.beta, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.362816234925851e-4, weight_decay=8.143270337695065e-5)
    loader = DataLoader(
        cp175.LineTrialDataset(train_bundle, include_atr=False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
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
        epoch_rows.append({"run_id": run_id, "epoch": epoch, "train_loss": float(np.mean(losses)), "batch_count": len(losses)})

    prediction = cp175.collect_predictions(
        model,
        cp175.LineTrialDataset(eval_bundle, include_atr=False),
        metadata=eval_bundle.metadata,
        device=device,
        batch_size=batch_size,
    )
    metrics = cp175.evaluate_prediction(
        candidate_id=run_id,
        split=split,
        prediction=prediction,
        train_q10_downside=train_q10,
    )
    checkpoint_path = CHECKPOINT_DIR / phase / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "config": {
                "source_cp": "CP209",
                "source_candidate": candidate.cp208z_candidate_id,
                "model": "patchtst",
                "timeframe": "1D",
                "horizon": 5,
                "seq_len": cp208z.SEQ_LEN,
                "dropout": 0.10,
                "band_mode": "direct",
                "num_tickers": 0,
                "ticker_emb_dim": 16,
                "output_role": "legacy",
                "use_revin": True,
                "ci_aggregate": "target",
                "target_channel_idx": 0,
                "ci_target_fast": False,
                "patch_len": cp208z.PATCH_LEN,
                "patch_stride": cp208z.PATCH_STRIDE,
                "patchtst_d_model": 128,
                "patchtst_n_heads": 8,
                "patchtst_n_layers": 3,
                "feature_columns": feature_names,
                "n_features": len(feature_names),
                "feature_set": candidate.feature_pack,
                "beta": candidate.beta,
                "alpha": 1.0,
                "seed": seed,
                "score_contract": "A_score_contract",
                "phase": phase,
            },
            "metrics": metrics,
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
        },
        checkpoint_path,
    )
    pred_frame = cp208z.prediction_to_frame(prediction, run_id)
    pred_path = ARTIFACT_DIR / phase / f"{run_id}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_parquet(pred_path, index=False, compression="snappy")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return metrics, checkpoint_path, epoch_rows


def build_prepared_splits() -> dict[str, Any]:
    payload = cp208z.load_current_payload()
    base_cols = resolve_feature_columns(cp208z.FEATURE_SET)
    base_train, base_val, base_test, _base_mean, _base_std = apply_feature_columns_to_splits(
        payload["train_raw"],
        payload["val_raw"],
        payload["test_raw"],
        payload["mean_raw"],
        payload["std_raw"],
        base_cols,
    )
    base_splits = (base_train, base_val, base_test)
    train_actual = cp175.collect_actual_h5(base_train, batch_size=2048)
    train_q10 = float(np.quantile(train_actual[np.isfinite(train_actual)], 0.10))
    extra_frame = cp208z.build_extra_feature_frame(payload["price"], payload["indicators"])
    split_cache: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        extras = cp208z.FEATURE_PACKS[candidate.feature_pack]
        train_pack, val_pack, test_pack, mean_pack, std_pack = cp208z.build_pack_splits(
            base_splits=base_splits,
            extra_frame=extra_frame,
            extra_names=extras,
        )
        split_cache[candidate.key] = {
            "train": train_pack,
            "validation": val_pack,
            "test": test_pack,
            "full": combine_datasets(train_pack, val_pack, test_pack),
            "mean": mean_pack,
            "std": std_pack,
            "feature_names": [*base_cols, *extras],
        }
    return {
        "payload": payload,
        "train_q10": train_q10,
        "split_cache": split_cache,
        "base_counts": {"train": len(base_train), "validation": len(base_val), "test": len(base_test)},
    }


def evaluate_seed_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = pd.DataFrame(rows)
    for candidate in CANDIDATES:
        part = frame[frame["candidate_key"] == candidate.key].copy()
        ic = part["ic"].astype(float)
        false_safe = part["false_safe"].astype(float)
        recall = part["severe_recall"].astype(float)
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=0))
        cv = float(ic_std / abs(ic_mean)) if abs(ic_mean) > 1e-12 else math.inf
        false_safe_std = float(false_safe.std(ddof=0))
        recall_std = float(recall.std(ddof=0))
        stable = bool(cv <= 0.30 and false_safe_std <= 0.02 and recall_std <= 0.02)
        out[candidate.key] = {
            "label": "STABLE" if stable else "UNSTABLE",
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_cv": cv,
            "false_safe_mean": float(false_safe.mean()),
            "false_safe_std": false_safe_std,
            "severe_recall_mean": float(recall.mean()),
            "severe_recall_std": recall_std,
            "seed_count": int(len(part)),
        }
    return out


def evaluate_walk_forward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = pd.DataFrame(rows)
    for candidate in CANDIDATES:
        part = frame[frame["candidate_key"] == candidate.key].copy()
        ic = part["ic"].astype(float)
        false_safe = part["false_safe"].astype(float)
        recall = part["severe_recall"].astype(float)
        temporal = bool(
            len(part) == 4
            and bool((ic > 0).all())
            and float(ic.max() - ic.min()) <= 0.02
            and bool((false_safe <= FALSE_SAFE_FOLD_MAX).all())
            and bool((recall >= SEVERE_RECALL_FOLD_MIN).all())
        )
        out[candidate.key] = {
            "label": "TEMPORAL_STABLE" if temporal else "TEMPORAL_DRIFT",
            "ic_min": float(ic.min()) if len(ic) else None,
            "ic_max": float(ic.max()) if len(ic) else None,
            "ic_range": float(ic.max() - ic.min()) if len(ic) else None,
            "false_safe_max": float(false_safe.max()) if len(false_safe) else None,
            "severe_recall_min": float(recall.min()) if len(recall) else None,
            "fold_count": int(len(part)),
            "directional_false_safe_max": FALSE_SAFE_FOLD_MAX,
            "directional_severe_recall_min": SEVERE_RECALL_FOLD_MIN,
        }
    return out


def ship_decision(seed_eval: dict[str, Any], wf_eval: dict[str, Any]) -> dict[str, Any]:
    f4_ok = seed_eval["F4_b4"]["label"] == "STABLE" and wf_eval["F4_b4"]["label"] == "TEMPORAL_STABLE"
    f6_ok = seed_eval["F6_b7"]["label"] == "STABLE" and wf_eval["F6_b7"]["label"] == "TEMPORAL_STABLE"
    if f4_ok and f6_ok:
        return {"label": "BOTH_STABLE_USER_CHOICE", "recommendation": "F4 β=4 균형형 권장, F6 β=7 보수형은 예비 후보"}
    if f4_ok:
        return {"label": "SHIP_F4_B4", "recommendation": "F4 β=4 ship 후보"}
    if f6_ok:
        return {"label": "SHIP_F6_B7", "recommendation": "F6 β=7 ship 후보"}
    return {"label": "KEEP_CP175_FROZEN", "recommendation": "두 후보 모두 안정성 기준 미달. CP175 frozen 유지"}


def write_report(
    *,
    source_hash: str,
    seed_rows: list[dict[str, Any]],
    wf_rows: list[dict[str, Any]],
    seed_eval: dict[str, Any],
    wf_eval: dict[str, Any],
    decision: dict[str, Any],
    operating_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# CP209 Pre-Ship Verification Report",
        "",
        f"- source hash: `{source_hash}`",
        "- 목적: CP208Z top 2 후보의 seed instability와 temporal drift를 ship 전 확인",
        "- 신규 연구 금지: 새 backbone, beta 추가 sweep, loss 교체 없음",
        f"- 진행 로그: `{PROGRESS_LOG}`",
        "",
        "## 재사용/재학습 판단",
        "",
        f"- seed=42: CP208Z full metric 재사용",
        f"- seed=7: {seed7_reuse_note()}",
        "",
        "## Seed Stability",
        "",
        "| 후보 | 판정 | IC mean | IC std | IC CV | false-safe std | severe recall std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in seed_eval.items():
        lines.append(
            f"| {key} | {row['label']} | {row['ic_mean']:.6f} | {row['ic_std']:.6f} | "
            f"{row['ic_cv']:.3f} | {row['false_safe_std']:.6f} | {row['severe_recall_std']:.6f} |"
        )
    lines.extend(["", "## Walk-Forward", "", "| 후보 | 판정 | IC range | false-safe max | severe recall min |", "|---|---|---:|---:|---:|"])
    for key, row in wf_eval.items():
        lines.append(
            f"| {key} | {row['label']} | {row['ic_range']:.6f} | {row['false_safe_max']:.6f} | {row['severe_recall_min']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 운영성 재확인",
            "",
            "| 후보 | checkpoint load/export | latest asof | row count |",
            "|---|---|---:|---:|",
        ]
    )
    for row in operating_rows:
        ok = bool(row.get("checkpoint_reload_pass")) and bool(row.get("predictions_line_export_pass"))
        lines.append(f"| {row['candidate_key']} | {ok} | {row.get('latest_asof_date')} | {row.get('row_count')} |")
    lines.extend(
        [
            "",
            "## 최종 결정 보조",
            "",
            f"- 라벨: `{decision['label']}`",
            f"- 권장: {decision['recommendation']}",
            "",
            "판단은 사용자 몫입니다. 이 보고서는 F4/F6 중 어느 후보가 ship 전 안정성 검증을 견디는지 보조합니다.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_only(args: argparse.Namespace) -> None:
    update_progress(
        "ops_resume_start",
        completed_candidates=16,
        total_candidates=16,
        next_stage="ops_recheck",
        eta="payload reload 후 후보 2개 export",
    )
    prepared = build_prepared_splits()
    split_cache = prepared["split_cache"]
    train_q10 = prepared["train_q10"]
    device = cp208z.device_from_arg(args.device)
    seed_rows = pd.read_csv(SEED_CSV).to_dict("records")
    wf_rows = pd.read_csv(WF_CSV).to_dict("records")
    seed_eval = evaluate_seed_stability(seed_rows)
    wf_eval = evaluate_walk_forward(wf_rows)
    operating_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cache = split_cache[candidate.key]
        part = [row for row in seed_rows if row.get("candidate_key") == candidate.key and str(row.get("source", "")).endswith(".pt")]
        if part:
            best = sorted(part, key=lambda row: (float(row.get("ic") or -999), -float(row.get("false_safe") or 999)), reverse=True)[0]
            checkpoint_path = Path(str(best["source"]))
        else:
            checkpoint_path = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp208z" / f"{candidate.cp208z_candidate_id}.pt"
        export = cp208z.export_candidate(
            checkpoint_path=checkpoint_path,
            bundle=cache["test"],
            train_q10=train_q10,
            device=device,
            batch_size=args.batch_size,
        )
        export.pop("test_metrics", None)
        cp209_export_dir = ARTIFACT_DIR / "exports"
        cp209_export_dir.mkdir(parents=True, exist_ok=True)
        src = Path(export["export_path"])
        dst = cp209_export_dir / src.name
        shutil.copy2(src, dst)
        export["cp209_export_path"] = str(dst)
        operating_rows.append({"candidate_key": candidate.key, "checkpoint_path": str(checkpoint_path), **export})
        update_progress(
            "ops_recheck_done",
            current_candidate=candidate.key,
            completed_candidates=len(operating_rows),
            total_candidates=2,
            last_metric_snapshot=operating_rows[-1],
            next_stage="judgment",
        )
    decision = ship_decision(seed_eval, wf_eval)
    payload = {
        "created_at": now_utc(),
        "source_hash": str(prepared["payload"].get("source_hash")),
        "seed_reuse_note": seed7_reuse_note(),
        "seed_eval": seed_eval,
        "walk_forward_eval": wf_eval,
        "decision": decision,
        "seed_rows": seed_rows,
        "walk_forward_rows": wf_rows,
        "operating_rows": operating_rows,
        "forbidden_actions": {
            "new_backbone": False,
            "beta_or_feature_sweep": False,
            "loss_change": False,
            "band_modified": False,
            "frontend_modified": False,
            "db_write": False,
            "live_fetch": False,
            "eodhd_fallback": False,
        },
    }
    write_json(METRICS_JSON, payload)
    write_report(
        source_hash=payload["source_hash"],
        seed_rows=seed_rows,
        wf_rows=wf_rows,
        seed_eval=seed_eval,
        wf_eval=wf_eval,
        decision=decision,
        operating_rows=operating_rows,
    )
    update_progress(
        "completed",
        completed_candidates=16,
        total_candidates=16,
        last_metric_snapshot=decision,
        next_stage="done",
        eta="완료",
    )


def run(args: argparse.Namespace) -> None:
    update_progress("started", total_candidates=16, next_stage="payload_load", eta="전체 예상 1.5~2.5시간")
    device = cp208z.device_from_arg(args.device)
    if device.type != "cuda":
        update_progress("started", risk="CUDA를 잡지 못했습니다. 사용자가 GPU 우선을 요청했으므로 환경 확인 필요", eta="중단 가능")
    else:
        update_progress("runtime_locked", current_candidate=torch.cuda.get_device_name(0), eta="GPU 사용")

    prepared = build_prepared_splits()
    split_cache = prepared["split_cache"]
    train_q10 = prepared["train_q10"]
    source_hash = str(prepared["payload"].get("source_hash"))
    update_progress(
        "baseline_reuse_done",
        completed_candidates=0,
        total_candidates=16,
        last_metric_snapshot={"source_hash": source_hash, **prepared["base_counts"]},
        next_stage="seed_stability",
        eta="seed 신규 8셀 시작",
    )

    seed_rows: list[dict[str, Any]] = list(load_cp208z_seed42_rows().values())
    seed_metric_rows: list[dict[str, Any]] = []
    training_log_rows: list[dict[str, Any]] = []
    completed = 0
    total_new = 16

    for candidate in CANDIDATES:
        cache = split_cache[candidate.key]
        for seed in SEEDS:
            if seed == 42:
                continue
            run_id = f"cp209_seed_{candidate.key}_seed{seed}"
            completed_hint = completed
            update_progress(
                f"seed_{candidate.key}_{seed}_start",
                current_candidate=run_id,
                completed_candidates=completed_hint,
                total_candidates=total_new,
                next_stage="seed_train",
                eta=f"seed 남은 신규 {8 - completed_hint if completed_hint < 8 else 0}셀",
            )
            metrics, checkpoint_path, epoch_rows = train_eval_checkpoint(
                run_id=run_id,
                candidate=candidate,
                seed=seed,
                train_bundle=cache["train"],
                eval_bundle=cache["validation"],
                split="validation",
                train_q10=train_q10,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                feature_names=cache["feature_names"],
                mean=cache["mean"],
                std=cache["std"],
                phase="seed_stability",
            )
            completed += 1
            training_log_rows.extend(epoch_rows)
            row = row_from_metrics(
                phase="seed_stability",
                candidate=candidate,
                seed=seed,
                split="validation",
                metrics=metrics,
                source=str(checkpoint_path),
                run_id=run_id,
                row_count=len(cache["validation"]),
            )
            seed_rows.append(row)
            seed_metric_rows.append(row)
            write_csv(SEED_CSV, seed_rows)
            update_progress(
                f"seed_{candidate.key}_{seed}_done",
                current_candidate=run_id,
                completed_candidates=completed,
                total_candidates=total_new,
                last_metric_snapshot=metric_snapshot(metrics),
                next_stage="seed_or_walk_forward",
                eta="진행 중",
            )

    seed_eval = evaluate_seed_stability(seed_rows)
    update_progress(
        "seed_stability_done",
        completed_candidates=completed,
        total_candidates=total_new,
        last_metric_snapshot=seed_eval,
        next_stage="walk_forward",
        eta="walk-forward 8셀 시작",
    )

    wf_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cache = split_cache[candidate.key]
        full = cache["full"]
        for fold, train_end, test_start, test_end in FOLDS:
            run_id = f"cp209_wf_{candidate.key}_{fold}_seed42"
            train_fold = subset_by_date(full, end=train_end)
            test_fold = subset_by_date(full, start=test_start, end=test_end)
            update_progress(
                f"wf_{candidate.key}_{fold}_start",
                current_candidate=run_id,
                completed_candidates=completed,
                total_candidates=total_new,
                last_metric_snapshot={"train_rows": len(train_fold), "test_rows": len(test_fold)},
                next_stage="wf_train",
                eta=f"walk-forward 남은 {total_new - completed}셀",
            )
            metrics, checkpoint_path, epoch_rows = train_eval_checkpoint(
                run_id=run_id,
                candidate=candidate,
                seed=42,
                train_bundle=train_fold,
                eval_bundle=test_fold,
                split=fold,
                train_q10=train_q10,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                feature_names=cache["feature_names"],
                mean=cache["mean"],
                std=cache["std"],
                phase="walk_forward",
            )
            completed += 1
            training_log_rows.extend(epoch_rows)
            row = row_from_metrics(
                phase="walk_forward",
                candidate=candidate,
                seed=42,
                split=fold,
                metrics=metrics,
                source=str(checkpoint_path),
                run_id=run_id,
                fold=fold,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                row_count=len(test_fold),
            )
            wf_rows.append(row)
            write_csv(WF_CSV, wf_rows)
            update_progress(
                f"wf_{candidate.key}_{fold}_done",
                current_candidate=run_id,
                completed_candidates=completed,
                total_candidates=total_new,
                last_metric_snapshot=metric_snapshot(metrics),
                next_stage="wf_or_ops",
                eta="진행 중",
            )

    wf_eval = evaluate_walk_forward(wf_rows)
    update_progress(
        "walk_forward_done",
        completed_candidates=completed,
        total_candidates=total_new,
        last_metric_snapshot=wf_eval,
        next_stage="ops_recheck",
        eta="운영성 2개 후보 확인",
    )

    operating_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cache = split_cache[candidate.key]
        candidate_seed_rows = [row for row in seed_rows if row["candidate_key"] == candidate.key and row["source"].endswith(".pt")]
        if candidate_seed_rows:
            best_row = sorted(candidate_seed_rows, key=lambda row: (float(row.get("ic") or -999), -float(row.get("false_safe") or 999)), reverse=True)[0]
            checkpoint_path = Path(str(best_row["source"]))
        else:
            checkpoint_path = PROJECT_ROOT / "ai" / "artifacts" / "checkpoints" / "cp208z" / f"{candidate.cp208z_candidate_id}.pt"
        export = cp208z.export_candidate(
            checkpoint_path=checkpoint_path,
            bundle=cache["test"],
            train_q10=train_q10,
            device=device,
            batch_size=args.batch_size,
        )
        export.pop("test_metrics", None)
        operating_rows.append({"candidate_key": candidate.key, "checkpoint_path": str(checkpoint_path), **export})
        update_progress(
            "ops_recheck_done",
            current_candidate=candidate.key,
            completed_candidates=len(operating_rows),
            total_candidates=2,
            last_metric_snapshot=operating_rows[-1],
            next_stage="judgment",
        )

    decision = ship_decision(seed_eval, wf_eval)
    payload = {
        "created_at": now_utc(),
        "source_hash": source_hash,
        "seed_reuse_note": seed7_reuse_note(),
        "seed_eval": seed_eval,
        "walk_forward_eval": wf_eval,
        "decision": decision,
        "seed_rows": seed_rows,
        "walk_forward_rows": wf_rows,
        "operating_rows": operating_rows,
        "training_logs": training_log_rows,
        "forbidden_actions": {
            "new_backbone": False,
            "beta_or_feature_sweep": False,
            "loss_change": False,
            "band_modified": False,
            "frontend_modified": False,
            "db_write": False,
            "live_fetch": False,
            "eodhd_fallback": False,
        },
    }
    write_json(METRICS_JSON, payload)
    write_report(
        source_hash=source_hash,
        seed_rows=seed_rows,
        wf_rows=wf_rows,
        seed_eval=seed_eval,
        wf_eval=wf_eval,
        decision=decision,
        operating_rows=operating_rows,
    )
    write_csv(ARTIFACT_DIR / "cp209_training_logs.csv", training_log_rows)
    update_progress(
        "completed",
        completed_candidates=completed,
        total_candidates=total_new,
        last_metric_snapshot=decision,
        next_stage="done",
        eta="완료",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP209 F4/F6 pre-ship verification")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        if args.finalize_only:
            finalize_only(args)
        else:
            run(args)
    except Exception as exc:
        update_progress("failed", risk=f"{type(exc).__name__}: {exc}", next_stage="사용자 확인 필요")
        raise


if __name__ == "__main__":
    main()
