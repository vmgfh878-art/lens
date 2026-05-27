from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch

bootstrap_torch()

import pandas as pd

V1_DIR = ROOT / "backend" / "data" / "v1"
DOCS_DIR = ROOT / "docs"

BASE_ARTIFACT_PATH = ROOT / "data" / "artifacts" / "cp210" / "exports" / "cp210_F4b4_ensemble_predictions_line_1d.parquet"
TARGET_PATH = V1_DIR / "predictions_line_1d.parquet"
CP175_BACKUP_PATH = V1_DIR / "predictions_line_1d_cp175_frozen_backup.parquet"
MARKET_SNAPSHOT_PATH = V1_DIR / "market_prices_1d.parquet"

LINE_MODEL_ID = "cp210_F4_b4_ensemble_mean"
LINE_SOURCE_CP = "CP208Z_CP209_F4B4"
LINE_SOURCE_NOTE = "CP208Z seed42와 CP209 seed stability 4개 checkpoint를 합친 F4 beta=4 5-seed ensemble"
TAIL_REFRESH_BUSINESS_DAYS = 10
LATEST_REFRESH_ROWS_PER_TICKER = 5

REQUIRED_COLUMNS = [
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


def utc_now() -> str:
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frame_summary(path: Path, *, date_column: str = "asof_date") -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    frame = pd.read_parquet(path)
    return summarize_frame(frame=frame, path=path, date_column=date_column)


def summarize_frame(frame: pd.DataFrame, *, path: Path | None = None, date_column: str = "asof_date") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if path is not None:
        summary["path"] = str(path)
        if path.exists():
            summary["size_mb"] = round(path.stat().st_size / 1024 / 1024, 3)
    if "ticker" in frame.columns:
        summary["ticker_count"] = int(frame["ticker"].nunique())
    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if not dates.empty:
            summary["min_date"] = str(dates.min().date())
            summary["max_date"] = str(dates.max().date())
    if "model_id" in frame.columns and len(frame):
        summary["model_ids"] = sorted(frame["model_id"].dropna().astype(str).unique().tolist())
    if "source_cp" in frame.columns and len(frame):
        summary["source_cps"] = sorted(frame["source_cp"].dropna().astype(str).unique().tolist())
    return summary


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"1D line parquet 필수 컬럼 누락: {missing}")

    normalized = frame[REQUIRED_COLUMNS].copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["asof_date"] = pd.to_datetime(normalized["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized = normalized.dropna(subset=["ticker", "asof_date"])
    for column in [
        "line_score",
        "safe_line_score",
        "line_rank_by_date",
        "safe_line_rank_by_date",
        "actual_h5_return",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["line_top_decile_flag"] = normalized["line_top_decile_flag"].fillna(False).astype(bool)
    normalized["safe_line_top_decile_flag"] = normalized["safe_line_top_decile_flag"].fillna(False).astype(bool)
    normalized["model_id"] = LINE_MODEL_ID
    normalized["source_cp"] = LINE_SOURCE_CP
    normalized = normalized.drop_duplicates(["ticker", "asof_date"], keep="last")
    return normalized.sort_values(["ticker", "asof_date"]).reset_index(drop=True)


def load_baseline_frame(source_path: Path, target_path: Path) -> tuple[pd.DataFrame, Path]:
    baseline_path = target_path if target_path.exists() else source_path
    if not baseline_path.exists():
        raise FileNotFoundError(f"기준 1D line parquet가 없습니다: {baseline_path}")
    frame = pd.read_parquet(baseline_path)
    return normalize_frame(frame), baseline_path


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp_path, index=False, compression="snappy")
    temp_path.replace(path)


def refresh_start_date(base_max_date: str) -> str:
    return str((pd.Timestamp(base_max_date) - pd.offsets.BDay(TAIL_REFRESH_BUSINESS_DAYS)).date())


def build_refreshed_tail(*, base_max_date: str, device_name: str, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    import ai.cp208z_lm_1d_line_final_smoke as cp208z
    import ai.cp209_lm_f4_f6_pre_ship_verification as cp209
    import ai.cp210_lm_ensemble_ship_verification as cp210

    candidate = next(item for item in cp209.CANDIDATES if item.key == "F4_b4")
    prepared = cp209.build_prepared_splits()
    refresh_start = refresh_start_date(base_max_date)
    full_bundle = prepared["split_cache"][candidate.key]["full"]
    tail_bundle = cp209.subset_by_date(full_bundle, start=refresh_start)
    device = cp208z.device_from_arg(device_name)

    seed_scores, base_prediction = cp210.collect_seed_predictions(
        candidate=candidate,
        bundle=tail_bundle,
        device=device,
        batch_size=batch_size,
    )
    mean_prediction = cp210.ensemble_prediction(seed_scores, base_prediction, "mean")
    historical_tail = cp208z.prediction_to_frame(mean_prediction, LINE_MODEL_ID)
    historical_tail["model_id"] = LINE_MODEL_ID
    historical_tail["source_cp"] = LINE_SOURCE_CP

    latest_tail = cp210.latest_ensemble_rows(
        candidate=candidate,
        bundle=full_bundle,
        device=device,
        recent_rows_per_ticker=LATEST_REFRESH_ROWS_PER_TICKER,
    )
    latest_tail["model_id"] = LINE_MODEL_ID
    latest_tail["source_cp"] = LINE_SOURCE_CP

    combined_tail = pd.concat([historical_tail, latest_tail], ignore_index=True)
    combined_tail = normalize_frame(combined_tail)

    diagnostics = {
        "candidate_key": candidate.key,
        "device": str(device),
        "refresh_start": refresh_start,
        "tail_bundle_rows": int(len(tail_bundle)),
        "tail_bundle_tickers": int(tail_bundle.metadata["ticker"].nunique()) if len(tail_bundle) else 0,
        "historical_tail_rows": int(len(historical_tail)),
        "latest_tail_rows": int(len(latest_tail)),
        "latest_refresh_rows_per_ticker": int(LATEST_REFRESH_ROWS_PER_TICKER),
        "tail_summary": summarize_frame(combined_tail),
        "source_hash": prepared["payload"].get("source_hash"),
        "base_counts": prepared["base_counts"],
    }
    return combined_tail, diagnostics


def market_latest_date() -> str | None:
    if not MARKET_SNAPSHOT_PATH.exists():
        return None
    frame = pd.read_parquet(MARKET_SNAPSHOT_PATH, columns=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return str(dates.max().date())


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# CP212 1D line refresh report",
        "",
        "CP212에서는 F4 beta=4 5-seed ensemble을 1D line serving 기준으로 사용한다.",
        "이번 스크립트는 고정 parquet 복사가 아니라 최근 구간과 최신일을 checkpoint ensemble로 다시 계산해 serving parquet를 갱신한다.",
        "",
        f"- final_status: `{metrics['final_status']}`",
        f"- apply: `{metrics['apply']}`",
        f"- baseline_path: `{metrics['baseline_path']}`",
        f"- target: `{metrics['target']}`",
        f"- model_id: `{LINE_MODEL_ID}`",
        f"- source_cp: `{LINE_SOURCE_CP}`",
        f"- source_note: {LINE_SOURCE_NOTE}",
        f"- cp175_backup: `{metrics.get('cp175_backup')}`",
        f"- market_latest_date: `{metrics.get('market_latest_date')}`",
        "",
        "## Baseline",
        "",
        f"- rows: `{metrics['baseline_summary'].get('rows')}`",
        f"- tickers: `{metrics['baseline_summary'].get('ticker_count')}`",
        f"- asof: `{metrics['baseline_summary'].get('min_date')}` ~ `{metrics['baseline_summary'].get('max_date')}`",
        "",
        "## Refreshed Tail",
        "",
        f"- refresh_start: `{metrics['refresh_diagnostics'].get('refresh_start')}`",
        f"- tail rows: `{metrics['refresh_diagnostics'].get('tail_summary', {}).get('rows')}`",
        f"- tail tickers: `{metrics['refresh_diagnostics'].get('tail_summary', {}).get('ticker_count')}`",
        f"- tail asof: `{metrics['refresh_diagnostics'].get('tail_summary', {}).get('min_date')}` ~ `{metrics['refresh_diagnostics'].get('tail_summary', {}).get('max_date')}`",
        "",
        "## Target",
        "",
        f"- rows: `{metrics['after_summary'].get('rows')}`",
        f"- tickers: `{metrics['after_summary'].get('ticker_count')}`",
        f"- asof: `{metrics['after_summary'].get('min_date')}` ~ `{metrics['after_summary'].get('max_date')}`",
        "",
        "## 금지 작업 확인",
        "",
        "- 새 학습 없음",
        "- 새 calibration 없음",
        "- DB/Supabase write 없음",
        "- 1W line 생성 없음",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_path)
    target_path = Path(args.target_path)
    baseline_frame, baseline_path = load_baseline_frame(source_path, target_path)
    baseline_summary = summarize_frame(baseline_frame, path=baseline_path)
    baseline_max_date = baseline_summary.get("max_date")
    if baseline_max_date is None:
        raise ValueError("기준 1D line parquet에서 최신 asof_date를 찾지 못했습니다.")

    refreshed_tail, refresh_diagnostics = build_refreshed_tail(
        base_max_date=baseline_max_date,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    combined = pd.concat([baseline_frame, refreshed_tail], ignore_index=True)
    combined = normalize_frame(combined)

    cp175_backup = None
    run_backup = None
    if args.apply:
        if target_path.exists() and not CP175_BACKUP_PATH.exists():
            shutil.copy2(target_path, CP175_BACKUP_PATH)
            cp175_backup = str(CP175_BACKUP_PATH)
        elif CP175_BACKUP_PATH.exists():
            cp175_backup = str(CP175_BACKUP_PATH)

        if target_path.exists():
            backup_dir = target_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{target_path.stem}_before_cp212_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
            shutil.copy2(target_path, backup_path)
            run_backup = str(backup_path)

        atomic_write_parquet(combined, target_path)

    after_summary = frame_summary(target_path) if args.apply else summarize_frame(combined, path=target_path)
    market_date = market_latest_date()
    final_status = "PASS_CP212_LINE_REFRESH_APPLIED" if args.apply else "PASS_CP212_LINE_REFRESH_DRY_RUN"
    if market_date and after_summary.get("max_date") != market_date:
        final_status = "WARN_CP212_LINE_REFRESH_DATE_MISMATCH" if args.apply else "WARN_CP212_LINE_REFRESH_DRY_RUN_MISMATCH"

    metrics = {
        "cp": "CP212-LG",
        "created_at": utc_now(),
        "apply": bool(args.apply),
        "final_status": final_status,
        "baseline_path": str(baseline_path),
        "source_artifact": str(source_path),
        "target": str(target_path),
        "line_model_id": LINE_MODEL_ID,
        "line_source_cp": LINE_SOURCE_CP,
        "line_source_note": LINE_SOURCE_NOTE,
        "market_latest_date": market_date,
        "baseline_summary": baseline_summary,
        "refresh_diagnostics": refresh_diagnostics,
        "after_summary": after_summary,
        "cp175_backup": cp175_backup,
        "run_backup": run_backup,
        "forbidden_actions_observed": {
            "new_training": False,
            "new_calibration": False,
            "db_write": False,
            "supabase_write": False,
            "line_1w_generation": False,
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP212 F4 beta=4 ensemble 1D line serving refresh")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-path", default=str(BASE_ARTIFACT_PATH))
    parser.add_argument("--target-path", default=str(TARGET_PATH))
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp212_line_refresh_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp212_line_refresh_report.md"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args)
    print(json.dumps({"status": metrics["final_status"], "asof_max": metrics["after_summary"].get("max_date")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
