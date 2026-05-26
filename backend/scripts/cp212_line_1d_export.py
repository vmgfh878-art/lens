from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "backend" / "data" / "v1"
DOCS_DIR = ROOT / "docs"

SOURCE_PATH = ROOT / "data" / "artifacts" / "cp210" / "exports" / "cp210_F4b4_ensemble_predictions_line_1d.parquet"
TARGET_PATH = V1_DIR / "predictions_line_1d.parquet"
CP175_BACKUP_PATH = V1_DIR / "predictions_line_1d_cp175_frozen_backup.parquet"

LINE_MODEL_ID = "cp210_F4_b4_ensemble_mean"
LINE_SOURCE_CP = "CP208Z_CP209_F4B4"
LINE_SOURCE_NOTE = "CP208Z seed42와 CP209 seed stability 4개 checkpoint를 CP210에서 5-seed mean ensemble로 export"

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
    summary: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }
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


def normalize_source_frame(source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not source_path.exists():
        raise FileNotFoundError(f"line source artifact가 없습니다: {source_path}")
    frame = pd.read_parquet(source_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"line source artifact 필수 컬럼 누락: {missing}")

    original_source_counts = frame["source_cp"].astype(str).value_counts(dropna=False).to_dict()
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["ticker", "asof_date"])
    for column in [
        "line_score",
        "safe_line_score",
        "line_rank_by_date",
        "safe_line_rank_by_date",
        "actual_h5_return",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["line_top_decile_flag"] = frame["line_top_decile_flag"].fillna(False)
    frame["safe_line_top_decile_flag"] = frame["safe_line_top_decile_flag"].fillna(False)
    frame["raw_source_cp"] = frame["source_cp"].astype(str)
    frame["source_cp"] = LINE_SOURCE_CP
    frame["model_id"] = LINE_MODEL_ID
    frame = frame.drop_duplicates(["ticker", "asof_date"], keep="last")
    frame = frame.sort_values(["ticker", "asof_date"]).reset_index(drop=True)

    finite_safe = pd.to_numeric(frame["safe_line_score"], errors="coerce")
    non_finite = int((~finite_safe.notna()).sum())
    diagnostics = {
        "original_source_cp_counts": original_source_counts,
        "duplicate_count_after_normalize": int(frame.duplicated(["ticker", "asof_date"]).sum()),
        "non_finite_safe_line_score_count": non_finite,
        "ticker_count": int(frame["ticker"].nunique()),
        "row_count": int(len(frame)),
        "asof_min": str(pd.to_datetime(frame["asof_date"]).min().date()) if len(frame) else None,
        "asof_max": str(pd.to_datetime(frame["asof_date"]).max().date()) if len(frame) else None,
    }
    if non_finite:
        raise ValueError(f"safe_line_score non-finite row가 있습니다: {non_finite}")
    return frame, diagnostics


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp_path, index=False, compression="snappy")
    temp_path.replace(path)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# CP212 1D line export report",
        "",
        "CP212에서는 F4 beta=4 5-seed ensemble을 1D line serving 후보로 고정했다.",
        "",
        f"- final_status: `{metrics['final_status']}`",
        f"- apply: `{metrics['apply']}`",
        f"- source_artifact: `{metrics['source_artifact']}`",
        f"- target: `{metrics['target']}`",
        f"- model_id: `{LINE_MODEL_ID}`",
        f"- source_cp: `{LINE_SOURCE_CP}`",
        f"- source_note: {LINE_SOURCE_NOTE}",
        f"- backup_cp175: `{metrics.get('cp175_backup')}`",
        "",
        "## Source",
        "",
        f"- rows: `{metrics['source_summary'].get('rows')}`",
        f"- tickers: `{metrics['source_summary'].get('ticker_count')}`",
        f"- asof: `{metrics['source_summary'].get('min_date')}` ~ `{metrics['source_summary'].get('max_date')}`",
        "",
        "## Target",
        "",
        f"- rows: `{metrics['after_summary'].get('rows')}`",
        f"- tickers: `{metrics['after_summary'].get('ticker_count')}`",
        f"- asof: `{metrics['after_summary'].get('min_date')}` ~ `{metrics['after_summary'].get('max_date')}`",
        "",
        "## 금지 작업 확인",
        "",
        "- 새 학습: 실행 안 함",
        "- 새 calibration: 실행 안 함",
        "- inference 재실행: 실행 안 함",
        "- DB/Supabase write: 실행 안 함",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_path)
    target_path = Path(args.target_path)
    before_summary = frame_summary(target_path)
    source_summary = frame_summary(source_path)
    frame, diagnostics = normalize_source_frame(source_path)

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
            run_backup_path = backup_dir / f"{target_path.stem}_before_cp212_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
            shutil.copy2(target_path, run_backup_path)
            run_backup = str(run_backup_path)
        atomic_write_parquet(frame, target_path)

    after_summary = frame_summary(target_path) if args.apply else {
        **diagnostics,
        "path": str(target_path),
        "model_ids": [LINE_MODEL_ID],
        "source_cps": [LINE_SOURCE_CP],
    }
    metrics = {
        "cp": "CP212-LG",
        "created_at": utc_now(),
        "apply": bool(args.apply),
        "final_status": "PASS_CP212_LINE_EXPORT_APPLIED" if args.apply else "PASS_CP212_LINE_EXPORT_DRY_RUN",
        "source_artifact": str(source_path),
        "target": str(target_path),
        "line_model_id": LINE_MODEL_ID,
        "line_source_cp": LINE_SOURCE_CP,
        "line_source_note": LINE_SOURCE_NOTE,
        "before_summary": before_summary,
        "source_summary": source_summary,
        "diagnostics": diagnostics,
        "after_summary": after_summary,
        "cp175_backup": cp175_backup,
        "run_backup": run_backup,
        "forbidden_actions_observed": {
            "new_training": False,
            "new_calibration": False,
            "inference_execution": False,
            "db_write": False,
            "supabase_write": False,
        },
    }
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP212 F4 beta=4 ensemble 1D line serving export")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-path", default=str(SOURCE_PATH))
    parser.add_argument("--target-path", default=str(TARGET_PATH))
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp212_line_export_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp212_line_export_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args)
    print(json.dumps({"status": metrics["final_status"], "asof_max": metrics["after_summary"].get("max_date")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
