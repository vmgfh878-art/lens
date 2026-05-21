from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "docs" / "calibration_artifacts"
CALIBRATION_MANIFEST_SCHEMA_VERSION = "calibration_artifact_manifest_v1"


def calibration_model_run_id(config: dict[str, Any], checkpoint_path: Path | None = None) -> str:
    run_id = config.get("run_id")
    if run_id:
        return str(run_id)
    if checkpoint_path is not None:
        return f"checkpoint:{checkpoint_path.stem}"
    return "unknown"


def default_calibration_manifest_path(*, role: str, timeframe: str, model_run_id: str) -> Path:
    safe_run_id = str(model_run_id).replace(":", "_").replace("/", "_").replace("\\", "_")
    return DEFAULT_CALIBRATION_DIR / f"{role}_{str(timeframe).upper()}_{safe_run_id}.manifest.json"


def build_calibration_artifact_manifest(
    *,
    role: str,
    timeframe: str,
    model_run_id: str,
    calibration_method: str,
    calibration_params: dict[str, Any],
    source_metrics_path: str,
    checkpoint_path: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "role": role,
        "timeframe": str(timeframe).upper(),
        "model_run_id": model_run_id,
        "calibration_method": calibration_method,
        "calibration_params": calibration_params,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_metrics_path": source_metrics_path,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics or {},
    }


def write_calibration_artifact_manifest(manifest: dict[str, Any], path: Path | None = None) -> Path:
    output_path = path or default_calibration_manifest_path(
        role=str(manifest["role"]),
        timeframe=str(manifest["timeframe"]),
        model_run_id=str(manifest["model_run_id"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_calibration_artifact_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"지원하지 않는 calibration manifest schema입니다: {manifest.get('schema_version')}")
    params = manifest.get("calibration_params")
    if not isinstance(params, dict):
        raise ValueError("calibration manifest에 calibration_params가 없습니다.")
    return manifest


def resolve_default_calibration_artifact_manifest(
    *,
    role: str,
    timeframe: str,
    model_run_id: str,
) -> dict[str, Any]:
    path = default_calibration_manifest_path(role=role, timeframe=timeframe, model_run_id=model_run_id)
    if not path.exists():
        raise FileNotFoundError(f"calibration artifact not found: {path}")
    return load_calibration_artifact_manifest(path)
