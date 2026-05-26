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


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _matches_calibration_target(payload: dict[str, Any], *, run_id: str, timeframe: str) -> bool:
    payload_timeframe = str(payload.get("timeframe") or "").upper()
    if payload_timeframe and payload_timeframe != str(timeframe).upper():
        return False
    ids = {
        str(payload.get("model_run_id") or ""),
        str(payload.get("run_id") or ""),
        str(payload.get("model_id") or ""),
    }
    return run_id in ids


def _calibration_from_payload(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    if payload.get("schema_version") == CALIBRATION_MANIFEST_SCHEMA_VERSION:
        params = payload.get("calibration_params")
        method = payload.get("calibration_method")
    elif "calibration_method" in payload and "calibration_params" in payload:
        params = payload.get("calibration_params")
        method = payload.get("calibration_method")
    elif "method" in payload and "params" in payload:
        params = payload.get("params")
        method = payload.get("method")
    else:
        return None
    if not isinstance(params, dict) or not method:
        return None
    return {
        "status": "calibration_applied",
        "applied": True,
        "method": str(method),
        "params": params,
        "artifact_path": str(source_path),
        "source": "product_band_calibration_artifact",
    }


def resolve_product_band_calibration(
    *,
    checkpoint_config: dict[str, Any],
    run_id: str,
    timeframe: str,
    enabled: bool = True,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "calibration_disabled", "applied": False}

    root = project_root or PROJECT_ROOT
    explicit_candidates = [
        checkpoint_config.get("calibration_artifact_path"),
        checkpoint_config.get("band_calibration_artifact"),
        checkpoint_config.get("calibration_path"),
    ]
    for raw_path in explicit_candidates:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        payload = _safe_json(path)
        if payload:
            artifact = _calibration_from_payload(payload, source_path=path)
            if artifact:
                return artifact

    default_path = default_calibration_manifest_path(role="band", timeframe=timeframe, model_run_id=run_id)
    payload = _safe_json(default_path)
    if payload:
        artifact = _calibration_from_payload(payload, source_path=default_path)
        if artifact:
            return artifact

    docs_dir = root / "docs"
    for pattern in ("*config_lock.json", "*calibration_params.json"):
        for path in sorted(docs_dir.glob(pattern)):
            payload = _safe_json(path)
            if not payload or not _matches_calibration_target(payload, run_id=run_id, timeframe=timeframe):
                continue
            artifact = _calibration_from_payload(payload, source_path=path)
            if artifact:
                return artifact

    return {
        "status": "calibration_missing_raw_band_output_used",
        "applied": False,
        "missing_reason": f"band calibration artifact not found for run_id={run_id}, timeframe={timeframe}",
    }


def _clamp_min(values: Any, minimum: float) -> Any:
    if hasattr(values, "clamp"):
        return values.clamp(min=minimum)
    return values if values >= minimum else minimum


def _elementwise_min(left: Any, right: Any) -> Any:
    if hasattr(left, "minimum"):
        return left.minimum(right)
    return left if left <= right else right


def _elementwise_max(left: Any, right: Any) -> Any:
    if hasattr(left, "maximum"):
        return left.maximum(right)
    return left if left >= right else right


def apply_product_band_calibration(
    *,
    lower_returns: Any,
    upper_returns: Any,
    calibration: dict[str, Any],
) -> tuple[Any, Any]:
    if not calibration.get("applied"):
        return lower_returns, upper_returns

    method = str(calibration.get("method") or "")
    params = calibration.get("params") if isinstance(calibration.get("params"), dict) else {}
    center = (lower_returns + upper_returns) / 2.0
    lower_width = _clamp_min(center - lower_returns, 1e-6)
    upper_width = _clamp_min(upper_returns - center, 1e-6)

    if method in {
        "scalar_width",
        "lower_focused",
        "separate_scale",
        "symmetric_expand",
        "upper_trimmed",
        "lower_breach_guard",
        "center_preserving_width_scale",
    }:
        scale = float(params.get("scale", 1.0))
        if method == "center_preserving_width_scale":
            scale = float(params.get("width_scale", scale))
        lower_scale = float(params.get("lower_scale", scale))
        upper_scale = float(params.get("upper_scale", scale))
        calibrated_lower = center - lower_width * lower_scale
        calibrated_upper = center + upper_width * upper_scale
    elif method == "conformal_residual":
        calibrated_lower = center + float(params["lower_offset"])
        calibrated_upper = center + float(params["upper_offset"])
    elif "global_shift" in params:
        calibrated_lower = lower_returns + float(params["global_shift"])
        calibrated_upper = upper_returns
    else:
        raise ValueError(f"지원하지 않는 band calibration method입니다: {method}")

    return _elementwise_min(calibrated_lower, calibrated_upper), _elementwise_max(calibrated_lower, calibrated_upper)
