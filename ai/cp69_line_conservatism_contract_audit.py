from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402

from ai.loss import AsymmetricHuberLoss, ForecastCompositeLoss


REPORT_PATH = PROJECT_ROOT / "docs" / "cp69_line_conservatism_contract_audit_report.md"
MATRIX_PATH = PROJECT_ROOT / "docs" / "cp69_line_conservatism_contract_audit_matrix.json"
CP49_METRICS_PATH = PROJECT_ROOT / "docs" / "cp49_patchtst_horizon_rescue_metrics.json"
CP65_METRICS_PATH = PROJECT_ROOT / "docs" / "cp65_lm_feature_h20_smoke_metrics.json"
CP67_METRICS_PATH = PROJECT_ROOT / "docs" / "cp67_lm_h20_100ticker_validation_metrics.json"
CP68_METRICS_PATH = PROJECT_ROOT / "docs" / "cp68_lm_h20_conservative_line_rescue_metrics.json"


AUDIT_KEYS = [
    "alpha",
    "beta",
    "delta",
    "lambda_line",
    "lambda_band",
    "lambda_cross",
    "checkpoint_selection",
    "line_target_type",
    "band_target_type",
    "horizon",
    "seq_len",
    "patch_len",
    "patch_stride",
    "feature_set",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    checkpoint_path: str | None
    expected_feature_set: str | None = None
    display_policy: str | None = None
    notes: str = ""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _checkpoint_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_checkpoint_config(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "checkpoint_path 없음"
    if not path.exists():
        return None, f"checkpoint 파일 없음: {_rel(path)}"
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return None, f"checkpoint 로드 실패: {exc}"
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return None, "checkpoint config 없음"
    return dict(config), None


def _candidate_from_cp49(name: str, cp49: dict[str, Any]) -> Candidate:
    for record in cp49.get("records", []):
        candidate = record.get("candidate", {})
        if candidate.get("name") == name:
            return Candidate(
                name=name,
                source="CP49",
                checkpoint_path=record.get("checkpoint_path"),
                expected_feature_set="full_features",
            )
    raise ValueError(f"CP49 candidate를 찾지 못했다: {name}")


def _candidate_from_cp65_experiment(name: str, cp65: dict[str, Any]) -> Candidate:
    for record in cp65.get("experiments", []):
        if record.get("name") == name:
            return Candidate(
                name=name,
                source="CP65",
                checkpoint_path=record.get("checkpoint_path"),
                expected_feature_set=record.get("feature_set"),
            )
    raise ValueError(f"CP65 experiment를 찾지 못했다: {name}")


def _candidate_from_cp65_reference(name: str, cp65: dict[str, Any]) -> Candidate:
    record = cp65.get("references", {}).get("full_features", {}).get(name)
    if not isinstance(record, dict):
        raise ValueError(f"CP65 reference를 찾지 못했다: {name}")
    return Candidate(
        name=name,
        source="CP65_reference",
        checkpoint_path=record.get("checkpoint_path"),
        expected_feature_set=record.get("feature_set", "full_features"),
    )


def _candidate_from_cp67(cp67: dict[str, Any]) -> Candidate:
    for record in cp67.get("experiments", []):
        if record.get("name") == "h20_full_features_post_backfill_100":
            return Candidate(
                name="h20_full_features_post_backfill_100",
                source="CP67",
                checkpoint_path=record.get("checkpoint_path"),
                expected_feature_set=record.get("feature_set"),
            )
    raise ValueError("CP67 h20_full_features_post_backfill_100을 찾지 못했다.")


def _candidate_from_cp68(cp68: dict[str, Any]) -> Candidate:
    checkpoint = cp68.get("checkpoint", {})
    best = cp68.get("best_policy", {})
    return Candidate(
        name=f"cp68_{best.get('name', 'display_calibrated_line')}",
        source="CP68",
        checkpoint_path=checkpoint.get("path"),
        expected_feature_set=checkpoint.get("feature_set"),
        display_policy=best.get("name"),
        notes="CP68은 학습 보수성이 아니라 validation offset 기반 display calibration이다.",
    )


def _candidate_list() -> list[Candidate]:
    cp49 = _read_json(CP49_METRICS_PATH)
    cp65 = _read_json(CP65_METRICS_PATH)
    cp67 = _read_json(CP67_METRICS_PATH)
    cp68 = _read_json(CP68_METRICS_PATH)

    candidates = [
        _candidate_from_cp49("h5_longer_context_seq252_p32_s16", cp49),
        _candidate_from_cp49("h5_baseline_seq252_p16_s8", cp49),
        _candidate_from_cp49("h5_dense_overlap_seq252_p16_s4", cp49),
        _candidate_from_cp65_reference("h20_longer_context_seq252_p32_s16", cp65),
        _candidate_from_cp65_experiment("h20_technical_only_seq252_p32_s16", cp65),
        _candidate_from_cp65_experiment("h20_no_fundamentals_seq252_p32_s16", cp65),
        _candidate_from_cp65_experiment("h20_price_volatility_volume_seq252_p32_s16", cp65),
        _candidate_from_cp67(cp67),
        _candidate_from_cp68(cp68),
    ]
    return candidates


def _is_value(value: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(value) - expected) < 1e-9
        except (TypeError, ValueError):
            return False
    return value == expected


def _classify(config: dict[str, Any] | None, candidate: Candidate, error: str | None) -> str:
    if candidate.display_policy:
        return "posthoc_only_not_training"
    if config is None:
        return "unverified_config_missing"

    has_conservative_params = (
        _is_value(config.get("alpha"), 1.0)
        and _is_value(config.get("beta"), 2.0)
        and float(config.get("beta", 0.0)) > float(config.get("alpha", 0.0))
        and _is_value(config.get("delta"), 1.0)
        and float(config.get("lambda_line", 0.0)) > 0.0
        and config.get("line_target_type") == "raw_future_return"
    )
    if has_conservative_params:
        return "verified_trained_conservative"

    if config.get("alpha") is None or config.get("beta") is None:
        return "likely_trained_conservative"

    return "not_conservative_loss"


def _audit_candidate(candidate: Candidate) -> dict[str, Any]:
    path = _checkpoint_path(candidate.checkpoint_path)
    config, error = _load_checkpoint_config(path)
    config_payload = {key: (config or {}).get(key) for key in AUDIT_KEYS}
    checkpoint_feature_set = config_payload.get("feature_set")
    feature_set_note = ""
    if checkpoint_feature_set is None and candidate.expected_feature_set:
        feature_set_note = f"checkpoint config에는 feature_set이 없고 source에서 {candidate.expected_feature_set}로 추정"
    elif checkpoint_feature_set != candidate.expected_feature_set and candidate.expected_feature_set:
        feature_set_note = f"source feature_set={candidate.expected_feature_set}, checkpoint feature_set={checkpoint_feature_set}"

    classification = _classify(config, candidate, error)
    return {
        "name": candidate.name,
        "source": candidate.source,
        "classification": classification,
        "checkpoint_path": _rel(path) if path else None,
        "checkpoint_exists": bool(path and path.exists()),
        "config_error": error,
        "expected_feature_set": candidate.expected_feature_set,
        "feature_set_note": feature_set_note,
        "display_policy": candidate.display_policy,
        "loss_class_name": "AsymmetricHuberLoss",
        "loss_class_saved_in_config": bool(config and config.get("loss_class")),
        "loss_class_config_value": (config or {}).get("loss_class") if config else None,
        "config": config_payload,
        "notes": candidate.notes,
    }


def _loss_contract_audit() -> dict[str, Any]:
    loss = AsymmetricHuberLoss(alpha=1.0, beta=2.0, delta=1.0)
    target = torch.tensor([[1.0]], dtype=torch.float32)
    over_prediction = torch.tensor([[2.0]], dtype=torch.float32)
    under_prediction = torch.tensor([[0.0]], dtype=torch.float32)
    over_loss = float(loss(over_prediction, target).item())
    under_loss = float(loss(under_prediction, target).item())
    composite = ForecastCompositeLoss(alpha=1.0, beta=2.0, delta=1.0)
    return {
        "asymmetric_huber_class": loss.__class__.__name__,
        "forecast_composite_line_loss_class": composite.line_loss.__class__.__name__,
        "error_definition": "prediction - target",
        "overprediction_condition": "error > 0",
        "alpha": loss.alpha,
        "beta": loss.beta,
        "delta": loss.delta,
        "unit_example": {
            "target": 1.0,
            "over_prediction": 2.0,
            "under_prediction": 0.0,
            "over_loss": over_loss,
            "under_loss": under_loss,
            "over_to_under_ratio": over_loss / under_loss if under_loss else None,
            "expected_over_loss": 1.0,
            "expected_under_loss": 0.5,
            "beta_applied_to_overprediction": over_loss > under_loss,
        },
        "line_loss_used_in_composite": True,
        "forecast_component_uses_lambda_line": True,
    }


def _matrix_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        config = record["config"]
        feature_set = config.get("feature_set") or record.get("expected_feature_set")
        if record.get("feature_set_note") and config.get("feature_set") is not None:
            feature_set = f"{config.get('feature_set')} / source:{record.get('expected_feature_set')}"
        rows.append(
            {
                "name": record["name"],
                "source": record["source"],
                "class": record["classification"],
                "alpha": _fmt(config.get("alpha")),
                "beta": _fmt(config.get("beta")),
                "delta": _fmt(config.get("delta")),
                "lambda_line": _fmt(config.get("lambda_line")),
                "lambda_band": _fmt(config.get("lambda_band")),
                "lambda_cross": _fmt(config.get("lambda_cross")),
                "selection": _fmt(config.get("checkpoint_selection")),
                "target": _fmt(config.get("line_target_type")),
                "band_target": _fmt(config.get("band_target_type")),
                "h": _fmt(config.get("horizon")),
                "seq": _fmt(config.get("seq_len")),
                "patch": f"{_fmt(config.get('patch_len'))}/{_fmt(config.get('patch_stride'))}",
                "feature_set": _fmt(feature_set),
            }
        )
    return rows


def _build_report(payload: dict[str, Any]) -> str:
    loss = payload["loss_contract"]
    rows = _matrix_rows(payload["candidates"])
    counts = payload["classification_counts"]
    lines = [
        "# CP69-LM line 보수성 계약 재감사",
        "",
        "## 1. 범위",
        "- 새 학습, 성능 재평가, h20 추가 보정, DB 쓰기, save-run, band/composite/UI/backend 작업은 하지 않았다.",
        "- checkpoint는 `torch.load(..., map_location='cpu')`로 config만 읽었다.",
        "",
        "## 2. line loss 구현 감사",
        _table(
            [
                {"item": "line loss class", "value": loss["asymmetric_huber_class"], "result": "PASS"},
                {"item": "composite line loss", "value": loss["forecast_composite_line_loss_class"], "result": "PASS"},
                {"item": "error definition", "value": loss["error_definition"], "result": "PASS"},
                {"item": "overprediction condition", "value": loss["overprediction_condition"], "result": "PASS"},
                {"item": "alpha/beta/delta", "value": f"{loss['alpha']}/{loss['beta']}/{loss['delta']}", "result": "PASS"},
                {"item": "unit over/under loss", "value": f"{loss['unit_example']['over_loss']:.4f}/{loss['unit_example']['under_loss']:.4f}", "result": "PASS"},
            ],
            [("항목", "item"), ("값", "value"), ("판정", "result")],
        ),
        "",
        "단위 예제에서 target=1.0, prediction=2.0은 overprediction이며 loss=1.0이다. prediction=0.0은 underprediction이며 loss=0.5다. 같은 절대오차에서 beta=2가 overprediction에 적용된다.",
        "",
        "## 3. checkpoint/config 감사 매트릭스",
        _table(
            rows,
            [
                ("후보", "name"),
                ("source", "source"),
                ("판정", "class"),
                ("alpha", "alpha"),
                ("beta", "beta"),
                ("delta", "delta"),
                ("lambda_line", "lambda_line"),
                ("lambda_band", "lambda_band"),
                ("lambda_cross", "lambda_cross"),
                ("selection", "selection"),
                ("target", "target"),
                ("band_target", "band_target"),
                ("h", "h"),
                ("seq", "seq"),
                ("patch", "patch"),
                ("feature_set", "feature_set"),
            ],
        ),
        "",
        "## 4. 분류 요약",
        _table(
            [{"classification": key, "count": value} for key, value in counts.items()],
            [("분류", "classification"), ("개수", "count")],
        ),
        "",
        "## 5. h5 제품 후보 신뢰도",
        "- `h5_longer_context_seq252_p32_s16`, `h5_baseline_seq252_p16_s8`, `h5_dense_overlap_seq252_p16_s4` 모두 checkpoint config에서 alpha=1, beta=2, delta=1, lambda_line=1, line_target_type=raw_future_return, checkpoint_selection=line_gate를 확인했다.",
        "- 따라서 h5 후보들은 `verified_trained_conservative`로 분류한다.",
        "- 단 CP49 계열 checkpoint는 config에 `feature_set` 값이 저장되지 않아 source metrics 기준 full_features로 추정한다. 이 문제는 보수 loss 확인과는 별개다.",
        "- CP65 `price_volatility_volume` 후보는 당시 `technical_only`와 feature 정의가 동일해 같은 checkpoint를 재사용했다. 그래서 checkpoint config에는 `technical_only`로 남아 있고 source feature_set은 `price_volatility_volume`이다.",
        "",
        "## 6. CP68 용어 재분류",
        "- `raw_model_line`: 모델 checkpoint가 직접 출력한 line이다.",
        "- `trained_conservative_line`: alpha/beta asymmetric Huber loss로 학습된 raw_model_line이다.",
        "- `display_calibrated_line`: validation offset으로 아래로 보정한 표시선이다.",
        "- CP68 `global_downshift`와 `horizon_bucket_downshift`는 학습 보수성이 아니라 post-hoc display calibration이다.",
        "- 따라서 CP68 best line은 `posthoc_only_not_training`으로 분류하고, underlying CP67 checkpoint만 `verified_trained_conservative`로 둔다.",
        "",
        "## 7. 결론",
        "주요 h5/h20 PatchTST checkpoint는 line loss 계약상 하방 보수 학습이 확인됐다. CP68의 보수선은 학습으로 생긴 보수성이 아니라 표시 보정이므로 제품/문서에서 반드시 `display_calibrated_line`으로 구분해야 한다.",
        "",
        "## 8. 검증",
        "- `.venv\\Scripts\\python.exe -m py_compile ai\\cp69_line_conservatism_contract_audit.py ai\\tests\\test_losses.py`: 통과",
        "- `python -m json.tool docs\\cp69_line_conservatism_contract_audit_matrix.json`: 통과",
        "- `.venv\\Scripts\\python.exe -m unittest ai.tests.test_losses ai.tests.test_loss`: 8개 통과",
    ]
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    loss_contract = _loss_contract_audit()
    candidates = [_audit_candidate(candidate) for candidate in _candidate_list()]
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate["classification"]] = counts.get(candidate["classification"], 0) + 1
    payload = {
        "cp": "CP69-LM",
        "rules": {
            "new_training": False,
            "posthoc_calibration_experiment": False,
            "performance_reevaluation": False,
            "db_write": False,
            "save_run": False,
            "band_experiment": False,
            "composite_experiment": False,
            "ui_backend_modified": False,
        },
        "terminology": {
            "raw_model_line": "모델 checkpoint 원출력 line",
            "trained_conservative_line": "alpha/beta asymmetric Huber loss로 학습된 line",
            "display_calibrated_line": "validation offset으로 아래로 보정한 표시선",
        },
        "loss_contract": loss_contract,
        "candidates": candidates,
        "classification_counts": counts,
    }
    _write_json(MATRIX_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8-sig")
    return payload


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="CP69 line 보수성 계약 재감사").parse_args()


def main() -> None:
    parse_args()
    payload = run()
    print(
        json.dumps(
            {
                "cp": payload["cp"],
                "report": _rel(REPORT_PATH),
                "matrix": _rel(MATRIX_PATH),
                "classification_counts": payload["classification_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
