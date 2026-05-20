from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()

from ai.models.common import (  # noqa: E402
    BandOutput,
    ForecastOutput,
    LineDistributionalOutput,
    LineMonotonicDistributionalOutput,
    LineRegimeOutput,
    LineV2Output,
    LineWarningOutput,
    MODEL_OUTPUT_ROLES,
)
from ai.preprocessing import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    SequenceDataset,
    SequenceDatasetBundle,
    resolved_market_data_provider,
)


PROVIDER_KEYS = (
    "market_data_provider",
    "train_market_data_provider",
    "data_provider",
    "provider",
)


@dataclass(frozen=True)
class FeatureSubsetResult:
    bundle: SequenceDatasetBundle | SequenceDataset
    report: dict[str, Any]


MODEL_ROLE_ALIASES = {
    "legacy": "legacy",
    "forecast": "legacy",
    "multihead": "legacy",
    "line": "line_v2",
    "line_v2": "line_v2",
    "line_model": "line_v2",
    "conservative_line": "line_v2",
    "line_regime": "line_regime",
    "regime": "line_regime",
    "line_warning": "line_warning",
    "warning": "line_warning",
    "line_distributional": "line_distributional",
    "distributional": "line_distributional",
    "line_v3": "line_distributional",
    "line_distributional_mono": "line_distributional_mono",
    "distributional_mono": "line_distributional_mono",
    "line_v3_mono": "line_distributional_mono",
    "band": "band",
    "band_model": "band",
}
ROLE_OUTPUT_TYPES = {
    "legacy": ForecastOutput,
    "line_v2": LineV2Output,
    "line_regime": LineRegimeOutput,
    "line_warning": LineWarningOutput,
    "line_distributional": LineDistributionalOutput,
    "line_distributional_mono": LineMonotonicDistributionalOutput,
    "band": BandOutput,
}
ROLE_ALLOWED_FIELDS = {
    "legacy": ("line", "lower_band", "upper_band", "direction_logit"),
    "line_v2": ("line", "downside_risk_logit", "downside_risk_prob"),
    "line_regime": ("line", "regime_logits", "regime_prob"),
    "line_warning": ("warning_logit", "warning_prob"),
    "line_distributional": ("quantiles",),
    "line_distributional_mono": ("line", "quantiles"),
    "band": ("lower_band", "upper_band", "bands"),
}
INFERENCE_PAYLOAD_SUPPORTED_ROLES = ("legacy", "line_v2", "band")
COMPOSITE_LINE_SUPPORTED_ROLES = ("legacy", "line_v2")
COMPOSITE_BAND_SUPPORTED_ROLES = ("legacy", "band")
BAND_CALIBRATION_SUPPORTED_ROLES = ("legacy", "band")
AMBIGUOUS_SELECTOR_ROLES = {"line_model", "band_model", "combined_model", "composite_model"}


def normalize_model_role(role: str | None, *, context: str = "model_role") -> str:
    raw_role = str(role or "legacy").strip().lower()
    resolved = MODEL_ROLE_ALIASES.get(raw_role)
    if resolved is None or resolved not in MODEL_OUTPUT_ROLES:
        raise ValueError(f"{context}: 지원하지 않는 model_role입니다: {raw_role}")
    return resolved


def resolve_model_role_from_config(config: dict[str, Any], *, context: str = "checkpoint") -> str:
    for key in ("model_role", "output_role"):
        value = config.get(key)
        if value is not None and str(value).strip():
            return normalize_model_role(str(value), context=f"{context}.{key}")

    role_value = config.get("role")
    if role_value is not None and str(role_value).strip():
        raw_role = str(role_value).strip().lower()
        if raw_role in AMBIGUOUS_SELECTOR_ROLES:
            raise ValueError(
                f"{context}.role={raw_role}은 checkpoint selector/storage label로도 쓰이므로 "
                "model_role 또는 output_role이 없으면 추론 역할을 확정할 수 없습니다."
            )
        return normalize_model_role(raw_role, context=f"{context}.role")

    return "legacy"


def assert_output_matches_model_role(output: object, model_role: str, *, context: str = "inference") -> None:
    resolved_role = normalize_model_role(model_role, context=context)
    expected_type = ROLE_OUTPUT_TYPES[resolved_role]
    if not isinstance(output, expected_type):
        raise TypeError(
            f"{context}: model_role={resolved_role}은 {expected_type.__name__}만 허용합니다. "
            f"actual={type(output).__name__}"
        )


def require_role_supported(model_role: str, allowed_roles: tuple[str, ...], *, context: str) -> str:
    resolved_role = normalize_model_role(model_role, context=context)
    if resolved_role not in allowed_roles:
        raise ValueError(
            f"{context}: model_role={resolved_role}은 현재 경로에서 지원하지 않습니다. "
            f"supported_roles={list(allowed_roles)}"
        )
    return resolved_role


def role_contract_table() -> dict[str, dict[str, Any]]:
    return {
        role: {
            "output_type": ROLE_OUTPUT_TYPES[role].__name__,
            "allowed_fields": list(ROLE_ALLOWED_FIELDS[role]),
            "inference_payload_supported": role in INFERENCE_PAYLOAD_SUPPORTED_ROLES,
            "composite_line_supported": role in COMPOSITE_LINE_SUPPORTED_ROLES,
            "composite_band_supported": role in COMPOSITE_BAND_SUPPORTED_ROLES,
            "band_calibration_supported": role in BAND_CALIBRATION_SUPPORTED_ROLES,
        }
        for role in ROLE_OUTPUT_TYPES
    }


def _iter_contract_dicts(source: dict[str, Any] | None):
    if not isinstance(source, dict):
        return
    yield source
    for key in ("config", "run_config", "artifact_meta", "data_contract", "feature_contract", "dataset_contract", "meta"):
        nested = source.get(key)
        if isinstance(nested, dict):
            yield from _iter_contract_dicts(nested)


def _provider_values(*sources: dict[str, Any] | None) -> list[str]:
    values: list[str] = []
    for source in sources:
        for contract in _iter_contract_dicts(source):
            for key in PROVIDER_KEYS:
                value = contract.get(key)
                if value is not None and str(value).strip():
                    values.append(resolved_market_data_provider(str(value)))
    return values


def resolve_checkpoint_market_data_provider(
    checkpoint_config: dict[str, Any],
    *,
    run_config: dict[str, Any] | None = None,
    artifact_meta: dict[str, Any] | None = None,
) -> str:
    values = _provider_values(checkpoint_config, run_config, artifact_meta)
    unique = sorted(set(values))
    if not unique:
        raise ValueError("checkpoint/run_config에 market_data_provider 계약이 없어 저장형 실행을 중단합니다.")
    if len(unique) > 1:
        raise ValueError(f"provider 계약 불일치: {unique}")
    return unique[0]


def resolve_execution_market_data_provider(
    checkpoint_config: dict[str, Any],
    *,
    requested_provider: str | None = None,
    run_config: dict[str, Any] | None = None,
    artifact_meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    train_provider = resolve_checkpoint_market_data_provider(
        checkpoint_config,
        run_config=run_config,
        artifact_meta=artifact_meta,
    )
    inference_provider = resolved_market_data_provider(requested_provider or train_provider)
    report = {
        "train_market_data_provider": train_provider,
        "inference_market_data_provider": inference_provider,
        "provider_match": train_provider == inference_provider,
    }
    if not report["provider_match"]:
        raise ValueError(
            "provider mismatch로 저장형 실행을 중단합니다: "
            f"train={train_provider}, inference={inference_provider}"
        )
    return inference_provider, report


def validate_checkpoint_runtime_contract(
    checkpoint_config: dict[str, Any],
    *,
    runtime_config: dict[str, Any] | None = None,
    runtime_timeframe: str | None = None,
    runtime_horizon: int | None = None,
    run_status: str | None = None,
) -> dict[str, Any]:
    if run_status is not None and str(run_status) != "completed":
        raise ValueError(f"completed run만 실행할 수 있습니다: status={run_status}")

    report: dict[str, Any] = {
        "checkpoint_status": "available",
        "run_status": run_status,
        "timeframe_match": True,
        "horizon_match": True,
        "target_type_match": True,
    }
    runtime_config = runtime_config or {}
    checkpoint_timeframe = checkpoint_config.get("timeframe")
    expected_timeframe = runtime_timeframe or runtime_config.get("timeframe")
    if checkpoint_timeframe is not None and expected_timeframe is not None:
        report["checkpoint_timeframe"] = str(checkpoint_timeframe)
        report["runtime_timeframe"] = str(expected_timeframe)
        report["timeframe_match"] = str(checkpoint_timeframe).upper() == str(expected_timeframe).upper()

    checkpoint_horizon = checkpoint_config.get("horizon")
    expected_horizon = runtime_horizon if runtime_horizon is not None else runtime_config.get("horizon")
    if checkpoint_horizon is not None and expected_horizon is not None:
        report["checkpoint_horizon"] = int(checkpoint_horizon)
        report["runtime_horizon"] = int(expected_horizon)
        report["horizon_match"] = int(checkpoint_horizon) == int(expected_horizon)

    target_keys = ("line_target_type", "band_target_type")
    target_mismatches = {
        key: (checkpoint_config.get(key), runtime_config.get(key))
        for key in target_keys
        if checkpoint_config.get(key) is not None
        and runtime_config.get(key) is not None
        and str(checkpoint_config.get(key)) != str(runtime_config.get(key))
    }
    report["target_type_match"] = not target_mismatches
    report["target_type_mismatches"] = target_mismatches
    if not report["timeframe_match"] or not report["horizon_match"] or target_mismatches:
        raise ValueError(f"checkpoint/runtime 계약 불일치: {report}")
    return report


def resolve_checkpoint_feature_columns(checkpoint_config: dict[str, Any]) -> list[str]:
    raw_columns = checkpoint_config.get("feature_columns")
    n_features = checkpoint_config.get("n_features")
    if raw_columns is None:
        if n_features is not None and int(n_features) != len(MODEL_FEATURE_COLUMNS):
            raise ValueError(
                "checkpoint feature_columns가 없는데 n_features가 전체 feature contract와 다릅니다: "
                f"n_features={n_features}, full={len(MODEL_FEATURE_COLUMNS)}"
            )
        return list(MODEL_FEATURE_COLUMNS)

    if not isinstance(raw_columns, list) or not all(isinstance(column, str) for column in raw_columns):
        raise ValueError("checkpoint feature_columns는 문자열 리스트여야 합니다.")
    columns = list(raw_columns)
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"checkpoint feature_columns에 중복이 있습니다: {duplicates}")
    missing = [column for column in columns if column not in MODEL_FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"missing_features={missing}")
    if n_features is not None and int(n_features) != len(columns):
        raise ValueError(f"checkpoint n_features와 feature_columns 수가 다릅니다: {n_features} != {len(columns)}")
    return columns


def _bundle_feature_count(bundle: SequenceDatasetBundle | SequenceDataset) -> int:
    if isinstance(bundle, SequenceDatasetBundle):
        return int(bundle.features.shape[2])
    if bundle.mean is not None:
        return int(bundle.mean.numel())
    if not bundle.ticker_arrays:
        return 0
    first = next(iter(bundle.ticker_arrays.values()))
    return int(first["features"].shape[1])


def select_bundle_features_for_columns(
    bundle: SequenceDatasetBundle | SequenceDataset,
    columns: list[str],
) -> FeatureSubsetResult:
    before_count = _bundle_feature_count(bundle)
    missing = [column for column in columns if column not in MODEL_FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"missing_features={missing}")

    extra_features_ignored = [column for column in MODEL_FEATURE_COLUMNS if column not in columns]
    report = {
        "checkpoint_feature_count": len(columns),
        "bundle_feature_count_before": before_count,
        "bundle_feature_count_after": len(columns),
        "missing_features": [],
        "extra_features_ignored": extra_features_ignored,
        "feature_order_matches_checkpoint": columns == MODEL_FEATURE_COLUMNS[: len(columns)],
    }

    if before_count == len(columns):
        if columns == MODEL_FEATURE_COLUMNS[: len(columns)]:
            report["bundle_feature_count_after"] = before_count
            return FeatureSubsetResult(bundle=bundle, report=report)
        if before_count != len(MODEL_FEATURE_COLUMNS):
            raise ValueError(
                "feature_columns order mismatch: 이미 subset된 bundle은 원본 feature 순서를 알 수 없어 "
                "checkpoint 순서로 안전하게 재정렬할 수 없습니다."
            )
    if before_count != len(MODEL_FEATURE_COLUMNS):
        raise ValueError(
            "bundle feature 수가 checkpoint 또는 전체 feature contract와 맞지 않습니다: "
            f"before={before_count}, checkpoint={len(columns)}, full={len(MODEL_FEATURE_COLUMNS)}"
        )

    indices = [MODEL_FEATURE_COLUMNS.index(column) for column in columns]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    if isinstance(bundle, SequenceDatasetBundle):
        selected_bundle = SequenceDatasetBundle(
            features=bundle.features.index_select(2, index_tensor).contiguous(),
            line_targets=bundle.line_targets,
            band_targets=bundle.band_targets,
            raw_future_returns=bundle.raw_future_returns,
            anchor_closes=bundle.anchor_closes,
            ticker_ids=bundle.ticker_ids,
            future_covariates=bundle.future_covariates,
            metadata=bundle.metadata.copy(),
        )
        return FeatureSubsetResult(bundle=selected_bundle, report=report)

    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        copied = dict(arrays)
        copied["features"] = arrays["features"][:, indices].copy()
        ticker_arrays[ticker] = copied
    selected_bundle = SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=list(bundle.sample_refs),
        metadata=bundle.metadata.copy(),
        seq_len=bundle.seq_len,
        horizon=bundle.horizon,
        mean=bundle.mean.index_select(0, index_tensor) if bundle.mean is not None else None,
        std=bundle.std.index_select(0, index_tensor) if bundle.std is not None else None,
        include_future_covariate=bundle.include_future_covariate,
        line_target_type=bundle.line_target_type,
        band_target_type=bundle.band_target_type,
    )
    return FeatureSubsetResult(bundle=selected_bundle, report=report)


def select_bundle_features_for_checkpoint(
    bundle: SequenceDatasetBundle | SequenceDataset,
    checkpoint_config: dict[str, Any],
) -> FeatureSubsetResult:
    columns = resolve_checkpoint_feature_columns(checkpoint_config)
    return select_bundle_features_for_columns(bundle, columns)
