from __future__ import annotations

from typing import Any

from ai.storage import STORAGE_CONTRACT_EVALUATION_BULK, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY


PRODUCT_RUN_ROLES = {"line_model", "band_model", "product_line", "product_band"}


def is_product_model_run(model_run: dict[str, Any], config: dict[str, Any]) -> bool:
    role = str(config.get("role") or config.get("model_role") or "").lower()
    storage_contract = str(config.get("storage_contract") or "").lower()
    product_flags = (
        config.get("product_candidate"),
        config.get("final_product_candidate"),
        model_run.get("product_candidate"),
    )
    if storage_contract == STORAGE_CONTRACT_PRODUCT_LATEST_ONLY:
        return True
    if role in PRODUCT_RUN_ROLES:
        return True
    return any(bool(flag) for flag in product_flags)


def resolve_inference_save_contract(
    *,
    save: bool,
    save_product_latest_only: bool,
    allow_bulk_evaluation_save: bool,
    model_run: dict[str, Any],
    config: dict[str, Any],
) -> str | None:
    if not save:
        return None
    if save_product_latest_only and allow_bulk_evaluation_save:
        raise ValueError("--save-product-latest-only와 --allow-bulk-evaluation-save는 동시에 사용할 수 없습니다.")
    if save_product_latest_only:
        return STORAGE_CONTRACT_PRODUCT_LATEST_ONLY
    if is_product_model_run(model_run, config):
        raise ValueError("제품 run은 bulk 저장을 허용하지 않습니다. --save-product-latest-only를 사용해 주세요.")
    if not allow_bulk_evaluation_save:
        raise ValueError(
            "--save는 기본적으로 bulk predictions 저장을 차단합니다. "
            "평가/레거시 bulk 저장이 필요하면 --allow-bulk-evaluation-save를 명시하세요."
        )
    return STORAGE_CONTRACT_EVALUATION_BULK
