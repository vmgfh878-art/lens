"""학습/평가 단계의 NaN/Inf 감지 유틸리티.

CP12에서 도입했다. validation/test/checkpoint 결과가 NaN/Inf로 오염되면 즉시
실패 시그널을 만들어 후속 저장을 차단하고, 디버그용 phase/metric/run_id/epoch/batch
정보를 함께 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch


@dataclass
class FiniteCheckResult:
    """finite 검증 결과 보고서."""

    ok: bool
    failed_metric: str | None = None
    failed_value: float | None = None
    phase: str = ""
    run_id: str = ""
    epoch: int = -1
    batch: int = -1
    extras: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        return {
            "failed_phase": self.phase,
            "failed_metric": self.failed_metric,
            "failed_value": self.failed_value,
            "failed_epoch": self.epoch,
            "failed_batch": self.batch,
            **({"extras": self.extras} if self.extras else {}),
        }

    def format_message(self) -> str:
        bits = [f"[NaN-GATE phase={self.phase}"]
        if self.run_id:
            bits.append(f"run_id={self.run_id}")
        if self.epoch >= 0:
            bits.append(f"epoch={self.epoch}")
        if self.batch >= 0:
            bits.append(f"batch={self.batch}")
        bits.append(f"metric={self.failed_metric}")
        bits.append(f"value={self.failed_value!r}]")
        return " ".join(bits)


def _is_nonfinite_scalar(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int,)):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(torch.isfinite(value).logical_not().any().item())
    return False


def check_metrics_finite(
    metrics: Mapping[str, Any],
    *,
    phase: str,
    run_id: str = "",
    epoch: int = -1,
    batch: int = -1,
) -> FiniteCheckResult:
    """metrics dict 안의 스칼라 값들을 isfinite 검사한다. 첫 실패 항목을 보고한다.

    None은 통과시킨다 (선택적 필드). list/dict/문자열도 통과 (호출자가 별도 검증).
    """
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if _is_nonfinite_scalar(value):
            float_value: float | None
            if isinstance(value, torch.Tensor):
                try:
                    float_value = float(value.detach().reshape(-1)[0].item())
                except Exception:
                    float_value = None
            else:
                try:
                    float_value = float(value)
                except Exception:
                    float_value = None
            return FiniteCheckResult(
                ok=False,
                failed_metric=key,
                failed_value=float_value,
                phase=phase,
                run_id=run_id,
                epoch=epoch,
                batch=batch,
            )
    return FiniteCheckResult(ok=True, phase=phase, run_id=run_id, epoch=epoch, batch=batch)


def assert_finite_metrics(
    metrics: Mapping[str, Any],
    *,
    phase: str,
    run_id: str = "",
    epoch: int = -1,
    batch: int = -1,
) -> None:
    """check_metrics_finite + 실패 시 RuntimeError로 즉시 던진다."""
    result = check_metrics_finite(metrics, phase=phase, run_id=run_id, epoch=epoch, batch=batch)
    if not result.ok:
        raise RuntimeError(result.format_message())


def tensor_finite_summary(named: Mapping[str, torch.Tensor | None]) -> dict[str, dict[str, float | bool | int]]:
    """이름→tensor 매핑에 대해 finite 비율과 min/max를 계산한다. None은 무시."""
    summary: dict[str, dict[str, float | bool | int]] = {}
    for name, tensor in named.items():
        if tensor is None:
            continue
        if tensor.numel() == 0:
            summary[name] = {
                "numel": 0,
                "finite_ratio": 1.0,
                "has_nan": False,
                "has_inf": False,
            }
            continue
        flat = tensor.detach()
        finite_mask = torch.isfinite(flat)
        has_nan = bool(torch.isnan(flat).any().item())
        has_inf = bool((flat.abs() == float("inf")).any().item())
        finite_ratio = float(finite_mask.float().mean().item())
        finite_values = flat[finite_mask]
        if finite_values.numel() > 0:
            try:
                value_min = float(finite_values.min().item())
                value_max = float(finite_values.max().item())
            except Exception:
                value_min = float("nan")
                value_max = float("nan")
        else:
            value_min = float("nan")
            value_max = float("nan")
        summary[name] = {
            "numel": int(flat.numel()),
            "finite_ratio": finite_ratio,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "min": value_min,
            "max": value_max,
        }
    return summary


def is_nan_safe_better(candidate: float | None, best: float | None, *, mode: str = "min", min_delta: float = 0.0) -> bool:
    """NaN을 항상 worse로 처리하는 비교. candidate가 best보다 *엄격히* 더 좋으면 True.

    NaN candidate는 절대 best가 될 수 없다. NaN best는 어떤 finite candidate에게도 진다.
    """
    if candidate is None:
        return False
    if isinstance(candidate, float) and not math.isfinite(candidate):
        return False
    if best is None:
        return True
    if isinstance(best, float) and not math.isfinite(best):
        return True
    if mode == "min":
        return candidate < (best - min_delta)
    if mode == "max":
        return candidate > (best + min_delta)
    raise ValueError(f"지원하지 않는 mode입니다: {mode}")


__all__ = [
    "FiniteCheckResult",
    "assert_finite_metrics",
    "check_metrics_finite",
    "is_nan_safe_better",
    "tensor_finite_summary",
]
