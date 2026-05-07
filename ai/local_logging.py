from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WarnFn = Callable[[str], None]


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if np is not None and isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if np is not None and isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return sanitize_for_json(tensor.item())
        return sanitize_for_json(tensor.tolist())
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    return str(value)


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(sanitize_for_json(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)


class LocalTrainingProgressLogger:
    """W&B 없이도 학습 진행률을 로컬 JSON 파일로 남기는 안전한 logger."""

    def __init__(
        self,
        *,
        run_id: str,
        base_dir: str | Path,
        enabled: bool = True,
        warn: WarnFn | None = None,
    ) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self._warn = warn or (lambda message: print(message))
        self._active = bool(enabled)
        self._warning_emitted = False
        if self._active:
            self._safe(lambda: self.run_dir.mkdir(parents=True, exist_ok=True))

    @property
    def active(self) -> bool:
        return self._active

    def _warn_once(self, exc: Exception) -> None:
        if self._warning_emitted:
            return
        self._warning_emitted = True
        self._warn(f"경고: local training log 기록 실패: {exc}")

    def _safe(self, action: Callable[[], None]) -> None:
        if not self._active:
            return
        try:
            action()
        except Exception as exc:  # pragma: no cover - 구체 예외는 파일 시스템별로 달라진다.
            self._active = False
            self._warn_once(exc)

    def write_config(self, payload: dict[str, Any]) -> None:
        def _write() -> None:
            self.config_path.write_text(dumps_json(payload) + "\n", encoding="utf-8")

        self._safe(_write)

    def write_epoch(self, payload: dict[str, Any]) -> None:
        def _write() -> None:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(dumps_json(payload) + "\n")

        self._safe(_write)

    def write_summary(self, payload: dict[str, Any]) -> None:
        def _write() -> None:
            self.summary_path.write_text(dumps_json(payload) + "\n", encoding="utf-8")

        self._safe(_write)
