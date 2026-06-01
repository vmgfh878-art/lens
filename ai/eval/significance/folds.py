"""CP216.2 — CP153 / CP178 stage5T walk-forward fold 정의.

docs/v1_operating_models_reproducibility.md §2~3 의 fold 정의 그대로.
운영 모델과 1:1 일치 (자가 점검 PASS 조건).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: str
    train_end: str    # exclusive 끝
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def in_train(self, d: str) -> bool:
        return self.train_start <= d < self.train_end

    def in_val(self, d: str) -> bool:
        return self.val_start <= d < self.val_end

    def in_test(self, d: str) -> bool:
        return self.test_start <= d < self.test_end

    def eval_range(self) -> tuple[str, str]:
        """val_start ~ test_end (OOS 전체 영역)."""
        return self.val_start, self.test_end


# CP153 stage5T 1D — train_start 5y 슬라이딩 + val 6m + test 6m
CP153_FOLDS: list[Fold] = [
    Fold(1, "2019-05-01", "2024-05-01", "2024-05-01", "2024-11-01", "2024-11-01", "2025-05-01"),
    Fold(2, "2019-11-01", "2024-11-01", "2024-11-01", "2025-05-01", "2025-05-01", "2025-11-01"),
    Fold(3, "2020-05-01", "2025-05-01", "2025-05-01", "2025-11-01", "2025-11-01", "2026-05-09"),
]

# CP178 stage5 1W — same calendar.
CP178_FOLDS: list[Fold] = list(CP153_FOLDS)


def assign_fold(asof_date: str, folds: List[Fold]) -> int | None:
    """asof_date 가 어느 fold 의 eval 영역(val+test)에 속하는지 fold_id 반환. 없으면 None."""
    for f in folds:
        if f.val_start <= asof_date < f.test_end:
            return f.fold_id
    return None


def assert_matches_reproducibility() -> None:
    """fold 정의가 reproducibility.md 의 §2 표와 1:1 일치하는지 sanity."""
    # 동치 fold 표 (assertion)
    expected = [
        (1, "2019-05-01", "2024-05-01", "2024-05-01", "2024-11-01", "2024-11-01", "2025-05-01"),
        (2, "2019-11-01", "2024-11-01", "2024-11-01", "2025-05-01", "2025-05-01", "2025-11-01"),
        (3, "2020-05-01", "2025-05-01", "2025-05-01", "2025-11-01", "2025-11-01", "2026-05-09"),
    ]
    for f, e in zip(CP153_FOLDS, expected):
        assert (f.fold_id, f.train_start, f.train_end, f.val_start, f.val_end, f.test_start, f.test_end) == e, (
            f"CP153 fold {f.fold_id} mismatch vs reproducibility.md"
        )


if __name__ == "__main__":
    assert_matches_reproducibility()
    print("CP153 / CP178 fold definitions match reproducibility.md")
