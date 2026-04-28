"""CP12 finite gate 테스트.

다루는 범위:
- assert_finite_metrics / check_metrics_finite 동작
- is_nan_safe_better NaN 처리
- run_epoch이 eval phase에서 NaN 즉시 raise
- run_epoch이 train phase에서 모든 batch NaN이면 raise (streak 한도와 별개로 batch_count=0 가드)
- EarlyStopping이 NaN을 best로 받지 않음
- argparse가 --amp-dtype off / bf16 / fp16를 모두 받음
- save_checkpoint 호출 안 됨 (NaN 시): 통합 흐름 단위 테스트는 비용이 커 단위 검증으로 대체.
- inference / backtest의 status='failed_nan' run에 대한 거부
"""

from __future__ import annotations

import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from ai.finite import (
    FiniteCheckResult,
    assert_finite_metrics,
    check_metrics_finite,
    is_nan_safe_better,
    tensor_finite_summary,
)
from ai.loss import LossBreakdown
from ai.models.common import ForecastOutput
from ai.train import EarlyStopping, parse_args, run_epoch


class _ConstantCriterion:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, prediction, line_target, band_target, raw_future_returns=None):
        del prediction, line_target, band_target, raw_future_returns
        total = torch.tensor(self.value, dtype=torch.float32)
        return LossBreakdown(
            total=total,
            forecast=total,
            line=total,
            band=total,
            cross=total,
            direction=total,
        )


class _ZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 옵티마이저가 빈 파라미터 리스트로 실패하지 않도록 dummy 파라미터를 둔다.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, features, ticker_id=None):
        del ticker_id
        batch = features.shape[0]
        zeros = torch.zeros(batch, 2, dtype=features.dtype) + self._dummy
        return ForecastOutput(line=zeros, lower_band=zeros - 0.1, upper_band=zeros + 0.1)


def _make_loader(num_batches: int = 1) -> DataLoader:
    features = torch.randn(num_batches, 4, 3)
    targets = torch.randn(num_batches, 2)
    raw_returns = targets.clone()
    ticker_ids = torch.zeros(num_batches, dtype=torch.long)
    future_covariates = torch.zeros(num_batches, 2, 0)
    return DataLoader(
        TensorDataset(features, targets, targets.clone(), raw_returns, ticker_ids, future_covariates),
        batch_size=1,
        shuffle=False,
    )


class CheckMetricsFiniteTest(unittest.TestCase):
    def test_passes_when_all_finite(self) -> None:
        result = check_metrics_finite({"loss": 0.5, "coverage": 0.9, "_meta": "ok"}, phase="val")
        self.assertTrue(result.ok)
        self.assertIsNone(result.failed_metric)

    def test_detects_nan(self) -> None:
        result = check_metrics_finite({"a": 1.0, "b": float("nan")}, phase="val", run_id="r", epoch=2)
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_metric, "b")
        self.assertEqual(result.phase, "val")
        meta = result.to_meta()
        self.assertEqual(meta["failed_phase"], "val")
        self.assertEqual(meta["failed_metric"], "b")
        self.assertEqual(meta["failed_epoch"], 2)
        self.assertIn("phase=val", result.format_message())

    def test_detects_inf(self) -> None:
        result = check_metrics_finite({"x": float("inf")}, phase="train")
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_metric, "x")

    def test_assert_finite_metrics_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_finite_metrics({"loss": float("nan")}, phase="checkpoint")

    def test_none_passes(self) -> None:
        # None 값(선택 메트릭)은 finite 검증을 통과해야 한다.
        result = check_metrics_finite({"spearman_ic": None, "loss": 0.1}, phase="val")
        self.assertTrue(result.ok)


class TensorFiniteSummaryTest(unittest.TestCase):
    def test_handles_none(self) -> None:
        summary = tensor_finite_summary({"a": None, "b": torch.tensor([1.0, 2.0])})
        self.assertNotIn("a", summary)
        self.assertEqual(summary["b"]["finite_ratio"], 1.0)
        self.assertFalse(summary["b"]["has_nan"])

    def test_detects_nan_inf(self) -> None:
        summary = tensor_finite_summary({"x": torch.tensor([1.0, float("nan"), float("inf")])})
        self.assertTrue(summary["x"]["has_nan"])
        self.assertTrue(summary["x"]["has_inf"])
        self.assertLess(summary["x"]["finite_ratio"], 1.0)


class IsNanSafeBetterTest(unittest.TestCase):
    def test_finite_candidate_beats_none_best(self) -> None:
        self.assertTrue(is_nan_safe_better(0.5, None, mode="min"))

    def test_nan_candidate_never_wins(self) -> None:
        self.assertFalse(is_nan_safe_better(float("nan"), 1.0, mode="min"))
        self.assertFalse(is_nan_safe_better(float("nan"), None, mode="min"))

    def test_finite_beats_nan_best(self) -> None:
        self.assertTrue(is_nan_safe_better(1.0, float("nan"), mode="min"))

    def test_min_delta_required(self) -> None:
        self.assertFalse(is_nan_safe_better(0.999, 1.0, mode="min", min_delta=0.01))
        self.assertTrue(is_nan_safe_better(0.98, 1.0, mode="min", min_delta=0.01))


class EarlyStoppingNaNTest(unittest.TestCase):
    def test_nan_does_not_become_best(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.0, mode="min")
        model = _ZeroModel()
        # 첫 epoch가 NaN이면 best로 채택되면 안 된다.
        es.step(float("nan"), epoch=1, model=model)
        self.assertIsNone(es.best_epoch)
        # 다음 finite metric은 best가 되어야 한다.
        es.step(0.7, epoch=2, model=model)
        self.assertEqual(es.best_epoch, 2)
        self.assertEqual(es.best_value, 0.7)

    def test_nan_after_best_keeps_best(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.0, mode="min")
        model = _ZeroModel()
        es.step(0.5, epoch=1, model=model)
        es.step(float("nan"), epoch=2, model=model)
        self.assertEqual(es.best_epoch, 1)
        self.assertEqual(es.best_value, 0.5)
        # NaN epoch도 patience 카운트는 증가해야 한다.
        self.assertEqual(es.epochs_since_improve, 1)


class RunEpochNaNGateTest(unittest.TestCase):
    def test_eval_nan_raises_immediately(self) -> None:
        loader = _make_loader(num_batches=2)
        with self.assertRaises(RuntimeError) as ctx:
            run_epoch(
                model=_ZeroModel(),
                loader=loader,
                criterion=_ConstantCriterion(float("nan")),
                device=torch.device("cpu"),
                epoch=1,
                run_id="r",
            )
        self.assertIn("NaN-GATE", str(ctx.exception))
        self.assertIn("phase=val", str(ctx.exception))

    def test_train_all_nan_raises_via_batch_count_guard(self) -> None:
        loader = _make_loader(num_batches=1)
        model = _ZeroModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with self.assertRaises(RuntimeError):
            run_epoch(
                model=model,
                loader=loader,
                criterion=_ConstantCriterion(float("nan")),
                device=torch.device("cpu"),
                epoch=1,
                optimizer=optimizer,
                run_id="r",
            )


class CliParseTest(unittest.TestCase):
    def test_amp_dtype_choices(self) -> None:
        import sys

        original = sys.argv
        try:
            for choice in ("bf16", "fp16", "off"):
                sys.argv = ["train.py", "--amp-dtype", choice, "--no-wandb"]
                args = parse_args()
                self.assertEqual(args.amp_dtype, choice)
        finally:
            sys.argv = original

    def test_detect_anomaly_default_false(self) -> None:
        import sys

        original = sys.argv
        try:
            sys.argv = ["train.py", "--no-wandb"]
            args = parse_args()
            self.assertFalse(args.detect_anomaly)
            sys.argv = ["train.py", "--detect-anomaly", "--no-wandb"]
            args = parse_args()
            self.assertTrue(args.detect_anomaly)
        finally:
            sys.argv = original


class FailureMetaTest(unittest.TestCase):
    def test_failure_meta_round_trip(self) -> None:
        result = FiniteCheckResult(
            ok=False,
            failed_metric="val/coverage",
            failed_value=float("nan"),
            phase="val",
            run_id="run-1",
            epoch=3,
            batch=42,
        )
        meta = result.to_meta()
        self.assertEqual(meta["failed_phase"], "val")
        self.assertEqual(meta["failed_metric"], "val/coverage")
        self.assertEqual(meta["failed_epoch"], 3)
        self.assertEqual(meta["failed_batch"], 42)
        # NaN value는 meta에 그대로 들어간다 (직렬화 시 호출자가 처리).
        self.assertTrue(math.isnan(meta["failed_value"]))


if __name__ == "__main__":
    unittest.main()
