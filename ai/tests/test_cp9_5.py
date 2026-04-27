import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from ai.loss import LossBreakdown
from ai.train import _summarize_predictions, run_epoch


class _DummyCriterion:
    def __init__(self, total_value: float) -> None:
        self.total_value = total_value

    def __call__(self, prediction, line_target, band_target, raw_future_returns=None):
        del prediction, line_target, band_target, raw_future_returns
        total = torch.tensor(self.total_value, dtype=torch.float32)
        return LossBreakdown(
            total=total,
            forecast=total,
            line=total,
            band=total,
            cross=total,
            direction=total,
        )


class _DummyModel(torch.nn.Module):
    def forward(self, features, ticker_id=None):
        del ticker_id
        batch_size = features.shape[0]
        horizon = 2
        zeros = torch.zeros(batch_size, horizon, dtype=features.dtype)
        from ai.models.common import ForecastOutput

        return ForecastOutput(
            line=zeros,
            lower_band=zeros - 0.1,
            upper_band=zeros + 0.1,
        )


class CP95TestCase(unittest.TestCase):
    def test_nan_loss_raises_runtime_error(self):
        features = torch.randn(2, 4, 3)
        targets = torch.randn(2, 2)
        ticker_ids = torch.zeros(2, dtype=torch.long)
        future_covariates = torch.zeros(2, 2, 0)
        raw_future_returns = targets.clone()
        loader = DataLoader(
            TensorDataset(features, targets, targets.clone(), raw_future_returns, ticker_ids, future_covariates),
            batch_size=1,
            shuffle=False,
        )

        with self.assertRaises(RuntimeError):
            run_epoch(
                model=_DummyModel(),
                loader=loader,
                criterion=_DummyCriterion(float("nan")),
                device=torch.device("cpu"),
                epoch=3,
                debug_label="trial=7",
            )

    def test_mae_metric_finite_for_zero_targets(self):
        metrics = _summarize_predictions(
            line_predictions=[torch.tensor([[0.0, 0.1]])],
            lower_predictions=[torch.tensor([[-0.1, 0.0]])],
            upper_predictions=[torch.tensor([[0.1, 0.2]])],
            line_targets=[torch.tensor([[0.0, 0.0]])],
            band_targets=[torch.tensor([[0.0, 0.0]])],
            raw_future_returns=[torch.tensor([[0.0, 0.0]])],
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["mae"])).item())
        self.assertTrue(torch.isfinite(torch.tensor(metrics["smape"])).item())


if __name__ == "__main__":
    unittest.main()
