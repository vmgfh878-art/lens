import unittest

import torch

from ai.loss import AsymmetricHuberLoss, ForecastCompositeLoss, PinballLoss
from ai.models.common import ForecastOutput


class LossTestCase(unittest.TestCase):
    def test_asymmetric_huber_penalizes_over_prediction_more(self):
        loss_fn = AsymmetricHuberLoss(alpha=1.0, beta=2.0, delta=1.0)
        over = loss_fn(torch.tensor([[2.0]]), torch.tensor([[1.0]])).item()
        under = loss_fn(torch.tensor([[0.0]]), torch.tensor([[1.0]])).item()
        self.assertGreater(over, under)

    def test_pinball_loss_is_non_negative(self):
        loss_fn = PinballLoss(0.1)
        loss = loss_fn(torch.tensor([[0.2, 0.3]]), torch.tensor([[0.1, 0.5]])).item()
        self.assertGreaterEqual(loss, 0.0)

    def test_composite_loss_returns_all_components(self):
        loss_fn = ForecastCompositeLoss()
        prediction = ForecastOutput(
            line=torch.tensor([[0.01, 0.02]]),
            lower_band=torch.tensor([[-0.01, 0.0]]),
            upper_band=torch.tensor([[0.03, 0.04]]),
        )
        breakdown = loss_fn(
            prediction,
            line_target=torch.tensor([[0.02, 0.01]]),
            band_target=torch.tensor([[0.015, 0.025]]),
        )
        self.assertGreaterEqual(breakdown.total.item(), 0.0)
        self.assertGreaterEqual(breakdown.band.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
