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
        self.assertGreaterEqual(breakdown.forecast.item(), 0.0)
        self.assertEqual(float(breakdown.direction.item()), 0.0)

    def test_width_loss_removed_from_composite(self):
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
        self.assertFalse(hasattr(breakdown, "width"))
        self.assertNotIn("width_loss", breakdown.to_log_dict())

    def test_direction_loss_added_only_when_direction_head_exists(self):
        loss_fn = ForecastCompositeLoss(lambda_direction=0.1)
        prediction = ForecastOutput(
            line=torch.tensor([[0.01, -0.02]]),
            lower_band=torch.tensor([[-0.01, -0.03]]),
            upper_band=torch.tensor([[0.03, 0.01]]),
            direction_logit=torch.tensor([[0.5, 0.5]]),
        )
        breakdown = loss_fn(
            prediction,
            line_target=torch.tensor([[0.02, -0.01]]),
            band_target=torch.tensor([[0.015, -0.005]]),
            raw_future_returns=torch.tensor([[0.01, -0.02]]),
        )
        self.assertGreater(breakdown.direction.item(), 0.0)
        self.assertGreater(breakdown.total.item(), breakdown.forecast.item())


if __name__ == "__main__":
    unittest.main()
