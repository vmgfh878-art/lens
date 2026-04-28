import unittest

import torch

from ai.train import (
    CheckpointSelector,
    coverage_gate_eligible,
)


def _metrics(
    *,
    coverage: float,
    upper: float,
    lower: float = 0.05,
    ic: float = 0.01,
    spread: float = 0.001,
    forecast_loss: float = 1.0,
    fee_return: float = 0.1,
) -> dict[str, float]:
    return {
        "coverage": coverage,
        "upper_breach_rate": upper,
        "lower_breach_rate": lower,
        "spearman_ic": ic,
        "long_short_spread": spread,
        "forecast_loss": forecast_loss,
        "total_loss": forecast_loss,
        "fee_adjusted_return": fee_return,
    }


class CheckpointSelectionTest(unittest.TestCase):
    def test_coverage_gate_eligible(self):
        self.assertTrue(
            coverage_gate_eligible(
                _metrics(coverage=0.82, upper=0.09, lower=0.10, ic=0.02, spread=0.003)
            )
        )
        self.assertFalse(coverage_gate_eligible(_metrics(coverage=0.70, upper=0.09)))
        self.assertFalse(coverage_gate_eligible(_metrics(coverage=0.82, upper=0.16)))
        self.assertFalse(coverage_gate_eligible(_metrics(coverage=0.82, upper=0.09, ic=0.0)))
        self.assertFalse(coverage_gate_eligible(_metrics(coverage=0.82, upper=0.09, spread=0.0)))

    def test_coverage_gate_prefers_low_upper_breach_among_eligible(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("coverage_gate")

        with torch.no_grad():
            model.weight.fill_(1.0)
        selector.update(epoch=1, metrics=_metrics(coverage=0.96, upper=0.02, forecast_loss=0.2), model=model)

        with torch.no_grad():
            model.weight.fill_(2.0)
        selector.update(epoch=2, metrics=_metrics(coverage=0.80, upper=0.09, forecast_loss=0.3), model=model)

        with torch.no_grad():
            model.weight.fill_(3.0)
        selector.update(epoch=3, metrics=_metrics(coverage=0.86, upper=0.12, forecast_loss=0.1), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "coverage_gate_eligible")
        self.assertFalse(selected.coverage_gate_failed)
        self.assertEqual(float(selected.candidate.state_dict["weight"][0, 0]), 2.0)
        self.assertEqual(selected.best_val_total_epoch, 3)

    def test_coverage_gate_falls_back_to_val_total_when_no_epoch_passes(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("coverage_gate")
        selector.update(epoch=1, metrics=_metrics(coverage=0.60, upper=0.20, forecast_loss=0.5), model=model)
        selector.update(epoch=2, metrics=_metrics(coverage=0.65, upper=0.18, forecast_loss=0.2), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "coverage_gate_failed_fallback_val_total")
        self.assertTrue(selected.coverage_gate_failed)
        self.assertEqual(selected.best_val_total_epoch, 2)

    def test_val_total_selector_keeps_legacy_behavior(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("val_total")
        selector.update(epoch=1, metrics=_metrics(coverage=0.82, upper=0.09, forecast_loss=0.4), model=model)
        selector.update(epoch=2, metrics=_metrics(coverage=0.78, upper=0.10, forecast_loss=0.3), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "val_total_best")
        self.assertFalse(selected.coverage_gate_failed)


if __name__ == "__main__":
    unittest.main()
