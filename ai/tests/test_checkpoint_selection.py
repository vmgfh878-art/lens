import unittest

import torch

from ai.train import (
    CheckpointSelector,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED_QUALITY_GATE,
    band_gate_eligible,
    combined_gate_eligible,
    coverage_gate_eligible,
    line_gate_eligible,
    resolve_persisted_run_status,
)


def _metrics(
    *,
    coverage: float = 0.82,
    upper: float = 0.09,
    lower: float = 0.10,
    ic: float = 0.02,
    spread: float = 0.003,
    mae: float = 0.04,
    smape: float = 1.2,
    avg_band_width: float = 0.08,
    band_loss: float = 0.02,
    forecast_loss: float = 1.0,
    fee_return: float = 0.1,
) -> dict[str, float]:
    return {
        "coverage": coverage,
        "upper_breach_rate": upper,
        "lower_breach_rate": lower,
        "spearman_ic": ic,
        "long_short_spread": spread,
        "mae": mae,
        "smape": smape,
        "avg_band_width": avg_band_width,
        "band_loss": band_loss,
        "forecast_loss": forecast_loss,
        "total_loss": forecast_loss,
        "fee_adjusted_return": fee_return,
    }


class CheckpointSelectionTest(unittest.TestCase):
    def test_line_gate_pass_fail(self):
        self.assertTrue(line_gate_eligible(_metrics(ic=0.02, spread=0.003)))
        self.assertFalse(line_gate_eligible(_metrics(ic=0.0, spread=0.003)))
        self.assertFalse(line_gate_eligible(_metrics(ic=0.02, spread=0.0)))
        self.assertFalse(line_gate_eligible(_metrics(ic=0.02, spread=0.003, mae=float("nan"))))

    def test_band_gate_pass_fail(self):
        self.assertTrue(band_gate_eligible(_metrics(coverage=0.85, upper=0.08, lower=0.10)))
        self.assertFalse(band_gate_eligible(_metrics(coverage=0.70, upper=0.08, lower=0.10)))
        self.assertFalse(band_gate_eligible(_metrics(coverage=0.85, upper=0.16, lower=0.10)))
        self.assertFalse(band_gate_eligible(_metrics(coverage=0.85, upper=0.08, lower=0.21)))
        self.assertFalse(band_gate_eligible(_metrics(coverage=0.85, upper=0.08, lower=0.10, avg_band_width=0.0)))

    def test_combined_gate_pass_fail(self):
        self.assertTrue(combined_gate_eligible(_metrics()))
        self.assertFalse(combined_gate_eligible(_metrics(ic=-0.01)))
        self.assertFalse(combined_gate_eligible(_metrics(coverage=0.99)))

    def test_coverage_gate_alias_matches_combined_gate(self):
        passing = _metrics()
        failing_line = _metrics(ic=-0.01)
        self.assertEqual(coverage_gate_eligible(passing), combined_gate_eligible(passing))
        self.assertEqual(coverage_gate_eligible(failing_line), combined_gate_eligible(failing_line))

    def test_band_gate_passes_even_when_ic_is_negative(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("band_gate")
        selector.update(epoch=1, metrics=_metrics(coverage=0.84, upper=0.07, lower=0.10, ic=-0.03), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 1)
        self.assertEqual(selected.gate_type, "band_gate")
        self.assertEqual(selected.selected_reason, "band_gate_eligible")
        self.assertFalse(selected.gate_failed)
        self.assertFalse(selected.line_gate_pass)
        self.assertTrue(selected.band_gate_pass)
        self.assertFalse(selected.combined_gate_pass)

    def test_line_gate_passes_even_when_coverage_fails(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("line_gate")
        selector.update(epoch=1, metrics=_metrics(coverage=0.99, upper=0.01, ic=0.03, spread=0.004), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 1)
        self.assertEqual(selected.gate_type, "line_gate")
        self.assertEqual(selected.selected_reason, "line_gate_eligible")
        self.assertFalse(selected.gate_failed)
        self.assertTrue(selected.line_gate_pass)
        self.assertFalse(selected.band_gate_pass)
        self.assertFalse(selected.combined_gate_pass)

    def test_combined_gate_prefers_legacy_sort_order(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("combined_gate")

        selector.update(epoch=1, metrics=_metrics(coverage=0.80, upper=0.10, ic=0.04, spread=0.002, forecast_loss=0.2), model=model)
        selector.update(epoch=2, metrics=_metrics(coverage=0.80, upper=0.08, ic=0.02, spread=0.002, forecast_loss=0.3), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "combined_gate_eligible")
        self.assertFalse(selected.gate_failed)

    def test_band_gate_sorts_by_target_coverage_then_breach_width_loss(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("band_gate")

        selector.update(epoch=1, metrics=_metrics(coverage=0.80, upper=0.04, lower=0.08, avg_band_width=0.05), model=model)
        selector.update(epoch=2, metrics=_metrics(coverage=0.85, upper=0.10, lower=0.08, avg_band_width=0.09), model=model)
        selector.update(epoch=3, metrics=_metrics(coverage=0.88, upper=0.02, lower=0.08, avg_band_width=0.04), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "band_gate_eligible")

    def test_coverage_gate_alias_preserves_legacy_reason_names(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("coverage_gate")
        selector.update(epoch=1, metrics=_metrics(), model=model)

        selected = selector.select()
        self.assertEqual(selected.gate_type, "combined_gate")
        self.assertEqual(selected.selected_reason, "coverage_gate_eligible")
        self.assertFalse(selected.coverage_gate_failed)

    def test_gate_falls_back_to_val_total_when_no_epoch_passes(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("band_gate")
        selector.update(epoch=1, metrics=_metrics(coverage=0.60, upper=0.20, forecast_loss=0.5), model=model)
        selector.update(epoch=2, metrics=_metrics(coverage=0.65, upper=0.18, forecast_loss=0.2), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "band_gate_failed_fallback_val_total")
        self.assertTrue(selected.gate_failed)
        self.assertTrue(selected.coverage_gate_failed)
        self.assertEqual(selected.best_val_total_epoch, 2)

    def test_val_total_selector_keeps_legacy_behavior(self):
        model = torch.nn.Linear(1, 1)
        selector = CheckpointSelector("val_total")
        selector.update(epoch=1, metrics=_metrics(forecast_loss=0.4), model=model)
        selector.update(epoch=2, metrics=_metrics(forecast_loss=0.3), model=model)

        selected = selector.select()
        self.assertEqual(selected.candidate.epoch, 2)
        self.assertEqual(selected.selected_reason, "val_total_best")
        self.assertFalse(selected.gate_failed)
        self.assertFalse(selected.coverage_gate_failed)

    def test_gate_failed_status_policy(self):
        self.assertEqual(
            resolve_persisted_run_status(gate_failed=True),
            RUN_STATUS_FAILED_QUALITY_GATE,
        )
        self.assertEqual(
            resolve_persisted_run_status(gate_failed=False),
            RUN_STATUS_COMPLETED,
        )
        self.assertEqual(
            resolve_persisted_run_status(coverage_gate_failed=True),
            RUN_STATUS_FAILED_QUALITY_GATE,
        )


if __name__ == "__main__":
    unittest.main()
