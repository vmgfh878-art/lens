import unittest
from unittest.mock import patch

from ai.composite_inference import (
    COMPOSITION_TOOL_STATUS,
    INDICATOR_LAYER_REPLACEMENT,
    LEGACY_REASON,
    _build_contract_checks,
    _save_composite_run,
    validate_legacy_save_allowed,
)


def _record_with_horizon(horizon: int) -> dict:
    meta = {
        "line_model_run_id": "line-run",
        "band_model_run_id": "band-run",
        "composition_policy": "raw_composite",
        "line_model_name": "patchtst",
        "band_model_name": "cnn_lstm",
        "band_calibration_method": "scalar_width",
        "band_calibration_params": {"lower_scale": 1.0, "upper_scale": 1.0},
        "prediction_composition_version": "line_band_v1",
        "deprecated_for_phase1_product_contract": True,
        "indicator_layer_replacement": "overlay_bundle",
        "legacy_reason": LEGACY_REASON,
        "composite_is_not_product_default": True,
    }
    values = [float(index) for index in range(horizon)]
    return {
        "forecast_dates": [f"2026-01-{index + 1:02d}" for index in range(horizon)],
        "line_series": values,
        "lower_band_series": values,
        "upper_band_series": values,
        "conservative_series": values,
        "meta": meta,
    }


class CompositeLegacyGuardTestCase(unittest.TestCase):
    def test_save_without_allow_flag_fails(self):
        with self.assertRaisesRegex(ValueError, "--allow-legacy-composite-save"):
            validate_legacy_save_allowed(save=True, allow_legacy_composite_save=False)

    def test_save_with_allow_flag_is_valid(self):
        validate_legacy_save_allowed(save=True, allow_legacy_composite_save=True)
        validate_legacy_save_allowed(save=False, allow_legacy_composite_save=False)

    def test_legacy_save_forces_deprecated_meta(self):
        line_config = {"timeframe": "1D", "horizon": 5, "feature_version": "v3_adjusted_ohlc"}
        band_config = {"q_low": 0.15, "q_high": 0.85, "band_mode": "direct"}

        with patch("ai.composite_inference.save_model_run") as save_model_run:
            _save_composite_run(
                run_id="composite-run",
                line_config=line_config,
                band_config=band_config,
                line_model_run_id="line-run",
                band_model_run_id="band-run",
                summary={"coverage": 0.8},
                calibration_params={"lower_scale": 1.0, "upper_scale": 1.0},
                composition_policy="raw_composite",
            )

        record = save_model_run.call_args.args[0]
        config = record["config"]
        self.assertEqual(config["phase1_contract_status"], COMPOSITION_TOOL_STATUS)
        self.assertTrue(config["deprecated_for_phase1_product_contract"])
        self.assertEqual(config["indicator_layer_replacement"], INDICATOR_LAYER_REPLACEMENT)
        self.assertEqual(config["role"], "composite_model")
        self.assertEqual(config["legacy_reason"], LEGACY_REASON)
        self.assertTrue(config["composite_is_not_product_default"])

    def test_horizon_dynamic_length_check_supports_h20(self):
        checks = _build_contract_checks(
            records=[_record_with_horizon(20)],
            lower_le_upper_flags=[True],
            line_inside_band_flags=[True],
            expected_horizon=20,
        )

        self.assertTrue(checks["series_length_matches_horizon"])
        self.assertTrue(checks["forecast_dates_length_matches_horizon"])
        self.assertEqual(checks["actual_line_lengths"], [20])
        self.assertNotIn("series_length_all_5", checks)

    def test_line_inside_band_false_is_diagnostic_not_failure(self):
        checks = _build_contract_checks(
            records=[_record_with_horizon(5)],
            lower_le_upper_flags=[True],
            line_inside_band_flags=[False],
            expected_horizon=5,
        )

        self.assertTrue(checks["lower_le_upper_all"])
        self.assertFalse(checks["line_inside_band_all"])
        self.assertTrue(checks["line_inside_band_diagnostic_only"])
        self.assertFalse(checks["line_inside_band_is_success_condition"])
        self.assertNotIn("line_inside_band_required", checks)


if __name__ == "__main__":
    unittest.main()
