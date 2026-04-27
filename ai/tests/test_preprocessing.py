import unittest

import torch  # noqa: F401
import pandas as pd

from ai.preprocessing import (
    CALENDAR_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    build_calendar_feature_frame,
    build_sequence_dataset,
    default_horizon,
    split_sequence_dataset,
)


def _build_feature_rows(length: int = 80, *, ticker: str = "AAPL") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    base_price = 100.0
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    for idx in range(length):
        rows.append(
            {
                "ticker": ticker,
                "timeframe": "1D",
                "date": dates[idx],
                "log_return": 0.01,
                "open_ratio": 0.01,
                "high_ratio": 0.02,
                "low_ratio": -0.01,
                "vol_change": 0.03,
                "ma_5_ratio": 0.01,
                "ma_20_ratio": 0.01,
                "ma_60_ratio": 0.01,
                "rsi": 0.5,
                "macd_ratio": 0.01,
                "bb_position": 0.5,
                "us10y": 4.0,
                "yield_spread": 1.0,
                "vix_close": 20.0,
                "credit_spread_hy": 3.0,
                "nh_nl_index": 0.2,
                "ma200_pct": 0.7,
                "regime_calm": 0.0,
                "regime_neutral": 1.0,
                "regime_stress": 0.0,
                "revenue": 100.0,
                "net_income": 10.0,
                "equity": 50.0,
                "eps": 1.0,
                "roe": 0.2,
                "debt_ratio": 0.4,
                "has_macro": True,
                "has_breadth": True,
                "has_fundamentals": True,
            }
        )
    feature_df = pd.DataFrame(rows)
    price_df = pd.DataFrame(
        {
            "ticker": [ticker] * length,
            "date": dates,
            "close": [base_price + idx for idx in range(length)],
            "adjusted_close": [base_price + idx for idx in range(length)],
            "open": [base_price + idx for idx in range(length)],
            "high": [base_price + idx + 1 for idx in range(length)],
            "low": [base_price + idx - 1 for idx in range(length)],
            "volume": [1000 + idx for idx in range(length)],
        }
    )
    return feature_df, price_df


class PreprocessingTestCase(unittest.TestCase):
    def test_default_horizon_matches_plan(self):
        self.assertEqual(default_horizon("1D"), 5)
        self.assertEqual(default_horizon("1W"), 4)
        with self.assertRaises(ValueError):
            default_horizon("1M")

    def test_build_sequence_dataset_shapes(self):
        feature_df, price_df = _build_feature_rows()
        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)
        train_bundle, val_bundle, test_bundle = split_sequence_dataset(dataset)

        self.assertEqual(dataset.features.shape[1:], (60, MODEL_N_FEATURES))
        self.assertEqual(dataset.future_covariates.shape[1:], (5, len(CALENDAR_FEATURE_COLUMNS)))
        self.assertEqual(dataset.line_targets.shape[1], 5)
        self.assertEqual(dataset.ticker_ids.shape[0], len(dataset))
        self.assertGreater(len(train_bundle), 0)
        self.assertGreater(len(val_bundle), 0)
        self.assertGreater(len(test_bundle), 0)

    def test_build_sequence_dataset_keeps_zero_filled_fundamentals_with_flag(self):
        feature_df, price_df = _build_feature_rows()
        feature_df[["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]] = 0.0
        feature_df["has_fundamentals"] = False

        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)

        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.features.shape[2], MODEL_N_FEATURES)
        self.assertTrue((dataset.ticker_ids == 0).all())

    def test_calendar_features_deterministic_from_date(self):
        dates = pd.to_datetime(["2026-01-16", "2026-01-16", "2026-03-20"])
        first = build_calendar_feature_frame(dates)
        second = build_calendar_feature_frame(dates)
        pd.testing.assert_frame_equal(first, second)

    def test_future_calendar_aligns_with_target_dates(self):
        feature_df, price_df = _build_feature_rows()
        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)

        first_row = dataset.metadata.iloc[0]
        expected = build_calendar_feature_frame(pd.to_datetime(first_row["forecast_dates"]))[CALENDAR_FEATURE_COLUMNS]
        actual = dataset.future_covariates[0].numpy()
        self.assertEqual(actual.shape, (5, len(CALENDAR_FEATURE_COLUMNS)))
        self.assertTrue((expected.to_numpy(dtype="float32") == actual).all())

    def test_future_calendar_no_value_leak(self):
        feature_df, price_df = _build_feature_rows()
        feature_df["log_return"] = range(len(feature_df))
        feature_df["us10y"] = 99.0
        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)

        future_cov = dataset.future_covariates[0]
        self.assertEqual(future_cov.shape[-1], len(CALENDAR_FEATURE_COLUMNS))
        self.assertTrue(float(future_cov.abs().max().item()) <= 1.0)


if __name__ == "__main__":
    unittest.main()
