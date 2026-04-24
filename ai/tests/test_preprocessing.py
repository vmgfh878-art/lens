import unittest

import pandas as pd

from ai.preprocessing import build_sequence_dataset, default_horizon, split_sequence_dataset


class PreprocessingTestCase(unittest.TestCase):
    def test_default_horizon_matches_plan(self):
        self.assertEqual(default_horizon("1D"), 5)
        self.assertEqual(default_horizon("1W"), 4)
        with self.assertRaises(ValueError):
            default_horizon("1M")

    def test_build_sequence_dataset_shapes(self):
        rows = []
        base_price = 100.0
        dates = pd.date_range("2026-01-01", periods=80, freq="D").strftime("%Y-%m-%d").tolist()
        for idx in range(80):
            rows.append(
                {
                    "ticker": "AAPL",
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
                "ticker": ["AAPL"] * 80,
                "date": dates,
                "close": [base_price + idx for idx in range(80)],
                "adjusted_close": [base_price + idx for idx in range(80)],
                "open": [base_price + idx for idx in range(80)],
                "high": [base_price + idx + 1 for idx in range(80)],
                "low": [base_price + idx - 1 for idx in range(80)],
                "volume": [1000 + idx for idx in range(80)],
            }
        )

        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)
        train_bundle, val_bundle, test_bundle = split_sequence_dataset(dataset)

        self.assertEqual(dataset.features.shape[1:], (60, 29))
        self.assertEqual(dataset.line_targets.shape[1], 5)
        self.assertGreater(len(train_bundle), 0)
        self.assertGreater(len(val_bundle), 0)
        self.assertGreater(len(test_bundle), 0)

    def test_build_sequence_dataset_keeps_zero_filled_fundamentals_with_flag(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=80, freq="D").strftime("%Y-%m-%d").tolist()
        for idx in range(80):
            rows.append(
                {
                    "ticker": "AAPL",
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
                    "revenue": 0.0,
                    "net_income": 0.0,
                    "equity": 0.0,
                    "eps": 0.0,
                    "roe": 0.0,
                    "debt_ratio": 0.0,
                    "has_macro": False,
                    "has_breadth": False,
                    "has_fundamentals": False,
                }
            )
        feature_df = pd.DataFrame(rows)
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 80,
                "date": dates,
                "close": [100.0 + idx for idx in range(80)],
                "adjusted_close": [100.0 + idx for idx in range(80)],
                "open": [100.0 + idx for idx in range(80)],
                "high": [101.0 + idx for idx in range(80)],
                "low": [99.0 + idx for idx in range(80)],
                "volume": [1000 + idx for idx in range(80)],
            }
        )

        dataset = build_sequence_dataset(feature_df, price_df, timeframe="1D", seq_len=60, horizon=5)

        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.features.shape[2], 29)


if __name__ == "__main__":
    unittest.main()
