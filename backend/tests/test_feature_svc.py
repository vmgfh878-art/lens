import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import build_features


class FeatureServiceTestCase(unittest.TestCase):
    def test_build_features_expands_regime_one_hot(self):
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 140,
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "open": [100 + idx for idx in range(140)],
                "high": [101 + idx for idx in range(140)],
                "low": [99 + idx for idx in range(140)],
                "close": [100 + idx for idx in range(140)],
                "adjusted_close": [100 + idx for idx in range(140)],
                "volume": [1000 + idx for idx in range(140)],
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "us10y": [4.0] * 140,
                "yield_spread": [1.0] * 140,
                "vix_close": [20.0] * 60 + [12.0] * 30 + [20.0] * 25 + [30.0] * 25,
                "credit_spread_hy": [3.0] * 140,
            }
        )
        breadth_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "nh_nl_index": [0.2] * 140,
                "ma200_pct": [0.7] * 140,
            }
        )

        features = build_features(price_df=price_df, macro_df=macro_df, breadth_df=breadth_df, timeframe="1D")

        self.assertFalse(features.empty)
        self.assertIn("regime_label", features.columns)
        self.assertIn("regime_calm", features.columns)
        self.assertIn("regime_neutral", features.columns)
        self.assertIn("regime_stress", features.columns)
        self.assertIn("has_macro", features.columns)
        self.assertIn("has_breadth", features.columns)
        self.assertIn("calm", set(features["regime_label"]))
        self.assertIn("neutral", set(features["regime_label"]))
        self.assertIn("stress", set(features["regime_label"]))
        self.assertTrue(((features["regime_calm"] + features["regime_neutral"] + features["regime_stress"]) == 1.0).all())
        self.assertTrue((features["has_macro"] == True).all())
        self.assertTrue((features["has_breadth"] == True).all())

    def test_build_features_applies_fundamental_asof_join_and_gate(self):
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 180,
                "date": pd.date_range("2026-01-01", periods=180, freq="D"),
                "open": [100 + idx for idx in range(180)],
                "high": [101 + idx for idx in range(180)],
                "low": [99 + idx for idx in range(180)],
                "close": [100 + idx for idx in range(180)],
                "adjusted_close": [100 + idx for idx in range(180)],
                "volume": [1000 + idx for idx in range(180)],
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=180, freq="D"),
                "us10y": [4.0] * 180,
                "yield_spread": [1.0] * 180,
                "vix_close": [20.0] * 180,
                "credit_spread_hy": [3.0] * 180,
            }
        )
        breadth_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=180, freq="D"),
                "nh_nl_index": [0.2] * 180,
                "ma200_pct": [0.7] * 180,
            }
        )
        fundamentals_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "date": f"2025-{quarter:02d}-28",
                    "filing_date": filing_date,
                    "revenue": 100.0 * index,
                    "net_income": 10.0 * index,
                    "equity": 50.0 * index,
                    "eps": 1.0 * index,
                    "total_liabilities": 20.0 * index,
                }
                for index, (quarter, filing_date) in enumerate(
                    [
                        (1, "2026-01-10"),
                        (2, "2026-01-20"),
                        (3, "2026-01-30"),
                        (4, "2026-02-09"),
                        (5, "2026-02-19"),
                        (6, "2026-03-01"),
                        (7, "2026-03-11"),
                        (8, "2026-03-21"),
                        (9, "2026-03-31"),
                    ],
                    start=1,
                )
            ]
        )

        features = build_features(
            price_df=price_df,
            macro_df=macro_df,
            breadth_df=breadth_df,
            fundamentals_df=fundamentals_df,
            timeframe="1D",
        )

        gated_row = features.loc[features["date"] == pd.Timestamp("2026-03-20")].iloc[0]
        released_row = features.loc[features["date"] == pd.Timestamp("2026-03-21")].iloc[0]
        latest_row = features.loc[features["date"] == pd.Timestamp("2026-04-10")].iloc[0]

        self.assertEqual(gated_row["revenue"], 0.0)
        self.assertEqual(gated_row["roe"], 0.0)
        self.assertFalse(gated_row["has_fundamentals"])
        self.assertEqual(released_row["revenue"], 800.0)
        self.assertEqual(released_row["net_income"], 80.0)
        self.assertEqual(released_row["equity"], 400.0)
        self.assertEqual(released_row["eps"], 8.0)
        self.assertAlmostEqual(released_row["roe"], 0.2)
        self.assertAlmostEqual(released_row["debt_ratio"], 0.4)
        self.assertTrue(released_row["has_fundamentals"])
        self.assertEqual(latest_row["revenue"], 900.0)
        self.assertEqual(latest_row["eps"], 9.0)
        self.assertTrue(latest_row["has_fundamentals"])

    def test_build_features_sets_zero_and_flag_when_fundamentals_missing(self):
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 140,
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "open": [100 + idx for idx in range(140)],
                "high": [101 + idx for idx in range(140)],
                "low": [99 + idx for idx in range(140)],
                "close": [100 + idx for idx in range(140)],
                "adjusted_close": [100 + idx for idx in range(140)],
                "volume": [1000 + idx for idx in range(140)],
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "us10y": [4.0] * 140,
                "yield_spread": [1.0] * 140,
                "vix_close": [20.0] * 140,
                "credit_spread_hy": [3.0] * 140,
            }
        )
        breadth_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=140, freq="D"),
                "nh_nl_index": [0.2] * 140,
                "ma200_pct": [0.7] * 140,
            }
        )

        features = build_features(price_df=price_df, macro_df=macro_df, breadth_df=breadth_df, timeframe="1D")

        self.assertFalse(features.empty)
        self.assertTrue((features["revenue"] == 0.0).all())
        self.assertTrue((features["net_income"] == 0.0).all())
        self.assertTrue((features["equity"] == 0.0).all())
        self.assertTrue((features["eps"] == 0.0).all())
        self.assertTrue((features["roe"] == 0.0).all())
        self.assertTrue((features["debt_ratio"] == 0.0).all())
        self.assertTrue((features["has_fundamentals"] == False).all())

    def test_build_features_preserves_rows_when_macro_or_breadth_missing(self):
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 220,
                "date": pd.date_range("2026-01-01", periods=220, freq="D"),
                "open": [100 + idx for idx in range(220)],
                "high": [101 + idx for idx in range(220)],
                "low": [99 + idx for idx in range(220)],
                "close": [100 + idx for idx in range(220)],
                "adjusted_close": [100 + idx for idx in range(220)],
                "volume": [1000 + idx for idx in range(220)],
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": pd.to_datetime([]),
                "us10y": pd.Series(dtype=float),
                "yield_spread": pd.Series(dtype=float),
                "vix_close": pd.Series(dtype=float),
                "credit_spread_hy": pd.Series(dtype=float),
            }
        )
        breadth_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-04-01", periods=90, freq="D"),
                "nh_nl_index": [0.2] * 90,
                "ma200_pct": [0.7] * 90,
            }
        )

        features = build_features(price_df=price_df, macro_df=macro_df, breadth_df=breadth_df, timeframe="1D")

        self.assertFalse(features.empty)
        self.assertLess(features["date"].min(), pd.Timestamp("2026-04-01"))
        self.assertTrue((features["credit_spread_hy"] == 0.0).all())
        self.assertTrue((features["has_macro"] == False).any())
        self.assertTrue((features["has_breadth"] == False).any())


if __name__ == "__main__":
    unittest.main()
