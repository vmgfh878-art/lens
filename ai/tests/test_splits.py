import unittest

import pandas as pd

from ai.splits import eligible_tickers, make_splits


def _ticker_frame(ticker: str, timeframe: str, row_count: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=row_count, freq="D")
    return pd.DataFrame(
        {
            "ticker": [ticker] * row_count,
            "timeframe": [timeframe] * row_count,
            "date": dates,
        }
    )


class SplitLogicTestCase(unittest.TestCase):
    def test_make_splits_rejects_gate_a_when_rows_too_short(self):
        frame = _ticker_frame("AAPL", "1D", 260)
        with self.assertRaisesRegex(ValueError, "Gate A"):
            make_splits(frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

    def test_make_splits_rejects_gate_b_and_gate_c_at_thresholds(self):
        gate_b_frame = _ticker_frame("AAPL", "1D", 381)
        with self.assertRaisesRegex(ValueError, "Gate B"):
            make_splits(gate_b_frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

        gate_c_frame = _ticker_frame("AAPL", "1D", 511)
        with self.assertRaisesRegex(ValueError, "Gate C"):
            make_splits(gate_c_frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

    def test_make_splits_keeps_gap_between_folds(self):
        frame = _ticker_frame("AAPL", "1D", 700)
        spec = make_splits(frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

        self.assertGreaterEqual(spec.val.start - spec.train.end, 20)
        self.assertGreaterEqual(spec.test.start - spec.val.end, 20)
        self.assertGreater(spec.train.count, 0)
        self.assertGreater(spec.val.count, 0)
        self.assertGreater(spec.test.count, 0)

    def test_timeframe_filters_are_independent(self):
        feature_df = pd.concat(
            [
                _ticker_frame("AAPL", "1D", 800),
                _ticker_frame("MSFT", "1D", 700),
                _ticker_frame("GOOG", "1D", 381),
                _ticker_frame("AAPL", "1W", 600),
                _ticker_frame("MSFT", "1W", 430),
                _ticker_frame("GOOG", "1W", 520),
            ],
            ignore_index=True,
        )

        eligible_1d, _ = eligible_tickers(feature_df, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)
        eligible_1w, _ = eligible_tickers(feature_df, timeframe="1W", seq_len=104, h_max=12, min_fold_samples=50)

        self.assertEqual(eligible_1d, ["AAPL", "MSFT"])
        self.assertEqual(eligible_1w, ["AAPL", "GOOG"])


if __name__ == "__main__":
    unittest.main()
