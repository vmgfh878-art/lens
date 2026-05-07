import unittest

import pandas as pd

from ai.splits import TIMEFRAME_ABSOLUTE_MIN_ROWS, build_split_specs, eligible_tickers, make_splits, required_history_rows


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
        gate_b_frame = _ticker_frame("AAPL", "1W", 100)
        with self.assertRaisesRegex(ValueError, "Gate B"):
            make_splits(gate_b_frame, timeframe="1W", seq_len=20, h_max=4, min_fold_samples=50)

        gate_c_frame = _ticker_frame("AAPL", "1W", 300)
        with self.assertRaisesRegex(ValueError, "Gate C"):
            make_splits(gate_c_frame, timeframe="1W", seq_len=20, h_max=4, min_fold_samples=50)

    def test_make_splits_keeps_gap_between_folds(self):
        frame = _ticker_frame("AAPL", "1D", 700)
        spec = make_splits(frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

        self.assertGreaterEqual(spec.val.start - spec.train.end, 20)
        self.assertGreaterEqual(spec.test.start - spec.val.end, 20)
        self.assertGreater(spec.train.count, 0)
        self.assertGreater(spec.val.count, 0)
        self.assertGreater(spec.test.count, 0)

    def test_make_splits_rejects_1d_absolute_history_below_450(self):
        frame = _ticker_frame("AAPL", "1D", 300)

        with self.assertRaisesRegex(ValueError, "insufficient_absolute_history_1D_min_450"):
            make_splits(frame, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=1)

    def test_make_splits_rejects_1w_absolute_history_below_78(self):
        frame = _ticker_frame("AAPL", "1W", 75)

        with self.assertRaisesRegex(ValueError, "insufficient_absolute_history_1W_min_78"):
            make_splits(frame, timeframe="1W", seq_len=60, h_max=12, min_fold_samples=1)

    def test_absolute_history_gate_keeps_sufficient_ticker_eligible(self):
        feature_df = pd.concat(
            [
                _ticker_frame("AAPL", "1D", 700),
                _ticker_frame("MSFT", "1D", 300),
            ],
            ignore_index=True,
        )

        eligible, excluded = eligible_tickers(feature_df, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)

        self.assertEqual(eligible, ["AAPL"])
        self.assertEqual(excluded["MSFT"], "insufficient_absolute_history_1D_min_450")

    def test_absolute_history_exclusion_reason_is_kept_in_split_specs(self):
        feature_df = pd.concat(
            [
                _ticker_frame("AAPL", "1W", 120),
                _ticker_frame("MSFT", "1W", 75),
            ],
            ignore_index=True,
        )

        specs, excluded = build_split_specs(feature_df, timeframe="1W", seq_len=60, h_max=12, min_fold_samples=1)

        self.assertIn("AAPL", specs)
        self.assertEqual(excluded["MSFT"], "insufficient_absolute_history_1W_min_78")

    def test_absolute_history_required_rows_are_separate_from_sequence_requirement(self):
        self.assertEqual(required_history_rows("1D", seq_len=252, h_max=20), 450)
        self.assertEqual(required_history_rows("1W", seq_len=60, h_max=12), 78)
        self.assertEqual(required_history_rows("1M", seq_len=24, h_max=3), 27)
        self.assertNotIn("1M", TIMEFRAME_ABSOLUTE_MIN_ROWS)

    def test_timeframe_filters_are_independent(self):
        feature_df = pd.concat(
            [
                _ticker_frame("AAPL", "1D", 800),
                _ticker_frame("MSFT", "1D", 700),
                _ticker_frame("GOOG", "1D", 381),
                _ticker_frame("AAPL", "1W", 600),
                _ticker_frame("MSFT", "1W", 430),
                _ticker_frame("GOOG", "1W", 520),
                _ticker_frame("AAPL", "1M", 140),
                _ticker_frame("MSFT", "1M", 40),
            ],
            ignore_index=True,
        )

        eligible_1d, _ = eligible_tickers(feature_df, timeframe="1D", seq_len=252, h_max=20, min_fold_samples=50)
        eligible_1w, _ = eligible_tickers(feature_df, timeframe="1W", seq_len=104, h_max=12, min_fold_samples=50)
        eligible_1m, _ = eligible_tickers(feature_df, timeframe="1M", seq_len=24, h_max=3, min_fold_samples=5)

        self.assertEqual(eligible_1d, ["AAPL", "MSFT"])
        self.assertEqual(eligible_1w, ["AAPL", "GOOG"])
        self.assertEqual(eligible_1m, ["AAPL"])


if __name__ == "__main__":
    unittest.main()
