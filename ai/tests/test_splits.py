import unittest

import pandas as pd

from ai.splits import (
    TIMEFRAME_ABSOLUTE_MIN_ROWS,
    build_split_specs,
    eligible_tickers,
    make_calendar_split_date_plan,
    make_splits,
    required_history_rows,
)


def _ticker_frame(ticker: str, timeframe: str, row_count: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=row_count, freq="D")
    return pd.DataFrame(
        {
            "ticker": [ticker] * row_count,
            "timeframe": [timeframe] * row_count,
            "date": dates,
        }
    )


def _sample_metadata(tickers: list[str], date_count: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=date_count).strftime("%Y-%m-%d").tolist()
    rows = []
    for ticker in tickers:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "asof_date": date,
                    "sample_index": index,
                }
            )
    return pd.DataFrame(rows)


def _legacy_split_dates(ticker: str, start: str) -> tuple[set[str], set[str], set[str]]:
    seq_len = 5
    h_max = 2
    row_count = 500
    frame = pd.DataFrame(
        {
            "ticker": [ticker] * row_count,
            "timeframe": ["1D"] * row_count,
            "date": pd.date_range(start, periods=row_count, freq="D"),
        }
    )
    spec = make_splits(frame, timeframe="1D", seq_len=seq_len, h_max=h_max, min_fold_samples=1)
    date_strings = frame["date"].dt.strftime("%Y-%m-%d").tolist()

    def _dates_for_slice(start_index: int, end_index: int) -> set[str]:
        return {
            date_strings[seq_len - 1 + sample_index]
            for sample_index in range(start_index, end_index)
        }

    return (
        _dates_for_slice(spec.train.start, spec.train.end),
        _dates_for_slice(spec.val.start, spec.val.end),
        _dates_for_slice(spec.test.start, spec.test.end),
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
        self.assertEqual(required_history_rows("1M", seq_len=24, h_max=3), 36)
        self.assertEqual(TIMEFRAME_ABSOLUTE_MIN_ROWS["1M"], 36)

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

    def test_calendar_split_no_cross_ticker_overlap(self):
        metadata = _sample_metadata(["A", "B", "C"], date_count=90)
        plan = make_calendar_split_date_plan(
            metadata,
            purge_gap_trading_days=3,
            min_fold_samples=1,
        )

        train_dates = set(plan.train_dates)
        validation_dates = set(plan.validation_dates)
        test_dates = set(plan.test_dates)

        self.assertFalse(train_dates & validation_dates)
        self.assertFalse(train_dates & test_dates)
        self.assertFalse(validation_dates & test_dates)
        self.assertEqual(plan.cross_split_date_overlap_count, 0)

    def test_calendar_split_order(self):
        metadata = _sample_metadata(["A", "B", "C"], date_count=90)
        plan = make_calendar_split_date_plan(
            metadata,
            purge_gap_trading_days=3,
            min_fold_samples=1,
        )

        self.assertLess(plan.train_end_date, plan.validation_start_date)
        self.assertLess(plan.validation_end_date, plan.test_start_date)

    def test_calendar_purge_gap(self):
        metadata = _sample_metadata(["A", "B", "C"], date_count=90)
        plan = make_calendar_split_date_plan(
            metadata,
            purge_gap_trading_days=4,
            min_fold_samples=1,
        )

        self.assertEqual(plan.train_validation_gap_trading_days, 4)
        self.assertEqual(plan.validation_test_gap_trading_days, 4)
        self.assertEqual(plan.purge_gap_trading_days, 4)

    def test_ticker_index_split_regression_guard(self):
        a_train, a_val, a_test = _legacy_split_dates("A", "2020-01-01")
        b_train, b_val, b_test = _legacy_split_dates("B", "2020-02-01")
        legacy_train_dates = a_train | b_train
        legacy_val_dates = a_val | b_val
        legacy_test_dates = a_test | b_test
        legacy_overlap = (
            (legacy_train_dates & legacy_val_dates)
            | (legacy_train_dates & legacy_test_dates)
            | (legacy_val_dates & legacy_test_dates)
        )
        self.assertGreater(len(legacy_overlap), 0)

        metadata = pd.concat(
            [
                pd.DataFrame(
                    {
                        "ticker": ["A"] * 494,
                        "asof_date": pd.date_range("2020-01-05", periods=494, freq="D").strftime("%Y-%m-%d"),
                        "sample_index": list(range(494)),
                    }
                ),
                pd.DataFrame(
                    {
                        "ticker": ["B"] * 494,
                        "asof_date": pd.date_range("2020-02-05", periods=494, freq="D").strftime("%Y-%m-%d"),
                        "sample_index": list(range(494)),
                    }
                ),
            ],
            ignore_index=True,
        )
        calendar_plan = make_calendar_split_date_plan(
            metadata,
            purge_gap_trading_days=2,
            min_fold_samples=1,
        )
        self.assertEqual(calendar_plan.cross_split_date_overlap_count, 0)


if __name__ == "__main__":
    unittest.main()
