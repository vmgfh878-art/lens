import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.preprocessing import (
    SequenceDatasetBundle,
    _source_aware_feature_index_from_frames,
    build_dataset_plan,
    build_cache_manifest_payload,
    fetch_feature_index_frame,
    is_cache_manifest_valid,
    resolve_cache_manifest_path,
    resolve_data_fingerprint,
    resolve_feature_cache_path,
    split_sequence_dataset_by_plan,
    write_cache_manifest,
)


class PreprocessingCacheIsolationTestCase(unittest.TestCase):
    def _fake_fetch_frame(self, table, **kwargs):
        if table == "indicators":
            return pd.DataFrame(
                {
                    "date": ["2026-01-02", "2026-01-05"],
                }
            )
        if table == "price_data":
            return pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL"],
                    "date": ["2026-01-02", "2026-01-05"],
                    "open": [100.0, 101.0],
                    "high": [102.0, 103.0],
                    "low": [99.0, 100.0],
                    "close": [101.0, 102.0],
                    "adjusted_close": [101.0, 102.0],
                    "volume": [1000, 1100],
                    "created_at": ["2026-01-06T00:00:00Z", "2026-01-06T00:00:00Z"],
                }
            )
        return pd.DataFrame()

    def test_provider_changes_source_data_hash_and_cache_path(self):
        with patch("ai.preprocessing._postgres_engine", side_effect=RuntimeError("no db")), patch(
            "ai.preprocessing.fetch_frame",
            side_effect=self._fake_fetch_frame,
        ), patch.dict(os.environ, {"MARKET_DATA_PROVIDER": "eodhd", "LENS_USE_LOCAL_SNAPSHOTS": "0"}, clear=False):
            eodhd_hash = resolve_data_fingerprint("1D", tickers=["AAPL"])
            eodhd_path = resolve_feature_cache_path(timeframe="1D", data_hash=eodhd_hash, tickers=["AAPL"])

        with patch("ai.preprocessing._postgres_engine", side_effect=RuntimeError("no db")), patch(
            "ai.preprocessing.fetch_frame",
            side_effect=self._fake_fetch_frame,
        ), patch.dict(os.environ, {"MARKET_DATA_PROVIDER": "yfinance", "LENS_USE_LOCAL_SNAPSHOTS": "0"}, clear=False):
            yfinance_hash = resolve_data_fingerprint("1D", tickers=["AAPL"])
            yfinance_path = resolve_feature_cache_path(timeframe="1D", data_hash=yfinance_hash, tickers=["AAPL"])

        self.assertNotEqual(eodhd_hash, yfinance_hash)
        self.assertNotEqual(eodhd_path.name, yfinance_path.name)

    def test_indicator_value_change_changes_source_hash_and_rejects_manifest(self):
        def fake_fetch_frame_factory(indicator_value):
            def fake_fetch_frame(table, **kwargs):
                if table == "indicators":
                    return pd.DataFrame(
                        {
                            "ticker": ["AAPL", "AAPL"],
                            "date": ["2026-01-02", "2026-01-05"],
                            "timeframe": ["1D", "1D"],
                            "source": ["yfinance", "yfinance"],
                            "provider": ["yfinance", "yfinance"],
                            "log_return": [0.01, indicator_value],
                            "open_ratio": [0.0, 0.0],
                        }
                    )
                if table == "price_data":
                    return pd.DataFrame(
                        {
                            "ticker": ["AAPL", "AAPL"],
                            "date": ["2026-01-02", "2026-01-05"],
                            "open": [100.0, 101.0],
                            "high": [102.0, 103.0],
                            "low": [99.0, 100.0],
                            "close": [101.0, 102.0],
                            "adjusted_close": [101.0, 102.0],
                            "volume": [1000, 1100],
                            "source": ["yfinance", "yfinance"],
                            "provider": ["yfinance", "yfinance"],
                            "updated_at": ["2026-01-06T00:00:00Z", "2026-01-06T00:00:00Z"],
                        }
                    )
                return pd.DataFrame()

            return fake_fetch_frame

        with patch("ai.preprocessing._postgres_engine", side_effect=RuntimeError("no db")), patch(
            "ai.preprocessing.fetch_frame",
            side_effect=fake_fetch_frame_factory(0.02),
        ), patch.dict(os.environ, {"LENS_USE_LOCAL_SNAPSHOTS": "0"}, clear=False):
            first_hash = resolve_data_fingerprint("1D", tickers=["AAPL"], market_data_provider="yfinance")

        with patch("ai.preprocessing._postgres_engine", side_effect=RuntimeError("no db")), patch(
            "ai.preprocessing.fetch_frame",
            side_effect=fake_fetch_frame_factory(0.03),
        ), patch.dict(os.environ, {"LENS_USE_LOCAL_SNAPSHOTS": "0"}, clear=False):
            second_hash = resolve_data_fingerprint("1D", tickers=["AAPL"], market_data_provider="yfinance")

        self.assertNotEqual(first_hash, second_hash)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "features_1D_test.pt"
            cache_path.write_bytes(b"not-used")
            write_cache_manifest(
                cache_path,
                build_cache_manifest_payload(
                    cache_kind="features",
                    timeframe="1D",
                    source_data_hash=first_hash,
                    feature_columns=["log_return"],
                    ticker_count=1,
                    date_min="2026-01-02",
                    date_max="2026-01-05",
                    provider="yfinance",
                ),
            )
            expected_manifest = build_cache_manifest_payload(
                cache_kind="features",
                timeframe="1D",
                source_data_hash=second_hash,
                feature_columns=["log_return"],
                ticker_count=1,
                date_min="2026-01-02",
                date_max="2026-01-05",
                provider="yfinance",
            )
            self.assertFalse(is_cache_manifest_valid(cache_path, expected_manifest))

    def test_manifest_provider_mismatch_rejects_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "features_1D_test.pt"
            cache_path.write_bytes(b"not-used")
            eodhd_manifest = build_cache_manifest_payload(
                cache_kind="features",
                timeframe="1D",
                source_data_hash="abc123",
                feature_columns=["log_return"],
                ticker_count=1,
                date_min="2026-01-02",
                date_max="2026-01-05",
                provider="eodhd",
            )
            write_cache_manifest(cache_path, eodhd_manifest)
            yfinance_manifest = build_cache_manifest_payload(
                cache_kind="features",
                timeframe="1D",
                source_data_hash="abc123",
                feature_columns=["log_return"],
                ticker_count=1,
                date_min="2026-01-02",
                date_max="2026-01-05",
                provider="yfinance",
            )

            self.assertFalse(is_cache_manifest_valid(cache_path, yfinance_manifest))
            self.assertTrue(resolve_cache_manifest_path(cache_path).exists())
            saved = json.loads(resolve_cache_manifest_path(cache_path).read_text(encoding="utf-8"))
            self.assertEqual(saved["provider"], "eodhd")
            self.assertEqual(saved["source"], "eodhd")

    def test_source_aware_feature_index_uses_provider_price_dates(self):
        indicator_index = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
                "timeframe": ["1D", "1D", "1D", "1D"],
                "date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-05"],
                "source": [None, "yfinance", "eodhd", "eodhd"],
            }
        )
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
                "date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-05"],
                "source": [None, "yfinance", "eodhd", "eodhd"],
            }
        )

        yfinance_index = _source_aware_feature_index_from_frames(
            indicator_index,
            price_df,
            timeframe="1D",
            provider="yfinance",
        )
        eodhd_index = _source_aware_feature_index_from_frames(
            indicator_index,
            price_df,
            timeframe="1D",
            provider="eodhd",
        )

        self.assertEqual(yfinance_index["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-05"])
        self.assertEqual(
            eodhd_index[["ticker", "date"]].assign(date=eodhd_index["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
            [
                {"ticker": "AAPL", "date": "2026-01-02"},
                {"ticker": "AAPL", "date": "2026-01-06"},
                {"ticker": "MSFT", "date": "2026-01-05"},
            ],
        )

    def test_1w_feature_index_uses_resampled_friday_label(self):
        indicator_index = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "timeframe": ["1W"],
                "date": ["2026-07-03"],
                "source": ["yfinance"],
                "provider": ["yfinance"],
            }
        )
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "date": ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"],
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "adjusted_close": [100.5, 101.5, 102.5, 103.5],
                "volume": [1000, 1000, 1000, 1000],
                "source": ["yfinance"] * 4,
                "provider": ["yfinance"] * 4,
            }
        )

        index_frame = _source_aware_feature_index_from_frames(
            indicator_index,
            price_df,
            timeframe="1W",
            provider="yfinance",
        )

        self.assertEqual(index_frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-03"])
        self.assertEqual(index_frame.attrs["alignment_audit"]["row_loss_count"], 0)
        self.assertEqual(index_frame.attrs["alignment_audit"]["join_basis"], "resampled_price_label_date")

    def test_1m_feature_index_uses_month_end_label_when_weekend(self):
        indicator_index = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "timeframe": ["1M"],
                "date": ["2026-02-28"],
                "source": ["yfinance"],
                "provider": ["yfinance"],
            }
        )
        price_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": ["2026-02-26", "2026-02-27"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "adjusted_close": [101.0, 102.0],
                "volume": [1000, 1100],
                "source": ["yfinance", "yfinance"],
                "provider": ["yfinance", "yfinance"],
            }
        )

        index_frame = _source_aware_feature_index_from_frames(
            indicator_index,
            price_df,
            timeframe="1M",
            provider="yfinance",
        )

        self.assertEqual(index_frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-02-28"])
        self.assertEqual(index_frame.attrs["alignment_audit"]["row_loss_count"], 0)

    def test_local_snapshot_feature_index_does_not_call_supabase(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir) / "snapshots"
            cache_dir = Path(tmp_dir) / "cache"
            snapshot_dir.mkdir()
            price_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL"],
                    "date": ["2026-01-02", "2026-01-05"],
                    "source": ["yfinance", "yfinance"],
                    "provider": ["yfinance", "yfinance"],
                    "open": [100.0, 101.0],
                    "high": [102.0, 103.0],
                    "low": [99.0, 100.0],
                    "close": [101.0, 102.0],
                    "adjusted_close": [101.0, 102.0],
                    "volume": [1000, 1100],
                    "updated_at": ["2026-01-06T00:00:00Z", "2026-01-06T00:00:00Z"],
                }
            )
            indicator_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL"],
                    "timeframe": ["1D", "1D"],
                    "date": ["2026-01-02", "2026-01-05"],
                    "source": ["yfinance", "yfinance"],
                    "provider": ["yfinance", "yfinance"],
                    "log_return": [0.0, 0.01],
                    "open_ratio": [0.0, 0.001],
                }
            )
            try:
                price_frame.to_parquet(snapshot_dir / "price_data_yfinance.parquet", index=False)
                indicator_frame.to_parquet(snapshot_dir / "indicators_yfinance_1D.parquet", index=False)
            except Exception as exc:
                self.skipTest(f"parquet 엔진을 사용할 수 없어 local snapshot 테스트를 건너뜁니다: {exc}")

            with patch.dict(
                os.environ,
                {
                    "LENS_DATA_BACKEND": "local",
                    "LENS_LOCAL_SNAPSHOT_DIR": str(snapshot_dir),
                    "MARKET_DATA_PROVIDER": "yfinance",
                },
                clear=False,
            ), patch("ai.preprocessing.CACHE_DIR", cache_dir), patch(
                "ai.preprocessing.fetch_frame",
                side_effect=AssertionError("Supabase REST 조회 금지"),
            ), patch(
                "ai.preprocessing._postgres_engine",
                side_effect=AssertionError("Postgres 조회 금지"),
            ):
                index_frame = fetch_feature_index_frame(
                    timeframe="1D",
                    tickers=["AAPL"],
                    market_data_provider="yfinance",
                )

            self.assertEqual(index_frame["ticker"].tolist(), ["AAPL", "AAPL"])
            self.assertEqual(index_frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02", "2026-01-05"])

    def test_local_snapshot_feature_index_uses_timeframe_price_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir) / "snapshots"
            cache_dir = Path(tmp_dir) / "cache"
            snapshot_dir.mkdir()
            daily_price_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": ["2026-01-30"],
                    "source": ["yfinance"],
                    "provider": ["yfinance"],
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "adjusted_close": [101.0],
                    "volume": [1000],
                    "updated_at": ["2026-02-01T00:00:00Z"],
                }
            )
            monthly_price_frame = daily_price_frame.copy()
            monthly_price_frame["date"] = ["2026-01-31"]
            indicator_frame = pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "timeframe": ["1M"],
                    "date": ["2026-01-31"],
                    "source": ["yfinance"],
                    "provider": ["yfinance"],
                    "log_return": [0.0],
                    "open_ratio": [0.0],
                }
            )
            try:
                daily_price_frame.to_parquet(snapshot_dir / "price_data_yfinance.parquet", index=False)
                monthly_price_frame.to_parquet(snapshot_dir / "price_data_yfinance_1M.parquet", index=False)
                indicator_frame.to_parquet(snapshot_dir / "indicators_yfinance_1M.parquet", index=False)
            except Exception as exc:
                self.skipTest(f"parquet 엔진을 사용할 수 없어 local snapshot 테스트를 건너뜁니다: {exc}")

            with patch.dict(
                os.environ,
                {
                    "LENS_DATA_BACKEND": "local",
                    "LENS_LOCAL_SNAPSHOT_DIR": str(snapshot_dir),
                    "MARKET_DATA_PROVIDER": "yfinance",
                },
                clear=False,
            ), patch("ai.preprocessing.CACHE_DIR", cache_dir), patch(
                "ai.preprocessing.fetch_frame",
                side_effect=AssertionError("Supabase REST 조회 금지"),
            ), patch(
                "ai.preprocessing._postgres_engine",
                side_effect=AssertionError("Postgres 조회 금지"),
            ):
                index_frame = fetch_feature_index_frame(
                    timeframe="1M",
                    tickers=["AAPL"],
                    market_data_provider="yfinance",
                )

            self.assertEqual(index_frame["ticker"].tolist(), ["AAPL"])
            self.assertEqual(index_frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-31"])

    def test_dataset_plan_records_provider_and_source(self):
        index_frame = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 150,
                "timeframe": ["1D"] * 150,
                "date": pd.bdate_range("2025-01-02", periods=150),
            }
        )

        plan = build_dataset_plan(
            index_frame,
            timeframe="1D",
            seq_len=60,
            horizon=5,
            min_fold_samples=1,
            market_data_provider="yfinance",
            source_data_hash="hash123",
        )

        self.assertEqual(plan.provider, "yfinance")
        self.assertEqual(plan.source, "yfinance")
        self.assertEqual(plan.source_data_hash, "hash123")
        self.assertEqual(plan.date_min, "2025-01-02")
        self.assertGreater(plan.estimated_usable_sample_count, 0)

    def test_empty_split_error_includes_source_diagnostics(self):
        bundle = SequenceDatasetBundle(
            features=torch.zeros(1, 2, 1),
            line_targets=torch.zeros(1, 1),
            band_targets=torch.zeros(1, 1),
            raw_future_returns=torch.zeros(1, 1),
            anchor_closes=torch.ones(1),
            ticker_ids=torch.zeros(1, dtype=torch.long),
            future_covariates=torch.zeros(1, 1, 0),
            metadata=pd.DataFrame({"ticker": ["AAPL"], "asof_date": ["2026-01-02"], "sample_index": [0]}),
        )

        with self.assertRaises(ValueError) as raised:
            split_sequence_dataset_by_plan(
                bundle,
                split_specs={},
                diagnostics={
                    "provider": "yfinance",
                    "source": "yfinance",
                    "date_min": "2026-01-02",
                    "date_max": "2026-01-02",
                    "seq_len": 60,
                    "horizon": 5,
                    "gap": 20,
                },
            )

        payload = json.loads(str(raised.exception))
        self.assertEqual(payload["provider"], "yfinance")
        self.assertEqual(payload["source"], "yfinance")
        self.assertEqual(payload["error"], "empty_split_result")
        self.assertEqual(payload["actual_usable_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
