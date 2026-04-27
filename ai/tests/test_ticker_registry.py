import tempfile
import unittest
from pathlib import Path

from ai.ticker_registry import build_registry, load_registry, lookup_id, save_registry


class TickerRegistryTestCase(unittest.TestCase):
    def test_ticker_registry_alphabetical(self):
        registry = build_registry(["MSFT", "AAPL", "GOOG"], "1D")
        self.assertEqual(list(registry["mapping"].keys()), ["AAPL", "GOOG", "MSFT"])
        self.assertEqual(registry["mapping"]["AAPL"], 0)
        self.assertEqual(registry["mapping"]["GOOG"], 1)
        self.assertEqual(registry["mapping"]["MSFT"], 2)

    def test_ticker_registry_oov_returns_num_tickers(self):
        registry = build_registry(["AAPL", "MSFT"], "1D")
        self.assertEqual(lookup_id("NVDA", registry), registry["num_tickers"])

    def test_ticker_registry_save_load_roundtrip(self):
        registry = build_registry(["AAPL", "MSFT"], "1D")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ticker_map.json"
            save_registry(registry, "1D", path)
            loaded = load_registry("1D", path)
        self.assertEqual(registry, loaded)
        self.assertEqual(lookup_id("AAPL", loaded), 0)

    def test_ticker_registry_1d_and_1w_independent(self):
        registry_1d = build_registry(["AAPL", "MSFT", "NVDA"], "1D")
        registry_1w = build_registry(["AAPL", "MSFT"], "1W")
        self.assertEqual(registry_1d["num_tickers"], 3)
        self.assertEqual(registry_1w["num_tickers"], 2)
        self.assertEqual(lookup_id("NVDA", registry_1w), registry_1w["num_tickers"])


if __name__ == "__main__":
    unittest.main()
