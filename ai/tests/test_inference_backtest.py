import unittest

import pandas as pd
import torch

from ai.backtest import run_rule_based_backtest
from ai.inference import decode_return_forecasts


class InferenceBacktestTestCase(unittest.TestCase):
    def test_decode_return_forecasts_converts_returns_to_prices(self):
        line, lower, upper = decode_return_forecasts(
            line_returns=torch.tensor([[0.1, 0.2]]),
            lower_returns=torch.tensor([[0.0, 0.05]]),
            upper_returns=torch.tensor([[0.2, 0.3]]),
            anchor_closes=torch.tensor([100.0]),
        )

        self.assertAlmostEqual(line[0][0], 110.0, places=4)
        self.assertAlmostEqual(line[0][1], 120.0, places=4)
        self.assertAlmostEqual(lower[0][0], 100.0, places=4)
        self.assertAlmostEqual(lower[0][1], 105.0, places=4)
        self.assertAlmostEqual(upper[0][0], 120.0, places=4)
        self.assertAlmostEqual(upper[0][1], 130.0, places=4)

    def test_rule_based_backtest_uses_realized_return(self):
        frame = pd.DataFrame(
            {
                "asof_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
                "signal": ["BUY", "SELL", "HOLD"],
                "realized_return": [0.1, -0.05, 0.02],
                "line_return": [0.08, -0.03, 0.01],
            }
        )

        result = run_rule_based_backtest(frame)

        self.assertEqual(result["num_trades"], 2)
        self.assertGreater(result["return_pct"], 0.0)
        self.assertGreaterEqual(result["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
