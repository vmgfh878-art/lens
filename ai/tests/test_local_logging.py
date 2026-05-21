import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ai.local_logging import LocalTrainingProgressLogger, sanitize_for_json
from ai.train import DEFAULT_LOCAL_LOG_DIR, parse_args


class LocalTrainingProgressLoggerTestCase(unittest.TestCase):
    def test_writes_config_metrics_and_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LocalTrainingProgressLogger(run_id="run-1", base_dir=tmpdir)

            logger.write_config({"run_id": "run-1", "tensor_scalar": torch.tensor(2.0)})
            logger.write_epoch(
                {
                    "run_id": "run-1",
                    "epoch": 1,
                    "train_total_loss": np.float32(0.25),
                    "nan_value": float("nan"),
                }
            )
            logger.write_summary({"run_id": "run-1", "status": "completed", "value": torch.tensor(3.0)})

            config_payload = json.loads((Path(tmpdir) / "run-1" / "config.json").read_text(encoding="utf-8"))
            metric_lines = (Path(tmpdir) / "run-1" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            summary_payload = json.loads((Path(tmpdir) / "run-1" / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(config_payload["tensor_scalar"], 2.0)
        self.assertEqual(len(metric_lines), 1)
        metric_payload = json.loads(metric_lines[0])
        self.assertEqual(metric_payload["train_total_loss"], 0.25)
        self.assertIsNone(metric_payload["nan_value"])
        self.assertEqual(summary_payload["status"], "completed")
        self.assertEqual(summary_payload["value"], 3.0)

    def test_sanitize_numpy_and_torch_values(self):
        payload = sanitize_for_json(
            {
                "np_scalar": np.float64(1.5),
                "torch_scalar": torch.tensor(4.0),
                "torch_vector": torch.tensor([1.0, float("nan")]),
            }
        )

        self.assertEqual(payload["np_scalar"], 1.5)
        self.assertEqual(payload["torch_scalar"], 4.0)
        self.assertEqual(payload["torch_vector"], [1.0, None])

    def test_logging_failure_warns_without_raising(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blocked_base = Path(tmpdir) / "not_a_directory"
            blocked_base.write_text("blocked", encoding="utf-8")
            warnings: list[str] = []

            logger = LocalTrainingProgressLogger(run_id="run-1", base_dir=blocked_base, warn=warnings.append)
            logger.write_epoch({"run_id": "run-1", "epoch": 1})
            logger.write_summary({"run_id": "run-1", "status": "completed"})

        self.assertFalse(logger.active)
        self.assertEqual(len(warnings), 1)
        self.assertIn("local training log", warnings[0])

    def test_cli_local_log_defaults_and_disable_option(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()
        self.assertTrue(args.local_log)
        self.assertEqual(args.local_log_dir, DEFAULT_LOCAL_LOG_DIR)

        with patch.object(sys, "argv", ["train.py", "--no-local-log", "--local-log-dir", "tmp/runs"]):
            args = parse_args()
        self.assertFalse(args.local_log)
        self.assertEqual(args.local_log_dir, "tmp/runs")


if __name__ == "__main__":
    unittest.main()
