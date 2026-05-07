import subprocess
import sys
import textwrap
import unittest


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TorchImportContractTest(unittest.TestCase):
    def test_bootstrap_loads_torch_before_numpy_pandas(self):
        result = _run_python(
            """
            import os
            import sys

            from ai.torch_bootstrap import bootstrap_torch

            torch = bootstrap_torch()
            import numpy  # noqa: F401
            import pandas  # noqa: F401

            assert "torch" in sys.modules
            assert os.environ["PYTHONUTF8"] == "1"
            assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
            assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
            assert os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
            assert torch.__name__ == "torch"
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_train_and_inference_import_after_bootstrap(self):
        result = _run_python(
            """
            import ai.train  # noqa: F401
            import ai.inference  # noqa: F401
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_data_only_cp103_import_does_not_import_torch(self):
        result = _run_python(
            """
            import sys

            import scripts.cp103_yfinance_new_day_append_gate  # noqa: F401

            assert "torch" not in sys.modules
            assert "scripts.cp99_1d_product_loop_thin_upload" not in sys.modules
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_data_only_cp115_import_does_not_import_torch(self):
        result = _run_python(
            """
            import sys

            import scripts.cp115_yfinance_completed_day_append  # noqa: F401

            assert "torch" not in sys.modules
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
