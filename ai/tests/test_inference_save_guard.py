import unittest

from ai.inference_save_guard import is_product_model_run, resolve_inference_save_contract
from ai.storage import STORAGE_CONTRACT_EVALUATION_BULK, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY


class InferenceSaveGuardTestCase(unittest.TestCase):
    def test_default_save_blocks_bulk_prediction_storage(self):
        with self.assertRaisesRegex(ValueError, "bulk predictions"):
            resolve_inference_save_contract(
                save=True,
                save_product_latest_only=False,
                allow_bulk_evaluation_save=False,
                model_run={"run_id": "run-1"},
                config={},
            )

    def test_explicit_bulk_flag_allows_evaluation_bulk_contract(self):
        contract = resolve_inference_save_contract(
            save=True,
            save_product_latest_only=False,
            allow_bulk_evaluation_save=True,
            model_run={"run_id": "run-1"},
            config={},
        )

        self.assertEqual(contract, STORAGE_CONTRACT_EVALUATION_BULK)

    def test_product_latest_flag_uses_product_contract(self):
        contract = resolve_inference_save_contract(
            save=True,
            save_product_latest_only=True,
            allow_bulk_evaluation_save=False,
            model_run={"run_id": "run-1"},
            config={"role": "line_model"},
        )

        self.assertEqual(contract, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY)

    def test_product_run_cannot_use_bulk_contract(self):
        with self.assertRaisesRegex(ValueError, "제품 run"):
            resolve_inference_save_contract(
                save=True,
                save_product_latest_only=False,
                allow_bulk_evaluation_save=True,
                model_run={"run_id": "run-1"},
                config={"role": "band_model"},
            )

    def test_conflicting_save_flags_raise(self):
        with self.assertRaisesRegex(ValueError, "동시에"):
            resolve_inference_save_contract(
                save=True,
                save_product_latest_only=True,
                allow_bulk_evaluation_save=True,
                model_run={"run_id": "run-1"},
                config={},
            )

    def test_product_run_detection_uses_config_contract_and_flags(self):
        self.assertTrue(is_product_model_run({"run_id": "run-1"}, {"storage_contract": "product_latest_only"}))
        self.assertTrue(is_product_model_run({"run_id": "run-1"}, {"final_product_candidate": True}))
        self.assertTrue(is_product_model_run({"run_id": "run-1", "product_candidate": True}, {}))
        self.assertFalse(is_product_model_run({"run_id": "run-1"}, {}))


if __name__ == "__main__":
    unittest.main()
