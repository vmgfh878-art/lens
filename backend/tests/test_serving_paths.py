from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from app.services.serving_paths import (
    activate_serving_data_dir,
    configured_serving_data_dir,
    get_serving_bootstrap_dir,
    get_serving_cache_dir,
    get_serving_data_dir,
    reset_active_serving_data_dir,
    serving_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def setup_function() -> None:
    reset_active_serving_data_dir()


def teardown_function() -> None:
    reset_active_serving_data_dir()


def test_default_paths_keep_existing_layout() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert configured_serving_data_dir() == (REPOSITORY_ROOT / "backend/data/v1").resolve()
        assert (
            get_serving_bootstrap_dir() == (REPOSITORY_ROOT / "backend/data/bootstrap/v1").resolve()
        )
        assert get_serving_cache_dir() == (REPOSITORY_ROOT / "backend/data/runtime/v1").resolve()


def test_relative_config_paths_use_repository_root() -> None:
    with patch.dict(
        os.environ,
        {
            "LENS_SERVING_DATA_DIR": "runtime/serving/current",
            "LENS_SERVING_BOOTSTRAP_DIR": "backend/data/bootstrap/custom",
            "LENS_SERVING_CACHE_DIR": "runtime/cache",
        },
        clear=True,
    ):
        assert get_serving_data_dir() == (REPOSITORY_ROOT / "runtime/serving/current").resolve()
        assert (
            get_serving_bootstrap_dir()
            == (REPOSITORY_ROOT / "backend/data/bootstrap/custom").resolve()
        )
        assert get_serving_cache_dir() == (REPOSITORY_ROOT / "runtime/cache").resolve()


def test_runtime_activation_overrides_config_until_reset() -> None:
    with patch.dict(
        os.environ,
        {"LENS_SERVING_DATA_DIR": "runtime/serving/configured"},
        clear=True,
    ):
        active = activate_serving_data_dir("runtime/serving/version-1")
        assert get_serving_data_dir() == active
        assert serving_file("market_prices_1d.parquet") == (active / "market_prices_1d.parquet")

        reset_active_serving_data_dir()
        assert get_serving_data_dir() == (REPOSITORY_ROOT / "runtime/serving/configured").resolve()
