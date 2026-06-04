"""CP235 — 도메인별 Settings 박제.

매 호출 재평가 계약, truthy 변형, CSV 파싱, 기본값/override 케이스.
"""

import os
import unittest
from unittest.mock import patch

from app.config import (
    get_admin_config,
    get_cache_config,
    get_cors_config,
    get_database_config,
    get_market_config,
)


class DatabaseConfigTestCase(unittest.TestCase):
    def test_default_when_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_database_config()
            self.assertIsNone(cfg.supabase_url)
            self.assertIsNone(cfg.supabase_key)
            self.assertFalse(cfg.force_local)

    def test_override(self):
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x", "SUPABASE_KEY": "k", "LENS_FORCE_LOCAL": "1"},
            clear=True,
        ):
            cfg = get_database_config()
            self.assertEqual(cfg.supabase_url, "https://x")
            self.assertEqual(cfg.supabase_key, "k")
            self.assertTrue(cfg.force_local)

    def test_force_local_truthy_variants(self):
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            (" 1 ", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
            ("y", False),  # 코드 원본은 'y' 미지원 (pydantic 기본과 다른 부분)
        ]:
            with patch.dict(os.environ, {"LENS_FORCE_LOCAL": raw}, clear=True):
                self.assertEqual(get_database_config().force_local, expected, raw)

    def test_clear_env_repeated_call_reevaluates(self):
        """import-time 싱글톤 검출: 매 호출 새로운 env 반영."""
        with patch.dict(os.environ, {"SUPABASE_URL": "first"}, clear=True):
            self.assertEqual(get_database_config().supabase_url, "first")
        with patch.dict(os.environ, {"SUPABASE_URL": "second"}, clear=True):
            self.assertEqual(get_database_config().supabase_url, "second")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_database_config().supabase_url)


class MarketConfigTestCase(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_market_config()
            self.assertEqual(cfg.market_data_provider, "yfinance")
            self.assertIsNone(cfg.local_snapshot_dir)

    def test_override(self):
        with patch.dict(
            os.environ,
            {"MARKET_DATA_PROVIDER": "eodhd", "LENS_LOCAL_SNAPSHOT_DIR": "/tmp/x"},
            clear=True,
        ):
            cfg = get_market_config()
            self.assertEqual(cfg.market_data_provider, "eodhd")
            self.assertEqual(cfg.local_snapshot_dir, "/tmp/x")


class CorsConfigTestCase(unittest.TestCase):
    def test_default_origins(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_cors_config()
            self.assertEqual(
                cfg.origins,
                [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "https://lens-kimjihyeong-s-projects.vercel.app",
                    "https://lens-ten-delta.vercel.app",
                ],
            )
            self.assertEqual(cfg.origin_regex, r"^https://lens(?:-[a-z0-9-]+)?\.vercel\.app$")

    def test_csv_split_strip_drop_empty(self):
        with patch.dict(
            os.environ,
            {"BACKEND_CORS_ORIGINS": "http://a , http://b,,http://c "},
            clear=True,
        ):
            self.assertEqual(
                get_cors_config().origins,
                ["http://a", "http://b", "http://c"],
            )

    def test_regex_override(self):
        with patch.dict(os.environ, {"BACKEND_CORS_ORIGIN_REGEX": r"^https://x$"}, clear=True):
            self.assertEqual(get_cors_config().origin_regex, r"^https://x$")


class AdminConfigTestCase(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_admin_config()
            self.assertEqual(cfg.reload_token, "")
            self.assertFalse(cfg.allow_local_reload)

    def test_token_strip(self):
        with patch.dict(os.environ, {"LENS_ADMIN_RELOAD_TOKEN": "  secret  "}, clear=True):
            self.assertEqual(get_admin_config().reload_token, "secret")

    def test_allow_local_truthy(self):
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("0", False),
            ("", False),
        ]:
            with patch.dict(os.environ, {"LENS_ALLOW_LOCAL_ADMIN_RELOAD": raw}, clear=True):
                self.assertEqual(get_admin_config().allow_local_reload, expected, raw)


class CacheConfigTestCase(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_cache_config()
            self.assertFalse(cfg.eager_v1_cache)
            self.assertEqual(cfg.gzip_minimum_size, 512)

    def test_eager_strict_equals_one(self):
        """main.py:78 정확히 '!= "1"' 게이트의 반전 → '1'만 활성."""
        for raw, expected in [
            ("1", True),
            ("0", False),
            ("true", False),
            ("yes", False),
            ("", False),
        ]:
            with patch.dict(os.environ, {"LENS_EAGER_V1_CACHE": raw}, clear=True):
                self.assertEqual(get_cache_config().eager_v1_cache, expected, raw)


if __name__ == "__main__":
    unittest.main()
