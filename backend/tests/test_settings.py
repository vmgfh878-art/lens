"""CP235 — 도메인별 Settings 박제.

매 호출 재평가 계약, truthy 변형, CSV 파싱, 기본값/override 케이스.
"""

import os
import unittest
from unittest.mock import patch

from app.config import (
    GitHubReleaseConfig,
    get_admin_config,
    get_cache_config,
    get_cors_config,
    get_database_config,
    get_github_release_config,
    get_market_config,
    get_r2_config,
    get_remote_sync_config,
    get_serving_data_config,
)
from app.services.serving_storage import get_remote_serving_config
from pydantic import ValidationError


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


class ServingDataConfigTestCase(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_serving_data_config()
            self.assertIsNone(cfg.data_dir)
            self.assertIsNone(cfg.bootstrap_dir)
            self.assertIsNone(cfg.cache_dir)

    def test_paths_strip_and_empty_to_none(self):
        with patch.dict(
            os.environ,
            {
                "LENS_SERVING_DATA_DIR": "  data/current  ",
                "LENS_SERVING_BOOTSTRAP_DIR": "   ",
                "LENS_SERVING_CACHE_DIR": " runtime/cache ",
            },
            clear=True,
        ):
            cfg = get_serving_data_config()
            self.assertEqual(cfg.data_dir, "data/current")
            self.assertIsNone(cfg.bootstrap_dir)
            self.assertEqual(cfg.cache_dir, "runtime/cache")


class R2ConfigTestCase(unittest.TestCase):
    def test_default_is_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_r2_config()
            self.assertFalse(cfg.configured)
            self.assertEqual(cfg.object_prefix, "serving/v1")
            self.assertEqual(cfg.region, "auto")
            self.assertIn("LENS_R2_BUCKET", cfg.missing_required_fields)

    def test_account_id_builds_official_endpoint(self):
        with patch.dict(
            os.environ,
            {
                "LENS_R2_ACCOUNT_ID": " account-123 ",
                "LENS_R2_BUCKET": " lens-private ",
                "LENS_R2_ACCESS_KEY_ID": " access ",
                "LENS_R2_SECRET_ACCESS_KEY": " secret ",
                "LENS_R2_PREFIX": "/snapshots/v1/",
            },
            clear=True,
        ):
            cfg = get_r2_config()
            self.assertTrue(cfg.configured)
            self.assertEqual(
                cfg.endpoint_url,
                "https://account-123.r2.cloudflarestorage.com",
            )
            self.assertEqual(cfg.bucket, "lens-private")
            self.assertEqual(cfg.object_prefix, "snapshots/v1")

    def test_explicit_endpoint_works_without_account_id(self):
        with patch.dict(
            os.environ,
            {
                "LENS_R2_ENDPOINT_URL": "http://127.0.0.1:9000/",
                "LENS_R2_BUCKET": "test",
                "LENS_R2_ACCESS_KEY_ID": "access",
                "LENS_R2_SECRET_ACCESS_KEY": "secret",
            },
            clear=True,
        ):
            cfg = get_r2_config()
            self.assertTrue(cfg.configured)
            self.assertEqual(cfg.endpoint_url, "http://127.0.0.1:9000")


class GitHubReleaseConfigTestCase(unittest.TestCase):
    def test_missing_token_is_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_github_release_config()
            self.assertEqual(cfg.repository, "vmgfh878-art/lens-serving")
            self.assertEqual(cfg.release_tag, "serving-snapshots")
            self.assertEqual(cfg.missing_required_fields, ["LENS_GITHUB_TOKEN"])
            self.assertFalse(cfg.configured)

    def test_token_is_trimmed_and_hidden_from_repr(self):
        with patch.dict(os.environ, {"LENS_GITHUB_TOKEN": "  private-test-token  "}, clear=True):
            cfg = get_github_release_config()
            self.assertTrue(cfg.configured)
            self.assertEqual(cfg.token, "private-test-token")
            self.assertNotIn("private-test-token", repr(cfg))

    def test_invalid_repository_and_tag_are_rejected(self):
        for key, value in (
            ("LENS_GITHUB_REPOSITORY", "https://github.com/owner/repo"),
            ("LENS_GITHUB_REPOSITORY", "../repo"),
            ("LENS_GITHUB_REPOSITORY", "owner/repo/extra"),
            ("LENS_GITHUB_RELEASE_TAG", "../snapshot"),
        ):
            with (
                self.subTest(key=key, value=value),
                patch.dict(os.environ, {key: value}, clear=True),
                self.assertRaises(ValidationError),
            ):
                get_github_release_config()

    def test_explicit_github_does_not_fall_back_to_r2_credentials(self):
        env = {
            "LENS_SERVING_STORAGE": "github_release",
            "LENS_R2_ACCOUNT_ID": "test",
            "LENS_R2_BUCKET": "test",
            "LENS_R2_ACCESS_KEY_ID": "test",
            "LENS_R2_SECRET_ACCESS_KEY": "test",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = get_remote_serving_config()
            self.assertIsInstance(cfg, GitHubReleaseConfig)
            self.assertFalse(cfg.configured)

    def test_github_serving_never_uses_supabase_reads(self):
        from app.services.data_backend import use_supabase

        with (
            patch.dict(os.environ, {"LENS_GITHUB_TOKEN": "test-token"}, clear=True),
            patch("app.services.data_backend.supabase_is_configured", return_value=True) as db,
        ):
            self.assertFalse(use_supabase())
            db.assert_not_called()


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


class RemoteSyncConfigTestCase(unittest.TestCase):
    def test_default_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_remote_sync_config().interval_seconds, 0)

    def test_interval_is_bounded(self):
        with patch.dict(os.environ, {"LENS_REMOTE_SYNC_INTERVAL_SECONDS": "300"}, clear=True):
            self.assertEqual(get_remote_sync_config().interval_seconds, 300)
        for value in ("1", "59", "3601", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"LENS_REMOTE_SYNC_INTERVAL_SECONDS": value}, clear=True),
                self.assertRaises(ValidationError),
            ):
                get_remote_sync_config()


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
