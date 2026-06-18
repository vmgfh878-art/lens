"""CP254 — Supabase REST 서빙 전환 단위/계약 테스트.

네트워크 0: FakeSupabase(인메모리 PostgREST 부분 구현)로 REST 경로를 검증한다.
- data_backend.use_supabase() env 우선순위 진리표
- rest_paging.paged_select 페이지 이어붙임
- prediction_repo: 슬롯 윈도우/부재 ticker None/horizon 빈 frame/product history 윈도우·limit
- REST 모드 엔드포인트 (line/band/tickers/product-history/stocks/prices/indicators):
  로컬 모드와 응답 골격 동일 (회귀 0 원칙)
- strategy_scan REST 분기: base/line/band/sector 입력 출처 교체
- import_v1_serving_to_supabase 빌더: 화이트리스트/on_conflict/NaN→None/날짜 str
- ETag/Cache-Control egress 가드 (P4)
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FakeSupabase — postgrest 빌더의 사용 부분집합(인메모리)
# ---------------------------------------------------------------------------


class FakeQuery:
    def __init__(self, rows: list[dict], columns: str | None, count: str | None):
        self._rows = rows
        self._columns = None if columns in (None, "*") else [c.strip() for c in columns.split(",")]
        self._count = count
        self._ops: list[tuple[str, str, object]] = []
        self._or: list[str] = []
        self._order: list[tuple[str, bool]] = []
        self._limit: int | None = None
        self._range: tuple[int, int] | None = None

    def eq(self, column, value):
        self._ops.append(("eq", column, value))
        return self

    def gte(self, column, value):
        self._ops.append(("gte", column, value))
        return self

    def lte(self, column, value):
        self._ops.append(("lte", column, value))
        return self

    def in_(self, column, values):
        self._ops.append(("in", column, list(values)))
        return self

    def ilike(self, column, pattern):
        self._ops.append(("ilike", column, pattern))
        return self

    def or_(self, expression):
        self._or.append(expression)
        return self

    def order(self, column, desc: bool = False):
        self._order.append((column, desc))
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    @staticmethod
    def _match(row, op, column, value) -> bool:
        actual = row.get(column)
        if op == "eq":
            return actual == value
        if op == "gte":
            return actual is not None and str(actual) >= str(value)
        if op == "lte":
            return actual is not None and str(actual) <= str(value)
        if op == "in":
            return actual in value
        if op == "ilike":
            prefix = str(value).rstrip("%").upper()
            return actual is not None and str(actual).upper().startswith(prefix)
        raise AssertionError(f"unsupported op {op}")

    def execute(self):
        rows = list(self._rows)
        for op, column, value in self._ops:
            rows = [r for r in rows if self._match(r, op, column, value)]
        for expression in self._or:
            if expression == "source.eq.eodhd,source.is.null":
                rows = [r for r in rows if r.get("source") in ("eodhd", None)]
            else:
                raise AssertionError(f"unsupported or_ {expression}")
        for column, desc in reversed(self._order):
            rows = sorted(
                rows,
                key=lambda r: (r.get(column) is None, str(r.get(column))),
                reverse=desc,
            )
        total = len(rows)
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        elif self._limit is not None:
            rows = rows[: self._limit]
        if self._columns is not None:
            rows = [{c: r.get(c) for c in self._columns} for r in rows]
        count = total if self._count else None
        return SimpleNamespace(data=[dict(r) for r in rows], count=count)


class FakeTable:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, columns="*", count: str | None = None, **_kwargs):
        return FakeQuery(self._rows, columns, count)


class FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables

    def table(self, name: str) -> FakeTable:
        return FakeTable(self.tables.get(name, []))


def _fake_tables() -> dict[str, list[dict]]:
    line = [
        {
            "ticker": "AAPL",
            "asof_date": d,
            "line_score": s,
            "safe_line_score": s,
            "line_rank_by_date": 0.5,
            "safe_line_rank_by_date": 0.5,
            "line_top_decile_flag": False,
            "safe_line_top_decile_flag": False,
            "actual_h5_return": None,
            "model_id": "m1",
            "source_cp": "CP254",
        }
        for d, s in [("2024-01-02", 0.01), ("2026-06-01", 0.02), ("2026-06-10", 0.03)]
    ]
    band_1d = [
        {
            "ticker": "AAPL",
            "asof_date": d,
            "forecast_date": d,
            "horizon_step": h,
            "band_lower": 90.0,
            "band_upper": 110.0,
            "actual_return": None,
            "actual_return_available": False,
            "model_id": "m2",
            "source_cp": "CP254",
        }
        for d in ("2026-06-09", "2026-06-10")
        for h in (1, 5)
    ]
    band_1w = [
        {
            "ticker": "AAPL",
            "asof_date": "2026-06-05",
            "horizon_step": h,
            "band_lower": -0.05,
            "band_upper": 0.05,
            "actual_return": None,
            "model_id": "m3",
            "source_cp": "CP254",
        }
        for h in (1, 2)
    ]
    history = []
    for d in ("2026-06-08", "2026-06-09", "2026-06-10"):
        history.append(
            {
                "ticker": "AAPL",
                "timeframe": "1D",
                "role": "line",
                "asof_date": d,
                "display_date": d,
                "display_horizon": 5,
                "run_id": "run-line",
                "line_value": 1.0,
                "lower_value": None,
                "upper_value": None,
            }
        )
        for h in (1, 2):
            history.append(
                {
                    "ticker": "AAPL",
                    "timeframe": "1D",
                    "role": "band",
                    "asof_date": d,
                    "display_date": d,
                    "display_horizon": h,
                    "run_id": "run-band",
                    "line_value": None,
                    "lower_value": 90.0,
                    "upper_value": 110.0,
                }
            )
    stock_info = [
        {"ticker": "AAPL", "sector": "Technology", "industry": "CE", "market_cap": 1.0},
        {"ticker": "MSFT", "sector": "Technology", "industry": "SW", "market_cap": 2.0},
        # FK placeholder (서빙 목록에 나오면 안 됨 — serving_stocks 분리 검증용)
        {"ticker": "ZZZZ", "sector": None, "industry": "Unknown", "market_cap": None},
    ]
    serving_stocks = [
        {"ticker": "AAPL", "sector": "Technology", "industry": "CE", "market_cap": 1.0},
        {"ticker": "MSFT", "sector": "Technology", "industry": "SW", "market_cap": 2.0},
    ]
    price_data = [
        {
            "ticker": "AAPL",
            "date": d,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "adjusted_close": 1.4,
            "volume": 100,
            "source": "yfinance",
        }
        for d in ("2026-06-09", "2026-06-10")
    ]
    indicators = [
        {
            "ticker": "AAPL",
            "timeframe": "1D",
            "date": d,
            "rsi": 0.5,
            "macd_ratio": 0.01,
            "bb_position": 0.4,
            "ma_5_ratio": 0.0,
            "ma_20_ratio": 0.0,
            "ma_60_ratio": 0.0,
            "vol_change": 0.1,
            "volume": 100,
            "atr_ratio": None,
            "regime_label": "neutral",
            "source": "yfinance",
        }
        for d in ("2026-06-09", "2026-06-10")
    ]
    return {
        "predictions_line_1d": line,
        "predictions_band_1d": band_1d,
        "predictions_band_1w": band_1w,
        "product_prediction_history_1d": history,
        "stock_info": stock_info,
        "serving_stocks": serving_stocks,
        "price_data": price_data,
        "indicators": indicators,
    }


# ---------------------------------------------------------------------------
# data_backend.use_supabase 진리표
# ---------------------------------------------------------------------------


class DataBackendToggleTestCase(unittest.TestCase):
    """순수 env 우선순위 진리표 — 자동 폴백은 꺼서(AUTOFALLBACK=0) 도달성과 분리."""

    def _call(self, env: dict[str, str]) -> bool:
        from app.services import data_backend

        merged = {"LENS_SUPABASE_AUTOFALLBACK": "0", **env}
        with patch.dict(os.environ, merged, clear=True):
            return data_backend.use_supabase()

    def test_unconfigured_is_local(self):
        self.assertFalse(self._call({}))

    def test_url_key_only_is_supabase(self):
        self.assertTrue(self._call({"SUPABASE_URL": "http://x", "SUPABASE_KEY": "k"}))

    def test_force_local_wins(self):
        self.assertFalse(
            self._call({"SUPABASE_URL": "http://x", "SUPABASE_KEY": "k", "LENS_FORCE_LOCAL": "1"})
        )

    def test_data_backend_local_wins(self):
        self.assertFalse(
            self._call(
                {"SUPABASE_URL": "http://x", "SUPABASE_KEY": "k", "LENS_DATA_BACKEND": "local"}
            )
        )

    def test_use_local_snapshots_wins(self):
        # .env.example 문서 의미 보존: "1 이면 SUPABASE 대신 local parquet".
        self.assertFalse(
            self._call(
                {
                    "SUPABASE_URL": "http://x",
                    "SUPABASE_KEY": "k",
                    "LENS_USE_LOCAL_SNAPSHOTS": "1",
                }
            )
        )

    def test_require_local_snapshots_wins(self):
        self.assertFalse(
            self._call(
                {
                    "SUPABASE_URL": "http://x",
                    "SUPABASE_KEY": "k",
                    "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
                }
            )
        )


class AutoFallbackTestCase(unittest.TestCase):
    """CP255 — Supabase 도달성 자동 폴백 (복구되면 DB, 죽으면 parquet, 자유 전환)."""

    def setUp(self):
        from app.services import data_backend

        data_backend.reset_reachable_cache()
        self.addCleanup(data_backend.reset_reachable_cache)

    def _use(self, *, reachable: bool, extra_env: dict | None = None):
        from app.services import data_backend

        env = {"SUPABASE_URL": "http://x", "SUPABASE_KEY": "k", **(extra_env or {})}
        if reachable:
            client = FakeSupabase({"stock_info": [{"ticker": "AAPL"}]})
            gs = patch("app.services.data_backend.get_supabase", return_value=client)
        else:
            gs = patch(
                "app.services.data_backend.get_supabase",
                side_effect=OSError("getaddrinfo failed"),
            )
        with patch.dict(os.environ, env, clear=True), gs as mock_gs:
            result = data_backend.use_supabase()
        return result, mock_gs

    def test_reachable_uses_supabase(self):
        result, _ = self._use(reachable=True)
        self.assertTrue(result)

    def test_unreachable_falls_back_to_local(self):
        # configured 지만 Supabase 死 → 자동 parquet (Render env 안 건드려도)
        result, _ = self._use(reachable=False)
        self.assertFalse(result)

    def test_force_local_overrides_even_if_reachable(self):
        result, mock_gs = self._use(reachable=True, extra_env={"LENS_FORCE_LOCAL": "1"})
        self.assertFalse(result)
        mock_gs.assert_not_called()  # 강제 로컬이면 ping 도 안 함

    def test_autofallback_disabled_skips_ping(self):
        # AUTOFALLBACK=0 이면 (구) 동작: configured 면 ping 없이 True
        result, mock_gs = self._use(reachable=False, extra_env={"LENS_SUPABASE_AUTOFALLBACK": "0"})
        self.assertTrue(result)
        mock_gs.assert_not_called()

    def test_reachability_is_cached(self):
        from app.services import data_backend

        client = FakeSupabase({"stock_info": [{"ticker": "AAPL"}]})
        env = {"SUPABASE_URL": "http://x", "SUPABASE_KEY": "k"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("app.services.data_backend.get_supabase", return_value=client) as mock_gs,
        ):
            data_backend.use_supabase()
            data_backend.use_supabase()
            data_backend.use_supabase()
        self.assertEqual(mock_gs.call_count, 1)  # 3회 호출, ping 은 1회(캐시)


# ---------------------------------------------------------------------------
# rest_paging
# ---------------------------------------------------------------------------


class PagedSelectTestCase(unittest.TestCase):
    def test_pages_are_stitched_in_order(self):
        from app.repositories.rest_paging import paged_select

        rows = [{"i": i} for i in range(2500)]
        calls: list[tuple[int, int]] = []

        class Query:
            def range(self, start, end):
                calls.append((start, end))
                self._slice = (start, end)
                return self

            def execute(self):
                start, end = self._slice
                return SimpleNamespace(data=rows[start : end + 1])

        out = paged_select(lambda: Query(), page_size=1000)
        self.assertEqual([r["i"] for r in out], list(range(2500)))
        self.assertEqual(calls, [(0, 999), (1000, 1999), (2000, 2999)])

    def test_runaway_guard(self):
        from app.repositories.rest_paging import paged_select

        class Query:
            def range(self, start, end):
                return self

            def execute(self):
                return SimpleNamespace(data=[{"i": 0}] * 10)

        with self.assertRaises(RuntimeError):
            paged_select(lambda: Query(), page_size=10, max_pages=3)


# ---------------------------------------------------------------------------
# prediction_repo
# ---------------------------------------------------------------------------


class PredictionRepoTestCase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSupabase(_fake_tables())
        self.patcher = patch(
            "app.repositories.prediction_repo.get_supabase", return_value=self.fake
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_line_window_filters_old_rows(self):
        from app.repositories import prediction_repo

        frame = prediction_repo.fetch_line_1d("AAPL", days=30)
        self.assertEqual(len(frame), 2)  # 2024-01-02 행은 윈도우 밖
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["asof_date"]))
        self.assertEqual(list(frame.columns), prediction_repo.LINE_1D_COLUMNS)

    def test_absent_ticker_returns_none(self):
        from app.repositories import prediction_repo

        self.assertIsNone(prediction_repo.fetch_line_1d("NOPE", days=30))

    def test_band_horizon_filter_returns_empty_frame_not_none(self):
        from app.repositories import prediction_repo

        frame = prediction_repo.fetch_band_1d("AAPL", days=30, horizon=3)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 0)

    def test_band_full_horizons(self):
        from app.repositories import prediction_repo

        frame = prediction_repo.fetch_band_1d("AAPL", days=30)
        self.assertEqual(len(frame), 4)
        self.assertEqual(sorted(frame["horizon_step"].unique().tolist()), [1, 5])

    def test_product_history_lookback_and_roles(self):
        from app.repositories import prediction_repo

        frame = prediction_repo.fetch_product_history(
            "AAPL", timeframe="1D", roles={"line", "band"}, lookback_days=1
        )
        # 최신 6/10 기준 1일 lookback → 6/09 + 6/10, role 3행/일 = 6행
        self.assertEqual(len(frame), 6)
        self.assertEqual(set(frame["role"].unique()), {"line", "band"})

    def test_product_history_limit_is_per_role(self):
        from app.repositories import prediction_repo

        frame = prediction_repo.fetch_product_history(
            "AAPL", timeframe="1D", roles={"line", "band"}, limit=2
        )
        counts = frame["role"].value_counts().to_dict()
        self.assertEqual(counts, {"line": 2, "band": 2})
        # 최근 행 우선 (asof desc limit) 후 asc 재정렬
        self.assertEqual(
            frame[frame["role"] == "line"]["asof_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-06-09", "2026-06-10"],
        )

    def test_list_tickers_prefix(self):
        from app.repositories import prediction_repo

        self.assertEqual(prediction_repo.list_tickers(search="AA"), ["AAPL"])
        # tickers 는 가격 유니버스(stock_info, placeholder 포함) 기준 — 큐레이션 아님.
        self.assertEqual(prediction_repo.list_tickers(), ["AAPL", "MSFT", "ZZZZ"])

    def test_scan_frames_thin_columns(self):
        from app.repositories import prediction_repo

        line = prediction_repo.fetch_line_frame_for_scan()
        self.assertEqual(list(line.columns), ["ticker", "asof_date", "line_score"])
        band = prediction_repo.fetch_band_frame_for_scan(horizon=5)
        self.assertEqual(
            list(band.columns),
            ["ticker", "asof_date", "horizon_step", "band_lower", "band_upper"],
        )
        self.assertTrue((band["horizon_step"] == 5).all())


# ---------------------------------------------------------------------------
# REST 모드 엔드포인트 — 로컬 모드와 응답 골격 동일
# ---------------------------------------------------------------------------


class RestModeEndpointTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient

        from app.main import app

        cls.client = TestClient(app)

    def _rest(self):
        fake = FakeSupabase(_fake_tables())
        return (
            patch("app.services.data_backend.use_supabase", return_value=True),
            patch("app.repositories.prediction_repo.get_supabase", return_value=fake),
            patch("app.repositories.market_repo.get_supabase", return_value=fake),
        )

    def test_line_endpoint_rest_shape(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/predictions/line/AAPL", params={"days": 30})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()["data"]
        self.assertEqual(body["slot"], "1D Line")
        self.assertEqual(body["rows"], 2)
        self.assertEqual(body["model_id"], "m1")
        self.assertEqual(body["data"][0]["asof_date"], "2026-06-01")
        self.assertEqual(resp.headers["cache-control"], "public, max-age=3600")
        self.assertIn("etag", resp.headers)

    def test_line_endpoint_rest_404(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/predictions/line/NOPE")
        self.assertEqual(resp.status_code, 404)

    def test_line_endpoint_rest_etag_304(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            first = self.client.get("/api/v1/predictions/line/AAPL", params={"days": 30})
            etag = first.headers["etag"]
            second = self.client.get(
                "/api/v1/predictions/line/AAPL",
                params={"days": 30},
                headers={"If-None-Match": etag},
            )
        self.assertEqual(second.status_code, 304)

    def test_band_1d_endpoint_rest_horizon_empty_is_200(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get(
                "/api/v1/predictions/band/1d/AAPL", params={"days": 30, "horizon": 3}
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()["data"]
        self.assertEqual(body["rows"], 0)
        self.assertEqual(body["horizons"], [])

    def test_band_1w_endpoint_rest(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/predictions/band/1w/AAPL", params={"days": 30})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()["data"]
        self.assertEqual(body["rows"], 2)
        self.assertEqual(body["horizons"], [1, 2])

    def test_tickers_endpoint_rest(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/predictions/tickers", params={"search": "AA"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"]["tickers"], ["AAPL"])

    def test_product_history_endpoint_rest(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get(
                "/api/v1/stocks/AAPL/predictions/product-history",
                params={"lookback_days": 1},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()["data"]
        self.assertEqual(body["source"], "product_rolling_replay")
        self.assertEqual(len(body["line_history"]), 2)
        self.assertEqual(len(body["band_history"]), 4)
        self.assertIn("etag", resp.headers)

    def test_stocks_endpoint_rest(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/stocks", params={"search": "MS"})
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()["data"]
        self.assertEqual([r["ticker"] for r in rows], ["MSFT"])

    def test_stocks_endpoint_rest_excludes_fk_placeholder(self):
        # /stocks 목록은 serving_stocks(큐레이션) 기준 — stock_info 의 FK
        # placeholder(ZZZZ, sector NULL)가 목록에 새면 안 된다.
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/stocks", params={"limit": 50})
        tickers = [r["ticker"] for r in resp.json()["data"]]
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        self.assertNotIn("ZZZZ", tickers)

    def test_prices_endpoint_rest(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get(
                "/api/v1/stocks/AAPL/prices",
                params={"start": "2026-06-01", "end": "2026-06-10"},
            )
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()["data"]["data"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["date"], "2026-06-10")

    def test_indicators_endpoint_rest_volume_no_second_query(self):
        p1, p2, p3 = self._rest()
        with p1, p2, p3:
            resp = self.client.get("/api/v1/stocks/AAPL/indicators", params={"limit": 5})
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()["data"]["data"]
        self.assertEqual(len(rows), 2)
        # CP254 — indicators.volume 컬럼이 채워져 있으면 그 값을 그대로 사용.
        self.assertEqual(rows[0]["volume"], 100)
        self.assertIn("etag", resp.headers)


# ---------------------------------------------------------------------------
# strategy_scan REST 분기
# ---------------------------------------------------------------------------


class StrategyScanRestTestCase(unittest.TestCase):
    def setUp(self):
        from app.services.strategy_backtest_svc import clear_strategy_cache

        clear_strategy_cache()
        self.addCleanup(clear_strategy_cache)

    def _price_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
        return pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "date": dates,
                "open": np.linspace(10, 11, 30),
                "high": np.linspace(10.5, 11.5, 30),
                "low": np.linspace(9.5, 10.5, 30),
                "close": np.linspace(10, 11, 30),
                "adjusted_close": np.linspace(10, 11, 30),
                "volume": [100] * 30,
            }
        )

    def _indicator_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
        return pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "date": dates,
                "rsi": [0.5] * 30,
                "macd_ratio": [0.01] * 30,
                "ma_5_ratio": [0.0] * 30,
                "ma_20_ratio": [0.0] * 30,
                "ma_60_ratio": [0.0] * 30,
                "bb_position": [0.4] * 30,
            }
        )

    def test_load_base_frame_uses_rest_inputs(self):
        from app.services import strategy_scan

        with (
            patch("app.services.data_backend.use_supabase", return_value=True),
            patch(
                "app.repositories.market_repo.fetch_price_frame_for_scan",
                return_value=self._price_frame(),
            ) as price_mock,
            patch(
                "app.repositories.market_repo.fetch_indicator_frame_for_scan",
                return_value=self._indicator_frame(),
            ) as ind_mock,
        ):
            frame = strategy_scan._load_base_frame()
        price_mock.assert_called_once()
        ind_mock.assert_called_once()
        self.assertIn("atr_ratio_calc", frame.columns)
        self.assertIn("rsi_norm", frame.columns)
        self.assertIn("roc_20", frame.columns)
        self.assertTrue((frame["ticker"] == "AAPL").all())

    def test_sector_map_uses_rest(self):
        from app.services import strategy_scan

        sector_frame = pd.DataFrame({"ticker": ["AAPL"], "sector": ["Technology"]})
        with (
            patch("app.services.data_backend.use_supabase", return_value=True),
            patch("app.repositories.market_repo.fetch_sector_frame", return_value=sector_frame),
        ):
            mapping = strategy_scan._sector_map()
        self.assertEqual(mapping, {"AAPL": "Technology"})

    def test_merge_band_rest_uses_horizon5_server_filter(self):
        from app.services import strategy_scan

        base = self._price_frame()[["ticker", "date", "close"]].copy()
        base["date"] = pd.to_datetime(base["date"])
        band_frame = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "asof_date": pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d"),
                "horizon_step": [5] * 30,
                "band_lower": np.linspace(9, 10, 30),
                "band_upper": np.linspace(11, 12, 30),
            }
        )
        with (
            patch("app.services.data_backend.use_supabase", return_value=True),
            patch(
                "app.repositories.prediction_repo.fetch_band_frame_for_scan",
                return_value=band_frame,
            ) as band_mock,
        ):
            merged = strategy_scan._merge_band(base)
        band_mock.assert_called_once_with(horizon=5)
        self.assertIn("band_lower_return", merged.columns)
        self.assertIn("band_width_percentile", merged.columns)


# ---------------------------------------------------------------------------
# import 스크립트 빌더
# ---------------------------------------------------------------------------


class ImportBuildersTestCase(unittest.TestCase):
    def test_serving_stocks_frame_preserves_nulls(self):
        # /stocks parity: serving_stocks 는 parquet 의 null 을 그대로 — fillna 금지.
        from backend.scripts import import_v1_serving_to_supabase as imp

        raw = pd.DataFrame(
            {
                "ticker": ["aapl", "msft"],
                "sector": ["Technology", None],
                "industry": [None, "SW"],
                "market_cap": [1.0, np.nan],
            }
        )
        frame, columns, on_conflict = imp.build_serving_stocks_frame(raw)
        records = imp._records(frame, columns)
        self.assertEqual(on_conflict, "ticker")
        self.assertEqual(records[0]["ticker"], "AAPL")
        self.assertIsNone(records[0]["industry"])  # 'Unknown' 으로 채우면 안 됨
        self.assertIsNone(records[1]["sector"])
        self.assertIsNone(records[1]["market_cap"])

    def test_price_frame_whitelist_and_source(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        raw = pd.DataFrame(
            {
                "ticker": pd.Categorical(["aapl"]),
                "date": pd.to_datetime(["2026-06-10"]),
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "adjusted_close": [np.nan],
                "volume": np.array([100], dtype="int64"),
            }
        )
        frame, columns, on_conflict = imp.build_price_frame(raw)
        records = imp._records(frame, columns)
        self.assertEqual(on_conflict, "ticker,date,source")
        self.assertEqual(
            set(records[0].keys()),
            {
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "source",
                "provider",
                "provider_adjustment_policy",
            },
        )
        self.assertEqual(records[0]["ticker"], "AAPL")
        self.assertEqual(records[0]["date"], "2026-06-10")
        self.assertIsNone(records[0]["adjusted_close"])
        self.assertEqual(records[0]["source"], "yfinance")
        self.assertIsInstance(records[0]["volume"], int)

    def test_indicator_frame_joins_price_volume(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        raw = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2026-06-10"]),
                "rsi": [0.5],
                "macd_ratio": [0.01],
                "bb_position": [0.4],
                "ma_5_ratio": [0.0],
                "ma_20_ratio": [0.0],
                "ma_60_ratio": [0.0],
                "vol_change": [np.nan],
                "regime_label": pd.Categorical(["neutral"]),
            }
        )
        price = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2026-06-10"]),
                "volume": np.array([12345], dtype="int64"),
            }
        )
        frame, columns, on_conflict = imp.build_indicator_frame(raw, price)
        records = imp._records(frame, columns)
        self.assertEqual(on_conflict, "ticker,timeframe,date,source")
        self.assertEqual(records[0]["volume"], 12345)
        self.assertEqual(records[0]["timeframe"], "1D")
        self.assertIsNone(records[0]["vol_change"])
        self.assertNotIn("log_return", records[0])  # 화이트리스트 밖

    def test_product_history_float32_promoted(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        raw = pd.DataFrame(
            {
                "ticker": pd.Categorical(["AAPL"]),
                "timeframe": pd.Categorical(["1D"]),
                "role": pd.Categorical(["band"]),
                "asof_date": pd.to_datetime(["2026-06-10"]),
                "display_date": pd.Categorical(["2026-06-11"]),
                "display_horizon": np.array([1], dtype="int8"),
                "run_id": pd.Categorical(["run-1"]),
                "line_value": np.array([np.nan], dtype="float32"),
                "lower_value": np.array([119.117927], dtype="float32"),
                "upper_value": np.array([122.389259], dtype="float32"),
                "source": ["x"],
            }
        )
        frame, columns, on_conflict = imp.build_product_history_frame(raw)
        records = imp._records(frame, columns)
        self.assertEqual(on_conflict, "ticker,timeframe,role,asof_date,display_horizon,run_id")
        self.assertNotIn("source", records[0])
        self.assertIsNone(records[0]["line_value"])
        self.assertIsInstance(records[0]["lower_value"], float)
        self.assertEqual(records[0]["display_horizon"], 1)
        self.assertEqual(records[0]["display_date"], "2026-06-11")

    def test_placeholder_uses_ignore_duplicates(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        captured: list[dict] = []

        class Table:
            def upsert(self, records, **kwargs):
                captured.append({"records": records, "kwargs": kwargs})
                return self

            def execute(self):
                return None

        client = SimpleNamespace(table=lambda name: Table())
        frame = pd.DataFrame({"ticker": ["AAPL", "ZZZZ"]})
        imp.ensure_stock_info_covers(client, frame, dry_run=False)
        self.assertTrue(captured)
        self.assertTrue(captured[0]["kwargs"]["ignore_duplicates"])
        self.assertEqual(captured[0]["kwargs"]["on_conflict"], "ticker")


class Phase4aTestCase(unittest.TestCase):
    """CP255 Phase 4a — product_history 제외, retention 불변식, publish 윈도우."""

    def test_default_tables_excludes_product_history(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        self.assertNotIn("product_prediction_history_1d", imp.DEFAULT_TABLES)
        self.assertEqual(len(imp.DEFAULT_TABLES), 7)
        # 화면이 쓰는 6개 + serving_stocks 는 들어 있어야
        for table in ("price_data", "indicators", "predictions_band_1d", "serving_stocks"):
            self.assertIn(table, imp.DEFAULT_TABLES)

    def test_run_import_default_skips_history(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        # dry_run 은 네트워크 0. 기본 호출이 product_history 를 selected 에서 빼는지.
        counts = imp.run_import(tickers=["AAPL"], dry_run=True)
        self.assertNotIn("product_prediction_history_1d", counts)
        self.assertIn("predictions_band_1d", counts)

    def test_run_import_opt_in_history(self):
        from backend.scripts import import_v1_serving_to_supabase as imp

        counts = imp.run_import(tickers=["AAPL"], dry_run=True, include_product_history=True)
        self.assertIn("product_prediction_history_1d", counts)

    def test_retention_invariant_rejects_short_window(self):
        from backend.scripts import supabase_retention as ret

        # 윈도우(100d) < 화면 요청(730d) → 데모 빈 구간 위험 → 즉시 예외
        with self.assertRaises(ValueError):
            ret._validate_invariants([("predictions_band_1w", "asof_date", 100, 730)])

    def test_retention_specs_satisfy_invariant(self):
        from backend.scripts import supabase_retention as ret

        ret._validate_invariants(ret.RETENTION_SPECS)  # 예외 없어야 통과
        bw = next(s for s in ret.RETENTION_SPECS if s[0] == "predictions_band_1w")
        self.assertGreaterEqual(bw[2], 730)  # band_1w 윈도우 ≥ 화면 730d 요청

    def test_publish_full_window_covers_retention(self):
        from backend.scripts import publish_serving_to_supabase as pub
        from backend.scripts import supabase_retention as ret

        max_window = max(s[2] for s in ret.RETENTION_SPECS)
        # 전체 재동기화 윈도우는 최장 retention 보관분을 덮어야 (빈 구간 방지)
        self.assertGreaterEqual(pub.FULL_WINDOW_DAYS, max_window)
        self.assertLess(pub.INCREMENTAL_LOOKBACK_DAYS, pub.FULL_WINDOW_DAYS)


if __name__ == "__main__":
    unittest.main()
