"""CP223 characterization snapshot 안전망 conftest.

목적: backend read-path 엔드포인트(9개)의 응답을 syrupy로 박제. CP225+
백엔드 분리 리팩토링의 동작 보존을 byte/tolerance 비교로 증명.

설계:
- sys.path 이중 보정: ROOT(`...\lens`) + BACKEND(`...\lens\backend`) 모두
  insert. app.* / backend.* / ai.* 혼합 import 동시 해결. (pyproject.toml의
  pytest pythonpath와 중복 보장.)
- `LENS_FORCE_LOCAL=1`: Supabase 경로 차단, local parquet only. v1 정책 준수.
- 세션 시작에 4개 캐시 클리어: parquet_store / local_market_svc /
  strategy_backtest_svc / product_prediction_history_svc. cold start
  결정성 확보.
- `FIXED_HEADERS`: `X-Request-Id="test-fixed"`로 request_id.py:9의 uuid4
  비결정성 회피. 운영 코드 수정 없이 호출 측에서만 해결.
- `normalize_floats()`: 응답 dict의 float 값을 round(v, 9)로 정규화.
  rtol≈1e-9 효과. numpy/pandas 버전 미세 변동 흡수.

운영 코드 수정 0. characterization 테스트만 추가.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

# --- sys.path 이중 보정 -----------------------------------------------------
# parents[0] = backend/tests, parents[1] = backend, parents[2] = lens(repo root)
ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- env 강제 ---------------------------------------------------------------
# Supabase 경로 차단, local parquet only. test_api.py의 env clear 테스트는
# patch.dict(clear=True)로 일시 비웠다가 복원하므로 충돌 없음.
os.environ["LENS_FORCE_LOCAL"] = "1"

# --- 공용 fixture 헬퍼 ------------------------------------------------------
FIXED_HEADERS = {"X-Request-Id": "test-fixed"}


def normalize_floats(obj, ndigits: int = 9):
    """응답 JSON의 float 값을 round(v, ndigits)로 정규화.

    NaN/Inf는 None으로(`_jsonable`/`_finite_or_none` 정책과 일치). dict는
    재귀, list는 element-wise. rtol≈10**-ndigits 효과. 기본 9자리.

    Args:
        obj: dict / list / float / int / str / None / bool 임의 중첩.
        ndigits: round 자릿수. 클수록 엄격. 작으면 미세 변동 흡수 폭 ↑.

    Returns:
        같은 구조의 새 객체 (float만 정규화).
    """
    if isinstance(obj, bool):
        # bool은 int 서브클래스라 먼저 처리.
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: normalize_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_floats(v, ndigits) for v in obj]
    return obj


@pytest.fixture(scope="session")
def client():
    """TestClient(app) — 세션 1개 공유 + 시작 시 캐시 cold reset.

    backend/data/v1/*.parquet 디스크 read-only. Supabase 미접속.
    """
    from app.main import app
    from app.services import (
        local_market_svc,
        parquet_store,
        product_prediction_history_svc,
        strategy_backtest_svc,
    )

    # 세션 시작 cold reset (결정성).
    parquet_store.clear_all()
    local_market_svc.clear_caches()
    strategy_backtest_svc.clear_strategy_cache()
    product_prediction_history_svc.clear_product_history_cache()

    from starlette.testclient import TestClient

    return TestClient(app)
