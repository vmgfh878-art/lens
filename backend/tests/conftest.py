"""CP222 보강 + CP223 characterization snapshot 안전망 conftest.

CP223: backend read-path 엔드포인트(9개)의 응답을 syrupy로 박제.
CP222 보강: 운영 v1 parquet 재오염 영구 가드 (session-scoped autouse).

설계:
- sys.path 이중 보정: ROOT(`.../lens`) + BACKEND(`.../lens/backend`) 모두
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
- `_guard_v1_parquet_integrity()` (CP222 보강): backend/data/v1/*.parquet의
  세션 시작 sha256을 박제 → 세션 종료 시 비교 → 변경된 파일을 `git checkout`
  으로 즉시 복원하고 경고 출력. 어떤 테스트가 운영 데이터를 만져도 모두
  자동 복원 → 런북 §0.8 "운영 parquet 덮어쓰기 금지" 영구 보장.

운영 코드 수정 0. characterization 테스트 + 안전 가드만 추가.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
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


# --- CP222 보강: 운영 v1 parquet 재오염 가드 ---------------------------------
# backend/data/v1/*.parquet는 frontend serving 운영 데이터. .gitignore가
# `!backend/data/v1/*.parquet` 로 명시 추적. 어떤 테스트도 덮어쓰면 안 됨.
# Step 3 pytest 실행 시 운영 데이터가 modified로 노출된 사례가 있었음 (런북
# §0.8 위반). 사용자 결정: 영구 가드 신설.

_V1_PARQUET_DIR = BACKEND / "data" / "v1"


def _sha256_of(path: Path) -> str | None:
    """파일 sha256 16진 헤시. 부재면 None."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _emit_pollution_warning(lines: list[str]) -> None:
    """경고 출력 — pytest의 stdout 캡처를 우회해 항상 보이게 stderr 직출력.

    pythonw/embedded 환경에서는 sys.__stderr__가 None일 수 있어 sys.stderr로
    폴백. 폴백한 stderr는 pytest 캡처를 받지만 pytest 종료 후 출력에 포함됨.
    """
    stream = sys.__stderr__ or sys.stderr
    stream.write("\n".join(lines) + "\n")
    stream.flush()


@pytest.fixture(scope="session", autouse=True)
def _guard_v1_parquet_integrity():
    """운영 v1 parquet 재오염 가드 (CP222 보강, 영구).

    - 세션 시작: backend/data/v1/*.parquet 각 파일의 sha256 박제.
    - 세션 종료 (yield 후): sha256 재비교.
    - 변경된 파일이 있으면 `git checkout -- <파일>` 로 즉시 복원하고 경고.
    - 복원 실패 시: 차단 트리거. 사용자 보고 필요 (pytest 종료 후 git
      status backend/data/v1/ 가 dirty 이면 발견됨).
    """
    files = sorted(_V1_PARQUET_DIR.glob("*.parquet"))
    baseline = {f.name: _sha256_of(f) for f in files}

    yield

    polluted: list[str] = []
    for f in files:
        if _sha256_of(f) != baseline.get(f.name):
            polluted.append(f.name)

    if not polluted:
        return

    msg = [
        "",
        "=" * 72,
        "[CP222 GUARD] v1 parquet pollution detected:",
        *[f"  - {name}" for name in polluted],
        "Attempting `git checkout` restore (runbook §0.8: 운영 parquet 보호).",
        "=" * 72,
    ]
    _emit_pollution_warning(msg)

    failed: list[str] = []
    for name in polluted:
        rel = (_V1_PARQUET_DIR / name).relative_to(ROOT).as_posix()
        try:
            subprocess.run(
                ["git", "checkout", "--", rel],
                cwd=str(ROOT),
                check=True,
                capture_output=True,
            )
            _emit_pollution_warning([f"  RESTORED: {rel}"])
        except subprocess.CalledProcessError as exc:
            failed.append(rel)
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            _emit_pollution_warning([f"  RESTORE FAILED: {rel}: {stderr}"])

    if failed:
        _emit_pollution_warning(
            [
                "",
                "[CP222 GUARD] 차단 트리거: 복원 실패한 파일이 있다. ",
                "  `git status backend/data/v1/` 확인 후 보고하라.",
                "=" * 72,
            ]
        )


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
