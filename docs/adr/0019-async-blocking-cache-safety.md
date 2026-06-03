# ADR-0019: CP229 — async blocking + cache safety (audit only, no fixes)

Status: Accepted
Date: 2026-06-03
Context: CP229 (refactoring runbook CP221~237)

## Context

CP229 지시서가 가정한 3 개 결함:
- (A) async def 라우트 + 무거운 동기 pandas → 이벤트 루프 정지.
- (B) 모듈 레벨 가변 전역 재할당에 `global` 키워드 누락 → 캐시 동작이 로컬 변수화.
- (C) `lru_cache(maxsize=...)` 의 동시 호출 안전성 / 캐시 미스 lock 외부 실행.

지시서 작성 시점 (2026-06-02 이전) 의 실측은 모두 부재였다. 본 CP 실행 시점 (2026-06-03, CP225~228 직후) 의 재실측:

- (A) `backend/app/routers/**/*.py` 의 `async def` → **0 건**. 코드베이스 전역 단 1 건 (`middleware/request_id.py:22` — pandas 없음, 순수 패스스루). **진단 A 부재 확정.**
- (B) `global` 선언 grep 결과: `db.py:27,40` (_client), `local_market_svc.py:57,93,101,109,117` (_PRICES_1D/1W/_INDICATORS_1D/_STOCK_INFO), `core/logging.py:41` (_CONFIGURED, CP228 신설). 모두 재할당 지점과 짝 정상. **진단 B 부재 확정.**
- (C) **지시서 가정과 실측 차이.** 본 시점 코드에는 `lru_cache` 6 건 / `Lock()` 2 건 존재:
  - `strategy_scan.py`: `lru_cache(maxsize=1)` × 2 (`_load_frame`, `_sector_map`) + `lru_cache(maxsize=16)` (`_strategy_results`) — CP226 분리 산출, 의도적 보존.
  - `product_prediction_history_svc.py`: `lru_cache(maxsize=2)` — OOM 방지 의도 (주석 명시).
  - `ai_repo.py`: `lru_cache(maxsize=1)`.
  - `parquet_store.py`: `Lock()` (per-store).
  - `local_market_svc.py`: `Lock()` (4 슬롯 보호).

## Decision

지시서의 본질 ("결함이 없으면 손대지 않는다, 결함이 발견되어도 새 캐시 인프라 신설은 범위 밖") 을 따라 **소스 코드 0 라인 변경**. 미래 회귀를 막는 정적 가드 1 건만 추가 (`backend/tests/test_async_safety.py`).

- 라우트 동시성 모델 (동기 `def` → FastAPI threadpool) 유지.
- 기존 전역 캐시 (`db._client`, `local_market_svc._PRICES_*` 등) 유지.
- 기존 `lru_cache` / `Lock` 패턴 유지 — CP226 facade 분리에서 이미 보존 검증됨 (.cache_clear attribute 보존, snapshot diff 0).

진단 C 의 lru_cache 동시 호출 안전성 / 캐시 미스 lock 외부 실행은 본 CP 의 차단 트리거 #7 ("캐시 인프라 신설 / 개선 명목 변경 → 범위 밖, 보고하고 멈춤") 대상. 추가 분석 / 튜닝은 별도 후속 CP 권고.

## Consequences

긍정:
- 동작 / 응답 schema 1bit 변경 없음. CP223 snapshot diff 0 자명 (소스 변경 0).
- 미래 회귀 가드 (`test_async_safety.py`) 가 라우트 8 모듈에서 `async def` + 무거운 동기 호출을 AST 정적 검사. 서버 기동 / 네트워크 불필요. 현재는 자명 통과.
- backend/tests 87 baseline + 6 신규 가드 = 93 passed.

부정 / 미해결:
- `strategy_scan._load_frame` (lru_cache maxsize=1) 의 동시 호출 안전성 실측 미확인. 첫 호출이 진행 중일 때 두 번째 동시 호출이 캐시 미스로 또 다시 무거운 로딩을 트리거할 수 있음 (lru_cache 는 내부 lock 으로 dict 동기화하지만 함수 본체 실행 중에 다른 호출이 같은 키로 들어오는 race 는 일부 Python 버전에서 가능). 본 CP 범위 밖. 후속 CP 권고.
- `product_prediction_history_svc.py` 의 `lru_cache(maxsize=2)` 도 동일 패턴.

대안 및 기각 이유:
- (a) **lru_cache 를 double-check lock 으로 감싸 동시성 안전성 강화** — 캐시 인프라 신설. 차단 트리거 #7. 범위 밖.
- (b) **정상 코드를 "정리" 목적으로 변경** — 회귀 위험. 지시서가 명시적으로 금지.

후속 권고:
- CP229 후속 CP: `strategy_scan` / `product_prediction_history_svc` 의 lru_cache 동시 호출 안전성 실측 + 필요 시 명시적 Lock 추가. 본 CP 의 가드 테스트로 라우트 변경에 따른 미래 회귀는 잡힘.
