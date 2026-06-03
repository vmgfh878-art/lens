# CP229 보고서 — async blocking + cache safety (검증 + 가드, 수리 0)

## 요구

`docs/cp229_async_blocking_cache_safety_directive.md`. 진단 A (async + pandas), B (global 누락), C (lru_cache 동시성) 를 grep 으로 재확인. 결함 없으면 no-op, 있으면 차단 보고. async-blocking 회귀 가드 1 건 추가.

## 한 일

5 step 중 4 commit (Step 1-3 검증만, Step 4 가드, Step 5 sanity 통합):

| Step | 커밋 | 작업 | 결과 |
|---|---|---|---|
| 1-3 | (검증, 커밋 없음) | grep 으로 async / global / lru_cache 재확인 | A/B 부재 확정, C 실측 차이 (보고서 기록) |
| 4 | 74459e5 | test_async_safety.py 추가 (AST 정적 가드) | 6 passed + 2 skipped (모듈 부재) |

## 핵심 컴포넌트 존재 체크리스트

### 진단 A — async def 라우트 + blocking pandas

```
backend/app/routers/**/*.py 의 async def: 0 건 (모든 라우트 def)
코드베이스 전역 async def: middleware/request_id.py:22 (pandas 호출 없음, 순수 패스스루)
```

→ **A 부재 확정.** FastAPI 가 `def` 라우트를 threadpool 에서 실행하므로 무거운 pandas 가 이벤트 루프를 막지 않음. 수정 대상 없음.

### 진단 B — 모듈 global 정합

```
backend/app/db.py:27, 40                       global _client (재할당 짝 정상)
backend/app/services/local_market_svc.py:
    57  global _PRICES_1D, _PRICES_1W, _INDICATORS_1D, _STOCK_INFO
    93  global _PRICES_1D
    101 global _PRICES_1W
    109 global _INDICATORS_1D
    117 global _STOCK_INFO
backend/app/core/logging.py:41                 global _CONFIGURED (CP228 신설)
```

→ **B 부재 확정.** 모든 모듈 전역 재할당이 `global` 선언과 짝. 수정 대상 없음.

### 진단 C — lru_cache / Lock 실측 (지시서 가정과 차이)

지시서: "lru_cache 0 건, Lock 0 건, parquet_store / strategy_backtest_svc 파일 없음." (작성 시점 가정)

본 CP 실측 (CP225~228 이후):

```
backend/app/services/strategy_scan.py (CP226 분리 산출):
    35   @lru_cache(maxsize=1)   # _load_frame
    182  @lru_cache(maxsize=1)   # _sector_map
    195  @lru_cache(maxsize=16)  # _strategy_results
backend/app/services/product_prediction_history_svc.py:
    82   @lru_cache(maxsize=2)   # OOM 방지 주석 명시
backend/app/repositories/ai_repo.py:
    19   @lru_cache(maxsize=1)
backend/app/services/parquet_store.py:
    50   _LOCK = Lock()         # per-store
backend/app/services/local_market_svc.py:
    33   _LOCK = Lock()         # 4 슬롯 보호
```

→ **C 지시서 가정과 차이 — 그러나 손대지 않음.** 차단 트리거 #7 ("캐시 인프라 신설 / 개선 명목 변경 → 범위 밖") 대상. CP226 facade 분리에서 이미 `.cache_clear` attribute 보존 + snapshot diff 0 검증 완료. 동시성 안전성 강화는 별도 후속 CP 권고.

### 가드 테스트

`backend/tests/test_async_safety.py`:
- 라우트 8 모듈 (prices, predict, v1.stocks, v1.health, v1.predictions, v1.strategies, v1.ai, v1.admin) 검사.
- 각 모듈에서 `@router.<method>` 데코레이터가 붙은 함수가 `async def` 이면 본문에 무거운 동기 호출 (HEAVY_SYNC_CALLS: `aggregate_prices`, `build_features`, `build_price_features`, `resample_price_frame`, `get_price_response_data`, `get_indicator_response_data`, `get_latest_prediction_data`, `_load_frame`, `_strategy_results` 등) 이 `await` 없이 등장하면 fail.
- 현재는 모든 라우트가 `def` 이므로 자명 통과. 미래 회귀만 잡음.
- 6 passed, 2 skipped (`app.routers.prices`, `app.routers.predict` 모듈 부재).

## 새 테스트 결과

`test_async_safety.py` 8 케이스: 6 passed + 2 skipped. 자명 통과 (현재 라우트 모두 def).

## dry-run 결과

소스 코드 0 라인 변경 → 응답 schema / 동작 무변경. CP223 snapshot diff 0 자명. 별도 dry-run 불필요.

## 기존 회귀 통과 건수

- `pytest backend/tests --ignore=test_services.py`: **93 passed** (87 baseline + 6 신규 가드), 11 failed (전부 pre-existing).
- `pytest backend/tests/test_characterization_api.py`: 9 passed, diff 0.
- `pytest backend/tests/test_async_safety.py`: 6 passed, 2 skipped.
- `ruff check`: pre-existing baseline 유지, 신규 위반 0.

## 결정

- **소스 코드 0 라인 변경.** 진단 A/B 부재 확정, C는 차단 트리거 #7 대상이라 손대지 않음.
- 미래 회귀 가드 `test_async_safety.py` 1 건만 추가.
- 라우트 동시성 모델 (동기 def → threadpool) 유지.
- 기존 lru_cache / Lock 패턴 유지 (CP226 facade 분리에서 이미 검증).

## 후속

1. **CP229 후속 CP — strategy_scan lru_cache 동시성**: `_load_frame` (maxsize=1) / `_strategy_results` (maxsize=16) 의 동시 호출 시 race condition 실측. 첫 호출이 진행 중일 때 두 번째 동시 호출이 캐시 미스로 다시 무거운 로딩을 트리거할 수 있는지 reproduce 테스트. 필요 시 명시적 Lock 추가.
2. **product_prediction_history_svc / ai_repo 동시성**: 동일 패턴 검토.
3. **가드 강화**: HEAVY_SYNC_CALLS 목록을 코드베이스 grep 으로 자동 갱신하는 fixture (현재는 manual list). 새 무거운 함수 추가 시 가드에서 빠질 수 있음.
