# CP226 보고서 — strategy_backtest_svc 4-module split

## 요구

`docs/cp226_strategy_backtest_svc_split_directive.md`. 590 줄 (실측 659 — CP214 `_align_date_dtype` 추가 이후) `backend/app/services/strategy_backtest_svc.py` 를 facade + 3 sibling 모듈 (indicators / engine / scan) 로 분리. 라우터 import 계약 (5 심볼) 0 변경. 계산 / 집계 / 시뮬레이션 동작 1bit 보존.

## 한 일

4 step, 4 commit (Step 5 dead code 제거는 CP224b 에서 이미 처리됨 → skip):

| Step | 커밋 | 작업 | 검증 |
|---|---|---|---|
| 1 | 7031052 | strategy_indicators.py 추출 (178 줄) — 7 함수 | ruff/mypy/snapshot PASS |
| 2 | b4e4116 | strategy_backtest_engine.py 추출 (206 줄) — FEE_RATE + 11 함수 | ruff/mypy/snapshot PASS |
| 3 | bfdfec1 | strategy_scan.py 추출 (263 줄) — MIN_EVAL_DAYS + 4 함수 (3 lru_cache 포함) | ruff/mypy/snapshot PASS |
| 4 | 3c70d05 | strategy_backtest_svc.py → facade (148 줄), docstring | ruff/mypy/snapshot PASS |

## 핵심 컴포넌트 존재 체크리스트

- **lru_cache 보존**: `_load_frame`(maxsize=1), `_sector_map`(maxsize=1), `_strategy_results`(maxsize=16) 셋 다 `strategy_scan.py` 로 데코레이터 그대로 이동. `clear_strategy_cache.cache_clear()` 호출 가능 (검증 통과).
- **CP214 dtype 가드**: `_align_date_dtype` 는 `strategy_indicators.py` 에 보존. `_load_frame` 이 머지 직전 4 source (price/indicators/line/band) 모두에 호출하는 패턴 유지.
- **전략 조건식 (값 보존)**: `_raw_target` 의 `indicator_balance_v2` (`ma60 >= 0.02`, `ma20 >= -0.02`, `macd >= 0.0`, `rsi < 75.0`, `bb <= 0.35`, `rsi < 55.0`, `ma60 <= -0.05`, `ma20 <= -0.05`, `atr >= 0.07`), `ai_balance_v2` (`line >= -0.02`, `lower >= -0.06` 등), `ai_band_defense_v1` (`rsi < 82.0`, `bb <= 0.45`, `width_expansion < 1.60` 등) 모두 그대로.
- **상태머신**: `_compute_signal_frame` 의 `entry_confirm_days` / `exit_confirm_days` 카운터 로직 그대로.
- **수수료 / 시뮬레이션**: `FEE_RATE = 0.001`, `shifted * returns - trades * FEE_RATE`, `np.cumprod(1.0 + nan_to_num)` 순서 그대로.
- **드로우다운**: `np.maximum.accumulate(equity)` 패턴 그대로.
- **샤프 / 소르티노**: `np.std(usable, ddof=1)` + `* sqrt(252)` 그대로.
- **MIN_EVAL_DAYS = 120**, 365 일 컷오프 (`end_date - pd.Timedelta(days=365)`) 그대로.
- **공개 facade 5 심볼**: `STRATEGIES`, `StrategyRule`, `get_strategy_scan`, `get_strategy_backtest`, `clear_strategy_cache` — 라우터 import 0 수정.

## 새 테스트 결과

이번 CP 는 신규 테스트 추가 없음 (구조 이동만). 기존 `backend/tests` 그대로 사용.

## dry-run 결과

`get_strategy_scan` / `get_strategy_backtest` 응답은 CP223 syrupy snapshot 으로 박제. 매 Step:
- `pytest backend/tests/test_characterization_api.py -k "strateg or scan or backtest"` → 2 snapshots passed, diff 0.
- 응답 dict 의 `cards[...]`, `portfolioMetrics`, `aggregateMetrics`, `contract`, `points`, `signals`, `tradeEvents`, `**metrics` 키·값·정렬 모두 보존.

`lower<=upper` / `line_preserved` 는 forecast 출력 검증 항목으로 본 CP 범위 밖. backtest 응답의 `feeAdjustedReturnPct` / `maxDrawdownPct` 등 수치 보존이 본 CP 의 핵심 fidelity 요건이며, snapshot 으로 박제됨.

## 기존 회귀 통과 건수

- `pytest backend/tests` (test_services.py 제외): **87 passed** — CP224b / CP225 baseline 그대로.
- `pytest backend/tests` (전체): 87 passed + 11 failed (전부 pre-existing 다른 파일, 본 CP 영향 0).
- facade import (`STRATEGIES, StrategyRule, get_strategy_scan, get_strategy_backtest, clear_strategy_cache`): 통과.
- lru_cache wrapper `.cache_clear` attribute: 보존 (검증 통과).
- `ruff check`: 4 errors (둘 다 pre-existing UP038 3 + B007 1, 위치만 모듈 분리에 따라 이동).
- `mypy`: 1 error (pre-existing var-annotated, 위치만 engine 으로 이동).

## 결정

- 4 모듈 단방향 의존: strategy_rules ← strategy_indicators ← strategy_backtest_engine ← strategy_scan ← strategy_backtest_svc (facade).
- `strategy_backtest_svc.py` 는 facade 로 라우터 import 계약 보존.
- `strategy_scan.py` 263 줄 — 목표 250 의 +13 (`_load_frame` merge 로직 본질적 크기). 추가 분리는 후속 권고.
- `strategy_backtest_svc.py` 148 줄 — 목표 120 의 +28 (docstring + 공개 함수 본체). facade 의도 명확.
- **CP226 Step 5 (dead code 제거) skip**: `_safe_gt/_safe_gte/_safe_lt/_safe_lte` 4 함수는 CP224b 에서 이미 grep-verified 제거됨. 본 CP 시작 시점 strategy_backtest_svc.py 에 부재.
- lru_cache maxsize 조정은 본 CP 범위 밖 (CP229).

## 후속

1. **CP229 — lru_cache 동시 호출 안전성**: `_load_frame` (maxsize=1) 의 동시 호출 시 race condition / 메모리 폭증 가능성 검토. 본 CP 는 위치만 이동, 동작 보존.
2. **caller 점진 이전**: 라우터를 새 모듈 경로 (예: `from app.services.strategy_scan import _strategy_results`) 로 옮기는 작업은 별도 후속 CP. 본 CP 의 facade 덕에 시한 압박 없음.
3. **`strategy_scan.py` 추가 분리 검토**: `_load_frame` 의 4 source merge 로직을 별도 함수 (`_load_price_frame`, `_load_indicators_frame`, `_load_line_frame`, `_load_band_frame`) 로 쪼개 size 조정 검토.
