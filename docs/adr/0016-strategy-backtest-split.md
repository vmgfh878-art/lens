# ADR-0016: strategy_backtest_svc 4-module split (indicators / engine / scan / facade)

Status: Accepted
Date: 2026-06-03
Context: CP226 (refactoring runbook CP221~237)

## Context

`backend/app/services/strategy_backtest_svc.py` 는 659 줄 (CP224b 직후, dead code 제거 후) 단일 파일에 책임 3 종 + 공개 API 를 묶고 있었다:

1. **순수 규칙 / 지표 / 상태머신** — `_align_date_dtype` (CP214 dtype 가드), `_jsonable`, `_normalize_rsi`, `_safe`, `_raw_target` (전략 진입/청산/위험 조건), `_reason`, `_compute_signal_frame` (confirm-day 상태머신).
2. **순수 백테스트 엔진** — `FEE_RATE`, `_total_return`, `_max_drawdown`, `_sharpe`, `_sortino`, `_large_loss_threshold`, `_signal_row`, `_ticker_metrics`, `_average_holding_days`, `_trade_events`, `_points`.
3. **I/O + 캐시 + 집계** — `MIN_EVAL_DAYS`, `_data_dir`, `_load_frame` (@lru_cache), `_sector_map` (@lru_cache), `_strategy_results` (@lru_cache).
4. **공개 API** — `get_strategy_scan`, `get_strategy_backtest`, `clear_strategy_cache`, `STRATEGIES`/`StrategyRule` 재노출.

라우터 (`strategies.py`, `admin.py`) 는 `from app.services.strategy_backtest_svc import STRATEGIES, get_strategy_backtest, get_strategy_scan, clear_strategy_cache` 로 facade 5 심볼만 사용.

## Decision

**facade 패턴**으로 분리. 원본 `strategy_backtest_svc.py` 는 공개 API 4 함수의 본체 + 3 sibling 모듈 re-export 만 남긴 얇은 facade 로 축소한다. 라우터 import 계약 (`app.services.strategy_backtest_svc.X`) 무변경.

```
strategy_indicators.py      (178 줄)  ← strategy_rules
        ↑
strategy_backtest_engine.py (206 줄)  ← strategy_rules + strategy_indicators
        ↑
strategy_scan.py            (263 줄)  ← strategy_rules + strategy_indicators + strategy_backtest_engine + parquet_store
        ↑
strategy_backtest_svc.py    (148 줄)  ← 3 sibling 모듈 + strategy_rules
```

의존 방향은 단방향. 순환 import 없음.

`clear_strategy_cache` 는 `strategy_scan` 의 세 lru_cache wrapper (`_load_frame`, `_sector_map`, `_strategy_results`) 의 `.cache_clear()` 를 그대로 호출 — `lru_cache` 데코레이터를 sibling 모듈로 옮긴 후에도 wrapper attribute 보존 (검증).

## Consequences

긍정:
- 라우터 / `admin` 0 라인 수정. facade import 계약 보존이 구조적으로 보장.
- 전략 조건식 (indicators) / 지표 계산 (engine) / parquet 로딩 (scan) 을 따로 읽고 따로 테스트할 수 있다.
- CP214 `_align_date_dtype` (날짜 컬럼 categorical 가드) 가 indicators 모듈에 그대로 보존 — 향후 다른 source 추가 시에도 머지 직전 호출 패턴 유지.

부정 / 미해결:
- `strategy_backtest_svc.py` 148 줄 — 목표 120 의 +28 (docstring + 공개 함수 본체 보존으로 자연 초과). facade 의도가 명확하므로 더 줄이는 것은 가치 낮음.
- `strategy_scan.py` 263 줄 — 목표 250 살짝 초과 (+13). `_load_frame` 이 5 source merge 로직을 담아 본질적으로 큼. 추가 분리는 별도 후속 CP.
- `lru_cache(maxsize=1/16)` 동시 호출 안전성은 CP229 에서 다룬다 (마스터플랜). 본 CP 는 데코레이터 위치 이동 + maxsize 보존만.

대안 및 기각 이유:
- (a) **`services/strategy/` 패키지로 옮기기** — 호출자 churn (`from app.services.strategy_backtest_svc` → `from app.services.strategy.X`) + 스냅샷 위험. 라우터 무수정 보장 안 됨.
- (b) **함수 그대로 두기** — Plan v3 fidelity 추적 / 변경 영향 분석 비용 누적. SRP 위반.

Dead code (`_safe_gt`, `_safe_gte`, `_safe_lt`, `_safe_lte`) 제거는 CP224b 에서 이미 처리됨 → 본 CP Step 5 는 skip.

Fidelity 보장:
- CP223 strategies scan/backtest 응답 syrupy snapshot, 매 Step diff 0.
- `_ticker_metrics` / `_points` 의 `FEE_RATE` 곱, `shift`, `cumprod` 순서 1 비트 변경 없음.
- `_raw_target` 의 전략별 임계값 (`ma60 >= 0.02`, `rsi < 75.0` 등) 그대로.
- `backend/tests` 87 passed (CP224b/CP225 baseline 유지).
