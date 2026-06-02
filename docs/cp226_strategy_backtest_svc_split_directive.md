# CP226 strategy_backtest_svc.py 분리 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 CP 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단까지 한다. 추측 금지. 모든 경로는 절대경로.

---

## 역할 고정

- **모드**: `code` (구현 모드). 이 CP는 순수 구조 리팩토링이다. 새 기능/새 수치 0.
- **권한**: 코드 수정, 로컬 검증(pytest / 스냅샷 / import 스모크 / 백엔드 기동)만.
- **금지(절대)**:
  - 새 모델 학습 / 새 calibration / 백테스트 파라미터(임계값·confirm_days·FEE_RATE·기간) 변경
  - DB write / Supabase 호출 (이 CP는 로컬 parquet만 읽는다)
  - 사용자가 직접 수정한 파일 revert
  - `_raw_target` 안의 전략 조건식, `_compute_signal_frame`의 상태머신 로직, 지표 계산식 **한 글자도 변경 금지** (값이 바뀌면 스냅샷이 깨진다 = 차단)
- **자가 점검(완료 후 보고)**: [Plan v3 정합] / [구조 결함] / [모델 영향] 각 PASS/WARN/FAIL + 사유.
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
  - 리팩토링 커밋과 기능 커밋을 절대 섞지 않는다. 이 CP는 전부 리팩토링 커밋.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `C:\Users\user\lens\.venv` (Python 3.10.0, torch 2.11.0+cu128). 활성화:
  `C:\Users\user\lens\.venv\Scripts\Activate.ps1`
- **백엔드 import root**: `C:\Users\user\lens\backend` (`app.*` 패키지). pytest/스모크는 이 디렉토리에서 실행.
- **백엔드 기동(필요 시에만)**: `C:\Users\user\lens\scripts\start_demo.ps1`, 또는
  `python -m uvicorn app.main:app --port 8010` (`backend` 디렉토리에서).
  - **포트 충돌 주의**: 8000은 사용자가 데모용으로 쓸 수 있다. 검증 기동은 `--port 8010` 등 비충돌 포트로.
- **프론트(이 CP 무관)**: `npm run dev`. 이 CP는 백엔드 전용이라 프론트 기동 불필요.
- **lint/type 도구**: 이 리포에는 `pyproject.toml` / `ruff.toml` / `mypy.ini`가 **없다**(확인함).
  ruff/mypy가 venv에 설치돼 있으면 best-effort로 돌리되, **없으면 통과로 간주하고 pytest + 스냅샷 + 컴파일 스모크로 대체**한다. 새 lint 설정 파일을 이 CP에서 만들지 않는다(범위 밖).

---

## 진단 (근거)

조사 출처: `C:\Users\user\lens\backend\app\services\strategy_backtest_svc.py` 직접 Read(전체 590줄),
`docs/refactoring_master_plan.md:62, 81`, 호출부 Grep.

대상 파일 `backend\app\services\strategy_backtest_svc.py` = **현재 590줄**(파일 끝 = 590행, 확인함).
한 파일에 책임 3개가 혼재 — SRP 위반:

1. **지표/규칙 계산 (순수)** — 부작용 없는 pandas/numpy 계산.
   - `_jsonable` (47–59), `_normalize_rsi` (62–64), `_safe` (67–68),
     `_safe_gt`/`_safe_gte`/`_safe_lt`/`_safe_lte` (71–84) **← 4개 모두 dead code**(아래 참고),
     `_align_date_dtype` (27–44, **CP214 dtype 충돌 픽스 헬퍼 — 보존 1순위**),
     `_raw_target` (239–280, 전략별 진입/청산/위험 조건식), `_reason` (283–292),
     `_compute_signal_frame` (295–332, 진입·청산 confirm-day 상태머신).
2. **백테스트 시뮬레이션 엔진 + 지표 산출 (순수)** —
   `_total_return` (87–89), `_max_drawdown` (92–98), `_sharpe` (101–106), `_sortino` (109–115),
   `_large_loss_threshold` (118–122), `_ticker_metrics` (354–396), `_average_holding_days` (399–412),
   `_trade_events` (415–431), `_points` (434–455), `_signal_row` (335–351).
   상수 `FEE_RATE = 0.001` (19), `MIN_EVAL_DAYS = 120` (20).
3. **데이터 로딩 + scan 집계 (I/O + 상태)** —
   `_data_dir` (23–24), `_load_frame` (125–223, `@lru_cache(maxsize=1)`, parquet + `parquet_store` 머지),
   `_sector_map` (226–236, `@lru_cache(maxsize=1)`), `_strategy_results` (458–518, `@lru_cache(maxsize=16)` — 티커별 시그널 + aggregate).
4. **공개 API (facade)** —
   `get_strategy_scan` (521–565), `clear_strategy_cache` (568–573), `get_strategy_backtest` (576–590).
   추가로 16행에서 `from app.strategies.strategy_rules import STRATEGIES, StrategyRule` 를 **재노출** —
   라우터가 `from app.services.strategy_backtest_svc import STRATEGIES` 로 끌어쓴다(아래 인터페이스 절).

**Dead code 확인**: `_safe_gt`/`_safe_gte`/`_safe_lt`/`_safe_lte` (71–84) 는 backend 전체 Grep 결과
**정의 외 참조 0건**. 이 CP에서 제거 후보로 다루되, 제거도 별도 step·별도 커밋으로(아래 Step 5).

**lru_cache 주의**: `_load_frame`(maxsize=1), `_sector_map`(maxsize=1), `_strategy_results`(maxsize=16).
`maxsize` 동시호출 안전성 문제는 `refactoring_master_plan.md:81`에 적혀 있고 **CP229에서 다룬다**.
이 CP226에서는 데코레이터·maxsize **그대로 보존만** 한다. 캐시 무효화는 `clear_strategy_cache`(568–573)가
세 캐시 `.cache_clear()`를 호출하므로, 분리 후에도 이 세 함수의 `.cache_clear` 접근이 깨지면 안 된다(차단 트리거).

---

## 선행 의존

- **CP223 (백엔드 characterization 스냅샷) 그린**: 필수. CP223이 strategies scan/backtest 응답을
  스냅샷으로 박제해 둔 상태여야 한다. **CP223 스냅샷이 없거나 빨강이면 이 CP를 시작하지 말고 즉시 보고**한다.
  - 시작 전 확인: `backend\tests\` 에서 strategies scan/backtest를 다루는 스냅샷/characterization 테스트가
    존재하고 현재 통과하는지 1회 실행해 그린을 확인(아래 검증 절 0번).
  - CP223 산출물(스냅샷 파일 경로·픽스처·tolerance 도구 = snaptol 여부)을 그대로 사용한다.
    이 CP에서 스냅샷 기대값(golden)을 **새로 갱신/재생성 금지**(구조만 바꾸므로 값은 불변이어야 한다).
- 그 외 선행: 없음.

---

## 범위

### 포함
- `backend\app\services\strategy_backtest_svc.py` (590줄)를 책임별 모듈로 분리.
- 새 모듈 3개 생성 + 기존 파일을 **facade(공개 API 유지)** 로 슬림화.
- dead code(`_safe_gt/gte/lt/lte`) 제거(별도 step).
- ADR 1장 + 리포트 1장.

### 제외 (건드리지 않음)
- `backend\app\strategies\strategy_rules.py` — 이미 분리된 dataclass(`StrategyRule` + `STRATEGIES`).
  **수정 금지.** 단, facade가 여기서 import해 재노출하는 구조는 유지(역할 정리는 문서로만).
- `lru_cache` maxsize 조정 → **CP229**. 여기선 보존만.
- Supabase 경로 — 이 CP는 로컬 parquet만. Supabase 호출/도입 보류.
- 전략 조건식·임계값·수치 로직 — **불변**.
- 라우터(`strategies.py`, `admin.py`) 로직 변경 — import 경로조차 바꾸지 않는다(facade로 흡수).
- 새 lint/type 설정 파일 생성.

---

## 분리 설계 (목표 모듈)

기존 flat `services/` 레이아웃과 **import 계약**(`from app.services.strategy_backtest_svc import ...`)을
지키기 위해, `strategy_backtest_svc.py`는 **facade로 유지**하고 내부를 sibling private 모듈로 추출한다.
이러면 호출자(라우터) 변경이 0이 되어 인터페이스 보존이 구조적으로 보장된다.

| 새 파일(절대경로 기준 `backend\app\services\`) | 담는 것 | 성격 |
|---|---|---|
| `strategy_indicators.py` | `_align_date_dtype`, `_jsonable`, `_normalize_rsi`, `_safe`, `_raw_target`, `_reason`, `_compute_signal_frame` | 순수 계산(규칙/지표/상태머신) |
| `strategy_backtest_engine.py` | `FEE_RATE`, `_total_return`, `_max_drawdown`, `_sharpe`, `_sortino`, `_large_loss_threshold`, `_average_holding_days`, `_ticker_metrics`, `_signal_row`, `_trade_events`, `_points` | 순수 백테스트 엔진 + 지표 산출 |
| `strategy_scan.py` | `MIN_EVAL_DAYS`, `_data_dir`, `_load_frame`(lru_cache 보존), `_sector_map`(lru_cache 보존), `_strategy_results`(lru_cache 보존) | I/O + 상태(캐시) + 집계 |
| `strategy_backtest_svc.py` (facade, 잔존) | `STRATEGIES`/`StrategyRule` 재노출, `get_strategy_scan`, `get_strategy_backtest`, `clear_strategy_cache` | 공개 API |

설계 근거(추출 순서 = 순수함수 → I/O 경계 → 상태 의존):
- `strategy_indicators` / `strategy_backtest_engine` 는 부작용 없는 순수 함수라 **먼저** 옮긴다(Step 1–2).
- `strategy_scan` 은 parquet I/O + lru_cache(상태)라 **나중에** 옮긴다(Step 3).
- facade의 `clear_strategy_cache` 는 `strategy_scan`의 세 캐시 함수 `.cache_clear()`를 호출해야 하므로,
  `strategy_scan` 추출 후 facade가 그 함수들을 import해 호출하도록 연결(Step 3에서 함께).

> 주의: `_strategy_results`(scan 모듈)는 `_compute_signal_frame`(indicators)·`_ticker_metrics`(engine)을
> 호출한다. 즉 `strategy_scan` → `strategy_indicators` + `strategy_backtest_engine` 단방향 의존이다.
> `get_strategy_scan`/`get_strategy_backtest`(facade)는 `_signal_row`/`_trade_events`/`_points`/`_ticker_metrics`(engine)와
> `_strategy_results`/`_sector_map`(scan)을 호출한다. **순환 import 금지** — 의존은 facade→scan→(indicators,engine),
> scan→(indicators,engine) 의 단방향만. indicators/engine은 서로/상위를 import하지 않는다.

---

## Sub-step (Strangler Fig, 작은 단위)

각 Step은 "옛 코드 옆 새 코드 공존 → caller 이전 → 옛 제거" 패턴. 한 Step = 한 commit = 한 revert 단위.
**모든 Step 끝에 검증 절의 명령을 돌려 pytest 그린 + 스냅샷 diff 0을 확인**한 뒤 커밋한다.

### Step 0 — 선행 그린 확인 (코드 변경 없음)
- 검증 절 0번 실행. CP223 strategies 스냅샷 테스트가 **그린**인지 확인. 빨강/부재면 **중단·보고**.
- 현재 `strategy_backtest_svc.py` 줄 수 기록(기대 590).
- 커밋 없음(확인 단계).

### Step 1 — `strategy_indicators.py` 추출 (순수 규칙/지표)
- 새 파일 `backend\app\services\strategy_indicators.py` 생성. 옮길 함수(원본에서 **잘라내기**):
  `_align_date_dtype`, `_jsonable`, `_normalize_rsi`, `_safe`, `_raw_target`, `_reason`, `_compute_signal_frame`.
  - 함수 본문은 **바이트 단위로 동일**하게 옮긴다(주석·docstring 포함). 로직 수정 0.
  - 새 파일 상단 import: `from __future__ import annotations`, `math`(필요 시), `numpy as np`, `pandas as pd`,
    `from fastapi import HTTPException`(`_raw_target`이 404 raise), `from typing import Any`,
    `from app.strategies.strategy_rules import StrategyRule`.
- 원본 `strategy_backtest_svc.py` 는 이 함수들을 **import로 대체**:
  `from app.services.strategy_indicators import (_align_date_dtype, _jsonable, _normalize_rsi, _safe, _raw_target, _reason, _compute_signal_frame)`.
  - `_load_frame`이 쓰는 `_align_date_dtype`, `_normalize_rsi`, `_strategy_results`가 쓰는 `_compute_signal_frame`,
    `get_strategy_scan`/`_signal_row`가 쓰는 `_jsonable` 등 **모든 내부 호출이 import된 이름을 가리키는지** 확인.
- 검증(검증 절 1·2·3) → 그린·diff 0이면 commit: `refactor(be): extract strategy_indicators from strategy_backtest_svc`.

### Step 2 — `strategy_backtest_engine.py` 추출 (순수 엔진 + 지표)
- 새 파일 `backend\app\services\strategy_backtest_engine.py` 생성. 옮길 것:
  `FEE_RATE` 상수, `_total_return`, `_max_drawdown`, `_sharpe`, `_sortino`, `_large_loss_threshold`,
  `_average_holding_days`, `_ticker_metrics`, `_signal_row`, `_trade_events`, `_points`.
  - `_signal_row`/`_points`/`_trade_events`/`_ticker_metrics` 가 쓰는 `_jsonable` 은
    `from app.services.strategy_indicators import _jsonable` 로 가져온다.
  - import: `from __future__ import annotations`, `math`, `numpy as np`, `pandas as pd`, `from typing import Any`,
    `from app.strategies.strategy_rules import StrategyRule`(`_signal_row(row, rule)` 시그니처).
- 원본/`strategy_scan`에서 `FEE_RATE` 와 위 함수들을 쓰는 곳을 engine import로 교체.
  - `FEE_RATE`는 `_ticker_metrics`/`_points` 내부에서만 쓰이므로 engine으로 옮기면 충분.
    facade에서 `FEE_RATE`를 직접 참조하던 코드는 없음(확인). 원본의 `FEE_RATE = 0.001` 라인 제거.
- 검증 → commit: `refactor(be): extract strategy_backtest_engine from strategy_backtest_svc`.

### Step 3 — `strategy_scan.py` 추출 (I/O + 캐시 + 집계)
- 새 파일 `backend\app\services\strategy_scan.py` 생성. 옮길 것:
  `MIN_EVAL_DAYS` 상수, `_data_dir`, `_load_frame`(`@lru_cache(maxsize=1)` 데코레이터 **그대로**),
  `_sector_map`(`@lru_cache(maxsize=1)`), `_strategy_results`(`@lru_cache(maxsize=16)`).
  - import: `from __future__ import annotations`, `from functools import lru_cache`, `from pathlib import Path`,
    `numpy as np`, `pandas as pd`, `from typing import Any`, `from fastapi import HTTPException`,
    `from app.services import parquet_store`,
    `from app.strategies.strategy_rules import STRATEGIES, StrategyRule`,
    `from app.services.strategy_indicators import _align_date_dtype, _normalize_rsi, _compute_signal_frame`,
    `from app.services.strategy_backtest_engine import _ticker_metrics`.
- facade(`strategy_backtest_svc.py`)에서 `_strategy_results`, `_sector_map`, `_load_frame`(cache_clear용)을 scan import로 교체:
  - `get_strategy_scan`/`get_strategy_backtest` 는 `from app.services.strategy_scan import _strategy_results, _sector_map`.
  - `clear_strategy_cache` 는 `from app.services.strategy_scan import _load_frame, _sector_map, _strategy_results` 후
    `_load_frame.cache_clear(); _strategy_results.cache_clear(); _sector_map.cache_clear()` 그대로 호출.
    **세 함수가 여전히 lru_cache 래퍼라 `.cache_clear` 속성이 살아있는지 확인**(차단 트리거).
- 검증 → commit: `refactor(be): extract strategy_scan (load/cache/aggregate) from strategy_backtest_svc`.

### Step 4 — facade 정리 + 재노출 보존
- 이 시점 `strategy_backtest_svc.py` 에 남는 것: `from __future__ import annotations`,
  `from typing import Any`, `import pandas as pd`,
  `from app.strategies.strategy_rules import STRATEGIES, StrategyRule` (재노출 유지),
  engine/scan/indicators 에서 필요한 심볼 import,
  그리고 `get_strategy_scan`, `clear_strategy_cache`, `get_strategy_backtest` 본문(로직 불변).
- `STRATEGIES`, `StrategyRule` 가 facade 네임스페이스에 그대로 노출돼
  `from app.services.strategy_backtest_svc import STRATEGIES` (strategies.py 6–10행)가 깨지지 않는지 확인.
- 파일 상단에 한 줄 주석으로 "이 파일은 facade. 계산은 strategy_indicators / 엔진은 strategy_backtest_engine /
  로딩·집계는 strategy_scan" 명시.
- 검증 → commit: `refactor(be): slim strategy_backtest_svc to public facade`.

### Step 5 — dead code 제거 (`_safe_gt/gte/lt/lte`)
- Step 1에서 `strategy_indicators.py`로 함께 옮겨졌을 4개 미사용 헬퍼(`_safe_gt`,`_safe_gte`,`_safe_lt`,`_safe_lte`)를 제거.
  - 제거 전 한 번 더 Grep으로 backend 전역 참조 0 재확인.
- 검증 → commit: `refactor(be): drop unused _safe_gt/gte/lt/lte helpers`.

> Step 5를 분리하는 이유: dead code 제거(동작 무관)와 구조 이동을 한 커밋에 섞지 않기 위함. 단독 revert 가능.

---

## 인터페이스 보존

다음을 **바꾸지 않는다**(바꿔야 하면 호출자 영향 분석 후 차단 보고):

1. **공개 함수 시그니처** (모듈 `app.services.strategy_backtest_svc` 네임스페이스에서 그대로 import 가능해야 함):
   - `get_strategy_scan(strategy_id: str, limit: int = 500) -> dict[str, Any]`
   - `get_strategy_backtest(strategy_id: str, ticker: str) -> dict[str, Any]`
   - `clear_strategy_cache() -> None`
   - 재노출 심볼: `STRATEGIES`, `StrategyRule`
   - 호출부: `backend\app\routers\v1\strategies.py:6-10` (`STRATEGIES, get_strategy_backtest, get_strategy_scan`),
     `backend\app\routers\v1\admin.py:16` (`clear_strategy_cache`).
2. **API 응답 schema**: `get_strategy_scan` 반환 dict의 키
   (`strategyId, strategyLabel, timeframe, asofDate, scopeTickerCount, usableSignalCount,
   latestValidTickerCount, cards[...], portfolioMetrics, aggregateMetrics, contract`)와
   `get_strategy_backtest` 반환(`points, signals, tradeEvents, **metrics`)의 **모든 키·타입·값** 불변.
   라우터가 `success_response(...)`로 `{"data": <위 dict>, "meta": {"request_id": ...}}` 래핑하는 것도 그대로.
3. **캐시 인터페이스**: `clear_strategy_cache`가 비우는 3개 lru_cache(`_load_frame`,`_strategy_results`,`_sector_map`).
   분리 후에도 `.cache_clear()` 호출 가능해야 한다.

> 라우터 파일은 import 줄조차 변경하지 않는다(facade 흡수). 만약 facade 유지가 불가능한 상황이 보이면
> 그냥 라우터를 고치지 말고 **중단·보고**한다.

---

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 |
|---|---:|---|
| `strategy_backtest_svc.py` 줄 수 | 590 | facade ≤ **120** |
| `strategy_indicators.py` 줄 수 | 0 | ≤ **250** |
| `strategy_backtest_engine.py` 줄 수 | 0 | ≤ **250** |
| `strategy_scan.py` 줄 수 | 0 | ≤ **250** |
| 4개 파일 각각 | — | **각 ≤ 250** (분리안 핵심: 590 → 각 250 이내) |
| strategies **scan** 응답 스냅샷 diff | 0 | **0** (CP223 golden 대비) |
| strategies **backtest** 응답 스냅샷 diff | 0 | **0** |
| 기존 pytest (`backend\tests\`) | green | **회귀 0** (전부 동일 통과) |
| import 스모크 (`from app.services.strategy_backtest_svc import ...`) | ok | ok |
| `python -m compileall` 새 4파일 | — | 에러 0 |
| mypy 추가 에러 | — | **0 추가** (도구 있으면) / tsc 해당 없음(BE) |
| 예상 시간 | — | **약 2.5시간** |

---

## 검증

`backend` 디렉토리에서 실행(venv 활성 상태). 명령은 PowerShell 기준.

### 0) 선행 그린 + 베이스라인 (Step 0)
```powershell
# CP223 strategies 스냅샷 테스트 그린 확인 (테스트 파일명은 CP223 산출물에 맞춰 조정)
python -m pytest backend\tests -k "strateg or scan or backtest or snapshot" -q
# 대상 파일 줄 수 베이스라인
(Get-Content backend\app\services\strategy_backtest_svc.py | Measure-Object -Line).Lines   # 기대 590
```
기대: 스냅샷 관련 테스트 **passed**(또는 해당 테스트가 존재). 빨강/부재면 중단·보고.

### 1) 컴파일 스모크 (각 Step 후)
```powershell
python -m compileall backend\app\services\strategy_backtest_svc.py `
  backend\app\services\strategy_indicators.py `
  backend\app\services\strategy_backtest_engine.py `
  backend\app\services\strategy_scan.py
```
기대: `Compiling ...` 만, 에러/Traceback 0.

### 2) import + facade 보존 스모크 (각 Step 후)
```powershell
python -c "from app.services.strategy_backtest_svc import STRATEGIES, StrategyRule, get_strategy_scan, get_strategy_backtest, clear_strategy_cache; print('facade ok', sorted(STRATEGIES))"
python -c "from app.routers.v1 import strategies, admin; print('routers import ok')"
python -c "from app.services.strategy_scan import _load_frame, _sector_map, _strategy_results; assert all(hasattr(f,'cache_clear') for f in (_load_frame,_sector_map,_strategy_results)); print('cache_clear ok')"
```
기대: `facade ok ['ai_balance_v2', 'ai_band_defense_v1', 'indicator_balance_v2']`, `routers import ok`, `cache_clear ok`.

### 3) 전체 회귀 + 스냅샷 (각 Step 후, 특히 scan/backtest)
```powershell
python -m pytest backend\tests -q
```
기대: 시작 시점과 **동일한 passed 수**, 신규 실패 0. scan/backtest 스냅샷 diff 0.

### 4) (선택) 런타임 동치 확인 — 데이터 있을 때
`backend\data\v1\*.parquet` 가 있으므로 facade를 직접 호출해 키 구조를 눈으로 확인 가능(값 비교는 스냅샷이 담당):
```powershell
python -c "from app.services.strategy_backtest_svc import get_strategy_scan; d=get_strategy_scan('indicator_balance_v2', limit=5); print(sorted(d.keys())); print('cards', len(d['cards']))"
```
기대: scan dict 키 집합이 분리 전과 동일.

### 5) lint/type (도구 있을 때만, best-effort)
```powershell
ruff check backend\app\services\strategy_indicators.py backend\app\services\strategy_backtest_engine.py backend\app\services\strategy_scan.py backend\app\services\strategy_backtest_svc.py
```
도구 미설치면 생략(범위 밖). 설치돼 있고 새 에러가 나면 그 에러만 정리(F401 미사용 import 등). 동작 변경 금지.

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고**한다. 그냥 넘어가기 절대 금지.

1. **CP223 strategies 스냅샷이 부재하거나 빨강** — 안전망 없이 분리 진행 금지. Step 0에서 멈춤.
2. **scan 응답 스냅샷 diff 발생** — `get_strategy_scan` 출력이 한 글자라도 바뀜 = 동작 변경.
   백테스트/스캔 수치는 사용자 신뢰의 핵심이다. 어떤 키/값/정렬/소수도 달라지면 멈추고 원인 보고.
3. **backtest 응답 스냅샷 diff 발생** — `get_strategy_backtest`의 points/signals/tradeEvents/metrics 중
   하나라도 달라지면 멈춤. (특히 `_ticker_metrics`/`_points`의 `FEE_RATE`·shift·cumprod 순서 변형 의심.)
4. **기존 pytest 다수 실패 / 신규 실패** — 분리 전 통과하던 테스트가 깨지면 멈춤.
5. **facade import 깨짐** — `from app.services.strategy_backtest_svc import STRATEGIES/get_strategy_scan/...`
   또는 라우터 import 실패 = 인터페이스 위반. 라우터를 고쳐 우회하지 말고 멈춤.
6. **순환 import / `.cache_clear` 소실** — 모듈 분리로 import 사이클이 생기거나
   `clear_strategy_cache`가 호출하는 세 함수의 `.cache_clear`가 사라지면(예: 헬퍼로 감싸 lru_cache 래퍼가 가려짐) 멈춤.
7. **백엔드 기동 실패(환경변수 누락 등)** — 검증용 기동 시 `LENS_*`/Supabase 미설정으로 import 단계가 죽으면,
   그게 이 CP 변경 탓인지 환경 탓인지 구분해 보고(분리 무관한 환경 문제면 그대로 보고).
8. **lru_cache maxsize·전략 조건식·임계값을 바꿔야만 통과하는 상황** — 그건 이 CP 범위가 아니다(CP229/기능). 멈춤.

---

## ADR

완료 후 `C:\Users\user\lens\docs\adr\0016-strategy-backtest-split.md` 작성(200~300단어).
- `docs\adr\` 디렉토리가 없으면 생성.
- 기록할 것: (1) 왜 facade 패턴(원본 파일을 공개 API로 남기고 내부만 sibling 모듈로 추출)을 택했는지 —
  라우터 import 계약 무변경 보장. (2) 모듈 경계 3분할 근거(순수 규칙/지표 vs 순수 백테스트 엔진 vs I/O+캐시+집계)와
  단방향 의존(facade→scan→{indicators,engine}). (3) `_align_date_dtype`(CP214) 보존 결정과 위치(indicators).
  (4) lru_cache maxsize는 보존만 하고 조정은 CP229로 미룬 이유. (5) dead code 제거를 별도 커밋으로 분리한 이유.
  대안(패키지 디렉토리 `services/strategy/`로 옮겨 import 경로 변경)을 기각한 이유(호출자 churn + 스냅샷 위험)도 한 줄.

---

## 자가 점검 결과 양식 (완료 후 채움)

- **[Plan v3 정합]** PASS/WARN/FAIL — 사유: ____
  (밴드 본체·fidelity·EODHD·α=1/β=2·backtest cost 철학에 영향 0이어야 PASS. 이 CP는 구조만.)
- **[구조 결함]** PASS/WARN/FAIL — 사유: ____
  (순환 import 0, 단방향 의존, facade 인터페이스 보존, 각 파일 ≤250줄.)
- **[모델 영향]** PASS/WARN/FAIL — 사유: ____
  (전략 수치·예측 출력 불변 = scan/backtest 스냅샷 diff 0. 학습/calibration 무관.)

---

## 산출물

1. 변경/생성 파일:
   - `backend\app\services\strategy_backtest_svc.py` (facade로 축소, ≤120줄)
   - `backend\app\services\strategy_indicators.py` (신규)
   - `backend\app\services\strategy_backtest_engine.py` (신규)
   - `backend\app\services\strategy_scan.py` (신규)
   - `docs\adr\0016-strategy-backtest-split.md` (신규)
2. `docs\cp226_report.md` — 섹션: **요구 / 한 일(Step별 커밋 해시) / 결정(ADR 요약) / 후속(CP229 lru_cache 등)**.
   필요한 만큼만. 각 파일 최종 줄 수 표, scan/backtest 스냅샷 diff 0 증빙, pytest passed 수(전/후) 포함.
