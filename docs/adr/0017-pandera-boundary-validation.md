# ADR-0017: pandera DataFrameModel at parquet read/merge boundaries

Status: Accepted
Date: 2026-06-03
Context: CP227 (refactoring runbook CP221~237)

## Context

CP214 사고 (2026-05-30): `parquet_store._compress_strings` 가 `asof_date`/`forecast_date` 까지 ordered Categorical 로 변환 → `strategy_backtest_svc._load_frame` 의 머지 단계에서 `Categorical[datetime] vs datetime64[ns]` 충돌로 1W 밴드 / 전략 스캔 / 백테스트가 라우터 500. 라우터 단의 응답까지 와서야 발견됐다.

근본 fix 는 `_compress_strings` 가 날짜 컬럼을 categorical 에서 제외하도록 변경 + `_align_date_dtype` 헬퍼를 머지 직전 방어선으로 두는 것이었다. 그러나:
- 새 source / 컬럼이 추가되거나 parquet 가 재생성되면 같은 사고가 silent 하게 다시 들어올 수 있다.
- 라우터에 도달하기 전 (read 단계) 에 dtype 계약 위반을 잡는 장치가 없다.

## Decision

`backend/app/schemas/frames.py` 에 5 개 `pandera.DataFrameModel` 정의 (line_1d / band_1d / band_1w / market_prices_1d / market_indicators_1d) + read/merge 경계 4 곳에 검증을 끼웠다:

1. `parquet_store._load`: `pd.read_parquet` 직후 / `_compress_strings` 직전.
2. `local_market_svc._load`: `pd.read_parquet` 직후 / `strftime` 직전.
3. `strategy_scan._load_frame`: `_align_date_dtype` 직후 / merge 직전 4 프레임 `date` dtype assert.
4. (Step 4) 재발생 / 삼킴 `except Exception` 5 곳에 logger.warning / logger.debug + exc_info=exc 추가 → 원인이 로그에 남아 진단 가능.

**핵심 제약 (어기면 CP214 회귀)**: 스키마는 `asof_date`/`forecast_date`/`date` 를 `Series[str]` + `coerce=True` 로 모델링. disk dtype 이 object(str) 이고 `parquet_store._compress_strings` 가 의도적으로 유지하기 때문. datetime 으로 강제하면 categorical 변환 단계 또는 라우터 비교 단계에서 다시 깨진다.

`Config: strict = False` — disk 의 추가 컬럼 (예: line_1d 의 `actual_h5_return`, `*_flag`) 을 허용. 핵심 컬럼만 계약화.

`lazy=True` — 모든 위반을 한 번에 수집해 `SchemaErrors` 로 전파.

## Consequences

긍정:
- CP214 류 dtype 사고를 read 순간 차단. 라우터 500 으로 늦게 표면화하지 않는다.
- 운영 parquet 5 종 검증 통과 확인 (line 182732 / band1d 597160 / band1w 186900 / prices 137632 / indicators 137565). 스키마는 현재 disk 와 byte-identical 계약.
- admin `debug-state` 의 traceback / 예외 메시지 / 예외 타입 노출 제거 → 익명 호출자가 내부 모듈 구조 / 절대경로를 추출할 수 없다.
- 5 곳의 except 로깅으로 운영 중 진짜 원인 (pyarrow 깨진 parquet vs Supabase 네트워크) 이 로그에 남는다.

부정 / 미해결:
- `strategy_scan._load_frame` 이 직접 `pd.read_parquet(market_*)` 하는 경로 (line 38, 66) 는 pandera 스키마를 우회. 대신 머지 직전 datetime64 assert 로 dtype 만 가드. `_load_frame` 의 자체 정규화 로직 (groupby pct_change, transform 등) 이 disk dtype 과 다른 출력을 만들기 때문에 read 직후 스키마를 강제하면 검증 실패. 후속에서 `local_market_svc` 경유로 통일하는 안 검토.
- `HTTPException` → `AppError` 표준화는 본 CP 범위 밖 (응답 schema 변경 위험). 별도 CP.
- Supabase 호출 경로의 `except Exception` 좁히기는 클라이언트 예외 타입 불명확 + Supabase 도입 재개 시 함께 검토 (Plan v3 — Supabase 보류).

대안 및 기각 이유:
- (a) **스냅샷 만으로 보호** (CP223 syrupy) — 정상 경로 응답을 박제하지만 dtype 계약 자체를 보호하지 않는다. 새 컬럼이 추가되거나 dtype 이 바뀌어도 응답이 동일하면 스냅샷은 통과. CP214 류 사고는 스냅샷 박제 *시점* 이후 데이터가 흔들리면 잡지 못한다.
- (b) **모든 함수에 직접 assert** — 중복 + 책임 분산. pandera 가 한 곳에 모은 schema spec 으로 검증을 위임하는 것이 명확하다.

Fidelity 보장:
- 검증은 `coerce=True` 지만 disk dtype 과 스키마가 일치하므로 정상 경로에서 dtype/값을 바꾸지 않는다.
- CP223 syrupy snapshot 9 endpoint, 본 CP 4 step 전체 diff 0.
- `backend/tests` 87 passed (CP226 baseline 유지).
