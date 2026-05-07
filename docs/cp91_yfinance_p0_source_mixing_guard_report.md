# CP91-D yfinance P0 데이터 혼합 차단 보고서

## 1. Executive Summary

CP91의 핵심 P0인 `price_data`/`indicators`의 yfinance/EODHD 혼합 저장 위험은 코드와 스키마 기준으로 차단했다. `price_data`는 `(ticker, date, source)`, `indicators`는 `(ticker, timeframe, date, source)`를 unique/upsert 기준으로 사용하도록 변경했다. `compute_indicators`는 provider/source를 받아 해당 source의 `price_data`만 읽고, 산출 indicator에도 같은 source/provider를 기록한다.

단, 운영 DB 진단 결과 CP89의 5티커 제한 write 당시 기존 `UNIQUE (ticker, date)` 때문에 최근 500거래일 EODHD row가 yfinance row로 대체된 상태가 확인됐다. 현재 5티커의 `price_data`는 2015-01-02~2024-05-02가 legacy null(EODHD로 해석), 2024-05-03~2026-05-01이 yfinance다. 이 구간의 기존 `indicators`는 source가 `eodhd`로 보이지만, source-aware 재계산 전까지 실제 provenance 신뢰도가 낮다.

최종 판정은 **WARN**이다. P0 차단 구조는 들어갔지만, 50티커 write 확대 전에는 5티커 source-aware indicator 재계산과 EODHD 최근 구간 복구 여부 결정을 먼저 해야 한다.

## 2. 구현 요약

| 영역 | 변경 | 근거 |
|---|---|---|
| `price_data` unique | `(ticker, date, source)` 병렬 저장 허용 | `backend/db/schema.sql:49`, `backend/db/scripts/ensure_runtime_schema.py` |
| `indicators` unique | `(ticker, timeframe, date, source)` 병렬 저장 허용 | `backend/db/schema.sql:191`, `backend/db/schema.sql:201` |
| 가격 sync upsert | `on_conflict="ticker,date,source"` | `backend/collector/jobs/sync_prices.py:361` |
| EODHD legacy null 가드 | legacy null 최신 구간은 backfill 전 overlap 없이 다음 날짜부터 sync | `backend/collector/jobs/sync_prices.py:73` |
| indicator 계산 | provider/source 인자 추가, source별 price frame만 사용 | `backend/collector/jobs/compute_indicators.py:40`, `backend/collector/jobs/compute_indicators.py:210` |
| indicator 저장 | source/provider 기록, `ticker,timeframe,date,source` upsert | `backend/collector/jobs/compute_indicators.py:241`, `backend/collector/jobs/compute_indicators.py:274` |
| feature index/split | price_data와 indicators 모두 source-aware 필터 적용 | `ai/preprocessing.py:465`, `ai/preprocessing.py:522`, `ai/preprocessing.py:808` |
| cron/CLI 경로 | daily/CLI indicator recompute에 provider 전달 | `backend/collector/pipelines/compute_indicators_cli.py`, `backend/collector/pipelines/daily_market_sync.py`, `backend/collector/pipelines/daily_sync.py` |

## 3. DB 마이그레이션 결과

실행 명령:

```powershell
python -m backend.db.scripts.ensure_runtime_schema
```

확인된 unique constraint:

| table | constraint | definition |
|---|---|---|
| `price_data` | `price_data_ticker_date_source_key` | `UNIQUE (ticker, date, source)` |
| `indicators` | `indicators_ticker_timeframe_date_source_key` | `UNIQUE (ticker, timeframe, date, source)` |

전역 source count:

| table | source | rows |
|---|---:|---:|
| `price_data` | `__null__` | 1,384,320 |
| `price_data` | `yfinance` | 2,500 |
| `indicators` | `eodhd` | 1,650,817 |

주의: `price_data`의 null source는 기존 EODHD legacy로 해석한다. `indicators`는 이번 DDL 이후 기존 row가 `eodhd`로 보인다. 명시적 대량 UPDATE/DELETE는 실행하지 않았지만, 운영 DB에서 기존 indicator row의 source provenance는 실제 계산 당시 provider를 보장하지 않는다.

## 4. CP89 5티커 상태 진단

대상: AAPL, MSFT, NVDA, TSLA, NFLX.

| ticker | source | rows | date_min | date_max |
|---|---|---:|---|---|
| AAPL | `__null__` | 2,349 | 2015-01-02 | 2024-05-02 |
| AAPL | `yfinance` | 500 | 2024-05-03 | 2026-05-01 |
| MSFT | `__null__` | 2,349 | 2015-01-02 | 2024-05-02 |
| MSFT | `yfinance` | 500 | 2024-05-03 | 2026-05-01 |
| NFLX | `__null__` | 2,349 | 2015-01-02 | 2024-05-02 |
| NFLX | `yfinance` | 500 | 2024-05-03 | 2026-05-01 |
| NVDA | `__null__` | 2,349 | 2015-01-02 | 2024-05-02 |
| NVDA | `yfinance` | 500 | 2024-05-03 | 2026-05-01 |
| TSLA | `__null__` | 2,349 | 2015-01-02 | 2024-05-02 |
| TSLA | `yfinance` | 500 | 2024-05-03 | 2026-05-01 |

같은 ticker/date에 source가 병렬 존재하는 sample은 현재 0건이다. 이는 병렬 저장이 잘 된 결과가 아니라, CP89 당시 old unique 때문에 최근 EODHD row가 yfinance row로 대체됐을 가능성이 높다는 뜻이다.

영향 가능 indicator row:

| timeframe | per ticker rows after 2024-05-03 | 5 ticker total |
|---|---:|---:|
| 1D | 500 | 2,500 |
| 1W | 105 | 525 |
| 1M | 25 | 125 |

이 row들은 DB상 `source='eodhd'`로 보이지만, source-aware recompute 전까지는 provenance를 신뢰하면 안 된다.

## 5. Legacy Null 정책

확정 정책:

- `price_data.source IS NULL`은 EODHD legacy로 해석한다.
- yfinance mode는 `source='yfinance'` row만 읽는다.
- EODHD mode는 `source='eodhd' OR source IS NULL` row를 읽는다.
- 대량 `NULL -> eodhd` backfill은 이번 CP에서 실행하지 않았다.
- EODHD sync는 legacy null 최신 row와 overlap write를 만들지 않도록, null latest row가 있으면 `latest_date + 1`부터 시작한다.

복구 정책 제안:

1. 사용자 승인 후 5티커의 2024-05-03~2026-05-01 EODHD 가격만 제한 복구한다.
2. 복구 row는 `source='eodhd'`로 저장한다.
3. 그 뒤 5티커 `provider=eodhd`와 `provider=yfinance` indicator를 각각 source-aware recompute한다.
4. 두 provider의 feature cache hash와 manifest가 분리되는지 재확인한다.

## 6. Source-aware Indicator/Feature 계약

`compute_indicators`는 이제 active provider를 받는다. yfinance mode에서는 yfinance 가격만 읽고, EODHD mode에서는 eodhd 또는 legacy null만 읽는다. 1W/1M은 리샘플 전에 가격 프레임이 source-filtered 되므로 긴 lookback이 다른 provider 경계를 넘어갈 수 없다.

AI preprocessing도 `price_data`와 `indicators` 양쪽 source를 맞춘다. 즉 yfinance mode feature index는 `price_data.source='yfinance'`와 `indicators.source='yfinance'`가 동시에 있는 날짜만 사용한다. source가 없는 구형 schema에서 yfinance mode는 안전하게 empty로 떨어진다.

## 7. 테스트 결과

실행:

```powershell
python -m py_compile backend\collector\jobs\sync_prices.py backend\collector\jobs\compute_indicators.py backend\db\scripts\ensure_runtime_schema.py ai\preprocessing.py
python -m unittest backend.tests.test_market_data_providers backend.tests.test_collector_jobs ai.tests.test_preprocessing_cache_isolation
$env:PYTHONPATH='backend'; python -m unittest discover backend\tests
python -m unittest discover ai\tests
```

결과:

- py_compile: PASS
- targeted unittest: 27 tests PASS
- backend unittest discover: 65 tests PASS
- ai unittest discover: 183 tests PASS

추가된 주요 테스트:

- price_data schema가 source-aware unique key를 선언하는지 확인
- sync_prices upsert conflict가 `ticker,date,source`인지 확인
- EODHD legacy null 최신 구간에서 overlap start를 피하는지 확인
- compute_indicators가 yfinance price row만 읽고 indicator source를 저장하는지 확인
- EODHD 1W/1M 계산이 yfinance row를 섞지 않는지 확인
- feature index가 indicator source와 price source를 함께 맞추는지 확인

## 8. 금지 사항 준수

실행하지 않음:

- 50티커 write
- 2015년 장기 yfinance write
- 전체 yfinance write
- EODHD row 삭제
- EODHD row 대량 복구
- 모델 학습
- live inference
- product run 교체

실행한 DB write는 `ensure_runtime_schema`의 DDL 마이그레이션뿐이다.

## 9. 남은 리스크

| 등급 | 리스크 | 설명 | 다음 조치 |
|---|---|---|---|
| P0 | CP89 5티커 최근 EODHD 가격 공백 | old unique로 2024-05-03 이후 EODHD row가 yfinance row에 대체된 상태 | 제한 EODHD 복구 또는 baseline 포기 결정 |
| P0 | 기존 5티커 indicator provenance 불신 | 최근 indicator가 `eodhd`로 보이나 yfinance 기반일 수 있음 | source-aware 5티커 recompute |
| P1 | API/차트 read path source 선택 정책 | 이번 CP는 학습/수집 경로 중심이며 제품 API source filter는 별도 점검 필요 | 다음 CP에서 app read path provider 선택 명문화 |
| P1 | legacy null 대량 backfill 미실행 | null/eodhd 의미 중복이 남아 있음 | 승인 기반 단계적 backfill 계획 |

## 10. 최종 판정

**WARN.**

P0 혼합 차단 구조는 들어갔다. 하지만 운영 DB에는 CP89에서 발생한 5티커 최근 구간 EODHD 공백과 indicator provenance 불신 구간이 남아 있다. 다음 단계는 50티커 write가 아니라, 5티커 source-aware indicator 재계산과 EODHD 복구 여부 결정이다. 그 검증이 끝나면 50티커 제한 write로 넘어갈 수 있다.

