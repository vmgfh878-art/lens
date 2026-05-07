# 데이터 마이그레이션 상태 인벤토리

작성일: 2026-05-04  
범위: CP86부터 CP96, CP-SUPA, CP-SUPA2까지의 데이터 provider 전환과 비용 가드 변경점 감사  
주의: 이 문서는 코드 수정, DB write, DB 대량 read 없이 기존 문서와 코드 계약을 기준으로 작성했다.

## 1. 요약

EODHD에서 yfinance로의 전환은 단순 provider 교체가 아니라, source-aware 병렬 저장, adjusted OHLC 재검증, indicator source 분리, feature cache 격리, Supabase egress 절감까지 포함한 데이터 아키텍처 전환이다.

현재 상태는 다음과 같이 분류한다.

| 영역 | 상태 | 판단 |
|---|---|---|
| yfinance 1D 가격 품질 | 100티커 장기 검증 통과 | 전체 universe 전 검증 확대 필요 |
| source-aware schema | 구현됨 | 운영 DB migration 상태는 별도 확인 필요 |
| indicators source 분리 | 구현됨 | 1D 중심 검증, 1W/1M 확대 전 조건 남음 |
| feature cache 격리 | 구현됨 | local snapshot 기반 재생성 정책 유지 필요 |
| Supabase egress guard | 구현됨 | cron 재개 전 사람 확인 필요 |
| Supabase slimming | 계획 수립 | 삭제와 pruning은 아직 금지 |

## 2. CP별 변경점

| CP | 문제 | 추가된 계약 또는 변경 | 검증 상태 | 아직 운영 반영이 안 된 것 |
|---|---|---|---|---|
| CP86-D | EODHD 비용 부담, yfinance 전환 필요 | provider abstraction, yfinance daily fetch, adjusted OHLC sanity, EODHD fallback 유지 | dry-run 비교와 provider 테스트 추가 | 운영 전체 write 없음 |
| CP87-D | CP86 dry-run만으로 운영 전환 판단 부족 | 데이터, feature, model smoke 3단계 검증 체계 | yfinance 전환은 WARN | provenance, source hash, cache isolation 부족 |
| CP88-D | yfinance/EODHD cache와 source 혼합 위험 | `price_data` provenance, source hash 강화, cache manifest | provider별 hash와 manifest mismatch 테스트 | 제한 write 전 상태 |
| CP88-G | provider migration 경험과 기록 방식 부재 | parallel run, reconciliation, audit trail, cutover playbook | 문서화 완료 | 기술 구현은 CP88-D 이후 단계에서 진행 |
| CP89-D | provider/source 계약을 실제 제한 write로 검증 필요 | 5티커 yfinance 제한 write, 1D indicator 재계산, feature cache 검증 | 제한 write 성공, smoke는 split 이슈 발생 | old unique 때문에 EODHD row 덮임 artifact 가능 |
| CP90-D | yfinance smoke split이 빈 결과로 실패 | source-aware feature index와 split planning | split diagnostic 개선 | PatchTST seq252는 기간 부족 가능성 명시 |
| CP91-D | price_data와 indicators에서 source 혼합 P0 위험 | unique `(ticker,date,source)`, indicator unique `(ticker,timeframe,date,source)`, source-aware compute_indicators | source mixing guard 테스트 추가 | CP89 old unique artifact 복구는 미실행 |
| CP92-D | CP89 5티커 덮임 artifact 처리 필요 | 5티커 yfinance source-aware indicator 재계산, artifact 기록 | 조건부 PASS | EODHD 최근 구간 복구는 보류 |
| CP93-D | 50티커 recent write 확대 검증 필요 | 50티커 recent yfinance write, 1D indicators, feature cache, smoke | PASS_WITH_NOTES | 장기 history 전환 전 |
| CP94-D | 50티커 장기 history 검증 필요 | 2015년 이후 50티커 write, 1D indicators, line/band smoke | PASS_WITH_NOTES | 전체 universe 전환 전 |
| CP95-D | EODHD 해지 판단을 위한 100티커 gate 필요 | 100티커 장기 write, 1D indicators, feature, line/band smoke | PASS_WITH_NOTES | 전체 universe write 금지 유지 |
| CP96-D | 전체 write 전 P1 데이터 guard 필요 | indicator value checksum, source/provider 조회 계약, 1W/1M partial period 제외 | 관련 테스트 추가 | P2 항목은 다음 CP 후보 |
| CP-SUPA | Supabase Free egress 제한 위험 | local parquet snapshot first, 대량 read 차단, frontend 기본 1년, readiness 대량 조회 금지 | egress incident 문서화 | Render cron suspend 확인은 사람 확인 필요 |
| CP-SUPA2 | Supabase DB 2GB Free 운영 구조 필요 | slimming plan, 유지/로컬 분리, pruning 정책, upsert `returning=minimal` | checklist와 upsert 테스트 추가 | 실제 삭제와 pruning은 금지 |

## 3. 문제에서 계약으로 바뀐 항목

| 기존 문제 | 현재 계약 |
|---|---|
| EODHD와 yfinance가 같은 ticker/date를 덮어쓸 수 있음 | `price_data` unique를 `(ticker,date,source)`로 분리 |
| indicators가 source 없는 전체 price history로 계산될 수 있음 | `indicators` unique를 `(ticker,timeframe,date,source)`로 분리하고 source-aware fetch 적용 |
| legacy EODHD null source 해석이 불명확함 | `source is null`은 EODHD legacy로 해석 |
| provider가 달라도 같은 feature cache를 재사용할 수 있음 | provider, source, policy, checksum 포함 hash와 manifest 검증 |
| indicator 값만 바뀌면 cache가 stale이어도 재사용될 수 있음 | indicator numeric value checksum을 fingerprint에 포함 |
| yfinance 최근 2년 데이터로 seq252 smoke가 비는 이유가 불명확함 | split diagnostic에 provider, source, date range, sample count 출력 |
| 1W/1M 진행 중 candle이 완성 기간처럼 저장될 수 있음 | latest daily date 기준 partial period 제외 |
| Supabase가 학습 데이터 창고처럼 반복 조회됨 | local parquet snapshot required guard |
| 대량 upsert가 응답 payload로 egress를 키움 | chunked upsert 기본 `returning=minimal` |
| parquet export 자체가 egress를 만들 수 있음 | guarded table export confirm gate |

## 4. 아직 운영 반영 전인 항목

| 항목 | 현재 상태 | 다음 조건 |
|---|---|---|
| 전체 universe yfinance 1D write | 금지 상태 | CP96 guard 통과 후 별도 제한 계획 |
| yfinance 1W/1M 전체 재계산 | 금지 상태 | source-aware resample과 partial period 검증 확대 |
| EODHD row 복구 | 보류 | CP89 artifact 복구 필요성이 확인될 때만 dry-run 후 승인 |
| EODHD 해지 | 보류 | 전체 yfinance 전환, fallback 축소 판단, 비용 검토 완료 |
| Supabase pruning | 보류 | parquet backup, count 검증, rollback 절차 완료 |
| Render cron 재개 | 보류 | egress guard, source-aware provider, 사람 확인 완료 |
| live inference 연결 | 보류 | yfinance 전체 전환과 제품 run 교체 계획 필요 |

## 5. 현재 기준 의사결정

| 의사결정 | 현재 판단 |
|---|---|
| yfinance를 primary 후보로 유지할지 | 유지 |
| EODHD를 즉시 삭제할지 | 삭제 금지 |
| Supabase를 원천 데이터 창고로 계속 쓸지 | 축소 |
| 로컬 parquet를 학습 원천으로 둘지 | 예 |
| 100티커 검증만으로 EODHD 해지 가능한지 | 아직 불가 |
| 전체 universe write를 바로 진행할지 | 아직 불가 |

## 6. 검증 근거 요약

| 가드 | 근거 파일 |
|---|---|
| yfinance raw/adjusted 계약 | `backend/tests/test_market_data_providers.py` |
| source-aware unique key | `backend/db/schema.sql`, `backend/db/scripts/ensure_runtime_schema.py` |
| sync upsert source conflict | `backend/collector/jobs/sync_prices.py`, `backend/tests/test_market_data_providers.py` |
| compute_indicators source filter | `backend/collector/jobs/compute_indicators.py`, `backend/tests/test_collector_jobs.py` |
| feature hash와 manifest | `ai/preprocessing.py`, `ai/tests/test_preprocessing_cache_isolation.py` |
| partial 1W/1M 제외 | `backend/app/services/feature_svc.py`, `backend/tests/test_services.py` |
| local snapshot required | `backend/collector/repositories/base.py`, `backend/collector/repositories/local_snapshots.py` |
| upsert egress 절감 | `backend/db/bootstrap.py`, `backend/tests/test_db_bootstrap.py` |

## 7. 감사 메모

이번 인벤토리는 변경 이력을 “PASS 여부”만으로 정리하지 않는다. CP89의 old unique artifact처럼 기능은 통과했지만 migration artifact가 남은 항목은 별도 운영 리스크로 유지한다.

다음 큰 전환은 전체 universe 1D yfinance write가 아니라, 그 전에 “Supabase egress를 만들지 않는 로컬 snapshot 기반 검증 루프”를 먼저 고정하는 것이다.
