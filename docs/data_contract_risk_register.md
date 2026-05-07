# 데이터 계약 리스크 레지스터

작성일: 2026-05-04  
범위: yfinance 전환, source-aware schema, local parquet, Supabase slimming 관련 남은 리스크와 gate

## 1. 리스크 등급 기준

| 등급 | 의미 |
|---|---|
| P0 | 데이터 오염, source 혼합, 제품 결과 왜곡, 복구 어려운 비용 사고 |
| P1 | 전체 write, 모델 실험, 데모 전에 막아야 하는 리스크 |
| P2 | 우회 가능하지만 유지보수와 운영 혼란을 키우는 리스크 |
| P3 | 문서와 절차로 관리 가능한 리스크 |

## 2. 리스크 목록

| ID | 리스크 | 등급 | 트리거 | 현재 가드 | 상태 | 다음 조치 |
|---|---|---|---|---|---|---|
| R-01 | yfinance와 EODHD price row가 같은 ticker/date에서 덮임 | P0 | old unique `(ticker,date)` 또는 source 없는 upsert | unique `(ticker,date,source)`, upsert conflict source 포함 | 구조 수리됨, CP89 artifact 남음 | 50/100 이후 전체 write 전 artifact 복구 필요성 판단 |
| R-02 | indicators가 yfinance와 EODHD price history를 섞어 계산 | P0 | source filter 없는 indicator recompute | compute_indicators source filter, unique `(ticker,timeframe,date,source)` | 1D 중심 검증됨 | 1W/1M 확대 전 source boundary 테스트 재확인 |
| R-03 | stale feature cache 재사용 | P0 | indicator 값 변경, provider 변경, 같은 count/date | provider/source/policy/hash manifest, indicator value checksum | 구조 수리됨 | CP95 cache 재생성 또는 mismatch 감지 확인 유지 |
| R-04 | Supabase 대량 read로 egress 재발 | P0 | feature build, readiness, export가 price/indicator 전체 조회 | local snapshot required guard, guarded bulk read block | 수리됨 | Render cron 재개 전 환경값 확인 |
| R-05 | 백업 없는 Supabase pruning | P0 | failed run, legacy row 삭제를 바로 실행 | slimming plan과 export confirm gate | 계획만 있음 | parquet export, count diff, rollback 문서 후 승인 |
| R-06 | 전체 universe yfinance write를 너무 빨리 실행 | P1 | 100티커 PASS만 보고 전체 write | 금지 정책과 단계 gate | 금지 유지 | 전체 write CP에서 batch, rollback, egress plan 작성 |
| R-07 | 1W/1M partial period가 완성 candle처럼 저장 | P1 | resample 후 진행 중 주/월 저장 | partial period 제외 테스트 | 수리됨 | yfinance 1W/1M source-aware 제한 재계산으로 검증 |
| R-08 | EODHD 해지 후 fallback과 검증 기준 상실 | P1 | 비용 때문에 즉시 key 삭제 | EODHD fallback/legacy 유지 정책 | 보류 | 전체 universe, 1W/1M, live inference 후 해지 판단 |
| R-09 | Render cron 재개가 egress와 source 혼합을 다시 유발 | P1 | daily_market_sync 재개 | cron suspend 확인 절차, source-aware path | 사람 확인 필요 | 재개 전 dry-run 명령과 provider 설정 점검 |
| R-10 | local parquet snapshot이 누락되거나 오래됨 | P1 | local required 모드에서 snapshot 부재, 잘못된 파일 사용 | snapshot path 후보와 required guard | 구조 있음 | snapshot manifest와 생성일 감사 강화 |
| R-11 | backtest가 product provider와 다른 source를 읽음 | P1 | source 미지정 price read | CP96 source/provider 조회 계약 | 수리됨 | backtest fixture 유지, 제품 provider env 문서화 |
| R-12 | 제품 API가 전체 2015 history를 기본 조회 | P1 | 프론트 또는 readiness가 전체 history 요청 | 기본 1년 조회, lazy load 원칙 | 수리됨 | 화면별 limit 계약 문서화 |
| R-13 | 대량 upsert 응답 payload가 egress 증가 | P1 | Supabase upsert 기본 representation 응답 | `returning=minimal` 기본값 | 수리됨 | 신규 upsert helper도 같은 계약 강제 |
| R-14 | guarded table parquet export가 egress를 만듦 | P1 | 확인 없이 price/indicator export | `--confirm-egress-export` gate | 수리됨 | export 전 용량 추정과 시간대 기록 |
| R-15 | stock_info 누락으로 yfinance universe 구성 실패 | P2 | SPY, QQQ 같은 ETF 또는 신규 ticker 누락 | 공식 stock_info 보강 경로 | 일부 CP에서 보강 | 전체 universe 전 누락 0 확인 |
| R-16 | CP89 5티커 EODHD 최근 row 덮임 artifact | P2 | old unique 시점 제한 write | artifact 문서화, 복구 보류 | 남아 있음 | 필요 시 EODHD dry-run 재수집 절차만 실행 |
| R-17 | fallback 사용 ticker가 yfinance 품질 통계에 섞임 | P2 | provider fallback이 조용히 성공 처리 | fallback_used 기록 정책 | 구조 있음 | metrics에 fallback ticker 분리 |
| R-18 | predictions/evaluations legacy row가 제품 run 선택을 혼란 | P2 | composite, failed, legacy run 다수 누적 | CP-SUPA2 pruning 후보 | 계획만 있음 | 백업 후 pruning CP 필요 |

## 3. 전환별 gate

### 3.1 전체 universe yfinance 1D write 전 조건

| 조건 | 현재 상태 | 판정 |
|---|---|---|
| source-aware schema 적용 | 구현됨 | 필요 시 운영 DB 확인 |
| price_data unique `(ticker,date,source)` | 구현됨 | 필수 |
| indicators unique `(ticker,timeframe,date,source)` | 구현됨 | 필수 |
| provider/source read 계약 | 수리됨 | 필수 |
| local snapshot required guard | 수리됨 | 필수 |
| indicator value checksum | 수리됨 | 필수 |
| 100티커 장기 yfinance PASS | PASS_WITH_NOTES | 전체 전환 전 충분조건 아님 |
| batch write plan | 별도 필요 | 미완료 |
| rollback 계획 | 별도 필요 | 미완료 |
| egress 영향 예측 | 별도 필요 | 미완료 |

판정: 아직 전체 universe write 금지.

### 3.2 1W/1M 전환 전 조건

| 조건 | 현재 상태 | 판정 |
|---|---|---|
| source-aware price fetch | 구현됨 | 필수 |
| partial week/month 제외 | 구현됨 | 필수 |
| yfinance 1W/1M 제한 indicator recompute | 별도 필요 | 미완료 |
| 1W/1M feature cache hash 분리 | 구조 있음 | 재검증 필요 |
| 1W/1M 제품 표시 계약 | 별도 필요 | 미완료 |

판정: 1W/1M 전체 재계산 금지.

### 3.3 Supabase pruning 전 조건

| 조건 | 현재 상태 | 판정 |
|---|---|---|
| 테이블별 용량 추정 SQL | 계획 있음 | 실행은 사람 판단 |
| parquet backup | 필요 | 미완료 |
| 백업 row count 검증 | 필요 | 미완료 |
| 삭제 후보 분류 | 계획 있음 | 보강 필요 |
| rollback 절차 | 필요 | 미완료 |
| 제품 run 보호 목록 | 필요 | 미완료 |

판정: DB row 삭제 금지.

### 3.4 EODHD 해지 전 조건

| 조건 | 현재 상태 | 판정 |
|---|---|---|
| yfinance 전체 1D write | 미완료 | 필수 |
| 1D indicators source-aware 재계산 | 100티커까지 검증 | 전체 필요 |
| feature cache 전체 검증 | 100티커까지 검증 | 전체 필요 |
| line/band smoke | 100티커까지 검증 | 추가 필요 |
| 1W/1M 정책 결정 | 미완료 | 필수 |
| fallback 축소 계획 | 미완료 | 필수 |
| EODHD baseline archive | 미완료 | 권장 |

판정: EODHD 즉시 해지 금지.

### 3.5 Render cron 재개 전 조건

| 조건 | 현재 상태 | 판정 |
|---|---|---|
| cron suspend 상태 사람 확인 | 필요 | 미완료 |
| provider 설정 명시 | 필요 | 미완료 |
| indicator lookback과 1W/1M partial guard 확인 | 필요 | 미완료 |
| Supabase egress budget 확인 | 필요 | 미완료 |
| dry-run 또는 제한 ticker rehearsal | 필요 | 미완료 |

판정: 확인 전 전체 sync 금지.

## 4. 테스트와 가드 현황

| 가드 | 목적 | 대표 테스트 또는 근거 | 상태 |
|---|---|---|---|
| source mixing guard | yfinance/EODHD price와 indicators 혼합 방지 | `test_compute_indicators_filters_yfinance_price_source_and_stores_source`, source-aware unique key 테스트 | 구현됨 |
| legacy null EODHD guard | 기존 null source를 EODHD로 해석 | `test_eodhd_legacy_null_start_date_avoids_overlap_until_backfill` | 구현됨 |
| partial period guard | 진행 중 1W/1M candle 제외 | `test_aggregate_prices_drops_partial_week_and_month` | 구현됨 |
| feature checksum guard | indicator 값 변경 시 cache 무효화 | `test_indicator_value_change_changes_source_hash_and_rejects_manifest` | 구현됨 |
| cache manifest provider guard | provider mismatch cache 재사용 금지 | `test_manifest_provider_mismatch_rejects_cache` | 구현됨 |
| source-aware split guard | yfinance availability 기준 split planning | `test_source_aware_feature_index_uses_provider_price_dates` | 구현됨 |
| local snapshot required guard | Supabase 대량 read 방지 | `test_local_snapshot_feature_index_does_not_call_supabase`, market repo local snapshot 테스트 | 구현됨 |
| upsert returning minimal | Supabase 응답 egress 절감 | `test_chunked_upsert_defaults_to_returning_minimal` 계열 | 구현됨 |
| export confirm gate | guarded table export 실수 방지 | `backend/db/scripts/export_parquet.py` 확인 gate | 구현됨 |

## 5. 지금 하면 안 되는 작업

| 작업 | 금지 이유 |
|---|---|
| 전체 yfinance write | 전체 batch, rollback, egress 계획 미완료 |
| indicators full recompute | source-aware 1W/1M 확대 검증 전 |
| Supabase pruning | parquet backup과 검증 전 |
| EODHD 삭제 | fallback과 검증 기준 상실 |
| Render cron 무조건 재개 | egress 재발 가능 |
| full model training | local snapshot과 cache 계약 검증 전에는 비용과 원천 혼합 위험 |
| live inference 저장 | product run 교체와 provider provenance 정책 필요 |

## 6. 다음 CP 추천

1. 전체 universe write가 아니라, 전체 universe 대상 local parquet yfinance snapshot 생성과 검증을 먼저 수행한다.
2. 1W/1M은 20티커 제한 source-aware recompute로 partial period와 indicator sanity를 먼저 확인한다.
3. Supabase pruning은 backup, row count diff, rollback 문서를 만든 뒤 별도 승인으로 진행한다.
4. Render cron은 사람이 suspend 상태를 확인한 뒤, 제한 ticker dry-run으로 재개 절차를 검증한다.

최종 판정: 현재 데이터 아키텍처는 yfinance primary 전환을 향해 가는 구조로 정리됐지만, 전체 write, 1W/1M 확대, Supabase pruning, EODHD 해지는 아직 gate 밖이다.
