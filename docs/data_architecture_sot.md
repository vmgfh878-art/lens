# Lens 데이터 아키텍처 SoT

작성일: 2026-05-04  
상태: 전환 중인 기준 계약  
범위: EODHD에서 yfinance 로컬 primary 후보로 전환하는 과정의 데이터 원천, 저장소, 캐시, 검증 계약

## 1. 핵심 결론

Lens의 데이터 구조는 이제 단일 원격 DB 중심 구조가 아니라, 원천 수집과 학습 검증은 로컬 파일 우선, 제품 표시는 얇은 Supabase DB 우선으로 나뉜다.

현재 기준 역할은 다음과 같다.

| 영역 | 기준 역할 | 상태 |
|---|---|---|
| yfinance | 개인 로컬 운영용 primary 가격 provider 후보 | 100티커 장기 검증 통과, 전체 universe 전환 전 |
| EODHD | fallback 및 검증용 legacy provider | 즉시 삭제 금지 |
| Supabase | 제품 결과와 얇은 조회용 DB | egress 절감 모드, 대량 원천 저장소 역할 축소 |
| 로컬 parquet | 원본 가격, 지표, 학습, 검증용 데이터 창고 | Supabase 대량 read 대체 경로 |
| feature cache | 모델 학습 입력 캐시 | provider, source, checksum 격리 필수 |
| checkpoint 및 artifacts | 실험 결과 파일 | DB 원천 데이터로 보지 않음 |

이 문서는 “어떤 데이터가 어디에 있어야 하는가”와 “어떤 조건을 통과해야 다음 전환 단계로 갈 수 있는가”를 고정한다.

## 2. 데이터 원천 계약

### 2.1 yfinance

yfinance는 로컬 개인 운영 기준 primary 가격 데이터 후보이다. 공식 SLA가 있는 상용 provider가 아니므로, 제품 운영에서는 fallback과 검증 로그를 유지해야 한다.

계약:

| 항목 | 계약 |
|---|---|
| 용도 | 로컬 primary 가격 데이터 후보 |
| 사용 범위 | 1D 가격 수집, source-aware indicator 재계산, feature cache 생성, smoke 검증 |
| 저장 source | `yfinance` |
| 저장 provider | `yfinance` |
| 조정 정책 | `yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc` |
| 가격 수집 정책 | raw OHLC와 `adjusted_close`를 같이 보존 |
| adjusted OHLC | `adjusted_close / close` factor로 재구성 |
| 실패 처리 | EODHD fallback은 유지하되 fallback 사용 ticker를 별도 기록 |
| 운영 상태 | 전체 universe write 전 |

### 2.2 EODHD

EODHD는 기존 primary였고, 현재는 fallback 및 검증용 legacy provider이다. 해지 전까지는 yfinance와 차이를 비교하는 golden reference 역할을 유지한다.

계약:

| 항목 | 계약 |
|---|---|
| 용도 | fallback, 검증용 baseline, legacy data |
| 저장 source | `eodhd` 또는 legacy `null` |
| 저장 provider | `eodhd` 또는 legacy `null` |
| 조정 정책 | `eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc` 또는 legacy 정책 |
| legacy 해석 | `source is null`은 EODHD legacy로 본다 |
| 삭제 정책 | 대량 삭제 금지 |
| 해지 조건 | yfinance 전체 전환, 검증, fallback 축소 판단 이후 |

### 2.3 Supabase

Supabase는 더 이상 전체 원천 가격과 전체 지표를 반복 조회하는 학습 데이터 창고가 아니다. Free egress와 DB 용량 제한을 고려해 제품 표시와 최신 결과 중심의 얇은 DB로 유지한다.

유지 후보:

| 테이블 | Supabase 유지 목적 |
|---|---|
| `stock_info` | 티커 검색, 이름, 섹터, 산업 |
| `price_data` | 제품 표시용 최신 가격 일부 |
| `indicators` | 제품 표시용 최신 지표 일부 |
| `model_runs` | 제품 run 요약과 provenance |
| `predictions` | 최신 제품 예측 |
| `prediction_evaluations` | 제품 run 평가 요약 |
| `backtest_results` | 제품 run 백테스트 요약 |

로컬 우선으로 내릴 후보:

| 데이터 | 로컬 우선 이유 |
|---|---|
| 전체 `price_data` history | 대량 read와 egress 원인 |
| 전체 `indicators` history | 대량 read와 재계산 비용 |
| 학습용 feature bundle | 반복 생성 비용과 원천 hash 필요 |
| 실험성 predictions/evaluations history | 제품 DB를 비대하게 만듦 |
| 실패 run, legacy composite run | 제품 표시와 직접 관계가 약함 |

### 2.4 로컬 parquet

로컬 parquet는 원본, 검증, 학습 데이터의 기준 창고이다. Supabase 대량 read를 막기 위해 `price_data`와 `indicators`는 로컬 snapshot 우선으로 읽는다.

계약:

| 항목 | 계약 |
|---|---|
| 기본 위치 | `LENS_LOCAL_SNAPSHOT_DIR` |
| 강제 모드 | `LENS_REQUIRE_LOCAL_SNAPSHOTS=1` |
| 데이터 backend | `LENS_DATA_BACKEND=local`, `parquet`, `snapshot` |
| 대상 테이블 | `price_data`, `indicators`, 필요 시 `stock_info` |
| 파일명 후보 | provider, timeframe, table 단위 파일명 허용 |
| 누락 시 정책 | local required 모드에서는 Supabase 대량 read 금지 |
| export 정책 | `price_data`, `indicators` export는 확인 gate 필요 |

## 3. 테이블과 파일 역할

| 대상 | 역할 | 원천성 | 주요 계약 |
|---|---|---|---|
| `price_data` | source-aware 원천 가격 저장 | 높음 | unique `(ticker,date,source)`, raw OHLC와 `adjusted_close` 보존 |
| `indicators` | source-aware 지표 저장 | 중간 | unique `(ticker,timeframe,date,source)`, provider별 가격만 사용 |
| `stock_info` | 티커 메타데이터 | 중간 | 제품 검색과 universe 구성 보조 |
| `model_runs` | 학습 run provenance와 상태 | 낮음 | `feature_version`, provider, checkpoint, status 보존 |
| `predictions` | 제품 예측 결과 | 낮음 | `run_id` 포함 unique 계약 |
| `prediction_evaluations` | 예측 평가 결과 | 낮음 | coverage, breach rate 등 평가 계약 |
| `backtest_results` | 백테스트 요약 | 낮음 | adjusted anchor와 포트폴리오 정의 필요 |
| 로컬 parquet snapshot | 원천 및 학습 read 기준 | 높음 | provider/source/timeframe 격리 |
| feature cache | 모델 입력 tensor와 index | 파생 | `source_data_hash`와 manifest 검증 필수 |
| checkpoint | 모델 가중치 | 파생 | ticker registry mapping과 feature contract 보존 |
| artifacts/logs | 실험 로그와 보고서 | 파생 | 제품 DB 원천으로 사용하지 않음 |

## 4. 가격 조정 계약

모델 입력 가격 피처와 indicator 가격 ratio는 adjusted OHLC 기준이다.

기준 계산:

```text
adjusted_factor = adjusted_close / close
adjusted_open = open * adjusted_factor
adjusted_high = high * adjusted_factor
adjusted_low = low * adjusted_factor
adj_close = adjusted_close
```

필수 sanity:

| 항목 | 기준 |
|---|---|
| 날짜 | null 없음 |
| raw OHLC | `open`, `high`, `low`, `close`, `adjusted_close` null 없음 |
| adjusted factor | null, Inf, 비양수 없음 |
| raw 가격 정합성 | `high >= max(open, close)`, `low <= min(open, close)` |
| adjusted 가격 정합성 | `adjusted_high >= max(adjusted_open, adjusted_close)`, `adjusted_low <= min(adjusted_open, adjusted_close)` |
| ratio 폭주 | `open_ratio`, `high_ratio`, `low_ratio` p99와 max sanity 통과 |
| 중복 | `(ticker,date,source)` 중복 0 |

raw OHLC와 adjusted previous close를 섞는 구현은 금지한다.

## 5. source/provider 계약

| 필드 | 위치 | 의미 | 필수 정책 |
|---|---|---|---|
| `source` | `price_data`, `indicators` | 저장된 row의 데이터 원천 | yfinance row는 `yfinance`, EODHD legacy null은 `eodhd`로 해석 |
| `provider` | `price_data`, `indicators` | 수집 adapter | 명시값 우선, null은 legacy EODHD |
| `provider_adjustment_policy` | `price_data` | adjusted OHLC 재구성 정책 | provider별 정책 문자열 기록 |
| `updated_at` | `price_data`, `indicators` | upsert 시각 | 신규 write에서는 누락 금지 |
| `feature_version` | feature cache, model run | 모델 feature 계약 | 현재 `v3_adjusted_ohlc` |
| `source_data_hash` | feature cache, model run | 입력 데이터 fingerprint | provider, policy, checksum 포함 |
| cache manifest | feature cache 옆 JSON | 캐시 재사용 계약 | provider/hash/version/columns 불일치 시 재생성 |
| ticker universe fingerprint | feature fingerprint | universe 구성 식별 | 같은 기간이어도 universe가 다르면 hash 변경 |

읽기 정책:

| 모드 | price read | indicator read |
|---|---|---|
| `yfinance` | `source='yfinance'`만 사용 | `source='yfinance'`만 사용 |
| `eodhd` | `source='eodhd'` 또는 legacy `null` 사용 | `source='eodhd'` 또는 legacy `null` 사용 |
| source 미지정 | 경고 대상 | 경고 대상 |

한 run, 한 feature cache, 한 indicator recompute 안에서 yfinance와 EODHD row를 섞으면 안 된다.

## 6. feature cache와 fingerprint 계약

feature cache는 provider가 다르면 같은 ticker/date라도 다른 파일명과 다른 hash를 가져야 한다.

`source_data_hash`에 포함해야 하는 값:

| 항목 | 필요 이유 |
|---|---|
| `market_data_provider` | yfinance/EODHD cache 혼합 방지 |
| `provider_adjustment_policy` | adjusted OHLC 정책 차이 반영 |
| `feature_contract_version` | feature list와 계산 계약 식별 |
| `timeframe` | 1D/1W/1M 혼합 방지 |
| ticker universe fingerprint | universe 변경 반영 |
| date range | 기간 변경 반영 |
| price checksum | price row 변경 반영 |
| indicator value checksum | 같은 count/date라도 값 변경 반영 |

cache manifest 필수 필드:

| 필드 | 계약 |
|---|---|
| `schema_version` | manifest schema 식별 |
| `cache_kind` | feature 또는 index cache 구분 |
| `provider` | market data provider |
| `source` | source filter |
| `provider_adjustment_policy` | 가격 조정 정책 |
| `source_data_hash` | 입력 fingerprint |
| `feature_version` | feature contract |
| `feature_columns` | 모델 입력 column 목록 |
| `ticker_count` | universe 크기 |
| `date_min`, `date_max` | cache 기간 |
| `created_at` | 생성 시각 |

manifest가 없거나 provider, source, policy, hash, feature column이 맞지 않으면 캐시를 재사용하지 않는다.

## 7. 1W/1M partial period 계약

1D는 최신 거래일을 그대로 사용할 수 있다. 1W와 1M은 현재 진행 중인 주나 월을 완성 candle처럼 저장하거나 표시하면 안 된다.

정책:

| timeframe | 저장 가능 조건 |
|---|---|
| 1D | 최신 거래일까지 허용 |
| 1W | `period_end <= latest_complete_week_end` |
| 1M | `period_end <= latest_complete_month_end` |

단순 resample 후 `dropna`만 하는 방식은 부족하다. latest daily date 기준으로 아직 종료되지 않은 bucket은 제외한다.

## 8. Supabase egress 절감 계약

Supabase 대량 read를 막는 기본 원칙:

| 상황 | 정책 |
|---|---|
| feature cache 생성 | local parquet snapshot 우선 |
| local snapshot required | Supabase `price_data`, `indicators` 대량 read 금지 |
| 제품 가격 조회 | 기본 1년 조회 |
| 전체 2015년 history | lazy load 또는 로컬 사용 |
| readiness | 대량 history 조회 금지 |
| parquet export | 명시 확인 gate 필요 |
| upsert | 기본 `returning=minimal` |

## 9. 운영 단계별 기준

| 단계 | 허용 | 금지 또는 보류 |
|---|---|---|
| 현재 | 100티커까지 검증된 yfinance 장기 데이터 판단 사용 | 전체 universe write |
| 전체 1D write 전 | source mixing, checksum, local snapshot, egress guard 통과 필요 | EODHD 삭제 |
| 1W/1M 전환 전 | source-aware resample과 partial period guard 검증 필요 | 1W/1M 전체 재계산 |
| Supabase pruning 전 | parquet export, row count 검증, rollback 절차 필요 | 백업 없는 삭제 |
| EODHD 해지 전 | yfinance 전체 write, indicators, feature, smoke, fallback 판단 완료 | provider 삭제 |
| Render cron 재개 전 | egress 제한, source-aware cron, local daily command 확인 | 무제한 sync |

## 10. 근거 파일

주요 근거는 다음 파일에 있다.

| 영역 | 파일 |
|---|---|
| schema | `backend/db/schema.sql`, `backend/db/scripts/ensure_runtime_schema.py` |
| 가격 sync | `backend/collector/jobs/sync_prices.py` |
| indicator 계산 | `backend/collector/jobs/compute_indicators.py` |
| feature cache | `ai/preprocessing.py` |
| local snapshot | `backend/collector/repositories/local_snapshots.py`, `backend/collector/repositories/base.py` |
| API read 경로 | `backend/app/repositories/market_repo.py` |
| egress guard | `backend/db/scripts/export_parquet.py`, `backend/db/bootstrap.py` |
| 테스트 | `backend/tests/test_market_data_providers.py`, `backend/tests/test_collector_jobs.py`, `backend/tests/test_services.py`, `ai/tests/test_preprocessing_cache_isolation.py`, `backend/tests/test_db_bootstrap.py` |

이 SoT는 코드 변경 없이 현재 구조와 변경 의도를 문서로 고정한다.
