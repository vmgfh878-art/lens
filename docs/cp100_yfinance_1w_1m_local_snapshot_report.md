# CP100-D yfinance local 1W/1M snapshot bootstrap

생성일: 2026-05-04

## 1. Executive Summary

판정: **WARN**

1D yfinance local parquet를 원천으로 1W/1M 가격 snapshot과 indicator snapshot을 생성했다. Supabase `price_data`/`indicators` 대량 read/write, DB indicator recompute, 모델 학습, inference 저장, EODHD fallback은 실행하지 않았다.

생성된 파일:

| 파일 | row 수 | ticker 수 | 날짜 범위 | 상태 |
|---|---:|---:|---|---|
| `data/parquet/price_data_yfinance_1W.parquet` | 59,200 | 100 | 2015-01-02 ~ 2026-05-01 | PASS |
| `data/parquet/indicators_yfinance_1W.parquet` | 53,300 | 100 | 2016-02-19 ~ 2026-05-01 | PASS |
| `data/parquet/price_data_yfinance_1M.parquet` | 13,600 | 100 | 2015-01-31 ~ 2026-04-30 | PASS |
| `data/parquet/indicators_yfinance_1M.parquet` | 7,700 | 100 | 2019-12-31 ~ 2026-04-30 | PASS |

WARN 사유는 1M 데이터 품질 문제가 아니라 split 정책이다. 1M은 `seq_len=24`, `horizon=3`, `min_fold_samples=50` strict 기준에서 Gate B로 탈락했다. 같은 데이터가 `min_fold_samples=5` 실험용 gate에서는 DB fallback 없이 통과했다. 2015년 시작 월봉은 60개월 MA 이후 usable row가 티커당 약 77개라, strict fold 50개를 만족하기 어렵다.

## 2. 변경 내용

이번 CP에서 로컬 snapshot 기반 1W/1M 실험 준비를 위해 아래 코드를 최소 보강했다.

| 파일 | 변경 |
|---|---|
| `scripts/cp100_yfinance_1w_1m_local_snapshot_bootstrap.py` | 1D yfinance parquet에서 1W/1M price/indicator parquet 생성 및 검증 |
| `ai/preprocessing.py` | `1M`을 AI timeframe으로 인식, local price snapshot을 timeframe별 parquet에서 우선 조회 |
| `ai/splits.py` | `MAX_HORIZON_BY_TIMEFRAME["1M"] = 3` 추가 |
| `ai/ticker_registry.py` | 1M ticker registry path 추가 |
| `ai/tests/test_preprocessing.py` | 1M default horizon 테스트 갱신 |
| `ai/tests/test_splits.py` | 1M split gate 테스트 추가 |
| `ai/tests/test_ticker_registry.py` | 1M ticker registry 테스트 추가 |
| `ai/tests/test_preprocessing_cache_isolation.py` | 1M local snapshot feature index가 timeframe price parquet를 쓰는 테스트 추가 |

핵심 수정 이유: 1W/1M indicator 날짜는 리샘플 period end다. feature index가 daily price snapshot과 날짜를 직접 조인하면 월말이 비거래일인 1M 샘플이 사라질 수 있다. 따라서 local mode에서는 `price_data_yfinance_1W.parquet`, `price_data_yfinance_1M.parquet`를 우선 읽도록 했다.

## 3. Source/Provider 계약

모든 신규 price snapshot에는 아래 계약을 기록했다.

| 항목 | 값 |
|---|---|
| `source` | `yfinance` |
| `provider` | `yfinance` |
| `provider_adjustment_policy` | `yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc` |
| `updated_at` | 생성 시각 |

중복 검증:

| timeframe | duplicate key | 결과 |
|---|---|---:|
| 1W price | `ticker,date,source` | 0 |
| 1M price | `ticker,date,source` | 0 |
| 1W indicators | `ticker,timeframe,date,source` | 0 |
| 1M indicators | `ticker,timeframe,date,source` | 0 |

## 4. Partial Period 제거

입력 1D snapshot의 최신 거래일은 `2026-05-01`이다.

| timeframe | latest complete period end | snapshot max date | 결과 |
|---|---|---|---|
| 1W | 2026-05-01 | 2026-05-01 | PASS |
| 1M | 2026-04-30 | 2026-04-30 | PASS |

2026년 5월 진행 중 월봉은 1M snapshot에 포함하지 않았다.

## 5. 가격 계약 검증

| timeframe | non-finite | high/low 위반 | adjusted OHLC 경계 위반 | partial rows | 결과 |
|---|---:|---:|---:|---:|---|
| 1W | 0 | 0 | 0 | 0 | PASS |
| 1M | 0 | 0 | 0 | 0 | PASS |

1W 가격 snapshot 검증에는 `1e-8` tolerance를 적용했다. 리샘플된 adjusted 값에서 부동소수점 단위의 미세한 경계 차이가 있었고, 실제 가격 관계 위반은 아니었다.

## 6. Indicator 검증

| timeframe | row 수 | ATR coverage | feature NaN/Inf | ratio p99 abs | 결과 |
|---|---:|---:|---:|---|---|
| 1W | 53,300 | 100.0% | 0 | open 0.0465 / high 0.1482 / low 0.1373 | PASS |
| 1M | 7,700 | 100.0% | 0 | open 0.0481 / high 0.3180 / low 0.2741 | PASS |

1M `high_ratio` max는 1.1684로 일부 월봉 급등 구간이 남아 있지만 p99는 0.3180이고 finite contract는 통과했다. 모델 피처 승격 전에는 1M ratio clipping 정책을 별도 판단하는 것이 좋다.

## 7. 모델 Feature 계약

| 항목 | 결과 |
|---|---|
| `MODEL_N_FEATURES` | 36 |
| `FEATURE_CONTRACT_VERSION` | `v3_adjusted_ohlc` |
| `atr_ratio in MODEL_FEATURE_COLUMNS` | false |

`atr_ratio`는 계속 차트/보조지표용 indicator로만 유지된다.

## 8. Source Data Hash

5개 검증 ticker 기준 source hash가 timeframe별로 분리됐다.

| timeframe | source_data_hash |
|---|---|
| 1D | `3e4ee198` |
| 1W | `cc395ce9` |
| 1M | `da6ff17b` |

## 9. Local Split Gate

DB fallback 방지를 위해 `ai.preprocessing.fetch_frame`과 `_postgres_engine`을 테스트 실행 중 차단했다.

| gate | 설정 | 결과 |
|---|---|---|
| 1W strict | `seq_len=104`, `horizon=4`, `min_fold_samples=50` | PASS: train 1,375 / val 295 / test 300 |
| 1M strict | `seq_len=24`, `horizon=3`, `min_fold_samples=50` | FAIL: Gate B |
| 1M experimental | `seq_len=24`, `horizon=3`, `min_fold_samples=5` | PASS: train 155 / val 30 / test 40 |

1M strict 실패는 데이터 혼합이나 DB fallback이 아니라 fold 크기 정책 문제다. 2015년 이후 월봉에서 60개월 MA를 만든 뒤 남는 행이 적어 기본 fold 50개를 만족하지 못한다.

## 10. 금지 작업 확인

| 금지 항목 | 실행 여부 |
|---|---|
| Supabase `price_data`/`indicators` 대량 read | 미실행 |
| Supabase 전체 write | 미실행 |
| indicators DB recompute | 미실행 |
| 모델 학습 | 미실행 |
| inference 저장 | 미실행 |
| EODHD fallback | 미실행 |
| 프론트 수정 | 미실행 |

## 11. 결론 및 다음 단계

1W는 EODHD 없이 local yfinance snapshot 기반 모델 실험 준비가 완료됐다.

1M은 snapshot/feature/hash/local split 경로는 열렸지만, 2015년 이후 데이터와 현행 `min_fold_samples=50` 기준으로는 strict training gate가 부족하다. 다음 CP에서는 1M 실험 정책을 정해야 한다.

추천:

1. 1W 모델 smoke는 local snapshot 기반으로 진행 가능.
2. 1M은 `min_fold_samples` 완화, `seq_len` 축소, 60개월 feature window 축소, 더 긴 history 확보 중 하나를 먼저 결정.
3. 1M ratio clipping/winsorization 정책을 별도 감사 후 모델 실험에 붙일 것.

## 12. 실행 명령

```powershell
python scripts\cp100_yfinance_1w_1m_local_snapshot_bootstrap.py
```

검증 로그와 상세 수치는 아래 파일에 저장했다.

```text
docs/cp100_yfinance_1w_1m_local_snapshot_metrics.json
logs/cp100_yfinance_1w_1m_local_snapshot/cp100_yfinance_1w_1m_local_snapshot_metrics.json
```
