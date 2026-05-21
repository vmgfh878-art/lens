# CP145-D EODHD 500티커 indicator + context generation 보고서

## 1. 요약

판정은 **WARN**이다.

500티커 EODHD 1D/1W indicators와 context 생성은 통과했지만 fundamentals coverage가 낮아 해석 제약이 필요하다.

## 2. 입력과 생성 파일

| 구분 | 경로 |
|---|---|
| input price | `data/parquet/price_data_eodhd_500.parquet` |
| 1D indicators | `data/parquet/indicators_eodhd_1D_500.parquet` |
| 1W indicators | `data/parquet/indicators_eodhd_1W_500.parquet` |
| context dir | `data/parquet/context/eodhd_500/` |
| feature inventory | `docs/cp145_eodhd_500ticker_feature_inventory.csv` |

## 3. indicator 검증

| 항목 | 1D | 1W |
|---|---:|---:|
| rows | 1355956 | 258410 |
| tickers | 503 | 502 |
| date range | 2015-03-30 ~ 2026-05-05 | 2016-02-19 ~ 2026-05-01 |
| duplicate | 0 | 0 |
| feature non-finite | 0 | 0 |
| atr_ratio coverage | 1.0 | 1.0 |
| ratio sanity | True | True |

## 4. context coverage

| 계열 | 상태 | rows | date range | 비고 |
|---|---|---:|---|---|
| macro | PASS | 13036 | 1976-06-01 ~ 2026-05-04 | 기존 local FRED context 재사용 |
| market_breadth | PASS | 2652 | 2015-10-16 ~ 2026-05-05 | EODHD 500 price 기준 재계산 |
| fundamentals | PASS | 1135 | 2014-12-31 ~ 2026-04-03 | 기존 local SEC EDGAR context를 EODHD universe로 필터 |
| sector_returns | PASS | 34202 | 2015-01-05 ~ 2026-05-05 | 모델 36개 feature에는 직접 미포함 |

## 5. split gate

| timeframe | status | eligible tickers | estimated samples | train/val/test |
|---|---|---:|---:|---|
| 1D | PASS | 501 | 1227379 | {'train': 859248, 'val': 184101, 'test': 184030} |
| 1W | PASS | 497 | 204949 | {'train': 143629, 'val': 31056, 'test': 30264} |

## 6. feature/cache 계약

`MODEL_N_FEATURES=36`이며 `atr_ratio`는 모델 feature에 포함하지 않았다. EODHD 산출물은 `_eodhd_*_500` 파일명과 manifest로 yfinance 산출물과 분리되어 있다. context 변경은 `context_hash=1aa6452d82369cc6`와 indicator checksum에 반영된다.

## 7. 제약과 다음 단계

fundamentals는 기존 SEC EDGAR local context를 재사용했기 때문에 coverage가 일부 ticker에 제한된다. 그러나 macro/breadth는 1D/1W full_features에서 실제 값으로 채워졌고, 1D/1W split gate가 통과했으므로 다음 CP에서 500티커 1D/1W 모델 학습 준비로 넘어갈 수 있다.

## 8. 금지 작업 미발생 확인

Supabase raw write, DB write, yfinance append, 모델 학습, inference 저장, 프론트 수정, W&B, EODHD 추가 대량 호출은 수행하지 않았다.
