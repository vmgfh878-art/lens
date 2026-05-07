# CP134-DG 로컬 일일 시장/컨텍스트 업데이트 파이프라인 보고서

## 1. 요약

판정은 **FAIL**이다.

local parquet 원천 구조에 맞춘 일일 업데이트 순서를 고정했고, 5티커 yfinance 완료 거래일 dry-run을 수행했다. 이번 실행에서는 새 완료 거래일 row가 있으면 append 후보로 기록하고, 없으면 `PASS_WITH_NO_NEW_DAY`로 처리한다. 단, yfinance fetch가 빈 응답이면 신규 거래일 없음으로 보지 않고 중단 조건으로 분리한다. Supabase 대량 read/write, EODHD fallback, 모델 학습, inference 저장은 수행하지 않았다.

## 2. 일일 파이프라인 계약

1. 가격 업데이트: yfinance만 사용하고 `row.date < current_date` 완료 거래일만 append한다.
2. 컨텍스트 업데이트: breadth/sector는 local price universe에서 재계산하고, macro는 FRED 최신 observation만 append/update하며, fundamentals는 SEC `filing_date` 기준 신규 filing만 반영한다.
3. 지표 갱신: 1D는 incremental refresh, 1W는 완료 주만 refresh, 1M은 daily job에서 skip한다.
4. 캐시/해시: indicator/context 값이 바뀌면 source_data_hash와 manifest mismatch로 기존 feature cache 재사용을 막는다.
5. 제품 추론: 별도 단계에서 1D line/band latest inference만 실행하고 `save_product_latest_predictions()`만 사용한다.
6. scanner: 500티커 운영 전까지 skip한다.
7. readiness: local price/indicator/context/product latest date와 EODHD fallback 0, Supabase bulk read/write 0을 확인한다.

## 3. 이번 리허설 결과

| 항목 | 값 |
|---|---|
| price dry-run status | FAIL |
| snapshot latest date | 2026-05-04 |
| current date gate | 2026-05-06 |
| completed append candidate rows | 0 |
| completed append candidate dates | [] |
| fallback used count | 0 |
| contract failures | [] |
| empty fetch count | 5 |
| empty fetch tickers | ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'NFLX'] |

## 4. 컨텍스트 업데이트 설계 검증

| context | update policy |
|---|---|
| macro | append/update local context parquet only when FRED observation changes |
| market_breadth | recompute from local yfinance price universe after price append |
| sector_returns | recompute from local yfinance price universe after price append |
| fundamentals | daily lightweight check for target tickers; weekly or filing-alert based broader SEC refresh |

## 5. 지표 갱신 게이트

| 항목 | 1D | 1W |
|---|---:|---:|
| current rows | 279005 | 53300 |
| candidate rows | 279005 | 53300 |
| current latest | 2026-05-04 | 2026-05-01 |
| candidate latest | 2026-05-04 | 2026-05-01 |
| non-finite count | 0 | 0 |
| checksum would change | False | False |
| partial week rows | - | 0 |

## 6. 제품 추론과 얇은 업로드 경계

CP134에서는 product checkpoint inference와 Supabase thin upload를 실행하지 않았다. 운영 단계에서는 1D line/band latest inference 후 `save_product_latest_predictions()`만 사용하고, 5티커 기준 predictions 10행과 evaluations 10행 수준을 예상한다. `ai.inference --save` bulk 저장, composite 저장, history bulk upload는 금지한다.

## 7. 준비 상태

| 항목 | 값 |
|---|---|
| local price latest | 2026-05-04 |
| local indicator 1D latest | 2026-05-04 |
| local indicator 1W latest | 2026-05-01 |
| macro latest | 2026-05-04 |
| breadth latest | 2026-05-01 |
| fundamentals latest filing_date | 2026-05-05 |
| sector latest | 2026-05-04 |
| product history latest | 2026-05-04 |
| EODHD fallback used | False |
| Supabase bulk read/write | False |

## 8. 최종 판단

yfinance 최신 가격 조회가 리허설 5티커 모두 빈 응답으로 끝났다. 신규 거래일 없음으로 판정하지 않고, daily append와 EODHD 해지 gate를 보류한다.

실패 목록: ['price update dry-run failed']

경고 목록: ['5 yfinance fetches returned empty', 'no new SEC filing rows for rehearsal tickers', 'product inference/thin upload left as dry-run boundary']

## 9. 금지 작업 미발생 확인

Supabase price_data/indicators/context 대량 write, Supabase 대량 read, 모델 학습, full inference 저장, DB row delete/update, EODHD 호출, 프론트 수정은 수행하지 않았다.
