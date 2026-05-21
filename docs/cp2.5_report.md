[CP2.5] 완료

## 1. price_data 분포
- 전체 ticker 수: 503
- row_count 분위수: p10=2842.0, p25=2843.0, p50=2843.0, p75=2843.0, p90=2843.0, max=2844.0
- min_date 분위수: 가장 이른 p10=2015-01-02, 가장 늦은 p90=2015-01-02
- 2015-01-01 이전부터 있는 ticker 수: 0
- 2020-01-01 이후 시작하는 ticker 수: 17

## 2. indicators 분포 (1D / 1W / 1M)
### 1D
- 전체 ticker 수: 503
- row_count p10/p50/p90: 753.0 / 757.0 / 757.0
- has_fundamentals=1 비율(time-weighted): 0.0003
- min_date p10/p90: 2023-04-18 / 2023-04-24

### 1W
- 전체 ticker 수: 502
- row_count p10/p50/p90: 157.0 / 158.0 / 158.0
- has_fundamentals=1 비율(time-weighted): 0.0003
- min_date p10/p90: 2023-04-21 / 2023-04-28

### 1M
- 전체 ticker 수: 494
- row_count p10/p50/p90: 37.0 / 37.0 / 37.0
- has_fundamentals=1 비율(time-weighted): 0.0004
- min_date p10/p90: 2023-04-30 / 2023-04-30

## 3. company_fundamentals
- 전체 ticker 수: 503
- 분기 수 분위수 p10/p50/p90: 12.0 / 12.0 / 12.0
- 8분기 이상 보유 ticker 수: 477
- 확인된 기대치 477, 실제 477

## 4. AAPL 직접 진단
- price_data: min_date=2015-01-02, max_date=2026-04-23, row_count=2843
- indicators 1D: min_date=2023-04-18, max_date=2026-04-23, row_count=757
- indicators 1W: min_date=2023-04-21, max_date=2026-04-24, row_count=158
- fundamentals: min_date=2021-09-25, max_date=2025-12-27, quarter_count=14

- 원시 파이프라인 row 감소 단계별
  N0 (price raw): 2843
  N1 (resample): 2843
  N2 (base dropna): 753
  N3 (regime merge): 753
  N4 (fundamentals merge + 8Q gate): 2843
  N5 (final dropna): 753

## 5. sync_state 힌트
- sync_prices:1D earliest last_cursor: None
- sync_prices:1D latest last_cursor: None
- compute_indicators:1D earliest last_cursor: 2026-04-22
- compute_indicators:1D latest last_cursor: 2026-04-23

## 6. 가설 판정
- 가설 H1 (price_data 자체가 짧다): [REJECTED]
  근거: price_data row_count 중앙값=2843.0
- 가설 H2 (price OK, indicators 짧음): [HOLDS]
  근거: price_data p50=2843.0, indicators 1D p50=757.0
- 가설 H3 (원천은 OK, 피처에서 줄어듦): [HOLDS]
  근거: AAPL 파이프라인 N1=2843, N5=753

**원인 결정**: H3

## 7. CP2.6 제안 (다음 CP 참고)
- feature_svc 단계별 drop 규칙 재조정
