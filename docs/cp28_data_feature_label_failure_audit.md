# CP28-D End-to-End Data/Feature/Label Failure Audit

## 한 줄 결론

이제는 PatchTST를 더 두들기기 전에, 이 데이터가 애초에 이길 수 있는 데이터인지 확인해야 합니다. 모델 하나로 줄였으니 오히려 데이터/피처/라벨/검증을 더 깊게 파는 게 맞습니다.

## 감사 범위

이번 감사는 `1D`, `seq_len=252`, `horizon=5`, `h_max=20` 기준으로 현재 모델 입력 36개 피처와 raw future return 라벨을 점검했다. 산출물은 다음 3개 표와 이 보고서다.

- `docs/cp28_feature_inventory.csv`: 36개 피처 정의, 분포, 위험, clipping 후보
- `docs/cp28_feature_ic_summary.csv`: 50/100/200 티커 기준 단면 rank IC 요약
- `docs/cp28_universe_shift_report.json`: 50→100→200 확장 시 universe, target, split, feature drift 요약

36개 피처 구성은 원천 피처 29개와 달력 파생 피처 7개다. 원천 피처는 가격/거래량, 매크로, 시장폭, VIX regime, 재무 원시값, 가용성 플래그로 구성된다.

## 핵심 판정

현재 병목은 PatchTST 구조보다 데이터 계약에 더 가깝다. 특히 `open_ratio`, `high_ratio`, `low_ratio`는 조정종가와 비조정 OHLC가 섞인 것으로 보이며, 평균이 0 근처가 아니라 각각 0.812, 0.834, 0.790이고 p99가 22.94~23.97까지 튄다. 이 값은 정상적인 일중 OHLC 비율이 아니라 corporate action 왜곡일 가능성이 높다.

raw 5일 수익률 라벨은 시장/섹터 공통 성분이 크다. 200 입력 universe 기준 전체 target variance 중 동일 일자 평균이 27.17%, 일자+섹터 평균이 42.12%를 설명한다. 이 상태로 raw return을 맞히면 모델은 종목 alpha보다 시장/섹터 방향과 universe 구성을 같이 맞히는 문제를 풀게 된다.

검증 split도 완전히 닫혀 있지 않다. per-ticker split에는 `h_max=20` gap이 있지만, ticker별 상장/데이터 길이가 달라 200 입력 universe에서 train-val 중복 일자 512개, train-test 중복 일자 121개, val-test 중복 일자 246개가 생긴다. shared macro/regime/date feature가 있는 구조에서는 global date purge/embargo가 필요하다.

## Universe 요약

| 입력 | eligible | 제외 | train | val | test | market date R2 | date+sector R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 48 | 2 | 80,459 | 17,231 | 17,288 | 0.3162 | 0.5086 |
| 100 | 93 | 7 | 156,229 | 33,459 | 33,566 | 0.2933 | 0.4544 |
| 200 | 188 | 12 | 313,286 | 67,095 | 67,311 | 0.2717 | 0.4212 |

제외 사유는 모두 재무 sufficiency gate다. 100 입력에서는 `AMP`, `APA`, `BALL`, `BK`, `BXP`, `CHTR`, `CINF`가 제외됐고, 200 입력에서는 여기에 `CPAY`, `DUK`, `EA`, `EME`, `EXE`가 추가로 제외됐다.

## 실질 Alpha 후보 Top 10

아래 순위는 200 입력 universe test rank IC, train/test 부호 안정성, 피처 위험도를 함께 본 것이다. 1~5는 현재 형태에서도 비교적 깨끗한 후보이고, 6~10은 조건부 후보다.

| 순위 | 피처 | 해석 | 200 test IC | 판정 |
|---:|---|---|---:|---|
| 1 | `log_return` | 단기 reversal 성격 | -0.0089 | 유지 |
| 2 | `rsi` | 과열/침체 reversal | -0.0076 | 유지 |
| 3 | `ma_20_ratio` | 20일 이격도 reversal | -0.0058 | 유지 |
| 4 | `bb_position` | 밴드 위치 reversal | -0.0053 | 약한 winsorization 후 유지 |
| 5 | `ma_5_ratio` | 초단기 이격도 reversal | -0.0050 | 유지 |
| 6 | `ma_60_ratio` | 중기 trend/reversal regime 혼재 | 0.0147 | regime별 재검증 |
| 7 | `macd_ratio` | trend 계열 | 0.0117 | regime별 재검증 |
| 8 | `vol_change` | 거래량 충격 | 0.0005 | 로그 변환 또는 clipping 후 재검증 |
| 9 | `net_income` | 품질/규모 proxy | 0.0064 | 원시값 금지, margin/sector z-score 후 후보 |
| 10 | `revenue` | 규모 proxy | 0.0181 | 원시값 금지, scale-normalized 후 후보 |

주의할 점은 IC 절대값이 작다는 것이다. 현재 깨끗한 후보들의 200 test rank IC는 대체로 0.005~0.009 수준이다. 단독으로 모델을 강하게 이길 정도의 alpha라기보다, 데이터 계약을 고친 뒤 ensemble feature family로 확인해야 하는 약한 신호다.

## 무의미하거나 위험한 피처

가장 위험한 피처는 `open_ratio`, `high_ratio`, `low_ratio`다. `feature_svc.py`에서 `close`는 조정종가로 바꾸지만 `open/high/low`는 그대로 둔 뒤 전일 `close`와 비교한다. 이 때문에 split-adjusted close와 raw OHLC가 섞이는 것으로 보인다. 50→100 확장에서도 추가 티커의 OHLC ratio p99가 19대에서 49대까지 커져 universe shift의 가장 큰 분포 변화로 잡혔다.

재무 피처 `revenue`, `net_income`, `equity`, `eps`, `roe`, `debt_ratio`는 대부분 median이 0이고 p99가 큰 원시 규모값이다. `has_fundamentals`도 200 입력 model rows에서 1 비중이 7.25%에 불과하다. 지금 형태로는 alpha라기보다 재무 데이터 가용성, 기업 규모, 상장 기간, 섹터 편향을 학습할 위험이 크다.

`has_macro`는 200 입력 model rows에서 완전 상수 1.0이고, `has_breadth`도 0.993으로 거의 상수다. 모델 입력에는 남아 있지만 정보량은 사실상 낮다. 매크로/시장폭/regime/달력 피처는 동일 일자에서 티커 간 값이 같아서 단면 rank IC로는 측정되지 않는다. raw market timing에는 쓸 수 있지만, 종목 alpha 검증에는 market/sector excess target이 필요하다.

## Clipping/Winsorization 필요 피처

즉시 clipping 또는 재정의가 필요한 피처는 다음이다.

- `open_ratio`, `high_ratio`, `low_ratio`: 먼저 adjusted OHLC로 재계산해야 한다. 단순 winsorization만으로는 부족하다.
- `vol_change`: `log1p(volume)` 차분 또는 [-3, 3] 수준의 robust clipping 후보.
- `bb_position`: 정상 범위 밖 값이 있으므로 약한 winsorization 후보.
- `credit_spread_hy`: 초기 0 대체 구간이 있어 0 flag와 실제 값 분리 필요.
- `revenue`, `net_income`, `equity`, `eps`, `roe`, `debt_ratio`: 원시값 사용 금지. sector/date z-score, per-share, margin, asset-normalized, stale-age feature로 재설계 필요.

## 50→100 확장 실패 원인 후보

첫째, 50티커 결과는 universe subset 효과가 컸다. 기존 CP26 보고서에서 50 기준 q25/q20/q15 coverage는 각각 0.9368, 0.9726, 0.9951이었지만, 100 최종 checkpoint에서는 0.5446, 0.6500, 0.7493으로 크게 낮아졌다. CP27의 coverage-aware selection으로 100티커는 복구됐지만, CP28 200티커 q20-b2는 다시 coverage gate가 실패했고 test coverage 0.5907, long-short spread 0.00076, fee-adjusted return -0.0433으로 약해졌다.

둘째, ticker 확장이 알파벳 순서 기반이라 섹터/상장연한/기업 규모가 균형 있게 늘지 않는다. 50→100 추가분에서 가장 큰 feature drift는 `open_ratio/high_ratio/low_ratio`였고, 이는 데이터 계약 버그와 universe shift가 동시에 작동했을 가능성을 키운다.

셋째, raw return label이 시장/섹터 공통 성분을 크게 포함한다. universe가 바뀌면 target 분산 구조와 long-short 평가가 함께 바뀐다. 즉 50에서 맞춘 calibration이 100/200으로 이동할 때 같은 의미를 유지하지 못한다.

넷째, ticker별 split 때문에 fold 간 날짜가 겹친다. 짧은 history ticker의 train이 긴 history ticker의 val/test 날짜와 겹치면, shared macro/date feature가 validation을 낙관적으로 만들거나 반대로 regime mismatch를 만들 수 있다.

## Market/Sector Excess Target 필요성

필요하다. 200 입력 기준 raw 5일 수익률의 market date R2는 0.2717이고, date+sector R2는 0.4212다. 이는 현재 label의 40% 안팎이 "어떤 종목인가"보다 "그 날짜와 섹터가 어땠는가"로 설명된다는 뜻이다.

다음 target을 병렬로 만들어야 한다.

- `raw_future_return`: market timing과 제품 데모용 유지
- `market_excess_return`: `ticker_return - universe_date_mean`
- `sector_excess_return`: `ticker_return - sector_date_mean`
- `rank_target`: 동일 일자 단면 rank 또는 z-score

PatchTST를 계속 쓸 수는 있지만, 모델이 맞히는 것이 price direction인지 cross-sectional alpha인지 target 차원에서 분리해야 한다.

## Purge/Embargo Split 필요성

필요하다. 현재 split은 ticker 내부 sample index 기준으로 `h_max=20` gap을 둔다. 이 점은 좋다. 하지만 전체 universe의 날짜 기준으로는 fold가 겹친다. 200 입력 기준 train-val overlap 512일, train-test overlap 121일, val-test overlap 246일이 확인됐다.

권장 split은 global date 기준이다.

- fold boundary를 전체 거래일 기준으로 먼저 정한다.
- label horizon 5일과 최대 lookback 252일을 고려해 boundary 주변을 purge한다.
- validation/test 시작 전 최소 20 거래일 embargo를 둔다.
- ticker별 짧은 history는 해당 global fold 안에서만 샘플을 만들고, 날짜를 앞으로 당기지 않는다.

## 다음 성능 개선 우선순위 3개

1. OHLC/adjusted close 데이터 계약을 고친다. `open_ratio/high_ratio/low_ratio`는 현재 가장 위험한 버그 후보이며, 수정 전 모델 비교는 신뢰하기 어렵다.

2. target을 `raw`, `market_excess`, `sector_excess`, `rank`로 분리하고 같은 split에서 baseline IC를 다시 낸다. raw return만 쓰면 모델이 시장/섹터 exposure를 alpha로 착각한다.

3. global purge/embargo split을 도입하고 50/100/200을 sector-balanced universe로 다시 만든다. 그 뒤에야 PatchTST 구조, loss, calibration을 다시 만지는 것이 맞다.

## 최종 판단

PatchTST를 새 모델로 바꾸기 전에 데이터가 먼저 이겨야 한다. 현재 36개 피처 안에는 약한 단기 reversal 신호가 있지만, OHLC 조정 오류 의심, 재무 원시값/0 대체, raw return target의 시장/섹터 성분, ticker별 split overlap이 같이 섞여 있다. 다음 CP는 모델 확장이 아니라 데이터 계약 수리와 label/split 재검증으로 가야 한다.
