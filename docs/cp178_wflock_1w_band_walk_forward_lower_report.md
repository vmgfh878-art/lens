# CP178-WFLOCK 1W Band Walk-Forward Lower Calibration

## 한 줄 결론

1W band는 실패가 아니다. walk-forward lower calibration 기준으로 보면 1D band보다 lower breach 안정성은 훨씬 좋은 후보권에 들어왔고, 현재 WARN은 1W에만 더 엄격한 strict 기준을 적용했기 때문에 남아 있다.

## 계약

- 대상: tide_s60_q10_q90_param
- calibration: walk_forward_lower_calibration
- 새 학습: 없음
- save-run / DB write / inference 저장 / live fetch / EODHD fallback / composite: 없음
- test 결과로 threshold/calibration 변경 없음

## 핵심 요약

- lower_breach mean/worst: 0.118452 / 0.138709
- coverage_abs_error mean/worst: 0.038757 / 0.062822
- p90_band_width mean/worst: 0.210630 / 0.221118
- band_width_ic mean/worst: 0.344148 / 0.322571
- downside_width_ic mean/worst: 0.060930 / 0.015942

## Bootstrap CI

- aggregate lower_breach CI95: 0.116842 ~ 0.118981
- aggregate coverage CI95: 0.764384 ~ 0.766866
- aggregate coverage_abs_error CI95: 0.033134 ~ 0.035616

## Fold별 결과

| fold | lower mean/worst | coverage error mean/worst | p90 width worst | downside IC mean |
|---|---:|---:|---:|---:|
| fold_1 | 0.132633/0.138709 | 0.053809/0.059893 | 0.213530 | 0.109993 |
| fold_2 | 0.098588/0.113205 | 0.009659/0.014794 | 0.210313 | 0.019267 |
| fold_3 | 0.124136/0.128579 | 0.052804/0.062822 | 0.221118 | 0.053528 |

## 1D Band Primary와 같은 표 비교

| timeframe | candidate | lower mean/worst | coverage error mean/worst | p90 width worst | band IC mean | downside IC mean |
|---|---|---:|---:|---:|---:|---:|
| 1W | tide_s60_q10_q90_param + walk_forward_lower_calibration | 0.118452/0.138709 | 0.038757/0.062822 | 0.221118 | 0.344148 | 0.060930 |
| 1D | tide_s60_q15_param | 0.142540/0.199795 | 0.025440/0.061201 | 0.102895 | 0.373995 | 0.086193 |

해석: 1W walk-forward lower는 1D primary보다 lower breach mean/worst가 낮다. coverage 평균은 1D보다 약하지만 worst coverage는 1D worst와 거의 같은 범위다.

## Ticker Concentration

- ticker_count: 451
- top10 lower breach share: 0.051153
- top10 any breach share: 0.036752
- top10 tickers: TTD, LULU, EPAM, NOW, SW, UNH, ACN, FISV, DECK, ORCL

## Stress / Calm

| bucket | lower mean/worst | coverage error mean/worst | p90 width worst | downside IC mean |
|---|---:|---:|---:|---:|
| calm | 0.093957/0.121107 | 0.022935/0.048751 | 0.205027 | 0.083393 |
| stress | 0.138030/0.160472 | 0.070580/0.092272 | 0.244866 | 0.050837 |

## 판정표

| 기준 | 실패 항목 | 판정 |
|---|---|---|
| strict 1W 기준 | lower_breach_mean, lower_breach_worst, coverage_abs_error_mean, coverage_abs_error_worst | CP178-LOSS로 이동 |
| 1D-1W 대칭 기준 | 없음 | 1W band product candidate로 이동 |

## 최종 분기

- 대칭 기준 채택 시: 1W band product candidate로 이동
- strict 기준 유지 시: CP178-LOSS로 이동
- 애매하면: research reserve 유지, 1D line v3 우선
