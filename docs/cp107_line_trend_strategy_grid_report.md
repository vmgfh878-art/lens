# CP107-P Line Trend 100티커 룰 탐색 보고서

작성 시각: 2026-05-04 19:35 KST

## 1. 요약

CP107-P는 보수적 예측선 중심 전략인 Line Trend 계열만 대상으로 100티커 룰 탐색을 수행한 작업이다. 프론트 화면, 모델 학습, inference, DB write는 수행하지 않았다.

결론부터 말하면, **holdout 기준 생존 후보는 없었다.** 다만 최상위 후보는 기존 Line Trend v1 유사 baseline보다 수익률 방어와 강한 상승 티커 추종이 개선됐다. 문제는 평균 시장 참여율이 36% 수준으로 생존 기준 45%에 못 미치고, 기준 통과 티커 비율도 3.16%에 그쳤다는 점이다.

이번 실험의 의미는 다음과 같다.

- 보수적 예측선만으로도 MDD 개선과 손실 회피 성격은 확인됐다.
- baseline보다 상승 추종을 개선한 룰은 있었다.
- 그러나 line 단독 전략은 아직 티커별 안정성이 부족하다.
- 다음 Balance 실험에서는 line을 단독 청산 신호로 쓰기보다 band 위험 필터와 함께 해석하는 쪽이 더 적절하다.

## 2. 산출물

- `ai/cp107_line_trend_strategy_grid.py`
- `docs/cp107_line_trend_strategy_grid_metrics.json`
- `docs/cp107_line_trend_strategy_grid_top_candidates.csv`
- `docs/cp107_line_trend_strategy_grid_report.md`

주의: 디버그 실행 중 최초 grid 설정이 4,320개로 과도하게 잡힌 것을 확인했고, 지시 범위에 맞게 최종 runner를 1,080개 coarse grid로 수정한 뒤 최종 산출물을 다시 생성했다. 보고서와 JSON/CSV는 최종 1,080개 실행 기준이다.

## 3. 금지 사항 준수

- GPU 사용 없음
- 모델 학습 없음
- inference 실행 없음
- DB write 없음
- Supabase `price_data` / `indicators` 대량 read 없음
- 프론트 수정 없음
- fake data 생성 없음

사용 데이터:

- 가격: `data/parquet/price_data_yfinance.parquet`
- 지표: `data/parquet/indicators_yfinance_1D.parquet`
- universe: `data/parquet/stock_info.parquet`
- LM prediction history: `patchtst-1D-efad3c29d803` 기준 read-only API

가격 컬럼은 `adjusted_close`를 우선 사용하고, 없으면 `close`로 fallback한다.

## 4. 사용 티커 수와 제외 티커

local yfinance universe:

- 전체 universe: 100티커
- 사용 가능 티커: 95티커
- 제외 티커: 5티커

제외 사유:

| 티커 | 사유 |
|---|---|
| LMT | LM prediction history 없음 |
| MS | LM prediction history 없음 |
| QQQ | LM prediction history 없음 |
| SPY | LM prediction history 없음 |
| T | LM prediction history 없음 |

## 5. Prediction History Coverage

제품 LM run:

- run_id: `patchtst-1D-efad3c29d803`
- 역할: 1D 보수적 예측선

coverage:

- 사용 티커 95개 모두 prediction row 200개
- min asof: 2025-06-16
- max asof: 2026-05-01
- 평균 prediction row: 200개

단, 기본 1개월 holdout에서는 price/prediction 날짜 매칭 usable 티커가 3개뿐이었다. 그래서 지시대로 최근 2개월 holdout 대안을 적용했다.

## 6. 기간 Split

기본 요청 split:

| 구분 | 기간 | usable 티커 |
|---|---:|---:|
| 탐색 | 2025-06-18 ~ 2026-03-31 | 95 |
| 기본 1개월 holdout | 2026-04-01 ~ 2026-05-01 | 3 |

최종 적용 split:

| 구분 | 기간 | usable 티커 |
|---|---:|---:|
| 탐색 | 2025-06-18 ~ 2026-02-28 | 95 |
| holdout | 2026-03-01 ~ 2026-05-01 | 95 |

선택 이유:

- 1개월 holdout usable ticker가 30개 미만이었다.
- 최근 2개월 holdout은 95티커를 평가할 수 있었다.
- 최근 1개월 holdout 결과는 JSON에 참고 지표로 별도 기록했다.

## 7. 실험한 파라미터 조합

최종 coarse grid:

- line_entry_threshold: -0.004, -0.002, 0.000, 0.002, 0.004
- line_exit_threshold: -0.006, -0.010, -0.014
- trend_floor: -0.03, -0.05, off
- trend_override: true, false
- rsi_entry_cap: off, 75, 80
- rsi_exit_guard: only_if_trend_weak
- exit_confirm_days: 1, 2
- reentry_confirm_days: 1, 2
- cooldown_days: 0

총 조합 수:

```text
5 * 3 * 3 * 2 * 3 * 1 * 2 * 2 * 1 = 1,080개
```

coarse grid에서 줄인 항목:

- `line_exit_threshold=-0.018` 제외
- `trend_floor=-0.08` 제외
- `rsi_exit_guard=off` 제외
- `cooldown_days=1/2` 제외

이유:

- CP107 지시가 500~1,500개 안쪽 coarse grid를 권장했다.
- RSI는 강한 추세 청산 사유로 과하게 쓰지 않기 위해 `only_if_trend_weak`만 유지했다.
- cooldown은 이번 실험의 핵심 질문이 아니므로 0일로 고정했다.

## 8. 신호 생성 방식

사용 신호:

- 보수적 예측선 수익률:
  - `conservative_series` 마지막 유효값 우선
  - 없으면 `line_series` 마지막 유효값 사용
  - `prediction_value / adjusted_close - 1`
- 60일 추세:
  - indicator `ma_60_ratio` 우선
  - 없으면 가격 기준 rolling 60일 평균 대비 괴리율 계산
- RSI:
  - indicator `rsi`
  - 0~1 범위면 0~100으로 정규화

lookahead leakage 방지:

- 신호는 prediction `asof_date` 기준으로 생성했다.
- 해당 asof 신호는 다음 거래일 수익률부터 적용했다.
- 미래 실현수익률은 신호 생성에 쓰지 않았다.

## 9. 큰 하락 구간 정의

CP105 정의를 유지했다.

```text
large_loss_threshold = min(-2%, 일간 수익률 하위 20%)
```

손실 회피율은 큰 하락일 중 전략이 현금 대기 상태였던 비율이다.

강한 상승 티커 정의:

- 평가 구간 Buy & Hold 수익률 상위 25%

하락/횡보 티커 정의:

- 평가 구간 Buy & Hold 수익률 하위 50%

## 10. 기존 Line Trend v1 유사 Baseline

baseline은 CP105 화면의 Line Trend v1과 유사한 방어형 규칙으로 계산했다.

baseline 규칙:

- 진입: 보수적 예측선 >= 0%
- 청산 기준: 보수적 예측선 < -3.0%
- trend_floor: -5%
- trend_override: true
- RSI entry cap: 80
- exit_confirm_days: 1
- reentry_confirm_days: 1

holdout 결과:

| 지표 | 값 |
|---|---:|
| 평가 티커 | 95 |
| 평균 전략 수익률 | 0.95% |
| 평균 Buy & Hold 수익률 | 2.37% |
| Buy & Hold 대비 수익률 비율 | 0.55 |
| 평균 MDD 개선폭 | 8.42%p |
| 평균 Sharpe | -0.21 |
| Buy & Hold Sharpe | -0.10 |
| 평균 Sortino | 0.00 |
| Buy & Hold Sortino | 0.20 |
| 손실 회피율 | 64.84% |
| 시장 참여율 | 34.39% |
| 평균 거래 횟수 | 0.91 |
| 기준 통과 티커 비율 | 3.16% |
| 강한 상승 티커 수익률 방어율 | 30.27% |
| 하락/횡보 티커 MDD 개선폭 | 10.64%p |

해석:

- MDD와 손실 회피는 좋다.
- 하지만 시장 참여율이 낮아 상승 구간을 너무 놓친다.
- 강한 상승 티커 수익률 방어율이 30.27%라 Line Trend 이름값에는 부족하다.

## 11. 상위 10개 후보

holdout score 기준 상위 후보:

| 순위 | entry | exit | trend | override | RSI cap | exit 확인 | 재진입 확인 | 수익률 | B&H | 수익률 비율 | MDD 개선 | 손실 회피 | 참여율 | 강한 상승 방어 | 생존 |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.000 | -0.014 | off | true | off | 2 | 2 | 4.17% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.35% | 62.41% | 아니오 |
| 2 | 0.000 | -0.014 | off | true | 80 | 2 | 2 | 4.17% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.35% | 62.41% | 아니오 |
| 3 | 0.000 | -0.014 | off | false | off | 2 | 2 | 4.17% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.35% | 62.41% | 아니오 |
| 4 | 0.000 | -0.014 | off | false | 80 | 2 | 2 | 4.17% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.35% | 62.41% | 아니오 |
| 5 | 0.000 | -0.014 | off | true | 75 | 2 | 2 | 4.16% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.33% | 62.41% | 아니오 |
| 6 | 0.000 | -0.014 | off | false | 75 | 2 | 2 | 4.16% | 2.37% | 0.80 | 8.38%p | 64.87% | 36.33% | 62.41% | 아니오 |
| 7 | 0.004 | -0.014 | off | true | off | 2 | 2 | 4.20% | 2.37% | 0.79 | 8.46%p | 65.05% | 35.57% | 62.36% | 아니오 |
| 8 | 0.004 | -0.014 | off | true | 80 | 2 | 2 | 4.20% | 2.37% | 0.79 | 8.46%p | 65.05% | 35.57% | 62.36% | 아니오 |
| 9 | 0.004 | -0.014 | off | false | off | 2 | 2 | 4.20% | 2.37% | 0.79 | 8.46%p | 65.05% | 35.57% | 62.36% | 아니오 |
| 10 | 0.004 | -0.014 | off | false | 80 | 2 | 2 | 4.20% | 2.37% | 0.79 | 8.46%p | 65.05% | 35.57% | 62.36% | 아니오 |

상위권 특징:

- trend filter를 끄는 쪽이 상위권에 많았다.
- exit threshold는 -1.4%가 좋았다.
- exit/reentry confirmation은 2일이 좋았다.
- RSI cap은 결과에 큰 차이를 만들지 못했다.

## 12. 최상위 후보 상세

최상위 규칙:

- line_entry_threshold: 0.0
- line_exit_threshold: -0.014
- trend_floor: off
- trend_override: true
- rsi_entry_cap: off
- rsi_exit_guard: only_if_trend_weak
- exit_confirm_days: 2
- reentry_confirm_days: 2
- cooldown_days: 0

holdout 결과:

| 지표 | 값 |
|---|---:|
| 평가 티커 | 95 |
| 평균 전략 수익률 | 4.17% |
| 평균 Buy & Hold 수익률 | 2.37% |
| Buy & Hold 대비 수익률 비율 | 0.80 |
| 평균 초과 수익률 | 1.80%p |
| 평균 전략 MDD | -4.70% |
| 평균 Buy & Hold MDD | -13.08% |
| 평균 MDD 개선폭 | 8.38%p |
| 평균 Sharpe | 0.28 |
| Buy & Hold Sharpe | -0.10 |
| 평균 Sortino | 0.40 |
| Buy & Hold Sortino | 0.20 |
| 손실 회피율 | 64.87% |
| 시장 참여율 | 36.35% |
| 평균 거래 횟수 | 0.45 |
| 평균 보유 기간 | 38.07일 |
| 기준 통과 티커 비율 | 3.16% |
| 강한 상승 티커 수익률 방어율 | 62.41% |
| 하락/횡보 티커 MDD 개선폭 | 10.90%p |

생존 판정:

- MDD 개선: 통과
- Buy & Hold 대비 수익률 비율: 통과
- 손실 회피율: 통과
- 강한 상승 티커 수익률 방어율: 통과
- 거래 횟수: 통과
- 시장 참여율: 실패
- 기준 통과 티커 비율: 실패

따라서 최상위 후보도 생존 후보로 채택하지 않는다.

## 13. AAPL / MSFT / NVDA 비교

최종 holdout 기간 2026-03-01 ~ 2026-05-01 기준이다.

| 티커 | 항목 | baseline | 최상위 후보 | 변화 |
|---|---|---:|---:|---:|
| AAPL | 전략 수익률 | 0.00% | 0.00% | 0.00%p |
| AAPL | B&H 수익률 | 5.83% | 5.83% | - |
| AAPL | 수익률 비율 | 0.00 | 0.00 | 0.00 |
| AAPL | MDD 개선폭 | 6.83%p | 6.83%p | 0.00%p |
| AAPL | 시장 참여율 | 0.00% | 0.00% | 0.00%p |
| MSFT | 전략 수익률 | 0.00% | 0.00% | 0.00%p |
| MSFT | B&H 수익률 | 3.99% | 3.99% | - |
| MSFT | 수익률 비율 | 0.00 | 0.00 | 0.00 |
| MSFT | MDD 개선폭 | 13.13%p | 13.13%p | 0.00%p |
| MSFT | 시장 참여율 | 0.00% | 0.00% | 0.00%p |
| NVDA | 전략 수익률 | 2.70% | 10.12% | +7.42%p |
| NVDA | B&H 수익률 | 8.76% | 8.76% | - |
| NVDA | 수익률 비율 | 0.31 | 1.16 | +0.85 |
| NVDA | MDD 개선폭 | -0.08%p | 0.00%p | +0.08%p |
| NVDA | 시장 참여율 | 86.05% | 97.67% | +11.63%p |

해석:

- NVDA에서는 상승 추종 문제가 개선됐다.
- AAPL/MSFT는 여전히 현금 대기 상태로 남아 baseline 대비 개선이 없었다.
- NVDA 개선도 시장 참여율이 97.67%라 사실상 Buy & Hold에 가까운 성격이 있다.

## 14. 기존 전략 대비 개선 여부

baseline 대비 최상위 후보:

- 평균 전략 수익률: 0.95% → 4.17%
- 수익률 방어율: 0.55 → 0.80
- Sharpe: -0.21 → 0.28
- Sortino: 0.00 → 0.40
- 강한 상승 티커 수익률 방어율: 30.27% → 62.41%
- MDD 개선폭: 8.42%p → 8.38%p로 거의 유지
- 손실 회피율: 64.84% → 64.87%로 거의 유지
- 시장 참여율: 34.39% → 36.35%로 소폭 개선이나 여전히 낮음

따라서 기존 Line Trend v1 유사 baseline보다는 개선됐지만, 제품 전략 후보로 올리기에는 부족하다.

## 15. Holdout 기준 생존 후보 여부

생존 후보:

```text
0개
```

실패한 핵심 이유:

- 시장 참여율이 생존 기준 45%에 못 미쳤다.
- 기준 통과 티커 비율이 50% 기준에 크게 못 미쳤다.
- AAPL/MSFT 같은 강한 상승 티커 중 일부는 여전히 전혀 따라가지 못했다.
- 최상위 후보도 티커별 안정성보다는 평균 지표 개선에 기대고 있다.

## 16. 다음 Balance 실험에 넘길 교훈

1. Line 단독 조건은 방어 성격이 강해지면 상승 추종을 쉽게 잃는다.
2. trend_floor를 강하게 걸면 평균 성능이 오히려 약해졌다. 제품 전략에서는 추세 필터를 청산 조건으로 과하게 쓰지 않는 편이 낫다.
3. 2일 확인 조건은 false exit를 줄이는 데 도움이 됐다.
4. line_entry_threshold는 0% 근처가 안정적이었다.
5. line_exit_threshold는 -1.4% 근처가 baseline -3.0%보다 상승 추종을 더 살렸다.
6. Balance 실험에서는 line 약화만으로 바로 청산하지 말고, AI 밴드 하방 위험 확대와 동시에 볼 필요가 있다.
7. AAPL/MSFT처럼 signal이 계속 보수적으로 나오는 티커는 line 단독 전략이 시장 참여를 만들지 못하므로, 가격 추세나 band 완화 조건을 재진입 보조로 쓰는 실험이 필요하다.

## 17. 남은 한계

- 최근 1개월 holdout usable ticker가 3개뿐이라 참고 지표로만 기록했다.
- 룰 탐색은 1,080개 coarse grid이며, 모든 후보를 완전 탐색한 것은 아니다.
- prediction history는 read-only API를 사용했으므로 API 응답 속도에 실행 시간이 영향을 받았다.
- 거래 비용은 10bp 고정이다.
- 포지션은 단일 티커 100% 또는 현금 100%만 사용했다.
- 좋은 후보가 나와도 이번 CP에서는 프론트 기본 전략에 반영하지 않았다.

## 18. 검증 결과

실행한 검증:

- `python -m py_compile ai\cp107_line_trend_strategy_grid.py`
- `python ai\cp107_line_trend_strategy_grid.py`
- `docs/cp107_line_trend_strategy_grid_metrics.json` JSON parse 확인
- `docs/cp107_line_trend_strategy_grid_top_candidates.csv` 생성 확인
- `Get-Process -Name python,pythonw` 확인

결과:

- py_compile 통과
- metrics JSON 생성 및 parse 가능
- top candidates CSV 생성
- CPU only 확인
- DB write 없음
- 모델 학습 없음
- inference 저장 없음
- Supabase price/indicator 대량 read 없음
- 프론트 수정 없음

남은 python 프로세스는 기존 백엔드 실행 프로세스 2개뿐이며, CP107 실험 runner 프로세스는 남아 있지 않았다.
