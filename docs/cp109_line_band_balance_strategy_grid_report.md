# CP109-P Line + Band Balance 해석 기반 전략 탐색 보고서

작성 시각: 2026-05-04 21:18 KST

## 1. 요약

CP109-P는 CP106 Band 단독 실패, CP107 Line 단독 실패, CP108 Lens 지표 해석 study를 바탕으로 line과 band를 함께 읽는 Balance 전략 후보를 100티커 기준으로 탐색한 작업이다. 이번 CP는 프론트 반영이 아니라 전략 후보 발굴과 해석 검증이다.

결론:

- **holdout 기준 생존 후보는 0개**였다.
- 다만 최상위 Balance 후보는 CP107 Line 단독 best보다 손실 회피율과 시장 참여율을 일부 개선했다.
- 동시에 CP107보다 MDD 개선폭과 평균 수익률은 낮아졌다.
- 가장 큰 실패 원인은 **기준 통과 티커 비율이 1.05%로 너무 낮다**는 점이다.
- 따라서 지금 바로 프론트 백테스트 기본 전략으로 반영할 후보는 없다.

최상위 후보는 “line 진입 + band 신규 진입 제한 + line 약화/band 위험 confirm” 구조였다. 방향 자체는 맞지만, 티커별 안정성이 부족하다.

## 2. 산출물

- `ai/cp109_line_band_balance_strategy_grid.py`
- `docs/cp109_line_band_balance_strategy_grid_metrics.json`
- `docs/cp109_line_band_balance_strategy_grid_top_candidates.csv`
- `docs/cp109_line_band_balance_strategy_grid_report.md`

## 3. 금지 사항 준수

- 프론트 수정 없음
- DB write 없음
- 모델 학습 없음
- inference 실행 없음
- Supabase `price_data` / `indicators` 대량 read 없음
- composite 모델/저장 없음
- fake data 생성 없음
- CPU-only 실행

사용 데이터:

- 가격: `data/parquet/price_data_yfinance.parquet`
- 보조지표: `data/parquet/indicators_yfinance_1D.parquet`
- universe: `data/parquet/stock_info.parquet`
- line prediction history: `patchtst-1D-efad3c29d803`
- band prediction history: `cnn_lstm-1D-d0c780dee5e8`

가격은 `adjusted_close`를 우선 사용하고, 없으면 `close`를 사용했다.

## 4. Coverage

| 항목 | 값 |
|---|---:|
| 요청 universe | 100 |
| 사용 가능 티커 | 95 |
| 제외 티커 | 5 |
| line prediction rows | 티커별 200개 |
| band prediction rows | 티커별 200개 |
| min asof | 2025-06-16 |
| max asof | 2026-05-01 |

제외 티커:

| 티커 | 사유 |
|---|---|
| LMT | line/band prediction history 없음 |
| MS | line/band prediction history 없음 |
| QQQ | line/band prediction history 없음 |
| SPY | line/band prediction history 없음 |
| T | line/band prediction history 없음 |

## 5. 기간 Split

기본 1개월 holdout은 usable ticker가 30개 미만인 구간이 생길 수 있어, CP107과 같은 기준으로 최근 2개월 holdout을 최종 선택했다.

| 구분 | 기간 | usable 티커 |
|---|---:|---:|
| 탐색 | 2025-06-18 ~ 2026-02-28 | 95 |
| holdout | 2026-03-01 ~ 2026-05-01 | 95 |

최근 1개월 holdout은 JSON에 참고 지표로 남겼다.

## 6. 실험한 전략군

총 rule 수:

```text
720개
```

전략군별 rule 수:

| 전략군 | rule 수 | 의도 |
|---|---:|---|
| Balance Entry Filter | 144 | line 양호 진입, band 확장/위험은 신규 진입 제한 |
| Balance Risk Confirm | 216 | line 약화와 lower/width 위험이 동시에 있을 때 청산 |
| Balance Trend Continuation | 72 | line이 양호하면 band 불확실만으로 매도하지 않음 |
| Balance Position Sizing | 144 | 100%, 50%, 현금 3단계 exposure 허용 |
| Balance Hybrid | 144 | 진입 제한, risk confirm, partial exposure 결합 |

CP108 해석 반영:

- `line_return_signal`: 진입 방향의 기본 신호
- `lower_band_risk`: line 약화 시 risk veto/confirm
- `band_width_expansion`: 불확실성 확장 confirm
- `upper_breach_event`: 사후 지표라 전략 입력 제외, 무조건 매도 신호로 쓰지 않음
- `line_band_disagreement`: Balance family 설계의 핵심

중요한 점:

- upper breach는 미래 사후 이벤트이므로 전략 입력으로 쓰지 않았다.
- 상단 밴드 돌파를 매도 신호로 쓰지 않는 철학은 “band 단독 매도 금지”와 “line 양호 시 보유 유지” 구조로만 반영했다.

## 7. 생존 기준

holdout 기준:

- 평균 MDD 개선폭 > 0
- 손실 회피율 >= 0.55
- 단순 보유 대비 수익률 방어율 >= 0.75
- 시장 참여율 0.45 ~ 0.90
- 강한 상승 티커 수익률 방어율 >= 0.60
- 기준 통과 티커 비율 >= 0.15
- 거래 횟수 과도하지 않을 것

결과:

```text
생존 후보 0개
```

조건별 통과 rule 수:

| 조건 | 통과 rule 수 / 720 |
|---|---:|
| MDD 개선 | 720 |
| 손실 회피율 | 432 |
| 수익률 방어율 | 232 |
| 시장 참여율 | 288 |
| 강한 상승 방어율 | 180 |
| 거래 횟수 | 720 |
| 기준 통과 티커 비율 | 0 |

가장 중요한 실패 지점은 `기준 통과 티커 비율`이다. 평균 지표는 일부 좋아도, 개별 티커 단위로 안정적으로 통과하는 후보가 없었다.

## 8. 최상위 후보

최상위 후보:

- family: `balance_entry_filter`
- line_entry_threshold: -0.2%
- line_hold_threshold: -1.4%
- lower_risk_threshold: -5%
- width_expansion_threshold: 1.10
- entry_band_filter: `block_expansion`
- exit_risk_mode: `lower_or_width`
- confirm_days: 2
- reentry_confirm_days: 2
- position_mode: binary

holdout 결과:

| 지표 | 값 |
|---|---:|
| 평가 티커 | 95 |
| 평균 전략 수익률 | 3.98% |
| 평균 단순 보유 수익률 | 2.37% |
| 단순 보유 대비 수익률 방어율 | 1.74 |
| 평균 초과 수익률 | 1.61%p |
| 전략 MDD | -5.93% |
| 단순 보유 MDD | -13.08% |
| MDD 개선폭 | 7.16%p |
| 전략 Sharpe | 0.07 |
| 단순 보유 Sharpe | -0.10 |
| 전략 Sortino | 0.37 |
| 단순 보유 Sortino | 0.20 |
| 손실 회피율 | 55.59% |
| 시장 참여율 | 44.65% |
| 평균 거래 횟수 | 0.92 |
| 평균 보유 기간 | 30.56일 |
| 강한 상승 티커 수익률 방어율 | 67.09% |
| 하락/횡보 티커 MDD 개선폭 | 9.97%p |
| 기준 통과 티커 비율 | 1.05% |

판정:

- MDD 개선 통과
- 손실 회피율 통과
- 수익률 방어율 통과
- 강한 상승 방어율 통과
- 거래 횟수 통과
- 시장 참여율은 45% 기준에 아주 조금 못 미침
- 기준 통과 티커 비율은 크게 실패

따라서 평균 성능만 보면 꽤 그럴듯하지만, 제품 후보로 올릴 안정성은 부족하다.

## 9. 전략군별 Best

| 전략군 | 수익률 | 수익률 방어율 | MDD 개선 | 손실 회피 | 시장 참여 | 강한 상승 방어 | 통과 티커 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entry Filter | 3.98% | 1.74 | 7.16%p | 55.59% | 44.65% | 67.09% | 1.05% |
| Risk Confirm | 2.62% | 1.17 | 6.73%p | 53.09% | 46.05% | 56.81% | 1.05% |
| Trend Continuation | 2.62% | 1.17 | 6.73%p | 53.09% | 46.05% | 56.81% | 1.05% |
| Position Sizing | 1.32% | 0.74 | 10.11%p | 79.11% | 20.07% | 24.71% | 0.00% |
| Hybrid | 1.06% | 0.99 | 9.37%p | 73.41% | 26.29% | 27.33% | 0.00% |

해석:

- Entry Filter가 가장 균형이 좋았다.
- Position Sizing과 Hybrid는 MDD와 손실 회피는 좋아졌지만 시장 참여율과 상승 추종이 너무 낮았다.
- Risk Confirm/Trend Continuation은 참여율은 기준 안에 들어오지만 손실 회피와 강한 상승 방어가 기준에 못 미쳤다.

## 10. CP106 Band 단독 대비 개선 여부

CP106 Band Risk best는 holdout usable ticker가 3개뿐이라 직접 비교는 제한적이다. 그래도 방향성은 다음과 같다.

Band Risk best:

- 수익률 방어율: 0.65
- MDD 개선폭: 3.32%p
- 손실 회피율: 33.33%
- 시장 참여율: 66.67%
- 기준 통과 티커 비율: 0%

CP109 Balance best:

- 수익률 방어율: 1.74
- MDD 개선폭: 7.16%p
- 손실 회피율: 55.59%
- 시장 참여율: 44.65%
- 기준 통과 티커 비율: 1.05%

개선점:

- line을 진입 방향으로 쓰면서 수익률 방어율이 크게 개선됐다.
- band 단독보다 손실 회피율이 좋아졌다.
- MDD 개선폭도 더 커졌다.

주의:

- CP106 holdout 표본이 3개뿐이라 정량 비교는 약하게 봐야 한다.
- Balance도 티커별 통과율은 여전히 낮다.

## 11. CP107 Line 단독 대비 개선 여부

CP107 Line Trend best:

- 평균 전략 수익률: 4.17%
- 수익률 방어율: 0.80
- MDD 개선폭: 8.38%p
- 손실 회피율: 64.87%
- 시장 참여율: 36.35%
- 강한 상승 방어율: 62.41%
- 기준 통과 티커 비율: 3.16%

CP109 Balance best:

- 평균 전략 수익률: 3.98%
- 수익률 방어율: 1.74
- MDD 개선폭: 7.16%p
- 손실 회피율: 55.59%
- 시장 참여율: 44.65%
- 강한 상승 방어율: 67.09%
- 기준 통과 티커 비율: 1.05%

개선점:

- 시장 참여율이 36.35%에서 44.65%로 올라왔다.
- 강한 상승 티커 수익률 방어율이 62.41%에서 67.09%로 올라왔다.
- 평균 수익률 방어율도 좋아졌다.

나빠진 점:

- MDD 개선폭은 8.38%p에서 7.16%p로 낮아졌다.
- 손실 회피율은 64.87%에서 55.59%로 낮아졌다.
- 기준 통과 티커 비율은 3.16%에서 1.05%로 낮아졌다.

해석:

- CP108 해석을 반영한 Balance 구조는 시장 참여율과 상승 추종을 살리는 데는 도움이 됐다.
- 하지만 티커별 일관성은 오히려 더 약해졌다.

## 12. AAPL / MSFT / NVDA 대표 티커 결과

최상위 후보 기준:

| 티커 | 전략 수익률 | 단순 보유 | 수익률 방어율 | MDD 개선 | 손실 회피 | 시장 참여 | 거래 횟수 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | -2.73% | 5.83% | -0.47 | 2.63%p | 66.67% | 11.63% | 2 |
| MSFT | 0.00% | 3.99% | 0.00 | 13.13%p | 100.00% | 0.00% | 0 |
| NVDA | 10.12% | 8.76% | 1.16 | 0.00%p | 0.00% | 97.67% | 1 |

해석:

- NVDA는 잘 따라갔다.
- AAPL은 방어는 했지만 수익 추종에 실패했다.
- MSFT는 사실상 항상 현금에 가까워 제품 전략으로는 좋지 않다.
- 대표 티커 3개에서도 전략 성격이 티커별로 크게 갈렸다.

## 13. CP108 해석 규칙이 성능에 도움이 됐는가?

부분적으로 도움이 됐다.

도움이 된 규칙:

- line을 진입 방향으로 쓴 것
- band width expansion을 신규 진입 제한으로 쓴 것
- band 단독 매도를 피하고 line 약화와 결합한 것
- 상단 돌파를 매도 신호로 쓰지 않은 것

도움이 약했던 규칙:

- lower risk와 width expansion threshold 차이가 상위 후보에서 크게 변별력을 만들지 못했다.
- partial position sizing은 손실 회피는 강했지만 시장 참여율을 너무 낮췄다.
- band 불확실성을 너무 강하게 반영하면 AAPL/MSFT처럼 상승 구간을 놓쳤다.

버려야 할 해석:

- band width 확장만으로 매도
- upper breach 또는 상단 돌파를 무조건 매도
- line 약화만으로 즉시 전량 청산
- 50% partial exposure를 과도하게 많이 쓰는 구조

## 14. 프론트 백테스트 화면 반영 여부

이번 CP 결과만으로는 **제품 기본 전략에 반영하지 않는다.**

이유:

- 생존 후보가 0개다.
- 기준 통과 티커 비율이 너무 낮다.
- 대표 티커에서 AAPL/MSFT/NVDA 성격이 너무 다르게 나왔다.
- 평균 지표는 좋아 보여도, 티커별 재현성이 약하다.

다만 다음 CP에서 검토할 만한 후보는 있다.

후보 이름:

- `Balance Entry Filter 후보`

룰 방향:

- line entry threshold: -0.2%
- line hold threshold: -1.4%
- band expansion은 신규 진입 제한
- line 약화 + lower/width 위험은 2일 confirm 후 청산

반영 조건:

- CP110에서 바로 UI 반영하지 말고, 먼저 티커별 실패 원인을 더 쪼갠다.
- 특히 AAPL/MSFT의 시장 참여율 0~11% 문제를 해결해야 한다.

## 15. 제품 문구 관점 결론

현재 결과만 보면 “AI 지표가 자동 매매 엔진으로 충분하다”고 말하면 안 된다.

더 정확한 제품 문구는 다음 쪽이다.

- 보수적 예측선은 방향 참고선이다.
- AI 밴드는 위험 범위와 불확실성 참고 지표다.
- 두 지표를 함께 읽으면 특정 구간의 위험 해석은 좋아질 수 있지만, 아직 자동 전략으로 일반화하기에는 부족하다.
- Lens는 매매 지시보다 리스크 해석 보조 도구로 설명하는 것이 맞다.

## 16. 남은 과제

- AAPL/MSFT처럼 상승을 놓치는 티커의 공통 원인 분석
- line entry threshold를 티커별 분포 기준으로 정규화할지 검토
- band width expansion을 절대 threshold가 아니라 티커별 regime으로 다루기
- partial position sizing의 과도한 현금화를 줄이기
- Balance 전략을 “자동 매매”가 아니라 “해석 레이어”로 화면에 보여주는 UX 검토

## 17. 검증 결과

실행한 검증:

- `python -m py_compile ai\cp109_line_band_balance_strategy_grid.py`
- `python ai\cp109_line_band_balance_strategy_grid.py`
- `docs/cp109_line_band_balance_strategy_grid_metrics.json` 생성 확인
- `docs/cp109_line_band_balance_strategy_grid_top_candidates.csv` 생성 확인

결과:

- py_compile 통과
- metrics JSON 생성
- top candidates CSV 생성
- rule count 720
- usable ticker 95
- survived count 0
- CPU-only 확인
- 프론트 수정 없음
- DB write 없음
- 모델 학습 없음
- inference 실행 없음
- Supabase price/indicator 대량 read 없음
- composite 사용 없음
