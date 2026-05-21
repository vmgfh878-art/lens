# CP106-BT Band Risk 100티커 룰 탐색 보고서

작성 시각: 2026-05-04 18:34 KST

## 1. 요약

CP106-BT는 프론트 수정이 아니라 Band Risk 계열 전략의 백테스트 실험 runner와 산출물 작성 작업이다.

핵심 질문은 다음이었다.

- AI 밴드 하단과 밴드 폭만으로 실제 위험 구간을 줄일 수 있는가?
- 여러 티커 평균에서 MDD를 낮추고 손실 회피율을 올릴 수 있는가?
- Buy & Hold 수익률을 과도하게 희생하지 않는 밴드 룰이 있는가?

결론부터 말하면, 이번 grid에서는 **holdout 기준 생존 후보가 없었다.**

다만 기존 Band Risk v1처럼 사실상 Buy & Hold에 가까운 룰보다는, 더 민감한 밴드 룰이 MDD를 줄이는 방향은 확인했다. 문제는 손실 회피율과 수익률 방어율이 생존 기준에 동시에 도달하지 못했다는 점이다.

## 2. 산출물

- `ai/cp106_band_risk_strategy_grid.py`
- `docs/cp106_band_risk_strategy_grid_metrics.json`
- `docs/cp106_band_risk_strategy_grid_top_candidates.csv`
- `docs/cp106_band_risk_strategy_grid_report.md`

## 3. 사용 데이터와 금지 사항 준수

사용 데이터:

- 가격: `data/parquet/price_data_yfinance.parquet`
- 보조지표: `data/parquet/indicators_yfinance_1D.parquet`
- universe: `data/parquet/stock_info.parquet`
- BM prediction history: `cnn_lstm-1D-d0c780dee5e8` 기준 read-only API

금지 사항 준수:

- GPU 사용 없음
- 모델 학습 없음
- inference 실행 없음
- DB write 없음
- Supabase `price_data` / `indicators` 대량 read 없음
- fake data 생성 없음
- 프론트 수정 없음

주의:

- 가격과 지표는 local parquet에서 읽었다.
- prediction history는 제품 BM run의 저장 결과를 `/api/v1/stocks/{ticker}/predictions/history` read-only API로 조회했다.
- 이 API는 thin prediction row 조회이며, Supabase 가격/지표 대량 read는 하지 않았다.

## 4. 사용 티커 수와 제외 티커

local yfinance universe:

- 전체 universe: 100티커
- 사용 가능 티커: 95티커
- 제외 티커: 5티커

제외 사유:

| 티커 | 사유 |
|---|---|
| LMT | BM prediction history 없음 |
| MS | BM prediction history 없음 |
| QQQ | BM prediction history 없음 |
| SPY | BM prediction history 없음 |
| T | BM prediction history 없음 |

## 5. 기간 split

요청 기준 split:

- 탐색 구간: 2025-06-18 ~ 2026-03-31
- holdout 구간: 2026-04-01 ~ 2026-05-01

실제 BM prediction range:

- 최소 asof: 2025-06-16
- 최대 asof: 2026-05-01

중요한 제한:

- 탐색 구간은 95티커 기준으로 평가 가능했다.
- holdout 구간은 가격 데이터는 100티커 모두 2026-05-01까지 있었지만, 저장된 BM prediction history가 가격과 충분히 맞는 티커가 3개뿐이었다.
- 따라서 holdout 결과는 생존 판단 기준으로 계산했지만, 표본 수가 작아 제품 결론으로 쓰기에는 부족하다.

보고서와 JSON에는 이 제한을 그대로 남겼다.

## 6. 실험한 파라미터 조합 수

전체 grid:

- lower_risk_threshold: 4개
- width_risk_threshold: 4개
- width_expansion_ratio: 4개
- risk_confirm_days: 3개
- reentry_confirm_days: 3개
- trend_filter: 3개
- rsi_filter: 3개

총 조합:

```text
4 * 4 * 4 * 3 * 3 * 3 * 3 = 5,184개
```

실제 비교한 룰 후보 수:

- 5,184개

## 7. 룰 해석 방식

Band Risk 계열만 실험했다.

핵심 입력:

- AI 밴드 하단 위험도:
  - `min(lower_band_series) / close - 1`
- AI 밴드 폭:
  - `(max(upper_band_series) - min(lower_band_series)) / close`
- 밴드 폭 확장:
  - 현재 밴드 폭이 최근 20개 signal 기준 median 대비 얼마나 커졌는지

위험 판단:

- 밴드 하단이 threshold보다 깊으면 위험
- 밴드 폭이 threshold보다 넓으면 위험
- 밴드 폭이 최근 기준 대비 expansion ratio 이상이면 위험

포지션:

- 단일 티커 100%
- 현금 100%
- 부분 비중 없음
- 포트폴리오 없음

거래 비용:

- 10bp

## 8. 큰 하락 구간 정의

CP105 정의를 유지했다.

큰 하락 기준:

- 일간 수익률 하위 20%
- 일간 수익률 -2%
- 위 둘 중 더 엄격한 값

```text
large_loss_threshold = min(-2%, daily_return_20_percentile)
```

손실 회피율:

- 큰 하락이 나온 날 중 전략이 현금 대기 상태였던 비율

## 9. 평가 지표 정의

각 룰 후보에 대해 탐색 구간과 holdout 구간을 모두 계산했다.

필수 집계:

- 평가 티커 수
- 평균 전략 수익률
- 평균 Buy & Hold 수익률
- 평균 Buy & Hold 대비 수익률 비율
- 평균 초과 수익률
- 평균 전략 MDD
- 평균 Buy & Hold MDD
- 평균 MDD 개선폭
- 평균 Sharpe
- 평균 Sortino
- 평균 손실 회피율
- 평균 시장 참여율
- 평균 거래 횟수
- 평균 보유 기간
- 기준 통과 티커 비율
- 강한 상승 티커에서의 수익률 방어율
- 하락/횡보 티커에서의 MDD 개선폭

수익률 비율 처리:

- Buy & Hold 수익률이 양수이면 `전략 수익률 / Buy & Hold 수익률`로 계산했다.
- Buy & Hold 수익률이 음수이면 전략이 Buy & Hold보다 낫거나 같은 경우 방어 성공으로 1.0, 더 나쁘면 0.0으로 처리했다.

## 10. 생존 기준

holdout 기준으로 판단했다.

생존 조건:

- 평균 MDD 개선폭 > 0
- 평균 손실 회피율 >= 0.50
- 평균 Buy & Hold 대비 수익률 비율 >= 0.70
- 평균 시장 참여율 0.45 ~ 0.85
- 기준 통과 티커 비율 >= 0.50
- 거래 횟수가 과도하지 않을 것
- 시장 참여율 0.95 이상이면 Buy & Hold 유사 후보로 탈락
- 시장 참여율 0.30 이하이면 너무 소극적인 후보로 탈락

결과:

- 생존 후보 수: 0개

## 11. 상위 10개 후보

아래 표는 holdout score 기준 상위 10개다.
단, holdout 평가 티커가 3개뿐이므로 순위는 제한적으로 해석해야 한다.

| 순위 | lower | width | expansion | risk confirm | reentry confirm | trend | RSI | holdout 전략 수익률 | holdout B&H 수익률 | 수익률 비율 | MDD 개선폭 | 손실 회피율 | 시장 참여율 | 생존 |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | -0.05 | 0.10 | 1.10 | 1 | 1 | off | off | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 2 | -0.05 | 0.10 | 1.10 | 1 | 1 | off | RSI>75 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 3 | -0.05 | 0.10 | 1.10 | 1 | 1 | off | RSI>80 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 4 | -0.05 | 0.10 | 1.10 | 2 | 1 | off | off | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 5 | -0.05 | 0.10 | 1.10 | 2 | 1 | off | RSI>75 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 6 | -0.05 | 0.10 | 1.10 | 2 | 1 | off | RSI>80 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 7 | -0.05 | 0.10 | 1.10 | 3 | 1 | off | off | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 8 | -0.05 | 0.10 | 1.10 | 3 | 1 | off | RSI>75 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 9 | -0.05 | 0.10 | 1.10 | 3 | 1 | off | RSI>80 진입 회피 | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |
| 10 | -0.05 | 0.10 | 1.25 | 1 | 1 | off | off | 3.36% | 4.27% | 0.65 | 3.32%p | 33.33% | 66.67% | 아니오 |

상위권이 거의 같은 이유:

- holdout 표본이 3티커로 작다.
- 짧은 holdout에서는 trend/RSI 필터와 일부 confirm day 차이가 실제 포지션에 영향을 거의 주지 않았다.

## 12. 기존 Band Risk v1 대비 개선 여부

기존 Band Risk v1 유사 baseline:

- lower risk: -0.10
- width risk: 0.25
- expansion risk: 사실상 off
- trend filter: off
- RSI filter: off

탐색 구간 baseline:

- 평균 전략 수익률: 9.18%
- 평균 Buy & Hold 수익률: 10.24%
- 평균 수익률 비율: 0.56
- 평균 MDD 개선폭: -0.17%p
- 평균 손실 회피율: 0.44%
- 평균 시장 참여율: 99.60%

해석:

- 기존 baseline은 거의 Buy & Hold처럼 움직였다.
- 손실 회피율이 0.44%로 Band Risk라는 이름값이 약했다.
- MDD도 평균적으로 Buy & Hold보다 약간 악화됐다.

상위 후보 탐색 구간 예시:

- 규칙: lower -0.05, width 0.10, expansion 1.10, risk confirm 1, reentry confirm 1
- 평균 전략 수익률: 3.24%
- 평균 Buy & Hold 수익률: 10.24%
- 평균 수익률 비율: 0.76
- 평균 MDD 개선폭: +4.64%p
- 평균 손실 회피율: 38.11%
- 평균 시장 참여율: 64.99%

개선된 점:

- 시장 참여율이 99.60%에서 약 65%로 내려와 실제 위험 회피 동작이 생겼다.
- MDD 개선폭이 -0.17%p에서 +4.64%p로 좋아졌다.
- 손실 회피율도 0.44%에서 38.11%로 개선됐다.

부족한 점:

- 손실 회피율이 생존 기준 50%에 못 미쳤다.
- 기준 통과 티커 비율이 7.37%로 낮았다.
- 수익률을 꽤 희생했다.

## 13. 가장 좋은 후보의 규칙 설명

상위 후보 규칙:

- AI 밴드 하단이 현재가 대비 -5% 이하로 깊어지면 위험으로 본다.
- AI 밴드 폭이 현재가 대비 10% 이상이면 위험으로 본다.
- 밴드 폭이 최근 기준 대비 1.10배 이상 확대되면 위험으로 본다.
- 위험은 1일 확인으로 바로 반영한다.
- 위험 완화도 1일 확인으로 재진입한다.
- trend filter는 끈다.
- RSI filter도 끈다.

해석:

- 기존 Band Risk v1보다 훨씬 민감하다.
- Buy & Hold처럼 계속 들고 있지는 않는다.
- MDD는 줄일 수 있지만 손실 회피율과 수익률 방어가 아직 생존 기준에 못 미친다.

## 14. 실패 원인

이번 grid에서 생존 후보가 없었던 이유는 크게 세 가지다.

1. AI 밴드 하단/폭만으로는 손실 회피율 50%를 안정적으로 넘기기 어려웠다.

탐색 구간에서 더 민감한 룰은 손실 회피율을 높였지만 평균 38% 수준에 그쳤다. 위험 구간을 일부 줄이긴 했지만 Band Risk 단독 지표로 충분한 회피력을 확보하지 못했다.

2. 수익률과 MDD 사이 trade-off가 컸다.

MDD를 줄이는 룰은 시장 참여율을 낮추면서 수익률을 희생했다. 특히 강한 상승 티커에서 수익률 방어가 약해졌다.

3. holdout 표본이 너무 작았다.

holdout 구간에서 유효 티커가 3개뿐이어서 100티커 일반화 판단에는 부족하다. 이번 holdout에서는 생존 후보가 없었지만, 이 결과는 “제품 후보 탈락”이라기보다 “저장 prediction history를 넓혀 다시 검증 필요”에 가깝다.

## 15. 다음 Line Trend / Balance 실험에 넘길 교훈

Band Risk 단독으로는 위험 회피 성격을 만들 수 있지만, 충분하지 않다.

다음 실험에 넘길 교훈:

- Band Risk는 MDD 완화 필터로는 쓸 수 있지만 단독 전략으로는 수익 추종력이 약해질 수 있다.
- 손실 회피율을 높이려면 AI 밴드만 보지 말고 line 약화, 추세 이탈, 실현 변동성 등을 같이 봐야 한다.
- Balance 전략에서는 band 위험이 커질 때 바로 현금화하기보다, line이 동시에 약해지는지 확인하는 조건이 필요하다.
- Line Trend 실험에서는 상승 유지 조건을 완화하되, Band Risk에서 확인한 width/하단 위험을 exit 보조로 넣는 것이 유효해 보인다.
- holdout 평가를 제대로 하려면 100티커 BM prediction history를 2026-04 이후로 더 넓게 저장해야 한다.

## 16. 검증

실행한 명령:

```powershell
.\.venv\Scripts\python.exe -m py_compile ai\cp106_band_risk_strategy_grid.py
```

결과:

- 통과

```powershell
.\.venv\Scripts\python.exe ai\cp106_band_risk_strategy_grid.py
```

결과:

- 5,184개 룰 후보 계산 완료
- metrics JSON 생성 완료
- top candidates CSV 생성 완료
- 실행 시간: 약 3,204.7초

주의:

- 전체 실행은 산출물 생성까지 완료했지만, 셸 wrapper가 긴 실행을 timeout으로 표시했다.
- 이후 JSON parse, CSV 확인, 산출물 timestamp 확인을 별도로 수행했다.

```powershell
.\.venv\Scripts\python.exe -m json.tool docs\cp106_band_risk_strategy_grid_metrics.json
```

결과:

- JSON parse 통과

프로세스 확인:

- runner 종료 후 남은 `python/pythonw` 실험 프로세스 없음
- 남은 Python 프로세스는 기존 백엔드 서버 계열만 확인됨
- `pythonw` 없음

서버 상태:

- 백엔드 `127.0.0.1:8000` 실행 중
- 프론트 `127.0.0.1:3000` 실행 중

## 17. 남은 한계

- holdout 유효 티커가 3개뿐이라 100티커 holdout 결론으로는 부족하다.
- full grid가 단일 프로세스 순차 계산이라 약 53분 걸렸다.
- top 후보가 holdout 3티커 기준으로 동률이 많아 세밀한 ranking 의미는 약하다.
- 다음에는 prediction history를 local artifact로 export해서 API 반복 조회 시간을 줄이는 편이 좋다.
- Band Risk 단독 전략보다 Balance 전략의 exit filter로 쓰는 편이 더 자연스러워 보인다.
