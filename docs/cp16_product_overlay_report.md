# CP16-P 주식 보기 AI Overlay 보고서

## 1. 변경 요약

- 주식 보기 차트에 실제 AI overlay를 연결했다.
- 1D/1W에서는 최신 완료 PatchTST run을 먼저 찾고, 해당 run의 prediction만 조회한다.
- 1M은 계속 가격 전용으로 유지한다.
- AI 밴드 토글과 보수적 예측선 토글이 실제 차트 표시 여부에 반영된다.
- 주식 보기와 백테스트 화면의 주요 사용자 문구를 한국어로 정리했다.

## 2. overlay에 사용한 prediction 필드명

- 날짜 축: `forecast_dates`
- 보수적 예측선: `line_series`
- 보수적 예측선 보조 fallback: `conservative_series`
- AI 밴드 상단: `upper_band_series`
- AI 밴드 하단: `lower_band_series`
- 밴드 분위수 정보: `band_quantile_low`, `band_quantile_high`

차트에는 `forecast_dates`와 각 series 길이가 정확히 맞을 때만 표시한다. 길이가 다르거나 값이 유효하지 않으면 임의로 보정하지 않고 overlay를 생략한다.

## 3. 1D / 1W / 1M 동작 결과

- `1D`: `fetchAiRuns(status=completed, modelName=patchtst, timeframe=1D)`로 최신 완료 run을 고른 뒤 `predictions/latest?run_id=...`를 조회한다.
- `1W`: 1D와 같은 방식으로 최신 완료 run의 주봉 prediction을 조회한다.
- `1M`: 가격 API만 호출한다. prediction API와 evaluation API는 호출하지 않고, AI 레이어 토글은 비활성 처리한다.

failed_nan run은 주식 보기 overlay에 섞이지 않는다. 프론트가 `completed` run만 선택하고, run_id prediction API도 백엔드에서 completed가 아닌 run을 거부한다.

## 4. prediction 없음 상태 결과

- 가격 데이터는 정상 표시한다.
- AI 상태는 `저장된 AI 결과 없음`으로 표시한다.
- 차트 overlay는 그리지 않는다.
- 커버리지, 평균 밴드 폭, 보수적 예측선 요약 값은 `-`로 표시한다.

## 5. AI 밴드 토글 결과

- ON: `upper_band_series`, `lower_band_series`를 보라색 점선 2개로 표시한다.
- OFF: 상단/하단 밴드 라인을 차트에서 제거한다.
- 1M: 토글은 비활성 상태로 표시한다.

## 6. 보수적 예측선 토글 결과

- ON: `line_series`를 검은색 실선으로 표시한다.
- OFF: 보수적 예측선을 차트에서 제거한다.
- 1M: 토글은 비활성 상태로 표시한다.

## 7. 한글화한 주요 라벨

- `Chart` -> `차트`
- `Results` -> `결과`
- `Runs` -> `학습 이력`
- `return` -> `수익률`
- `sharpe` -> `샤프`
- `mdd` -> `최대낙폭`
- `win_rate` -> `승률`
- `turnover` -> `회전율`
- `trades` -> `거래 수`
- `equity curve` -> `수익 곡선`
- `drawdown` -> `낙폭 차트`
- `fee bps` -> `수수료(bp)`

주식 보기 화면에는 target type, raw, volatility-normalized 용어를 노출하지 않는다.

## 8. npm run build 결과

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 막혔다. 승인된 실행에서는 정상 통과했다.

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
Route (app) / 78.3 kB, First Load JS 165 kB
```

dev 서버 확인:

```text
http://127.0.0.1:3000
HTTP/1.1 200 OK
```

## 9. band fill 구현 여부

band fill은 이번 CP에서 구현하지 않았다.

`lightweight-charts` 기본 series만으로는 두 선 사이 영역을 안정적으로 채우는 구현이 부담되고, 잘못 처리하면 가격 스케일과 hover 동작을 흔들 수 있다. 이번 단계에서는 안전하게 `upper_band_series`와 `lower_band_series`를 상하단 점선으로 표시했다. 실제 fill은 CP16 이후 별도 polygon/canvas overlay 또는 커스텀 series 방식으로 확장하는 것이 좋다.
