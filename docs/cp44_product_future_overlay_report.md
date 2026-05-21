CP44-P는 예측선을 과거 가격 구간에 겹치지 않고, `forecast_dates` 기준 미래 구간에 분리해 표시하는 CP다.

# 1. 변경 요약

- 가격 차트는 예측 overlay가 있을 때 첫 `forecast_dates` 이전 실제 가격까지만 렌더링한다.
- `prediction.forecast_dates`를 차트 x축 오른쪽에 whitespace 구간으로 추가하고, 그 구간에 예측선을 표시한다.
- AI 밴드 상단/하단과 보수적 예측은 미래 구간에서 점선 스타일로 표시한다.
- 첫 forecast date 위치에 `예측 시작` 기준선을 표시한다.
- `forecast_dates`와 시리즈 길이가 맞지 않으면 overlay를 숨기고 조용한 경고 문구를 표시하도록 방어했다.
- 하단 RSI/MACD 보조지표는 가격 차트와 같은 timeline을 사용하고, 미래 예측 구간에는 값을 채우지 않는다.

# 2. 변경 파일

- `frontend/src/components/Chart.tsx`
- `frontend/src/components/IndicatorPanel.tsx`
- `frontend/src/components/StockView.tsx`
- `frontend/src/app/globals.css`

# 3. 실제 API 응답 확인

AAPL 데모 prediction:

- `run_id`: `patchtst-1D-239b58ab90f0`
- `asof_date`: `2026-03-30`
- `forecast_dates`: `2026-03-31`, `2026-04-01`, `2026-04-02`, `2026-04-06`, `2026-04-07`
- `line_series`: 5개
- `upper_band_series`: 5개
- `lower_band_series`: 5개
- `conservative_series`: 5개

현재 가격 API는 `2026-04-28`까지 실제 가격을 내려준다.
이 상태에서 전체 가격을 그대로 그리면 예측이 과거 구간에 보이므로, 차트 overlay가 활성화된 경우 가격 차트는 첫 forecast date 이전까지만 표시한다.

# 4. 렌더링 방식

- 실제 가격: 첫 forecast date 이전 실제 가격 구간.
- 미래 예측: `forecast_dates` 5개 지점.
- AI 밴드: upper/lower 두 점선.
- 보수적 예측: 점선 line series.
- 예측 시작 기준선: 첫 forecast date 좌표에 세로 dashed marker.
- 1M: 기존대로 price-only이며 AI overlay는 비활성화.

밴드 fill은 이번 CP에서 추가하지 않았다. 현재는 upper/lower line으로 밴드 범위를 보이게 처리했다.

# 5. 보조지표 x축 동기화

- `StockView`에서 가격 차트용 timeline을 만들고 `IndicatorPanel`에 같은 timeline을 전달한다.
- RSI/MACD 값은 실제 과거 날짜에만 표시한다.
- 미래 forecast date 구간은 축만 확장하고 보조지표 값은 비운다.
- 일부 날짜에 지표 값이 없으면 SVG path를 끊어 보간하지 않는다.
- 가격 차트 visible range 변경 이벤트를 받아 하단 보조지표 timeline도 같은 범위로 갱신한다.

# 6. 브라우저 확인

`http://127.0.0.1:3000`에서 AAPL 1D 확인:

- 가격 차트 표시 확인.
- 마지막 실제 가격 구간 오른쪽에 미래 forecast 5개 지점 표시 확인.
- AI 밴드 상단/하단 표시 확인.
- 보수적 예측선 표시 확인.
- `예측 시작` 기준선 표시 확인.
- 백엔드 연결 오류 배너 없음.
- RSI/MACD 보조지표가 가격 차트와 같은 종료 축 `04-07`로 정렬되는 것 확인.
- 1M 전환 시 가격 전용 상태와 AI 토글 비활성화 확인.

# 7. 검증 결과

- `cd C:\Users\user\lens\frontend; npm run build`
  - 샌드박스에서는 Next worker 생성이 `spawn EPERM`으로 막혀 권한 승인 경로로 실행했다.
  - 최종 빌드 성공.
- `cd C:\Users\user\lens; powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - health, CORS, frontend, 1D prices, indicators, 1M prices, completed runs, demo prediction, evaluation 통과.
  - stock search는 기존처럼 503.
  - latest run에는 AAPL prediction이 없어 demo run fallback 경고 유지.
  - backtest row 없음 경고 유지.

# 8. 남은 리스크

- 데모 prediction의 band 값 범위가 가격 대비 매우 넓어 가격 차트가 다소 압축된다. 이번 CP에서는 prediction 값 생성 계약을 수정하지 않았다.
- latest completed run에 AAPL prediction이 없어 fallback run을 사용한다.
- stock search 503과 backtest row 없음은 이번 CP 범위 밖 기존 리스크다.
