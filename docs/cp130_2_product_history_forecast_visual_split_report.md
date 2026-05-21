# CP130-2-P product history와 latest forecast 연결부 시각 분리 보고서

## 1. 목표

주식 보기 1D 차트에서 product rolling history와 latest future forecast가 하나의 끊기지 않은 예측선처럼 보이지 않도록 시각 위계를 정리했다.

데이터 계약은 변경하지 않았다. product history는 product-history API 응답 그대로 사용하고, latest future forecast는 기존 latest prediction API 응답 그대로 사용한다.

## 2. 변경 파일

- `frontend/src/components/Chart.tsx`
- `frontend/src/app/globals.css`

## 3. 구현 내용

### History와 Future Forecast 분리 유지

기존 계약대로 과거 예측 이력과 최신 미래 예측은 서로 다른 series로 렌더링한다.

- 과거 예측 이력
  - `line_history.value`
  - `band_history.lower`
  - `band_history.upper`
  - 각 asof_date의 대표 horizon 값만 사용
- 최신 5일 예측
  - latest prediction의 `forecast_dates`
  - latest prediction의 line/band series
  - 모델 기준일과 최신 가격일 이후 point만 표시

## 4. 시각 분리 방식

### 모델 기준일 marker

차트 marker 문구를 `모델 기준일`에서 `모델 기준일 경계`로 바꿨다.

CSS도 아래처럼 조정했다.

- marker 선을 2px 파란 dashed line으로 강화
- marker 오른쪽에 아주 옅은 blue gradient를 넣어 history/future 경계를 보이게 함
- label 색상을 파란 계열로 변경
- tooltip: `과거 예측 이력과 최신 5일 예측을 나누는 기준일`

### 과거 예측 이력

과거 이력은 낮은 강조로 조정했다.

- line history: `rgba(4, 120, 87, 0.38)`, `lineWidth=1`, dotted
- band history: `rgba(30, 64, 175, 0.34)`, `lineWidth=1`, dotted

### 최신 미래 예측

최신 forecast는 기존처럼 굵은 dashed line을 유지했다.

- 최신 보수적 예측선: 진한 청록 dashed
- 최신 AI 위험 범위: 진한 남색 dashed
- history보다 훨씬 높은 시각 강조를 유지

## 5. Legend/Tooltip 문구

legend를 짧게 정리했다.

- `최신 5일 예측`
  - title: `현재 모델 기준일 이후 예측`
- `최신 AI 위험 범위`
  - title: `현재 모델 기준일 이후 AI 위험 범위`
- `과거 예측 이력`
  - title: `과거 각 날짜에서 모델이 본 5일 뒤 기준값`
- `모델 기준일 경계`

화면에 긴 설명 문단은 추가하지 않았다.

## 6. 검증 결과

실행 명령:

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 샌드박스 내부 첫 실행은 `spawn EPERM`으로 실패
- 권한 밖 재실행에서 build 통과

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과:

- backend health OK
- CORS OK
- frontend 200 OK
- frontend-static CSS 200 OK
- AAPL 1D prices OK
- indicators OK
- 1M prices OK
- LM/BM run, prediction, evaluation, history OK

브라우저 확인:

- AAPL 1D
  - 최신 5일 예측 legend 표시
  - 최신 AI 위험 범위 legend 표시
  - 과거 예측 이력 legend 표시
  - 모델 기준일 경계 표시
- MSFT 1D
  - 동일 항목 모두 표시 확인
- NVDA 1D
  - 동일 항목 모두 표시 확인
- 1W
  - product history/forecast legend 미표시
  - 주간 예측 준비 중 안내 유지
- 1M
  - 가격 전용 안내 유지
  - product history/forecast legend 미표시
- browser console error/warn 0건

## 7. 금지 사항 준수

- 데이터 값 수정 없음
- product history 계약 변경 없음
- inference 재실행 없음
- DB write 없음
- 모델 수정 없음
- fake data 없음

## 8. 남은 아쉬운 점

- canvas 내부 선 자체의 연결감을 자동 픽셀 검사로 판정하지는 않았다.
- 이번 CP는 시각 강조와 legend/marker 정리만 수행했다.
- 더 섬세한 구간 분리는 이후에 hover tooltip 또는 crosshair 동기화가 들어갈 때 같이 다듬는 것이 좋다.
