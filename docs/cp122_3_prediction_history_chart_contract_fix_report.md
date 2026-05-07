# CP122-3-P 예측 history/forecast 차트 series 계약 버그 수정 보고서

## 1. 목표

주식 보기 차트에서 AI 예측선과 AI 밴드가 과거 가격 구간에 섞이거나, rolling history와 latest future forecast가 같은 series처럼 이어져 톱니 형태로 보이는 문제를 줄였다.

이번 CP는 모델 값, prediction DB, inference 결과를 수정하지 않고 프론트 표시 직전의 차트 series 계약만 고쳤다.

## 2. 수정 파일

- `frontend/src/components/Chart.tsx`
- `frontend/src/components/StockView.tsx`

## 3. 핵심 변경

### Future forecast 분리

- latest prediction 1건의 `forecast_dates`와 series만 future forecast로 사용한다.
- `forecast_date`가 `model asof_date` 이하이면 future forecast에서 제외한다.
- `forecast_date`가 최신 가격 날짜 이하이면 future forecast에서 제외한다.
- 즉, 최신 캔들 이후 날짜만 future forecast series로 그린다.

### Rolling history 분리

- rolling prediction history는 `forecast_dates` 전체 h1~h5를 이어 그리지 않는다.
- 각 prediction row의 `asof_date`에 대표 horizon 1개만 찍는다.
- 대표 horizon은 h5를 우선 사용하고, 길이가 짧으면 마지막 available horizon을 사용한다.
- line history, upper band history, lower band history 모두 같은 기준을 적용했다.

### Band history 계약

- upper/lower band history도 각 row의 `asof_date`와 같은 대표 horizon 값을 사용한다.
- upper/lower가 서로 다른 날짜 기준으로 연결되지 않게 동일한 history 계약을 적용했다.

### Marker 정리

- 오늘 marker가 예측 시작선처럼 보이지 않도록 제거했다.
- 차트 marker는 `모델 기준일`만 표시한다.
- 우측 패널의 `모델 기준일`은 기존처럼 prediction의 실제 `asof_date`를 따른다.

### 정렬/필터 방어

- 모든 chart series는 `time` 오름차순으로 정렬한다.
- 중복 time은 마지막 값 하나만 남긴다.
- invalid date, NaN, null 값은 `setData` 전에 제거한다.
- 가격 candlestick, line, volume, future forecast, rolling history 모두 동일한 정렬 방어를 탄다.

## 4. 실제 데이터 확인

AAPL 1D 기준:

- 최신 가격일: `2026-05-04`
- line model 기준일: `2026-05-04`
- band model 기준일: `2026-05-04`
- latest forecast dates: `2026-05-05`부터 5거래일

따라서 future forecast는 최신 캔들 이후 구간에만 표시 가능한 상태다.

## 5. 브라우저 확인 결과

- AAPL 1D 화면 로드 확인
- `보수적 예측선`, `AI 위험 범위`, `모델 기준일` legend 표시 확인
- 1D → 1W → 1M → 1D 전환 확인
- browser console error/warn: 0건

## 6. 검증 명령

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과: 통과.

참고: 샌드박스 안에서는 Next.js child process spawn이 `EPERM`으로 막혀, 권한 밖에서 동일 명령을 재실행해 통과 확인했다.

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과: 통과.

초기 readiness는 백엔드가 내려가 있어 실패했고, 백엔드를 다시 띄운 뒤 재실행했다.

최종 readiness 주요 결과:

- backend health: OK
- CORS: OK
- frontend root: OK
- frontend static stylesheet: OK
- AAPL 1D prices: OK
- AAPL indicators: OK
- AAPL 1M prices: OK
- LM prediction/history/evaluation: OK
- BM prediction/history/evaluation: OK

## 7. 남은 리스크

- rolling history는 h5 대표값으로 줄였지만, 값 자체가 실제 가격과 민감하게 움직이면 여전히 다소 출렁일 수 있다.
- future forecast와 rolling history는 series로 분리됐지만, lightweight-charts 특성상 같은 가격 축에서 겹쳐 보일 수 있다.
- 제품적으로는 이후 tooltip/crosshair에서 `과거 예측`과 `최신 예측`을 더 명확히 분리하는 UX가 필요하다.

## 8. 금지사항 준수

- 모델 값 수정 없음
- prediction DB 수정 없음
- fake data 생성 없음
- inference 재실행 없음
- 전략 룰 변경 없음
- 백엔드 schema 변경 없음
