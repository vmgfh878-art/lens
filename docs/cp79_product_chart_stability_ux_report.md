CP79-P 차트 안정화 및 시각 UX 정리 보고서

## 1. 발견된 버그

- 1W 버튼 클릭 시 `lightweight-charts` 런타임 오류가 발생했다.
- 오류 메시지:
  - `Assertion failed: data must be asc ordered by time`
- 오류 위치:
  - `frontend/src/components/Chart.tsx`
  - `candleSeries.setData(...)`

## 2. 원인

- `lightweight-charts`는 모든 `setData` 입력이 time 오름차순이어야 한다.
- 기존 구현은 API 응답과 forecast/rolling history를 그대로 합쳐 넘겼고, timeframe 전환 중 정렬·중복 제거·invalid 제거가 일관되게 적용되지 않았다.
- 가격 API는 기본 1년치만 내려주며, `start=2015` 단일 요청은 Supabase 반환 제한 때문에 과거 일부 구간만 받는 문제가 있었다. 프론트에서 연도별 조회를 합치는 방식으로 전체 히스토리 표시를 보강했다.

## 3. 수정한 파일별 상세 변경

### `frontend/src/components/Chart.tsx`

- `setData` 직전 공통 방어 로직을 추가했다.
  - time 오름차순 정렬
  - 같은 time 중복 시 마지막 값 유지
  - invalid date 제거
  - NaN/null 가격·예측 값 제거
- 적용 대상:
  - 캔들 가격 series
  - 라인 가격 series
  - 최신 예측선
  - 최신 AI 밴드 upper/lower
  - rolling 예측선 이력
  - rolling 밴드 이력
  - 보수적 기준선
- 초기 표시 범위는 최근 구간으로 잡되, 전체 data는 chart series에 그대로 넣도록 변경했다.
  - 1D: 최근 260개 내외
  - 1W: 최근 156개 내외
  - 1M: 최근 84개 내외
- 예측선, AI 위험 범위, 보수적 기준선을 별도 series로 분리할 수 있게 정리했다.
- `conservative_series`가 `line_series`와 완전히 같으면 중복 선을 그리지 않도록 했다.

### `frontend/src/components/StockView.tsx`

- 가격 데이터를 연도별 read-only API 호출로 가져와 합친다.
  - 2015년부터 현재 연도까지 연도별 `start/end` window로 조회
  - 프론트 표시 직전 정렬·중복 제거
- forecast 시작일 기준으로 가격 데이터를 잘라내던 구조를 제거했다.
- indicator와 AI 밴드 폭 포인트도 date 기준 정렬·중복 제거를 적용했다.
- 1W/1M 안내 문구는 우측 notice 한 곳에서만 보이도록 유지했다.

### `frontend/src/components/IndicatorPanel.tsx`

- 보조지표 SVG series도 표시 직전에 date 오름차순 정렬·중복 제거·invalid 제거를 적용했다.
- RSI/MACD/AI 밴드 폭 등 하단 보조지표가 가격 chart visible range와 같은 timeline을 쓰도록 기존 계약을 유지했다.
- 패널 높이를 112px에서 96px 기준으로 줄여 밀도를 높였다.

### `frontend/src/app/globals.css`

- AI 밴드 legend 색을 더 진한 파란 계열로 조정했다.
- 예측선 legend 색을 별도로 추가했다.
- 보수적 기준선은 진회색으로 낮은 강조를 유지했다.
- 보조지표 패널 padding/gap/min-height를 줄여 한 화면에서 더 많이 보이게 했다.

## 4. 1D/1W/1M 전환 검증 결과

- 브라우저에서 아래 순서를 확인했다.
  - 1D → 1W → 1M → 1D → 1W → 1M → 1D
- 결과:
  - 1W 런타임 오류 재현 안 됨
  - 1M 가격 전용 화면 정상
  - 빨간 백엔드 오류 배너 없음
  - 브라우저 console error/warn 0건

## 5. 중복 문구 제거 결과

- `주간 AI 예측은 준비 중입니다.` 표시 count: 1
- `월간 화면은 현재 가격 전용입니다.` 표시 count: 1
- `rolling prediction history` 같은 구현 설명 문구는 화면에서 제거했다.

## 6. 차트 데이터 범위 확인 결과

- 가격 API를 연도별 window로 합쳤을 때 AAPL 기준 프론트가 구성 가능한 데이터 범위:
  - 1D: 2,849개, 2015-01-02 ~ 2026-05-01
  - 1W: 592개, 2015-01-02 ~ 2026-05-01
  - 1M: 137개, 2015-01-31 ~ 2026-05-31
- 차트에는 전체 히스토리를 넣고, 초기 viewport만 최근 구간으로 제한했다.
- forecast 5일 구간 때문에 과거 가격 데이터가 잘리는 구조는 제거했다.

## 7. AI 밴드/보수적 예측선 시각 구분 변경

- 최신 AI 밴드 upper/lower를 기존보다 진한 파란 점선으로 표시한다.
- 과거 밴드 이력은 옅은 파란 선으로 유지한다.
- 예측선은 초록 점선으로 표시한다.
- 보수적 기준선은 진회색 dotted line으로 낮은 강조를 유지한다.
- 현재 AAPL 1D의 `line_series`와 `conservative_series`는 값이 동일해서 보수적 기준선은 중복 표시하지 않는다.
- band fill은 구현하지 않았다. `lightweight-charts` 기본 line series만으로 upper/lower 사이 면을 안정적으로 채우기 어렵고, 이번 CP는 안정화 우선이므로 선 색/두께/투명도로 구분했다.

## 8. 보조지표 간격 변경

- 보조지표 패널 전체 padding을 줄였다.
- 카드 gap과 min-height를 줄였다.
- SVG 높이를 96px로 줄였다.
- 가격 차트와 보조지표 사이 간격을 줄여 한 화면 내 연결감을 조금 높였다.

## 9. 남은 아쉬운 점

- 보조지표 전체 히스토리는 현재 indicator API의 `limit=1000` 범위 안에서만 표시된다.
- AI 밴드 fill은 아직 없다.
- 보수적 기준선은 현재 저장 데이터가 예측선과 같으면 별도로 보이지 않는다.
- 브라우저 자동화에서 실제 pan 동작을 좌표 drag로 검증하려 했으나 in-app browser 좌표 변환 제한으로 수행하지 못했다. 대신 전체 히스토리 data를 chart에 넣고 초기 visible range만 조정하는 코드와 전환 오류 0건을 확인했다.

## 10. 다음 CP로 넘길 항목

- 보조지표 전체 히스토리 API 또는 pagination.
- AI 밴드 fill/영역 표시 방식 설계.
- 차트 pan/zoom 상태를 더 명확히 확인할 수 있는 디버그 없는 QA 도구.
- 보수적 기준선이 예측선과 다를 때의 사용자 설명 문구 추가.

## 검증 명령

- `npm run build` 통과
- `powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1` 통과
- 브라우저 확인:
  - AAPL 1D
  - AAPL 1W
  - AAPL 1M
  - 1D → 1W → 1M → 1D 반복
  - console error/warn 0건

## 수정하지 않은 것

- 모델 학습, inference 실행, DB 쓰기, fake data 생성은 하지 않았다.
- API/DB schema는 수정하지 않았다.
- 백엔드 API 코드는 수정하지 않았다.
- 다크모드, 백테스트 전략, 모델 학습 화면 대개편은 하지 않았다.
