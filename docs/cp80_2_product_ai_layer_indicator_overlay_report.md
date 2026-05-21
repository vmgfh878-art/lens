CP80-2-P는 AI 예측선과 AI 밴드를 제품 핵심 레이어로 더 선명하게 보이게 하고, 기본 차트 보조 정보의 밀도를 높인 프론트 안정화 CP다.

## 1. 중복 설명 제거 목록

- 기본 예측 출처 패널에서 아래 설명 블록을 제거했다.
  - `예측선: 향후 수익 방향을 참고하기 위한 AI 예측선입니다.`
  - `AI 밴드: 예상 변동 범위를 보여주는 리스크 참고 지표입니다.`
  - `보수적 예측선: 하방 위험을 더 조심스럽게 보기 위한 기준선입니다.`
- 기본 패널에는 모델명, 모델 기준일, 예측 기간만 남겼다.
- `예측선 상태`, `밴드 상태`는 기본 화면에서 제거하고 `상세 정보` 안으로 내렸다.
- `표시 중`, `확인 중`, `없음`처럼 토글과 중복되는 큰 상태 문구는 제거하고 패널 제목을 `모델 정보`로 바꿨다.
- 정상 연결 상태에서는 하단 notice를 숨기고, 예측선 범위 이탈이나 준비/없음 상태일 때만 표시하도록 줄였다.

## 2. AI 레이어 색/두께 변경 상세

- 최신 예측선
  - 색상: `#006b57`
  - 두께: 4
  - 역할: 가격선과 동급으로 보이는 핵심 예측 레이어
- 과거 예측선
  - 색상: `rgba(4, 120, 87, 0.62)`
  - 두께: 2
  - 역할: 희미한 장식이 아니라 rolling prediction history로 읽히게 강화
- 최신 AI 밴드 upper/lower
  - 색상: `#172554`
  - 두께: 4
  - 역할: 모델 기반 위험 범위를 가장 또렷하게 표시
- 과거 AI 밴드 history
  - 색상: `rgba(30, 64, 175, 0.58)`
  - 두께: 2
  - 역할: 과거 밴드 이력의 신뢰감을 높임
- 보수적 기준선
  - 색상: `rgba(51, 65, 85, 0.82)`
  - 두께: 2
  - 스타일: dotted 유지

## 3. 가격선과 AI 레이어 위계 조정

- 라인 차트 가격선은 `#1f2937`, 두께 2로 유지했다.
- AI 예측선과 AI 밴드는 두께 4로 올려 가격선보다 약하지 않게 조정했다.
- 캔들 차트의 가격 가독성은 유지하되, forecast 구간의 AI 레이어가 한눈에 들어오도록 latest AI 선을 더 강하게 만들었다.
- legend 색상도 예측선, AI 위험 범위, 보수적 기준선, 거래량이 서로 구분되도록 맞췄다.

## 4. 예측범위/예측 출처 패널 축소 내용

- 패널 상단을 `예측 레이어 / 모델 정보`로 정리했다.
- 기본 표시 항목:
  - 예측선 모델 v1
  - AI 밴드 모델 v1
  - 모델 기준일
  - 예측 기간
- 상세 정보로 내린 항목:
  - 표시 기준
  - 예측선 상태
  - 밴드 상태
  - 예측선 실행 ID
  - 밴드 실행 ID
- 설명 문장 반복을 줄여 차트와 토글이 먼저 읽히게 했다.

## 5. 거래량 bar 추가 여부

- 추가했다.
- `frontend/src/components/Chart.tsx`에서 `PriceBar.volume` 값을 사용한다.
- volume 값이 없거나 0 이하이면 조용히 숨긴다.
- lightweight-charts `HistogramSeries`를 사용했고, 별도 `volume` price scale을 하단 18% 영역에 배치했다.
- 색상은 중립 회색을 기본으로 하되, 상승/하락일에 따라 아주 낮은 opacity의 초록/빨강을 적용했다.
- 가격 차트와 같은 x축을 사용한다.

## 6. Bollinger overlay 추가 여부

- 추가하지 않았다.
- 현재 read-only indicator API와 타입에는 `bb_position`만 있다.
- 가격 차트 overlay에 필요한 Bollinger upper/lower 가격선 필드는 없다.
- 이번 CP 원칙상 프론트에서 임의 계산하지 않았고, fake data도 만들지 않았다.
- 필요 데이터:
  - `bb_upper`
  - `bb_middle`
  - `bb_lower`
  - 또는 가격 overlay가 가능한 동등한 Bollinger band time series

## 7. ATR/AI band width 설명 정리

- ATR 설명을 `최근 가격 변동 폭을 보는 전통 변동성 지표`로 줄였다.
- AI 밴드 폭 설명을 `AI가 보는 예상 변동 범위의 넓이`로 줄였다.
- 두 설명은 보조지표 선택 영역에만 남겨 차트/예측 패널 설명과 중복되지 않게 했다.

## 8. 구현하지 못한 항목과 이유

- Bollinger overlay
  - API/타입/DB 조회 응답에 upper/lower 가격 series가 없어 보류했다.
  - `bb_position`만으로는 실제 가격 overlay 선을 안전하게 복원할 수 없다.
- 보조지표 전체 대개편
  - 이번 CP 금지 범위에 따라 구조 대개편은 하지 않았다.
- AI band fill
  - 이번 CP는 라인 강조와 거래량 bar 중심으로 처리했고, fill은 다음 UI 정리 후보로 남겼다.

## 9. 남은 보조지표 대개편 backlog

- 보조지표가 여러 개 켜지면 가격 차트와 함께 보기 어렵다.
- 패널 간격과 높이를 더 줄일 필요가 있다.
- TradingView처럼 하단 indicator stack 구조를 더 연구할 필요가 있다.
- crosshair/tooltip 동기화가 필요할 수 있다.
- 선택 UX가 아직 checkbox 느낌이라 더 다듬을 여지가 있다.
- Bollinger는 가격 차트 overlay로 옮기기 위해 upper/lower series API 보강이 필요하다.
- Volume은 이번 CP에서 들어갔지만, 향후 별도 compact pane으로 더 자연스럽게 다듬을 수 있다.

## 10. 검증 결과

- `cd C:\Users\user\lens\frontend; npm run build`
  - 통과
- `cd C:\Users\user\lens; powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - 통과
- 브라우저 확인
  - AAPL 1D: 가격 차트, 예측선, AI 위험 범위, 보수적 기준선, 거래량 bar 확인
  - AAPL 1W: 가격 표시, 주간 AI 준비 안내, 거래량 bar 확인
  - AAPL 1M: 가격 전용 안내, 오늘 이후 예측 미표시, 거래량 bar 확인
  - rolling history가 이전보다 더 진하게 표시되는 것 확인
  - 중복 설명 문구가 기본 화면에서 제거된 것 확인
  - Bollinger는 데이터 부재로 미표시 확인
  - console error/warn 0건
- backend unittest
  - 이번 CP는 프론트 표시 변경만 했고 API/백엔드 코드를 수정하지 않아 생략했다.
