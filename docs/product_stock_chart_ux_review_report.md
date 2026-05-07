# 주식 보기 차트 UX 리뷰 보고서

## 1. Executive Summary

이번 리뷰는 주식 보기 화면의 가격 차트, 보조지표, 보수적 예측선, AI 밴드가 일반 투자 사용자에게 명확하게 전달되는지 정적 코드 기준으로 확인한 결과다. 브라우저 실행이나 스크린샷 검수는 하지 않았고, `StockView.tsx`, `Chart.tsx`, `IndicatorPanel.tsx`, `globals.css`의 렌더링 구조와 문구를 근거로 판단했다.

전체 방향은 좋아졌다. 첫 진입 한 줄 설명이 있고, 가격 차트가 가장 큰 영역을 차지하며, 보수적 예측선과 AI 밴드를 오른쪽 패널에서 켜고 끄는 구조도 이해 가능하다. 거래량 bar가 가격 차트 안에 들어가고, RSI/MACD/AI 밴드 폭이 기본 하단 지표로 선택되는 점도 일반 사용자에게 과하지 않다. 근거: `frontend/src/components/StockView.tsx:83`, `frontend/src/components/StockView.tsx:917`, `frontend/src/components/StockView.tsx:1008`, `frontend/src/components/StockView.tsx:1060`

다만 반드시 잡아야 할 혼란이 있다. 가장 큰 문제는 차트가 실제 예측 시작일 대신 현재 달력 날짜를 “오늘” 마커로 추가하고, 범례에 “오늘 이후 예측”이라고 표시한다는 점이다. 모델 기준일과 forecast date가 따로 있는데도 사용자는 예측이 오늘부터 시작한다고 이해할 수 있다. 근거: `frontend/src/components/Chart.tsx:376`, `frontend/src/components/Chart.tsx:380`, `frontend/src/components/Chart.tsx:627`, `frontend/src/components/Chart.tsx:637`

두 번째 핵심 문제는 예측 레이어가 별도 가격 축으로 표시될 수 있다는 점이다. 범례에 안내는 있지만, 일반 투자자는 같은 차트 위 선이면 같은 가격축이라고 가정한다. 이 상태에서 보수적 예측선이나 AI 밴드를 실제 가격과 직접 비교하면 잘못 읽을 수 있다. 근거: `frontend/src/components/Chart.tsx:48`, `frontend/src/components/Chart.tsx:438`, `frontend/src/components/Chart.tsx:513`, `frontend/src/components/Chart.tsx:638`

## 2. P1/P2/P3 UX 이슈 표

| 우선순위 | 이슈 | 화면/컴포넌트 | 사용자가 이해하기 어려운 이유 | 근거 | 제안 방향 |
|---|---|---|---|---|---|
| P1 | “오늘” 마커가 실제 예측 시작일처럼 보임 | 가격 차트 | 현재 달력 날짜를 차트 timeline에 넣고 “오늘 이후 예측”이라고 표시한다. 모델 기준일, forecast 시작일, 오늘이 서로 다를 수 있어 예측 구간을 잘못 이해할 수 있다. | `Chart.tsx:376`, `Chart.tsx:380`, `Chart.tsx:627`, `Chart.tsx:637`, `StockView.tsx:1034` | 마커 기준을 `forecast_dates[0]` 또는 모델 기준일 다음 거래일로 맞추고, 문구를 “예측 시작” 또는 “모델 기준일 이후”로 바꾼다. |
| P1 | 예측 레이어가 별도 가격 축으로 표시될 수 있음 | 가격 차트 | 같은 차트 위에 있는 선은 같은 가격축이라고 생각하기 쉽다. 별도 축이면 보수적 예측선/밴드와 실제 가격의 위치 비교가 직관적으로 깨진다. | `Chart.tsx:48`, `Chart.tsx:438`, `Chart.tsx:513`, `Chart.tsx:546`, `Chart.tsx:638` | 별도 축 사용 시 차트 상단 경고 배지와 축 라벨을 강하게 표시하거나, 제품 화면에서는 별도 축 표시를 제한한다. |
| P2 | 보수적 예측선과 AI 밴드가 둘 다 굵은 dashed line이라 순간 구분이 어렵다 | 가격 차트 | 보수적 예측선도 dashed 4px, AI 밴드 상·하단도 dashed 4px다. 색은 다르지만 선 스타일이 같아 AI 밴드가 “두 개의 목표가 선”처럼 보일 수 있다. | `Chart.tsx:544`, `Chart.tsx:555`, `Chart.tsx:566`, `Chart.tsx:567`, `Chart.tsx:631`, `Chart.tsx:632` | 보수적 예측선은 단일 실선/점선, AI 밴드는 옅은 면 영역 또는 확실한 상하 범위 스타일로 분리한다. |
| P2 | 과거 rolling 예측 이력이 신뢰보다 복잡함을 줄 수 있음 | 가격 차트 | 최근 90개 예측 이력을 한 레이어에 얹고, 범례는 “과거 예측” 한 단어만 보여준다. 현재 예측과 과거 예측의 의미 차이가 충분히 설명되지 않는다. | `StockView.tsx:84`, `StockView.tsx:813`, `StockView.tsx:820`, `Chart.tsx:190`, `Chart.tsx:205`, `Chart.tsx:634` | 과거 예측은 기본 접기 또는 별도 토글로 분리하고, “과거 1일차 예측 이력”처럼 무엇을 보여주는지 구체화한다. |
| P2 | 1D/1W/1M 전환 기대와 실제 표시가 완전히 맞지 않음 | 타임프레임 버튼/AI 레이어 | 1W 버튼은 1D와 같은 위치에 있지만 실제 AI 예측은 “준비 중”이다. 1M은 가격 전용이다. 사용자는 버튼을 누른 뒤에야 예측이 사라지는 이유를 알게 된다. | `StockView.tsx:73`, `StockView.tsx:319`, `StockView.tsx:673`, `StockView.tsx:950` | 버튼 옆이나 AI 패널 상단에 “AI 예측은 현재 1D만 제공” 배지를 미리 노출한다. |
| P2 | 첫 로딩 중 “가격 데이터가 없습니다”로 보일 수 있음 | 차트 패널 empty state | 초기 `priceData`가 비어 있는 동안 `isLoading`과 무관하게 empty state가 보인다. 첫 사용자에게 데이터가 없는 종목처럼 보일 수 있다. | `StockView.tsx:587`, `StockView.tsx:631`, `StockView.tsx:1008` | `isLoading`일 때는 “가격 데이터를 불러오는 중입니다”로 분리한다. |
| P2 | “예측가”, “커버리지”, “평균 밴드 폭”이 투자자에게 오해 또는 장벽이 됨 | 오른쪽 예측 레이어 패널 | “예측가”는 목표가처럼 들릴 수 있고, “커버리지”는 통계 용어라 설명 없이 이해하기 어렵다. | `StockView.tsx:1077`, `StockView.tsx:1081`, `StockView.tsx:1089` | “보수적 참고값”, “실제 수익률 포함률”, “AI 범위 평균 폭”처럼 의미를 풀어쓴다. |
| P2 | AI 밴드 폭이 raw 숫자로 보임 | 하단 지표 | 최신값이 `0.0607` 같은 숫자로 보이면 사용자가 넓은지 좁은지 판단하기 어렵다. | `StockView.tsx:233`, `StockView.tsx:235`, `StockView.tsx:541`, `IndicatorPanel.tsx:209` | “넓음/보통/좁음” 상태 라벨이나 최근 평균 대비 해석을 붙인다. |
| P3 | ATR은 제공되지만 기본 선택에는 없음 | 하단 지표 | 리스크 중심 사용자에게 ATR은 AI 밴드 폭과 함께 보고 싶은 지표지만 기본값은 RSI/MACD/AI 밴드 폭이다. | `StockView.tsx:83`, `StockView.tsx:222`, `StockView.tsx:225` | 다음 UX 실험에서 기본 지표 세트를 RSI/MACD/ATR/AI 밴드 폭 중 어떤 조합으로 둘지 비교한다. |
| P3 | 하단 지표 미니 차트는 요약형이라 전문 차트 사용자는 가볍게 느낄 수 있음 | IndicatorPanel | 96px SVG 카드로 시작/끝 날짜와 최신값은 보이지만, TradingView류 하단 패널처럼 crosshair와 정밀 축을 제공하지 않는다. | `IndicatorPanel.tsx:159`, `IndicatorPanel.tsx:174`, `IndicatorPanel.tsx:177`, `globals.css:1668` | 현재는 유지하되, 데모 이후 고급 모드에서 full-width 지표 패널을 검토한다. |

## 3. 지금 유지해도 되는 부분

첫 진입 설명은 유지하는 편이 좋다. “가격 차트 위에 보수적 예측선과 AI 밴드를 겹쳐 보며, 리스크를 먼저 확인”한다는 문구는 이 화면의 목적을 빠르게 전달한다. 근거: `frontend/src/components/StockView.tsx:916`, `frontend/src/components/StockView.tsx:917`

가격 차트를 중심에 두고 오른쪽에 예측 레이어와 하단 지표 선택을 모은 구조는 유지해도 된다. 가격이 먼저 보이고, AI는 보조 레이어로 위치한다는 정보 위계가 맞다. 근거: `frontend/src/components/StockView.tsx:1000`, `frontend/src/components/StockView.tsx:1021`, `frontend/src/components/StockView.tsx:1058`

상세 정보를 접어둔 방식은 좋다. 실행 ID는 검토자에게 필요하지만 일반 투자 사용자에게는 과하다. 현재처럼 “상세 정보” 아래에 넣는 방향은 유지하되, 기본 노출은 더 줄이는 편이 좋다. 근거: `frontend/src/components/StockView.tsx:1041`, `frontend/src/components/StockView.tsx:1049`, `frontend/src/components/StockView.tsx:1051`

거래량 bar 토글은 유지할 만하다. 가격 차트 아래쪽에 volume scale을 따로 두고, 사용자가 끌 수 있는 구조라 밀도 조절이 가능하다. 근거: `frontend/src/components/Chart.tsx:490`, `frontend/src/components/Chart.tsx:491`, `frontend/src/components/StockView.tsx:1101`, `frontend/src/components/StockView.tsx:1103`

RSI/MACD/AI 밴드 폭을 기본 하단 지표로 둔 것도 당장은 괜찮다. 처음 보는 사용자가 너무 많은 지표를 동시에 보지 않게 제한한다. 근거: `frontend/src/components/StockView.tsx:83`, `frontend/src/components/IndicatorPanel.tsx:188`

## 4. 다음 CP에서 고칠 후보

1. P1: “오늘” 마커를 없애거나 `forecast_dates[0]` 기준 “예측 시작” 마커로 바꾼다. 모델 기준일과 예측 기간 표기도 같은 기준으로 통일한다.

2. P1: 별도 가격 축이 켜질 때 사용자가 오해하지 않도록 강한 경고 배지, 축 라벨, 또는 제품 화면 제한 정책을 정한다.

3. P2: 보수적 예측선과 AI 밴드 시각 언어를 분리한다. 보수적 예측선은 단일 선, AI 밴드는 범위 영역으로 읽히게 하는 것이 좋다.

4. P2: 과거 예측 이력은 기본 off 또는 별도 토글로 분리한다. 기본 차트는 현재 가격, 현재 보수적 예측선, 현재 AI 밴드만 먼저 보이게 한다.

5. P2: 1W/1M 전환 전에 “AI 예측은 현재 1D만 제공” 상태를 미리 보여준다. 버튼을 누른 뒤 사라지는 느낌을 줄인다.

6. P2: 로딩 상태를 empty state와 분리한다. 첫 진입에서 “가격 데이터가 없습니다”가 아니라 “가격 데이터를 불러오는 중입니다”가 먼저 보여야 한다.

7. P2: 용어를 투자자 언어로 다듬는다. “예측가”는 “보수적 참고값”, “커버리지”는 “실제 수익률 포함률”, “평균 밴드 폭”은 “AI 범위 평균 폭”이 더 안전하다.

## 5. 수정하지 말아야 할 것

가격 차트를 AI보다 뒤로 밀면 안 된다. Lens의 신뢰는 가격을 먼저 보여주고 AI를 보조 레이어로 얹는 구조에서 나온다.

AI 밴드를 매수/매도 신호처럼 보이게 하면 안 된다. 현재 “리스크 참고 지표”라는 설명 방향은 유지해야 한다. 근거: `frontend/src/components/StockView.tsx:1072`

실행 ID나 모델 내부 정보를 기본 화면에 더 많이 노출하면 안 된다. 필요한 경우 접힌 상세로 두는 현재 방향이 맞다.

1W/1M 예측을 준비 중인데 fake data로 채우면 안 된다. 준비 중 상태를 솔직하게 보여주는 편이 제품 신뢰에 더 좋다. 근거: `frontend/src/components/StockView.tsx:319`, `frontend/src/components/StockView.tsx:673`

보조지표를 한 번에 너무 많이 기본 표시하면 안 된다. 기본은 간단하게 두고, 고급 사용자가 추가하는 구조를 유지하는 것이 좋다.

## 6. 코드 수정 없이 리뷰만 했다는 확인

이번 작업에서 코드 파일, 스타일 파일, 테스트 파일은 수정하지 않았다. 새로 작성한 파일은 이 보고서 `docs/product_stock_chart_ux_review_report.md` 하나뿐이다.

리뷰는 정적 코드 기준으로만 수행했다. 확인한 주요 파일은 `frontend/src/components/StockView.tsx`, `frontend/src/components/Chart.tsx`, `frontend/src/components/IndicatorPanel.tsx`, `frontend/src/app/globals.css`다.
