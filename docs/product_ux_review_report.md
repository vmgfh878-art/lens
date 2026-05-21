# Lens 제품 UX 리뷰 보고서

## 1. Executive Summary

이번 리뷰는 실행 화면을 직접 띄운 시각 검수가 아니라, 현재 프론트 코드에 노출된 화면 구조, 문구, 상태 처리, 차트 레이어 구성 기준의 정적 UX 감사다. 목적은 디자인 취향 평가가 아니라 “처음 보는 투자 보조도구 사용자가 이해하고 신뢰하며 쓸 수 있는가”를 확인하는 것이다.

전체적으로 Lens는 이미 주식 보기, 보조지표, AI 밴드, 모델 학습, 백테스트를 한 제품 안에 묶는 골격이 있고, empty/error 상태도 완전히 방치되어 있지는 않다. 특히 주식 화면은 가격 차트가 중심이고, AI 밴드와 보수적 예측을 토글로 분리하며, 모델 학습 화면은 Line 후보와 Band 후보를 드러내는 구조가 있다.

하지만 현재 UI는 일반 투자 사용자용 제품과 프로젝트 검토자용 콘솔이 같은 언어를 섞어 쓰고 있다. `run_id`, `feature_set`, `wandb_status`, `lambda_band`, `completed_line_watch`, `product layer run` 같은 내부 용어가 주식 보기와 모델 학습 화면에 그대로 노출된다. 이 때문에 발표/제출용으로는 “많이 만들었다”가 보이지만, 첫 사용자에게는 “이걸 투자 판단에 어떻게 읽어야 하지?”라는 부담이 생긴다.

가장 큰 UX 리스크는 AI 예측선과 AI 밴드가 수익 예측처럼 보일 수 있다는 점이다. 코드상 설명은 “하방 리스크”, “변동 가능 범위”라고 되어 있어 방향은 맞지만, 차트와 사이드 패널에서 “실현 수익 보장 아님”, “리스크 보조 신호”, “검증 품질 기준” 같은 신뢰 단서가 아직 충분히 앞단에 있지 않다.

## 2. 가장 먼저 고쳐야 할 UX 문제 TOP 5

| 우선순위 | 문제 | 근거 | 왜 중요한가 | 제안 |
|---|---|---|---|---|
| P1 | 첫 진입 10초 설명이 약함 | 기본 진입은 `stocks`이고 메타 설명은 “AI 보조지표 기반 주식 분석 대시보드” 수준이다. `frontend/src/components/DashboardPage.tsx:11`, `frontend/src/app/layout.tsx:8` | 사용자는 Lens가 가격 예측기인지, 리스크 보조도구인지, 백테스트 플랫폼인지 즉시 구분하기 어렵다. | 주식 화면 상단에 “가격 차트 위에 검증된 AI 리스크 범위와 방어적 예측선을 겹쳐 보는 도구” 같은 한 줄 설명을 넣는다. |
| P1 | 일반 사용자 화면에 내부 실행 용어가 과다 노출됨 | 예측 출처 패널에 `band run`, `product layer run`, `feature contract`가 노출된다. `frontend/src/components/StockView.tsx:1050`, `frontend/src/components/StockView.tsx:1060`, `frontend/src/components/StockView.tsx:1075` | 신뢰를 주려던 provenance가 오히려 디버그 콘솔처럼 보여 초심자에게 불안감을 줄 수 있다. | 기본 화면은 “예측선 모델 / 밴드 모델 / 검증 상태”만 보여주고, run_id 등은 “상세 보기”로 접는다. |
| P1 | 1D/1W/1M 전환의 제품 계약이 직관적이지 않음 | 타입상 1D/1W 예측 가능이지만 실제 화면은 1D 후보만 예측 layer를 표시한다. `frontend/src/api/client.ts:112`, `frontend/src/components/StockView.tsx:664`, `frontend/src/components/StockView.tsx:974` | 사용자는 1W 버튼을 눌렀을 때 예측이 사라지면 데이터가 없는지, 기능이 막힌 건지, 오류인지 헷갈린다. | 타임프레임 버튼 옆에 “AI 예측은 현재 1D 후보만 제공” 배지를 미리 보여준다. |
| P1 | AI 예측선/밴드의 해석 기준이 차트 안에서 부족함 | 차트 범례는 “AI 밴드 상단/하단”, “보수적 예측”, “별도 scale” 정도만 표시한다. `frontend/src/components/Chart.tsx:492`, `frontend/src/components/Chart.tsx:494`, `frontend/src/components/Chart.tsx:500` | 투자 사용자는 선과 밴드가 가격 목표인지, 위험 범위인지, 모델 불확실성인지 즉시 알아야 한다. | 범례나 사이드 패널에 “가격 목표가 아니라 리스크 판단 보조 범위”를 짧게 명시한다. |
| P2 | 모델 학습 화면이 제품 설명보다 개발 로그에 가까움 | 화면 제목이 “run 콘솔”이고 지표명이 대부분 원문 metric이다. `frontend/src/components/TrainingView.tsx:432`, `frontend/src/components/TrainingView.tsx:515`, `frontend/src/components/TrainingView.tsx:576` | 검토자에게는 유용하지만 발표 중 비전문 청중에게는 이해 비용이 크다. | “제품 후보 요약”을 먼저, 세부 run/config/evaluation은 아래로 배치한다. |

## 3. 화면별 리뷰

### 주식 보기 화면

장점은 가격 정보의 우선순위가 분명하다는 점이다. 상단에 티커, 현재가, 등락률, 거래량이 있고, 그 아래 큰 차트가 나온다. `frontend/src/components/StockView.tsx:948`, `frontend/src/components/StockView.tsx:1020`

P1 리스크는 AI 설명의 위치다. AI 관련 설명이 오른쪽 패널의 “예측 출처”와 “예측 범위” 아래에 있지만, 처음 사용자는 차트를 먼저 보고 선과 밴드의 의미를 추측하게 된다. `frontend/src/components/StockView.tsx:1045`, `frontend/src/components/StockView.tsx:1050`

P1 리스크는 내부 provenance 문구가 많다는 점이다. `Line: ... / Band: ...`, `band run`, `product layer run`, `feature contract`가 그대로 보이며, 이는 검토자에게는 좋지만 투자 사용자에게는 디버그 상태처럼 보일 수 있다. `frontend/src/components/StockView.tsx:900`, `frontend/src/components/StockView.tsx:910`, `frontend/src/components/StockView.tsx:1060`

P2 장점은 실패 상태가 비교적 솔직하다는 점이다. 가격 데이터 없음, 예측 row 없음, 가격 범위 밖 숨김, 티커 검색 실패 등 상태 문구가 있다. `frontend/src/components/StockView.tsx:737`, `frontend/src/components/StockView.tsx:1005`, `frontend/src/components/StockView.tsx:1031`

### 보조지표 패널

장점은 RSI, MACD, ATR, AI 밴드 폭이 독립 패널로 읽히게 되어 있다는 점이다. 보조지표 선택 메뉴와 지표 카드가 분리되어 있고, 지표 설명도 초심자에게 과하지 않다. `frontend/src/components/IndicatorPanel.tsx:176`, `frontend/src/components/IndicatorPanel.tsx:177`, `frontend/src/components/StockView.tsx:1140`

P2 리스크는 전문 투자자 관점에서 TradingView류의 시간축 정렬 보조 패널보다 요약 카드에 가깝게 느껴질 수 있다는 점이다. SVG 미니 차트와 시작/끝 날짜 라벨은 있지만, 지표별 축, 크로스헤어, 가격 차트와의 정밀 동기화 해석은 제한적이다. `frontend/src/components/IndicatorPanel.tsx:127`, `frontend/src/components/IndicatorPanel.tsx:162`, `frontend/src/components/IndicatorPanel.tsx:165`

P2 리스크는 AI 밴드 폭이 다른 보조지표와 같은 레벨로 보이면서도 산출 근거가 충분히 설명되지 않는 점이다. “예측 범위가 넓을수록 불확실성이 크다”는 설명은 좋지만, 밴드 모델 품질과 연결되는 신뢰 단서가 약하다. `frontend/src/components/StockView.tsx:158`, `frontend/src/components/StockView.tsx:1104`

### AI 예측선 / AI 밴드 차트

장점은 forecast 구간을 구분하려는 장치가 있다. 차트에는 “예측 시작” 기준선, rolling prediction history 범례, 별도 scale 안내가 있다. `frontend/src/components/Chart.tsx:488`, `frontend/src/components/Chart.tsx:497`, `frontend/src/components/Chart.tsx:500`

P1 리스크는 별도 scale 안내가 작게 범례에만 들어간다는 점이다. 예측 밴드가 가격 범위와 달라 별도 scale을 쓰면, 사용자는 같은 차트 위 선을 실제 가격축으로 오해할 수 있다. `frontend/src/components/Chart.tsx:202`, `frontend/src/components/Chart.tsx:262`, `frontend/src/components/Chart.tsx:500`

P2 리스크는 rolling history가 얇은 선으로 표시되지만, “과거 예측 이력”과 “현재 forecast”의 의미 차이가 일반 사용자에게 충분히 설명되지 않는 점이다. 얇은 선 안내가 영어 섞인 “rolling prediction history”로 나온다. `frontend/src/components/Chart.tsx:497`

P2 리스크는 밴드가 면 영역이 아니라 상단/하단 dashed line으로 표시된다. 현재 범례는 밴드 영역처럼 보이지만 실제 렌더링은 상단/하단 선 중심이라, 초심자가 “범위”로 즉시 인지하기 어렵다. `frontend/src/components/Chart.tsx:405`, `frontend/src/components/Chart.tsx:409`, `frontend/src/components/Chart.tsx:419`

### 모델 학습 화면

장점은 Line 후보와 Band 후보를 분리해서 보여주는 방향이 맞다. 제품 후보 카드가 있고, Legacy / 기타를 제품 기본 후보에서 제외한다고 표시한다. `frontend/src/components/TrainingView.tsx:478`, `frontend/src/components/TrainingView.tsx:485`, `frontend/src/components/TrainingView.tsx:492`, `frontend/src/components/TrainingView.tsx:506`

P1 리스크는 화면의 첫 인상이 제품 설명보다 run 로그에 가깝다는 점이다. 제목이 “run 콘솔”이고, 상태 탭도 `완료(completed)`, `NaN 실패(failed_nan)`, `품질 게이트 실패(failed_quality_gate)`처럼 내부 상태를 노출한다. `frontend/src/components/TrainingView.tsx:432`, `frontend/src/components/TrainingView.tsx:443`, `frontend/src/components/TrainingView.tsx:453`

P2 리스크는 metric 이름이 전문가에게도 설명 없이 촘촘하다. `ic_mean`, `false_safe_tail_rate`, `asymmetric_interval_score`, `downside_width_ic` 등은 모델 검토자에게 의미가 있지만 발표 청중에게는 핵심 품질 메시지를 가린다. `frontend/src/components/TrainingView.tsx:34`, `frontend/src/components/TrainingView.tsx:41`

P2 장점은 fake 값을 만들지 않았다고 명시한 점이다. Baseline 비교 준비 필요 문구는 투명성 측면에서 신뢰에 도움이 된다. 다만 이 문구는 발표 화면에서는 “미완성”으로 크게 보일 수 있어 위치 조정이 필요하다. `frontend/src/components/TrainingView.tsx:592`, `frontend/src/components/TrainingView.tsx:595`

### 백테스트 화면

장점은 수수료 반영 수익률, 수수료 반영 샤프, 최대낙폭, 승률, 회전율, 거래 수를 상단 카드로 보여줘 리스크 관리 관점의 핵심 지표가 있다. `frontend/src/components/BacktestView.tsx:187`

P1 리스크는 전략 선택지가 `band_breakout_v1` 하나로 노출되어 제품 언어보다 실험 코드명처럼 보인다는 점이다. `frontend/src/components/BacktestView.tsx:8`, `frontend/src/components/BacktestView.tsx:156`

P2 리스크는 백테스트가 PatchTST run만 대상으로 잡혀 있어, 화면에서 Line/Band 분리 구조와 백테스트 대상의 관계가 흐릿하다. `frontend/src/components/BacktestView.tsx:10`, `frontend/src/components/BacktestView.tsx:83`

P2 리스크는 수익 곡선과 낙폭 차트가 placeholder다. 발표 때 “여기에 표시합니다” 문구가 보이면 완성도가 낮아 보일 수 있다. `frontend/src/components/BacktestView.tsx:242`, `frontend/src/components/BacktestView.tsx:245`, `frontend/src/components/BacktestView.tsx:249`

### 데이터셋 / 라이브러리 / 리포트 화면

현재 확인한 `frontend/src/components/DashboardPage.tsx` 기준으로 화면은 `stocks`, `backtests`, `training` 세 개다. 데이터셋, 라이브러리, 리포트 전용 화면은 확인되지 않았다. `frontend/src/components/DashboardPage.tsx:11`, `frontend/src/components/DashboardPage.tsx:15`

P3 리스크는 발표자가 “데이터셋/라이브러리/리포트도 있다”고 말할 경우 실제 메뉴 구조와 어긋날 수 있다는 점이다. 지금은 “모델 학습 화면 안에 실행 로그와 평가 테이블이 있다”고 설명하는 편이 안전하다.

## 4. 사용자 시나리오별 막히는 지점

### 처음 들어온 사용자

P1: 첫 화면에서 Lens가 “AI 수익 예측기”인지 “리스크 보조 대시보드”인지 즉시 고정되지 않는다. 메타 설명과 브랜드 캡션은 있지만 화면 안의 제품 한 줄 정의가 부족하다. `frontend/src/components/AppShell.tsx:167`, `frontend/src/components/AppShell.tsx:168`, `frontend/src/app/layout.tsx:8`

P1: AAPL에서 가격, 예측선, AI 밴드의 구분은 범례와 토글로 가능하지만, 예측선이 “목표가”가 아니라 “보수적 예측”이라는 점은 오른쪽 설명을 읽어야 확실해진다. `frontend/src/components/Chart.tsx:492`, `frontend/src/components/Chart.tsx:494`, `frontend/src/components/StockView.tsx:1118`, `frontend/src/components/StockView.tsx:1123`

### 투자 보조지표 사용자

P2: RSI/MACD/AI 밴드 폭 설명은 친절하지만, 가격 차트와 보조지표 패널 사이의 시각적 연결은 미니 차트 중심이다. TradingView류 사용자는 하단 지표 축과 동기화된 crosshair를 기대할 수 있다. `frontend/src/components/IndicatorPanel.tsx:127`, `frontend/src/components/IndicatorPanel.tsx:196`

P2: 지표를 켜고 끄는 흐름은 오른쪽 패널의 checkbox로 명확하지만, “왜 기본값이 RSI/MACD/AI 밴드 폭인지”에 대한 안내는 없다. `frontend/src/components/StockView.tsx:82`, `frontend/src/components/StockView.tsx:1137`

### 리스크 관리 중심 사용자

P1: 위험 구간을 빠르게 파악하는 목적이라면 AI 밴드 폭, 커버리지, 평균 밴드 폭이 가격 차트보다 한 단계 늦게 보인다. 현재는 오른쪽 패널의 레이어 메트릭 안에 들어간다. `frontend/src/components/StockView.tsx:1104`, `frontend/src/components/StockView.tsx:1108`

P1: “하방 리스크를 우선 반영한 방어적 예측선” 문구는 좋지만, 밴드와 함께 볼 때 어떤 조합이면 watch인지 avoid인지까지는 안내하지 않는다. `frontend/src/components/StockView.tsx:1123`

### 모델/프로젝트 검토자

P1: Line model과 Band model 분리는 이해 가능하다. 다만 제품 화면과 학습 화면이 서로 다른 수준의 언어를 쓴다. 주식 화면은 “예측선과 AI 밴드를 별도 제품 후보 run에서 표시”라고 하고, 학습 화면은 “Line layer / Band layer”, `role`, `checkpoint_selection`, `wandb_status`를 보여준다. `frontend/src/components/StockView.tsx:910`, `frontend/src/components/TrainingView.tsx:269`, `frontend/src/components/TrainingView.tsx:515`

P2: composite를 쓰지 않는 구조는 Legacy / 기타 섹션과 notice로 전달되지만, “왜 composite를 제외했는지” 한 줄 근거는 없다. `frontend/src/components/TrainingView.tsx:492`, `frontend/src/components/TrainingView.tsx:525`

## 5. 문구/용어 개선 제안

| 우선순위 | 현재 문구 | 제안 문구 | 이유 |
|---|---|---|---|
| P1 | 주식 분석 | AI 리스크 밴드 기반 주식 분석 | 서비스 정체성을 더 빨리 전달한다. |
| P1 | 예측 범위 / 표시 중 | AI 리스크 범위 / 표시 중 | 수익 예측처럼 보이는 느낌을 줄인다. |
| P1 | 모델이 예측한 향후 가격 변동 가능 범위입니다. | 모델이 보는 향후 변동 가능 범위입니다. 투자 판단을 보조하는 참고 범위입니다. | 목표가 오해를 줄인다. |
| P1 | 보수적 예측 | 방어적 예측선 | 리스크 관리 목적이 더 선명하다. |
| P2 | band run | 밴드 모델 실행 ID | 내부 용어를 사용자 언어로 바꾼다. |
| P2 | product layer run | 제품 후보 모델 | 디버그 느낌을 줄인다. |
| P2 | completed_line_watch | 예측선 관찰 후보 | 상태값을 사람이 읽는 문장으로 바꾼다. |
| P2 | run 콘솔 | 모델 후보 검토 | 발표/검토자에게 의도를 더 잘 전달한다. |
| P2 | NaN 실패(failed_nan) | 학습 실패: 값 오류 | 일반 사용자에게 원인을 더 쉽게 전달한다. |
| P3 | rolling prediction history | 최근 예측 이력 | 영어 혼용을 줄인다. |

## 6. 차트/보조지표 개선 제안

P1: AI 밴드와 예측선이 별도 scale로 표시될 때는 범례의 작은 문구가 아니라 차트 상단 배지로 알려야 한다. “예측 레이어는 가격축과 다른 축으로 표시 중”을 명확히 하지 않으면 같은 가격 차트 위의 선을 실제 가격으로 오해할 수 있다. `frontend/src/components/Chart.tsx:500`

P1: forecast 구간은 기준선만으로는 약하다. 예측 시작 이후 배경을 아주 옅게 음영 처리하고, tooltip 또는 legend에 “이 구간은 미래 예측 구간”을 표시하는 편이 안전하다. `frontend/src/components/Chart.tsx:488`

P1: AI 밴드는 상단/하단 선만이 아니라 면 영역으로 읽히게 해야 한다. 지금처럼 dashed line 두 개만 있으면 범위보다 두 개의 가격선처럼 보일 수 있다. `frontend/src/components/Chart.tsx:405`, `frontend/src/components/Chart.tsx:416`

P2: 보조지표 패널은 현재 요약 카드로는 좋지만, 전문 투자자 데모에서는 “가격 차트와 같은 날짜를 보고 있다”는 연결감이 약하다. 차트 hover 날짜와 보조지표 최신값을 동기화하거나, 선택 지표 하나를 가격 차트 아래 full-width로 확장하는 모드가 있으면 신뢰가 오른다. `frontend/src/components/IndicatorPanel.tsx:196`

P2: AI 밴드 폭은 리스크 관리 핵심 지표이므로 기본 보조지표 카드 중 하나가 아니라 오른쪽 AI 패널에도 “현재 밴드 폭 상태: 보통/넓음/매우 넓음”처럼 해석 라벨을 붙이는 편이 좋다. `frontend/src/components/StockView.tsx:1108`

P3: 월봉은 가격 전용이라는 정책은 좋다. 다만 1M을 누른 뒤 AI 토글이 비활성화될 때 “월봉은 가격 지표만 제공됩니다” 외에 “AI 후보는 현재 1D 기준 검증 중”이라는 이유를 붙이면 덜 고장처럼 보인다. `frontend/src/components/StockView.tsx:338`

## 7. 모델 학습 화면 개선 제안

P1: 화면 상단을 “제품 후보 요약”으로 시작해야 한다. 현재는 run 목록과 콘솔 느낌이 먼저 오기 때문에, 발표에서는 모델이 잘 정리되어 있다는 인상보다 로그를 뒤지는 인상이 강할 수 있다. `frontend/src/components/TrainingView.tsx:432`, `frontend/src/components/TrainingView.tsx:468`

P1: Line 후보와 Band 후보 카드에는 각각 “무엇을 담당하는가”를 한 문장으로 붙여야 한다. 예시는 “Line 후보: 방향성과 방어적 예측선 담당”, “Band 후보: 불확실성 범위와 커버리지 담당”이다. `frontend/src/components/TrainingView.tsx:269`, `frontend/src/components/TrainingView.tsx:509`

P2: metric은 두 층으로 나눠야 한다. 기본 카드에는 “방향성 품질”, “하방 리스크 탐지”, “커버리지 오차”, “밴드 폭 안정성”처럼 사람이 읽는 이름을 쓰고, 원래 metric명은 상세 표에 둔다. `frontend/src/components/TrainingView.tsx:34`, `frontend/src/components/TrainingView.tsx:41`

P2: Baseline 비교는 지금처럼 fake 값을 만들지 않은 점은 좋지만, “준비 필요”가 화면 중앙에 크면 발표 리스크가 크다. baseline이 없을 때는 “비교 기준 미연결”로 작게 표시하고, 준비된 뒤에는 가장 중요한 2개 지표만 보여주는 편이 낫다. `frontend/src/components/TrainingView.tsx:592`, `frontend/src/components/TrainingView.tsx:595`

P2: local logs 섹션은 개발자에게는 유용하지만 제품 데모에서는 너무 내부 구현처럼 보인다. 제출/검토자 모드에서는 접힌 상세로 두고, 발표 기본 동선에서는 숨기는 것이 좋다. `frontend/src/components/TrainingView.tsx:601`

## 8. 발표 데모 동선 제안

1. 주식 보기에서 AAPL을 연다. 먼저 “Lens는 가격 차트 위에 AI 리스크 밴드와 방어적 예측선을 얹어 보는 투자 보조도구”라고 정의한다.

2. 가격 차트를 보여주고, 캔들/라인 전환과 1D/1M 전환을 짧게 보여준다. 이때 1M은 가격 전용이라고 먼저 말해 혼란을 막는다.

3. AI 밴드를 켜고 끄며 “밴드가 넓으면 모델이 보는 불확실성이 크다”고 설명한다. 보수적 예측선은 목표가가 아니라 하방 리스크를 반영한 참고선이라고 말한다.

4. 보조지표 패널에서 RSI, MACD, AI 밴드 폭을 보여준다. “전통 지표와 AI 불확실성 지표를 같은 흐름에서 본다”가 핵심 메시지다.

5. 모델 학습 화면으로 이동한다. Line 후보와 Band 후보가 분리되어 있고, composite는 제품 기본 후보가 아니라 legacy 참고 산출물이라고 설명한다.

6. 백테스트 화면으로 이동한다. 수수료 반영 수익률, 샤프, 최대낙폭을 먼저 보여주고, “수익률보다 비용 반영과 낙폭까지 함께 본다”는 리스크 관리 메시지로 마무리한다.

데모 중 막힐 수 있는 부분은 `run_id`, `wandb_status`, `feature_set`, `baseline 준비 필요`, placeholder 수익 곡선이다. 발표 전에는 이 네 가지가 보이는 위치를 접거나, “검토자 상세 영역”이라고 먼저 프레이밍하는 것이 안전하다.

## 9. 지금 고치지 않아도 되는 것

P3: 전체 레이아웃과 밝은 금융 대시보드 톤은 당장 갈아엎을 필요가 없다. 가격 차트가 큰 영역을 차지하고, 오른쪽에 AI/지표 제어가 붙는 구조는 사용자가 이해하기 쉽다. `frontend/src/app/globals.css:885`, `frontend/src/app/globals.css:915`, `frontend/src/app/globals.css:1017`

P3: 사이드바 메뉴 수는 현재 적절하다. 데이터셋/리포트 메뉴를 무리하게 추가하기보다, 현재 세 화면의 메시지를 정리하는 것이 먼저다. `frontend/src/components/AppShell.tsx:41`, `frontend/src/components/AppShell.tsx:47`

P3: empty state 자체는 이미 존재하므로 전면 재설계보다 문구 개선이 우선이다. `frontend/src/app/globals.css:1650`, `frontend/src/components/BacktestView.tsx:257`, `frontend/src/components/TrainingView.tsx:498`

P3: 반응형 처리는 기본적으로 고려되어 있다. 모바일에서 세로 스택으로 내려가는 규칙이 있으므로, 우선순위는 반응형 레이아웃보다 AI 의미 전달과 용어 정리다. `frontend/src/app/globals.css:2030`, `frontend/src/app/globals.css:2044`

## 10. 코드 수정 금지 확인

이번 작업에서 코드 파일은 수정하지 않았다. 새로 작성한 파일은 이 보고서 `docs/product_ux_review_report.md` 하나뿐이다.

리뷰는 읽기 전용 명령으로만 수행했다. 확인한 주요 파일은 `frontend/src/components/StockView.tsx`, `frontend/src/components/Chart.tsx`, `frontend/src/components/IndicatorPanel.tsx`, `frontend/src/components/TrainingView.tsx`, `frontend/src/components/BacktestView.tsx`, `frontend/src/components/AppShell.tsx`, `frontend/src/components/DashboardPage.tsx`, `frontend/src/api/client.ts`, `frontend/src/app/globals.css`, `frontend/src/app/layout.tsx`, `frontend/src/app/page.tsx`다.
