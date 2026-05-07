# AI 모델 페이지 UX 리뷰 보고서

## 1. 리뷰 범위

이번 리뷰는 `frontend/src/components/TrainingView.tsx`에 구현된 AI 모델 페이지를 정적 코드 기준으로 검토했다. 스크린샷이나 브라우저 실행은 하지 않았고, 코드·문구·상태 처리·접힘 구조만 기준으로 판단했다.

목표는 일반 투자 사용자가 “이 모델을 어떻게 읽어야 하는지” 이해할 수 있는지, 프로젝트 검토자가 “왜 이 모델이 제품 후보이고 이전 실험은 왜 제외됐는지” 납득할 수 있는지 확인하는 것이다.

## 2. 종합 판단

현재 AI 모델 페이지는 이전의 run 콘솔형 화면보다 훨씬 제품 설명에 가까워졌다. 메뉴명도 “AI 모델”로 정리됐고, 상단 문구에서 예측선 모델과 AI 밴드 모델을 분리 평가한다고 설명한다. 근거: `frontend/src/components/AppShell.tsx:50`, `frontend/src/components/TrainingView.tsx:1859`, `frontend/src/components/TrainingView.tsx:1861`

4슬롯 구조도 명확하다. 1D 보수적 예측선, 1D AI 밴드, 1W 보수적 예측선 준비 중, 1W AI 밴드 준비 중이 카드로 고정되어 있어 현재 제품 범위를 빠르게 파악할 수 있다. 근거: `frontend/src/components/TrainingView.tsx:87`, `frontend/src/components/TrainingView.tsx:98`, `frontend/src/components/TrainingView.tsx:109`, `frontend/src/components/TrainingView.tsx:120`

다만 신뢰 UX 관점에서 중요한 문제가 남아 있다. 제품 슬롯은 정적으로 “사용 중”을 표시하고, 상세 데이터가 없어도 일부 지표는 fallback 숫자로 보일 수 있다. 사용자는 실제 API/DB 상태와 무관하게 모델이 검증 완료된 것처럼 받아들일 수 있다. 이건 다음 CP에서 P1로 막아야 한다.

## 3. 중점 리뷰 결과

### 3.1 제품 모델 현황 4슬롯

현재 상태 전달은 전반적으로 좋다. 1D 두 모델은 “사용 중”, 1W 두 모델은 “준비 중”으로 분리되어 있고, 제품 확장 계획도 자연스럽게 보인다. 근거: `frontend/src/components/TrainingView.tsx:19`, `frontend/src/components/TrainingView.tsx:87`, `frontend/src/components/TrainingView.tsx:98`, `frontend/src/components/TrainingView.tsx:110`, `frontend/src/components/TrainingView.tsx:121`

P1 리스크는 4슬롯의 상태가 실제 로딩 결과와 직접 묶이지 않는다는 점이다. `PRODUCT_SLOTS`는 정적 상수이고, 제품 상세 fetch가 실패해도 슬롯 자체는 “사용 중”으로 남을 수 있다. 근거: `frontend/src/components/TrainingView.tsx:83`, `frontend/src/components/TrainingView.tsx:1803`, `frontend/src/components/TrainingView.tsx:1808`, `frontend/src/components/TrainingView.tsx:1872`

### 3.2 1D 보수적 예측선과 1D AI 밴드 차이

차이는 대체로 이해된다. 예측선은 “수익 방향과 종목 순위”, AI 밴드는 “예상 변동 범위”라고 설명한다. 근거: `frontend/src/components/TrainingView.tsx:93`, `frontend/src/components/TrainingView.tsx:104`

예측선 상세도 “수익 방향과 종목 순위 판단에는 사용할 수 있지만 위험 회피 품질은 개선 중”이라고 밝히고, AI 밴드는 “위험 범위 보조지표”라고 밝힌다. 근거: `frontend/src/components/TrainingView.tsx:1177`, `frontend/src/components/TrainingView.tsx:1291`

P2 리스크는 “보수적 예측선”이라는 말이 일반 사용자에게 가격 목표선처럼 들릴 수 있다는 점이다. 상세에는 “단독 매매 신호가 아니라 참고선”이라는 좋은 방어 문구가 있지만, 슬롯 카드 단계에서는 그 경고가 약하다. 근거: `frontend/src/components/TrainingView.tsx:87`, `frontend/src/components/TrainingView.tsx:1210`, `frontend/src/components/TrainingView.tsx:1269`

### 3.3 이전 실험 접힘 구조

이전 실험을 `details/summary`로 접어두고, 예측선 실험과 밴드 실험을 나눈 구조는 자연스럽다. 일반 사용자는 제품 모델만 보고 지나갈 수 있고, 검토자는 필요할 때 펼칠 수 있다. 근거: `frontend/src/components/TrainingView.tsx:1112`, `frontend/src/components/TrainingView.tsx:1903`, `frontend/src/components/TrainingView.tsx:1905`

P2 리스크는 페이지 로딩 시 이전 실험 상세를 여러 개 미리 fetch하는 구조라, 실험 수가 늘면 “AI 모델 정보를 불러오는 중입니다” 상태가 길어질 수 있다는 점이다. 이건 기능보다 UX 체감 문제다. 근거: `frontend/src/components/TrainingView.tsx:1813`, `frontend/src/components/TrainingView.tsx:1818`, `frontend/src/components/TrainingView.tsx:1889`

### 3.4 제품 모델 대비 이전 실험 비교

비교 설명은 꽤 납득 가능하다. 제품 모델과 이 실험의 값, 차이, 해석을 한 테이블에 보여주고, 최종 판단 문구도 제공한다. 근거: `frontend/src/components/TrainingView.tsx:1616`, `frontend/src/components/TrainingView.tsx:1624`, `frontend/src/components/TrainingView.tsx:1709`

P2 리스크는 이전 실험 이름에 아직 실험실 언어가 남아 있다는 점이다. `no fundamentals`, `Line Gate`, `Val Total`, `CNN-LSTM q...` 같은 표현은 검토자에게는 힌트지만 일반 사용자에게는 막힌다. 근거: `frontend/src/components/TrainingView.tsx:551`, `frontend/src/components/TrainingView.tsx:566`, `frontend/src/components/TrainingView.tsx:569`, `frontend/src/components/TrainingView.tsx:579`

### 3.5 상세 정보 깊이

상세 정보는 접혀 있어서 기본 UX를 망치지는 않는다. “모델 설정·평가 지표·저장 정보”로 설명하는 것도 검토자용 상세라는 방향이 맞다. 근거: `frontend/src/components/TrainingView.tsx:981`, `frontend/src/components/TrainingView.tsx:983`

P2 리스크는 상세를 열었을 때 `CI 집계 기준`, `빠른 CI 목표`, `밴드 손실 가중치`, `실험 추적 ID`, `q_low/q_high` 같은 표현이 여전히 어렵다는 점이다. 라벨은 한국어화했지만, 일반 사용자가 의미를 이해하기에는 설명이 부족하다. 근거: `frontend/src/components/TrainingView.tsx:143`, `frontend/src/components/TrainingView.tsx:145`, `frontend/src/components/TrainingView.tsx:737`, `frontend/src/components/TrainingView.tsx:738`, `frontend/src/components/TrainingView.tsx:742`

### 3.6 실패/기준 미달 실험 표시

기준 미달 실험을 숨기지 않고 “목표 기준에 미치지 못해 현재 제품 화면에는 쓰지 않는 실험”이라고 설명하는 방향은 신뢰에 도움이 된다. 근거: `frontend/src/components/TrainingView.tsx:591`, `frontend/src/components/TrainingView.tsx:598`, `frontend/src/components/TrainingView.tsx:1609`

P1 리스크는 `failed_quality_gate`만 불러오고 `failed_nan`은 AI 모델 페이지의 이전 실험 목록에 포함하지 않는다는 점이다. 실패/기준 미달 실험을 보여준다는 기대가 있다면, NaN 실패 같은 실제 실패 유형을 누락하면 검토자에게 선택적으로 보여주는 화면처럼 보일 수 있다. 근거: `frontend/src/api/client.ts:84`, `frontend/src/components/TrainingView.tsx:250`, `frontend/src/components/TrainingView.tsx:1793`, `frontend/src/components/TrainingView.tsx:1795`

## 4. 다음 CP에서 꼭 고쳐야 할 P1/P2 UX 문제

| 우선순위 | 문제 | 이유 | 근거 |
|---|---|---|---|
| P1 | 제품 4슬롯 상태를 실제 API 상세 로딩 결과와 연결해야 한다. | 현재는 정적 슬롯이 “사용 중”을 표시하므로, run 상세 fetch 실패나 DB 누락 상황에서도 제품 모델이 정상처럼 보일 수 있다. | `frontend/src/components/TrainingView.tsx:83`, `frontend/src/components/TrainingView.tsx:1803`, `frontend/src/components/TrainingView.tsx:1872` |
| P1 | 상세 데이터가 없을 때 fallback 지표값을 제품 성과처럼 보여주면 안 된다. | `LineModelDetail`과 `BandModelDetail`은 detail이 null이어도 기본 숫자를 표시할 수 있다. 사용자는 실제 저장 지표로 오해한다. | `frontend/src/components/TrainingView.tsx:1166`, `frontend/src/components/TrainingView.tsx:1171`, `frontend/src/components/TrainingView.tsx:1276`, `frontend/src/components/TrainingView.tsx:1284` |
| P1 | 실패 실험 표시 범위에 `failed_nan` 포함 여부를 결정해야 한다. | 현재는 기준 미달만 보여주고 NaN 실패는 목록 로딩 대상이 아니다. 실패 실험을 보여주는 페이지라면 누락으로 보일 수 있다. | `frontend/src/api/client.ts:84`, `frontend/src/components/TrainingView.tsx:1793`, `frontend/src/components/TrainingView.tsx:1795` |
| P2 | 1D 보수적 예측선 슬롯에 “가격 목표가 아님”을 카드 단계에서 알려야 한다. | 상세에는 안전 문구가 있지만, 첫 카드에서는 “예측선”이 목표가처럼 읽힐 수 있다. | `frontend/src/components/TrainingView.tsx:87`, `frontend/src/components/TrainingView.tsx:93`, `frontend/src/components/TrainingView.tsx:1269` |
| P2 | 이전 실험 이름을 사용자용 이름과 검토자용 태그로 분리해야 한다. | `no fundamentals`, `Line Gate`, `Val Total`, `q10/q90` 같은 표현이 목록 첫 줄에 나오면 일반 사용자가 흐름을 놓친다. | `frontend/src/components/TrainingView.tsx:548`, `frontend/src/components/TrainingView.tsx:551`, `frontend/src/components/TrainingView.tsx:566`, `frontend/src/components/TrainingView.tsx:579` |
| P2 | 상세 정보 섹션을 “검토자 상세”로 명확히 라벨링해야 한다. | 접힘 구조는 좋지만, 내부 용어가 많아 일반 사용자에게는 너무 기술적이다. 열기 전부터 검토자용임을 알려야 한다. | `frontend/src/components/TrainingView.tsx:981`, `frontend/src/components/TrainingView.tsx:983`, `frontend/src/components/TrainingView.tsx:777` |
| P2 | 이전 실험 상세 선로딩을 줄이거나 로딩 메시지를 분리해야 한다. | 실험 수가 많아지면 제품 4슬롯 확인까지 늦어질 수 있다. 사용자는 제품 상태와 실험 목록 로딩을 구분하지 못한다. | `frontend/src/components/TrainingView.tsx:1813`, `frontend/src/components/TrainingView.tsx:1818`, `frontend/src/components/TrainingView.tsx:1889` |

## 5. 용어 개선 제안

| 현재 표현 | 제안 표현 | 우선순위 | 이유 |
|---|---|---|---|
| 보수적 예측선 | 방어적 예측 참고선 | P2 | 목표가 오해를 줄인다. |
| AI 밴드 | AI 위험 범위 | P2 | 밴드가 수익 기회가 아니라 위험 범위임을 강화한다. |
| no fundamentals | 재무 피처 제외 실험 | P2 | 영어 실험명을 사용자 언어로 바꾼다. |
| Line Gate | 예측선 기준 선택 실험 | P2 | 내부 selector 느낌을 줄인다. |
| Val Total | 전체 검증 손실 기준 실험 | P2 | 선택 기준을 설명형으로 바꾼다. |
| CI 집계 기준 | 채널 집계 방식 | P2 | 약어 장벽을 낮춘다. |
| 빠른 CI 목표 | 빠른 채널 목표 | P2 | 약어 해석 부담을 줄인다. |
| 밴드 손실 가중치 | 밴드 학습 가중치 | P2 | 의미는 유지하되 덜 수식적으로 보인다. |

## 6. 좋은 점

제품 모델과 이전 실험의 정보 위계가 좋아졌다. 제품 4슬롯을 먼저 보여주고, 이전 실험은 아래에서 접어두는 구조는 일반 사용자와 검토자를 동시에 배려한다. 근거: `frontend/src/components/TrainingView.tsx:1869`, `frontend/src/components/TrainingView.tsx:1903`

Line/Band 분리도 이전보다 훨씬 설명적이다. 예측선은 방향·순위 판단, 밴드는 변동 범위라는 역할 구분이 코드 문구에 들어가 있다. 근거: `frontend/src/components/TrainingView.tsx:1177`, `frontend/src/components/TrainingView.tsx:1291`

과장 방지 문구도 좋다. “투자 조언이 아니라 보조 판단선”, “AI 밴드는 수익 목표가 아니라 위험 범위” 같은 문구는 금융 UX 신뢰에 필요하다. 근거: `frontend/src/components/TrainingView.tsx:1269`, `frontend/src/components/TrainingView.tsx:1383`

## 7. 코드 수정 금지 확인

이번 작업에서 코드 파일, 스타일 파일, 테스트 파일은 수정하지 않았다. 새로 작성한 파일은 이 보고서 `docs/product_ai_model_page_review_report.md` 하나뿐이다.

읽기 전용으로 확인한 주요 파일은 `frontend/src/components/TrainingView.tsx`, `frontend/src/components/AppShell.tsx`, `frontend/src/components/DashboardPage.tsx`, `frontend/src/api/client.ts`, `frontend/src/app/globals.css`다.
