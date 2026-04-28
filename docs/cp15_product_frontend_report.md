# CP15-P 프론트 앱 구조 전환 보고서

## 1. 변경 파일

- `frontend/src/api/client.ts`
- `frontend/src/app/globals.css`
- `frontend/src/components/DashboardPage.tsx`
- `frontend/src/components/Chart.tsx`
- `frontend/src/components/AppShell.tsx`
- `frontend/src/components/StockView.tsx`
- `frontend/src/components/BacktestView.tsx`
- `frontend/src/components/TrainingView.tsx`
- `frontend/src/components/MetricCard.tsx`
- `frontend/src/components/LayerToggle.tsx`
- `frontend/src/components/ModelSelector.tsx` 제거

## 2. 화면 구조

- 첫 화면은 hero 소개가 아니라 `주식 보기` 화면이다.
- 왼쪽 내비게이션으로 `주식 보기`, `백테스트`, `모델 학습` 3개 화면을 전환한다.
- 모바일 폭에서는 왼쪽 내비게이션이 상단 탭 형태로 접힌다.

## 3. API client 추가 내용

- `fetchAiRuns`
- `fetchAiRun`
- `fetchRunEvaluations`
- `fetchRunBacktests`
- `fetchPrediction`에 `runId` 옵션 추가

기본 API 주소는 기존처럼 `NEXT_PUBLIC_BACKEND_URL`을 우선 사용하고, 없으면 `http://localhost:8000`을 사용한다.

## 4. 1M 처리 방식

- `1M`은 `fetchPrices`로 가격 차트를 정상 조회한다.
- `1M`에서는 `fetchPrediction`을 호출하지 않는다.
- AI 밴드와 보수적 예측선 토글은 비활성 처리한다.
- 차트에는 가격 데이터만 전달되므로 AI layer가 꺼져도 차트가 깨지지 않는다.

## 5. target type 숨김 처리

- `주식 보기` 화면에서는 target type, raw, volatility-normalized 같은 용어를 노출하지 않는다.
- 모델 선택 UI도 제거했고, 주식 보기 화면은 PatchTST 기본 호출만 사용한다.
- target type은 `모델 학습`의 read-only run 목록에서만 확인한다.

## 6. 백테스트/모델 학습 연결 수준

- 백테스트 화면은 latest completed PatchTST run을 기준으로 `fetchRunBacktests`를 호출한다.
- 백테스트 결과가 없으면 빈 상태를 표시한다.
- equity curve와 drawdown chart는 자리만 잡았다.
- 모델 학습 화면은 `completed`와 `failed_nan` status 탭을 제공한다.
- run 상세는 `fetchAiRun`, ticker/asof 평가 테이블은 `fetchRunEvaluations`로 읽는다.
- 학습 실행 버튼과 파라미터 수정 UI는 만들지 않았다.

## 7. build 결과

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 막혔고, 승인된 실행에서 빌드가 통과했다.

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
Route (app) / 77.5 kB, First Load JS 164 kB
```

dev 서버도 확인했다.

```text
http://127.0.0.1:3000
HTTP/1.1 200 OK
```

## 8. 에이전트 질문 답변

1. 현재 프론트에서 실제 CP14-P API를 바로 호출 가능한가?
   - 가능하다. 백엔드가 `NEXT_PUBLIC_BACKEND_URL` 또는 기본값 `http://localhost:8000`에서 실행 중이면 바로 호출한다.
2. run 목록이 비어 있을 때 어떤 empty state를 보여주는가?
   - 모델 학습은 `선택한 상태의 run이 없습니다.`를 보여준다. 백테스트는 latest completed run 또는 저장된 백테스트가 없다는 빈 상태를 보여준다.
3. latest completed run을 어떤 기준으로 선택하는가?
   - `fetchAiRuns({ modelName: "patchtst", status: "completed", limit: 20 })` 결과의 첫 번째 run을 선택한다. CP14 API가 `created_at desc`로 정렬하므로 가장 최근 completed run이다.
4. 1M에서 AI layer가 꺼질 때 차트가 깨지지 않는가?
   - 깨지지 않는다. 1M에서는 예측 API를 호출하지 않고 가격 데이터만 차트에 전달한다.
5. 모바일 폭에서 왼쪽 내비게이션은 어떻게 접히는가?
   - 860px 이하에서 사이드바가 상단 탭 내비게이션으로 전환된다.
6. 차트 overlay는 이번 CP에서 구현했는가, 아니면 props 자리만 열었는가?
   - 이번 CP에서는 props 자리만 열었다. 실제 AI band/line overlay 렌더링은 CP16-P에서 연결하면 된다.

## 9. 남은 TODO

- CP16-P에서 AI 밴드와 보수적 예측선 overlay를 실제 차트 시리즈로 연결한다.
- 백테스트 equity curve와 drawdown chart용 저장 데이터가 정해지면 차트로 연결한다.
- 모델 학습 화면의 W&B/Optuna 요약은 저장 필드가 확정되면 표시 항목을 넓힌다.
- 주요 화면 스크린샷은 dev 서버가 응답 중이므로 브라우저 확인 단계에서 바로 촬영 가능하다.
