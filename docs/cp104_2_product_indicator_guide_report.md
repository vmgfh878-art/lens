# CP104-2-P 지표 가이드 재구성 보고서

## 목적

CP104 리포트 화면이 프로젝트 진행 보고서처럼 보이고, 사용자가 실제로 차트에서 봐야 할 보수적 예측선과 AI 밴드의 의미가 뒤로 밀린 문제를 수정했다. 이번 작업은 “Lens 리포트”를 “AI 지표 가이드” 성격으로 바꾸는 데 집중했다.

## 변경 사항

### 지표 가이드 중심 재구성

- 사이드바 메뉴명을 `Lens 리포트`에서 `지표 가이드`로 변경했다.
- 첫 화면 제목을 `예측선은 방향을 보고, 밴드는 흔들림을 봅니다.`로 바꿨다.
- 현재 상태 카드 4개를 제거하고, 보수적 예측선과 AI 밴드 설명을 최상단에 배치했다.
- 보수적 예측선은 “앞으로 유리한 방향인지 보는 선”으로 설명했다.
- AI 밴드는 “가격이 흔들릴 수 있는 범위”로 설명했다.
- 사용자가 실제로 해석할 수 있도록 세 가지 상황을 추가했다.
  - 예측선이 위쪽이고 밴드가 좁을 때
  - 예측선은 괜찮지만 밴드가 넓을 때
  - 예측선이 약하고 밴드 하단이 낮을 때

### 내부 보고서성 문구 축소

- composite, CP 진행 과정, 데이터 provider 전환, Supabase 상세 구조를 기본 화면에서 제거하거나 축소했다.
- “둘을 합치지 않는 이유”는 composite 같은 내부 용어 대신 방향 지표와 흔들림 지표의 차이로 설명했다.
- 데이터 설명은 조정 가격 기준과 현재 1D 중심 상태만 남겼다.
- local parquet / Supabase thin DB 설명은 `조금 더 기술적인 설명` 접기 영역으로 이동했다.

### 모델 설명 정리

- PatchTST는 보수적 예측선에 쓰는 모델로 설명했다.
- CNN-LSTM은 AI 밴드에 쓰는 모델로 설명했다.
- TiDE는 제품 기본이 아니라 비교 실험 모델로 짧게 정리했다.

### 주식 보기 URL 방어

- `NEXT_PUBLIC_BACKEND_URL`에 `127.0.0.1:8000`처럼 프로토콜 없는 값이 들어와도 `http://127.0.0.1:8000`으로 정규화되도록 보강했다.
- 잘못된 URL 값이 들어오면 기본값 `http://127.0.0.1:8000`으로 fallback한다.
- 스크린샷의 `Failed to construct 'URL': Invalid URL` 재발 가능성을 줄였다.

## 검증

- `npm run build`: 통과
- `scripts/check_demo_readiness.ps1`: 실행 완료
  - health: OK
  - CORS: OK
  - frontend: OK
  - stock search: OK
  - LM/BM run, prediction, evaluation, history: OK
  - AAPL 1D/1M 가격: 현재 DB 상태 기준 404 유지

## 남은 이슈

- AAPL 가격 404는 이번 지표 가이드 화면 문제가 아니라 현재 데이터/API 상태 문제다.
- 주식 보기 화면은 URL 정규화 방어를 넣었지만, 가격 데이터가 없을 때의 empty state는 데이터 루프 정리 이후 다시 확인해야 한다.
- 리포트 화면은 제품 설명 중심으로 바꿨고, 발표용 데이터 전환 서사는 별도 문서에서 다루는 편이 맞다.

## 수정 파일

- `frontend/src/api/client.ts`
- `frontend/src/components/AppShell.tsx`
- `frontend/src/components/ReportView.tsx`
- `frontend/src/app/globals.css`
- `docs/cp104_2_product_indicator_guide_report.md`
