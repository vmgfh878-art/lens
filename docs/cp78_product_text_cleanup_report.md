CP78-P 한글 깨짐 복구 및 제품 용어 정리 보고서

## 1. 깨진 한글 복구 대상

- `frontend/src`, `README.md`를 UTF-8 기준으로 확인했다.
- 최종 문자열 검증에서 지정 패턴은 0건이다.
  - `遺`, `諛`, `紐`, `二쇱`, `?덉`, `?쒕`, `?꾨`, `�`
- README는 상단/개요/실행 부분에 깨진 한글이 없어 수정하지 않았다.
- 프론트 사용자 노출 문구는 `StockView`, `TrainingView`, `BacktestView`, `Chart` 중심으로 제품 문구를 다시 고정했다.

## 2. 내부 용어 제거/대체 목록

- 주식 보기 기본 화면
  - `patchtst-1D-efad3c29d803` 직접 노출 → `예측선 모델 v1`
  - `cnn_lstm-1D-d0c780dee5e8` 직접 노출 → `AI 밴드 모델 v1`
  - `product layer run` → `제품 후보`
  - `legacy demo artifact` → `이전 실험 결과`
  - `completed_line_watch` → `예측선 후보`
  - `band run`, `run 모델`, `Line 판정` 기본 노출 제거
- 모델 학습 화면
  - `run 콘솔` → `모델 후보`
  - `Line 후보` → `예측선 후보`
  - `Band 후보` → `AI 밴드 후보`
  - `Legacy / 기타` → `이전 실험 / 기타`
  - `role` → `역할`
  - `feature_set` → `사용 데이터`
  - `wandb_status` → `실험 추적`
  - `run_id` → `실행 ID`
  - `baseline 비교 준비 필요` → `기준 모델 비교는 준비 중입니다.`

## 3. 예측 출처 패널 변경 내용

- 기본 표시를 run ID 중심에서 사용자 친화적 출처로 변경했다.
- 표시 항목:
  - 예측선: `예측선 모델 v1`
  - AI 밴드: `AI 밴드 모델 v1`
  - 기준일
  - 예측 기간
  - 예측선 상태: `예측선 후보` 또는 `예측 저장 대기`
  - 밴드 상태: `위험 범위 후보` 또는 `예측 저장 대기`
- 설명 문구:
  - 예측선: `향후 수익 방향을 참고하기 위한 AI 예측선입니다.`
  - AI 밴드: `예상 변동 범위를 보여주는 리스크 참고 지표입니다.`
  - 보수적 예측선: `하방 위험을 더 조심스럽게 보기 위한 기준선입니다.`
- 실행 ID는 기본 노출하지 않고 `상세 정보` 접힘 영역에만 남겼다.

## 4. 1D/1W/1M 문구 정리

- 1D: 저장된 예측선과 AI 밴드가 있으면 표시 가능.
- 1W: `주간 AI 예측은 준비 중입니다.`
- 1M: `월간 화면은 현재 가격 전용입니다.`
- 1M 브라우저 확인 결과, 제품 후보 run ID는 기본 화면에 노출되지 않았다.

## 5. 남은 용어/레이아웃 문제

- 모델 학습 화면에는 실행 ID, 실험 추적, 체크포인트 등 연구/검수에 필요한 용어가 남아 있다.
- 실행 로그 파일은 아직 프론트에서 직접 읽지 못하며, 읽기 전용 로그 API가 필요하다.
- metric 원천 키는 코드와 API 계약에는 남아 있으나, 주요 카드 라벨은 한국어로 병기했다.

## 6. 검증 결과

- `npm run build`
  - 샌드박스 내부 첫 실행은 Next.js worker spawn 제한으로 `EPERM` 실패.
  - 승인된 실행으로 재검증했고 빌드 통과.
- `powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - health, CORS, frontend 200, AAPL 1D 가격, indicators, 1M 가격, stock search 모두 OK.
  - LM/BM run, prediction, evaluation, history, band width 모두 OK.
  - composite는 `LEGACY_OK`.
- 브라우저 확인
  - AAPL 1D: 첫 진입 설명, `예측선 모델 v1`, `AI 밴드 모델 v1`, `보수적 예측선`, `예측 기간` 확인.
  - 1M: `월간 화면은 현재 가격 전용입니다.` 확인, run ID 기본 노출 없음, 백엔드 오류 배너 없음.
  - 모델 후보 화면: `모델 후보`, `예측선 후보`, `AI 밴드 후보`, 기준 모델 준비 문구 확인.
  - 브라우저 console error/warn: 0건.
- 문자열 검증
  - `frontend/src`, `README.md` 기준 지정 깨짐 패턴 0건.

## 7. 수정하지 않은 것

- 모델 학습, inference 실행, DB 쓰기, fake data 생성은 하지 않았다.
- API/DB schema는 수정하지 않았다.
- README는 깨진 문자열이 없어 수정하지 않았다.
- 대규모 레이아웃 리워크, 다크모드, 백테스트 전략 추가는 하지 않았다.
