CP84-P는 AI 모델 화면에서 실제 저장된 run/evaluation이 없는 값이 제품 성과처럼 보이지 않게 고친 신뢰도 수정 CP다.

## 1. 목표

AI 모델 화면의 제품 슬롯 상태와 평가 카드를 실제 API 결과에 기반하도록 바꿨다. 제품 성과처럼 보이는 fallback 숫자를 제거하고, 저장된 metric이 없으면 `저장된 평가 없음`으로 표시한다.

## 2. 변경 파일

- `frontend/src/components/TrainingView.tsx`
- `frontend/src/app/globals.css`
- `docs/cp84_product_ai_model_trust_fix_report.md`

## 3. 제품 슬롯 상태 변경

기존에는 제품 4슬롯의 상태가 정적으로 들어가 있었다.

변경 후 상태 기준:

- `사용 중`: 제품 후보 run 상세가 있고, 저장된 평가 metric이 확인된 경우
- `데이터 확인 중`: 제품 후보 run 상태를 불러오는 중
- `연결 필요`: 제품 후보 run_id는 있으나 상세 또는 저장된 평가 metric이 확인되지 않은 경우
- `준비 중`: 1W처럼 제품 후보 run_id가 없는 경우

현재 확인 결과:

- 1D 보수적 예측선: `사용 중`
- 1D AI 밴드: `사용 중`
- 1W 보수적 예측선: `준비 중`
- 1W AI 밴드: `준비 중`

## 4. fallback metric 제거

제품 상세의 목표 대비 평가 카드에서 아래처럼 하드코딩되어 있던 fallback 숫자를 제거했다.

- 예측선: `0.0406`, `0.0063`, `39.3%`, `60.6%`, `0.1749`
- 밴드: `70%`, `71.4%`, `1.4%p`, `15.0%`, `13.6%`, `0.0607`, `0.1244`, `0.3724`, `0.0673`

변경 후 평가 카드는 `test_metrics` 또는 `val_metrics`에 실제로 저장된 값이 있을 때만 생성된다. 값이 없으면 숫자 대신 `저장된 평가 없음`을 보여준다.

## 5. 신뢰 문구 추가

평가 카드 위에 다음 문구를 작게 추가했다.

`이 값은 저장된 평가 결과 기준입니다. 평가가 없으면 성능을 판단하지 않습니다.`

저장된 평가가 없을 때는 다음처럼 표시한다.

`저장된 평가 없음`

제품 모델의 설명도 저장 metric이 없는 경우 성능을 판단하지 않는 문구로 바뀌도록 했다.

## 6. 이전 실험 실패 처리

이번 화면에서는 `failed_nan`을 일반 이전 실험 목록에 섞지 않는다. 현재 `TrainingView`는 completed와 failed_quality_gate만 조회하며, composite는 legacy 판정으로 계속 숨긴다.

시스템 오류성 실패를 제품 설명 화면에 섞지 않는 정책을 유지했다. 필요하면 별도 하단 접힘 영역인 `시스템 실패 로그`로 분리할 수 있지만, 이번 CP에서는 새 영역을 추가하지 않았다.

## 7. CSS 변경

추가 스타일:

- `status-pill--warning`: `연결 필요` 상태용
- `trust-note`: 저장된 평가 기준 안내 문구
- `empty-state--compact`: metric이 없을 때 작은 empty state

## 8. 검증 결과

### 빌드

명령:

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 통과
- 컴파일, 타입 검사, 정적 페이지 생성 통과

### readiness

명령:

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과:

- backend health: OK
- CORS: OK
- frontend 200: OK
- AAPL 1D 가격: OK
- indicators: OK
- 1M price-only: OK
- stock search: OK
- LM run/prediction/evaluation/history: OK
- BM run/prediction/evaluation/history: OK
- band width 계산 가능: OK
- legacy composite는 제품 기본이 아니라 `LEGACY_OK`로만 확인

### 브라우저

확인 URL:

- `http://127.0.0.1:3000`

확인 결과:

- AI 모델 화면 진입 확인
- 1D 보수적 예측선 슬롯 `사용 중` 확인
- 1D AI 밴드 슬롯 `사용 중` 확인
- 1W 보수적 예측선 슬롯 `준비 중` 확인
- 1W AI 밴드 슬롯 `준비 중` 확인
- 저장된 평가 기준 안내 문구 확인
- console error/warn 0건 확인

화면에 보이는 수치가 남아 있는 것은 API에 저장된 metric이 존재하기 때문이다. 소스 코드의 제품 fallback 숫자는 제거했다.

## 9. 남은 과제

- 저장된 evaluation row 자체를 별도 API로 직접 확인해 제품 슬롯 상태에 반영하려면 `fetchRunEvaluations`를 함께 쓰는 방식도 가능하다.
- 현재는 run detail의 `test_metrics` / `val_metrics`를 저장된 평가 metric으로 판단한다.
- failed_nan을 별도 `시스템 실패 로그` 접힘 영역으로 보여줄지는 다음 UX 판단이 필요하다.

## 10. 수정하지 않은 것

- 모델 학습 없음
- inference 실행 없음
- DB 쓰기 없음
- fake data 생성 없음
- composite 제품 기본 복구 없음
- 백테스트 추가 변경 없음
