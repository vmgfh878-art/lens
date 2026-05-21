CP83-4-P는 AI 모델 화면의 이전 실험 상세를 제품 모델 대비 비교 리포트로 바꾸는 CP다.

## 1. 목표

AI 모델 화면에서 이전 실험을 클릭했을 때 단순히 `기준 미달`처럼 보이지 않게 하고, 현재 제품 모델보다 무엇이 부족했는지와 무엇이 더 나았는지를 같이 설명하도록 수정했다.

기준 제품 모델은 다음 두 개로 고정했다.

- 1D 보수적 예측선 v1: `patchtst-1D-efad3c29d803`
- 1D AI 밴드 v1: `cnn_lstm-1D-d0c780dee5e8`

1W 제품 모델은 아직 준비 중이므로, 1W 실험은 제품 대비 우열 비교처럼 과장하지 않고 `제품 기준 미확정`으로만 처리한다.

## 2. 변경 파일

- `frontend/src/components/TrainingView.tsx`
- `frontend/src/app/globals.css`
- `docs/cp83_4_product_experiment_comparison_report.md`

## 3. 이전 실험 상세 구조 변경

이전 실험 상세 패널을 아래 구조로 바꿨다.

- 실험 요약
- 역할: 예측선 실험 또는 밴드 실험
- 실험에서 바꾼 것
- 제품 모델 대비 부족했던 점
- 제품 모델 대비 좋았던 점
- 비교 지표 표
- 최종 판단
- 다음 확인 방향
- 전문가용 상세

제품 모델 상세와 이전 실험 상세가 같은 메인 상세 패널을 쓰도록 유지했다. 이전 실험을 클릭하면 상세 패널이 해당 실험의 비교 리포트로 바뀌고, 제품 모델 슬롯을 다시 클릭하면 제품 모델 설명으로 돌아간다.

## 4. 비교 지표

예측선 실험은 1D 보수적 예측선 v1과 비교한다.

- 순위 상관
- 상위-하위 수익 차
- 수수료 반영 샤프
- 위험 오판율
- 큰 위험 오판율
- 큰 하락 포착률

밴드 실험은 1D AI 밴드 v1과 비교한다.

- 실제 포함률
- 포함률 오차
- 하단 이탈률
- 상단 이탈률
- 비대칭 구간 점수
- 평균 밴드 폭
- 밴드 폭 반응도
- 하방 폭 반응도

표는 `항목 / 제품 모델 / 이 실험 / 차이 / 해석` 구조로 만들었다. `계산 불가`는 표시하지 않고, 제품 모델과 실험 값이 모두 있는 지표만 비교 표에 표시한다.

## 5. 자연어 비교 설명

비교 결과를 `better`, `worse`, `similar`, `neutral`로 나눈 뒤 자연어 문장을 만든다.

- 예측선에서 순위 상관이 낮으면 방향/순위 구분력이 약하다고 설명한다.
- 상위-하위 수익 차가 낮으면 좋은 종목과 나쁜 종목을 나누는 힘이 약하다고 설명한다.
- 위험 오판율이 높으면 위험 구간을 안전하다고 본 비율이 높다고 설명한다.
- 큰 하락 포착률이 낮으면 큰 하락을 포착하는 힘이 약하다고 설명한다.
- 밴드에서 포함률 오차가 크면 목표 포함률과 실제 포함률 차이가 크다고 설명한다.
- 하단 이탈률이 높으면 하방 위험을 충분히 덮지 못했다고 설명한다.
- 밴드 폭 반응도가 낮으면 변동성이 커질 때 밴드가 같이 넓어지는 반응이 약하다고 설명한다.

제품 모델보다 나은 값이 있으면 `제품 모델 대비 좋았던 점` 영역에 같이 표시한다. 한 지표가 더 좋더라도 핵심 비교 지표가 약하면 최종 판단에서 왜 제품 모델이 아닌지 설명한다.

## 6. 이전 실험 목록 이름 정리

이전 실험 목록이 `PatchTST 예측선 실험`처럼 반복되어 보이지 않도록 조건 기반 이름을 보강했다.

예측선 실험 이름 기준:

- horizon이 20이면 `PatchTST h20 예측선 실험`
- feature set이 다르면 `PatchTST h5 no fundamentals 예측선`
- patch 길이가 길면 `PatchTST h5 긴 패치 예측선`
- stride가 작으면 `PatchTST h5 Dense 예측선`
- seq_len이 60이고 epoch가 길면 `PatchTST h5 seq60 장기 학습 예측선`
- checkpoint 선택 기준이 line gate면 `PatchTST h5 Line Gate 예측선`
- checkpoint 선택 기준이 val total이면 `PatchTST h5 Val Total 예측선`

밴드 실험 이름 기준:

- q_low/q_high가 있으면 `CNN-LSTM q15 가격·변동성 AI 밴드`처럼 표시
- 가격·변동성·거래량 feature set이면 해당 데이터 구성을 이름에 반영

동일한 이름과 그룹으로 중복되는 실험은 한 번만 노출되게 정리했다.

## 7. 숨김 정책

- NaN 실패는 기본 UI에 노출하지 않는다.
- composite는 제품 UI와 이전 실험 기본 목록에 노출하지 않는다.
- 제품 기준이나 비교 카드가 부족한 실험은 이전 실험 목록에서 숨긴다.
- 1W 제품 모델은 준비 중 슬롯으로만 보여주며, 1W 실험은 제품 대비 비교처럼 과장하지 않는다.

## 8. CSS 변경

`globals.css`에 비교 표 전용 스타일을 추가했다.

- `.comparison-table-wrap`
- `.comparison-table`

긴 해석 문장이 넘치지 않도록 최소 폭, 줄바꿈, 숫자 컬럼 정렬을 분리했다. 모바일에서는 표가 패널 안에서 가로 스크롤되도록 두어 전체 화면 레이아웃이 깨지지 않게 했다.

## 9. 검증 결과

### 빌드

명령:

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 통과
- 최초 샌드박스 실행은 Next.js worker `spawn EPERM`으로 실패
- 승인 권한으로 재실행 후 컴파일, 타입 검사, 정적 페이지 생성 통과

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
- legacy composite는 `LEGACY_OK`로만 확인

### 브라우저

확인 URL:

- `http://127.0.0.1:3000`

확인 결과:

- AI 모델 화면 진입 확인
- 제품 모델 4슬롯 확인
- 예측선 실험 보기 버튼 확인
- 예측선 실험 클릭 시 제품 대비 비교 상세로 전환 확인
- 밴드 실험 보기 버튼 확인
- 밴드 실험 클릭 시 제품 대비 비교 상세로 전환 확인
- 제품 모델 슬롯 재클릭 시 제품 모델 상세로 복귀 확인
- `계산 불가` 문구 미노출 확인
- composite 기본 노출 없음 확인
- console error/warn 0건 확인

## 10. 남은 아쉬운 점

- 일부 과거 run은 config가 거의 같아 이름만으로 완전히 실험 의도를 설명하기 어렵다.
- run별 실험 메모가 별도 저장되어 있지 않아 `실험에서 바꾼 것`은 config와 metric에서 확인 가능한 내용으로만 제한했다.
- 제품 대비 비교 표는 현재 test/val metric 중 API에서 찾을 수 있는 값을 중심으로 만든다. 더 깊은 비교를 하려면 실험 리포트 JSON 또는 별도 read-only 실험 메모 API가 있으면 좋다.

## 11. 수정하지 않은 것

- 모델 학습 실행 없음
- inference 실행 없음
- DB 쓰기 없음
- fake data 생성 없음
- composite 제품 기본 복구 없음
- 신규 데이터셋/리포트 페이지 추가 없음
