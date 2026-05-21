CP83-5-P는 AI 모델 화면의 상세 영역을 디버그 패널이 아니라 모델 설명 페이지처럼 정리하는 CP다.

## 1. 목표

AI 모델 화면에서 `전문가용 상세`처럼 보이던 영역을 `상세 정보`로 바꾸고, 모델 구조와 설정, 평가 지표, 저장 정보를 사용자가 이해하기 쉬운 순서로 정리했다. 상세 정보는 기본 접힘 상태로 두어 사용자가 필요할 때만 열어보게 했다.

이번 CP는 새 기능 추가가 아니라 화면 polish다. 모델 학습, inference 실행, DB 쓰기, fake data 생성은 하지 않았다.

## 2. 변경 파일

- `frontend/src/components/TrainingView.tsx`
- `frontend/src/app/globals.css`
- `docs/cp83_5_product_model_detail_page_report.md`

## 3. 문구 변경

제거한 사용자 노출 문구:

- `전문가용 상세`
- `전문가용 정보`

대체 문구:

- `상세 정보`
- `학습 설정`
- `데이터 설정`
- `모델 구조`
- `손실/평가 설정`
- `평가 지표`
- `저장 정보`

이전 실험의 다음 확인 방향 문구도 `전문가용 정보에서 확인`이 아니라 `상세 정보에서 확인`으로 바꿨다.

## 4. 공통 상세 정보 컴포넌트

제품 모델과 이전 실험이 같은 상세 정보 컴포넌트를 사용하게 정리했다.

적용 대상:

- 1D 보수적 예측선 v1
- 1D AI 밴드 v1
- 예측선 이전 실험
- 밴드 이전 실험

제품 모델은 먼저 제품 설명과 목표 대비 평가 카드를 보여주고, 그 아래에 상세 정보를 둔다. 이전 실험은 제품 모델 대비 비교 리포트를 먼저 보여주고, 그 아래에 같은 상세 정보 구조를 둔다.

상세 정보는 기본으로 펼치지 않는다. 요약 줄에는 `상세 정보`, `모델 설정·평가 지표·저장 정보`, `열기/접기` 상태만 보이고, 내부 설정 카드는 사용자가 직접 열었을 때만 보인다.

## 5. 상세 정보 구성

상세 정보는 다음 순서로 표시한다.

1. 모델 이름 / 역할 / 버전 / 상태
2. 이 모델이 무엇을 예측하는지
3. 모델 구조 설명
4. 학습 설정
5. 데이터 설정
6. 모델 구조 파라미터
7. 손실/평가 설정
8. 추가 설정
9. 평가 지표
10. 저장 정보

PatchTST, CNN-LSTM, TiDE는 각각 간단한 구조 설명을 제공한다.

## 6. 파라미터 표시 방식

config 값을 고정 목록으로만 보지 않고, `config_summary`와 run 기본 필드를 합쳐 가능한 값을 동적으로 표시한다.

그룹:

- 학습 설정: `epochs`, `batch_size`, `lr`, `learning_rate`, `weight_decay`, `dropout`, `device`, `amp_dtype`, `seed`, `best_epoch`, `best_val_total`
- 데이터 설정: `timeframe`, `horizon`, `seq_len`, `feature_set`, `feature_version`, `n_features`, `target`, `line_target_type`, `band_target_type`, `ci_aggregate`, `ci_target_fast`
- 모델 구조: `model_name`, `model_ver`, `patch_len`, `patch_stride`, `stride`, `d_model`, `n_heads`, `n_layers`, `hidden_size`, `kernel_size`, `fp32_modules`, `band_mode`
- 손실/평가 설정: `alpha`, `beta`, `lambda_line`, `lambda_band`, `q_low`, `q_high`, `checkpoint_selection`
- 저장 정보: `run_id`, `status`, `checkpoint 존재 여부`, `wandb_status`, `wandb_run_id`, `created_at`

값 표시 규칙:

- `null`, `undefined`, 빈 문자열은 숨긴다.
- legacy composite 관련 값은 숨긴다.
- 긴 내부 path인 `checkpoint_path`는 직접 노출하지 않고 체크포인트 저장 여부만 표시한다.
- boolean은 `사용` / `미사용`으로 표시한다.
- `feature_set`은 `사용 피처 묶음`으로 라벨을 바꾸고, 값도 `전체 피처`, `가격·변동성·거래량`처럼 바꾼다.
- `checkpoint_selection`은 `체크포인트 선택 기준`으로 표시한다.
- run_id는 저장 정보 아래에서 작은 monospace 값으로 표시한다.

## 7. 평가 지표 표시

raw metric dump 대신 `평가 지표` 카드로 정리했다.

각 지표는 다음을 보여준다.

- 지표 이름
- 목표 또는 해석 기준
- test 값
- val 값

예측선 지표는 순위 상관, 상위-하위 수익 차, 위험 오판율, 큰 하락 포착률, 수수료 반영 샤프를 보여준다.

밴드 지표는 목표 포함률, 실제 포함률, 포함률 오차, 하단/상단 이탈률, 평균 밴드 폭, 비대칭 구간 점수, 밴드 폭 반응도, 하방 폭 반응도를 보여준다.

## 8. 스타일 변경

`globals.css`에 상세 정보 전용 스타일을 추가했다.

- `.model-run-details`
- `.model-run-details__header`
- `.detail-status-card`
- `.model-detail-section`
- `.detail-field-grid`
- `.detail-field`
- `.detail-metric-grid`
- `.detail-field__mono`

긴 파라미터가 박스 밖으로 튀어나가지 않도록 `overflow-wrap`, `text-overflow`, `auto-fit` 그리드를 적용했다. 좁은 화면에서는 카드가 자동으로 줄바꿈된다. 상세 정보는 기본 접힘 `<details>` 구조로 바꿨다.

## 9. 검증 결과

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
- 1D 보수적 예측선 상세 정보 확인
- 1D AI 밴드 상세 정보 확인
- 예측선 이전 실험 상세 정보 확인
- 밴드 이전 실험 상세 정보 확인
- `상세 정보`, `학습 설정`, `데이터 설정`, `모델 구조`, `평가 지표`, `저장 정보` 노출 확인
- `전문가용` 문구 미노출 확인
- `계산 불가` 문구 미노출 확인
- console error/warn 0건 확인

## 10. 남은 아쉬운 점

- 현재 API가 `config_summary` 중심이라 원본 config의 모든 세부값을 보여주지는 않는다.
- 실험별 사람이 쓴 설명 메모가 없기 때문에 일부 실험의 의도는 config 값에서 추론 가능한 범위까지만 표시한다.
- 모델 구조 설명은 현재 화면 문구 기반이며, 나중에 별도 라이브러리/모델 설명 페이지가 생기면 더 자세히 분리할 수 있다.

## 11. 수정하지 않은 것

- 모델 학습 없음
- inference 실행 없음
- DB 쓰기 없음
- fake data 생성 없음
- composite 제품 기본 복구 없음
- 백테스트 추가 변경 없음
- 신규 데이터셋/리포트 페이지 추가 없음
