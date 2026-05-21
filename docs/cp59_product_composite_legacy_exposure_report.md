CP59-P는 CP57-M에서 composite를 Phase 1 본류에서 내린 결정을 제품/백엔드 조회 경로에 반영한 CP다.

# 1. 목표

`line_band_composite`가 최신 completed 모델이나 제품 기본 후보처럼 보이지 않도록 차단했다.

이번 CP에서는 새 학습, AI 모델 구조 변경, AI 학습/추론 스크립트 수정, DB schema 변경, fake data 생성, legacy composite run 삭제를 하지 않았다.

# 2. Composite를 제품 기본 후보에서 제외한 경로

백엔드:

- `backend/app/repositories/ai_repo.py`
  - `is_legacy_composite_run()` 추가
  - `model_name=line_band_composite`
  - `role=composite_model`
  - `deprecated_for_phase1_product_contract=true`
  - 위 조건에 걸리는 run을 legacy composite로 판정
  - `fetch_model_runs(..., include_legacy=False)` 기본값에서는 legacy composite 제외

- `backend/app/routers/v1/ai.py`
  - `GET /api/v1/ai/runs`에 `include_legacy=false` 기본 옵션 추가
  - 기본 목록에서 legacy composite 재필터
  - `include_legacy=true` 요청 때만 legacy composite 포함
  - 응답에 `role`, `deprecated_for_phase1_product_contract`, `indicator_layer_replacement`, `is_legacy` 포함

프론트:

- `frontend/src/components/StockView.tsx`
  - 제품 기본 AI run 후보에서 `line_band_composite` 제거
  - 기본 후보는 PatchTST 중심으로 제한

- `frontend/src/components/TrainingView.tsx`
  - 일반 completed run 목록에서 legacy composite 제외
  - `includeLegacy=false`로 조회

- `frontend/src/components/BacktestView.tsx`
  - CP46에서 들어간 composite 후보 기준을 제품 기본 기준에 맞춰 PatchTST 중심으로 되돌림

# 3. Legacy Composite Run 보존 여부

기존 composite run은 삭제하지 않았다.

확인한 동작:

```text
GET /api/v1/ai/runs?model_name=&status=completed&timeframe=1D&limit=5
```

기본 응답:

- `line_band_composite` 제외
- `cnn_lstm`, `patchtst` 등 non-legacy run만 반환

명시 응답:

```text
GET /api/v1/ai/runs?model_name=&status=completed&timeframe=1D&limit=5&include_legacy=true
```

결과:

- `composite-1D-3a44b5e51ed2`
- `composite-1D-a0786769a07a`
- 두 run 모두 `is_legacy=true`로 표시

# 4. StockView 변경 내용

주식 보기 화면은 더 이상 `line_band_composite`를 제품 기본 AI 모델 후보로 선택하지 않는다.

변경 내용:

- `AI_RUN_MODEL_CANDIDATES`를 `patchtst`로 제한
- `fetchAiRuns(..., includeLegacy: false)` 사용
- legacy prediction이 명시적으로 들어오는 경우 `legacy demo artifact`로 표시
- composite를 최종 제품 모델처럼 설명하지 않도록 provenance 문구 변경

아직 overlay bundle API는 없다. 따라서 line layer와 band layer를 완전히 분리해 조립하는 구조는 다음 CP로 남겼다.

# 5. TrainingView 변경 내용

모델 학습 화면의 일반 completed run 목록에서 composite를 제외했다.

변경 내용:

- `TRAINING_RUN_MODELS`를 `patchtst`로 제한
- `includeLegacy: false`로 run 목록 조회
- `run.is_legacy`가 true인 항목은 목록에서 제외
- line/band 모델명 고정 문구인 `PatchTST`, `CNN-LSTM`은 meta/config 기반 표시로 변경

legacy composite를 별도 legacy 탭으로 보여주는 작업은 하지 않았다. 이번 CP에서는 일반 목록에서 제외하는 방식으로 처리했다.

# 6. Backend Latest Run 필터 변경 내용

`fetch_model_runs()`는 기본적으로 legacy composite를 제외한다.

판정 기준:

- `model_name == line_band_composite`
- `role == composite_model`
- `config.role == composite_model`
- `config.deprecated_for_phase1_product_contract`가 truthy
- `indicator_layer_replacement`가 있고 model이 composite인 경우

직접 run_id detail 조회는 막지 않았다. 기존 legacy artifact 보존과 디버깅을 위해 명시 조회는 허용한다.

# 7. Readiness 변화

`scripts/check_demo_readiness.ps1`을 수정했다.

변경 내용:

- product candidate model을 `patchtst`로 제한
- `/api/v1/ai/runs?...&include_legacy=false` 기준으로 제품 기본 run 확인
- legacy composite는 별도 `LEGACY_OK` 항목으로 표시
- legacy composite가 있어도 정상 제품 기본 산출물로 요구하지 않음

주의:

- readiness는 이번 작업 중 실제 서버를 새로 띄워 실행하지 않았다.
- UI 서버 실행은 이번 CP에서 필수가 아니므로 residual risk로 남긴다.

# 8. 검증 결과

백엔드 unittest:

```text
$env:PYTHONPATH="C:\Users\user\lens\backend"
python -m unittest discover backend\tests
```

결과:

- 52 tests OK

프론트 빌드:

```text
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 성공
- `/` size: 85.8 kB
- First Load JS: 173 kB

실제 DB 기반 TestClient 확인:

- 기본 `/api/v1/ai/runs`에서 composite 제외 확인
- `include_legacy=true`에서 composite 포함 확인
- legacy composite `is_legacy=true` 확인

# 9. 남은 과제

- overlay bundle API 설계
- role별 AI inference meta는 CP60-M에서 처리
- rolling AI band history
- h5/h20 horizon 결정
- legacy composite 전용 UI 섹션이 필요할지 제품 관점에서 재검토

# 10. 금지사항 준수 확인

이번 CP에서 하지 않은 일:

- 새 학습 실행
- AI 모델 구조 변경
- `ai/composite_inference.py` 같은 AI 스크립트 수정
- DB schema 변경
- fake data 생성
- legacy composite run 삭제
- 대규모 UI 리디자인
