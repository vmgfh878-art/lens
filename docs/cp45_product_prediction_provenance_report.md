CP45-P는 디자인 대공사가 아니라, 주식 보기 화면에서 현재 표시 중인 예측의 출처와 해석 가능성을 보강하는 CP다.

# 1. 목표

주식 보기 화면에서 예측선과 예측 밴드가 어떤 저장 run/model 조합에서 온 것인지 사용자가 이해할 수 있도록 정리했다.

이번 CP에서는 모델 학습 코드, DB schema, prediction 생성 로직, fake data를 건드리지 않았다.

# 2. 변경 파일

- `backend/app/schemas/stocks.py`
- `backend/app/repositories/prediction_repo.py`
- `backend/app/services/api_service.py`
- `backend/app/routers/v1/ai.py`
- `backend/tests/test_services.py`
- `frontend/src/api/client.ts`
- `frontend/src/components/StockView.tsx`
- `frontend/src/components/TrainingView.tsx`
- `frontend/src/components/Chart.tsx`
- `frontend/src/app/globals.css`
- `docs/cp45_product_prediction_provenance_report.md`

# 3. 주식 보기 화면 변경

오른쪽 예측 패널에 `예측 출처` 영역을 추가했다.

브라우저 확인 기준 AAPL 1D에서 표시된 내용:

- 저장된 예측 run: `composite-1D-3a44b5e51ed2`
- 기준일: `2026-03-30`
- 예측 구간: `5거래일`
- 표시 기준: `latest run`
- run 모델: `line_band_composite`
- 예측 모델: `Composite`

composite prediction일 때 meta에서 아래 정보를 읽어 표시한다.

- 예측선 모델: `PatchTST`
- 밴드 모델: `CNN-LSTM`
- 밴드 보정: `scalar width calibration`
- 조합 정책: `risk_first_lower_preserve`
- feature contract: `v3_adjusted_ohlc`

사용자에게 불안하게 보이는 오류 문구 대신 조용한 정보 문구로 처리했다.

# 4. Prediction Meta 처리

백엔드 prediction 응답에 `meta`를 포함하도록 보강했다.

확인한 API:

```text
GET /api/v1/stocks/AAPL/predictions/latest?run_id=composite-1D-3a44b5e51ed2
```

확인 결과:

- `forecast_dates`: 5개
- `line_series`: 5개
- `upper_band_series`: 5개
- `lower_band_series`: 5개
- `conservative_series`: 5개
- `feature_contract`: `v3_adjusted_ohlc`
- `composition_policy`: `risk_first_lower_preserve`
- `band_calibration_method`: `scalar_width`

# 5. Latest / Fallback 처리

주식 보기 화면은 completed run을 최신순으로 조회하고, `line_band_composite`와 `patchtst` run 중 실제 AAPL prediction row가 있는 run을 사용한다.

브라우저 확인에서는 `composite-1D-3a44b5e51ed2`가 AAPL 1D prediction을 가지고 있어 해당 run을 표시했다.

readiness 스크립트는 아직 기존 patchtst demo run 기준으로 점검하므로 아래 메시지가 남는다.

- latest PatchTST run: `patchtst-1D-41d584bcb3cb`
- usable demo run: `patchtst-1D-239b58ab90f0`

이 차이는 readiness 스크립트의 점검 기준이 composite UI 선택 기준보다 좁아서 생긴다. fake data는 생성하지 않았다.

# 6. 차트 압축 완화

예측 밴드가 실제 가격 범위 대비 지나치게 넓거나 가격 범위를 크게 벗어나면 prediction overlay를 별도 price scale로 표시하도록 했다.

중요한 원칙:

- prediction 값은 바꾸지 않았다.
- band 값을 클리핑하지 않았다.
- 가격 데이터와 prediction 데이터는 그대로 유지했다.
- 화면 표시 scale만 분리할 수 있게 했다.

AAPL 1D 브라우저 확인에서는 가격 차트와 미래 예측 5개 지점이 함께 보였고, 화면 전체가 무너지지 않았다.

# 7. 1M Price-Only 확인

1M 전환 시 기존 정책대로 가격 전용 화면으로 동작한다.

브라우저 확인 결과:

- 가격 캔들 차트 표시
- 예측 밴드 토글 비활성
- 보수적 예측 토글 비활성
- 예측 run 없음 안내 표시
- 빨간 백엔드 연결 오류 배너 없음

# 8. Stock Search 503 처리

현재 `GET /api/v1/stocks?search=AAPL`은 503을 반환한다.

화면에서는 이를 전체 백엔드 오류로 보이지 않게 유지했다.

표시 방식:

- 작은 보조 안내: 티커 검색을 사용할 수 없지만 직접 입력하면 가격 조회 가능
- 가격 API와 prediction API가 정상일 때 빨간 백엔드 연결 실패 배너는 표시하지 않음

# 9. 모델 학습 화면 변경

상태 탭을 아래처럼 구분했다.

- 완료(completed)
- NaN 실패(failed_nan)
- 품질 게이트 실패(failed_quality_gate)

composite run에서는 조합 정보를 볼 수 있는 영역을 추가했다.

표시 항목:

- `run_id`
- `model_name`
- `feature_version`
- `line_model_run_id`
- `band_model_run_id`
- `composition_policy`
- `band_calibration_method`
- `prediction_composition_version`

또한 line 모델 지표와 band 모델 지표를 분리해서 보여줄 수 있는 자리만 정리했다. 복잡한 랭킹 시스템은 만들지 않았다.

# 10. 브라우저 확인 결과

확인 URL:

- 프론트: `http://127.0.0.1:3000`
- 백엔드: `http://127.0.0.1:8000`

AAPL 1D 확인:

- 가격 차트 표시
- 미래 5개 지점 예측 표시
- 예측 밴드 상단/하단 표시
- 보수적 예측선 표시
- 예측 시작 기준선 표시
- 예측 출처 패널 표시
- composite meta 표시
- 빨간 백엔드 연결 오류 배너 없음

AAPL 1M 확인:

- 가격 전용 차트 표시
- AI overlay 비활성
- 화면 깨짐 없음
- 빨간 백엔드 연결 오류 배너 없음

모델 학습 화면 확인:

- completed / failed_nan / failed_quality_gate 탭 표시
- composite run provenance 영역 표시
- line / band 모델별 지표 자리 표시

# 11. 검증 결과

프론트 빌드:

```text
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 성공
- `/` size: 85.6 kB
- First Load JS: 173 kB

백엔드 테스트:

```text
$env:PYTHONPATH="C:\Users\user\lens\backend"
python -m unittest discover backend\tests
```

결과:

- 47 tests OK

데모 readiness:

```text
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과 요약:

- OK health
- OK CORS
- OK frontend
- OK AAPL 1D prices
- OK indicators
- OK AAPL 1M prices
- FAIL stock search: 503
- OK completed runs
- OK prediction
- OK evaluation
- WARN latest PatchTST run has no usable AAPL prediction
- WARN backtest row missing

# 12. 남은 리스크

- stock search API는 여전히 503이다. 가격 조회와 예측 조회는 가능하므로 화면 전체 오류로 처리하지 않는다.
- readiness 스크립트는 아직 patchtst demo run 기준이라 composite prediction 선택 결과와 기준이 다르다.
- stored backtest row가 없어 readiness에서 WARN으로 남는다.
- composite run의 line/band 지표 비교는 자리만 정리했고, 상세 랭킹이나 성능 비교 UI는 다음 CP에서 다루는 것이 좋다.
