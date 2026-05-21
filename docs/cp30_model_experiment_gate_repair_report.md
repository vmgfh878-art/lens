# CP30-G 모델 실험 게이트 수리 보고서

## 한 줄 결론

CP29-D로 피처가 깨끗해졌더라도, 저장/재현/백테스트 계약이 흔들리면 다음 모델 실험 결과를 믿을 수 없다. CP30-G에서는 새 실험을 늘리지 않고, run 저장과 inference, backtest의 P1 신뢰성 게이트를 먼저 막았다.

## 범위

이번 CP에서는 full 473티커 실행, W&B sweep, 대형 성능 비교, UI 수정은 하지 않았다.

수정 대상은 DB schema, 저장 upsert key, feature version 기록, checkpoint ticker registry 재사용, adjusted backtest anchor, 포트폴리오 단위 turnover 정의, coverage gate 실패 run 상태 정책이다.

## 결정한 계약

| 항목 | 결정 | 이유 |
|---|---|---|
| `model_runs.band_mode` | 정식 컬럼으로 추가 | config JSON만으로는 head 계약을 안정적으로 조회하기 어렵다. |
| `prediction_evaluations.lower_breach_rate`, `upper_breach_rate` | 정식 컬럼으로 추가 | inference 저장 레코드와 DB schema를 일치시킨다. |
| `model_runs.feature_version` | `ai.preprocessing.FEATURE_CONTRACT_VERSION` 기준 저장 | CP29-D 이후 실제 학습 입력은 `v3_adjusted_ohlc`다. |
| `predictions` unique/upsert | `run_id,ticker,model_name,timeframe,horizon,asof_date` | run별 prediction 이력을 보존해야 backtest와 화면 재현이 가능하다. |
| inference ticker id | checkpoint의 `ticker_registry_path`를 로드해 사용 | subset inference가 새 registry를 만들면 ticker embedding id가 바뀔 수 있다. |
| backtest anchor | `adjusted_close` 우선, 없으면 `close` fallback | CP29-D 이후 예측 price decode와 realized return 기준을 adjusted 가격으로 맞춘다. |
| backtest position/turnover | 날짜별 equal absolute exposure 포트폴리오 | ticker row 순서에 따라 turnover가 바뀌는 오류를 제거한다. |
| coverage gate 실패 run | `failed_quality_gate` | 학습은 완료됐더라도 제품 inference/backtest 대상인 `completed`와 분리한다. |

## 코드 변경

- `backend/db/schema.sql`, `backend/db/scripts/ensure_runtime_schema.py`
  - `model_runs.band_mode` 추가
  - `prediction_evaluations.lower_breach_rate`, `upper_breach_rate` 추가
  - `model_runs.status`에 `failed_quality_gate` 추가
  - `predictions` old unique constraint 제거 후 run-aware unique constraint 추가
  - `model_runs.feature_version` 기본값을 `v3_adjusted_ohlc`로 변경
- `ai/storage.py`
  - `save_predictions()` upsert key에 `run_id` 포함
- `ai/train.py`
  - checkpoint/config/model_runs 저장 feature version을 `FEATURE_CONTRACT_VERSION`으로 통일
  - coverage gate 실패 fallback run은 `failed_quality_gate`로 저장
- `ai/inference.py`, `ai/preprocessing.py`
  - checkpoint registry를 inference dataset 생성에 주입
  - checkpoint registry의 `timeframe`, `num_tickers`, mapping 크기가 config와 다르면 실패
  - 학습 registry는 ticker 집합 fingerprint가 들어간 파일로 저장해 timeframe 공용 registry overwrite를 피함
- `ai/backtest.py`
  - anchor를 adjusted close 기준으로 변경
  - row 단위 shift turnover를 날짜별 포트폴리오 turnover로 변경
- `backend/app/repositories/ai_repo.py`, `backend/app/routers/v1/ai.py`, `backend/app/schemas/ai.py`
  - 새 run/evaluation 컬럼을 API 조회 계약에 반영

## 검증

실제 schema migration을 실행했고 성공했다.

```text
python -m backend.db.scripts.ensure_runtime_schema
[OK] ai_tables=['backtest_results', 'model_runs', 'prediction_evaluations']
[OK] prediction_writer_columns=['band_quantile_high', 'band_quantile_low', 'line_series', 'run_id']
[OK] model_run_contract_columns=['band_mode', 'feature_version', 'status']
[OK] prediction_evaluation_contract_columns=['lower_breach_rate', 'upper_breach_rate']
```

단위 검증:

```text
python -m unittest ai.tests.test_inference_backtest ai.tests.test_storage_contracts ai.tests.test_checkpoint_selection
```

결과: 12건 통과.

검증 내용:

- `save_model_run` 1건 dry-run 성격의 upsert 호출 테스트 통과
- `save_predictions`가 run-aware conflict key를 사용하는지 확인
- inference `--save` 경로가 prediction/evaluation 저장 함수를 호출하는지 확인
- checkpoint ticker registry mismatch 시 실패하는지 확인
- backtest가 adjusted anchor를 쓰는지 확인
- backtest가 날짜별 포트폴리오 turnover를 쓰는지 확인
- coverage gate 실패 run status 정책 확인

## Backtest 계약

이전 구현은 전체 prediction row를 날짜순으로 정렬한 뒤 바로 이전 row와 position을 비교했다. 이 방식은 AAPL 다음 행이 MSFT이면 AAPL 포지션 변화가 아니라 ticker가 바뀐 효과를 turnover로 계산할 수 있었다.

CP30-G 이후 backtest는 날짜별 포트폴리오로 계산한다.

- `BUY=+1`, `SELL=-1`, `HOLD=0`
- 같은 날짜의 활성 포지션은 절대 노출 합이 1이 되도록 정규화
- 날짜별 gross return은 ticker별 weight와 realized return의 합
- turnover는 이전 날짜 weights와 현재 날짜 weights의 L1 변화량
- fee는 날짜별 turnover에 적용

이 계약은 현재 단순 signal 기반 전략을 위한 최소 신뢰성 수리다. 더 정교한 long-only, long-short, risk cap, band-size position sizing은 별도 CP에서 다룬다.

## Inference 재현성 계약

학습 시 저장된 checkpoint config의 `ticker_registry_path`가 inference의 단일 기준이다. `--tickers MSFT`처럼 subset inference를 하더라도 새 registry에서 MSFT를 0으로 다시 매기지 않는다. checkpoint registry에서 MSFT가 1이면 inference도 1을 쓴다.

registry mismatch는 조용히 fallback하지 않고 실패한다. ticker embedding id가 틀린 채 저장되는 prediction은 run 재현성을 깨기 때문이다.

또한 CP30-G 이후 학습 plan은 `ai/cache/ticker_id_map_1d.json` 같은 timeframe 공용 파일을 덮어쓰지 않고, ticker 집합 fingerprint가 포함된 registry 파일을 checkpoint config에 남긴다. registry path 자체가 run artifact 성격을 갖도록 바꾼 것이다.

## RevIN

RevIN은 이번 CP에서 수정하지 않았다. 별도 ablation 계획만 남긴다.

권장 ablation:

| 실험 | 목적 | 제한 |
|---|---|---|
| PatchTST `use_revin=True` 기준 재확인 | CP29/CP30 계약 수리 이후 현 기준선 확인 | 최대 50티커 smoke |
| PatchTST `use_revin=False` | adjusted feature 이후 RevIN이 여전히 도움이 되는지 확인 | 같은 split, 같은 seed |
| target channel fast + RevIN on/off | target channel 경로와 RevIN 상호작용 확인 | 성능 비교보다 finite/calibration 우선 |

## 다음 순서

1. 저장 run 1건을 작은 ticker subset으로 만들 때 `feature_version=v3_adjusted_ohlc`, `band_mode`, `status`가 DB에 남는지 확인한다.
2. 같은 run으로 inference `--save`와 backtest `--save`를 아주 작은 ticker 집합에서 재확인한다.
3. 그 다음에야 CP29-D clean feature 기준 PatchTST smoke 실험을 재개한다.
