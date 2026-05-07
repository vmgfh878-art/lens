# CP131-G inference save 경로 latest-only guard 강화 보고서

생성일: 2026-05-06

## 1. 최종 판정

판정: PASS_WITH_TORCH_TEST_NOTE

`ai/inference.py --save`가 기본값으로 bulk `predictions` / `prediction_evaluations`를 저장하지 못하도록 막았다. 이제 저장은 둘 중 하나를 명시해야 한다.

- 제품 저장: `--save-product-latest-only`
- 평가/레거시 bulk 저장: `--allow-bulk-evaluation-save`

제품 run으로 식별되는 run은 bulk flag가 있어도 bulk 저장을 거부한다.

## 2. 변경 요약

변경 파일:

- `ai/inference.py`
- `ai/inference_save_guard.py`
- `ai/storage.py`
- `ai/tests/test_inference_save_guard.py`
- `ai/tests/test_inference_backtest.py`
- `ai/tests/test_storage_contracts.py`

핵심 변경:

- `--save` 단독 사용 시 저장 실패
- `--save-product-latest-only` 사용 시 `save_product_latest_predictions()` 호출
- `--allow-bulk-evaluation-save` 사용 시에만 평가용 bulk 저장 허용
- product run bulk 저장 차단
- prediction `meta.storage_contract`에 `product_latest_only` 또는 `evaluation_bulk` 기록
- product latest-only helper는 기존 단일 `asof_date`, row 수 제한, composite 금지 검증을 유지

## 3. 현재 저장 계약

| 저장 목적 | 허용 방식 | storage_contract | 비고 |
|---|---|---|---|
| 제품 최신 예측 | `--save-product-latest-only` 또는 helper 직접 호출 | `product_latest_only` | 단일 `asof_date`, line/band layer만 |
| 평가/레거시 bulk | `--allow-bulk-evaluation-save` 명시 | `evaluation_bulk` | 기본 차단 |
| 제품 run bulk 저장 | 금지 | 없음 | role/config가 제품 run이면 실패 |
| composite 저장 | 금지 유지 | 없음 | product latest-only validation에서 차단 |

## 4. 코드 계약

`ai/inference_save_guard.py`를 추가해 torch 없이 저장 계약만 검증할 수 있게 했다.

주요 함수:

- `is_product_model_run()`
- `resolve_inference_save_contract()`

`ai/inference.py`는 inference 결과를 만든 뒤, 이 함수가 반환한 계약에 따라 저장 경로를 선택한다.

`ai/storage.py`는 `with_prediction_storage_contract()`를 추가해 prediction record의 `meta.storage_contract`를 저장 직전에 주입한다. 입력 record는 mutate하지 않는다.

## 5. 검증 결과

실행:

```powershell
.\.venv\Scripts\python.exe -m py_compile ai\inference.py ai\storage.py ai\inference_save_guard.py ai\tests\test_storage_contracts.py ai\tests\test_inference_save_guard.py ai\tests\test_inference_backtest.py
.\.venv\Scripts\python.exe -m unittest ai.tests.test_storage_contracts ai.tests.test_inference_save_guard
```

결과:

- py_compile PASS
- `ai.tests.test_storage_contracts` PASS
- `ai.tests.test_inference_save_guard` PASS
- 총 12개 단위 테스트 PASS

참고:

`ai.tests.test_inference_backtest`는 현재 로컬 Windows 환경에서 `torch` import 중 `c10.dll` `WinError 1114`가 발생해 테스트 본문 실행 전 차단됐다. 이번 CP에서는 실제 inference 실행이 금지되어 있어 이 문제를 우회해 실행하지 않았고, 저장 guard 자체는 torch-free 테스트로 검증했다.

## 6. 금지 작업 확인

| 금지 작업 | 발생 |
|---|---:|
| DB write | false |
| Supabase upload | false |
| Supabase 대량 read | false |
| 실제 inference 실행 | false |
| 모델 학습 | false |
| 프론트 수정 | false |

## 7. 남은 주의점

제품 저장 스크립트는 계속 `save_product_latest_predictions()`를 직접 쓰는 현재 방식이 맞다. 일반 `ai/inference.py --save`는 이제 평가 bulk 저장도 명시 flag 없이는 막히므로, 운영자가 실수로 test split 전체 prediction history를 Supabase에 밀어 넣는 경로는 닫혔다.
