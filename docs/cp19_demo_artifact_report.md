# CP19 데모용 실제 AI 산출물 확보 보고서

## 1. 결론

fake data 없이 실제 저장 산출물이 있는 데모 후보를 확보했다.

- 선택 demo run_id: `patchtst-1D-fc096a026a1e`
- 선택 demo ticker: `AAPL`
- timeframe: `1D`
- 프론트 표시 방식: 최신 completed run에 prediction이 없으면 최근 completed run들을 순서대로 확인해 실제 prediction row가 있는 run을 사용한다.
- 1차 성공 기준인 prediction overlay 데이터는 확보됐다.
- backtest row도 실제 `ai.backtest` 저장 경로로 생성했다.

## 2. 현재 completed PatchTST run 점검

`GET /api/v1/ai/runs?model_name=patchtst&status=completed&timeframe=1D&limit=20`

확인된 run:

| run_id | 상태 | 산출물 상태 |
| --- | --- | --- |
| `patchtst-1D-94d61c4e84d3` | completed | AAPL prediction/evaluation/backtest 없음 |
| `patchtst-1D-fc096a026a1e` | completed | AAPL prediction/evaluation 있음, CP19에서 backtest 생성 |

최신 run인 `patchtst-1D-94d61c4e84d3`에 inference 저장을 시도했지만 실패했다. 실패 위치는 checkpoint load 단계이며, 현재 `PatchTST` 코드 구조와 기존 checkpoint의 state_dict shape가 맞지 않았다.

```text
RuntimeError: Error(s) in loading state_dict for PatchTST
Missing key(s): revin.gamma, revin.beta, mixed_patch_proj...
size mismatch: position_embedding, line_head, band_head, patch_proj...
```

모델 구조 변경 금지 조건 때문에 이 checkpoint 호환 문제를 코드로 우회하지 않았다.

## 3. 선택한 데모 산출물

선택한 run은 `patchtst-1D-fc096a026a1e`이다.

- checkpoint: `ai\artifacts\checkpoints\patchtst_1D_patchtst-1D-fc096a026a1e.pt`
- checkpoint 존재 여부: 존재
- model: `patchtst`
- timeframe: `1D`
- horizon: `5`
- ticker: `AAPL`
- prediction rows: 9개
- evaluation rows: 9개

target type 확인 결과:

- `model_runs.config`와 checkpoint config에 `line_target_type`, `band_target_type` 명시값은 없다.
- 현재 inference 코드는 해당 값이 없으면 `raw_future_return` 기본값을 사용한다.
- 저장된 prediction row는 가격 레벨의 `line_series`, `lower_band_series`, `upper_band_series`와 `signal`을 포함한다.
- 따라서 non-raw target checkpoint의 score_only 제한 대상이 아니라, 데모 overlay에 사용할 수 있는 실제 저장 산출물로 판단했다.

## 4. prediction row 검증

`GET /api/v1/stocks/AAPL/predictions/latest?run_id=patchtst-1D-fc096a026a1e`

확인 결과:

- prediction 저장 여부: 저장되어 있음
- latest prediction asof_date: `2026-04-16`
- signal: `SELL`
- forecast_dates 길이: 5
- line_series 길이: 5
- upper_band_series 길이: 5
- lower_band_series 길이: 5
- forecast_dates와 series 길이 일치: 일치

프론트 차트 overlay에 필요한 필드는 모두 존재한다.

## 5. evaluation / backtest row 검증

evaluation:

- API: `GET /api/v1/ai/runs/patchtst-1D-fc096a026a1e/evaluations?ticker=AAPL&timeframe=1D&limit=1`
- 결과: 1건 조회
- coverage: `0.600000023841858`
- avg_band_width: `263.860015869141`

backtest:

- CP19에서 실행한 명령:

```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.backtest --run-id patchtst-1D-fc096a026a1e --timeframe 1D --strategy-name band_breakout_v1 --save
```

- API: `GET /api/v1/ai/runs/patchtst-1D-fc096a026a1e/backtests?timeframe=1D&limit=1`
- 결과: 1건 조회
- return_pct: `-4.071193639014`
- sharpe: `-0.219942266551737`
- mdd: `-0.04071193639014`
- win_rate: `0.111111111111111`
- num_trades: `4`
- meta.fee_bps: `10.0`
- meta.gross_return_pct: `-3.396888409655252`
- meta.gross_sharpe: `-0.17973689806023255`

## 6. 프론트 반영

수정 파일:

- `frontend/src/components/StockView.tsx`
- `scripts/check_demo_readiness.ps1`

주식 보기 화면은 기존처럼 completed PatchTST run을 사용한다. 다만 최신 completed run에 해당 ticker prediction row가 없으면 최근 completed run 최대 10개를 순서대로 확인해 실제 prediction row가 있는 run을 사용한다.

이 수정으로 현재 로컬 DB 상태에서는 다음 흐름이 된다.

1. 최신 run `patchtst-1D-94d61c4e84d3` 확인
2. AAPL prediction row 없음
3. 다음 run `patchtst-1D-fc096a026a1e` 확인
4. AAPL prediction row 있음
5. `line_series`, `upper_band_series`, `lower_band_series`를 차트 overlay props로 전달

프론트에서 AI 밴드와 보수적 예측선 표시가 가능한 상태다.

## 7. readiness check 결과

`powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`

```text
OK   health - live 200
OK   frontend - 200
OK   1D prices - AAPL 5 rows
OK   1M prices - price-only view available, 5 rows
FAIL stock search - 503 UPSTREAM_UNAVAILABLE
OK   completed runs - 2 rows, latest patchtst-1D-94d61c4e84d3
OK   demo run - patchtst-1D-fc096a026a1e
OK   prediction - forecast=5 line=5 upper=5 lower=5
WARN latest run - latest patchtst-1D-94d61c4e84d3 has no usable AAPL prediction
OK   evaluation - 1 rows
OK   backtest - 1 rows
```

남은 실패는 `stock_info` 기반 티커 검색 503이다. 가격 조회와 AI overlay용 prediction 조회에는 영향이 없고, 주식 보기 화면은 검색 실패 시 직접 입력 fallback을 사용한다.

## 8. 검증

백엔드:

- `GET http://127.0.0.1:8000/api/v1/health/live`: 200

프론트:

- `GET http://127.0.0.1:3000`: 200

데모 ticker:

- `GET /api/v1/stocks/AAPL/prices?timeframe=1D&limit=5`: 200
- `GET /api/v1/stocks/AAPL/predictions/latest?run_id=patchtst-1D-fc096a026a1e`: 200
- `GET /api/v1/ai/runs/patchtst-1D-fc096a026a1e/evaluations?ticker=AAPL&timeframe=1D&limit=1`: 200
- `GET /api/v1/ai/runs/patchtst-1D-fc096a026a1e/backtests?timeframe=1D&limit=1`: 200

프론트 빌드:

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
```

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 실패했고, 승인된 실행에서는 정상 통과했다.

## 9. 다음 조치

- `patchtst-1D-94d61c4e84d3` checkpoint는 현재 모델 코드와 호환되지 않는다. 이 run을 데모로 쓰려면 해당 checkpoint를 생성한 당시의 모델 구조를 보존한 별도 호환 로더가 필요하다.
- `stock_info` 기반 티커 검색 503 원인을 백엔드에서 별도 확인해야 한다.
- CP20 이후에는 데모 run을 문서에만 두지 말고, 운영 화면에서 “사용 가능한 최신 prediction run” 기준을 명시적으로 정리하는 것이 좋다.
