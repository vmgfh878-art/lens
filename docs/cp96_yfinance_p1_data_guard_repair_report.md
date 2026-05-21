# CP96-D: yfinance 전체 전환 전 P1 데이터 가드 수리 보고서

## 1. Executive Summary

CP96-D는 전체 universe yfinance write와 live inference 연결 전에 남은 P1 데이터 가드를 수리했다. 최종 판정은 **PASS**다. 전체 yfinance write, 100티커 추가 write, EODHD row 삭제, full retraining, save-run, live inference, product run 교체는 실행하지 않았다.

핵심 변경은 세 가지다.

- feature cache fingerprint가 indicator row count/min/max만 보지 않고 indicator 값 checksum을 포함한다.
- 가격/API/backtest/daily sync 조회가 product provider 또는 명시 provider를 따라 yfinance/eodhd source를 분리한다.
- 1W/1M resample과 API display에서 아직 끝나지 않은 현재 주/월 bucket을 완성 candle처럼 저장하거나 보여주지 않는다.

CP95 yfinance 100티커 cache도 새 checksum 계약으로 기존 `source_data_hash=5be36437`에서 `b6ad28de`로 바뀌는 것을 확인했다. 새 cache path는 존재하지 않아 기존 CP95 cache가 그대로 재사용되지 않는다.

## 2. Indicator Value Checksum

수정 파일: `ai/preprocessing.py`

`resolve_data_fingerprint()`에 indicator value checksum을 추가했다. checksum 범위는 provider/source, timeframe, ticker universe, date range에 종속된다. Postgres 경로는 ticker/date와 indicator 값 컬럼을 row hash로 만들고 ticker별 checksum을 다시 합성한다. REST fallback 경로도 같은 의미의 row payload checksum을 만든다.

포함 컬럼은 모델 입력 feature column과 indicator 보조 확인용 `atr_ratio`, `volume`이다. 따라서 indicator numeric value가 바뀌면 row count와 date min/max가 같아도 `source_data_hash`가 달라지고, `features_*.pt`와 `feature_index_*.pt` cache path도 달라진다.

한계: Postgres checksum은 row hash aggregation을 사용하므로 단순 count/min/max보다 비용이 높다. 다만 현재 100티커와 1D full history 기준으로 read-only fingerprint 계산은 통과했고, 전체 universe 전환 시에는 실행 시간을 모니터링해야 한다.

검증:

- 같은 ticker/date/count에서 indicator 값만 바뀌는 fixture로 hash 변경 확인
- manifest의 `source_data_hash`가 다르면 cache 재사용 거부 확인
- CP95 yfinance 100티커 기준 기존 hash `5be36437` → 새 hash `b6ad28de`

## 3. Source/Provider 조회 계약

수정 파일:

- `backend/app/repositories/market_repo.py`
- `backend/app/services/api_service.py`
- `ai/backtest.py`
- `backend/collector/pipelines/daily_market_sync.py`
- `backend/collector/repositories/base.py`

정책:

- product/local 기본 provider는 `MARKET_DATA_PROVIDER`를 따른다.
- yfinance mode는 `source='yfinance'` row만 읽는다.
- eodhd mode는 `source='eodhd'` 또는 legacy `source IS NULL` row를 읽는다.
- provider/source 없이 repository를 직접 호출하면 warning을 남긴다.
- stock_info fallback은 stock_info가 없을 때 price_data fallback 조회에만 source filter를 적용한다.
- backtest anchor close도 동일 provider source를 기준으로 읽는다.
- daily_market_sync coverage gate도 active provider 기준 최신 가격 coverage를 계산한다.

검증:

- yfinance/eodhd 병렬 row fixture에서 `fetch_price_rows(..., market_data_provider='yfinance')`는 yfinance row만 반환
- eodhd 조회는 eodhd/null legacy row만 반환
- backtest anchor close가 yfinance/eodhd provider별로 분리됨
- daily_market_sync coverage가 yfinance source filter를 사용함

## 4. 1W/1M Partial Period 제외

수정 파일:

- `backend/app/services/feature_svc.py`
- `backend/app/services/api_service.py`

정책:

- 1D는 기존처럼 유지한다.
- 1W는 latest daily date 기준으로 해당 주 금요일이 아직 오지 않았으면 현재 주 bucket을 제외한다.
- 1M은 latest daily date 기준으로 해당 월 말일이 아직 오지 않았으면 현재 월 bucket을 제외한다.
- `dropna()`만으로 완성 candle을 판단하지 않는다.
- API display도 같은 공통 helper를 사용해 partial 1W/1M candle을 완성 데이터처럼 보여주지 않는다.

검증:

- 수요일 daily data로 1W resample 시 그 주 금요일 candle 저장 금지
- 월 중간 daily data로 1M resample 시 그 달 month-end candle 저장 금지
- 이미 끝난 주/월 bucket은 유지

## 5. CP95 Cache 재사용 차단 확인

CP95 yfinance 100티커 기준:

| 항목 | 값 |
|---|---|
| 기존 CP95 source_data_hash | `5be36437` |
| CP96 checksum 포함 후 source_data_hash | `b6ad28de` |
| hash 변경 여부 | true |
| 기존 CP95 feature cache | `ai/cache/features_1D_4ba45fc9bb15_5be36437.pt` |
| CP96 feature cache 후보 | `ai/cache/features_1D_4ba45fc9bb15_b6ad28de.pt` |
| CP96 후보 존재 여부 | false |
| CP96 feature index 후보 | `ai/cache/feature_index_1D_90b5620146f0_b6ad28de.pt` |
| CP96 index 후보 존재 여부 | false |

즉, CP95의 기존 cache는 checksum 계약 변경 후 같은 파일명으로 재사용되지 않는다. 다음 feature build에서 새 hash 기준으로 재생성된다.

## 6. P2 분리

이번 CP에서는 P1만 수리했다. 아래 항목은 다음 CP 후보로 남긴다.

- `prediction_repo` 조회 계약을 `run_id/timeframe/horizon/model_name` 묶음 기준으로 더 엄격히 정리
- band calibration product inference 재사용 방식 확정
- fresh DB에서 `ensure_runtime_schema`가 predictions table을 완전 생성하는지 재점검
- cache/checkpoint/gitignore 정리

## 7. 검증

실행한 검증:

- `python -m py_compile ai\preprocessing.py ai\backtest.py backend\app\repositories\market_repo.py backend\app\services\feature_svc.py backend\app\services\api_service.py backend\collector\pipelines\daily_market_sync.py backend\collector\repositories\base.py`
- `$env:PYTHONPATH='backend'; python -m unittest ai.tests.test_preprocessing_cache_isolation ai.tests.test_inference_backtest backend.tests.test_feature_svc backend.tests.test_services backend.tests.test_collector_jobs`
- `python -m unittest ai.tests.test_preprocessing_cache_isolation ai.tests.test_inference_backtest backend.tests.test_feature_svc backend.tests.test_collector_jobs`
- CP95 yfinance 100티커 `resolve_data_fingerprint('1D', market_data_provider='yfinance')` read-only 확인

결과:

- py_compile PASS
- 관련 unittest 50개 PASS
- backend path 없이 실행 가능한 관련 unittest 38개 PASS
- DB write 없음
- 모델 학습 없음
- save-run 없음
- live inference 없음

## 8. 최종 판정

**PASS: 전체 universe 1D yfinance write로 진행 가능.**

단, 전체 write 직전에는 이번 checksum 변경 때문에 yfinance 100티커 cache가 새 hash로 재생성되는지 한 번 더 확인하고, 전체 universe에서 checksum query 시간이 과도하지 않은지 관찰해야 한다.
