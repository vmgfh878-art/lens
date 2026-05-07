# CP88-D yfinance provider/source provenance 및 cache 격리 보강 보고서

작성일: 2026-05-03

## 1. Executive Summary

CP88-D에서는 CP87의 WARN 원인이었던 `price_data` provenance 부재와 feature cache/source hash 혼용 위험을 보강했다. yfinance primary 전환 방향은 유지하고, EODHD fallback도 유지했다.

이번 CP에서 운영 DB 전체 write, 기존 EODHD row 삭제, 모델 학습, live inference, product run 교체는 실행하지 않았다.

최종 판정은 `READY_FOR_SCHEMA_MIGRATION_THEN_LIMITED_WRITE`다. 코드 계약은 준비됐지만, 실제 운영 DB에는 아직 schema migration을 실행하지 않았으므로 yfinance write 전 `ensure_runtime_schema` 실행이 필요하다.

## 2. price_data provenance 현재 상태

read-only 확인 결과 현재 운영 `price_data` 컬럼은 다음과 같다.

`adjusted_close`, `amount`, `close`, `created_at`, `date`, `high`, `id`, `low`, `open`, `pbr`, `per`, `ticker`, `volume`

즉 현재 DB에는 `source`, `provider`, `provider_adjustment_policy`, `updated_at`이 없다.

## 3. 선택한 schema 방식

선택한 방식은 A안, 즉 `price_data` 자체에 provenance 컬럼을 추가하는 방식이다.

추가 계약:

| 컬럼 | 목적 |
|---|---|
| `source` | 저장 row의 실제 source. 예: `yfinance`, `eodhd` |
| `provider` | provider 표시용 중복 provenance. 현재는 `source`와 같은 값 |
| `provider_adjustment_policy` | raw/adjusted 재구성 정책 |
| `updated_at` | source_data_hash에 반영할 변경 시각 |

`backend/db/schema.sql`와 `backend/db/scripts/ensure_runtime_schema.py`에 반영했다. 기존 EODHD row의 대량 source backfill은 이번 CP에서 하지 않았다. 다음 제한 write 전 별도 migration/backfill 판단이 필요하다.

## 4. sync 저장 계약

`backend/collector/jobs/sync_prices.py`의 price record 생성에서 다음 값을 저장하도록 바꿨다.

- `source`
- `provider`
- `provider_adjustment_policy`
- `updated_at`

yfinance 정책은 `yfinance_auto_adjust_false_adj_close_factor_v3_adjusted_ohlc`다. EODHD 정책은 `eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc`다.

## 5. source_data_hash 강화

`ai/preprocessing.py`의 `resolve_data_fingerprint()`를 강화했다.

hash payload에는 이제 다음이 포함된다.

- `market_data_provider`
- `provider_adjustment_policy`
- `feature_contract_version`
- `timeframe`
- `ticker_universe_fingerprint`
- price date range
- indicator date range/count
- price row count
- price ticker count
- price `updated_at` 또는 `created_at`
- price data checksum

provider가 다르면 같은 ticker/date 범위라도 hash가 달라진다.

예시:

| provider | source_data_hash | feature cache path |
|---|---|---|
| `eodhd` | `baace9f6` | `features_1D_0d93d6a47d96_baace9f6.pt` |
| `yfinance` | `18d5c982` | `features_1D_b4358bf2224b_18d5c982.pt` |

## 6. cache manifest

feature cache와 feature index cache 옆에 manifest JSON을 저장하도록 했다.

manifest suffix:

`*.pt.manifest.json`

manifest 필드:

- `provider`
- `provider_adjustment_policy`
- `source_data_hash`
- `feature_version`
- `feature_columns`
- `ticker_count`
- `timeframe`
- `date_min`
- `date_max`
- `created_at`

cache load 시 manifest가 없거나 provider/hash/feature_version/feature_columns가 맞지 않으면 기존 cache를 재사용하지 않고 재생성한다. 이로써 EODHD cache와 yfinance cache가 같은 파일명 또는 stale manifest로 섞이는 일을 막는다.

## 7. 혼합 방지 방식

방어선은 세 겹이다.

1. `price_data` row에 `source/provider`를 저장한다.
2. source_data_hash에 provider와 adjustment policy를 넣는다.
3. cache manifest mismatch 시 load를 거부한다.

source 컬럼이 존재하는 DB에서는 yfinance fingerprint와 학습용 price frame read가 `source='yfinance'` row를 기준으로 계산된다. EODHD는 기존 legacy null row를 EODHD로 해석할 수 있게 하되, yfinance hash/read에는 null source row를 섞지 않는다.

## 8. 제한 write 가능 여부

이번 CP에서는 제한 write를 실행하지 않았다.

제한 write 전 필수 순서:

1. `python -m backend.db.scripts.ensure_runtime_schema`
2. 5티커 yfinance write
3. indicators 재계산
4. data quality check
5. feature cache hash와 manifest 확인

후보 명령:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --write --tickers AAPL MSFT NVDA TSLA AMZN --start-date 2026-04-01 --metrics-path docs/cp88_limited_write_rehearsal_metrics.json
```

## 9. 테스트

실행한 검증:

- `python -m py_compile ai/preprocessing.py backend/collector/jobs/sync_prices.py backend/collector/sources/market_data_providers.py backend/db/scripts/ensure_runtime_schema.py`
- `python -m unittest ai.tests.test_preprocessing_cache_isolation`
- `python -m unittest backend.tests.test_market_data_providers`
- `python -m unittest backend.tests.test_collector_jobs`
- `python -m unittest backend.tests.test_feature_svc`

결과는 모두 PASS다.

## 10. 다음 단계

다음 CP에서는 schema migration을 실제 운영 DB에 적용한 뒤, 5티커 제한 write와 indicators 재계산을 실행해야 한다. 그 다음 `source='yfinance'` 기준으로 feature cache가 새 hash/manifest를 쓰는지 확인하고, 문제가 없을 때 50티커 제한 전환으로 넓히는 순서가 안전하다.
