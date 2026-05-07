# CP148-0-DG3 cache/provider WARN closure 보고서

## 1. Executive Summary

판정: **PASS**

CP148-0-DG2에서 남은 실험 신뢰도 WARN 중 cache/provider 관련 항목을 닫았다. 1W EODHD 500 local parquet 기준으로 feature index cache와 feature cache를 새로 refresh했고, CP150/CP151에서 사용할 1W registry도 EODHD eligible 기준으로 재확인했다.

Fundamentals coverage WARN은 이번 CP의 해결 대상이 아니라 **feature 해석 WARN**으로 분리한다.

최종 판단:

- CP150-LM-1W 실행 가능
- CP151-BM-1W 실행 가능
- CP148/149도 cache/provider 충돌 없음
- 남은 WARN은 실험 차단이 아니라 해석 주의다

## 2. 금지 작업 확인

이번 CP에서 하지 않은 작업:

- 모델 학습 없음
- W&B/Optuna 실행 없음
- inference 저장 없음
- DB/Supabase write 없음
- EODHD/yfinance fetch 없음
- 프론트 수정 없음
- legacy cache 삭제 없음

수행한 작업:

- EODHD 500 local parquet snapshot alias 기준 1W cache refresh
- 1W EODHD cache manifest 검증
- CP150/CP151 registry 재확인
- legacy cache non-use assertion 작성

## 3. EODHD 1W Cache Refresh

실행 환경:

- `MARKET_DATA_PROVIDER=eodhd`
- `LENS_DATA_BACKEND=local`
- `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`
- `LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\docs\cp146_lm_eodhd500_line_full_training_logs\snapshot_alias`

생성/확정된 cache:

| kind | path |
|---|---|
| feature_index | `ai/cache/feature_index_1W_ad2ca650d771_2a5d5a33.pt` |
| feature_index manifest | `ai/cache/feature_index_1W_ad2ca650d771_2a5d5a33.pt.manifest.json` |
| features | `ai/cache/features_1W_90ba70362fe5_2a5d5a33.pt` |
| features manifest | `ai/cache/features_1W_90ba70362fe5_2a5d5a33.pt.manifest.json` |

Manifest 핵심값:

| 항목 | 값 |
|---|---|
| provider | `eodhd` |
| source | `eodhd` |
| timeframe | `1W` |
| feature_version | `v3_adjusted_ohlc` |
| source_data_hash | `2a5d5a33` |
| feature columns | 36 |
| feature rows | 258,410 |
| feature ticker count | 502 |
| feature date range | 2016-02-19 ~ 2026-05-01 |
| feature_index rows | 250,635 |
| feature_index ticker count | 502 |

`is_cache_manifest_valid()` 결과:

- feature index manifest: true
- feature manifest: true

추가로 manifest에 CP150/CP151 eligible ticker count와 registry path/hash를 기록했다. 기존 manifest 비교 로직은 extra field를 무시하므로 cache validation은 유지된다.

## 4. 1W Registry 확정

| CP | registry | count | SW 포함 | zero-based continuous | ticker_order_hash |
|---|---|---:|---|---|---|
| CP150-LM-1W | `ai/cache/ticker_id_map_1w_221af228cd24.json` | 447 | true | true | `f8ae1b50e3c5b356` |
| CP151-BM-1W | `ai/cache/ticker_id_map_1w_4ea1266146e9.json` | 453 | true | true | `6107f3b2dfae4792` |

이전 1W generic/latest registry `ticker_id_map_1w_9915e57df87e.json`는 446개였고 `SW`가 빠져 있었다. 이번 CP 이후 CP150/CP151은 위 scenario별 registry를 사용해야 한다. yfinance/generic registry와 혼동하면 안 된다.

## 5. Legacy Cache 안전화

Manifest 없는 legacy `.pt` cache:

- count: 75
- 삭제하지 않음
- 전체 목록: `docs/cp148_0_dg3_cache_provider_warn_closure_metrics.json`의 `legacy_cache_non_use_assertion.stale_manifest_missing_candidates`

Non-use assertion:

- CP148~151에서 사용할 cache path와 manifest 없는 legacy cache의 overlap: 0
- 1D/1W 최신 실험 대상 cache 중 yfinance latest risk: 0
- 1M yfinance cache는 남아 있지만 CP148~151 대상이 아니므로 실험 차단 아님

Provider-aware cache path 확인:

- `resolve_feature_cache_path()`는 payload에 `market_data_provider`, `market_data_source`, `provider_adjustment_policy`, `feature_contract_version`을 포함한다.
- `resolve_feature_index_cache_path()`도 동일하게 provider/source/policy를 포함한다.
- 따라서 runner가 `market_data_provider=eodhd` 또는 `MARKET_DATA_PROVIDER=eodhd`를 명시하면 generic latest yfinance cache가 자동 선택되지 않는다.

## 6. CP148~151 사용 경로

| CP | cache/provider 상태 |
|---|---|
| CP148-LM-1D | latest 1D feature/index cache가 eodhd. registry 473 일치 |
| CP149-BM-1D | provider 충돌 없음. registry 476 사용. 기존 473 feature cache를 BM 476 실험에 그대로 쓰면 안 되며 provider-aware 새 cache path 확인 필요 |
| CP150-LM-1W | 새 EODHD 1W cache `2a5d5a33` + registry 447 고정 |
| CP151-BM-1W | 새 EODHD 1W cache `2a5d5a33` + registry 453 고정 |

CP149의 주의점은 provider 혼합이 아니라 eligible count 차이다. BM 1D가 476 eligible로 가면 첫 실행에서 476 tickers 기준 cache path가 새로 생성되는지 preflight에서 확인해야 한다.

## 7. Full Features WARN 분리

Fundamentals coverage 낮음은 cache/provider 결함이 아니다. 이번 CP에서는 실험 차단 항목에서 분리한다.

정책:

- `full_features`는 기본 후보가 아니라 비교 후보로 둔다.
- 제품 설명에서 “펀더멘털 충분 활용” 금지.
- 추천 우선 feature_set:
  - `price_volatility_volume`
  - `no_fundamentals`
  - `technical_only`
  - `price_volatility`
- `context_light`는 `design_needed` 유지. 지금 억지로 만들지 않는다.

## 8. CP148~151 실행 전 Preflight Checklist

실행 직전 사람이 확인할 항목:

1. `MARKET_DATA_PROVIDER=eodhd` 또는 명시 인자 `market_data_provider=eodhd`
2. `LENS_DATA_BACKEND=local`
3. `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`
4. `LENS_LOCAL_SNAPSHOT_DIR`가 EODHD 500 alias 또는 동등한 EODHD 500 parquet 디렉터리
5. 1W feature cache manifest:
   - provider/source = `eodhd`
   - timeframe = `1W`
   - source_data_hash = `2a5d5a33`
6. CP150 registry:
   - `ai/cache/ticker_id_map_1w_221af228cd24.json`
   - count 447
   - `SW` 포함
7. CP151 registry:
   - `ai/cache/ticker_id_map_1w_4ea1266146e9.json`
   - count 453
   - `SW` 포함
8. generic latest cache 이름이 아니라 provider-aware resolved cache path 사용
9. full_features 결과 해석에는 fundamentals coverage WARN 표기

## 9. 남은 WARN 구분

실험 차단 WARN 제거됨:

- 1W generic latest yfinance cache risk
- 1W registry 447 vs 446 mismatch
- CP150/151 cache path 불명확성

실험 차단이 아닌 해석 WARN:

- fundamentals coverage 낮음
- `context_light` 미정의
- legacy manifest 없는 cache 75개 존재

legacy cache는 사용 경로와 겹치지 않으므로 삭제하지 않아도 CP148~151 실험 신뢰도에는 직접 영향을 주지 않는다.

## 10. 최종 판정

**PASS**

CP150-LM-1W와 CP151-BM-1W는 이제 실행 가능하다. CP148/149도 cache/provider 충돌은 없다. 남은 WARN은 fundamentals/context 해석 주의이며, 실험 차단 사유가 아니다.
