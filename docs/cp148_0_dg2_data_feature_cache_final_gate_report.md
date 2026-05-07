# CP148-0-DG2 데이터/피처/cache/registry 최종 게이트 보고서

## 1. Executive Summary

판정: **WARN**

EODHD 500 local parquet 기준 원천 가격, 1D/1W indicators, feature finite contract, split/target finite, 작은 forward shape 검증은 통과했다. 따라서 데이터 자체가 CP148~CP151 실험을 막는 상태는 아니다.

다만 아래 두 가지는 WARN으로 남긴다.

- `full_features`의 macro/breadth는 실제 값이 들어가 있지만, fundamentals coverage가 낮다. 따라서 full_features 모델을 “펀더멘털을 충분히 활용한 모델”이라고 설명하면 안 된다.
- 1W cache/registry는 EODHD 실험 전에 명시적 refresh가 필요하다. 현재 generic latest 1W feature/index cache 중 일부는 yfinance이고, 기존 EODHD 1W feature cache ticker_count가 새 eligible count와 다르다.

결론적으로 **1D 실험은 바로 진행 가능**, **1W 실험은 EODHD provider/source 고정과 cache/registry refresh 확인 후 진행 가능**이다.

## 2. 금지 작업 확인

이번 CP에서 하지 않은 작업:

- 모델 full training 없음
- W&B/Optuna 실행 없음
- inference 저장 없음
- DB/Supabase write 없음
- 프론트 수정 없음
- EODHD/yfinance fetch 없음
- 기존 cache 삭제 없음

허용 범위에서 수행한 작업:

- EODHD 500 local parquet 기준 read-only 감사
- CP148~CP151 scenario eligible ticker 기준 registry 파일 저장/확인
- CPU-only tiny forward shape 1회

## 3. 원천 가격 데이터 검증

대상:

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/price_data_eodhd_500.manifest.json`

| 항목 | 결과 |
|---|---:|
| ticker count | 503 |
| row count | 1,387,834 |
| date range | 2015-01-02 ~ 2026-05-05 |
| source/provider | eodhd / eodhd |
| duplicate ticker/date/source | 0 |
| adjusted OHLC violation | 0 |
| raw high/low violation | 0 |
| adjusted factor nonfinite/nonpositive | 0 / 0 |
| volume null/negative | 0 / 0 |
| volume zero | 2,342 |

Adjusted factor 범위는 `0.01428558 ~ 4.93523810`이다. 이는 장기 split/corporate action 반영 결과로 볼 수 있고, adjusted OHLC 재구성 violation은 0이다.

## 4. Indicator 검증

대상:

- `data/parquet/indicators_eodhd_1D_500.parquet`
- `data/parquet/indicators_eodhd_1W_500.parquet`

| 항목 | 1D | 1W |
|---|---:|---:|
| ticker count | 503 | 502 |
| row count | 1,355,956 | 258,410 |
| date range | 2015-03-30 ~ 2026-05-05 | 2016-02-19 ~ 2026-05-01 |
| duplicate ticker/timeframe/date/source | 0 | 0 |
| model feature columns | 36 | 36 |
| feature NaN/Inf | 0 / 0 | 0 / 0 |
| atr_ratio non-null | 1,355,956 | 258,410 |
| atr_ratio p99 abs / max abs | 0.0810 / 0.8106 | 0.1719 / 1.0966 |
| open_ratio p99 abs / max abs | 0.0486 / 0.8109 | 0.0505 / 0.5524 |
| high_ratio p99 abs / max abs | 0.0761 / 0.8471 | 0.1674 / 1.5573 |
| low_ratio p99 abs / max abs | 0.0752 / 0.6941 | 0.1543 / 0.7118 |

1W partial week check:

- latest 1W date: 2026-05-01
- non-Friday rows: 0
- partial week row: 0

`atr_ratio`는 indicator에는 존재하지만 `MODEL_FEATURE_COLUMNS`에는 포함되지 않는다. `MODEL_N_FEATURES=36` 유지도 확인했다.

## 5. Context / Full Features 해석

Context source:

- macro: `data/parquet/context/eodhd_500/macroeconomic_indicators.parquet`
- breadth: `data/parquet/context/eodhd_500/market_breadth.parquet`
- sector returns: `data/parquet/context/eodhd_500/sector_returns.parquet`
- fundamentals: `data/parquet/context/eodhd_500/company_fundamentals.parquet`

| context | coverage |
|---|---:|
| macro rows | 13,036 |
| market breadth rows | 2,652 |
| sector returns rows | 34,202 |
| sector count | 12 |
| fundamentals rows | 1,135 |
| fundamentals ticker coverage | 98 / 503 = 19.48% |
| 1D has_macro true rate | 100.00% |
| 1D has_breadth true rate | 95.23% |
| 1D has_fundamentals true rate | 1.14% |
| 1W has_macro true rate | 100.00% |
| 1W has_breadth true rate | 100.00% |
| 1W has_fundamentals true rate | 1.24% |

해석:

- macro/breadth는 실제 값으로 채워져 있다.
- sector_returns parquet는 존재하지만 현재 36개 model feature에는 직접 컬럼으로 들어가지 않는다.
- fundamentals는 local SEC 기반 값이 일부만 들어가므로, full_features 후보는 “macro/breadth + 일부 fundamentals + missing flag”로 해석해야 한다.
- `context_light`는 아직 별도 feature_set으로 정의되지 않았다. 지금 급히 만들기보다 이번 sweep은 `price_volatility_volume`과 `full_features`의 해석 차이를 명확히 기록하고, 필요하면 다음 CP에서 `context_light`를 별도 ablation으로 만드는 편이 낫다.

금지 문구:

- full_features를 “펀더멘털을 충분히 활용한 모델”이라고 설명 금지.

허용 문구:

- “macro/breadth는 활성화되어 있고, fundamentals는 coverage가 낮아 missingness-aware context로만 제한 해석한다.”

## 6. Cache / Registry 검증

원천 hash:

- price manifest source_data_hash: `80dde8302f9e87befb0e60b0d666f13f863c0529a7f361a5d2d8119c50ffb9ce`
- 1D indicator checksum: `8e3a5d23e95a1f4b`
- 1W indicator checksum: `d98b27d90725442e`
- context hash: `1aa6452d82369cc6`

EODHD cache manifest:

| kind | timeframe | path | ticker_count | provider |
|---|---|---|---:|---|
| feature_index | 1D | `ai/cache/feature_index_1D_8b288711c67b_7db1be0f.pt` | 503 | eodhd |
| features | 1D | `ai/cache/features_1D_b841b673f7e6_33f6334e.pt` | 473 | eodhd |
| feature_index | 1W | `ai/cache/feature_index_1W_ad2ca650d771_e891fc62.pt` | 502 | eodhd |
| features | 1W | `ai/cache/features_1W_87a97b5fc0c0_141cef32.pt` | 441 | eodhd |

중요 WARN:

- generic latest cache 기준으로 보면 1W `feature_index`와 1W `features` 최신 manifest가 yfinance다.
- 따라서 CP150/CP151은 반드시 `market_data_provider=eodhd`와 EODHD 500 local snapshot alias를 명시해야 한다.
- 기존 EODHD 1W feature cache는 ticker_count 441이라 CP150 eligible 447, CP151 eligible 453과 맞지 않는다. 1W 실험 전 refresh_required다.

Manifest 없는 cache:

- manifest 없는 `.pt` cache 후보는 75개다.
- 삭제하지 않았다.
- 전체 목록은 `docs/cp148_0_dg2_data_feature_cache_final_gate_metrics.json`의 `cache.stale_manifest_missing_candidates`에 기록했다.

## 7. Registry 검증 및 1W 447 vs 446 원인

Scenario별 registry:

| CP | eligible | registry path | 상태 |
|---|---:|---|---|
| CP148-LM-1D | 473 | `ai/cache/ticker_id_map_1d_8737d2a1935a.json` | 기존 일치 |
| CP149-BM-1D | 476 | `ai/cache/ticker_id_map_1d_a2fcbcf4e338.json` | 기존 일치 |
| CP150-LM-1W | 447 | `ai/cache/ticker_id_map_1w_221af228cd24.json` | 신규 생성/확인 |
| CP151-BM-1W | 453 | `ai/cache/ticker_id_map_1w_4ea1266146e9.json` | 신규 생성/확인 |

모든 scenario registry는 zero-based continuous mapping을 만족한다.

Ticker order hash:

| CP | ticker_order_hash |
|---|---|
| CP148-LM-1D | `e07fc3b47dcda208` |
| CP149-BM-1D | `8eedbb564ff39415` |
| CP150-LM-1W | `f8ae1b50e3c5b356` |
| CP151-BM-1W | `6107f3b2dfae4792` |

1W 447 vs 446 원인:

- 이전 최신 1W registry: `ai/cache/ticker_id_map_1w_9915e57df87e.json`
- ticker count: 446
- CP150-LM-1W eligible: 447
- 차이 ticker: `SW`

CP151-BM-1W에서는 seq_len=60이라 eligible이 453으로 늘어나며, 기존 446 registry에 없던 ticker는 다음 7개다.

- `AMCR`
- `DELL`
- `FTV`
- `HWM`
- `SW`
- `TTD`
- `VST`

따라서 원인은 데이터 결함이 아니라 scenario별 seq_len/horizon/gate 차이를 반영하지 않은 이전 registry 재사용이다.

## 8. Split / Target 검증

| CP | split | eligible | train | val | test | target nonfinite | date overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| CP148-LM-1D | 1D h5 seq252 | 473 | 799,739 | 171,363 | 171,837 | 0 | 0 |
| CP149-BM-1D | 1D h5 seq60 | 476 | 864,386 | 184,807 | 185,738 | 0 | 0 |
| CP150-LM-1W | 1W h4/h6 seq104 | 447 | 122,808 | 26,346 | 26,796 | 0 | 0 |
| CP151-BM-1W | 1W h4 seq60 | 453 | 138,145 | 29,345 | 30,245 | 0 | 0 |

CP148-0 absolute minimum gate 반영:

- 1D absolute min rows: 450
- 1W absolute min rows: 78
- CP149-BM-1D excluded reason에 `insufficient_absolute_history_1D_min_450` 1건 반영
- CP151-BM-1W excluded reason에 `insufficient_absolute_history_1W_min_78` 1건 반영

Band target sanity:

- 현재 band target type은 `raw_future_return`이다.
- 별도 lower/upper label이 아니라 horizon future return 벡터를 쓰므로, 이번 audit에서는 future return finite와 horizon min/max order sanity로 확인했다.
- CP149/CP151 모두 target nonfinite 0, min/max order violation 0이다.

## 9. Tiny Forward Shape 검증

학습 없이 CPU-only로 작은 입력 한 batch에 대해 forward shape만 확인했다.

| 모델 | input shape | output shape | finite |
|---|---|---|---|
| PatchTST | `[2, 32, 36]` | line/lower/upper `[2, 5]` | true |
| CNN-LSTM | `[2, 32, 11]` | line/lower/upper `[2, 5]` | true |
| TiDE | `[2, 32, 36]`, future cov `[2, 5, 7]` | line/lower/upper `[2, 5]` | true |
| TCNQuantile | `[2, 32, 11]` | line/lower/upper `[2, 5]` | true |

첫 시도는 pandas/numpy import 뒤 Torch DLL 초기화 오류가 났고, 별도 프로세스에서 `bootstrap_torch(cpu_only=True)`를 가장 먼저 호출해 통과했다. sweep runner에서도 Windows에서는 Torch bootstrap 순서를 유지하는 것이 안전하다.

## 10. CP148~CP151 실행 가능 여부

| CP | 판정 | 조건 |
|---|---|---|
| CP148-LM-1D | 가능 | EODHD 1D feature/index cache와 473 registry 일치 |
| CP149-BM-1D | 가능 | 476 registry 사용. 기존 1D features cache는 473이라 새 BM run cache path 확인/재생성 권장 |
| CP150-LM-1W | 가능, refresh 후 | provider=eodhd 고정, 447 registry 사용, EODHD 1W feature cache refresh 필요 |
| CP151-BM-1W | 가능, refresh 후 | provider=eodhd 고정, 453 registry 사용, EODHD 1W feature cache refresh 필요 |

## 11. 최종 구분

### “데이터가 문제없다”라고 말해도 되는 범위

- EODHD 500 price parquet의 adjusted OHLC contract
- 1D/1W indicator finite contract
- duplicate/source mixing 없음
- 1W partial week 없음
- 1D h5, 1W h4/h6 target finite
- train/val/test date overlap 0
- PatchTST/CNN-LSTM/TiDE/TCNQuantile tiny forward shape

### 아직 WARN으로 해석해야 하는 범위

- fundamentals coverage가 낮은 full_features 해석
- `context_light` 미정의
- generic latest 1W cache가 yfinance인 점
- 기존 EODHD 1W features cache ticker_count와 CP150/151 eligible 불일치
- manifest 없는 cache 75개

### 실험 전에 사람이 확인해야 하는 항목

1. CP150/CP151 실행 명령에 `market_data_provider=eodhd`가 들어가는지 확인한다.
2. EODHD 500 snapshot alias 또는 명시 path가 사용되는지 확인한다.
3. 1W 실험 전 EODHD provider/source 기준 feature cache refresh가 일어나는지 확인한다.
4. full_features 결과를 발표할 때 fundamentals 기여를 과장하지 않는다.
5. generic latest cache 자동 선택 로직에 의존하지 않는다.

## 12. 최종 판정

**WARN**

1D/1W 모두 실험 가능한 데이터 상태다. 하지만 1W는 cache/registry refresh_required이고, full_features는 fundamentals coverage 때문에 해석상 WARN이 붙는다. CP148-LM-1D와 CP149-BM-1D는 진행 가능, CP150-LM-1W와 CP151-BM-1W는 EODHD cache refresh 확인 후 진행 가능하다.
