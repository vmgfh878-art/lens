# yfinance 데이터 전환 구조 리뷰 보고서

작성일: 2026-05-03
범위: CP86~CP90 흐름의 EODHD → yfinance 로컬 primary 전환 구조 전체 리뷰
모드: 코드 수정 / DB 쓰기 / 모델 학습 / provider 설정 변경 모두 금지. 읽기 전용 정적 분석 + CP 보고서 교차 검증.
관점: "yfinance가 위험하니 포기"가 아니라 "안전하게 전환하려면 무엇을 더 막아야 하는가."

## 1. Executive Summary

**현재 위치 평가: WARN, 즉시 전환 확대 금지.**

CP86~CP88은 provider abstraction과 cache 격리 *코드 계약*을 잘 깔았다. CP89가 `prepare_dataset_splits` 실패를 잡아낸 것도 의미 있다. 그러나 다음 두 가지가 동시에 깨져 있어, 50티커 write 확대와 live inference 연결을 *지금 시점에는* 차단해야 한다:

1. **`indicators` 계산 경로가 source-aware하지 않다.** `compute_indicators`는 `price_data`를 source 필터 없이 읽는다 (`backend/collector/jobs/compute_indicators.py:164-169`). 결과 `indicators` row는 **mixed-source rolling feature**를 담을 수 있고, `indicators` 테이블 자체에 source 컬럼이 없어 *DB 안에서 구분도 불가능*하다.
2. **`price_data` UNIQUE key가 여전히 `(ticker, date)`다** (`backend/db/schema.sql:33`, `backend/collector/jobs/sync_prices.py:308`). yfinance write는 같은 ticker/date의 EODHD row를 *조용히 덮어쓴다*. provider parallel row는 schema상 불가능하고, "yfinance write만 제거"하는 rollback은 EODHD 원본을 복구하지 못한다 (다시 fetch해야 함).

이 둘이 결합되면 다음이 발생한다 — **mixed-source indicator → mixed-source feature → 모델이 source 경계에서 discontinuity를 학습**. CP89에서 1D 5티커 1 epoch 짧은 검증은 yfinance window가 좁고 lookback이 작아 우연히 mixed가 안 되었지만, 1W (550일) / 1M (2100일) 또는 lookback 확장 시 mixed가 *반드시* 발생한다.

CP90은 새 기능 추가가 아니라 위 두 항목 + 부수 가드 보강에 집중해야 한다.

**좋았던 결정** (보존 권장):
- Provider Protocol 추상화와 `prepare_provider_price_frame` 정규화
- `validate_adjusted_ohlc_contract` 8단 gate
- cache manifest의 provider/hash mismatch 거부 정책
- `source_data_hash`에 provider + adjustment_policy + feature_contract_version 포함
- CP86 → 87 → 88 → 89 단계 분리와 *제한 write*만 허용한 점
- EODHD 코드 / fallback 보존, `auto_adjust=False` (CP29 교훈)

## 2. 리스크 표

P0 = CP90 안에서 반드시 고친다 / P1 = 50티커 write 전 / P2 = 전체 yfinance 전환 전 / P3 = 운영 정착기.

| ID | 등급 | 영역 | 리스크 | 근거 (파일:라인 또는 함수) |
|---|---|---|---|---|
| R-01 | **P0** | indicators | `compute_indicators`가 `price_data`를 source 필터 없이 읽는다. 같은 ticker에 EODHD + yfinance row가 공존하면 rolling MA/RSI/ATR이 source 경계를 가로질러 계산되고, 결과 indicator row는 source 컬럼이 없어 DB 안에서 구분 불가. | `backend/collector/jobs/compute_indicators.py:164-169` (price_frame fetch), `backend/db/schema.sql:126-167` (indicators 컬럼 목록 — source 없음) |
| R-02 | **P0** | split planning | `prepare_dataset_splits`가 `fetch_feature_index_frame`(indicators 전체 history, source-blind)으로 split plan을 만들고, `fetch_training_frames`(price_data, source-filtered)로 데이터셋을 만든다. 두 frame의 date range가 어긋나면 split spec이 dataset 범위 밖을 가리켜 train/val/test가 모두 empty. CP89 line/band smoke가 정확히 이 사유로 실패. | `ai/preprocessing.py:1532-1568` (prepare_dataset_splits), `ai/preprocessing.py:493-555` (fetch_feature_index_frame, source 필터 없음), `ai/preprocessing.py:710-716` (price_source_filter, training용에는 적용) |
| R-03 | **P0** | provenance | `price_data` UNIQUE key가 `(ticker, date)`. yfinance write가 같은 키의 EODHD row를 in-place 덮어씀. provider parallel row 불가, "yfinance만 제거" rollback이 원본 EODHD 복구를 보장하지 못한다. CP89 시점에 5티커 × 약 500일 = 2,500건 이미 덮어쓰기 발생. | `backend/db/schema.sql:33` (`UNIQUE (ticker, date)`), `backend/collector/jobs/sync_prices.py:308` (`on_conflict="ticker,date"`), `docs/cp89_yfinance_limited_write_indicator_validation_report.md:107` |
| R-04 | **P0** | provenance | legacy null source backfill 누락. CP88-D 보고서는 "기존 EODHD row 대량 backfill은 하지 않았다"고 명시. `_price_source_filter_clause`는 eodhd 쿼리에 한해 NULL을 EODHD로 해석하지만, 같은 행이 yfinance로 덮어써지면 source가 'yfinance'로 변하면서 EODHD 쿼리에서 빠진다. EODHD 쿼리 결과에 *덮어쓰기 hole*이 생기는데 lineage가 없어 추적 불가. | `ai/preprocessing.py:710-716`, `docs/cp88_yfinance_source_provenance_cache_isolation_report.md:34` ("backfill 하지 않았다") |
| R-05 | P1 | provenance / fingerprint | `_price_meta_query`의 `MAX(updated_at)`이 fingerprint payload에 들어간다. 그런데 `ALTER TABLE ... ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW()`는 마이그레이션 시점에 *기존 모든 행*에 NOW()를 채운다 (`ensure_runtime_schema.py:32-34`). 즉 마이그레이션 직후 EODHD fingerprint가 갑자기 갱신됨. 이후 yfinance write로 또 갱신. 마이그레이션~첫 안정화 구간 동안 EODHD cache가 무효화되며, 갱신 타이밍에 따라 fingerprint가 비결정적으로 흔들린다. | `ai/preprocessing.py:731-757` (price_meta_query), `backend/db/scripts/ensure_runtime_schema.py:32-34` |
| R-06 | P1 | sync logic | `_get_ticker_start_date`가 `get_latest_date('price_data', ticker=...)`를 source 필터 없이 호출. provider=yfinance인데 EODHD가 더 최근 row를 가지면 yfinance backfill 시작점이 EODHD 좌표에 종속됨. yfinance가 새 ticker에 대해 0부터 backfill해야 할 상황을 놓칠 수 있다. | `backend/collector/jobs/sync_prices.py:21-32` |
| R-07 | P1 | fallback | fallback이 일어나도 `record_source = fetch_result.provider` (실제 provider)이라 yfinance run 안에서도 일부 ticker가 `source='eodhd'`로 저장된다. `model_runs`나 `sync_state` *summary* meta에는 provider/fallback 의도가 남지만, *price_data row 자체*에는 "이 행은 yfinance run 중 fallback으로 들어온 EODHD 행"이라는 표시가 없다. 학습 파이프라인이 `source='yfinance'`만 읽으면 *이 ticker는 누락*되어 selection bias가 생기고, 그 사실이 fingerprint에 반영되지 않는다. | `backend/collector/jobs/sync_prices.py:301-307`, `backend/collector/sources/market_data_providers.py:182-203` |
| R-08 | P1 | provider switch | provider 선택이 단일 env var `MARKET_DATA_PROVIDER`로 *전 파이프라인 동시 전환*된다. daily_sync, daily_market_sync, bootstrap_backfill, ai/preprocessing.py가 모두 같은 env를 본다. 잘못된 환경에서 큰 universe write가 트리거될 위험. provider lock/safe-mode가 없다. | `backend/collector/pipelines/daily_sync.py:136-137`, `daily_market_sync.py:146-147`, `bootstrap_backfill.py:269-270`, `ai/preprocessing.py:674-675` |
| R-09 | P1 | dual cron | Render cron(원래 EODHD)과 로컬 cron(yfinance)이 같은 `price_data`를 쓰면, 같은 ticker/date에 대해 *마지막 write가 source를 결정*한다. 매일 source가 뒤바뀔 수 있고, fingerprint가 매일 바뀌어 cache가 매일 무효화. | `daily_market_sync.py:134-148`, `daily_sync.py:122-138` (둘 다 같은 `run_prices` 호출), `run_daily_local_market_sync.ps1:13` (env 강제) |
| R-10 | P1 | fingerprint REST 폴백 | `_price_meta_via_rest`가 fetch_frame을 source 필터 없이 호출하고 `source_column_present=False`만 반환. postgres 직접 연결 실패 시 fingerprint가 *전 source 혼합* 값으로 계산. provider isolation 약속이 degraded mode에서 깨진다. | `ai/preprocessing.py:760-803` |
| R-11 | P1 | fallback gate 비대칭 | dry-run gate 기준은 `fallback_used=False`이지만, `--write` 본 실행은 fallback 사용 여부와 무관하게 진행됨. dry-run에서 본 12티커 PASS와 write 시 실제 fallback 비율이 다를 수 있다. | `yfinance_price_sync.py:233-237` (dry-run pass 조건에 fallback 포함), `yfinance_price_sync.py:346-377` (write 본 실행, fallback gate 재적용 없음) |
| R-12 | P2 | error log leak | provider 실패 시 `f"{candidate}:{type(exc).__name__}:{exc}"` 형태로 sync_state.meta에 저장. 일부 yfinance/EODHD 예외 메시지에 URL/key/cookie 단편이 들어갈 수 있어 sanitize 필요. | `backend/collector/sources/market_data_providers.py:179` |
| R-13 | P2 | ratio sanity dead gate | `validate_adjusted_ohlc_contract`의 `P99_RATIO_ABS_LIMIT=1.0` (= 100% 일중 변동). 미국 주식 p99는 사실상 5% 이내. 사실상 트립 안 됨. corporate action 일자가 *그대로* 통과 가능. | `backend/collector/sources/price_contract.py:13-14` |
| R-14 | P2 | metrics path 충돌 | `sync_yfinance_prices.ps1` 기본 `--metrics-path docs/cp86_yfinance_local_primary_migration_metrics.json`. 이후 매 yfinance run이 CP86 archive metrics를 덮어씀 → CP86 reference artifact 손실. | `scripts/sync_yfinance_prices.ps1:6`, `scripts/run_daily_local_market_sync.ps1:8` |
| R-15 | P2 | EODHD baseline 미동결 | CP86/87/88 baseline이 "현재 DB"를 그대로 사용. EODHD가 daily sync로 갱신되면 baseline 자체가 움직인다. cutover 전 EODHD snapshot artifact (parquet 등) 동결 권장. | `yfinance_price_sync.py:167-180` (load_baseline_price_frame은 라이브 DB), CP86 보고서 §11 ("baseline manifest와 hash snapshot을 cutover artifact로 고정") |
| R-16 | P2 | yfinance pin 갱신 압박 | `yfinance==0.2.58`이 Yahoo 응답 변경에 따라 깨질 수 있고 (CP86에서 0.2.40이 깨졌음), 1.x로 가면 `websockets>=13` 요구로 Supabase realtime과 충돌. provider 자체 깨짐 시 fallback이 자동 흡수해도 alarm 경로 없음. | `docs/cp86_yfinance_local_primary_migration_report.md:209-217` |
| R-17 | P3 | indicator lookback × provider 경계 | 1W (`source_history_days=550`), 1M (`source_history_days=2100`)는 lookback이 yfinance write window를 가로지른다. R-01과 결합되면 1W/1M indicator는 *반드시* mixed-source. | `backend/collector/jobs/compute_indicators.py:14-16` |
| R-18 | P3 | EODHD price_data history loss after rollback | yfinance write가 EODHD를 덮어쓴 뒤 rollback 시 `DELETE WHERE source='yfinance'`로 yfinance만 지워도 EODHD 원본은 *복구되지 않음*. 다시 EODHD에서 fetch해야 함. 사전 snapshot 없으면 cost와 시간이 든다. | R-03 + R-04 결합 |
| R-19 | P3 | live inference 미정의 | local primary 전환 후 live inference가 yfinance feature를 쓸지 EODHD를 쓸지 정책 부재. CP86/87 보고서가 "live inference는 다음 단계" 한 줄로 미룸. provider별 cache 분리는 됐지만 *어느 cache를 product run이 본다*는 결정이 빠짐. | `cp86_..._report.md:235`, CP90 instruction 8번 |
| R-20 | P3 | bootstrap_backfill 동일 위험 | `bootstrap_backfill.py`도 daily_sync와 같은 `run_prices` 호출 + 같은 env. 향후 backfill 단계에서 동일 issue 재발. | `bootstrap_backfill.py:258-272` |

## 3. 가장 위험한 데이터 혼합 시나리오

### 시나리오 A — Mixed-source rolling feature (가장 위험)

전제: schema migration 적용 완료, 5~50티커 yfinance 제한 write 진행, 1W/1M indicator 재계산 트리거.

1. yfinance write가 `[2024-05-03, 2026-05-01]` 범위에서 `(ticker, date)` 키로 EODHD row를 덮어씀. 같은 ticker의 `[~, 2024-05-02]` 범위 EODHD row는 그대로 남는다.
2. `compute_indicators_cli --timeframes 1W` 호출. 1W의 `source_history_days=550`이라 lookback이 약 1.5년. lookback window가 *yfinance/EODHD 경계*를 가로지른다.
3. `compute_indicators` price_frame fetch (R-01)는 source 필터가 없으므로 EODHD pre-2024-05-03 + yfinance post-2024-05-03을 *하나의 시계열로* 읽는다.
4. AAPL/MSFT의 adjusted_close 정책 차이가 약 0.7%~1.3% (CP86 보고서 §5)이고, NFLX는 split adjustment 정책 차이로 raw close가 0.9 단위까지 다름.
5. 이 mixed series 위에서 RSI, MACD, MA, ATR, BB가 계산되고, 경계 근처에 *인공 점프*가 들어간다.
6. 결과 `indicators` row는 source 컬럼이 없어 (R-01) DB에서 구분 불가능. 이후 어느 학습이 이 row를 읽었는지 추적도 불가능.
7. 학습 파이프라인은 `_price_source_filter_clause`로 `source='yfinance'` price만 읽지만, `feature_df`(indicators) 측은 source-blind라서 mixed indicator를 그대로 사용. **price와 indicator가 다른 source 경계를 가짐**.
8. 모델은 source 경계의 인공 점프를 학습 신호로 받아들임. CP9.5/CP11에서 잡았던 NaN과 다른 종류의 *조용한 데이터 오염*이라 finite gate에도 안 걸린다.

### 시나리오 B — Render + 로컬 cron race

1. Render cron이 04:00 UTC에 daily_sync (env=eodhd) 실행, 오늘자 EODHD row를 `(ticker, date)`에 upsert.
2. 같은 날 09:00 UTC에 로컬 `run_daily_local_market_sync.ps1` 실행 (env=yfinance). 같은 오늘자 row를 yfinance로 덮어씀.
3. 다음 날 Render가 어제+오늘 EODHD를 다시 upsert. 어제자 yfinance가 EODHD로 다시 바뀜.
4. `price_data` 안에서 같은 ticker의 인접 일자가 *수일 단위로 source가 토글*되는 상태가 된다.
5. yfinance fingerprint = (price meta MAX(updated_at) + checksum)이 매 cron마다 바뀌고, EODHD fingerprint도 같이 흔들림.
6. cache manifest가 매일 invalidate → feature/index cache 매일 재생성 → 학습 reproducibility 사라짐.
7. 더 나쁜 경우: Render의 EODHD upsert가 `source='eodhd'` 기록을 유지하면 R-04가 일부 완화되지만, 현재 schema migration이 아직 Render에 적용 안 되어 있다면 Render 측 record는 source 컬럼을 채우지 못해 NULL로 들어간다 → R-04 폭발.

### 시나리오 C — Rollback 시 EODHD hole

1. CP90 이후 yfinance 50티커 write 확장, 운영 1주 진행.
2. yfinance에 데이터 품질 문제 발견, "yfinance row 전체 제거"를 결정.
3. `DELETE FROM price_data WHERE source='yfinance'` 실행 → 약 50티커 × 1년 = 약 12,500 row 제거.
4. 하지만 같은 (ticker, date)에 원래 있던 EODHD row는 yfinance write로 덮어쓰여졌다 (R-03). 즉 EODHD 데이터가 *영구 손실*.
5. 복구를 위해 EODHD 재 fetch가 필요. EODHD API quota / 시간 / 비용이 든다. Phase 1 EODHD 한도 안에서 가능한지 사전 확인 안 됨.
6. snapshot 동결 (R-15)이 안 되어 있어 *손실 전 상태가 무엇이었는지*도 확정 못 함.

## 4. CP90에서 반드시 고쳐야 할 것

CP90의 핵심 5건. 새 기능이 아닌 *기존 결함 봉합*.

1. **R-01 fix — `compute_indicators` source-aware**
   - `backend/collector/jobs/compute_indicators.py`의 `fetch_frame("price_data", ...)`에 active provider 기준 source 필터 추가. legacy null은 EODHD로 해석.
   - `indicators` 테이블에 `source VARCHAR(30)` 컬럼 추가하고 새 indicator row에 채움. UNIQUE key를 `(ticker, timeframe, date, source)`로 확장하거나, *active provider의 source만 indicator를 갱신*하도록 정책 결정.
   - migration 시 기존 indicator row의 source는 NULL로 남기고 *deprecated/legacy*로 표시 — 새 indicator만 신뢰.

2. **R-02 fix — split planning을 source-aware**로**
   - `prepare_dataset_splits`가 `build_dataset_plan` 호출 전에 *price availability와 merge된 frame*에서 ticker/date eligible을 결정하도록 변경. 또는 `fetch_feature_index_frame`에 `provider` 파라미터를 받아 indicators ↔ price_data inner join 결과로 index를 만듦.
   - eligible window를 `min(price_min_date, indicator_min_date)` ~ `max(price_max_date, indicator_max_date)`가 아니라 `intersect`로 좁힘.
   - CP89 5티커 line/band smoke가 PASS될 때까지 검증 — 이게 fix 여부 판정 gate.

3. **R-03 fix — `price_data` UNIQUE key 확장**
   - `(ticker, date)` → `(ticker, date, source)` 변경. provider parallel row를 schema 차원에서 허용.
   - `sync_prices.py:308`의 `on_conflict="ticker,date"`를 `"ticker,date,source"`로 변경.
   - 이전 unique constraint drop은 별도 migration 단계로 안전하게 (현 yfinance/EODHD overlapping row가 없는 상태에서만).

4. **R-04 fix — legacy null source backfill 정책 확정 + 실행**
   - 옵션 A: 기존 NULL source row를 모두 `source='eodhd'`로 backfill.
   - 옵션 B: NULL을 EODHD로 해석하는 코드 정책을 *모든* read 경로에 일관 적용 (현재는 `_price_source_filter_clause`만).
   - 둘 중 하나 *명시적 결정*. 보고서로 남기고 cutover gate에 포함.

5. **부수 — R-05 / R-06 / R-10 동시 수정**
   - `_get_ticker_start_date` source-aware (yfinance latest 따로).
   - `_price_meta_via_rest`도 source 필터 적용.
   - `ALTER TABLE ... ADD COLUMN updated_at` 마이그레이션 시 *백필 시점을 별도로 표시* (예: created_at으로 backfill, updated_at은 NULL 허용 + COALESCE 사용).

CP90 안에서 위 5개를 하나의 CP로 묶어도 좋고, R-01 / R-02 / R-03 세 핵심을 CP90-A, R-04와 부수를 CP90-B로 나눠도 된다. 50티커 write 전에 *전부 통과*되어야 한다.

## 5. 50티커 write 전 필수 gate

CP90 fix 완료 후, 50티커 write 직전에 통과해야 할 gate:

| Gate | PASS 조건 | 검증 방법 |
|---|---|---|
| G-1 | R-01 ~ R-04 fix 완료 | 단위 테스트 + integration test (mixed source price → indicator → split smoke) |
| G-2 | EODHD baseline snapshot 동결 | `SELECT * FROM price_data WHERE source IN ('eodhd', NULL)` parquet export, hash 기록, CP90 metrics에 첨부 |
| G-3 | rollback drill 1 ticker | 임시 ticker (e.g. AMD)로 yfinance write → indicator 재계산 → rollback `DELETE WHERE source='yfinance'` → EODHD 재 fetch → row count 일치 확인 |
| G-4 | provider parallel row 검증 | 같은 (ticker, date)에 EODHD/yfinance row 동시 존재 가능, 각각 fingerprint 분리 확인 |
| G-5 | CP89 line/band smoke 재실행 PASS | 5티커 yfinance window에서 split-empty 없이 학습 1 epoch 완료. NaN 없음. coverage 합리적 |
| G-6 | EODHD fallback parallel coverage 1주 | `MARKET_DATA_PROVIDER=yfinance, MARKET_DATA_FALLBACK_PROVIDER=eodhd`로 5티커 daily sync를 7일간 실행, fallback_used 카운트와 fingerprint 안정성 기록 |
| G-7 | metrics path 격리 | `--metrics-path` 기본값을 CP archive에서 `logs/yfinance_runs/` 또는 timestamped 경로로 변경 |
| G-8 | provider lock/safe mode | env 변경이 *명시적 confirm 없이* 큰 universe write를 트리거하지 못하도록 가드. 최소한 `--write --universe`가 둘 다 필요한 현 구조를 유지하고, env-only 자동 trigger 없는지 재확인 |

## 6. 전체 yfinance 전환 전 필수 gate

50티커 write 1~2주 안정화 이후, Render cron 또는 product run 교체 전 통과 gate:

| Gate | PASS 조건 |
|---|---|
| G-9 | Render env과 로컬 env가 *서로 다른 provider*를 보지 못하도록 lock. 최소한 하나의 cron만 active. dual-write race (시나리오 B) 차단 |
| G-10 | local daily sync 1개월 안정 — fingerprint drift, fallback rate, indicator coverage 90일 통계로 판정 |
| G-11 | EODHD fallback 실제 trigger 검증 — 의도적으로 yfinance를 한 ticker 차단 후 fallback 작동, model_runs/sync_state에 lineage 보존 확인 |
| G-12 | live inference cache 결정 — product run이 어느 provider cache를 보는지 명문화. cache hot-swap 절차 정의 |
| G-13 | yfinance 장애 SLA — Yahoo HTML/API 변경 시 alarm + auto-fallback 동작 확인. 테스트 케이스 (`yf.download` mock failure → fallback path 진입) 추가 |
| G-14 | rollback 절차 문서화 — yfinance row 제거 + EODHD 재 fetch full procedure 1쪽 SOP |
| G-15 | EODHD pin 정책 — Plan v3는 EODHD Phase 1~2 유지로 박혀 있음 (`memory/project_lens_plan_v3.md`). yfinance 전체 전환은 *Plan v3 명시적 변경*이 필요. 사용자 승인 + CP 명세 |
| G-16 | bootstrap_backfill 동일 fix 적용 (R-20) |

## 7. 좋았던 구조적 선택

리뷰 중 *유지/확장 권장*인 항목:

- **`MarketDataProvider` Protocol** (`market_data_providers.py:20-31`) — provider 추가가 클래스 한 개 추가로 끝남. 깔끔.
- **`prepare_provider_price_frame`** (`price_contract.py:32-48`) — provider 응답 정규화 (Volume null → 0, Amount 자동 계산, MultiIndex 평탄화)가 한 곳에 모임. Volume null 정책을 metric에도 명시 (`volume_null_policy` 필드).
- **`validate_adjusted_ohlc_contract` 8단 gate** — date null/duplicate, required null/non-finite, volume negative, adjusted_factor invalid, adjusted high/low/open/close 정합성, ratio p99/max sanity. 충실. 단 R-13의 임계값 보강 필요.
- **cache manifest의 8 키 mismatch 검사** (`is_cache_manifest_valid`, `preprocessing.py:951-969`) — provider/policy/hash/feature_version/columns/timeframe까지 본다. CP86~88에서 가장 잘 깔린 부분.
- **`source_data_hash` payload** (`resolve_data_fingerprint`, `preprocessing.py:805-895`) — provider, adjustment_policy, feature_contract_version, ticker_universe_fingerprint, date_range, checksum, updated_at 모두 포함. 단 R-05 / R-10 보강 필요.
- **CP86~89 단계 분리** — provider 추가 → 3단계 검증 → provenance/cache → 제한 write로 차근차근. Migration 사례 일반론 (CP88-G playbook)과 정합.
- **EODHD 코드와 fallback 보존** — `auto_adjust=False` (CP29 교훈), EODHD 즉시 삭제 안 함, fallback provider 명시. 비용보다 fidelity 원칙 (Plan v3) 정합.
- **테스트** `test_market_data_providers.py` — provider 출력 형식, dry-run compare, source provenance record까지 단위 테스트로 잡음.
- **`test_preprocessing_cache_isolation.py`** — provider별 cache path/hash 분리, manifest mismatch 거부. cache 격리 핵심 테스트.

## 8. 부족한 테스트

다음은 현재 fixture가 *재현하지 못하는* 위험. P1/P2 우선순위로 추가 필요.

| 테스트 ID | 우선순위 | 목적 | 시나리오 |
|---|---|---|---|
| T-1 | P1 | R-01 회귀 방지 | 같은 ticker에 EODHD pre-T0 + yfinance post-T0 row를 가진 가짜 price_data를 만들고 `compute_indicators` 실행. 결과 indicator row가 source-aware하게 계산되거나 *명시적으로 거부*되는지 |
| T-2 | P1 | R-02 회귀 방지 | feature_index_frame이 indicators 전체 history를 가지고, price_df는 yfinance 좁은 window만 가질 때 `prepare_dataset_splits`가 split-empty 없이 통과하는지. CP89 실패 재현 → fix 검증 |
| T-3 | P1 | R-03 회귀 방지 | price_data UNIQUE 정책이 `(ticker, date, source)`로 변경된 후, 같은 (ticker, date)에 EODHD/yfinance 두 row 동시 저장 가능 + 각 read query가 자기 source만 가져오는지 |
| T-4 | P1 | R-07 회귀 방지 | provider=yfinance 호출 중 ticker 한 개가 yfinance 실패 → fallback EODHD로 넘어가 row 저장. 이때 price_data row의 source 컬럼이 'eodhd'로 들어가고, sync_state.meta에 `fallback_used=True`가 기록되는지. 추가로 학습 측이 이 mixed run을 감지할 수 있도록 fingerprint에 `fallback_count > 0` 표시가 들어가는지 |
| T-5 | P1 | R-04 회귀 방지 | legacy NULL source row가 있는 상태에서 EODHD 쿼리는 NULL 포함, yfinance 쿼리는 NULL 제외. 같은 row가 yfinance write로 덮어써진 후 EODHD 쿼리에서 빠지는 동작이 *의도된 정책*임을 명시한 단위 테스트 (정책 문서 + assertion) |
| T-6 | P2 | R-05 회귀 방지 | `updated_at`이 NULL인 legacy row가 섞인 price_data에서 fingerprint가 결정적인지. ALTER TABLE migration 시뮬레이션 |
| T-7 | P2 | R-13 회귀 방지 | adjusted_high가 close보다 30% 이상 떨어진 가짜 corporate action row가 들어왔을 때 `validate_adjusted_ohlc_contract`가 잡는지. 현재 ratio limit=1.0이라 안 잡힘 |
| T-8 | P2 | R-09 race 시뮬레이션 | 같은 ticker/date에 EODHD upsert → yfinance upsert → EODHD upsert 순으로 호출. 최종 row의 source와 fingerprint 변화 검증 |
| T-9 | P3 | R-18 rollback drill | yfinance write → EODHD 재 fetch 시나리오 단위 테스트 (mocked source) |
| T-10 | P3 | R-12 leak prevention | provider exception 메시지에 sentinel 토큰을 넣고 sync_state.meta에 그 토큰이 *나타나지 않는지* (sanitizer 작동) |

추가로, CP86~89 fixture는 모두 5~50티커 *PASS 케이스* 위주. **failure injection** (provider 응답 NaN, 행 누락, 잘못된 timezone, 갑작스러운 row count drop)을 일부러 주는 fixture가 없다. CP90에 1~2건 도입 권장.

## 9. 코드 수정 / DB 쓰기 / 모델 학습 / provider 설정 변경 미수행 확인

본 리뷰는 다음을 *모두* 수행하지 않았다.

- **코드 수정**: 0건. Edit / Write로 ai/, backend/, scripts/ 어떤 파일도 변경하지 않았다. `docs/yfinance_migration_architecture_review_report.md` *신규 작성*만 수행.
- **DB 쓰기**: 0건. `python -m backend.db.scripts.ensure_runtime_schema`, `yfinance_price_sync --write`, `compute_indicators_cli` 모두 실행하지 않았다.
- **모델 학습**: 0건. `python -m ai.train`, `ai.sweep` 실행하지 않았다.
- **provider 설정 변경**: 0건. `MARKET_DATA_PROVIDER`, `MARKET_DATA_FALLBACK_PROVIDER` env 변경 없음. `.env`, `backend/collector/config.py` 변경 없음.
- **테스트 실행**: 0건. unittest 실행하지 않았다 (정적 read-only 분석만).
- **shell 명령**: `git status`, `ls`, `grep`, `wc` 등 *읽기/조회만* 사용. 어떤 데이터/파일도 변경하지 않았다.

## 10. 읽기 전용으로 확인한 파일 목록

CP86~CP90 흐름 및 관련 수직 파일을 읽었다. 굵은 항목은 *전체 read*, 나머지는 grep + spot-check.

문서:
- **docs/cp86_yfinance_local_primary_migration_report.md** (전체)
- **docs/cp87_yfinance_three_stage_validation_report.md** (전체)
- **docs/cp88_yfinance_source_provenance_cache_isolation_report.md** (전체)
- **docs/cp89_yfinance_limited_write_indicator_validation_report.md** (전체)
- **docs/cp88_data_provider_migration_case_study_playbook.md** (전체)

provider / contract / sync 코드:
- **backend/collector/sources/market_data_providers.py** (전체)
- **backend/collector/sources/price_contract.py** (전체)
- **backend/collector/jobs/sync_prices.py** (전체)
- backend/collector/jobs/compute_indicators.py (필요 부분: price fetch / source 처리)

파이프라인 / 스크립트:
- **backend/collector/pipelines/yfinance_price_sync.py** (전체)
- **backend/collector/pipelines/daily_market_sync.py** (전체)
- backend/collector/pipelines/daily_sync.py (provider 호출부)
- backend/collector/pipelines/bootstrap_backfill.py (provider 호출부)
- backend/collector/pipelines/compute_indicators_cli.py (인자 / 호출부)
- **scripts/sync_yfinance_prices.ps1** (전체)
- **scripts/run_daily_local_market_sync.ps1** (전체)

DB schema:
- **backend/db/schema.sql** (전체)
- **backend/db/scripts/ensure_runtime_schema.py** (필요 부분: price_data 컬럼, model_runs status 마이그레이션)

AI 측 데이터 계약:
- ai/preprocessing.py (provider/cache/fingerprint/split planning 핵심 구간 — line 35, 390-560, 670-1064, 1324-1572)

테스트:
- **ai/tests/test_preprocessing_cache_isolation.py** (전체)
- **backend/tests/test_market_data_providers.py** (전체)
- backend/tests/test_collector_jobs.py (provider/source 관련 grep)

메모리 (자가 리뷰 정합 확인용):
- C:\Users\user\.claude\projects\C--Users-user-lens\memory\project_lens_plan_v3.md
- C:\Users\user\.claude\projects\C--Users-user-lens\memory\project_lens_workflow.md

추가 grep 대상 (수정 사항 없음 확인용):
- ai/inference.py / ai/backtest.py — provider 의존성 없음 확인
- backend/collector/pipelines/data_coverage_report.py — 본 리뷰에서는 spot-check만

---

## 부록 A — Plan v3 정합 자가 점검

`memory/project_lens_plan_v3.md`는 EODHD Phase 1~2 유지 / 공개 배포 시 FMP Pro 검토라고 박혀 있음. yfinance 전체 전환은 명시적으로 다루지 않음. CP86~CP89 흐름은 "개인 로컬 운영 비용 절감을 위한 *primary 후보*"로 한정해 Plan v3와 충돌하지 않음 (EODHD fallback 유지 = Plan v3 정합). 그러나 **G-15** (전체 전환 gate)는 Plan v3 명시적 갱신 + 사용자 승인이 선행되어야 한다. 이것을 빼먹고 Render cron까지 yfinance로 넘기면 Plan v3 위반.

## 부록 B — 한 줄 결론

**현 상태에서 50티커 write 확장 / Render cron 전환 / live inference 연결은 모두 차단해야 한다.** CP90은 R-01 ~ R-04 fix 완료까지로 좁히고, fix 검증 후에야 50티커 write를 재개한다. Provider 추상화·cache 격리·source provenance 코드 계약은 잘 깔렸으니, 그 위에 *indicators source-awareness*와 *price_data parallel row*만 얹으면 안전한 cutover가 가능하다.
