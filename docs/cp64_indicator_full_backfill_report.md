# CP64-D ATR full backfill 및 indicator 가격 피처 재계산 보고서

## 1. Executive Summary

CP64-D는 모델 학습 없이 `public.indicators`의 ATR 계열과 오래된 가격 ratio 이상치를 정리한 운영 데이터 정합성 작업이다. 계산은 `backend.collector.jobs.compute_indicators.run()` 공식 경로만 사용했고, ad-hoc SQL 계산식으로 indicator 값을 우회 생성하지 않았다.

핵심 결과는 성공이다. `atr_ratio` non-null은 407,861 / 1,584,191 rows에서 1,649,317 / 1,649,317 rows로 올라갔다. 1D 최신 coverage는 AAPL 포함 503 / 503, 1W는 502 / 502, 1M은 494 / 494다. 1D `open_ratio/high_ratio/low_ratio` 폭주는 p99 기준 약 14배 수준에서 0.05-0.08 수준으로 정상화됐다.

중요한 발견도 있었다. 단순 upsert만으로는 과거 split/adjusted 계약 이전 stale row가 제거되지 않았다. 따라서 `force_full_backfill=True`일 때만 ticker/timeframe/source range의 기존 indicator row를 삭제한 뒤 공식 계산 결과를 다시 upsert하도록 `compute_indicators`를 보강했다.

모델 feature contract는 바꾸지 않았다. `MODEL_N_FEATURES=36` 유지, `atr_ratio in MODEL_FEATURE_COLUMNS=False`다. `atr_ratio`는 이번 CP에서 차트 보조지표 및 향후 BM feature 승격 후보로만 정리했다.

## 2. 읽기/수정 범위

수정한 파일:

- `backend/collector/jobs/compute_indicators.py`
- `backend/tests/test_collector_jobs.py`
- `docs/cp64_indicator_full_backfill_report.md`
- `docs/cp64_indicator_full_backfill_metrics.json`
- `docs/project_journal.md`
- `docs/model_architecture.md`
- `docs/training_hyperparameters.md`

읽기 전용으로 확인한 파일:

- `backend/app/services/feature_svc.py`
- `ai/preprocessing.py`
- `backend/db/scripts/ensure_runtime_schema.py`
- `backend/collector/pipelines/daily_market_sync.py`
- `render.yaml`

금지 항목 확인:

- 모델 학습 없음
- save-run 없음
- AI checkpoint 생성 없음
- 프론트/UI 수정 없음
- backend API schema 변경 없음
- fake data 생성 없음
- ad-hoc SQL 계산식 우회 없음

## 3. 계산 경로 계약 확인

`feature_svc.build_features()`는 `atr_ratio`를 output에 포함한다. 근거는 `backend/app/services/feature_svc.py:95-96`의 `_INDICATOR_ONLY_COLUMNS = ["atr_ratio"]`와 `_OUTPUT_COLUMNS`다. `atr_ratio` 계산은 같은 파일 `backend/app/services/feature_svc.py:315`에서 `atr / close`로 수행된다.

모델 입력 feature에는 포함되지 않는다. `ai/preprocessing.py:25`는 `FEATURE_COLUMNS`를 `SOURCE_FEATURE_COLUMNS`로 가져오고, `ai/preprocessing.py:44-46`은 `MODEL_FEATURE_COLUMNS = SOURCE_FEATURE_COLUMNS + calendar`, `MODEL_N_FEATURES = len(MODEL_FEATURE_COLUMNS)`로 구성한다. `atr_ratio`는 `FEATURE_COLUMNS`가 아니라 indicator-only output이므로 모델 입력 36개에는 들어가지 않는다.

가격 ratio 재계산은 adjusted OHLC 기준이다. `backend/app/services/feature_svc.py:161-177`에서 `adjusted_close / close` factor를 적용해 open/high/low/close를 adjusted 기준으로 통일하고, `backend/app/services/feature_svc.py:210`, `backend/app/services/feature_svc.py:246`, `backend/app/services/feature_svc.py:287` 경로에서 1D/1W/1M feature 계산에 같은 계약을 적용한다.

`compute_indicators`는 `backend/collector/jobs/compute_indicators.py:187`에서 `build_features()`를 호출하고, `backend/collector/jobs/compute_indicators.py:226`에서 `indicators`에 upsert한다. CP64-D에서는 `force_full_backfill=True`일 때 stale row 제거를 위해 `backend/collector/jobs/compute_indicators.py:75-88`, `backend/collector/jobs/compute_indicators.py:218-224`의 prune 경로를 추가했다.

`atr_ratio` 컬럼은 runtime schema에 존재한다. `backend/db/scripts/ensure_runtime_schema.py:56`에서 `ADD COLUMN IF NOT EXISTS atr_ratio DOUBLE PRECISION`을 확인했다.

## 4. 백필 전/후 비교

| timeframe | 항목 | 백필 전 | 백필 후 | 판단 |
|---|---:|---:|---:|---|
| 전체 | rows | 1,584,191 | 1,649,317 | 재계산 후 stale 범위 정리와 현재 feature output 기준으로 증가 |
| 전체 | atr_ratio non-null | 407,861 | 1,649,317 | 전 row coverage 확보 |
| 1D | atr latest coverage | 0 / 503 | 503 / 503 | 성공 |
| 1W | atr latest coverage | 502 / 502 | 502 / 502 | 유지 |
| 1M | atr latest coverage | 494 / 494 | 494 / 494 | 유지 |
| 1D | open_ratio abs p99 / max | 14.0473 / 77.3238 | 0.0485 / 0.8109 | 폭주 제거 |
| 1D | high_ratio abs p99 / max | 14.2371 / 82.7344 | 0.0760 / 0.8471 | 폭주 제거 |
| 1D | low_ratio abs p99 / max | 13.9232 / 77.0295 | 0.0752 / 0.6941 | 폭주 제거 |
| 1W | open_ratio abs p99 / max | 11.6508 / 50.7422 | 0.0505 / 0.5524 | 폭주 제거 |
| 1W | high_ratio abs p99 / max | 12.0451 / 64.2207 | 0.1673 / 1.5573 | 폭주 제거, 주간 real outlier 잔존 |
| 1W | low_ratio abs p99 / max | 11.2991 / 50.5597 | 0.1543 / 0.7118 | 폭주 제거 |
| 1M | open_ratio abs p99 / max | 0.0505 / 0.3170 | 0.0505 / 0.3170 | 기존에도 정상 범위 |
| 1M | high_ratio abs p99 / max | 0.3537 / 2.3712 | 0.3540 / 2.3712 | 월간 real outlier 감시 |
| 1M | low_ratio abs p99 / max | 0.3180 / 0.8870 | 0.3180 / 0.8870 | 월간 real outlier 감시 |
| 1D | atr_ratio p99 / max | 0.0733 / 0.2402 | 0.0811 / 0.8106 | 전체 coverage 반영 후 정상 |
| 1W | atr_ratio p99 / max | 0.1636 / 1.0966 | 0.1719 / 1.0966 | 정상, 고변동 주간 outlier 유지 |
| 1M | atr_ratio p99 / max | 0.3671 / 8.4681 | 0.3671 / 8.4681 | 월간 고변동 outlier 별도 감시 |

비정상 가격 ratio의 원인은 `raw open/high/low`와 adjusted 기준 계산물이 섞여 남은 과거 indicator row였다. CP29 이후 학습 cache는 이미 adjusted OHLC v3 계약을 사용했지만, DB `indicators`에는 예전 계산 결과가 남아 있었다.

## 5. Batch 실행 내역

공통 실행 형태:

```powershell
$env:PYTHONPATH='C:\Users\user\lens'
.venv\Scripts\python.exe -c "from backend.collector.jobs.compute_indicators import run; run(lookback_days=60, tickers=<batch>, force_full_backfill=True, full_start_date='2015-01-01', timeframes=[<timeframe>])"
```

실제 batch 요약:

| timeframe | 계획 batch | 실행 결과 |
|---|---:|---|
| 1D | 120 ticker 이하 | 최초 4개 대형 batch 성공, 마지막 23 ticker는 소배치 5/5/5/5/3으로 재시도 성공 |
| 1D repair | 50 ticker 이하, 이후 15 ticker | stale row 제거 방식 보강 전 null repair 160 ticker, 보강 후 최종 null repair 15 ticker 성공 |
| 1W | 40 ticker 이하 | 13개 batch 성공, 이후 AMCR/SW repair 2 ticker 성공 |
| 1M | 25 ticker 이하 | 21개 batch 성공 |

세부 batch 수치와 duration은 `docs/cp64_indicator_full_backfill_metrics.json`의 `backfill_execution`에 기록했다. 명시적 statement timeout은 관찰하지 않았고, 1D 마지막 batch는 큰 단위 실행 종료 응답이 안정적이지 않아 소배치로 줄여 재시도했다.

## 6. 백필 후 검증

최종 DB 상태:

| timeframe | rows | atr_ratio non-null | latest date | latest atr coverage | non-finite |
|---|---:|---:|---|---:|---:|
| 1D | 1,354,123 | 1,354,123 | 2026-04-29 | 503 / 503 | 0 |
| 1W | 258,410 | 258,410 | 2026-05-01 | 502 / 502 | 0 |
| 1M | 36,784 | 36,784 | 2026-04-30 | 494 / 494 | 0 |

AAPL 최신 sample:

| timeframe | date | open_ratio | high_ratio | low_ratio | atr_ratio |
|---|---|---:|---:|---:|---:|
| 1D | 2026-04-29 | -0.011673 | 0.001219 | -0.013557 | 0.020635 |
| 1W | 2026-05-01 | -0.018335 | 0.008006 | -0.022098 | 0.050080 |
| 1M | 2026-04-30 | 0.001143 | 0.086607 | -0.031877 | 0.101143 |

1D 학습용 feature cache와 DB indicators 비교도 정상이다. AAPL cache `ai\cache\features_1D_ee9615c5c6cb_0c1d7f52.pt`의 abs p99/max는 DB AAPL 1D와 사실상 일치했다. 예를 들어 open_ratio abs p99는 cache 0.0468585, DB 0.0468542이고 high/low도 같은 범위다. row 수 1개 차이는 cache snapshot/index 경계 차이로 보이며 ratio sanity에는 영향이 없다.

## 7. 기간 종료 라벨 점검

1W latest date는 2026-05-01이고, CP64-D 실행 기준일은 2026-04-30이다. 이는 미래 raw trade date가 아니라 주간 resample의 금요일 period-end label로 판단한다. 1M latest date 2026-04-30도 월말 period label이다.

다만 제품/보고서에서는 1W/1M label이 실제 관측일처럼 보일 수 있다. 따라서 향후 chart/API 문서에서는 `date`가 resample period end label인지, 마지막 실거래 관측일인지 구분하는 메타가 필요하다. 이번 CP에서는 backend API schema 변경 금지 때문에 수정하지 않았다.

## 8. 남은 극단값과 원인 추정

1D에는 abs ratio 5 초과 ticker가 남지 않았다. split-adjusted 혼용으로 보이는 폭주는 제거됐다.

1W에는 `high_ratio_abs_max=1.5573`, `atr_ratio_max=1.0966`이 남아 있다. 주간 aggregation에서 저가/급등/고변동 구간이 섞인 real outlier로 추정한다. split bug와 달리 p99가 정상화되어 전체 분포를 망가뜨리지는 않는다.

1M에는 `high_ratio_abs_max=2.3712`, `atr_ratio_max=8.4681`이 남아 있다. 월간 데이터는 분모가 낮은 구간과 위기/급등 구간이 크게 반영될 수 있어, 모델 feature로 승격할 경우 winsorization 또는 log transform 검토가 필요하다.

`daily_range_ratio`와 `intraday_range_ratio` 컬럼은 현재 DB에 없다. 현재는 `high_ratio - low_ratio`가 range proxy지만 별도 저장 계약은 없다. 이 둘을 BM feature로 승격하려면 schema, feature contract, cache digest, feature version 변경이 먼저 필요하다.

## 9. Cron 정합성

Render cron은 `render.yaml:22-27`에서 `lens-daily-market-sync`가 `python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60`을 실행한다.

`daily_market_sync`는 `backend/collector/pipelines/daily_market_sync.py:17`에서 `compute_indicators.run`을 import하고, `backend/collector/pipelines/daily_market_sync.py:181` 경로에서 indicator recompute를 호출한다. `--indicator-lookback-days 60`은 일일 incremental 계산에 충분하다. `compute_indicators` 자체 source history는 1D 250일, 1W 550일, 1M 2100일이라 ATR 14와 1W/1M 재계산에 필요한 context를 확보한다.

이번에 추가한 prune은 `force_full_backfill=True`에서만 작동한다. 따라서 daily cron incremental path는 기존처럼 upsert 중심으로 유지되며, 다음 cron부터 `atr_ratio`와 adjusted OHLC 기준 가격 ratio가 유지될 수 있다.

삭제 후보로 기록할 별도 임시 백필 자동화는 이번 CP에서 제거하지 않았다. 불필요한 backfill helper가 확인되더라도 삭제는 별도 CP로 분리한다.

## 10. BM feature 승격 판단

`atr_ratio`는 차트 보조지표와 향후 BM feature 후보로 승격 준비가 되었다. 이유는 세 가지다.

- DB coverage가 1D/1W/1M 전체 row에서 100%다.
- AAPL 포함 최신 1D/1W/1M sample이 non-null이다.
- 계산 경로가 adjusted OHLC 공식 feature builder와 daily cron에 연결되어 있다.

다만 아직 모델 feature로 추가하면 안 된다. 현 계약은 `MODEL_N_FEATURES=36`이고, `atr_ratio`는 indicator-only다. 승격 시에는 `FEATURE_COLUMNS`, `MODEL_FEATURE_COLUMNS`, cache fingerprint, feature version, checkpoint 호환성, winsorization 정책을 한 CP에서 함께 바꿔야 한다.

`daily_range_ratio` 또는 `intraday_range_ratio`는 아직 컬럼이 없으므로 승격 준비 완료가 아니다. 먼저 이름, 계산식, timeframe 의미, adjusted 기준, clipping 정책을 정한 뒤 schema/feature contract를 추가해야 한다.

## 11. 검증

실행한 검증:

```powershell
.venv\Scripts\python.exe -m py_compile backend/app/services/feature_svc.py backend/collector/jobs/compute_indicators.py
$env:PYTHONPATH='C:\Users\user\lens;C:\Users\user\lens\backend'; .venv\Scripts\python.exe -m unittest discover backend/tests
.venv\Scripts\python.exe -m json.tool docs/cp64_indicator_full_backfill_metrics.json
```

결과:

- `py_compile` 통과
- backend unittest 53개 통과
- metrics JSON 파싱 통과
- 문서 UTF-8 확인 통과

참고로 PYTHONPATH 없이 `unittest discover backend/tests`를 실행하면 일부 테스트에서 `app` import가 실패했다. 올바른 backend test 명령은 repo root와 backend path를 함께 PYTHONPATH에 넣는 방식이다.

## 12. 최종 판단

CP64-D는 성공이다. `public.indicators`의 ATR coverage는 full coverage가 되었고, 과거 adjusted/raw 혼용으로 남아 있던 1D/1W 가격 ratio 폭주는 제거됐다. 모델 feature contract는 바뀌지 않았으며, `atr_ratio`는 아직 모델 입력이 아니라 indicator-only 후보로 유지한다.

다음 BM 실험에서는 `atr_ratio`를 바로 모델에 넣기보다, 별도 feature contract CP에서 clipping/winsorization과 feature_version bump를 정한 뒤 승격하는 것이 맞다.
