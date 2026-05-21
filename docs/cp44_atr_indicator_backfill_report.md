# CP44-D ATR ratio 지표 백필 및 운영 수집 경로 반영 보고서

## 1. 한 줄 결론

`atr_ratio`는 계산식은 있었지만 indicator upsert 출력 컬럼에 빠져 있어 DB에는 채워지지 않았다. 이번 CP에서 `atr_ratio`를 indicator-only 출력으로 추가하고 최근 1D/1W/1M 구간을 백필했다. 모델 feature contract는 변경하지 않았다.

## 2. 범위와 금지사항 준수

- `MODEL_N_FEATURES` 변경 없음.
- `ai/preprocessing.py` feature list 변경 없음.
- `backend.app.services.feature_svc.FEATURE_COLUMNS`에 `atr_ratio`를 추가하지 않음.
- 모델 학습 실행 없음.
- full 473 모델 run 없음.
- 프론트 UI 수정 없음.
- `predictions`, `model_runs` schema 변경 없음.

확인값:

| 항목 | 값 |
|---|---:|
| source feature count | 29 |
| `MODEL_N_FEATURES` | 36 |
| `atr_ratio in FEATURE_COLUMNS` | false |
| `atr_ratio in MODEL_FEATURE_COLUMNS` | false |

## 3. 원인

`backend/app/services/feature_svc.py`의 `_compute_features_for_single_ticker()`는 이미 `atr_ratio = ATR14 / close`를 계산하고 있었다. 하지만 `_OUTPUT_COLUMNS`가 `["ticker", "date", "timeframe", "regime_label", *FEATURE_COLUMNS]`로만 구성되어 있어 `build_features()` 반환값과 `compute_indicators.py` upsert record에 `atr_ratio`가 포함되지 않았다.

수정은 다음처럼 제한했다.

- `_INDICATOR_ONLY_COLUMNS = ["atr_ratio"]` 추가.
- `_OUTPUT_COLUMNS`에 indicator-only 컬럼을 포함.
- `FEATURE_COLUMNS`는 그대로 유지.
- 1M indicator 계산은 월봉 특성상 `high_ratio` 분포 max가 커질 수 있어, finite 검증은 유지하되 p99/max hard fail은 1D/1W 모델 피처 검증에 한정했다.

## 4. 백필 전 DB 상태

직접 DB 조회 기준이다.

| 항목 | 값 |
|---|---:|
| `public.indicators` 전체 row 수 | 1,580,066 |
| `atr_ratio` non-null row 수 | 0 |

timeframe별 상태:

| timeframe | rows | `atr_ratio` non-null | min date | max date |
|---|---:|---:|---|---|
| 1D | 1,297,172 | 0 | 2015-03-27 | 2026-04-28 |
| 1W | 248,026 | 0 | 2016-02-19 | 2026-05-01 |
| 1M | 34,868 | 0 | 2019-12-31 | 2026-04-30 |

최신 AAPL도 1D/1W/1M 모두 null이었다.

## 5. 백필 실행 범위와 명령

기존 indicator 계산 경로인 `backend.collector.jobs.compute_indicators.run()`만 사용했다. ad-hoc SQL 계산은 하지 않았다.

전체 known universe는 503개 ticker였다.

### 1D

추천 범위인 최근 500 거래일에 맞춰 2024-01-01부터 전체 universe를 재계산했다.

```powershell
$env:PYTHONPATH='C:\Users\user\lens'
@'
from backend.collector.jobs.compute_indicators import run
from backend.collector.universe import list_known_tickers

run(
    lookback_days=500,
    tickers=list_known_tickers(),
    force_full_backfill=True,
    full_start_date="2024-01-01",
    timeframes=["1D"],
)
'@ | python -
```

### 1W

처음에는 전체 universe를 한 번에 실행했지만 Supabase statement timeout이 발생했다. 같은 `compute_indicators.run()` 경로를 유지하고 ticker batch size 40으로 나눠 재실행했다.

```powershell
$env:PYTHONPATH='C:\Users\user\lens'
@'
from backend.collector.jobs.compute_indicators import run
from backend.collector.universe import list_known_tickers

tickers = list_known_tickers()
for start in range(0, len(tickers), 40):
    run(
        lookback_days=1092,
        tickers=tickers[start:start + 40],
        force_full_backfill=True,
        full_start_date="2021-01-01",
        timeframes=["1W"],
    )
'@ | python -
```

### 1M

60개월 UI 표시보다 안전하게 긴 히스토리를 주기 위해 2015-01-01부터 batch size 25로 재계산했다. 월봉 기간 라벨은 월말 기준이다.

```powershell
$env:PYTHONPATH='C:\Users\user\lens'
@'
from backend.collector.jobs.compute_indicators import run
from backend.collector.universe import list_known_tickers

tickers = list_known_tickers()
for start in range(0, len(tickers), 25):
    run(
        lookback_days=2100,
        tickers=tickers[start:start + 25],
        force_full_backfill=True,
        full_start_date="2015-01-01",
        timeframes=["1M"],
    )
'@ | python -
```

## 6. 백필 후 DB 상태

| 항목 | 백필 전 | 백필 후 |
|---|---:|---:|
| 전체 rows | 1,580,066 | 1,583,688 |
| `atr_ratio` non-null rows | 0 | 407,861 |
| `atr_ratio` null rows | 1,580,066 | 1,175,827 |

timeframe별 최종 상태:

| timeframe | rows | `atr_ratio` non-null | `atr_ratio` null | min date | max date |
|---|---:|---:|---:|---|---|
| 1D | 1,297,172 | 262,176 | 1,034,996 | 2015-03-27 | 2026-04-28 |
| 1W | 249,732 | 108,901 | 140,831 | 2016-02-19 | 2026-05-01 |
| 1M | 36,784 | 36,784 | 0 | 2019-12-31 | 2026-04-30 |

최신 timeframe coverage:

| timeframe | 최신 label date | rows | `atr_ratio` non-null |
|---|---|---:|---:|
| 1D | 2026-04-28 | 503 | 503 |
| 1W | 2026-05-01 | 502 | 502 |
| 1M | 2026-04-30 | 494 | 494 |

현재 날짜가 2026-04-29이므로 1W `2026-05-01`, 1M `2026-04-30`은 미래 raw 거래일이 아니라 `W-FRI`, 월말 리샘플의 기간 종료 라벨이다.

## 7. AAPL 및 주요 ticker 확인

AAPL 최신 `atr_ratio`:

| timeframe | date | `atr_ratio` |
|---|---|---:|
| 1D | 2026-04-28 | 0.0208684042490667 |
| 1W | 2026-05-01 | 0.0499770444912849 |
| 1M | 2026-04-30 | 0.100940911992957 |

MSFT/NVDA도 1D/1W/1M 최신 값이 non-null임을 확인했다.

## 8. 분포 sanity

| timeframe | n | min | p50 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 1D | 262,176 | 0.001612 | 0.023449 | 0.073267 | 0.240202 |
| 1W | 108,901 | 0.005900 | 0.056432 | 0.163629 | 1.096561 |
| 1M | 36,784 | 0.048077 | 0.119383 | 0.367066 | 8.468068 |

극단값 출처:

| timeframe | ticker | date | `atr_ratio` |
|---|---|---|---:|
| 1D | SMCI | 2024-11-14 | 0.240202 |
| 1W | CVNA | 2022-12-23 | 1.096561 |
| 1M | CVNA | 2022-12-31 | 8.468068 |

1D p99는 정상 범위다. 1W/1M max는 CVNA 2022년 급락 구간에서 발생한 월봉/주봉 비율 극단값이다. chart indicator 값으로는 finite이며 계산 계약을 깨지는 않지만, 향후 `atr_ratio`를 모델 피처로 올릴 경우 timeframe별 winsorization 또는 clipping 기준을 먼저 정해야 한다.

## 9. API 확인

read-only 서비스 레이어에서 다음을 확인했다.

```python
from app.services.api_service import get_indicator_response_data

get_indicator_response_data("AAPL", timeframe="1D", limit=1)
get_indicator_response_data("AAPL", timeframe="1W", limit=1)
get_indicator_response_data("AAPL", timeframe="1M", limit=1)
```

결과:

| timeframe | rows | date | `atr_ratio` key | `atr_ratio` |
|---|---:|---|---|---:|
| 1D | 1 | 2026-04-28 | true | 0.0208684042490667 |
| 1W | 1 | 2026-05-01 | true | 0.0499770444912849 |
| 1M | 1 | 2026-04-30 | true | 0.100940911992957 |

`backend/app/repositories/market_repo.py`의 indicator select list와 `backend/app/schemas/stocks.py`의 `IndicatorPoint`에는 이미 `atr_ratio`가 포함되어 있었다.

## 10. Render daily_market_sync 영향

`render.yaml`의 cron은 다음 경로다.

```yaml
name: lens-daily-market-sync
startCommand: python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60
```

확인 결과:

- `daily_market_sync.py`가 `backend.collector.jobs.compute_indicators.run`을 `run_indicators`로 호출한다.
- `compute_indicators.run()`은 기본 timeframes로 `("1D", "1W", "1M")`을 처리한다.
- `build_features()` 출력에 `atr_ratio`가 포함되므로 이후 cron upsert에서도 `atr_ratio`가 유지된다.
- `--indicator-lookback-days 60`은 upsert cutoff overlap에 쓰인다.
- source history는 timeframe별로 별도 확보된다: 1D 250일, 1W 550일, 1M 2100일.
- ATR14 기준 1D/1W는 충분하다. 1M도 2100일 source history가 있어 MA60/ATR warmup에 대체로 충분하다.

따라서 이번 CP에서는 `render.yaml`을 변경하지 않았다. 첫 운영 cron 이후 `atr_ratio` 최신 coverage만 다시 확인하면 된다.

## 11. 데이터 품질 감사 자동화 반영

Codex 자동화 `Lens 데이터 품질 감사` 프롬프트를 갱신했다. 새 점검 항목은 다음이다.

- `atr_ratio` 컬럼 존재 여부.
- 전체 및 timeframe별 non-null count.
- 최신 1D/1W/1M coverage.
- p99/max sanity.

백필 자동화는 새로 만들지 않았다.

## 12. 테스트

통과:

```powershell
python -m unittest backend.tests.test_feature_svc
python -m unittest backend.tests.test_feature_svc backend.tests.test_collector_jobs
python -m py_compile backend\app\services\feature_svc.py backend\collector\jobs\compute_indicators.py
```

추가 테스트:

- `atr_ratio`가 `build_features()` output에는 포함되지만 `FEATURE_COLUMNS`에는 포함되지 않는지 확인.
- 1M indicator 계산에서 큰 월봉 ratio가 있어도 finite이면 통과하는지 확인.
- `compute_indicators.run(..., timeframes=["1D"])`로 백필 timeframe 제한이 가능한지 확인.

## 13. 추후 모델 feature로 추가할 때 필요한 변경

이번 CP에서는 `atr_ratio`를 모델 feature로 추가하지 않았다. 나중에 학습 피처로 올리려면 별도 CP에서 최소 다음을 해야 한다.

- `FEATURE_COLUMNS`와 `ai/preprocessing.py` feature contract 갱신.
- `MODEL_N_FEATURES` 변경 확인.
- feature contract version bump.
- 기존 feature/precomputed cache 무효화.
- 1D/1W/1M별 `atr_ratio` winsorization 또는 clipping 정책 정의.
- finite contract와 p99/max threshold 테스트 추가.
- 50티커 이하 smoke 후 모델 비교로 진행.

## 14. 최종 판정

성공 기준은 충족했다.

- `atr_ratio` 컬럼 존재 확인.
- 최신 indicators에서 `atr_ratio` 값 확인.
- 1D는 AAPL 포함 주요 ticker에서 non-null 확인.
- 1W/1M도 AAPL 최신 값 non-null 확인.
- `daily_market_sync` 경로에서 앞으로 `atr_ratio`가 유지될 수 있음 확인.
- 모델 feature contract 변경 없음.
