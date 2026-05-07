# CP86-D yfinance 로컬 primary 데이터 전환 보고서

작성일: 2026-05-03

## 1. 요약

CP86-D에서는 개인 로컬 운영 비용을 줄이기 위해 가격 수집 provider 계층을 추가하고, yfinance를 로컬 primary provider로 선택할 수 있게 만들었다. EODHD 코드는 삭제하지 않았고, `MARKET_DATA_PROVIDER=yfinance`, `MARKET_DATA_FALLBACK_PROVIDER=eodhd` 구조로 fallback/검증용 provider로 남겼다.

핵심 결과는 다음과 같다.

- yfinance fetch는 `auto_adjust=False`로 고정해 raw OHLC와 `Adj Close`를 함께 보존한다.
- 저장 전 adjusted OHLC sanity gate를 추가했다.
- 12티커 dry-run 비교는 `overall_pass=true`다.
- adjusted OHLC sanity violation은 0건이다.
- fallback 사용은 0건이다.
- DB write는 수행하지 않았다. write mode는 구현했지만, 실제 저장은 사용자가 `--write`를 명시할 때만 실행된다.
- yfinance write 후 indicators 재계산이 필요하므로 local daily script와 compute indicators CLI를 추가했다.

이번 CP의 판단은 "상용 SLA가 있는 운영 provider 교체"가 아니라 "개인 로컬 운영에서 비용을 낮추는 primary provider 전환"이다. yfinance는 비공식 Yahoo Finance 공개 데이터 기반이므로 장애 가능성을 인정하고, EODHD를 즉시 삭제하지 않는다.

## 2. 구현 변경

### provider abstraction

추가:

- `backend/collector/sources/market_data_providers.py`
- `backend/collector/sources/price_contract.py`

provider는 최소 두 개를 지원한다.

| provider | 역할 |
|---|---|
| `yfinance` | 로컬 primary 후보. `auto_adjust=False`로 raw OHLC와 adjusted_close를 함께 받는다. |
| `eodhd` | 기존 provider. fallback/검증용으로 유지한다. |

설정:

- `MARKET_DATA_PROVIDER=yfinance`
- `MARKET_DATA_FALLBACK_PROVIDER=eodhd`

`backend/collector/config.py`에 `market_data_provider`, `market_data_fallback_provider`를 추가했다. primary가 `yfinance`인데 fallback 설정이 없으면 기본 fallback은 `eodhd`다.

### EODHD 코드 유지

`backend/collector/sources/eodhd_prices.py`의 EODHD fetch와 valuation attach 경로는 유지했다. 기존 파일 안에 직접 있던 yfinance fallback 호출은 provider 계층으로 이동했다. 이로써 EODHD 전용 코드와 yfinance provider 책임이 분리됐다.

### sync_prices 연결

`backend/collector/jobs/sync_prices.py`는 이제 provider/fallback provider를 받아 `fetch_market_data()`를 호출한다. 저장 전 다음 gate를 통과해야 한다.

- 날짜 null 없음
- 날짜 중복 없음
- `Open/High/Low/Close/Adj Close` null/Inf 없음
- volume 음수 없음
- `adjusted_factor = Adj Close / Close` finite, 양수
- adjusted high/low 정합성
- adjusted OHLC 기준 `open_ratio/high_ratio/low_ratio` p99/max sanity

기존 `_validate_price_frame()`도 유지되어 raw OHLC 범위, volume, extreme jump를 추가로 검사한다.

### local CLI와 스크립트

추가:

- `backend/collector/pipelines/yfinance_price_sync.py`
- `backend/collector/pipelines/compute_indicators_cli.py`
- `scripts/sync_yfinance_prices.ps1`
- `scripts/run_daily_local_market_sync.ps1`

dry-run 예:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --start-date 2024-05-03 --end-date 2026-05-03 --metrics-path docs/cp86_yfinance_local_primary_migration_metrics.json
```

write 예:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --write --tickers AAPL MSFT NVDA --start-date 2026-04-01
```

로컬 일일 실행 예:

```powershell
.\scripts\run_daily_local_market_sync.ps1 -Universe backend/data/universe/sp500.csv -StartDate 2026-04-24 -PriceBatchLimit 80
```

## 3. yfinance primary가 가능한 이유

가능하다고 판단한 이유는 세 가지다.

1. yfinance가 raw OHLC와 `Adj Close`를 동시에 제공한다.
2. Lens feature contract는 DB raw 값을 직접 모델에 먹이는 구조가 아니라, `adjusted_close / close`로 adjusted OHLC를 재구성한 뒤 feature를 계산한다.
3. dry-run에서 yfinance provider의 adjusted OHLC sanity violation이 0건이었다.

즉 provider의 raw close가 기존 EODHD DB와 다르더라도, adjusted_close와 adjusted OHLC 재구성 결과가 sane이면 모델 feature contract는 유지될 수 있다. NFLX에서 이 점이 확인됐다. 기존 DB는 split-adjusted close처럼 보이고 yfinance는 raw close와 adjusted close를 분리해 주지만, adjusted_close 비교는 사실상 일치했다.

## 4. EODHD를 즉시 삭제하지 않는 이유

EODHD를 삭제하지 않는 이유는 다음과 같다.

- yfinance는 비공식 공개 데이터 wrapper라 장애, rate limit, HTML/API 변경에 취약하다.
- 기존 DB와 yfinance 간 adjusted 정책 차이가 실제로 관찰됐다.
- Render cron은 아직 EODHD 운영 경로를 유지하는 편이 안전하다.
- provider 장애 시 fallback/비교 기준값이 필요하다.
- 상용/공개 서비스 SLA는 yfinance에 기대하면 안 된다.

따라서 CP86의 전환 범위는 개인 로컬 primary이며, EODHD는 fallback/검증용으로 남긴다.

## 5. dry-run 비교 결과

실행 명령:

```powershell
python -m backend.collector.pipelines.yfinance_price_sync --provider yfinance --fallback-provider eodhd --start-date 2024-05-03 --end-date 2026-05-03 --metrics-path docs/cp86_yfinance_local_primary_migration_metrics.json --allow-fail
```

결과 파일:

- `docs/cp86_yfinance_local_primary_migration_metrics.json`

요약:

| 항목 | 결과 |
|---|---|
| overall_pass | true |
| provider | yfinance |
| fallback_provider | eodhd |
| fallback_used | 0 |
| adjusted OHLC sanity violations | 0 |
| baseline rows | 1000 |
| 비교 ticker | 12 |
| DB write | 수행하지 않음 |

티커별 상태:

| 상태 | 티커 | 의미 |
|---|---|---|
| pass | AMD, AMZN, AVGO, GOOGL, META, NVDA, TSLA | raw close와 adjusted_close 모두 기준 내 |
| dividend_adjustment_policy_diff | AAPL, MSFT | raw close는 일치하지만 yfinance adjusted_close가 기존 DB보다 배당 조정 factor를 반영 |
| split_adjustment_policy_diff | NFLX | adjusted_close는 일치하지만 raw close/factor 정책이 split 때문에 크게 다름 |
| baseline_missing_contract_only | SPY, QQQ | yfinance contract는 PASS지만 기존 DB baseline이 없어 EODHD 대비 비교 불가 |

주요 수치:

| ticker | status | date coverage | adjusted_close median diff | close median diff |
|---|---|---:|---:|---:|
| AAPL | dividend_adjustment_policy_diff | 1.0 | 0.0076746 | 0.000000024 |
| MSFT | dividend_adjustment_policy_diff | 1.0 | 0.0133402 | 0.000000019 |
| NFLX | split_adjustment_policy_diff | 1.0 | 0.000000023 | 0.900000001 |
| NVDA | pass | 1.0 | 0.00048197 | 0.000000019 |

해석:

- AAPL/MSFT는 기존 DB의 `adjusted_close`가 최근 구간에서 `close`와 같았다. yfinance는 배당 조정 factor를 반영해 adjusted_close가 낮다. raw close는 거의 완전 일치하므로 가격 오류라기보다 adjustment policy 차이다.
- NFLX는 split 이후 raw/adjusted 정책 차이가 크다. yfinance는 raw close와 adjusted_close를 분리하고, 기존 DB는 split-adjusted close처럼 보인다. adjusted_close는 거의 일치하므로 feature contract 관점에서는 yfinance가 더 명시적인 구조다.
- SPY/QQQ는 기존 `price_data` baseline이 없어 EODHD 대비 비교는 불가능했다. yfinance provider contract만 확인했다.

## 6. adjusted OHLC sanity 결과

12개 ticker 모두 다음 항목을 통과했다.

- `adjusted_factor` invalid count 0
- adjusted high/low violation 0
- duplicate date 0
- required price null/non-finite 0
- ratio p99/max sanity PASS

이 결과 때문에 write gate는 PASS로 판단했다. 다만 정책 차이 ticker는 metrics에 별도 status로 남겼다.

## 7. DB write 여부

DB write는 수행하지 않았다.

이유:

- CP86에서는 write mode를 구현하고 dry-run PASS를 확인하는 것이 우선이다.
- 기존 DB의 AAPL/MSFT/NFLX에서 provider adjustment policy 차이가 관찰되어, 실제 write는 사용자가 의도적으로 `--write`를 붙여 실행하는 것이 안전하다.
- `price_data`에는 provider/source 컬럼이 없다. 따라서 yfinance row와 EODHD row를 같은 `ticker,date` key로 구분 없이 upsert하게 된다.

schema 한계:

- `price_data`에는 `source`, `provider`, `source_symbol`, `source_data_hash`가 없다.
- 대신 `sync_state` meta에는 provider/fallback_used/adjusted gate metrics를 남기도록 했다.
- provider provenance를 row 단위로 보존하려면 별도 CP에서 schema 확장이 필요하다.

## 8. indicators 재계산 여부

indicators 재계산은 실행하지 않았다. DB write를 하지 않았기 때문이다.

yfinance price_data write 후에는 반드시 indicators를 다시 계산해야 한다.

권장 명령:

```powershell
python -m backend.collector.pipelines.compute_indicators_cli --lookback-days 60 --timeframes 1D 1W 1M
```

확인해야 할 항목:

- `open_ratio/high_ratio/low_ratio` p99/max sanity PASS
- `atr_ratio` coverage 유지
- feature contract `v3_adjusted_ohlc` 유지
- 모델 입력 feature 수 36 유지
- `atr_ratio`는 여전히 indicator-only이며 모델 feature로 승격하지 않음

## 9. 의존성 판단

기존 `yfinance==0.2.40`은 2026-05-03 dry-run에서 Yahoo 응답을 처리하지 못해 전 ticker가 `YFTzMissingError`로 실패했다.

검토 중 `yfinance==1.2.2`도 설치해 보았지만 `websockets>=13`을 요구해 Supabase `realtime` 패키지의 `websockets<13` 요구조건과 충돌했다. 따라서 최종 pin은 `yfinance==0.2.58`로 정했다. 이 버전은 0.2.40보다 최신이고, `pip check` 기준 broken requirement가 없다.

수정:

- `backend/collector/requirements.txt`: `yfinance==0.2.58`
- `backend/db/requirements-crawler.txt`: `yfinance==0.2.58`

## 10. 남은 리스크

| 리스크 | 등급 | 설명 |
|---|---|---|
| provider provenance 부재 | P1 | `price_data` row별 source가 없어 yfinance/EODHD 혼합 이력을 DB에서 직접 구분할 수 없다. |
| yfinance 비공식 API | P1 | Yahoo 응답 변경, rate limit, cookie 문제로 장애가 날 수 있다. |
| adjusted policy drift | P1 | AAPL/MSFT/NFLX에서 기존 DB와 정책 차이가 관찰됐다. feature는 sane하지만 과거 지표 재계산 시 분포가 달라질 수 있다. |
| SPY/QQQ baseline 부재 | P2 | 샘플 ETF는 yfinance contract만 확인했고 EODHD DB 대비 비교는 못 했다. |
| Render cron 미전환 | P2 | 이번 CP는 로컬 primary 준비이며 Render cron은 유지한다. |
| indicators 미재계산 | P2 | write를 하지 않았으므로 재계산도 하지 않았다. write 후 별도 실행 필요. |

## 11. 다음 단계

1. 제한된 ticker로 `--write` 실행을 명시적으로 수행한다.
2. write 후 `compute_indicators_cli`로 1D/1W/1M indicators를 재계산한다.
3. indicators ratio/ATR coverage를 재검증한다.
4. 1D live inference/cron 연결은 다음 CP에서 붙인다.
5. provider provenance를 보존할지 별도 schema CP로 결정한다.

## 12. 검증

실행한 검증:

```powershell
python -m py_compile backend/collector/sources/price_contract.py backend/collector/sources/market_data_providers.py backend/collector/jobs/sync_prices.py backend/collector/pipelines/yfinance_price_sync.py backend/collector/pipelines/compute_indicators_cli.py
python -m unittest backend.tests.test_market_data_providers
python -m unittest backend.tests.test_collector_jobs backend.tests.test_feature_svc
python -m pip check
$env:PYTHONPATH='backend'; python -m unittest backend.tests.test_api
```

참고:

- `python -m unittest backend.tests.test_api ...`를 `PYTHONPATH=backend` 없이 한 번 실행했을 때 `ModuleNotFoundError: No module named 'app'`가 발생했다. 같은 테스트를 `PYTHONPATH=backend`로 재실행해 PASS했다.
- 모델 학습은 실행하지 않았다.
- live inference 연결은 하지 않았다.
- 1W/1M 모델 실험은 실행하지 않았다.
- fake data는 만들지 않았다.

