# yfinance 50티커 장기 전환 리뷰 보고서

## 1. Executive Summary

판정: **100티커 장기 write는 조건부 가능, 전체 universe write와 live inference 연결은 아직 금지**가 맞다.

CP91에서 발생했던 source 혼합 P0는 현재 구조상 상당히 닫혔다. `price_data`는 `(ticker, date, source)`, `indicators`는 `(ticker, timeframe, date, source)` unique 계약을 갖고 있고, 가격 수집, indicator 재계산, AI feature cache 모두 provider/source를 분리한다. CP94의 50티커 장기 결과도 `2015-01-02~2026-05-01`, 142,448개 가격 row, 139,505개 1D indicator row, duplicate 0, source/provider 누락 0, 기본 split gate PASS, 1epoch smoke PASS까지 확인했다.

다만 이것은 "50티커 1D 장기 yfinance 경로가 돈다"는 증거이지, 전체 universe나 live inference에 바로 연결해도 된다는 증거는 아니다. 특히 1W/1M 미검증, 장기 dry-run 비교 gate 한계, full universe 실패 시 rollback/fallback 운영 절차, live read path의 source 선택 정책은 아직 필수 게이트로 남겨야 한다.

## 2. P0/P1/P2/P3 리스크 표

| 심각도 | 리스크 | 근거 | 판단 | 권장 조치 |
|---|---|---|---|---|
| P0 | 신규 P0는 확인되지 않음 | `backend/db/schema.sql:37`, `backend/db/schema.sql:187`, `backend/collector/jobs/sync_prices.py:361`, `backend/collector/jobs/compute_indicators.py:274`, `ai/preprocessing.py:535` | source-aware 병렬 저장과 feature cache 분리는 구조적으로 들어가 있다. | 100티커 확대 시 같은 gate를 유지한다. |
| P1 | 장기 dry-run 비교 gate를 아직 신뢰 가능한 확대 차단기로 보기 어렵다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:53`, `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json:4`, `backend/collector/pipelines/yfinance_price_sync.py:207` | CP94는 dry-run exit code 1을 "provider 정책 차이 + baseline row cap 한계"로 분류했다. 데이터 contract는 통과했지만, 비교 gate 자체는 아직 P1 개선 대상이다. | 100티커 write에서는 dry-run 실패 사유를 기록하고, 사후 contract/duplicate/source/hash/smoke gate로 통제한다. 전체 universe 전에는 baseline row cap 제거 또는 날짜 범위별 비교로 고쳐야 한다. |
| P1 | 1W/1M yfinance 장기 indicator가 아직 검증되지 않았다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:98`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:327`, `backend/collector/jobs/compute_indicators.py:150` | CP94는 1D만 재계산했다. 코드상 source-aware timeframe 처리는 있지만 장기 yfinance 1W/1M 결과는 없다. | 전체 universe 또는 1W/1M 모델/차트 연결 전 50티커와 100티커에서 1W/1M 재계산 sanity를 별도 통과시킨다. |
| P1 | live inference source 선택 정책이 아직 닫히지 않았다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:188`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:265`, `ai/preprocessing.py:1842` | 학습 feature plan은 provider/source/hash를 기록하지만, product run 선택, inference, API/차트 read path가 yfinance와 EODHD를 섞지 않는지는 이번 리뷰 범위에서 검증되지 않았다. | live 전 model_run/prediction/evaluation/API 응답에 provider/source/source_data_hash를 고정 표시하고, 기본 source 선택 정책을 테스트로 막는다. |
| P1 | full universe 실패 시 rollback/fallback 운영 절차가 부족하다 | `backend/collector/jobs/sync_prices.py:316`, `backend/collector/jobs/sync_prices.py:347`, `backend/collector/jobs/sync_prices.py:376`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:64` | 코드에는 fallback_used 기록이 있지만 CP94에서는 fallback 0건만 검증됐다. 대량 write 실패 후 source 단위 삭제/재시도/부분 실패 격리 절차는 아직 운영 gate로 부족하다. | 전체 universe 전 source/ticker/date 범위별 rollback runbook, fallback 허용 기준, 부분 실패 재시도 기준을 문서화한다. |
| P2 | CP94 feature row와 index row가 22건 차이난다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:144`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:146`, `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json:19` | smoke는 통과했지만, CP93에서는 feature/index row가 맞았던 흐름 대비 CP94의 `139,483` vs `139,505` 차이가 설명되지 않았다. | 100티커 전후로 이 차이가 어떤 join/drop에서 발생하는지 원인과 허용 기준을 기록한다. |
| P2 | AMD 2개 row 제외는 정상 gate로 보이지만 예외 관리가 필요하다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:66`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:73`, `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json:3395` | adjusted OHLC 위반은 0이고 invalid_volume/extreme_jump 차단이라 전환 실패는 아니다. 다만 universe 확대 시 예외가 누적될 수 있다. | ticker/date/reason 예외 목록과 허용률 threshold를 둔다. |
| P2 | CP89 5티커 EODHD artifact는 yfinance 확대에는 치명적이지 않지만 baseline 비교에는 계속 영향을 준다 | `docs/cp91_yfinance_p0_source_mixing_guard_report.md:7`, `docs/cp91_yfinance_p0_source_mixing_guard_report.md:147`, `docs/cp92_yfinance_5ticker_source_indicator_recalc_report.md:41` | 최근 500거래일 EODHD row가 yfinance로 대체된 known artifact가 남아 있다. | EODHD baseline 비교에서는 해당 구간을 제외하거나 EODHD 복구 여부를 별도 결정한다. |
| P2 | 50티커는 대형/유동성 높은 종목 중심이라 전체 universe의 delisted/illiquid/coverage 실패를 대표하지 못한다 | `docs/cp94_yfinance_50ticker_long_history_validation_report.md:261` | CP94 50티커 성공만으로 503개 전체의 Yahoo coverage, metadata FK, 긴 중단 구간을 보장할 수 없다. | 100티커에서 섹터/거래소/ETF/결측 위험을 더 섞고, 전체 전 coverage 실패율 기준을 만든다. |
| P3 | CP94 metrics/report 일부 한글 필드가 리뷰 환경에서 깨져 보인다 | `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json:3395`, `docs/cp94_yfinance_50ticker_long_history_validation_report.md:53` | 데이터 품질 리스크는 아니지만 자동 보고/발표 산출물 신뢰도를 낮출 수 있다. 리뷰 환경 기준 관찰이므로 인코딩 재현 여부 확인이 필요하다. | metrics JSON은 ASCII key/value 또는 UTF-8 검증을 추가한다. |

## 3. 100티커 write 가능 여부

**조건부 가능**이다. 단, "product run 교체"나 "live inference 연결"이 아니라 **장기 yfinance 데이터 경로 확대 검증**으로만 진행해야 한다.

진행 조건:

- `backend.collector.jobs.sync_prices.run()`의 official provider 경로를 사용한다. CP94도 이 경로에서 `source='yfinance'`, `provider='yfinance'`, `on_conflict="ticker,date,source"`를 사용했다.
- 100티커 write 후 가격 contract를 다시 확인한다. 최소 gate는 row 수, ticker 수, date range, duplicate `(ticker,date,source)=0`, raw OHLC violation 0, adjusted OHLC tolerance violation 0, source/provider/policy/updated_at 누락 0이다.
- 1D indicators는 반드시 source-aware full backfill로 재계산한다. CP94는 1D만 검증했으므로 100티커도 1D 기준으로 먼저 닫는다.
- feature cache는 provider/source/source_data_hash/manifest가 CP94와 새 100티커 데이터에 맞게 새로 분리되는지 확인한다.
- seq_len 60과 252 split gate를 기본 `min_fold_samples=50`으로 통과해야 한다.
- band/line smoke는 save-run 없이, product run 교체 없이, W&B/DB 쓰기 영향을 제한한 상태에서 다시 통과시킨다.
- 실패 ticker, fallback 사용 ticker, 제외 row는 ticker/date/reason으로 남긴다. fallback이 1건이라도 발생하면 source가 fallback provider로 저장되는지와 yfinance 집계에서 제외되는지 별도 확인한다.

현재 기준으로 100티커를 막는 P0는 없다. 다만 dry-run 비교 gate를 그대로 "PASS 아니면 무조건 중단"으로 쓰기에는 장기 비교 한계가 이미 드러났으므로, 100티커에서는 dry-run을 진단 자료로 남기고 사후 데이터 contract gate를 더 강하게 적용하는 방식이 안전하다.

## 4. 전체 universe write 가능 여부

**아직 금지**가 맞다.

금지 이유:

- CP94는 50티커만 검증했고, 전체 universe의 ticker mapping, delisted/illiquid ticker, Yahoo coverage 실패, 긴 결측 구간은 검증하지 않았다.
- 1W/1M source-aware long-history indicator가 아직 실제 데이터로 통과하지 않았다.
- dry-run 비교 gate가 장기 구간에서 신뢰 가능한 차단기로 정리되지 않았다.
- 전체 write는 실패 시 row 수와 indicator upsert 비용이 커서 migration artifact가 크게 남을 수 있다.
- rollback/fallback runbook이 아직 "실행 가능한 운영 절차" 수준으로 닫히지 않았다.

전체 universe로 가기 전 최소 순서:

1. 100티커 장기 write와 1D indicator/feature/split/smoke 통과.
2. 50티커 또는 100티커 local daily sync rehearsal 통과.
3. 1W/1M yfinance source-aware indicator 재계산 sanity 통과.
4. dry-run 비교 gate 개선 또는 대체 검증 도구 확정.
5. source 단위 rollback/fallback runbook 확정.
6. 전체 universe 대상 coverage 실패율, 제외 row 허용률, metadata 누락 허용률 threshold 확정.

## 5. live inference 전 필수 조건

live inference는 아직 연결하면 안 된다.

필수 조건:

- 학습 run, checkpoint, prediction, evaluation, backtest, API 응답에 `provider`, `source`, `source_data_hash`, `feature_contract_version`이 고정되어야 한다.
- inference가 현재 DB 상태의 default source를 암묵적으로 읽지 않고, 학습 시점 source/hash와 같은 데이터 계약을 사용해야 한다.
- product chart/API read path가 `source='yfinance'`와 `source='eodhd'`를 섞지 않는지 테스트가 필요하다.
- yfinance product run은 EODHD product run과 별도 slot 또는 명확한 provider badge로 노출해야 한다.
- fallback 사용 row가 product 모델에 섞일 경우, source provenance를 예측/평가/백테스트 단위에서 볼 수 있어야 한다.
- 100티커 장기 smoke는 "학습 가능성"만 본 것이므로, live 전에는 저장 run 없이 끝내지 말고 별도 staged product run으로 evaluation/backtest를 검증해야 한다.

## 6. 좋았던 구조

- DB 계약이 source-aware로 잘 바뀌었다. `price_data`와 `indicators` 모두 source 포함 unique를 쓰고 old unique를 제거한다.
- runtime schema도 같은 방향이다. `ensure_runtime_schema.py`가 source/provider 컬럼과 source-aware unique/index를 맞춘다.
- 가격 수집 경로가 provider abstraction을 통한다. yfinance는 `auto_adjust=False`, `Adj Close` 기반 adjusted OHLC 정책을 명시한다.
- `sync_prices`가 source/provider/policy/updated_at을 저장하고 `on_conflict="ticker,date,source"`로 upsert한다.
- `compute_indicators`가 provider/source별 가격만 읽고 indicator도 source별로 저장한다.
- AI preprocessing cache가 provider/source/source_data_hash/manifest를 반영한다. CP93 yfinance hash `2c075526`, CP94 yfinance hash `bef59538`, EODHD hash `3e90764c`가 분리된 점은 좋은 증거다.
- 단위 테스트가 핵심 혼합 방지 계약을 잘 덮고 있다. source provenance, source-aware upsert, schema unique, yfinance indicator source filtering, cache manifest mismatch, feature index provider date 사용을 확인한다.

## 7. 부족한 테스트

- 100티커급 source-aware write를 mock DB 또는 staging DB에서 end-to-end로 검증하는 테스트가 부족하다.
- 1W/1M yfinance long-history indicator full backfill 테스트가 부족하다.
- fallback 사용 시 `requested_provider=yfinance`, `provider=eodhd`, `fallback_used=true` row가 어떤 source로 저장되고 feature cache에서 어떻게 분리되는지 테스트가 필요하다.
- live inference/API/chart read path가 provider/source를 섞지 않는지 테스트가 없다.
- CP94의 `feature_df_rows=139,483`, `index_rows=139,505` 차이를 재현하고 허용할지 실패시킬지 정하는 테스트가 없다.
- AMD처럼 price quality gate에서 row가 제외될 때 exception registry와 허용률 threshold를 검증하는 테스트가 없다.
- source 단위 rollback dry-run 테스트가 없다.

## 8. 코드 수정/DB 쓰기/학습 미실행 확인

- 코드 파일은 수정하지 않았다.
- DB write, delete, migration 실행, provider 설정 변경은 하지 않았다.
- 모델 학습, smoke 재실행, W&B 실행, inference 실행은 하지 않았다.
- 이번 작업에서 생성한 산출물은 이 보고서 `docs/yfinance_50ticker_long_history_review_report.md`뿐이다.

## 9. 읽기 전용 확인 파일 목록

확인한 요청 대상 파일:

- `docs/cp91_yfinance_p0_source_mixing_guard_report.md`
- `docs/cp92_yfinance_5ticker_source_indicator_recalc_report.md`
- `docs/cp93_yfinance_50ticker_limited_write_validation_report.md`
- `docs/cp94_yfinance_50ticker_long_history_validation_report.md`
- `docs/cp94_yfinance_50ticker_long_history_validation_metrics.json`
- `backend/db/schema.sql`
- `backend/db/scripts/ensure_runtime_schema.py`
- `backend/collector/jobs/sync_prices.py`
- `backend/collector/jobs/compute_indicators.py`
- `ai/preprocessing.py`
- `backend/tests/test_market_data_providers.py`
- `backend/tests/test_collector_jobs.py`
- `ai/tests/test_preprocessing_cache_isolation.py`

추가로 확인한 관련 파일:

- `backend/collector/pipelines/yfinance_price_sync.py`
- `backend/collector/sources/market_data_providers.py`
- `backend/collector/sources/price_contract.py`

읽기 전용으로 실행한 명령 유형:

- `git status --short`
- `Get-Content ... | Select-Object ...`
- `Select-String -Path ... -Pattern ...`
- `ConvertFrom-Json` 기반 metrics 구조 확인
