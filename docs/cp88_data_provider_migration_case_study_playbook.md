# CP88-G 데이터 provider 전환 사례 리서치 및 Lens 전환 플레이북

작성일: 2026-05-03

범위: EODHD에서 yfinance 로컬 primary로 전환하는 과정을 데이터 migration 경험으로 기록하기 위한 외부 사례 리서치 및 Lens 적용 플레이북 작성.

제한 확인: 이번 CP에서는 코드 수정, DB 쓰기, 모델 학습, provider 설정 변경을 수행하지 않았다. 이 문서는 리서치와 전환 기록 방식만 정리한다.

## 1. Executive Summary

EODHD에서 yfinance로 바꾸는 일은 단순한 API 교체가 아니다. Lens 기준으로는 가격 데이터의 원천, 조정 정책, corporate action 처리, feature cache, 모델 smoke 결과까지 이어지는 데이터 migration이다.

외부 사례에서 반복되는 핵심 패턴은 명확하다.

- 기존 source를 baseline으로 고정한다.
- 새 source를 parallel 또는 shadow로 실행한다.
- row count와 coverage만 보지 않고, primary key 기준 field-level diff를 남긴다.
- hash/checksum은 빠른 gate로 쓰되, 차이 원인 분석을 대체하지 않는다.
- exception list를 만들고 원인을 분류한다.
- cutover는 source-target이 검증 기준을 통과하고 fallback/rollback이 살아 있을 때만 수행한다.
- 금융 데이터에서는 corporate action, adjusted price, lineage, audit trail이 전환 판단의 핵심이다.

Lens의 현재 위치는 **CP87 기준 WARN**이다. yfinance 가격 품질, feature sanity, line/band smoke는 통과했지만, WARN 원인은 `price_data` provider/source provenance와 `source_data_hash`/cache isolation 부족이었다. CP88-D에서 코드 계약은 보강되었으나, 운영 DB schema 적용, 제한 write, indicators 재계산, yfinance source 기준 cache 검증은 아직 cutover gate로 남아 있다.

따라서 다음 판단은 "yfinance가 무료라서 바꾼다"가 아니라 "EODHD baseline과 yfinance shadow를 reconcile하고, lineage와 rollback이 보이는 상태에서 제한 전환한다"가 되어야 한다.

## 2. 외부 사례 요약

| 사례 | 문서 기준 핵심 내용 | Lens에 가져올 점 | 주의 |
|---|---|---|---|
| AWS DMS data validation | full load 이후 source-target row를 비교하고, CDC 변경분도 검증한다. validation state, pending/failed/suspended record count, validation failure table을 제공한다. | `ticker,date`를 key로 row 단위 비교, 실패 row와 실패 유형을 별도 exception list로 남긴다. | validation은 source/target에 추가 부하를 만든다. Lens에서도 473 전체 전에 제한 universe로 나눠야 한다. |
| Google Database Migration Service | initial snapshot 이후 continuous replication을 하고, source와 destination이 sync된 상태에서 primary 전환한다. | EODHD baseline을 유지한 상태에서 yfinance shadow를 돌리고, sync/coverage 기준 통과 후 cutover한다. | Google 문서는 DB migration 사례다. Lens는 provider migration이라 dual-write보다 shadow 비교와 제한 write가 더 적합하다. |
| dbt audit-helper | row count, relation equality, column diff, all-column value comparison을 제공한다. primary key는 unique/not-null이어야 한다. | `ticker,date` unique를 전제로 row count, missing from source/target, column별 conflict count를 같은 형식으로 만든다. | Lens의 price_data는 provider별 key까지 들어가면 `ticker,date,source` 계약을 명확히 해야 한다. |
| Datafold data-diff | row count는 빠르지만 record 내부 stale 값을 못 잡고, table hash는 차이 위치를 설명하지 못한다. chunk hash와 value-level diff가 보완책이다. | `source_data_hash`는 cache gate로 쓰고, 실제 provider 전환 판단은 field-level diff와 exception 원인 분류로 한다. | vendor 문서 기준이다. 홍보 수치를 그대로 Lens 성능 추정에 쓰면 안 된다. |
| BIS BCBS 239 | 금융기관 risk data는 정확성, 무결성, 적시성, governance가 중요하며 위기 때 의사결정 품질에 직접 영향을 준다. | 모델 feature용 가격도 risk data처럼 lineage와 audit trail을 남긴다. "어느 provider의 어떤 조정 정책으로 만든 feature인가"가 기록되어야 한다. | Lens는 규제 대상 은행은 아니지만, 발표/운영 신뢰성 기준으로 원칙을 차용할 수 있다. |
| GoldenSource reference/corporate actions 문서 | 금융 reference/market data는 fragmented source를 통합해 golden source, lineage, auditability, corporate action validation을 제공한다고 설명한다. | yfinance/EODHD 차이를 오류로 단정하지 않고, golden record 후보와 provider policy diff를 분리한다. | vendor 제품 설명 기준이다. 독립 검증된 성능 수치처럼 쓰지 않는다. |
| Broadridge corporate actions case 문서 | 대형 은행의 corporate actions 처리에서 여러 legacy system을 단일 플랫폼으로 통합하고 single source of truth를 만든 사례를 제시한다. | split/dividend/corporate action은 price migration의 별도 검증 축으로 둔다. | vendor case study 기준이다. Lens에는 "단일 정답 강제"보다 provider별 provenance 보존이 더 안전하다. |
| Yahoo/yfinance 문서 | yfinance `download()`는 기본 `auto_adjust=True`이며 OHLC 자동 조정 옵션과 dividend/split actions 옵션이 있다. Yahoo adjusted close는 split과 dividend multiplier를 반영한다고 설명한다. | Lens는 CP29 사고 재발 방지를 위해 `auto_adjust=False`로 raw OHLC와 `Adj Close`를 함께 받고, `adjusted_close / close`로 adjusted OHLC를 재구성한다. | yfinance는 비공식 Yahoo 공개 데이터 wrapper이며 개인/연구 용도에 적합하다. SLA가 있는 상용 feed처럼 취급하면 안 된다. |

## 3. 사례별 핵심 검증 단계

외부 사례를 Lens 방식으로 번역하면 다음 순서가 된다.

| 공통 패턴 | 외부 사례에서의 의미 | Lens 적용 방식 |
|---|---|---|
| parallel run | 기존 source를 운영하면서 새 target/source를 나란히 만든다. | EODHD baseline 유지, yfinance dry-run/shadow feature 생성. |
| row count / coverage 비교 | source와 target의 row 수, pending/failed row 수, migration completeness를 본다. | ticker별 거래일 coverage, missing date count, duplicate count, latest date 비교. |
| field-level diff | row가 있어도 column 값이 다를 수 있으므로 column별 차이를 본다. | raw open/high/low/close, adjusted_close, adjusted_factor, volume, ratio p99/max diff. |
| aggregate diff | 개별 row diff가 작아도 분포가 바뀌면 downstream 모델이 흔들릴 수 있다. | feature mean/std/p01/p50/p99, target distribution, absolute return tail 비율 비교. |
| hash/checksum | 전체 또는 chunk 단위 동일성을 빠르게 판단한다. | `source_data_hash`, feature hash, provider/policy/date range/ticker universe fingerprint. |
| exception list | mismatch를 실패 row table이나 audit table로 남긴다. | ticker/date/field/status/reason/severity/action이 들어간 migration exception list. |
| 원인 분류 | 불일치를 오류, 정책 차이, coverage 차이, corporate action 차이로 나눈다. | `pass`, `dividend_adjustment_policy_diff`, `split_adjustment_policy_diff`, `baseline_missing`, `provider_error` 등으로 분류. |
| audit report | 검증 결과와 실패 row를 사람이 검토할 수 있게 남긴다. | CP별 metrics JSON과 markdown 보고서, 실행 명령, provider/version/policy 기록. |
| cutover 기준 | source-target sync와 validation 통과 뒤 primary 전환한다. | yfinance 제한 write, indicators 재계산, cache manifest 검증, smoke PASS 후 local primary write 범위 확대. |
| rollback/fallback | 전환 후 문제가 생기면 이전 source로 돌아갈 수 있어야 한다. | EODHD fallback 유지, provider별 cache 분리, 기존 EODHD row 삭제 금지, product run 교체 지연. |

## 4. Lens에 적용할 원칙

### 4.1 Provider 전환 원칙

1. EODHD baseline은 cutover 판단 전까지 삭제하지 않는다.
2. yfinance 데이터는 먼저 dry-run/shadow로 검증하고, 운영 write는 제한 universe부터 시작한다.
3. `ticker,date`만으로 provider를 구분하지 않는다. 운영 DB에서는 `source` 또는 `provider`가 provenance 계약에 포함되어야 한다.
4. raw OHLC와 adjusted OHLC를 혼용하지 않는다. Lens feature 계약은 adjusted OHLC 기준이다.
5. provider 차이는 곧 오류가 아니다. split/dividend 정책 차이, ETF baseline 부재, raw close 정책 차이를 원인 분류한다.
6. `source_data_hash`는 provider, adjustment policy, feature contract, timeframe, universe fingerprint, date range, checksum 또는 updated_at을 포함해야 한다.
7. cache manifest가 provider/hash mismatch를 잡지 못하면 cutover를 허용하지 않는다.
8. 모델 smoke는 데이터 migration gate의 마지막 확인이지, 데이터 검증을 대체하지 않는다.
9. EODHD fallback은 yfinance 장애와 Yahoo policy drift를 흡수하기 위한 rollback 장치로 남긴다.
10. Render 운영 cron 전환은 local 제한 write와 live inference rehearsal 이후 별도 판단한다.

### 4.2 전환 성공 기준

| 기준 | PASS 조건 |
|---|---|
| adjusted OHLC sanity | violation 0 |
| duplicate key | duplicate ticker/date 0, provider 컬럼 도입 후에는 provider별 unique 계약 명확화 |
| date coverage | 기준 universe에서 99% 이상 또는 결측 원인 전부 분류 |
| split/dividend | 차이 ticker의 corporate action 원인 분류 완료 |
| feature finite | feature/target NaN/Inf 0 |
| ratio sanity | open_ratio/high_ratio/low_ratio p99 sanity PASS |
| model smoke | line smoke와 band smoke exit code 0, NaN 없음 |
| hash isolation | provider가 다르면 `source_data_hash`와 cache path가 달라짐 |
| cache manifest | provider/hash mismatch 시 cache load 거부 |
| fallback | EODHD fallback 경로 유지, fallback count 기록 |

### 4.3 전환 기록 양식

다음 양식을 CP 보고서와 metrics JSON에 함께 남긴다.

| 필드 | 기록 내용 |
|---|---|
| provider | `eodhd`, `yfinance`, fallback provider |
| period | start/end date, timezone/date 정책 |
| ticker universe | ticker 수, universe hash, split/dividend 샘플 포함 여부 |
| adjusted policy | raw OHLC 보존 여부, adjusted_close source, adjusted factor 계산식 |
| data hash | provider/policy/date/universe/checksum 기반 hash |
| feature hash | feature contract version, columns, timeframe, cache path |
| 검증 결과 | data sanity, feature sanity, smoke 결과 |
| 발견된 차이 | ticker/date/field별 diff summary와 exception list |
| 원인 분류 | provider policy diff, corporate action diff, baseline missing, provider error 등 |
| 최종 판정 | PASS, WARN, FAIL |
| rollback 가능 여부 | EODHD fallback, 기존 cache 보존, product run 미교체 여부 |

## 5. Lens yfinance 전환 현재 위치

현재 상태는 다음과 같이 정리한다.

- EODHD는 여전히 primary baseline이다.
- yfinance provider 연결은 CP86-D에서 완료됐다.
- CP86-D의 12티커 dry-run은 overall PASS였고 adjusted OHLC sanity violation은 0건이었다.
- CP87-D의 50티커 3단계 검증은 data/feature/model smoke를 통과했지만 최종 판정은 WARN이었다.
- WARN 원인은 provider/source provenance와 source_data_hash/cache isolation 부족이었다.
- CP88-D에서는 source provenance와 cache isolation 코드 계약을 보강했지만, 운영 DB schema migration과 제한 write 검증은 아직 cutover gate로 남아 있다.

| Stage | 내용 | 현재 판정 | 근거/남은 일 |
|---|---|---|---|
| Stage 0 | 기존 EODHD baseline 고정 | 부분 완료 | 기존 DB/보고서는 있으나 baseline manifest와 hash snapshot을 cutover artifact로 고정해야 한다. |
| Stage 1 | yfinance dry-run 비교 | 완료 | CP86/CP87에서 dry-run 비교와 50티커 data sanity를 수행했다. |
| Stage 2 | shadow feature 생성 | 완료 | 운영 DB overwrite 없이 yfinance 기반 shadow feature를 검증했다. |
| Stage 3 | model smoke | 완료 | 1D line/band smoke가 exit code 0으로 통과했다. |
| Stage 4 | provider/source provenance와 cache isolation | 코드 보강 완료, 운영 검증 전 | CP88-D에서 계약은 보강됐지만 DB migration, 제한 write, 실제 cache manifest 검증이 남았다. |
| Stage 5 | 제한 write | 대기 | 5~20티커 yfinance write rehearsal 필요. 기존 EODHD row 삭제 금지. |
| Stage 6 | indicators 재계산 | 대기 | 제한 write 직후 compute_indicators와 open/high/low ratio, atr_ratio coverage 재검증 필요. |
| Stage 7 | live inference 연결 | 대기 | source=`yfinance` feature cache로 smoke와 API 조회가 안정화된 뒤 연결. |
| Stage 8 | EODHD fallback 축소 또는 제거 판단 | 대기 | 최소 며칠 이상의 local daily sync parallel run 이후 판단. 즉시 제거 금지. |

현재 최종 문구는 **WARN: 개인 로컬 dry-run과 shadow 검증은 가능하지만, 운영 write와 product 연결 전 provenance/cache cutover gate가 필요하다**가 맞다.

## 6. 다음 CP에서 해야 할 기술 작업

다음 CP는 제한 write와 audit artifact 생성에 집중해야 한다.

1. 운영 DB에 `price_data.source`, `price_data.provider`, `price_data.provider_adjustment_policy`, `price_data.updated_at` schema migration을 적용한다.
2. 기존 EODHD row의 대량 backfill은 바로 하지 말고, 최소한 legacy null source를 EODHD로 해석하는 정책을 보고서에 명시한다.
3. AAPL, MSFT, NVDA, TSLA, NFLX 같은 split/dividend 포함 5~20티커로 yfinance 제한 write를 수행한다.
4. write 전후 `ticker,date,source` 기준 row count, duplicate, coverage, adjusted OHLC sanity를 비교한다.
5. 같은 ticker에 대해 indicators를 재계산하고 open_ratio/high_ratio/low_ratio p99/max와 atr_ratio coverage를 확인한다.
6. yfinance source 기준 feature bundle을 생성하되 save-run과 full training은 금지한다.
7. cache manifest에서 provider/hash/feature_version/feature_columns mismatch가 실제로 load 거부되는지 확인한다.
8. EODHD fallback을 일부러 한두 ticker 실패 상황에서 검증하고 fallback count를 metrics에 남긴다.
9. CP 보고서에는 exception list와 원인 분류표를 반드시 포함한다.
10. live inference 연결은 제한 write와 indicators 재계산이 PASS한 뒤 별도 CP로 분리한다.

## 7. 발표/보고서에 쓸 수 있는 데이터 전환 서사

Lens의 yfinance 전환은 "무료 데이터로 갈아탔다"가 아니라 "모델 입력 데이터의 원천을 감사 가능한 방식으로 migration했다"로 설명하는 편이 정확하다.

서사의 핵심은 다음과 같다.

- CP29에서 raw/adjusted 혼용 사고 가능성을 발견하면서, 모델 성능 문제보다 데이터 계약 문제가 먼저라는 판단을 세웠다.
- CP86에서 yfinance를 로컬 primary 후보로 연결하되 EODHD를 삭제하지 않고 fallback으로 남겼다.
- CP87에서 데이터 sanity, feature sanity, model smoke를 단계적으로 검증했고, 데이터 자체는 가능성이 있지만 provenance/cache 격리 부족 때문에 WARN으로 판정했다.
- CP88-D에서 provider/source와 cache hash 계약을 보강했다.
- CP88-G에서는 이 과정을 일반적인 migration/reconciliation 관점으로 정리했다.
- 최종 전환은 비용 절감보다 재현성과 감사 가능성을 유지하는 cutover gate를 통과해야 한다.

발표용 한 문장:

> yfinance 전환은 비용 절감 작업이 아니라, EODHD baseline과 yfinance shadow를 reconcile하고 provider lineage, adjusted OHLC 계약, feature cache isolation까지 검증한 데이터 migration 작업이다.

## 8. 참고 링크

- [AWS DMS data validation](https://docs.aws.amazon.com/dms/latest/userguide/CHAP_Validating.html): source-target row validation, validation state, validation failure table.
- [Google Database Migration Service overview](https://docs.cloud.google.com/database-migration/docs/overview): initial snapshot, continuous replication, source-target sync 후 cutover.
- [dbt-labs audit-helper](https://github.com/dbt-labs/dbt-audit-helper): row count, column diff, all-column comparison, stored failures.
- [Datafold open source data-diff blog](https://www.datafold.com/blog/open-source-data-diff/): row count/hash/value-level diff의 장단점. vendor 문서 기준으로만 사용.
- [BIS BCBS 239](https://www.bis.org/publ/bcbs239.htm): 금융 risk data aggregation/reporting의 정확성, 무결성, governance 원칙.
- [GoldenSource Reference Data](https://www.thegoldensource.com/reference-data/): golden source, lineage, corporate action/time-series validation 관점. vendor 문서 기준.
- [Broadridge corporate actions case study](https://www.broadridge.com/insights/caip-case-study): corporate actions 처리와 single source of truth 사례. vendor case 문서 기준.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): `auto_adjust`, `actions`, interval/start/end 계약.
- [Yahoo adjusted close help](https://help.yahoo.com/kb/SLN28256.html): split/dividend multiplier 기반 adjusted close 설명.
- [yfinance price repair](https://ranaroussi.github.io/yfinance/advanced/price_repair.html): missing adjustment, split/dividend repair, false positive 주의.

