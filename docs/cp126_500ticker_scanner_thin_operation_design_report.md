# CP126-DG 500티커 운영 Inference / Scanner Thin 설계 감사

생성일: 2026-05-06  
상태: PASS_DESIGN_WITH_GATES  
범위: 설계 감사 전용. DB schema 변경, DB write, 모델 학습, inference 저장은 하지 않았다.

## 1. Executive Summary

결론: 500티커 운영은 가능하지만, Supabase에 500티커 raw prediction을 저장하는 방식은 금지해야 한다. 운영 구조는 `500티커 local forward-only inference -> local full prediction archive -> scanner top-k 산출 -> Supabase top-k thin upload`가 맞다.

현재 local parquet는 100티커까지만 준비되어 있다. 다만 `backend/data/universe/sp500.csv`에는 503개 티커가 있고, CP117 기준 Supabase `stock_info`는 507행이므로 500티커 universe 후보 자체는 있다. 다음 단계는 Supabase write가 아니라 500티커 local snapshot bootstrap과 forward-only timing rehearsal이다.

판정:
- 설계: PASS
- 즉시 운영 실행: WARN
- WARN 이유: local 500 snapshot이 아직 없고, scanner table schema는 제안 단계이며, selected prediction latest-only가 기존 `predictions` table에서는 asof_date별 row 누적으로 이어질 수 있다.

## 2. 금지 작업 확인

| 항목 | 발생 |
|---|---:|
| DB schema 실제 변경 | false |
| DB write | false |
| 모델 학습 | false |
| inference 저장 | false |
| 전체 500티커 Supabase upload | false |
| EODHD 호출 | false |

## 3. 현재 Local Parquet Coverage

읽은 파일은 `data/parquet` 아래 local snapshot뿐이다.

| snapshot | rows | tickers | date min | date max | size |
|---|---:|---:|---|---|---:|
| `stock_info.parquet` | 100 | 100 | - | - | 7,088 B |
| `price_data_yfinance.parquet` | 284,905 | 100 | 2015-01-02 | 2026-05-04 | 10,783,676 B |
| `indicators_yfinance_1D.parquet` | 279,005 | 100 | 2015-03-30 | 2026-05-04 | 30,491,865 B |
| `price_data_yfinance_1W.parquet` | 59,200 | 100 | 2015-01-02 | 2026-05-01 | 3,717,582 B |
| `indicators_yfinance_1W.parquet` | 53,300 | 100 | 2016-02-19 | 2026-05-01 | 6,433,851 B |
| `price_data_yfinance_1M.parquet` | 13,600 | 100 | 2015-01-31 | 2026-04-30 | 853,753 B |
| `indicators_yfinance_1M.parquet` | 7,700 | 100 | 2019-12-31 | 2026-04-30 | 916,756 B |

source/provider:
- 모든 local price/indicator row는 `source=yfinance`, `provider=yfinance`.
- 현재 500티커 coverage는 아직 없다.

500티커 확장 가능성:
- `backend/data/universe/sp500.csv`: 503 rows
- CP117 Supabase `stock_info`: 507 rows
- 현재 local `stock_info.parquet`: 100 rows
- 따라서 500티커 운영 전에는 `stock_info.parquet`, `price_data_yfinance.parquet`, `indicators_yfinance_1D/1W.parquet`를 500티커 local-only로 확장해야 한다.

## 4. 500티커 Local 용량 추정

현재 100티커 파일을 단순 5배 scaling한 추정이다. 압축률과 ticker별 상장 기간 차이는 반영하지 않았다.

| 대상 | 500티커 rows 추정 | size 추정 |
|---|---:|---:|
| 1D price | 1,424,525 | 53.9 MB |
| 1D indicators | 1,395,025 | 152.5 MB |
| 1W price | 296,000 | 18.6 MB |
| 1W indicators | 266,500 | 32.2 MB |
| 1M price | 68,000 | 4.3 MB |
| 1M indicators | 38,500 | 4.6 MB |

판단:
- local parquet 용량은 500티커 1D/1W 기준 약 257 MB로 충분히 작다.
- 1M까지 포함해도 약 266 MB 수준이다.
- full raw prediction daily archive도 Supabase가 아니라 local parquet로 두면 비용 리스크는 낮다.

## 5. Daily 운영 시간 추정

이번 CP에서는 yfinance 호출과 inference를 실행하지 않았다. 아래는 기존 CP 기록과 현재 코드 구조 기반 추정이다.

| 단계 | 추정 | 신뢰도 | 근거 |
|---|---:|---|---|
| yfinance 500티커 daily append | 3~15분 | 중하 | 기존 DB sync 경로는 ticker별 loop와 `sleep_seconds=0.3`이 있어 500티커 최소 150초 + 네트워크/재시도 |
| 1D indicator incremental refresh | 1~8분 | 중 | 완전 재계산은 약 139만 row, 증분 lookback 80일이면 내부 계산 약 4만 row |
| 1D line/band product inference 500 | GPU 1~5분, CPU 5~20분 | 중하 | CP116 5티커 통과, 500티커 forward-only는 아직 미측정 |
| 1W line/band product inference 500 | GPU 1~3분, CPU 3~12분 | 중하 | 1W rows는 1D보다 작지만 1W product closure 전 |
| scanner signal 계산 | 고정 룰 10개 이하 10~60초 | 중 | CP106은 95티커 x 5,184룰 grid에 3,204.714초. 운영은 grid가 아니라 고정 policy |

중요:
- `backend/collector/jobs/sync_prices.py`의 기존 경로는 Supabase `price_data` write와 `stock_info/company_fundamentals` read가 들어가므로 500티커 scanner 운영 경로로 쓰면 안 된다.
- 500티커 운영용 append는 CP115 계열처럼 local parquet first, Supabase price/indicator write 없음, EODHD fallback 없음이어야 한다.

## 6. Scanner Thin 저장 설계

제안 schema:
- `scanner_runs`
- `scanner_signals`

상세 DDL은 `docs/scanner_thin_schema_proposal.md`에 분리했다. 이번 CP에서는 schema를 실제 변경하지 않았다.

계약:
- `scanner_runs`: 하루/타임프레임/score_policy 단위 run 요약 1행
- `scanner_signals`: run별 top-k 신호만 저장
- 기본 `top_k=50`, 최대 `top_k=100`
- full 500 raw prediction은 local parquet만 보관
- API는 top-k만 반환

권장 local artifact:

```text
data/scanner/YYYY-MM-DD/1D/full_predictions.parquet
data/scanner/YYYY-MM-DD/1D/ranked_universe.parquet
data/scanner/YYYY-MM-DD/1D/topk_signals.parquet
data/scanner/YYYY-MM-DD/1W/full_predictions.parquet
data/scanner/YYYY-MM-DD/1W/ranked_universe.parquet
data/scanner/YYYY-MM-DD/1W/topk_signals.parquet
```

## 7. Supabase Upload Row 수

기본안: `top_k=50`, line/band 2 layer, evaluation은 forward-only 단계에서는 저장하지 않는다.

| 범위 | scanner_runs | scanner_signals | selected predictions | evaluations | 총 row |
|---|---:|---:|---:|---:|---:|
| 1D 1회 | 1 | 50 | 100 | 0 | 151 |
| 1W 1회 | 1 | 50 | 100 | 0 | 151 |
| 1D+1W 같은 날 | 2 | 100 | 200 | 0 | 302 |

선택:
- top-k 목록 화면만 필요하면 `scanner_runs + scanner_signals`만으로 충분하다.
- ticker 상세 차트까지 바로 보여야 하면 selected ticker에 한해 existing `predictions` latest row를 저장한다.
- 실제값이 없는 forward-only 시점에는 `prediction_evaluations` 저장을 권장하지 않는다.

## 8. Supabase 용량 및 Egress 추정

CP116 기준 5티커 line prediction payload는 약 6,279 bytes, band도 약 6,279 bytes였다. 즉 prediction row는 대략 1.2~1.3 KB 수준이다. Postgres JSONB/index overhead를 넉넉히 2.5배로 잡았다.

| 항목 | 추정 |
|---|---:|
| 1D top_k=50 1회 payload | 약 188 KB |
| 1D top_k=50 DB 반영량 | 약 469 KB |
| 1D+1W top_k=50 1회 DB 반영량 | 약 938 KB |
| 1D 30거래일 보존 | 약 14 MB |
| 1W 52주 보존 | 약 24 MB |

egress:
- scanner top-k API 1회 응답은 top_k=50 기준 대략 50~150 KB로 제한 가능하다.
- full 500 prediction API는 만들지 않는다.
- full history chart는 기존처럼 ticker detail에서 필요한 ticker만 조회한다.
- `Cache-Control`/ETag를 유지하면 scanner list 반복 조회 egress는 작게 유지 가능하다.

## 9. Selected Ticker Latest Prediction 저장 방식

현재 근거:
- `backend/db/schema.sql`의 `predictions` unique key는 `run_id,ticker,model_name,timeframe,horizon,asof_date`다.
- `ai/storage.py`에는 product latest guard가 있으며 기본 max row는 100이다.
- 이 guard는 1D top_k=50 line+band 100 row에는 딱 맞지만, 1D+1W 동시 저장 또는 top_k=100에는 부족하다.

권장:
1. scanner top-k 저장은 `scanner_signals`를 canonical output으로 둔다.
2. selected prediction은 ticker 상세 차트용 보조 저장으로만 둔다.
3. meta에 `storage_contract=scanner_selected_latest_only`, `scanner_run_id`, `scanner_rank`, `layer`를 넣는다.
4. asof_date 누적을 막기 위해 별도 CP에서 pruning 또는 `product_latest_predictions` 계열 latest table을 결정한다.

주의:
- 기존 `predictions` table만 쓰면 asof_date가 바뀔 때 row가 누적된다.
- CP117에서 이미 product history excess가 크게 생겼던 만큼, 500 scanner 전에는 latest-only pruning gate가 필수다.

## 10. API 계약

제안 API:
- `GET /api/v1/scanner/runs/latest?timeframe=1D`
- `GET /api/v1/scanner/signals/latest?timeframe=1D&limit=50`
- `GET /api/v1/scanner/runs/{scanner_run_id}/signals?limit=50`

응답 계약:
- top-k만 반환
- 기본 limit 50, 최대 100
- full forecast series 제외
- summary 값만 포함: score, signal, lower_return, upper_return, band_width_return, conservative_return, rsi, atr_ratio, ma60_ratio
- 상세 chart는 기존 ticker prediction latest API를 사용

## 11. Pruning 기준

Supabase:
- `scanner_runs`: 1D 최근 90거래일, 1W 최근 52주
- `scanner_signals`: run별 top-k만 저장하므로 run retention과 같이 삭제
- `predictions`: active selected latest만 유지. 과거 asof는 백업 후 pruning
- `prediction_evaluations`: forward-only scanner에는 기본 저장하지 않음

Local:
- full 500 raw prediction: 최근 1년
- ranked universe: 최근 1년
- 오래된 daily full prediction은 월 단위 archive 후 삭제 후보

삭제 실행 전:
- parquet export
- row count 확인
- checksum 기록
- product run 보호 anti-join
- 사용자 승인

## 12. 리스크와 Gate

| 등급 | 리스크 | 판단 | 다음 조치 |
|---|---|---|---|
| P1 | 현재 local snapshot이 100티커뿐 | 500 운영 불가 | 500 local snapshot bootstrap |
| P1 | selected prediction이 `predictions`에 asof별 누적 | thin DB 목적 훼손 | pruning 또는 latest table 결정 |
| P1 | 기존 DB sync 경로를 500 scanner에 재사용 | Supabase write/egress 위험 | local-only append runner 필요 |
| P2 | yfinance 500 ticker 일부 실패 | universe coverage 흔들림 | exception list와 retry policy |
| P2 | 1W product loop closure 전 | 1W 저장 확대 위험 | 1W latest-only rehearsal |
| P2 | scanner score가 test-driven으로 과최적화 | 제품 신호 신뢰도 저하 | validation/regime stability 필드 저장 |

## 13. 다음 CP 제안

1. CP127-D: 500티커 local yfinance snapshot bootstrap dry-run
   - Supabase write 없음
   - EODHD 없음
   - coverage/failure ticker list 생성

2. CP128-DG: scanner schema migration dry-run + API 설계 확정
   - `scanner_runs`, `scanner_signals` 실제 migration은 별도 승인 후
   - `returning=minimal` upsert 확인

3. CP129-D: 500티커 forward-only inference timing rehearsal
   - 저장은 local parquet만
   - top-k 산출
   - Supabase upload는 dry-run row count만

4. CP130-D: top-k scanner thin upload 제한 저장
   - top_k=20부터 시작
   - API read 확인
   - egress/row count 확인

## 14. 읽기 전용 명령 목록

- `Get-ChildItem data/parquet`
- pandas `read_parquet`로 local snapshot row/date/source 통계 산출
- `Import-Csv backend/data/universe/sp500.csv`
- `Select-String`으로 schema/storage/API 계약 확인
- `Get-Content`으로 CP100/CP115/CP116/CP117/CP106 보고서와 metrics 확인

`rg`는 현재 세션에서 `Access is denied`로 실행되지 않아 PowerShell `Select-String`으로 대체했다.
