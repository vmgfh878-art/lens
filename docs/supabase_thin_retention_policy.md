# Supabase 얇은 DB 보존 정책

생성일: 2026-05-06

## 1. 결론

Supabase는 더 이상 Lens의 원천 데이터 창고가 아니다.

운영 기준:
- 원천 가격/지표/학습 데이터: local parquet
- 제품 표시 DB: Supabase thin DB
- 제품 예측 저장: latest-only 기본
- 실험 history: local parquet/archive 우선
- pruning/delete: 반드시 parquet backup + row count/checksum 검증 후 별도 승인

## 2. 테이블별 보존 정책

| 테이블 | Supabase 보존 범위 | local/parquet 보존 범위 | 정책 |
|---|---|---|---|
| `stock_info` | 전체 유지 | `data/parquet/stock_info.parquet` | 작고 제품 검색에 필요하므로 Supabase 유지 |
| `model_runs` | 제품 run 2개 + 최근 completed 실험 최대 10개 + 최근 감사 필요 run | 전체 실험 메타 archive | failed/composite/legacy/smoke는 backup 후 삭제 후보 |
| `predictions` | 제품 활성 실행의 최신 asof 기본 | 전체 rolling prediction 이력 | 제품 최신값 전용 기본, rolling 이력은 Supabase에 기본 미저장 |
| `prediction_evaluations` | 제품 활성 실행의 최신 asof 기본 | 전체 rolling evaluation 이력 | 제품 최신값 전용 기본 |
| `backtest_results` | 제품 후보 최종 요약만 유지 | 전체 backtest 이력 | 전략 탐색/실험성 결과는 archive 우선 |
| `price_data` | 제품 표시용 최소 구간 또는 미저장 | 전체 source-aware price 이력 | 전체 저장소 역할 제거 |
| `indicators` | 제품 표시용 최소 1D 구간 또는 미저장 | 전체 source/timeframe indicator 이력 | 전체 저장소 역할 제거 |
| `job_runs` | 최근 운영 진단만 유지 | 필요 시 로그 archive | 오래된 success job은 삭제 후보 |

## 3. 제품 실행 보호 목록

절대 삭제 금지:

| layer | run_id | model | timeframe |
|---|---|---|---|
| line | `patchtst-1D-efad3c29d803` | PatchTST | 1D |
| band | `cnn_lstm-1D-d0c780dee5e8` | CNN-LSTM | 1D |

현재 제품 latest 기준:
- expected latest asof: `2026-05-04`
- latest rows per run: 5 rows
- active validation tickers: AAPL, MSFT, NVDA, TSLA, NFLX

## 4. 예측과 평가 보존 정책

기본 정책:
- Supabase에는 제품 최신값 전용 행만 유지한다.
- 제품 rolling AI band/line 이력이 필요해지면 먼저 로컬 Parquet archive에서 제공한다.
- Supabase rolling 이력은 명시 요구가 생길 때만 최근 N개 asof로 제한한다.

권장 기본값:
- `predictions`: active product run별 latest `asof_date`만 유지
- `prediction_evaluations`: active product run별 latest `asof_date`만 유지
- rolling 이력 N: 기본 0
- 제품 데모용 임시 N이 필요하면 최대 20 trading days로 제한

현재 dry-run 후보:
- `predictions_product_history_except_latest`: 357,474 rows
- `predictions_non_product`: 3,289 rows
- `prediction_evaluations_product_history_except_latest`: 357,474 rows
- `prediction_evaluations_non_product`: 3,289 rows

삭제는 이번 CP에서 하지 않았다.

## 5. 가격과 지표 보존 정책

목표:
- Supabase `price_data`/`indicators`는 전체 이력 저장소가 아니다.
- 로컬 Parquet가 원천 기록이다.

운영 선택지:

| 선택지 | 내용 | 권장 |
|---|---|---|
| A | Supabase에 price/indicator를 거의 두지 않고 backend가 로컬 Parquet에서 serving | 최종 목표 |
| B | Supabase에 제품 표시용 최근 1년 1D price/indicator만 유지 | 배포 제약이 있으면 임시 허용 |
| C | Supabase에 전체 price/indicator 유지 | 금지 방향 |

CP117 기준 판단:
- 단기: 제품 화면이 Supabase price/indicator에 의존한다면 최근 1년, active universe, active source만 남기는 정책을 사용한다.
- 중기: backend API가 local parquet를 읽어 제품 화면에 제공하게 하고 Supabase 원본 price/indicator 의존을 제거한다.
- 장기: Supabase는 예측/메타/검색 thin DB로만 유지한다.

## 6. 모델 실행 보존 정책

보존:
- 제품 실행 2개
- 최근 completed 실험 최대 10개
- 논문/발표/감사에 쓰는 후보 run
- 최신 1W 후보가 확정되면 해당 run 추가

삭제 후보:
- `failed_nan`
- `failed_quality_gate`
- composite legacy
- smoke/debug run
- 제품 후보에서 탈락했고 Parquet archive가 끝난 실행

CP117 dry-run:
- `model_runs` total: 22
- 제품 실행 행: 2
- 비제품 행: 20
- failed quality rows: 9

## 7. Pruning 절차

삭제 전 필수:

1. 후보 row count 산출
2. 후보 row parquet export
3. export row count 확인
4. export checksum 기록
5. product run anti-join 검증
6. 사용자 승인
7. delete 실행
8. delete 후 count 재확인
9. Supabase dashboard에서 용량/egress 확인

삭제 금지:
- product run 최신 row
- `stock_info`
- backup/checksum 없는 row
- source/provider가 불분명한 상태의 price/indicator row

## 8. SQL 실행

테이블별 실제 bytes 측정은 Supabase SQL Editor에서 실행한다. REST client는 임의 SQL 실행이 불가능하므로 CP117 스크립트는 row count dry-run까지만 수행했다.

대표 SQL은 `docs/cp117_local_parquet_supabase_thin_closure_report.md`에 포함했다.

## 9. 최종 정책 문장

Supabase는 제품이 지금 보여줘야 하는 최신 상태만 가진다.  
Lens가 학습하고 검증하고 되돌릴 수 있는 전체 데이터는 local parquet와 archive에 둔다.
