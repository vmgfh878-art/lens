# CP102-DG 제품 저장 계약 및 로컬/깃 정리 감사

생성일: 2026-05-04

## 1. Executive Summary

최종 판정: WARN

CP101-D에서 EODHD 없이 1D 제품 루프는 통과했지만, yfinance가 local snapshot 최신일인 2026-05-01 이후 새 거래일 row를 아직 주지 않아 다음 세 항목은 미검증으로 남았다.

- yfinance 신규 거래일 row append
- `data/parquet/price_data_yfinance.parquet` local append
- `data/parquet/indicators_yfinance_1D.parquet` 1D indicator incremental refresh append

제품 저장 계약은 기존 `ai.inference --save`가 전체 prediction history를 저장할 수 있는 구조라서 제품용 latest-only 경로와 분리할 필요가 있었다. 이번 CP에서 최소 구현으로 `ai.storage.save_product_latest_predictions()`를 추가했고, CP99/CP101 thin upload 스크립트가 이 헬퍼를 사용하도록 바꿨다. 이 헬퍼는 row 수 제한, 단일 `asof_date`, `line`/`band` layer만 허용, composite 저장 금지를 검사한다.

작업트리는 매우 큰 dirty 상태다. `git status --short` 기준 총 148개 항목이며, tracked 변경 70개, untracked 78개, 삭제 표시 2개가 있다. 따라서 지금은 하나의 커밋으로 묶으면 추적성이 무너진다. 데이터 전환, Supabase 비용/가드, 제품 latest loop, 모델/평가, 프론트 UI, 문서 묶음으로 나누는 커밋 계획이 필요하다.

로컬 산출물은 `ai/cache`가 7.4GB로 가장 크다. 삭제는 하지 않았고, 삭제 후보와 보관 후보만 분리했다. 특히 `*.pt.manifest.json`, `logs/runs/`, `frontend/tsconfig.tsbuildinfo`, `backend/data/cache/yfinance/*.db*`는 gitignore/추적 정책 정리가 필요하다.

Supabase Pro는 지금 선결제하기보다, egress 방화벽과 local parquet 전환이 이미 작동하는 상태라면 402/read-only가 실제로 발생하거나 발표/운영 일정상 복구 시간이 더 중요해지는 시점에 결제하는 전략이 합리적이다. 단, 2026-05-04 공식 문서 기준 Free DB size 제한은 500MB로 안내되어 있어 기존 "2GB Free DB" 전제는 재확인이 필요하다.

## 2. CP101 후속 Gate

### EODHD 없이 통과한 항목

- `MARKET_DATA_PROVIDER=yfinance`, `MARKET_DATA_FALLBACK_PROVIDER=` 조건에서 설정 로딩 통과
- EODHD API key 없이 provider 경로 동작
- local yfinance snapshot 기반 1D line/band inference 성공
- AAPL/MSFT/NVDA/TSLA/NFLX 5티커 latest-only thin upload 성공
- AAPL 1D line/band API 조회 성공
- Supabase `price_data`/`indicators` 대량 read guard 통과
- composite 저장 없음

근거:
- `docs/cp101_eodhd_off_1d_operation_rehearsal_metrics.json`: `final_decision.eodhd_off_pass=true`, `inference_pass=true`, `thin_upload_pass=true`, `bulk_read_guard_pass=true`
- `backend/collector/config.py:42`: `MARKET_DATA_FALLBACK_PROVIDER` 환경값을 명시적으로 읽음
- `backend/collector/config.py:44`: fallback 기본값은 env 미지정일 때만 eodhd로 보정

### 아직 미검증인 항목

CP101 실행 시 yfinance 최신 조회는 성공했지만, 조회된 최신 거래일이 이미 snapshot 최신일인 2026-05-01과 같았다. 따라서 local append는 `price_rows_written=0`, reason은 `no_new_rows`였다.

미검증 항목:
- 신규 거래일 row가 실제로 생긴 날 yfinance fetch가 새 row를 반환하는지
- 새 row가 `price_data_yfinance.parquet`에 중복 없이 append되는지
- append 후 1D indicator refresh가 해당 ticker만 증분 갱신되는지
- append 후 `source_data_hash`가 갱신되고 제품 inference가 최신 `asof_date`로 이동하는지

### 다음 장 종료 후 확인 절차

다음 미국장 종료 후, 한국 시간 기준 장 마감 데이터가 Yahoo Finance에 반영된 뒤 아래 절차로 확인한다.

권장 시점:
- 미국 정규장 종료 후 최소 3~6시간 뒤
- 한국 시간 다음날 오전 또는 오후

절차:
1. EODHD 비활성 환경 확인
2. CP101 스크립트를 5티커로 재실행하되 local parquet append는 제한 범위만 허용
3. `latest_yfinance_update.local_update.price_rows_written > 0` 확인
4. `indicator_rebuild_tickers`에 대상 티커가 기록되는지 확인
5. snapshot latest date가 이전보다 증가했는지 확인
6. AAPL 1D line/band API 조회 `asof_date`가 새 snapshot 최신일로 이동했는지 확인

예시 명령:

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:MARKET_DATA_FALLBACK_PROVIDER=""
$env:LENS_DATA_BACKEND="local"
$env:LENS_REQUIRE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\Users\user\lens\data\parquet"
python scripts\cp101_eodhd_off_1d_operation_rehearsal.py --tickers AAPL MSFT NVDA TSLA NFLX --apply-local-update
```

판정 제안:
- 위 append gate가 PASS하면 EODHD 해지 가능 후보로 본다.
- append gate가 WARN이면 EODHD 결제 해지 전 1~2거래일 더 관찰한다.
- yfinance fetch 실패, source 혼합, indicator refresh 실패가 있으면 EODHD 해지를 보류한다.

## 3. 제품용 Latest-only 저장 계약

### 기존 위험

`ai.inference --save`는 일반 실험 저장 경로다. `ai/inference.py:70`의 `--save` 옵션이 켜지면 `ai/inference.py:479`와 `ai/inference.py:480`에서 생성된 `prediction_records`, `evaluation_records` 전체를 저장한다. 이 경로는 split, ticker 범위, 생성 레코드 수에 따라 전체 history 저장으로 이어질 수 있다.

따라서 제품 화면용 latest-only 저장은 일반 inference 저장과 분리하는 것이 맞다.

### 이번 CP 최소 구현

추가/변경:
- `ai/storage.py:10`: 제품 latest 저장 허용 layer를 `line`, `band`로 제한
- `ai/storage.py:53`: prediction row 수, 단일 `asof_date`, composite 금지, layer 검증
- `ai/storage.py:85`: evaluation row 수와 단일 `asof_date` 검증
- `ai/storage.py:95`: `save_product_latest_predictions()` 추가
- `scripts/cp99_1d_product_loop_thin_upload.py:370`: CP99 thin upload 저장 함수가 제품용 helper를 사용
- `scripts/cp99_1d_product_loop_thin_upload.py:374`: prediction/evaluation 각각 20 row 제한
- `ai/tests/test_storage_contracts.py:70`: 제품 latest 저장 guard 테스트 추가

의도:
- 제품 latest 저장은 `line`/`band` layer만 허용
- composite 저장 금지
- 여러 `asof_date`가 섞인 history 저장 금지
- row 수 제한 초과 시 저장 전 실패
- 기존 `save_predictions()`와 `save_prediction_evaluations()`는 일반 실험 경로로 유지

남은 결정:
- `ai.inference` CLI에 `--save-product-latest-only`를 추가할지는 다음 CP에서 판단한다.
- 지금은 CP99/CP101 thin upload 스크립트가 제품 저장의 공식 경로이므로, 일반 `--save`를 제품 운영 명령으로 쓰지 않는 운영 규칙을 문서화한다.

## 4. Git 작업트리 감사 요약

실행 결과:
- `git status --short`: 총 148개 항목
- tracked changed: 70
- untracked: 78
- deleted 표시: 2
- `git status --ignored --short`: 총 666개 항목, ignored 518

상위 dirty 범주:
- `ai`: 76개
- `backend`: 40개
- `docs`: 15개
- `frontend`: 9개
- `scripts`: 7개
- `logs`: 1개

분류:
- 데이터 전환: yfinance provider, source-aware schema, local snapshots, preprocessing cache/hash, split planning
- Supabase 비용/가드: local backend, export parquet, upsert returning minimal, readiness 대량 read 차단
- 제품 저장/운영: CP98/CP99/CP101/CP102 스크립트와 latest-only 저장 helper
- 모델/평가: line/band metric, local logging, sweep/cache 관련 변경
- 제품 UI: StockView, Chart, TrainingView, BacktestView 등 대형 UI 변경
- 문서/보고서: CP 보고서, 데이터 아키텍처 SoT, Supabase 보고서
- 로컬 cache/artifact: `ai/cache`, `logs/runs`, yfinance cookie/tz DB

실제 `git add`, `git commit`, `git reset`, 파일 삭제는 하지 않았다.

자세한 커밋 분리안은 `docs/git_cleanup_plan.md`에 작성했다.

## 5. 로컬 산출물 감사 요약

용량 큰 디렉터리:

| 경로 | 파일 수 | 용량 |
|---|---:|---:|
| `ai/cache` | 156 | 7400.73MB |
| `.venv` | 43901 | 5592.73MB |
| `ai/artifacts` | 192 | 460.79MB |
| `frontend/node_modules` | 9958 | 264.55MB |
| `frontend/.next` | 139 | 90.62MB |
| `data/parquet` | 7 | 50.71MB |
| `logs` | 138 | 14.54MB |
| `docs` | 373 | 7.25MB |
| `wandb` | 388 | 4.49MB |

상위 대형 파일은 대부분 `.venv`의 PyTorch CUDA DLL과 `ai/cache/features_*.pt`다. 제품 checkpoint가 있는 `ai/artifacts`는 무작정 삭제하면 안 된다. `data/parquet`는 현재 yfinance/local 제품 루프의 핵심 원본 창고이므로 보관한다.

자세한 삭제 후보/보관 후보/gitignore 후보는 `docs/local_artifact_cleanup_plan.md`에 작성했다.

## 6. Supabase Pro 판단

판정: 지금 즉시 선결제는 보류, 402/read-only 또는 발표 일정상 복구 시간이 더 중요해지는 시점에 결제.

근거:
- CP98~CP101에서 `price_data`/`indicators` 대량 read 없이 local parquet 기반 1D 루프가 동작했다.
- 제품 thin upload는 5티커 latest-only 수준이라 DB egress/write 부담이 작다.
- 목표 구조는 Free 복귀 가능한 thin DB다.
- Pro를 먼저 켜면 egress/DB slimming 압력이 줄어들어 근본 정리가 늦어질 수 있다.

단, 2026-05-04 공식 문서 기준 Supabase Free Plan은 Database size 500MB 제한을 명시한다. 기존 "Free DB 2GB" 전제는 현재 공식 기준과 다를 수 있으므로 Supabase dashboard의 실제 Usage 수치로 재확인해야 한다.

Pro를 켜야 하는 조건:
- 402 또는 read-only가 실제로 발생해 제품 thin upload/API 확인이 막힘
- 발표/시연 일정상 장애 복구 시간을 돈으로 사는 편이 유리함
- pruning/export 전까지 단기적으로 500MB 이하 감축이 불가능함

Spend Cap:
- Pro로 전환한다면 Spend Cap은 켜는 것을 권장한다.
- egress/DB 실험을 다시 크게 돌릴 계획이 있으면 Spend Cap이 제한으로 작동할 수 있으므로, 끄기 전에는 row 수와 예상 egress를 따로 계산해야 한다.

자세한 판단 메모는 `docs/supabase_pro_decision_note.md`에 작성했다.

## 7. 검증

실행한 검증:

```powershell
python -m py_compile ai\storage.py ai\inference.py scripts\cp99_1d_product_loop_thin_upload.py scripts\cp101_eodhd_off_1d_operation_rehearsal.py
python -m unittest ai.tests.test_storage_contracts
```

결과:
- py_compile PASS
- `ai.tests.test_storage_contracts`: 5 tests PASS
- CP102 metrics JSON parse PASS 예정

금지 항목 확인:
- DB row 삭제 없음
- Supabase `price_data`/`indicators` 대량 read 없음
- 전체 yfinance write 없음
- indicators full recompute 없음
- full model training 없음
- live inference 대량 저장 없음
- checkpoint/cache 실제 삭제 없음
- EODHD row 삭제 없음
- 프론트 UI 수정 없음

## 8. 읽기 전용/검증 명령 목록

- `git status --short`
- `git status --ignored --short`
- `git diff --stat`
- 로컬 디렉터리 용량 집계 PowerShell 명령
- `Select-String` 기반 코드 경로 확인
- `Get-Content docs\cp101_eodhd_off_1d_operation_rehearsal_metrics.json`
- `Get-Content docs\cp99_1d_product_loop_thin_upload_metrics.json`
- `Get-Content docs\cp98_local_parquet_snapshot_bootstrap_metrics.json`
- `python -m py_compile ...`
- `python -m unittest ai.tests.test_storage_contracts`

외부 참고:
- Supabase 공식 Billing 문서
- Supabase 공식 Cost Control/Spend Cap 문서
- Supabase 공식 Database and Disk Size 문서

## 9. 최종 판단

CP102는 WARN이다.

latest-only 저장 계약은 최소 구현으로 막았다. 하지만 EODHD 해지 전 마지막 핵심 gate인 "신규 거래일 yfinance append + local parquet append + 1D indicator incremental refresh"는 아직 시장 데이터 타이밍상 실전 검증되지 않았다. 또한 작업트리가 너무 커서 커밋 분리 없이 진행하면 재현성과 롤백성이 나빠진다.

다음 CP 추천:
1. 다음 거래일 append gate 실전 확인
2. gitignore/커밋 분리 정리
3. 제품 latest-only CLI 또는 운영 명령 고정
