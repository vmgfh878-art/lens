# 로컬 산출물 정리 계획

생성일: 2026-05-04

## 원칙

- 이번 CP에서는 삭제하지 않는다.
- checkpoint, cache, parquet, logs는 먼저 역할을 분류한 뒤 삭제 여부를 별도 승인으로 결정한다.
- yfinance/local parquet 전환이 안정화되기 전에는 `data/parquet`를 보관한다.
- 제품 checkpoint 후보는 삭제하지 않는다.

## 용량 현황

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
| `artifacts` | 2 | 0.25MB |
| `backend/data/cache/yfinance` | 2 | 0.02MB |

## 대형 파일 TOP

| 파일 | 용량 |
|---|---:|
| `.venv\Lib\site-packages\torch\lib\torch_cuda.dll` | 773.96MB |
| `.venv\Lib\site-packages\torch\lib\cublasLt64_12.dll` | 643.41MB |
| `ai\cache\features_1D_043044b40107_960f3bc8.pt` | 562.16MB |
| `ai\cache\features_1D_bcf2a9a91171_960f3bc8.pt` | 562.16MB |
| `ai\cache\features_1D_bcf2a9a91171.pt` | 562.16MB |
| `ai\cache\features_1D_4b3997385f0e_3ac43945.pt` | 535.40MB |
| `ai\cache\features_1D_a6c1541a31c8_3ac43945.pt` | 534.74MB |
| `.venv\Lib\site-packages\torch\lib\cudnn_engines_precompiled64_9.dll` | 458.73MB |
| `.venv\Lib\site-packages\torch\lib\cusparse64_12.dll` | 361.95MB |
| `.venv\Lib\site-packages\torch\lib\cufft64_11.dll` | 263.33MB |
| `frontend\node_modules\@next\swc-win32-x64-msvc\next-swc.win32-x64-msvc.node` | 129.50MB |
| `data\parquet\indicators_yfinance_1D.parquet` | 29.02MB |
| `data\parquet\price_data_yfinance.parquet` | 10.31MB |

## 보관 후보

반드시 보관:
- `data/parquet/stock_info.parquet`
- `data/parquet/price_data_yfinance.parquet`
- `data/parquet/indicators_yfinance_1D.parquet`
- `data/parquet/price_data_yfinance_1W.parquet`
- `data/parquet/price_data_yfinance_1M.parquet`
- `data/parquet/indicators_yfinance_1W.parquet`
- `data/parquet/indicators_yfinance_1M.parquet`

이유:
- CP98~CP101 제품 1D loop와 CP100 1W/1M local snapshot의 원본 창고다.
- Supabase 대량 read를 피하는 핵심 파일이다.

보관하되 별도 목록화:
- `ai/artifacts/checkpoints/`의 제품 run checkpoint
- CP98~CP102 보고서와 metrics JSON
- `logs/cp98_*`, `logs/cp99_*`, `logs/cp101_*` 실행 로그

주의:
- `ai/artifacts`는 460.79MB로 크지 않지만 제품 checkpoint가 섞여 있으므로 일괄 삭제 금지다.

## 삭제 후보

승인 후 삭제 후보:
- 오래된 `ai/cache/features_*.pt`
- 오래된 `ai/cache/feature_index_*.pt`
- stale `ai/cache/*.manifest.json`
- `frontend/.next/`
- `logs/cp*` 중 보고서로 요약이 끝난 오래된 probe log
- `wandb/` 로컬 run cache
- `logs/runs/` 중 학습 재현에 필요 없는 run log

조건:
- 삭제 전 각 cache의 provider/source/hash/feature_version을 목록화한다.
- 제품 checkpoint와 연결된 cache인지 확인한다.
- 삭제 전 최소한 파일명/크기/생성 시점을 metrics로 남긴다.

재생성 가능하지만 즉시 삭제 비권장:
- `.venv/`
- `frontend/node_modules/`

이유:
- 용량은 크지만 개발 환경 복구 시간이 든다.
- 삭제는 디스크 압박이 있을 때만 승인 후 수행한다.

## gitignore 필요 후보

현재 untracked 또는 tracked cache 위험:

- `ai/cache/*.pt.manifest.json`
- `logs/runs/`
- `frontend/tsconfig.tsbuildinfo`
- `backend/data/cache/yfinance/*.db`
- `backend/data/cache/yfinance/*.db-shm`
- `backend/data/cache/yfinance/*.db-wal`

현재 `.gitignore` 근거:
- `docs/cp*`
- `*.pt`
- `*.pth`
- `backend/data/parquet/`
- `*.parquet`
- `wandb/`
- `logs/cp*/`

보강안:

```gitignore
ai/cache/*.manifest.json
ai/cache/*.pt.manifest.json
logs/runs/
frontend/tsconfig.tsbuildinfo
backend/data/cache/yfinance/*.db
backend/data/cache/yfinance/*.db-shm
backend/data/cache/yfinance/*.db-wal
```

## backend/data/cache/yfinance 상태

현재 파일:
- `cookies.db` 8192 bytes
- `tkr-tz.db` 8192 bytes

`git status`에는 `cookies.db`, `tkr-tz.db` modified, `cookies.db-shm`, `cookies.db-wal` deleted가 보인다. 이는 yfinance 실행 중 생기는 로컬 SQLite cache 성격으로 보이며, 코드/데이터 계약 산출물이 아니다.

권장:
- 먼저 현재 tracked 여부를 확인한다.
- tracked라면 별도 승인 후 `git rm --cached` 후보로 분리한다.
- 삭제나 reset은 이번 CP에서 하지 않는다.

## 정리 우선순위

1. gitignore 보강
2. `backend/data/cache/yfinance` 추적 정책 결정
3. `ai/cache` provider/source/hash별 inventory 생성
4. 제품 checkpoint 목록 고정
5. 오래된 feature cache 삭제 승인 절차
6. 오래된 logs/wandb 정리

## 이번 CP에서 하지 않은 것

- 파일 삭제 없음
- cache 삭제 없음
- checkpoint 삭제 없음
- DB 삭제 없음
- Supabase 대량 read 없음
