# CP212 후속 마감 보고 - backend, task, artifact, Render, CSS

작성 시각: 2026-05-27 KST

## 결론

현재 local과 Render backend는 AAPL 기준 가격, 1D line, 1D band, 1W band endpoint를 모두 200으로 응답한다. Windows task는 append-only 구경로가 아니라 line 포함 통합 wrapper를 가리킨다. CP175 line backup도 존재하고, 현재 serving line artifact는 F4 beta=4 ensemble provenance로 교체되어 있다.

단, CP212의 엄격 PASS 조건인 `price latest date = 1D line latest asof = 1D band latest asof`는 아직 아니다. 가격과 1D band는 `2026-05-26`, 1D line은 `2026-05-22`다. 따라서 최종 판정은 `WARN_CP212_AAPL_PARTIAL`이다.

## backend 상태

### local

Base URL: `http://127.0.0.1:8000`

| endpoint | 상태 | 기준값 |
|---|---:|---|
| `/api/v1/health/live` | 200 | 정상 |
| `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=10` | 200 | latest date `2026-05-26` |
| `/api/v1/predictions/line/AAPL?days=30` | 200 | latest asof `2026-05-22` |
| `/api/v1/predictions/band/1d/AAPL?days=30&horizon=5` | 200 | latest asof `2026-05-26`, forecast max `2026-06-02` |
| `/api/v1/predictions/band/1w/AAPL?days=60&horizon=4` | 200 | latest asof `2026-05-22` |

### Render

Base URL: `https://lens-backend-7stj.onrender.com`

| endpoint | 상태 | 기준값 |
|---|---:|---|
| `/api/v1/health/live` | 200 | 정상 |
| `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=10` | 200 | latest date `2026-05-26` |
| `/api/v1/predictions/line/AAPL?days=30` | 200 | latest asof `2026-05-22` |
| `/api/v1/predictions/band/1d/AAPL?days=30&horizon=5` | 200 | latest asof `2026-05-26`, forecast max `2026-06-02` |
| `/api/v1/predictions/band/1w/AAPL?days=60&horizon=4` | 200 | latest asof `2026-05-22` |

PowerShell의 Render HTTPS 호출은 로컬 proxy/cert 문제로 실패할 수 있어, 검증에는 proxy를 제거한 Python `urllib` 호출을 사용했다.

## serving artifact

| 산출물 | 상태 | 확인값 |
|---|---|---|
| `backend/data/v1/predictions_line_1d.parquet` | 존재 | max asof `2026-05-22`, AAPL rows 378 |
| `backend/data/v1/predictions_line_1d_cp175_frozen_backup.parquet` | 존재 | max asof `2026-05-01`, AAPL rows 239 |
| `backend/data/v1/predictions_band_1d.parquet` | 존재 | max asof `2026-05-26`, AAPL rows 1,255 |
| `backend/data/v1/predictions_band_1w.parquet` | 존재 | max asof `2026-05-22`, AAPL rows 420 |
| `backend/data/v1/product_prediction_history_1D.parquet` | 존재 | max asof `2026-05-26`, AAPL rows 1,633 |
| `backend/data/v1/ai_runs_mock.json` | 존재 | line metadata 갱신 확인 |

## line provenance

현재 1D line serving 후보:

- serving model_id: `cp210_F4_b4_ensemble_mean`
- source CP: `CP208Z_CP209_F4B4`
- serving target: `backend/data/v1/predictions_line_1d.parquet`
- 기존 CP175 backup: `backend/data/v1/predictions_line_1d_cp175_frozen_backup.parquet`
- test IC: `0.03249`
- false-safe: `0.20481`
- severe recall: `0.77275`
- WF IC positive folds: `4/4`
- WF stability: moderate

## Windows task binding

Task 이름: `Lens yfinance 500 daily append`

현재 action:

`C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\user\lens\scripts\run_v1_unified_refresh_local.ps1" -Apply -ReloadLocalBackend`

확인한 wrapper 단계:

- append
- market snapshot rebuild
- 1D band refresh
- 1W band refresh
- 1D line refresh/export
- product history rebuild
- ai_runs metadata rebuild
- admin reload 호출 로직

판정: `PASS_CP212_TASK_BOUND`

주의: 최근 실행에서는 `LENS_ADMIN_RELOAD_TOKEN`이 없어 reload가 `SKIPPED_TOKEN_MISSING`이었다. task binding 문제는 아니고 runtime token 환경 문제다.

## frontend / CSS 404

현재 local frontend:

- root `http://127.0.0.1:3000`: 200
- HTML 내 `_next/static/...css`: 200
- 브라우저 AAPL 1D/1W 화면: 스타일 적용 확인
- 브라우저 console error/warn: 0건

반복 CSS 404 원인:

`npm run build`가 `.next` 산출물을 다시 만들거나 build id를 바꾼 뒤 기존 `next dev` 서버가 계속 떠 있으면, 브라우저가 받은 HTML과 `_next/static` asset 경로가 서로 안 맞아 root는 200이지만 CSS/JS static이 404가 날 수 있다.

운영 원칙:

- `npm run build` 후 브라우저 검증 전에는 dev server를 clean restart한다.
- readiness는 root 200만 보지 말고 HTML의 stylesheet 링크를 따라가 CSS 200까지 확인한다.
- `.next` 삭제는 기본 복구가 아니라 수동 복구 옵션으로만 둔다.

이번 검증에서도 `npm run build` 직후 동일한 현상이 재현되었다.

- build 전 readiness: frontend-static OK
- `npm run build` 후 readiness: frontend-static FAIL, stylesheet 404
- 3000 dev server clean restart 후 readiness: frontend-static OK

따라서 이 문제는 CSS 코드 문제가 아니라 stale next dev server와 `.next` 산출물 기준 불일치 문제로 확정한다.

## 브라우저 확인

인앱 브라우저 `http://127.0.0.1:3000/`에서 확인했다.

### AAPL 1D

- 가격 표시: 확인
- 1D 보수적 기준선 표시: 확인
- 1D AI 밴드 표시: 확인
- 가격 최신일: `2026-05-26`
- 1D AI 밴드 기준일: `2026-05-26`, 상태 `fresh`
- 1D 보수적 기준선 기준일: `2026-05-22`, 상태 `fresh`로 표시됨
- console error/warn: 0건

주의: UI 상태는 현재 구현 기준으로 `fresh`지만, CP212 엄격 정합성 기준에서는 line 기준일이 가격/밴드보다 뒤처졌으므로 보고서 판정은 WARN으로 둔다.

### AAPL 1W

- 가격 표시: 확인
- 1W AI 밴드 표시: 확인
- 1W 보수적 기준선 비표시: 확인
- 1W 보수적 기준선 상태: `deferred`
- 1W AI 밴드 기준일: `2026-05-22`, 상태 `fresh`
- console error/warn: 0건

## 검증 명령

| 명령 | 결과 |
|---|---|
| `npm run build` | PASS. 최초 sandbox 실행은 `spawn EPERM`으로 실패했고, 권한 상승 실행에서 PASS |
| `powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1` | PASS. build 직후에는 frontend-static 404를 재현했고, dev server 재시작 후 PASS |
| `git diff --check` | PASS. `logs/run_backend_demo.ps1` CRLF 변환 경고만 있음 |
| local endpoint recheck | PASS. health/prices/line/band 200 |
| Render endpoint recheck | PASS. health/prices/line/band 200 |

## Render α 경로

현재 Render endpoint는 local과 같은 serving 값을 반환한다. 따라서 배포 backend는 현재 산출물을 읽고 있다.

다만 이번 마감에서 새 serving refresh를 실행하지 않았고, line latest asof가 `2026-05-22`로 남아 있으므로 “완전 정합 PASS”라고 쓰면 틀리다. Render α 경로의 다음 조건은 line export가 가격 최신일 `2026-05-26`까지 따라오는 것이다.

## 최종 판정

`WARN_CP212_AAPL_PARTIAL`

완료:

- local backend endpoint 200
- Render backend endpoint 200
- frontend CSS 200
- AAPL 1D/1W 화면 확인
- Windows task binding PASS
- CP175 backup/provenance 확인
- `npm run build` PASS
- readiness PASS after frontend dev restart

남은 리스크:

- 1D line latest asof `2026-05-22`가 가격/1D band `2026-05-26`보다 늦다.
- task runtime reload token이 없어 자동 실행 직후 live cache reload가 skipped 될 수 있다.
- 1W band API는 현재 `forecast_date`를 응답에 직접 노출하지 않는다.
- PowerShell HTTPS 호출은 로컬 proxy/cert 환경 때문에 Render 직접 확인에 실패할 수 있다. Python no-proxy 호출로는 200 확인됨.
