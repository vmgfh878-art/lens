# CP212-LG 후속 검증 - Windows task 통합 wrapper 재확인

검증 시각: 2026-05-27 KST

## 질문

현재 Windows task가 append만 도는 예전 경로가 아니라, market rebuild + 1D band + 1W band + 1D line + reload까지 포함한 통합 wrapper를 실제로 실행하느냐?

## 현재 task

- task 이름: `Lens yfinance 500 daily append`
- 실행 파일: `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe`
- action command: `-NoProfile -ExecutionPolicy Bypass -File "C:\Users\user\lens\scripts\run_v1_unified_refresh_local.ps1" -Apply -ReloadLocalBackend`
- working directory: `C:\Users\user\lens`
- 다음 실행 시각: `2026-05-28 08:20:00 KST`
- 마지막 실행 시각: `2026-05-27 08:20:00 KST`
- 마지막 실행 결과 코드: `0`

## 실제 wrapper 경로

`C:\Users\user\lens\scripts\run_v1_unified_refresh_local.ps1`

## wrapper 단계 확인

| 단계 | 포함 여부 | 확인 근거 |
|---|---:|---|
| append | 포함 | `cp151_yfinance_500_backfill.py` 실행 후 step name `append` 기록 |
| market snapshot rebuild | 포함 | `backend\scripts\build_v1_market_local.py --asof-date $RunDate` |
| 1D band refresh | 포함 | `backend\scripts\cp210_band_forward_refresh.py` 내부에서 `predictions_band_1d.parquet` 생성 |
| 1W band refresh | 포함 | `backend\scripts\cp210_band_forward_refresh.py` 내부에서 `predictions_band_1w.parquet` 생성 |
| 1D line refresh/export | 포함 | `backend\scripts\cp212_line_1d_export.py` |
| product history rebuild | 포함 | `backend\scripts\rebuild_product_history_parquet.py` |
| ai_runs metadata rebuild | 포함 | `backend\scripts\build_ai_runs_mock.py` |
| unified reload | wrapper에 포함 | task action에 `-ReloadLocalBackend`, wrapper에 `/api/v1/admin/reload` 호출 로직 존재 |

## band wrapper 세부 확인

`backend\scripts\cp210_band_forward_refresh.py`는 다음 산출물을 명시적으로 다룬다.

- `backend/data/v1/predictions_band_1d.parquet`
- `backend/data/v1/predictions_band_1w.parquet`
- `backend/data/v1/product_prediction_history_1D.parquet`

따라서 현재 task가 호출하는 wrapper는 1D band와 1W band를 둘 다 포함한다.

## static fallback 여부

현재 task action은 static fallback rebuild 또는 append-only 경로를 직접 호출하지 않는다.

확인한 현재 경로:

`Task Scheduler -> run_v1_unified_refresh_local.ps1 -> append -> market_snapshot -> band_refresh -> line_export -> product_history_rebuild -> ai_runs_mock_rebuild -> reload`

## 최근 실행 상태

최근 CP212 실행 로그 기준:

- append: `PASS`
- market_snapshot: `PASS`
- band_refresh: `PASS`
- line_export: `PASS`
- product_history_rebuild: `PASS`
- ai_runs_mock_rebuild: `PASS`
- reload_status: `SKIPPED_TOKEN_MISSING`
- final_status: `WARN_UNIFIED_REFRESH_PARTIAL`

reload는 wrapper와 task action에 포함되어 있지만, 최근 실행에서는 `LENS_ADMIN_RELOAD_TOKEN`이 없어 실제 reload 호출이 건너뛰어졌다.

## 수정 여부

수정하지 않았다.

이유:

- 현재 Windows task binding은 이미 통합 wrapper를 정확히 가리킨다.
- append-only, band-only, static fallback 구경로가 아니다.
- 문제는 task binding이 아니라 reload token 미설정으로 인한 runtime reload skip이다.

## 판정

`PASS_CP212_TASK_BOUND`

현재 Windows task는 line 포함 통합 wrapper를 정확히 가리킨다.

보조 경고:

`WARN_RUNTIME_RELOAD_TOKEN_MISSING`

최근 실행에서 reload가 `SKIPPED_TOKEN_MISSING`으로 끝났으므로, 자동 실행 직후 살아 있는 backend cache까지 즉시 reload하려면 `LENS_ADMIN_RELOAD_TOKEN` 환경을 task 실행 컨텍스트에 제공해야 한다.
