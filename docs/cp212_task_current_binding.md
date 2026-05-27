# CP212 Windows task 현재 binding

검증 시각: 2026-05-27 KST

## 대상 task

- task 이름: `Lens yfinance 500 daily append`
- task path: `\`
- 상태: `Ready`

## 현재 action

- 실행 파일: `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe`
- 실행 인자: `-NoProfile -ExecutionPolicy Bypass -File "C:\Users\user\lens\scripts\run_v1_unified_refresh_local.ps1" -Apply -ReloadLocalBackend`
- 시작 위치: `C:\Users\user\lens`

## 스케줄 상태

- 다음 실행 시각: `2026-05-28 08:20:00 KST`
- 마지막 실행 시각: `2026-05-27 08:20:00 KST`
- 마지막 실행 결과 코드: `0`
- trigger: weekly, `08:20`, 화요일~토요일로 해석됨

## 현재 binding 판정

현재 task는 append-only 또는 band-only 구경로가 아니라 `scripts/run_v1_unified_refresh_local.ps1` 통합 wrapper를 실행한다.

단, 최근 CP212 실행 산출물 기준 reload 단계는 `LENS_ADMIN_RELOAD_TOKEN` 부재로 `SKIPPED_TOKEN_MISSING`이었다. 이것은 task action이 구경로를 가리키는 문제는 아니고, reload runtime 토큰/환경 문제다.
