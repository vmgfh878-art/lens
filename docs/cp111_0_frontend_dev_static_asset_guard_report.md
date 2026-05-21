# CP111-0-P 프론트 dev 서버 static asset 안정화 보고서

CP111-0-P는 백테스트 전략 반영 전에, 프론트 dev 서버가 200을 반환해도 CSS/JS static asset이 404가 나서 무스타일 HTML로 보이는 반복 문제를 readiness 단계에서 잡고 복구하는 작업이다.

## 1. 원인

- `npm run build`가 `frontend/.next` 산출물을 다시 만들거나 덮어쓴 뒤에도 기존 `next dev` 서버가 계속 떠 있으면, 브라우저가 받는 HTML과 `/_next/static/...` 파일 경로가 서로 어긋날 수 있다.
- 이 경우 `http://127.0.0.1:3000` 루트는 200을 반환하지만, HTML 안의 stylesheet 링크가 가리키는 CSS 파일은 404가 된다.
- readiness가 기존처럼 프론트 root 200만 확인하면 이 상태를 정상으로 오판한다.

## 2. 변경 파일

### `scripts/check_demo_readiness.ps1`

- 프론트 root 200 확인 뒤 HTML에서 `rel="stylesheet"` CSS 링크를 추출한다.
- 추출한 CSS URL을 실제로 요청해 200인지 확인한다.
- CSS 요청이 실패하면 `FAIL frontend-static`으로 표시한다.
- `npm run build` 후에는 기존 `next dev` 서버를 재시작해야 static asset 404를 피할 수 있다는 안내를 readiness 출력 끝에 추가했다.
- Windows PowerShell에서 한글 안내가 깨지지 않도록 UTF-8 BOM을 적용했다.

### `scripts/restart_frontend_dev.ps1`

- 신규 추가했다.
- 3000 포트를 점유한 기존 프로세스를 확인하고, `node`, `cmd`, `npm` 계열 dev 서버 프로세스만 종료한다.
- 프론트 dev 서버를 `NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000`으로 다시 실행한다.
- stdout/stderr 로그를 아래 파일에 남긴다.
  - `logs/frontend_dev.out.log`
  - `logs/frontend_dev.err.log`
- dev 서버 시작 뒤 root 200과 CSS asset 200이 될 때까지 polling한다.
- `.next` 삭제는 기본 동작에 넣지 않았다.

### `scripts/start_demo.ps1`

- 백엔드는 기존처럼 8000 포트를 확인한 뒤 없으면 실행한다.
- 프론트는 직접 띄우지 않고 `restart_frontend_dev.ps1`을 거치도록 바꿨다.
- CORS, local snapshot 관련 데모 환경변수는 유지했다.
- 실행 안내와 `npm run build` 후 dev 서버 재시작 필요 문구를 정상 한글로 정리했다.

## 3. 검증 결과

### 빌드

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

- sandbox 안에서는 Next worker spawn이 `EPERM`으로 실패했다.
- 일반 PowerShell 권한으로 재실행해 통과했다.
- 결과: `Compiled successfully`

### build 직후 static mismatch 감지

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

- root: `OK frontend - 200`
- CSS: `FAIL frontend-static - stylesheet 404 ... /_next/static/css/app/layout.css?...`
- 즉, 기존 반복 버그를 readiness에서 잡는 것을 확인했다.

### 프론트 재시작 후 복구

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restart_frontend_dev.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

- `OK frontend - 200`
- `OK frontend-static - stylesheet 200`
- `OK health`
- `OK cors`
- `OK 1D prices`
- `OK indicators`
- `OK 1M prices`
- `OK stock search`
- LM/BM prediction, evaluation, history, band width 모두 OK

### start_demo 경로

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

- `start_demo.ps1` 실행 후에도 `frontend-static`이 OK로 유지되는 것을 확인했다.
- 프론트 dev 로그는 `logs/frontend_dev.out.log`, 에러 로그는 `logs/frontend_dev.err.log`에 기록된다.

### 브라우저 확인

- 인앱 브라우저에서 `http://127.0.0.1:3000` 새로고침 확인.
- 주식 보기 화면이 정상 앱 구조로 로드됐다.
- console error/warn: 0건.

## 4. 현재 서버 상태

- 백엔드: `http://127.0.0.1:8000`
- 프론트: `http://127.0.0.1:3000`
- readiness 기준:
  - backend health OK
  - frontend root OK
  - frontend static CSS OK
  - AAPL 1D 가격 OK
  - AAPL 1M 가격 OK
  - stock search OK

## 5. 운영 메모

- `npm run build` 후 기존 `next dev` 서버를 계속 쓰면 static asset mismatch가 재발할 수 있다.
- build 이후 브라우저 검증 전에는 아래 중 하나를 실행해야 한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restart_frontend_dev.ps1
```

또는

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

- `.next` 삭제는 기본 복구 절차가 아니다.
- 그래도 복구되지 않을 때만 수동 옵션으로 검토한다.

## 6. 남은 리스크

- Codex sandbox 안에서는 Next.js build/dev worker spawn이 `EPERM`으로 실패할 수 있다.
- 그래서 `npm run build`, `restart_frontend_dev.ps1`, `start_demo.ps1`은 실제 데모 검증 시 일반 PowerShell 권한으로 실행하는 것이 안정적이다.
- CP111 백테스트 UI 반영 검증은 이 static guard를 통과한 뒤 이어서 확인해야 한다.
