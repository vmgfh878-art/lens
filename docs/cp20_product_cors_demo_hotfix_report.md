# CP20-P CORS 데모 핫픽스 보고서

## 1. 원인

이번 문제는 백엔드 다운이 아니었다. 백엔드는 `127.0.0.1:8000`에서 정상 응답하고 있었지만, 브라우저 origin은 `http://127.0.0.1:3000`이고 백엔드 CORS 기본 허용값은 `http://localhost:3000`뿐이었다.

PowerShell readiness는 CORS 적용 대상이 아니라 직접 호출로 200을 받았고, 브라우저 JS만 CORS 정책 때문에 응답을 사용할 수 없었다. 그래서 화면에는 실제 백엔드 장애처럼 `백엔드에 연결할 수 없습니다`가 표시됐다.

## 2. 변경 파일

- `backend/app/main.py`
- `scripts/start_demo.ps1`
- `scripts/check_demo_readiness.ps1`

## 3. 변경 내용

`backend/app/main.py`

- `_parse_cors_origins()` 기본값을 아래처럼 변경했다.

```text
http://localhost:3000,http://127.0.0.1:3000
```

- `BACKEND_CORS_ORIGINS` 환경변수가 있으면 기존처럼 환경변수 값을 우선 사용한다.

`scripts/start_demo.ps1`

- 백엔드 runner에 아래 환경변수를 추가했다.

```powershell
$env:BACKEND_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
```

- 프론트는 기존처럼 `NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:8000"`을 유지한다.

`scripts/check_demo_readiness.ps1`

- `Origin: http://127.0.0.1:3000` 헤더를 붙여 `/api/v1/health/live`를 호출한다.
- 응답 헤더 `Access-Control-Allow-Origin`이 `http://127.0.0.1:3000`인지 확인한다.
- 없거나 다르면 `FAIL cors`로 표시한다.

## 4. CORS 검증 결과

수정 전, 기존 백엔드 프로세스에서는 다음처럼 실패했다.

```text
OK   health - live 200
FAIL cors - Access-Control-Allow-Origin=
```

백엔드를 새 CORS 설정으로 재시작한 뒤 결과:

```text
OK   health - live 200
OK   cors - Access-Control-Allow-Origin=http://127.0.0.1:3000
OK   frontend - 200
OK   1D prices - AAPL 5 rows
OK   1M prices - price-only view available, 5 rows
FAIL stock search - 503 UPSTREAM_UNAVAILABLE
OK   demo run - patchtst-1D-fc096a026a1e
OK   prediction - forecast=5 line=5 upper=5 lower=5
OK   evaluation - 1 rows
OK   backtest - 1 rows
```

`stock search` 503은 그대로 남아 있다. 이는 이번 CORS 문제와 별개이며, 화면에서는 작은 안내로만 표시된다.

## 5. 브라우저 확인 결과

브라우저에서 `http://127.0.0.1:3000`을 새로고침해 확인했다.

- `백엔드에 연결할 수 없습니다` 문구: 0건
- `가격 데이터가 없습니다.` 문구: 0건
- 차트 canvas: 7개 렌더링
- `AI 연결` 문구 표시
- `저장된 AI 예측 연결됨` 문구 표시
- `티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 가격 조회는 계속 가능합니다.`는 작은 안내로 표시
- 브라우저 console error: 없음

기대 상태인 가격 차트, AI 밴드/보수적 예측선 표시 가능 상태로 돌아왔다.

## 6. 빌드 결과

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 실패했다. 승인된 실행에서는 정상 통과했다.

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
```

## 7. 남은 이슈

- `stock_info` 기반 티커 검색 API는 여전히 503을 반환한다.
- 이 문제는 백엔드 연결 실패가 아니며, 가격 API와 prediction API는 정상이다.
- 주식 보기 화면은 검색 실패를 전체 화면 오류로 표시하지 않고 직접 입력 fallback을 유지한다.

## 8. 결론

이 이슈는 백엔드 다운이 아니라 `localhost`와 `127.0.0.1` origin 불일치로 생긴 CORS 버그였다. `127.0.0.1:3000`을 CORS 허용 origin에 추가했고, readiness와 브라우저 기준으로 정상 동작을 확인했다.
