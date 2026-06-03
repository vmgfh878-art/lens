# ADR 0020 — Frontend Characterization: Playwright + Vitest

- **상태**: 채택 (2026-06-03, CP230 Step 1~5)
- **컨텍스트**: CP231~234 프론트 분리(거대 뷰 BacktestView 1643 / TrainingView 1560 / StockView 1097 + globals.css 5208)를 시작하려면 분리 전후 화면과 순수 함수 동작이 보존됨을 기계적으로 증명할 안전망 필요. CP222 baseline 직후 frontend 도구는 0이었음 (test runner / e2e / lint 어느 것도 없음).

## 결정 요지

| 항목 | 선택 | 근거 |
|---|---|---|
| E2E 도구 | **Playwright 1.60.0** + chromium 1종 | Next.js 14와 자연스러운 결합, screenshot baseline 안정성, `toHaveScreenshot({ mask })` 마스킹 1급 지원, 결정론 옵션(`animations:"disabled"`, `timezoneId`, `locale`) 풍부. Cypress보다 worker/headless 안정. |
| 단위 도구 | **Vitest 4.1.8** + `@vitest/coverage-v8` | Next.js 14에서 별도 jest 설정 없이 ESM 그대로. `environment:"node"`(jsdom 미설치)로 순수 함수만 빠르게. |
| 서버 관리 | **webServer 미사용** — 선행 `start_demo.ps1` | `webServer`를 두면 Playwright가 3000을 또 띄워 포트 충돌. start_demo.ps1이 이미 모든 dev에서 사용 중인 경로. |
| 결정론 옵션 | viewport 1280x800, locale ko-KR, TZ Asia/Seoul, animations disabled, maxDiffPixelRatio 0.01 | 머신/시간 영향 차단. ReportView가 1순위 결정성 화면이라 마스킹 없이 baseline. |
| 동적 영역 마스킹 | canvas, `.freshness`, `.asof`, `[data-freshness]`, `[data-asof]`, `.stale-tag`, `.stale-badge` | 차트(`requestAnimationFrame` 마커), staleness 배지, asof 텍스트 비결정 회피. |
| console.error 가드 | pageerror 0 + console.error 0, 단 `/Failed to load resource:.+404/` 화이트리스트 | TrainingView가 mock 미설정 시 `/api/v1/ai/runs` 404 → 브라우저 generic console.error 발생. CP223에서도 ai 라우터 명시 제외됨 → 가드 완화. |
| 단위 테스트 대상 | 현존 5 lib (evaluators / predictionHelpers / formatters / training-formatters / staleness) + CP231~233 forward stub 3개 | 분기 경계값 + NaN/Inf 필터 + Intl 로케일 동작 박제. stub은 `describe.skip + it.todo`로 추출 후 즉시 활성 가능. |
| 베이스라인 위치 | `frontend/tests/e2e/screens.smoke.spec.ts-snapshots/*.png` (git 추적), `__pycache__` / `test-results/` / `playwright-report/` 등은 ignore | baseline diff 가시성과 잔여물 깔끔 분리. |

## 대안과 거부 이유

- **Cypress** — Playwright와 비교해 결정론 옵션이 약하고 chromium 외 안정성 낮음. 거부.
- **jest + @testing-library/react** — DOM 환경 필요(jsdom), 순수 함수에 과함. Vitest로 단순화. 거부.
- **Playwright webServer 사용** — start_demo.ps1과 이중 기동, 포트 충돌. 거부.
- **컴포넌트 단위(React) 렌더 테스트** — 거대 뷰 자체는 분리 후 CP에서. 현재는 순수 함수 + 스크린샷으로 충분. 거부(스코프 외).
- **시간 의존 함수(`dateUtils.buildDefaultPriceWindow`) 단위 테스트** — `vi.setSystemTime` 고정 없이는 매일 깨짐. CP230 범위에서 제외, 보고서 후속에 명시.

## 결과 (CP230 baseline)

| 항목 | 실측 |
|---|---|
| Playwright smoke | **4/4 통과** (report/stocks/backtests/training), pageerror 0 |
| Screenshot baseline | 4 PNG 생성, **2회 연속 재현 diff 0** (각 ~5.2s) |
| Vitest 단위 | **107 passed / 12 todo / 0 failed** (5 files passed + 3 stub skipped, 240ms) |
| `npx tsc --noEmit` | 0 에러 (테스트 파일 포함) |
| 프로덕션 코드 변경 | 0 라인 (`frontend/src/**` 무수정) |

## 후속

- **CP231**: `simulationEngine` 추출 + simulationEngine.test.ts skip 해제.
- **CP232**: `signalBuilder` 추출 + signalBuilder.test.ts skip 해제.
- **CP233**: `experimentBuilder` 추출 + experimentBuilder.test.ts skip 해제.
- **별도**: `dateUtils.buildDefaultPriceWindow`/`buildFullPriceWindows`의 `new Date()` 의존을 `vi.setSystemTime` 고정 또는 주입식 변경(코드 변경 CP).
- **별도**: 로케일/TZ 환경 의존 단정의 강화(예: CI 환경에서 `Intl` 결과 정확 비교).
