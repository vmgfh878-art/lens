import { defineConfig, devices } from "@playwright/test";

/**
 * CP230 Playwright 설정.
 *
 * 결정론:
 * - chromium 1종, viewport 1280x800, locale ko-KR, TZ Asia/Seoul 고정.
 * - 애니메이션 비활성, screenshot diff 허용 ≤1%.
 * - workers 1, retries 0 (재현성·디버그 용이).
 *
 * 서버 관리:
 * - webServer 미사용. 선행 조건으로 scripts/start_demo.ps1 가 백엔드 8000 +
 *   프론트 3000을 띄워 둔 상태에서 실행한다 (이중 기동 방지).
 * - baseURL = http://127.0.0.1:3000.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.01,
      animations: "disabled",
    },
  },
  use: {
    baseURL: "http://127.0.0.1:3000",
    viewport: { width: 1280, height: 800 },
    locale: "ko-KR",
    timezoneId: "Asia/Seoul",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
