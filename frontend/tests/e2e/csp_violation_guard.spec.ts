import { expect, test } from "@playwright/test";

/**
 * CP240 — Content-Security-Policy violation guard.
 *
 * 선행 조건:
 *   powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
 *   (backend 127.0.0.1:8000 + frontend 127.0.0.1:3000 ready)
 *
 * 검증:
 * - 메인 화면 + 4 view 로드 시 CSP violation console error 0
 * - script-src / connect-src / img-src 등 directive 가 lightweight-charts /
 *   Next.js hydration / Microsoft Clarity / backend 호출을 차단 안 함
 *
 * 회귀 차단:
 * - CSP 직접 좁히기 (예: 'unsafe-inline' 제거) → Next.js hydration 깨짐 → console
 *   에 "Refused to execute inline script" 출력 → 본 테스트 RED
 * - Clarity 도메인 누락 → console 에 "Refused to load the script ... clarity.ms"
 *   → RED
 */

const VIEWS = ["report", "stocks", "training", "backtests"] as const;

for (const view of VIEWS) {
  test(`CSP violation 0 on view=${view}`, async ({ page }) => {
    const cspErrors: string[] = [];
    page.on("console", (msg) => {
      const text = msg.text();
      if (
        msg.type() === "error" &&
        (text.includes("Content Security Policy") ||
          text.includes("Refused to") ||
          text.includes("violates"))
      ) {
        cspErrors.push(text);
      }
    });
    page.on("pageerror", (err) => {
      const text = err.message;
      if (text.includes("Content Security Policy") || text.includes("CSP")) {
        cspErrors.push(text);
      }
    });

    await page.goto(`/?view=${view}`, { waitUntil: "networkidle" });
    // Clarity init / hydration 마무리 대기.
    await page.waitForTimeout(2000);

    expect(
      cspErrors,
      `CSP violation 감지 (view=${view}):\n${cspErrors.join("\n")}`,
    ).toEqual([]);
  });
}
