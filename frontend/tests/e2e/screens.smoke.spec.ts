import { expect, test } from "@playwright/test";

/**
 * CP230 4화면 load smoke + screenshot baseline.
 *
 * 선행 조건:
 *   powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
 *   (백엔드 127.0.0.1:8000 + 프론트 127.0.0.1:3000 ready 상태에서 실행.)
 *
 * 결정론:
 * - playwright.config.ts: viewport 1280x800, locale ko-KR, TZ Asia/Seoul,
 *   animations disabled, maxDiffPixelRatio 0.01.
 * - 데이터 fetch가 있는 3화면은 networkidle 대기 + 동적 영역 마스킹
 *   (canvas, freshness/asof 배지).
 * - ReportView는 fetch 0 + 시간 의존 0 → 마스킹 없이 baseline.
 */
type Case = {
  id: string;
  path: string;
  // Playwright getByRole / getByText / locator 셀렉터로 변환되는 정보.
  headerHint:
    | { kind: "h1Text"; text: string }
    | { kind: "selector"; selector: string };
  // ReportView 외의 화면은 데이터 fetch 후 정착 대기 + 동적 영역 마스킹 필요.
  hasFetch: boolean;
};

const CASES: Case[] = [
  {
    id: "report",
    path: "/?view=report",
    headerHint: {
      kind: "h1Text",
      text: "예측선은 방향을, 밴드는 흔들림을 봅니다.",
    },
    hasFetch: false,
  },
  {
    id: "stocks",
    path: "/?view=stocks",
    headerHint: { kind: "selector", selector: ".stock-intro" },
    hasFetch: true,
  },
  {
    id: "backtests",
    path: "/?view=backtests",
    headerHint: { kind: "h1Text", text: "전략 신호" },
    hasFetch: true,
  },
  {
    id: "training",
    path: "/?view=training",
    headerHint: { kind: "h1Text", text: "AI 모델" },
    hasFetch: true,
  },
];

// 동적 영역 마스킹 셀렉터. canvas(차트) + freshness/asof 배지.
const DYNAMIC_MASKS = [
  "canvas",
  ".freshness",
  ".asof",
  "[data-freshness]",
  "[data-asof]",
  // staleness/시간 표시 잠재 영역.
  ".stale-tag",
  ".stale-badge",
];

for (const c of CASES) {
  test(`${c.id} loads + baseline screenshot`, async ({ page }) => {
    const pageErrors: Error[] = [];
    const consoleErrors: string[] = [];

    // 데모 환경에서 발생하는 정상 fetch 실패는 무시 (운영 코드 무수정 원칙).
    // /api/v1/ai/runs는 mock 데이터 미설정 시 404를 의도적으로 반환 — TrainingView가
    // StatusInline 으로 사용자 가시 에러 표시. ai 라우터는 CP223에서도 명시 제외됨.
    const ignoredConsoleErrorPatterns: RegExp[] = [
      /Failed to load resource:.+\b404\b/,
    ];

    page.on("pageerror", (err) => {
      pageErrors.push(err);
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        const text = msg.text();
        if (!ignoredConsoleErrorPatterns.some((p) => p.test(text))) {
          consoleErrors.push(text);
        }
      }
    });

    await page.goto(c.path);

    const headerLocator =
      c.headerHint.kind === "h1Text"
        ? page.getByRole("heading", { level: 1, name: c.headerHint.text })
        : page.locator(c.headerHint.selector);
    await expect(headerLocator.first()).toBeVisible({ timeout: 15000 });

    if (c.hasFetch) {
      await page.waitForLoadState("networkidle", { timeout: 20000 });
    }

    // 안정성 가드: pageerror 0, console.error 0.
    expect(pageErrors, `[${c.id}] pageerror: ${pageErrors.join(", ")}`).toEqual(
      [],
    );
    expect(
      consoleErrors,
      `[${c.id}] console.error: ${consoleErrors.join(" | ")}`,
    ).toEqual([]);

    // Screenshot baseline.
    if (c.hasFetch) {
      const masks = DYNAMIC_MASKS.map((sel) => page.locator(sel));
      await expect(page).toHaveScreenshot(`${c.id}.png`, {
        mask: masks,
        animations: "disabled",
      });
    } else {
      await expect(page).toHaveScreenshot(`${c.id}.png`);
    }
  });
}
