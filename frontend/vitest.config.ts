import path from "node:path";
import { defineConfig } from "vitest/config";

/**
 * CP230 Vitest 설정 — 순수 함수 단위 테스트만.
 *
 * - environment: "node" (DOM/jsdom 미사용. 순수 함수 대상이라 불필요).
 * - include: src/**\/*.test.ts (Playwright의 tests/e2e/** 제외).
 * - alias @ -> src (tsconfig paths와 일치).
 * - globals false: it/describe/expect 등 명시 import 강제.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    globals: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
