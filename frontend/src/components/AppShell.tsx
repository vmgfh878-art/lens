"use client";

import { ReactNode } from "react";

export type AppView = "stocks" | "backtests" | "training";

interface AppShellProps {
  activeView: AppView;
  onViewChange: (view: AppView) => void;
  children: ReactNode;
}

const NAV_ITEMS: Array<{ id: AppView; label: string; eyebrow: string }> = [
  { id: "stocks", label: "주식 보기", eyebrow: "차트" },
  { id: "backtests", label: "백테스트", eyebrow: "결과" },
  { id: "training", label: "모델 학습", eyebrow: "학습 이력" },
];

export default function AppShell({ activeView, onViewChange, children }: AppShellProps) {
  return (
    <main className="app-shell">
      <aside className="side-nav">
        <div className="brand">
          <div className="brand__mark">L</div>
          <div>
            <div className="brand__name">Lens</div>
            <div className="brand__caption">주식 분석</div>
          </div>
        </div>

        <nav className="nav-list" aria-label="주요 화면">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item${activeView === item.id ? " nav-item--active" : ""}`}
              onClick={() => onViewChange(item.id)}
            >
              <span className="nav-item__eyebrow">{item.eyebrow}</span>
              <span className="nav-item__label">{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <section className="workspace">{children}</section>
    </main>
  );
}
