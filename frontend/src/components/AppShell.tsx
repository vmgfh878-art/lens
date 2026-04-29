"use client";

import { ReactNode, useState } from "react";

export type AppView = "stocks" | "backtests" | "training";

interface AppShellProps {
  activeView: AppView;
  onViewChange: (view: AppView) => void;
  children: ReactNode;
}

interface NavItem {
  id: AppView | string;
  label: string;
  icon: IconName;
  badge?: string;
  disabled?: boolean;
}

interface NavSection {
  label: string;
  items: NavItem[];
}

type IconName =
  | "chart"
  | "star"
  | "beaker"
  | "cpu"
  | "database"
  | "doc"
  | "search"
  | "bell"
  | "settings"
  | "chevron"
  | "panel-left";

const NAV_SECTIONS: NavSection[] = [
  {
    label: "분석",
    items: [
      { id: "stocks", label: "주식 보기", icon: "chart" },
      { id: "watchlist", label: "관심 종목", icon: "star", badge: "12", disabled: true },
    ],
  },
  {
    label: "리서치",
    items: [
      { id: "backtests", label: "백테스트", icon: "beaker" },
      { id: "training", label: "모델 학습", icon: "cpu" },
    ],
  },
  {
    label: "라이브러리",
    items: [
      { id: "datasets", label: "데이터셋", icon: "database", disabled: true },
      { id: "reports", label: "리포트", icon: "doc", disabled: true },
    ],
  },
];

const ACTIVE_VIEWS = new Set<string>(["stocks", "backtests", "training"]);

function Icon({ name }: { name: IconName }) {
  const props = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  switch (name) {
    case "chart":
      return (
        <svg {...props}>
          <path d="M4 19V5" />
          <path d="M4 19h16" />
          <path d="M8 15l3-4 3 2 5-7" />
        </svg>
      );
    case "star":
      return (
        <svg {...props}>
          <path d="M12 4l2.5 5.2 5.5.8-4 4 1 5.5L12 17l-5 2.5 1-5.5-4-4 5.5-.8z" />
        </svg>
      );
    case "beaker":
      return (
        <svg {...props}>
          <path d="M9 4h6" />
          <path d="M10 4v6L5 19a2 2 0 002 3h10a2 2 0 002-3l-5-9V4" />
          <path d="M7 14h10" />
        </svg>
      );
    case "cpu":
      return (
        <svg {...props}>
          <rect x="6" y="6" width="12" height="12" rx="2" />
          <rect x="9" y="9" width="6" height="6" />
          <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
        </svg>
      );
    case "database":
      return (
        <svg {...props}>
          <ellipse cx="12" cy="6" rx="8" ry="3" />
          <path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
          <path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
        </svg>
      );
    case "doc":
      return (
        <svg {...props}>
          <path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z" />
          <path d="M14 3v5h5" />
          <path d="M9 13h6M9 17h4" />
        </svg>
      );
    case "search":
      return (
        <svg {...props}>
          <circle cx="11" cy="11" r="6" />
          <path d="M20 20l-4-4" />
        </svg>
      );
    case "bell":
      return (
        <svg {...props}>
          <path d="M6 8a6 6 0 0112 0v5l1.5 2h-15L6 13z" />
          <path d="M10 19a2 2 0 004 0" />
        </svg>
      );
    case "settings":
      return (
        <svg {...props}>
          <circle cx="12" cy="12" r="3" />
          <path d="M19 12a7 7 0 00-.1-1.2l2-1.5-2-3.4-2.3.9a7 7 0 00-2-1.2L14 3h-4l-.6 2.6a7 7 0 00-2 1.2l-2.3-.9-2 3.4 2 1.5A7 7 0 005 12c0 .4 0 .8.1 1.2l-2 1.5 2 3.4 2.3-.9c.6.5 1.3.9 2 1.2L10 21h4l.6-2.6c.7-.3 1.4-.7 2-1.2l2.3.9 2-3.4-2-1.5c.1-.4.1-.8.1-1.2z" />
        </svg>
      );
    case "chevron":
      return (
        <svg {...props}>
          <path d="M9 6l6 6-6 6" />
        </svg>
      );
    case "panel-left":
      return (
        <svg {...props}>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M9 4v16" />
        </svg>
      );
    default:
      return null;
  }
}

export default function AppShell({ activeView, onViewChange, children }: AppShellProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [search, setSearch] = useState("");

  const filteredSections = NAV_SECTIONS.map((section) => ({
    ...section,
    items: section.items.filter((item) =>
      item.label.toLowerCase().includes(search.toLowerCase()),
    ),
  })).filter((s) => s.items.length > 0);

  const activeLabel =
    NAV_SECTIONS.flatMap((s) => s.items).find((i) => i.id === activeView)?.label ?? "Lens";

  return (
    <main className={`app-shell${collapsed ? " app-shell--collapsed" : ""}`}>
      <aside className="side-nav">
        <div className="side-nav__brand">
          <div className="brand__mark" aria-hidden="true" />
          <div className="brand__text">
            <div className="brand__name">Lens</div>
            <div className="brand__caption">주식 분석</div>
          </div>
        </div>

        <div className="side-search">
          <span className="side-search__icon" aria-hidden="true">
            <Icon name="search" />
          </span>
          <input
            placeholder="검색..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="메뉴 검색"
          />
          <span className="side-search__kbd">⌘K</span>
        </div>

        <nav className="nav-scroll" aria-label="주요 화면">
          {filteredSections.map((section) => (
            <div className="nav-section" key={section.label}>
              <div className="nav-section__label">{section.label}</div>
              <div className="nav-list">
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    disabled={item.disabled && !ACTIVE_VIEWS.has(item.id)}
                    className={`nav-item${activeView === item.id ? " nav-item--active" : ""}`}
                    onClick={() => {
                      if (ACTIVE_VIEWS.has(item.id)) {
                        onViewChange(item.id as AppView);
                      }
                    }}
                    title={collapsed ? item.label : undefined}
                  >
                    <span className="nav-item__icon">
                      <Icon name={item.icon} />
                    </span>
                    <span className="nav-item__label">{item.label}</span>
                    {item.badge ? (
                      <span className="nav-item__badge">{item.badge}</span>
                    ) : null}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="nav-footer">
          <div className="nav-user" title="계정">
            <div className="nav-user__avatar">VM</div>
            <div className="nav-user__meta">
              <span className="nav-user__name">vmgfh878</span>
              <span className="nav-user__email">research@lens.io</span>
            </div>
          </div>
          <button
            className="nav-collapse"
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "사이드바 펼치기" : "사이드바 접기"}
          >
            <Icon name="panel-left" />
            <span>{collapsed ? "펼치기" : "접기"}</span>
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar__crumbs">
            <span>Lens</span>
            <span className="topbar__crumb-sep" aria-hidden="true">
              <Icon name="chevron" />
            </span>
            <strong>{activeLabel}</strong>
          </div>
          <div className="topbar__actions">
            <button className="icon-btn" type="button" title="알림">
              <Icon name="bell" />
              <span className="icon-btn__dot" aria-hidden="true" />
            </button>
            <button className="icon-btn" type="button" title="설정">
              <Icon name="settings" />
            </button>
          </div>
        </header>
        <div className="workspace__body">{children}</div>
      </section>
    </main>
  );
}
