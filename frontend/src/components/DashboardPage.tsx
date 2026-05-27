"use client";

import { useEffect, useState } from "react";

import AppShell, { AppView } from "@/components/AppShell";
import BacktestView from "@/components/BacktestView";
import ReportView from "@/components/ReportView";
import StockView from "@/components/StockView";
import TrainingView from "@/components/TrainingView";

function isAppView(value: string | null): value is AppView {
  return value === "stocks" || value === "backtests" || value === "training" || value === "report";
}

export default function DashboardPage() {
  const [activeView, setActiveView] = useState<AppView>("stocks");

  useEffect(() => {
    const nextView = new URLSearchParams(window.location.search).get("view");
    if (isAppView(nextView)) {
      setActiveView(nextView);
    }
  }, []);

  function handleViewChange(nextView: AppView) {
    setActiveView(nextView);
    const url = new URL(window.location.href);
    url.searchParams.set("view", nextView);
    window.history.replaceState(null, "", url);
  }

  return (
    <AppShell activeView={activeView} onViewChange={handleViewChange}>
      {activeView === "stocks" ? <StockView /> : null}
      {activeView === "backtests" ? <BacktestView /> : null}
      {activeView === "training" ? <TrainingView /> : null}
      {activeView === "report" ? <ReportView /> : null}
    </AppShell>
  );
}
