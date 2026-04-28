"use client";

import { useState } from "react";

import AppShell, { AppView } from "@/components/AppShell";
import BacktestView from "@/components/BacktestView";
import StockView from "@/components/StockView";
import TrainingView from "@/components/TrainingView";

export default function DashboardPage() {
  const [activeView, setActiveView] = useState<AppView>("stocks");

  return (
    <AppShell activeView={activeView} onViewChange={setActiveView}>
      {activeView === "stocks" ? <StockView /> : null}
      {activeView === "backtests" ? <BacktestView /> : null}
      {activeView === "training" ? <TrainingView /> : null}
    </AppShell>
  );
}
