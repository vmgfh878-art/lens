import type { Metadata } from "next";
import type { ReactNode } from "react";

import ClarityInit from "./ClarityInit";
import "./styles/base.css";
import "./styles/shell.css";
import "./styles/stock.css";
import "./styles/backtest.css";
import "./styles/training.css";
import "./styles/components.css";
import "./styles/report-model-role-card.css";
import "./styles/report-archive.css";
import "./styles/report-view.css";
import "./styles/significance.css";
import "./styles/reproducibility.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lens",
  description: "AI 보조지표 기반 주식 분석 대시보드",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>
        <ClarityInit />
        {children}
      </body>
    </html>
  );
}
