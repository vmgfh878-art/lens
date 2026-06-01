"use client";

import { ReactNode } from "react";

// 지표 가이드 = 프로덕트/사용자 관점. 모델 구조·실험·수치는 AI 모델 화면에서 다룬다.
// 직관 해석은 각 지표의 정의(보수적 asymmetric 추정 / 예측 분위수 구간)에서
// 직접 따라오는 것만 적는다. 조합 해석은 신호가 아니라 참고로 표기.

function ReportSection({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: string;
  children: ReactNode;
}) {
  return (
    <section className="report-section">
      <div className="report-section__heading">
        <h2>{title}</h2>
        {lead ? <p>{lead}</p> : null}
      </div>
      {children}
    </section>
  );
}

export default function ReportView() {
  return (
    <div className="view-stack report-view">
      <header className="view-header report-hero report-hero--guide">
        <div className="view-header__title">
          <div className="eyebrow">AI 지표 가이드</div>
          <h1>예측선은 방향을, 밴드는 흔들림을 봅니다.</h1>
          <p>
            Lens는 가격 차트 위에 두 가지 AI 보조지표를 겹쳐 봅니다. AI가 가격을 맞히는 도구가 아니라,
            딥러닝이 학습한 패턴을 <strong>보조지표</strong>로 보여주는 도구입니다. 최종 판단은 사용자가 합니다.
          </p>
        </div>
      </header>

      {/* 1) 누가 / 언제 / 어떻게 — 가장 중요, 맨 앞. 박스 그리드 */}
      <ReportSection title="누가, 언제, 어떻게">
        <div className="report-usage-grid">
          <article>
            <h3>누가</h3>
            <p>
              미국 주식을 직접 차트로 보는 개인 투자자, 그리고 RSI·MACD 같은 보조지표에 더해 AI 관점을 하나 더 두고
              싶은 사람을 위한 도구입니다. 차트 분석에 관심이 있다면 딥러닝을 몰라도 바로 읽을 수 있게 만들었습니다.
            </p>
          </article>
          <article>
            <h3>언제</h3>
            <p>
              오늘 이 종목을 더 들여다볼지, 관망할지, 위험을 줄일지 결정하기 전에 봅니다. 매일 새로 갱신된 예측선과
              위험 범위를 기준으로, 진입·보유·청산 판단을 빠르게 보조받고 싶을 때 적합합니다.
            </p>
          </article>
          <article>
            <h3>어떻게</h3>
            <p>
              예측선으로 방향을, AI 밴드로 흔들림을 나눠서 봅니다. 기존 RSI·MACD와 함께 읽고, 백테스트 화면에서 규칙
              기반 전략으로 과거 성과까지 확인한 뒤 참고합니다. 단타·실시간 트레이딩, 자동매매의 근거, 정확한 목표가
              예측으로는 적합하지 않으며 투자 자문을 대체하지 않습니다. 신호 하나로 매매하지 말고 늘 다른 근거와 함께
              보조지표로만 사용하세요.
            </p>
          </article>
        </div>
      </ReportSection>

      {/* 2) 두 핵심 지표 — 직관 해석 + 장점(정성). 계획서의 asymmetric loss / quantile 근거 반영 */}
      <section className="report-guide-grid" aria-label="주요 AI 지표 설명">
        <article className="report-guide-card report-guide-card--line">
          <span>보수적 기준선</span>
          <h2>앞으로 5거래일의 보수적 기준</h2>
          <p>
            앞으로 5거래일 동안 가격이 갈 만한 자리를 모델이 평균보다 조금 낮게 잡은 기준선입니다. 과하게 낙관적인
            예측을 억제하도록 학습해서, 일부러 보수적으로 봅니다. 가격이 이 선보다 위면 정상 범위, 아래로 내려가면
            주의해서 봅니다.
          </p>
          <ul>
            <li>큰 하락 구간을 단순 기준선보다 뚜렷이 잘 잡아냅니다.</li>
            <li>낙관 쪽 오차에 더 큰 페널티를 줘서 일부러 보수적으로 학습했습니다.</li>
            <li>매수·매도 명령이 아니라 정상 범위를 보는 참고선입니다.</li>
          </ul>
        </article>

        <article className="report-guide-card report-guide-card--band">
          <span>AI 밴드</span>
          <h2>예상 변동 범위</h2>
          <p>
            앞으로 가격이 흔들릴 수 있는 범위를, 높은 확률로 들어올 구간(예측 분위수)으로 그린 것입니다. 밴드가
            넓을수록 모델이 보는 불확실성이 크고, 가격이 밴드를 벗어나면 예상 범위를 넘어선 움직임입니다.
          </p>
          <ul>
            <li>실제 변동성을 잘 따라갑니다. 흔들린 만큼 밴드 폭이 넓어지고 좁아집니다.</li>
            <li>예상 변동 범위를 목표 포함 비율에 맞춰 보정합니다.</li>
            <li>밴드 폭 자체가 시장 불안정성을 읽는 신호가 됩니다.</li>
          </ul>
        </article>
      </section>

      {/* 4) 표시 기준 */}
      <ReportSection title="표시 기준">
        <div className="report-mini-grid">
          <article>
            <h3>매일 갱신되는 예측선</h3>
            <p>예측선은 매일 새로 계산되어 차트 끝에서 앞으로 이어집니다. 오늘 화면의 미래 구간은 가장 최근 예측 기준입니다.</p>
          </article>
          <article>
            <h3>현재는 일간(1D) 중심</h3>
            <p>일간 화면에 기준선과 밴드가 함께 있습니다. 주간(1W)은 밴드만 제공하고, 기준선은 준비 중입니다.</p>
          </article>
        </div>
      </ReportSection>

      {/* 5) 면책 */}
      <div className="report-callout report-callout--disclaimer">
        Lens는 학술·연구 목적의 AI 보조지표 시각화 도구입니다. 투자 자문이나 매매 권유가 아니며, 모든 투자 판단과 책임은
        사용자에게 있습니다.
      </div>
    </div>
  );
}
