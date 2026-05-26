"use client";

import { ReactNode } from "react";

const readingCases = [
  {
    title: "예측선이 위쪽이고 밴드가 좁을 때",
    body: "방향과 흔들림이 비교적 안정적으로 맞아떨어진 구간입니다. 바로 매수 신호로 보지는 않고, 가격 흐름과 다른 보조지표를 함께 확인합니다.",
  },
  {
    title: "예측선은 괜찮지만 밴드가 넓을 때",
    body: "방향은 나쁘지 않아도 예상 변동 폭이 큰 구간입니다. 진입 가격, 손절 기준, 포지션 크기를 더 엄격하게 보아야 합니다.",
  },
  {
    title: "예측선이 약하고 밴드 하단이 낮을 때",
    body: "모델이 하방 위험을 크게 보는 상태입니다. 신규 진입보다 관망, 리스크 관리, 기존 포지션 점검을 먼저 생각하는 해석이 자연스럽습니다.",
  },
];

const modelCards = [
  {
    name: "Line v2",
    role: "보수적 기준선",
    body: "5거래일 후 도착가를 보수적으로 추정하는 제품 line 모델입니다. F4 β=4 ensemble 기준이며 수익률 단위 score를 가격으로 환산해 차트에 표시합니다.",
  },
  {
    name: "TiDE",
    role: "AI 밴드",
    body: "가격과 변동성 흐름을 보고 예상 변동 범위를 계산하는 밴드 모델입니다. 1D와 1W 제품 밴드 슬롯에서 사용합니다.",
  },
  {
    name: "PatchTST / CNN-LSTM",
    role: "비교 실험",
    body: "이전 실험에서 비교한 모델입니다. 현재 제품 기본 화면에는 직접 쓰지 않고, 역할과 성능을 비교하는 참고 후보로 남깁니다.",
  },
];

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
          <h1>예측선은 방향을 보고, 밴드는 흔들림을 봅니다.</h1>
          <p>
            Lens는 가격 차트 위에 두 종류의 AI 보조지표를 올려 봅니다. 하나는 5거래일 후 도착가를 보수적으로 추정하고,
            다른 하나는 가격이 흔들릴 수 있는 범위를 보여줍니다.
          </p>
        </div>
        <div className="report-hero__note">
          <strong>먼저 이렇게 보면 됩니다</strong>
          <span>
            보수적 기준선이 좋아도 밴드가 넓으면 조심합니다. 기준선이 약하고 밴드 하단이 낮으면 위험을 먼저 봅니다.
          </span>
        </div>
      </header>

      <section className="report-guide-grid" aria-label="주요 AI 지표 설명">
        <article className="report-guide-card report-guide-card--line">
          <span>보수적 기준선</span>
          <h2>5거래일 후 도착가의 보수적 추정</h2>
          <p>
            이 선은 5거래일 후 도착가의 보수적 추정값입니다. 출력은 수익률 단위 score이고, 화면에서는
            asof_date 종가 × (1 + h5 보수 수익률 추정)으로 가격 환산해 표시합니다.
          </p>
          <ul>
            <li>학습 방식: asymmetric MSE 계열, F4 β=4 ensemble</li>
            <li>보수의 의미: 평균보다 약간 아래로 기울어진 예측을 의도합니다.</li>
            <li>해석: 가격이 이 선보다 위면 정상 영역, 아래로 내려가면 stop-loss 참고 신호로 봅니다.</li>
            <li>주의: 모델 정확도는 100%가 아니며, 투자 판단의 보조지표로만 사용합니다.</li>
          </ul>
        </article>

        <article className="report-guide-card report-guide-card--band">
          <span>AI 밴드</span>
          <h2>예상 변동 범위를 보는 리스크 지표</h2>
          <p>
            AI 밴드는 모델이 보는 예상 변동 범위입니다. 평균 수익률보다 불확실성과 하방 위험 폭을 먼저 봅니다.
            밴드가 넓어지는 구간은 모델이 더 큰 변동 가능성을 보는 구간입니다.
          </p>
          <ul>
            <li>밴드가 좁으면 모델이 보는 불확실성이 낮은 편입니다.</li>
            <li>밴드가 넓으면 방향보다 변동성을 먼저 확인합니다.</li>
            <li>하단 밴드가 깊으면 손실 구간을 더 보수적으로 봅니다.</li>
          </ul>
        </article>
      </section>

      <ReportSection
        title="delay-aligned 표시 방식"
        lead="차트 위 선은 모델이 만든 과거 예측과 최신 예측을 시간축에 맞춰 보여줍니다."
      >
        <div className="report-callout">
          차트의 오늘 날짜 위 line 점은 4거래일 전에 만든 h5 예측 중 내일에 해당하는 부분입니다. 매일 새 예측이
          들어오면 차트 끝에서 5거래일 forward로 선이 이어집니다. 이 방식은 가격을 새로 만드는 것이 아니라,
          저장된 수익률 score를 기준 종가로 환산해 시간축에 맞춰 보여주는 표시 방식입니다.
        </div>
      </ReportSection>

      <ReportSection
        title="두 지표를 같이 읽는 법"
        lead="둘 중 하나만 보는 것보다 방향과 흔들림을 나누어 보는 쪽이 실제 판단에 더 가깝습니다."
      >
        <div className="report-reading-grid">
          {readingCases.map((item) => (
            <article key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </ReportSection>

      <ReportSection
        title="왜 하나로 합치지 않나요?"
        lead="방향을 보는 지표와 흔들림을 보는 지표는 서로 다른 질문에 답합니다."
      >
        <div className="report-callout">
          Lens에서는 예측선과 밴드를 억지로 하나의 점수로 합치지 않습니다. 예측선은 방향이 괜찮은지를 보고,
          밴드는 그 과정에서 얼마나 흔들릴 수 있는지를 봅니다. RSI와 MACD를 따로 읽듯이, 두 AI 지표도 따로 켜고
          끄며 해석하는 방식이 더 자연스럽습니다.
        </div>
      </ReportSection>

      <ReportSection
        title="어떤 모델을 쓰나요?"
        lead="모델 이름보다 중요한 것은 각 모델이 화면에서 맡는 역할입니다."
      >
        <div className="report-model-list">
          {modelCards.map((item) => (
            <article key={item.name}>
              <span className="report-model-role">{item.role}</span>
              <h3>{item.name}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </ReportSection>

      <ReportSection
        title="데이터는 어떻게 보나요?"
        lead="사용자가 해석할 때 꼭 필요한 데이터 계약만 짧게 남깁니다."
      >
        <div className="report-mini-grid">
          <article>
            <h3>조정 가격 기준</h3>
            <p>
              주식 분할이나 배당 때문에 과거 가격이 왜곡되지 않도록 조정 가격 기준으로 보조지표와 모델 입력을 맞춥니다.
            </p>
          </article>
          <article>
            <h3>현재는 1D 중심</h3>
            <p>
              일간 화면에는 보수적 기준선과 AI 밴드 후보가 연결되어 있습니다. 1W는 AI 밴드만 활성이고, 1W 보수적 기준선은
              아직 비워둡니다.
            </p>
          </article>
        </div>
      </ReportSection>

      <details className="report-details">
        <summary>조금 더 기술적인 설명</summary>
        <p>
          보수적 기준선은 가격 단위가 아니라 수익률 단위 score입니다. 화면에서는 기준 종가를 곱해 가격 차트 위에 올립니다.
          이 방식은 multi-modal collapse 위험과 종목별 score 분포 차이가 있어 v2에서 개선 예정입니다.
        </p>
      </details>
    </div>
  );
}
