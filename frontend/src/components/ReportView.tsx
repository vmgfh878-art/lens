"use client";

import { ReactNode } from "react";

const readingCases = [
  {
    title: "예측선이 위쪽이고 밴드가 좁을 때",
    body: "모델이 보는 방향과 흔들림이 비교적 안정적으로 맞아떨어지는 구간입니다. 그래도 매수 신호가 아니라, 다른 가격 흐름과 함께 확인할 만한 상태로 봅니다.",
  },
  {
    title: "예측선은 괜찮지만 밴드가 넓을 때",
    body: "방향은 나쁘지 않아도 예상 흔들림이 큽니다. 수익 가능성만 보지 말고 손절 기준이나 진입 가격을 더 엄격하게 잡아야 하는 구간입니다.",
  },
  {
    title: "예측선이 약하고 밴드 하단이 낮을 때",
    body: "모델이 하방 여지를 크게 보고 있는 상태입니다. 새 진입보다는 포지션 축소, 관망, 리스크 점검을 먼저 생각하는 해석이 자연스럽습니다.",
  },
];

const modelCards = [
  {
    name: "PatchTST",
    role: "보수적 예측선",
    body: "가격과 보조지표의 흐름을 일정한 조각으로 나눠 읽는 시계열 모델입니다. Lens에서는 앞으로의 방향을 보는 선에 사용합니다.",
  },
  {
    name: "CNN-LSTM",
    role: "AI 밴드",
    body: "짧은 구간의 모양과 시간 순서를 함께 보는 모델입니다. Lens에서는 앞으로 흔들릴 수 있는 범위를 그리는 밴드에 사용합니다.",
  },
  {
    name: "TiDE",
    role: "비교 실험",
    body: "같은 문제를 다른 구조로 풀어보기 위해 실험한 모델입니다. 현재 기본 화면에는 쓰지 않지만, 비교 후보로 남겨두었습니다.",
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
            Lens는 가격 차트 위에 두 개의 AI 보조지표를 올려 봅니다. 하나는 앞으로의 방향을 조심스럽게
            보고, 다른 하나는 가격이 얼마나 흔들릴 수 있는지 보여줍니다.
          </p>
        </div>
        <div className="report-hero__note">
          <strong>먼저 이렇게 보면 됩니다</strong>
          <span>
            예측선이 좋아도 밴드가 넓으면 조심합니다. 예측선이 약하고 밴드 하단이 낮으면 위험을 먼저
            봅니다.
          </span>
        </div>
      </header>

      <section className="report-guide-grid" aria-label="주요 AI 지표 설명">
        <article className="report-guide-card report-guide-card--line">
          <span>보수적 예측선</span>
          <h2>앞으로 유리한 방향인지 보는 선</h2>
          <p>
            초록색 선은 앞으로의 수익 방향을 참고하기 위한 AI 예측선입니다. 모델이 너무 낙관적으로
            말하지 않도록, 실제보다 좋게 예측하는 실수에 더 큰 벌점을 주고 학습했습니다.
          </p>
          <ul>
            <li>선이 위쪽이면 방향은 긍정적으로 봅니다.</li>
            <li>선이 약하거나 아래쪽이면 진입을 서두르지 않습니다.</li>
            <li>단독 매매 신호가 아니라 가격 흐름과 같이 봅니다.</li>
          </ul>
        </article>

        <article className="report-guide-card report-guide-card--band">
          <span>AI 밴드</span>
          <h2>가격이 흔들릴 수 있는 범위</h2>
          <p>
            파란색 밴드는 모델이 보는 예상 변동 범위입니다. 평균적으로 오를지보다, 앞으로 흔들림이
            얼마나 커질 수 있는지와 하단 위험이 어디까지 열려 있는지를 봅니다.
          </p>
          <ul>
            <li>밴드가 좁으면 모델이 보는 불확실성이 작습니다.</li>
            <li>밴드가 넓으면 방향보다 변동성을 먼저 봅니다.</li>
            <li>밴드 하단이 낮으면 손실 구간을 더 보수적으로 봅니다.</li>
          </ul>
        </article>
      </section>

      <ReportSection
        title="두 지표를 같이 읽는 법"
        lead="둘 중 하나만 보는 것보다, 방향과 흔들림을 나눠 보는 쪽이 실제 판단에 더 가깝습니다."
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
          Lens에서는 예측선과 밴드를 억지로 하나의 점수로 합치지 않습니다. 예측선은 “방향이 괜찮은가”를
          보고, 밴드는 “그 과정에서 얼마나 흔들릴 수 있는가”를 봅니다. RSI와 MACD를 따로 읽듯이, 두 AI
          지표도 따로 켜고 끄며 해석하는 방식이 더 자연스럽습니다.
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
              주식 분할이나 배당 때문에 과거 가격이 왜곡되지 않도록 조정 가격 기준으로 보조지표와 모델
              입력을 맞춥니다.
            </p>
          </article>
          <article>
            <h3>현재는 1D 중심</h3>
            <p>
              일간 화면에는 보수적 예측선과 AI 밴드 후보가 연결되어 있습니다. 주간과 월간 모델은 별도
              데이터 기준을 확인한 뒤 붙입니다.
            </p>
          </article>
        </div>
      </ReportSection>

      <details className="report-details">
        <summary>조금 더 기술적인 설명</summary>
        <p>
          대량 학습 데이터는 로컬 파일 기준으로 관리하고, 화면에 필요한 최신 예측 결과만 가볍게
          저장합니다. 이 구조는 비용을 줄이기 위한 선택이기도 하지만, 같은 데이터로 다시 실험할 수
          있게 만드는 재현성 장치이기도 합니다.
        </p>
      </details>
    </div>
  );
}
