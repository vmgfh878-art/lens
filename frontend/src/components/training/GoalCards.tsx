// 목표 대비 평가 카드 (GoalCard) + 그리드 (GoalCardGrid) + 박스 (StoredEvaluationSection)
// + StaticGoalCard → GoalCardData 어댑터. CP216 정적 평가 화면의 표현 묶음.

import type { GoalCardData, GoalCardProps } from "@/lib/training/cardTypes";
import type { StaticGoalCard } from "@/lib/training/staticEvaluation";

export function GoalCard({ title, target, actual, diff, judgement, description, tone = "neutral", source, targetRationale }: GoalCardProps) {
  return (
    <article className={`goal-card goal-card--${tone}`}>
      <div className="goal-card__topline">
        <strong>{title}</strong>
        <span>{judgement}</span>
      </div>
      {source ? <span className="goal-card__source">{source}</span> : null}
      <div className="goal-card__rows">
        <div>
          <span>목표</span>
          <strong>{target}</strong>
        </div>
        <div>
          <span>실제</span>
          <strong>{actual}</strong>
        </div>
      </div>
      {targetRationale ? <p className="goal-card__rationale"><em>목표 근거</em><br />{targetRationale}</p> : null}
      <p>{description}</p>
    </article>
  );
}

function GoalCardGrid({ cards }: { cards: GoalCardData[] }) {
  return (
    <div className="goal-grid">
      {cards.map((card) => (
        <GoalCard
          key={card.id}
          title={card.title}
          target={card.target}
          actual={card.actual}
          diff={card.diff}
          judgement={card.judgement}
          description={card.description}
          tone={card.tone}
          source={card.source}
          targetRationale={card.targetRationale}
        />
      ))}
    </div>
  );
}

/** CP216 — StaticGoalCard 를 GoalCardData 로 변환. judgement / tone 셋이 호환되므로 단순 매핑. */
export function staticCardsToGoalCardData(cards: StaticGoalCard[]): GoalCardData[] {
  return cards.map((card) => ({
    id: card.id,
    title: card.title,
    target: card.target,
    actual: card.actual,
    diff: card.diff,
    judgement: card.judgement,
    description: card.description,
    tone: card.tone,
    source: card.source,
    targetRationale: card.targetRationale,
  }));
}

export function StoredEvaluationSection({ cards }: { cards: GoalCardData[] }) {
  return (
    <section>
      <div className="panel-heading panel-heading--compact">
        <h3>목표 대비 평가</h3>
      </div>
      <div className="trust-note">이 값은 저장된 평가 결과 기준입니다. 평가가 없으면 성능을 판단하지 않습니다.</div>
      {cards.length > 0 ? <GoalCardGrid cards={cards} /> : <div className="empty-state empty-state--compact">저장된 평가 없음</div>}
    </section>
  );
}
