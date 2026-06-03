// 목표 대비 평가 카드 인터페이스. GoalCard 컴포넌트의 props + 빌더 출력 형식.
// TrainingView (GoalCard / GoalCardGrid / StoredEvaluationSection)과 experimentBuilder가 공유.

export interface GoalCardProps {
  title: string;
  target: string;
  actual: string;
  diff: string;
  judgement: "통과" | "보통" | "개선 필요" | "준비 중" | "저장 없음";
  description: string;
  tone?: "good" | "neutral" | "warn";
  /** CP216 — 출처 배지 (예: "CP210"). */
  source?: string;
  /** 목표를 왜 이 값으로 잡았는지. */
  targetRationale?: string;
}

export interface GoalCardData extends GoalCardProps {
  id: string;
}
