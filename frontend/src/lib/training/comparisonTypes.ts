// 실험 detail과 제품 모델 비교 행. TrainingView 본문(buildComparisonRows / describe* /
// getComparisonVerdictTag / getFinalJudgement / getMetricInterpretation)과
// 분리된 ComparisonTable 컴포넌트가 공유한다.

export interface ComparisonRow {
  id: string;
  label: string;
  productValue: number;
  experimentValue: number;
  productText: string;
  experimentText: string;
  diffText: string;
  interpretation: string;
  result: "better" | "worse" | "similar" | "neutral";
}
