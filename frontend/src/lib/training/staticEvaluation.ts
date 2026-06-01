/**
 * CP216 — AI 모델 "목표 대비 평가" 정적 데이터.
 *
 * 운영 모델 (CP210 라인 / CP153 1D 밴드 / CP178 1W 밴드) 의 평가 카드는 v1 동안 바뀌지 않으므로
 * `detail.metrics.stored_evaluation` 동적 읽기 대신 코드로 박는다.
 *
 * 출처: `docs/v1_operating_models_reproducibility.md` §1~3 (모델별 metrics) + §4.5 (PPT 매핑).
 */
import type { ProductSlotId } from "@/lib/productSlots";

export interface StaticGoalCard {
  id: string;
  title: string;
  metricKey: string;
  target: string;
  actual: string;
  diff: string;
  judgement: "통과" | "보통" | "개선 필요";
  description: string;
  source: string;
  tone: "good" | "neutral" | "warn";
  /** 목표를 왜 이 값으로 잡았는지. 베이스라인 수치 또는 절대 임계값 사유 (case-by-case). */
  targetRationale?: string;
}

export interface StaticEvaluationBlock {
  slotId: "line-1d" | "band-1d" | "band-1w";
  modelLabel: string;
  cards: StaticGoalCard[];
  note?: string;
}

export interface PptMappingRow {
  pptMetric: string;
  pptTarget: string;
  v1Reality: string;
  diff: string;
}

export interface V1ExtraIndicator {
  metricKey: string;
  title: string;
  value: string;
  note: string;
  source: string;
}

/**
 * CP217 / CP217.2 — CP216.2 통계 검정 결과 (베이스라인별).
 * 출처: docs/cp216_2_significance/cp216_significance_summary.csv (+ metrics.json gw_*).
 * 판정은 Bonferroni 보정 p 기준 (다중비교 보수적, n_tests=11).
 */
export type SignificanceVerdict = "통계 우위" | "통계 동등" | "통계 열위";

export interface SignificanceRow {
  /** 한글 표시명 (예: "과거 분위수"). */
  baseline: string;
  /** CSV 의 baseline_id 그대로 (예: "historical_quantile"). 표 셀에 monospace 작은 글씨. */
  baselineCode: string;
  opsValue: string;
  baselineValue: string;
  /** Paired mean diff (CSV `mean_diff`, 4dp round). ops/baseline 4dp round 차이와 살짝 어긋날 수 있음 (DM paired diff 가 산술적으로 정확). */
  delta: string;
  /** Cohen's d. 부호 + 크기 라벨 (예: "+0.93 (large)"). */
  cohensD: string;
  /** Bonferroni 보정 p (two-sided). 예: "<0.001", "0.500", "1.00". */
  bonferroniP: string;
  /** Cluster bootstrap 95% CI (예: "[0.0005, 0.0008]"). 라인은 cluster_n=0 라 "—". */
  ciCluster: string;
  /** Block sqrt-t bootstrap 95% CI. 모든 비교 가용. */
  ciBlock: string;
  verdict: SignificanceVerdict;
  /** true 면 이 행은 walk-forward 한정 구간 비교 — 우리 값도 그 구간 기준. */
  partialWindow?: boolean;
}

/** GW (Giacomini–White) regime-conditional test. 1W 만 (1D 는 asof 1년 backfill 미완). */
export interface GwRegimeRow {
  /** 한글 regime 라벨. */
  regime: string;
  /** CSV/metrics.json 의 regime code (예: "vix_high"). */
  regimeCode: string;
  /** Regime indicator β coefficient. 부호 명시. */
  betaCoef: string;
  /** Wald test Bonferroni p (regime indicator joint test). */
  bonferroniP: string;
  /** "차이 systematic" / "차이 없음". */
  verdict: string;
}

/** 모델별 자유 개수의 결과 박스. 박스 = 검정 1종 (DM / Bootstrap CI / GW). */
export interface SignificanceFinding {
  /** 박스 제목 (예: "효과 차이 (DM test)"). */
  title: string;
  /** 이 검정이 무엇을 묻는가 — 1줄. 사용자/면접관이 검정을 모를 가능성을 가정. */
  question: string;
  /** 답 한 줄 (예: "비슷함" / "1승 2패 1무" / "차이 systematic"). 일상어. */
  verdict: string;
  /** 색 톤. */
  tone: "good" | "neutral" | "warn" | "unknown";
  /** 모델의 구체 답 한 줄. 수치/근거 명시. */
  detail: string;
}

export interface SignificanceBlock {
  slotId: "line-1d" | "band-1d" | "band-1w";
  metricLabel: string;
  metricDirection: string;
  /** ops 컬럼 라벨 (예: "1D line", "1D band", "1W band"). */
  opsLabel: string;
  /** 한 줄 요약 (학계 톤). 다른 narrative/trust-note 없음. */
  headline: string;
  /** 모델별 자유 개수의 결과 박스. CP217.2 — Q1/Q2/Q3 강제 통일 폐기. */
  findings: SignificanceFinding[];
  rows: SignificanceRow[];
  /** 1W 만 — walk-forward GARCH 베이스라인 대비 3 regime. */
  gwRegime?: GwRegimeRow[];
  /** GW 표 아래 자세 해석 (regime별 무엇이 어떻게 좋고 나쁜지). 1W 만. */
  gwInterpretation?: {
    baselineMeanDiff: string;
    paragraphs: Array<{ heading: string; body: string }>;
    triggerImplication: string;
  };
  /** 데이터 출처 표시 (footnote 용, 현재 미노출). */
  sourceCsv: string;
}

/**
 * 라인 (CP210 F4 β=4 5-seed ensemble · test ensemble mean).
 * WF IC range 만 fail → ship 판정 NO_SHIP 사유. 그대로 정직하게 노출.
 */
const LINE_CP210: StaticEvaluationBlock = {
  slotId: "line-1d",
  modelLabel: "CP210 F4 β=4 5-seed ensemble",
  cards: [
    {
      id: "line-ic",
      title: "수익 순위 예측력",
      metricKey: "ic",
      target: "≥ 0.030",
      actual: "0.0325",
      diff: "+0.0025",
      judgement: "통과",
      description: "모델이 매긴 순위가 실제 수익 순위와 얼마나 일치하는지. 통계 베이스라인 0.042~0.044 보다는 낮지만 v1 목표 0.030 통과.",
      source: "Information Coefficient",
      tone: "good",
      targetRationale: "통계 베이스라인 historical_mean 0.024 위, 이전 모델 CP175 0.042 보다 보수적으로 잡음. β=4 비대칭 손실로 severe recall 우선이라 IC 약간 양보 허용.",
    },
    {
      id: "line-severe-recall",
      title: "큰 하락 포착률",
      metricKey: "severe_recall",
      target: "≥ 0.75",
      actual: "0.7727",
      diff: "+0.0227",
      judgement: "통과",
      description: "큰 하락 구간을 미리 잡아낸 비율. 통계 베이스라인·이전 모델 0.62~0.79 평균 위.",
      source: "Severe Recall",
      tone: "good",
      targetRationale: "통계 베이스라인 0.62~0.67 위 약 0.10 위로 잡음. v1 최우선 metric 이라 통계보다 의미있게 위로. 이전 모델 CP175 0.79 와 비슷한 수준 유지.",
    },
    {
      id: "line-false-safe",
      title: "위험 오판율",
      metricKey: "false_safe",
      target: "≤ 0.210",
      actual: "0.2048",
      diff: "-0.0052",
      judgement: "통과",
      description: "안전이라고 판정한 구간에서 실제 큰 손실이 난 비율. 낮을수록 좋음.",
      source: "False-Safe Rate",
      tone: "good",
      targetRationale: "이전 모델 CP175 0.197 와 유사하게 잡음. 위험 회피 품질 지표라 CP175 수준 이하 유지가 목표 — 새 모델이 기존보다 위험 회피 약해지면 안 되기 때문.",
    },
    {
      id: "line-spread",
      title: "상위 하위 수익 차",
      metricKey: "spread",
      target: "> 0",
      actual: "0.0055",
      diff: "+0.0055",
      judgement: "통과",
      description: "상위 decile 과 하위 decile 의 일평균 수익 차. 양수이면 좋은 종목과 나쁜 종목을 가르는 힘이 있다는 뜻.",
      source: "Long-Short Spread",
      tone: "good",
      targetRationale: "절대 임계값. 양수면 상위·하위 decile 구분력 있음을 의미. 0 이하면 random 과 동등 — 이 모델의 최소 존재 이유.",
    },
    {
      id: "line-wf-ic-range",
      title: "시장 구간 별 안정성",
      metricKey: "wf_ic_range",
      target: "≤ 0.040",
      actual: "0.0457",
      diff: "+0.0057",
      judgement: "개선 필요",
      description: "walk-forward 4 fold IC 의 max-min. 시장 국면이 바뀌어도 일관된 성능이 나오는지. 목표 초과 = NO_SHIP 사유.",
      source: "WF IC Range",
      tone: "warn",
      targetRationale: "ship 정책 기준선. fold 간 IC 편차가 0.040 이상이면 시장 국면 변화에 모델이 취약하다 판정 → 새 학습 권고 임계값. CP210 은 0.0457 로 초과 → NO_SHIP.",
    },
  ],
};

/**
 * 1D 밴드 (CP153 TiDE · save-run test).
 * Coverage 목표(q15~q85, 즉 70%) 거의 일치. width_ic 강한 양수.
 */
const BAND_CP153: StaticEvaluationBlock = {
  slotId: "band-1d",
  modelLabel: "CP153 TiDE save-run",
  cards: [
    {
      id: "band1d-coverage-abs-error",
      title: "목표 포함률 오차",
      metricKey: "coverage_abs_error",
      target: "≤ 0.05",
      actual: "0.0099",
      diff: "-0.0401",
      judgement: "통과",
      description: "목표 포함률 0.70 대비 실제 포함률의 절대 오차. 거의 정확히 맞음. Stage 5T 참조값 0.0254 대비 0.016 개선.",
      source: "Coverage Abs Error",
      tone: "good",
      targetRationale: "calibration 학계 standard 임계값. 5%p 이내면 conformal 보정이 정상 작동했다고 봄. CP153 Stage 5T 참조값 0.0254 보다 더 엄격하게 잡음.",
    },
    {
      id: "band1d-width-ic",
      title: "밴드 폭의 변동성 반응도",
      metricKey: "band_width_ic",
      target: "> 0",
      actual: "0.376",
      diff: "+0.376",
      judgement: "통과",
      description: "폭이 실제 변동성에 얼마나 반응하는지. 강한 양수 = 위험 확장 잘 반영. Stage 5T 참조값 0.374 대비 약간 우위.",
      source: "Band Width IC",
      tone: "good",
      targetRationale: "절대 임계값. 양수면 폭이 실제 변동성과 같은 방향으로 움직임 = 위험 신호 잡음. 0 이하면 random 과 동등 — 위험 시각화 도구의 최소 존재 이유.",
    },
    {
      id: "band1d-lower-breach",
      title: "하단 이탈률",
      metricKey: "lower_breach",
      target: "≈ 0.15",
      actual: "0.1586",
      diff: "+0.0086",
      judgement: "보통",
      description: "실제 종가가 하단 q15 분위수 아래로 내려간 비율. 목표 0.15 에 근접하지만 약간 초과.",
      source: "Lower Breach Rate",
      tone: "neutral",
      targetRationale: "q15 분위수 정의 그 자체. 하단을 정확히 q15 (=하위 15%) 로 학습했으면 실제 이탈률도 0.15 가 정상. 0.15 에서 멀수록 calibration 어긋남.",
    },
    {
      id: "band1d-upper-breach",
      title: "상단 이탈률",
      metricKey: "upper_breach",
      target: "≈ 0.15",
      actual: "0.1513",
      diff: "+0.0013",
      judgement: "보통",
      description: "실제 종가가 상단 q85 분위수 위로 올라간 비율. 목표 0.15 에 거의 일치.",
      source: "Upper Breach Rate",
      tone: "neutral",
      targetRationale: "q85 분위수 정의 그 자체. 상단을 정확히 q85 (=상위 15%) 로 학습했으면 실제 이탈률 0.15 가 정상. lower 와 합치면 목표 coverage 0.70 의 좌우 대칭 점검.",
    },
    {
      id: "band1d-downside-width-ic",
      title: "하방 폭의 손실 반영도",
      metricKey: "downside_width_ic",
      target: "> 0",
      actual: "0.0866",
      diff: "+0.0866",
      judgement: "통과",
      description: "하방 폭 즉 종가에서 하단까지의 거리가 실제 손실 크기와 얼마나 연동되는지. 양수 = 하방 위험 반영.",
      source: "Downside Width IC",
      tone: "good",
      targetRationale: "절대 임계값. 양수면 하방 폭이 실제 손실 크기와 같은 방향으로 움직임 = 큰 손실 구간에서 밴드 아래쪽이 넓어짐. 0 이하면 위험 회피 도구로서 기능 X.",
    },
  ],
};

/**
 * 1W 밴드 (CP178 TiDE WFLOCK). 운영 metrics 일부만 모음.
 */
const BAND_CP178: StaticEvaluationBlock = {
  slotId: "band-1w",
  modelLabel: "CP178 TiDE WFLOCK",
  cards: [
    {
      id: "band1w-coverage-abs-error",
      title: "운영 평균 포함률 오차",
      metricKey: "coverage_abs_error",
      target: "≤ 0.05",
      actual: "0.039",
      diff: "-0.011",
      judgement: "통과",
      description: "목표 포함률 0.80 q10~q90 대비 실제 포함률의 절대 오차. 주간 난이도가 높지만 목표 안에 들어옴.",
      source: "Coverage Abs Error",
      tone: "good",
      targetRationale: "1D 와 같은 학계 standard 임계값. 1W 는 주간 난이도 (표본 수가 1/5 수준) 가 높아 더 어렵지만 같은 기준 유지.",
    },
    {
      id: "band1w-coverage-fold2",
      title: "안정 구간 포함률",
      metricKey: "coverage_fold2",
      target: "≈ 0.80",
      actual: "0.8002",
      diff: "+0.0002",
      judgement: "통과",
      description: "WFLOCK walk-forward fold_2 의 3 seed 평균. 목표 0.80 거의 정확히. bootstrap 95% CI 폭 ±0.005p 로 같은 seed 내 모델 일관성 높음.",
      source: "Coverage at fold_2",
      tone: "good",
      targetRationale: "q10~q90 분위수 정의 그 자체 (=80% 목표 coverage). fold 별 평가에서도 같은 분위수 정의를 적용해 정확히 0.80 근처면 모델 calibration 의도대로 작동.",
    },
    {
      id: "band1w-coverage-fold1",
      title: "분포 이동 구간 포함률",
      metricKey: "coverage_fold1",
      target: "≈ 0.80",
      actual: "0.7462",
      diff: "-0.0538",
      judgement: "개선 필요",
      description: "WFLOCK fold_1 의 3 seed 평균. 분포 이동 구간이라 목표 미달. fold_3 도 비슷한 패턴 0.7472.",
      source: "Coverage at fold_1",
      tone: "warn",
      targetRationale: "같은 q10~q90 정의 (=0.80) 적용. 단 fold_1 은 분포 이동 (시장 국면 변화) 구간이라 실제 미달 — 목표 달성보다 '국면 변화에 calibration 흔들림 정직 노출' 이 이 카드의 목적.",
    },
    {
      id: "band1w-width-ic",
      title: "밴드 폭의 변동성 반응도",
      metricKey: "band_width_ic",
      target: "> 0",
      actual: "0.34",
      diff: "+0.34",
      judgement: "통과",
      description: "주간 폭이 실제 변동성에 얼마나 반응하는지. 양수 = 위험 확장 반영. 1D 0.376 대비 약함, 주봉 표본 수가 적은 영향.",
      source: "Band Width IC",
      tone: "good",
      targetRationale: "1D 와 동일한 절대 임계값. 양수면 폭이 변동성 신호 잡음. 1W 는 주봉이라 1D 보다 약하지만 양수 유지면 도구로서 OK.",
    },
    {
      id: "band1w-lower-breach-fold2",
      title: "안정 구간 하단 이탈률",
      metricKey: "lower_breach_fold2",
      target: "≈ 0.10",
      actual: "0.0922",
      diff: "-0.0078",
      judgement: "통과",
      description: "WFLOCK fold_2 의 3 seed 평균. 목표 0.10 에 근접. fold_1 은 0.124 로 약간 초과.",
      source: "Lower Breach at fold_2",
      tone: "good",
      targetRationale: "q10 분위수 정의 그 자체 (=하위 10%). 정확히 학습됐다면 하단 이탈률 0.10 이 정상 — 1W 는 q10/q90 사용이라 0.15 가 아닌 0.10.",
    },
    {
      id: "band1w-bootstrap-stability",
      title: "같은 시드 내 모델 일관성",
      metricKey: "bootstrap_ci_width",
      target: "좁을수록 일관",
      actual: "±0.003~0.005p",
      diff: "—",
      judgement: "통과",
      description: "fold/seed 별 coverage 의 bootstrap 95% CI 폭. 좌우대칭 1%p 미만.",
      source: "Bootstrap 95% CI",
      tone: "good",
      targetRationale: "절대 정성 임계값. 같은 fold/seed 의 bootstrap 95% CI 폭이 좁으면 모델 추론 자체가 안정적이라는 의미. 1%p 미만이면 양호 — 학계 conformal prediction 관행.",
    },
  ],
  note: "fold_2 는 목표 0.80 정확 달성. fold_1·fold_3 는 분포 이동 구간이라 미달. WFLOCK calibration 으로 fold_2 에서 안정성 회복.",
};

export const STATIC_EVALUATIONS: Record<string, StaticEvaluationBlock> = {
  "line-1d": LINE_CP210,
  "band-1d": BAND_CP153,
  "band-1w": BAND_CP178,
};

export function getStaticEvaluation(slotId: ProductSlotId | string | null | undefined): StaticEvaluationBlock | null {
  if (!slotId) {
    return null;
  }
  return STATIC_EVALUATIONS[slotId] ?? null;
}

/**
 * CP217 — CP216.2 통계 검정 (베이스라인별).
 * 값 출처: docs/cp216_2_significance/cp216_significance_summary.csv.
 * 판정 기준: Bonferroni 보정 p < 0.05 일 때만 우위/열위, 아니면 동등 (다중비교 보수적).
 */
const CP216_2_CSV = "docs/cp216_2_significance/cp216_significance_summary.csv";

const SIGNIFICANCE_LINE: SignificanceBlock = {
  slotId: "line-1d",
  metricLabel: "IC (순위 예측력, daily)",
  metricDirection: "높을수록 좋음",
  opsLabel: "1D line",
  rows: [
    {
      baseline: "과거 평균",
      baselineCode: "historical_mean",
      opsValue: "0.0324",
      baselineValue: "0.0243",
      delta: "+0.0081",
      cohensD: "+0.12 (negligible)",
      bonferroniP: "1.00",
      ciCluster: "—",
      ciBlock: "[−0.0090, +0.0264]",
      verdict: "통계 동등",
    },
    {
      baseline: "이전 모델 (CP175)",
      baselineCode: "cp175_beta5",
      opsValue: "0.0365",
      baselineValue: "0.0392",
      delta: "−0.0027",
      cohensD: "−0.07 (negligible)",
      bonferroniP: "1.00",
      ciCluster: "—",
      ciBlock: "[−0.0095, +0.0048]",
      verdict: "통계 동등",
    },
  ],
  headline:
    "단순 통계와 비슷한 정확도. 모델의 차별점은 큰 하락 포착에 있습니다.",
  findings: [
    {
      title: "효과 차이 (DM 검정)",
      question: "모델 평균 정확도가 비교군과 의미있게 다른가?",
      verdict: "비슷함",
      tone: "neutral",
      detail: "과거 평균·이전 모델 CP175 둘 다 Bonferroni p=1.00, Cohen's d<0.2. 5-seed 앙상블의 추가 가치가 통계로 보이지 않음.",
    },
    {
      title: "차이 범위 (Bootstrap 신뢰구간)",
      question: "이 작은 차이를 데이터 흔들림 안에서 얼마나 확신할 수 있나?",
      verdict: "확신 못 함 (0 포함)",
      tone: "neutral",
      detail: "block CI [−0.0090, +0.0264] · 음수~양수 다 가능. 우열 못 가림. (cluster CI 라인엔 미적용)",
    },
    {
      title: "모델 차별점",
      question: "그래서 라인 모델이 잘 하는 건 어디인가?",
      verdict: "큰 하락 포착 0.7727",
      tone: "good",
      detail: "severe recall 0.7727 vs 통계 베이스라인 0.62~0.67. 비대칭 손실(β=4)로 낙관 억제하며 직접 학습한 결과.",
    },
  ],
  sourceCsv: CP216_2_CSV,
};

const SIGNIFICANCE_BAND_1D: SignificanceBlock = {
  slotId: "band-1d",
  metricLabel: "Pinball loss (daily, q15 + q85)",
  metricDirection: "낮을수록 좋음",
  opsLabel: "1D band",
  rows: [
    {
      baseline: "볼린저 밴드",
      baselineCode: "bollinger",
      opsValue: "0.0161",
      baselineValue: "0.0211",
      delta: "−0.0050",
      cohensD: "−1.97 (large)",
      bonferroniP: "<0.001",
      ciCluster: "[−0.00521, −0.00486]",
      ciBlock: "[−0.00565, −0.00426]",
      verdict: "통계 우위",
    },
    {
      baseline: "과거 분위수",
      baselineCode: "historical_quantile",
      opsValue: "0.0161",
      baselineValue: "0.0155",
      delta: "+0.0006",
      cohensD: "+0.93 (large)",
      bonferroniP: "<0.001",
      ciCluster: "[+0.000528, +0.000791]",
      ciBlock: "[+0.000504, +0.000820]",
      verdict: "통계 열위",
    },
    {
      baseline: "GARCH(1,1) 정적",
      baselineCode: "garch_p_q_1_1",
      opsValue: "0.0161",
      baselineValue: "0.0158",
      delta: "+0.0003",
      cohensD: "+0.24 (small)",
      bonferroniP: "0.500",
      ciCluster: "[+0.000138, +0.000453]",
      ciBlock: "[−0.000105, +0.000703]",
      verdict: "통계 동등",
    },
    {
      baseline: "GARCH walk-forward",
      baselineCode: "garch_walkforward",
      opsValue: "0.0155",
      baselineValue: "0.0152",
      delta: "+0.0005",
      cohensD: "+0.45 (small)",
      bonferroniP: "<0.001",
      ciCluster: "[+0.000188, +0.000437]",
      ciBlock: "[+0.000139, +0.000755]",
      verdict: "통계 열위",
      partialWindow: true,
    },
  ],
  headline:
    "예측 오차는 단순 통계와 비슷한 수준. 밴드가 실제값을 덮는 정확성(coverage)은 목표 통과. 위기 구간 반응은 데이터 더 필요.",
  findings: [
    {
      title: "효과 차이 (DM 검정)",
      question: "베이스라인 4종과 예측 오차 평균이 다른가?",
      verdict: "1승 2패 1무",
      tone: "neutral",
      detail: "볼린저 대비 우위 (d=−1.97). 과거 분위수·walk-forward GARCH 대비 열위 (d 0.45~0.93, p<0.001). 정적 GARCH는 동등.",
    },
    {
      title: "차이 범위 (Bootstrap 신뢰구간)",
      question: "이 차이를 얼마나 확신할 수 있나?",
      verdict: "방향 확실 (정적 GARCH 제외)",
      tone: "neutral",
      detail: "cluster·block CI 부호 일치하고 0 포함하지 않음 — 우위/열위 판정 신뢰 가능. 정적 GARCH만 block CI 0 걸쳐 동등 판정.",
    },
    {
      title: "Regime 반응 (GW 검정)",
      question: "위기 구간(VIX↑, drawdown↓)에서 다르게 반응하나?",
      verdict: "검증 보류",
      tone: "unknown",
      detail: "운영 데이터 1년치라 VIX>30 구간 2일, drawdown ≤−10% 0일. 표본 부족으로 검정 불가. CP216.3 backfill 후속.",
    },
  ],
  sourceCsv: CP216_2_CSV,
};

const SIGNIFICANCE_BAND_1W: SignificanceBlock = {
  slotId: "band-1w",
  metricLabel: "Pinball loss (daily, q10 + q90)",
  metricDirection: "낮을수록 좋음",
  opsLabel: "1W band",
  rows: [
    {
      baseline: "볼린저 밴드",
      baselineCode: "bollinger",
      opsValue: "0.0257",
      baselineValue: "0.0282",
      delta: "−0.0027",
      cohensD: "−0.66 (medium)",
      bonferroniP: "0.002",
      ciCluster: "[−0.00308, −0.00237]",
      ciBlock: "[−0.00456, −0.00108]",
      verdict: "통계 우위",
    },
    {
      baseline: "과거 분위수",
      baselineCode: "historical_quantile",
      opsValue: "0.0257",
      baselineValue: "0.0239",
      delta: "+0.0015",
      cohensD: "+1.17 (large)",
      bonferroniP: "<0.001",
      ciCluster: "[+0.00122, +0.00178]",
      ciBlock: "[+0.00111, +0.00199]",
      verdict: "통계 열위",
    },
    {
      baseline: "GARCH(1,1) 정적",
      baselineCode: "garch_p_q_1_1",
      opsValue: "0.0257",
      baselineValue: "0.0242",
      delta: "+0.0015",
      cohensD: "+0.61 (medium)",
      bonferroniP: "0.004",
      ciCluster: "[+0.00113, +0.00187]",
      ciBlock: "[+0.000642, +0.00251]",
      verdict: "통계 열위",
    },
    {
      baseline: "GARCH walk-forward",
      baselineCode: "garch_walkforward",
      opsValue: "0.0253",
      baselineValue: "0.0314",
      delta: "−0.0069",
      cohensD: "−0.98 (large)",
      bonferroniP: "<0.001",
      ciCluster: "[−0.01082, −0.00224]",
      ciBlock: "[−0.01071, −0.00288]",
      verdict: "통계 우위",
      partialWindow: true,
    },
  ],
  gwRegime: [
    {
      regime: "고변동 구간 (VIX > 30)",
      regimeCode: "vix_high",
      betaCoef: "+0.00648",
      bonferroniP: "<0.001",
      verdict: "꾸준한 차이 (우연 아님)",
    },
    {
      regime: "큰 손실 구간 (DD ≤ −10%)",
      regimeCode: "drawdown_low",
      betaCoef: "+0.01037",
      bonferroniP: "<0.001",
      verdict: "꾸준한 차이 (우연 아님)",
    },
    {
      regime: "고변동·손실 결합",
      regimeCode: "combined",
      betaCoef: "+0.00797",
      bonferroniP: "<0.001",
      verdict: "꾸준한 차이 (우연 아님)",
    },
  ],
  gwInterpretation: {
    baselineMeanDiff: "평상시 평균 차이 d̄ = −0.0069 (모델 우위: 모델 pinball loss 가 GARCH 보다 0.0069 작음)",
    paragraphs: [
      {
        heading: "고변동 구간 (VIX > 30)",
        body:
          "regime 효과 β = +0.0065 → 평상시 우위 (−0.0069) 가 거의 사라짐 (d ≈ −0.0004). 모델과 walk-forward GARCH 가 사실상 같은 수준. 변동성 급등기에는 ML 의 추가 가치 소멸.",
      },
      {
        heading: "큰 손실 구간 (drawdown ≤ −10%)",
        body:
          "regime 효과 β = +0.0104 → 평상시 우위가 완전히 역전 (d ≈ +0.0035). 모델 pinball loss 가 walk-forward GARCH 보다 더 큼 → 모델이 GARCH 보다 못함. 위기에서는 단순 시계열 변동성 모델이 더 정확.",
      },
      {
        heading: "결합 구간 (둘 중 하나라도 발생)",
        body:
          "regime 효과 β = +0.0080 → 우위 역전 (d ≈ +0.0010). 위기/고변동 어느 한쪽이라도 들어가면 모델 우위 잃음.",
      },
    ],
    triggerImplication:
      "→ Selective output 정량 근거: 평상시엔 1W 밴드 출력해도 안전하지만, 위기 구간(VIX>30 또는 drawdown ≤−10%) 에서는 출력을 줄이거나 보류해야 함. 3 구간 모두 Bonferroni p<0.001 로 우연 아닌 꾸준한 차이.",
  },
  headline:
    "예측 오차는 모델 따라 갈림(2승 2패). 위기 구간에서 모델이 다르게 반응 — 이 모델의 차별점.",
  findings: [
    {
      title: "효과 차이 (DM 검정)",
      question: "베이스라인 4종과 예측 오차 평균이 다른가?",
      verdict: "2승 2패",
      tone: "neutral",
      detail: "볼린저·walk-forward GARCH 대비 우위 (d=−0.66, −0.98, p<0.001). 과거 분위수·정적 GARCH 대비 열위 (d=+1.17, +0.61).",
    },
    {
      title: "차이 범위 (Bootstrap 신뢰구간)",
      question: "이 차이를 얼마나 확신할 수 있나?",
      verdict: "결과 신뢰 가능",
      tone: "good",
      detail: "4개 비교 모두 cluster·block CI 부호 일치 + 0 포함하지 않음. 우위/열위 판정이 데이터 흔들림 안에서 일관됨.",
    },
    {
      title: "Regime 반응 (GW 검정)",
      question: "위기 구간(VIX↑, drawdown↓)에서 다르게 반응하나?",
      verdict: "꾸준한 차이 (3 구간 모두)",
      tone: "good",
      detail: "VIX>30 · drawdown ≤−10% · 결합 3 regime 모두 wald p<0.001. 위기에서 모델이 평상시와 다르게 반응하는 게 통계로 확인됨.",
    },
  ],
  sourceCsv: CP216_2_CSV,
};

const STATIC_SIGNIFICANCE: Record<string, SignificanceBlock> = {
  "line-1d": SIGNIFICANCE_LINE,
  "band-1d": SIGNIFICANCE_BAND_1D,
  "band-1w": SIGNIFICANCE_BAND_1W,
};

export function getStaticSignificance(slotId: ProductSlotId | string | null | undefined): SignificanceBlock | null {
  if (!slotId) {
    return null;
  }
  return STATIC_SIGNIFICANCE[slotId] ?? null;
}

/**
 * 초기 계획(PPT) 평가지표 매핑 — 모델별로 그 모델에 해당하는 지표만.
 * 출처: docs/v1_operating_models_reproducibility.md §4.5.
 */
const PPT_MAPPING_BY_SLOT: Record<string, PptMappingRow[]> = {
  "line-1d": [
    {
      pptMetric: "MAPE",
      pptTarget: "< 5%",
      v1Reality: "미사용",
      diff: "라인은 가격 절대값이 아니라 수익률 score 라 MAPE 가 맞지 않습니다.",
    },
    {
      pptMetric: "방향 정확도",
      pptTarget: "> 55%",
      v1Reality: "수익 순위 예측력 0.0325",
      diff: "순위 상관으로 대체. 약하지만 방향 구분력은 있습니다.",
    },
    {
      pptMetric: "모델 간 비교",
      pptTarget: "각각 확인",
      v1Reality: "PatchTST 채택",
      diff: "CP208Z baseline lock 에서 결정. CNN-LSTM 은 비교 기준.",
    },
    {
      pptMetric: "지지/저항 일치",
      pptTarget: "시각적 확인",
      v1Reality: "경험적 확인",
      diff: "대부분 구간에서 기준선이 지지·저항 영역을 따라갑니다.",
    },
    {
      pptMetric: "API 응답 속도",
      pptTarget: "< 3초",
      v1Reality: "라인 9ms",
      diff: "목표 3초 대비 충분한 마진.",
    },
  ],
  "band-1d": [
    {
      pptMetric: "밴드 포함율",
      pptTarget: "> 90%",
      v1Reality: "70% (q15~q85)",
      diff: "5거래일 90% CI 가 너무 넓어 q-pair 좁히고 conformal 보정.",
    },
    {
      pptMetric: "모델 간 비교",
      pptTarget: "각각 확인",
      v1Reality: "TiDE 채택",
      diff: "CP153 model zoo 에서 결정. CNN-LSTM·TCN 은 비교 기준.",
    },
    {
      pptMetric: "지지/저항 일치",
      pptTarget: "시각적 확인",
      v1Reality: "경험적 확인",
      diff: "대부분 구간에서 밴드가 지지·저항 영역을 따라갑니다.",
    },
    {
      pptMetric: "API 응답 속도",
      pptTarget: "< 3초",
      v1Reality: "1D 밴드 27ms",
      diff: "목표 3초 대비 충분한 마진.",
    },
  ],
  "band-1w": [
    {
      pptMetric: "밴드 포함율",
      pptTarget: "> 90%",
      v1Reality: "80% (q10~q90)",
      diff: "주간 90% CI 가 너무 넓어 q-pair 좁히고 보정.",
    },
    {
      pptMetric: "모델 간 비교",
      pptTarget: "각각 확인",
      v1Reality: "TiDE 채택",
      diff: "CP153 model zoo 에서 결정. CNN-LSTM·TCN 은 비교 기준.",
    },
    {
      pptMetric: "지지/저항 일치",
      pptTarget: "시각적 확인",
      v1Reality: "경험적 확인",
      diff: "대부분 구간에서 밴드가 지지·저항 영역을 따라갑니다.",
    },
    {
      pptMetric: "API 응답 속도",
      pptTarget: "< 3초",
      v1Reality: "1W 밴드 10ms",
      diff: "목표 3초 대비 충분한 마진.",
    },
  ],
};

export function getPptMapping(slotId: ProductSlotId | string | null | undefined): PptMappingRow[] {
  if (!slotId) {
    return [];
  }
  return PPT_MAPPING_BY_SLOT[slotId] ?? [];
}

/**
 * PPT 미수록 v1 추가 지표 (모델별).
 */
const V1_EXTRA_BY_SLOT: Record<string, V1ExtraIndicator[]> = {
  "line-1d": [
    {
      metricKey: "severe_recall",
      title: "큰 하락 포착률",
      value: "0.7727",
      note: "위험 회피 모델의 핵심 지표. 통계 베이스라인·이전 모델 0.62~0.79 평균 위.",
      source: "Severe Recall",
    },
    {
      metricKey: "false_safe",
      title: "위험 오판율",
      value: "0.2048",
      note: "안전 판정 구간의 실제 손실 비율. 낮을수록 좋음.",
      source: "False-Safe Rate",
    },
    {
      metricKey: "wf_ic_range",
      title: "시장 구간 별 안정성",
      value: "0.0457",
      note: "walk-forward 4 fold IC 의 max-min. 시장 국면 변화에 일관된 성능이 나오는지. ship 판정 기준 0.040 초과.",
      source: "WF IC Range",
    },
  ],
  "band-1d": [
    {
      metricKey: "band_width_ic",
      title: "밴드 폭의 변동성 반응도",
      value: "0.376",
      note: "폭이 실제 변동성에 얼마나 반응하는지.",
      source: "Band Width IC",
    },
    {
      metricKey: "lower_breach",
      title: "하단 이탈률",
      value: "0.1586",
      note: "q15 분위수 목표 대비 실제 하단 이탈 비율. 포함률을 상하단으로 분리 측정.",
      source: "Lower Breach Rate",
    },
  ],
  "band-1w": [
    {
      metricKey: "band_width_ic",
      title: "밴드 폭의 변동성 반응도",
      value: "0.34",
      note: "주간 단위. 1D 0.376 대비 약하지만 양수 유지.",
      source: "Band Width IC",
    },
    {
      metricKey: "wflock_method",
      title: "fold 별 보정 적용",
      value: "fold-by-fold",
      note: "단일 calibration 대신 fold 별 lower calibration 별도 적용. fold_2 에서 목표 80% 적중 회복.",
      source: "Walk-Forward Lower Calibration",
    },
    {
      metricKey: "fold_variation",
      title: "fold 간 coverage 편차",
      value: "0.7462 / 0.8002 / 0.7472",
      note: "fold_1 · fold_2 · fold_3 평균. 분포 이동 구간 1·3 이 fold_2 보다 어렵다는 정량 증거.",
      source: "WFLOCK Fold Summary",
    },
  ],
};

export function getV1ExtraIndicators(slotId: ProductSlotId | string | null | undefined): V1ExtraIndicator[] {
  if (!slotId) {
    return [];
  }
  return V1_EXTRA_BY_SLOT[slotId] ?? [];
}
