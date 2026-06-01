"use client";

import { useEffect, useMemo, useState } from "react";

import { AiRunDetail, AiRunSummary, fetchAiRun, fetchAiRuns } from "@/api/client";
import { PRODUCT_RUN_IDS } from "@/lib/productSlots";
import type { ProductSlotId } from "@/lib/productSlots";
import {
  BAND_COMPARISON_METRICS,
  BAND_METRICS,
  CONFIG_KEYS_BAND,
  CONFIG_KEYS_COMMON,
  CONFIG_KEYS_LINE,
  CONFIG_LABELS,
  ComparisonMetricDefinition,
  ExperimentCategory,
  ExperimentKind,
  LINE_COMPARISON_METRICS,
  LINE_METRICS,
  MetricDefinition,
  PRODUCT_BAND_1D_RUN_ID,
  PRODUCT_LINE_1D_RUN_ID,
  PRODUCT_SLOTS,
  ProductSlot,
  ProductSlotStatus,
  TRAINING_RUN_MODELS,
} from "@/lib/training/constants";
import {
  buildAdditionalFields,
  buildDetailFields,
  DETAIL_GROUPS,
  DetailField,
  getMetricTargetLabel,
  getPredictionDescription,
  getStructureDescription,
} from "@/lib/training/detailFields";
import {
  extractErrorMessage,
  formatComparisonDiff,
  formatConfigLabel,
  formatFeatureSet,
  formatKoreanDateTime,
  formatMetric,
  formatModelLabel,
  formatRoleLabel,
  formatSignedNumber,
  formatSignedPctPoint,
  formatStatusLabel,
  formatValue,
} from "@/lib/training/formatters";
import {
  formatDetailValue,
  shouldShowDetailValue,
} from "@/lib/training/detailFields";
import {
  formatConfigValue,
  formatExperimentName,
  getChangedExperimentFields,
  getConfigKeys,
  getConfigValue,
  getExperimentDescription,
  getExperimentKind,
  getExperimentTag,
  getRunRole,
  isLegacyRun,
} from "@/lib/training/runUtils";
import StatusInline from "@/components/StatusInline";
import ExperimentTimeline from "@/components/training/ExperimentTimeline";
import LineExperimentArchive from "@/components/training/LineExperimentArchive";
import Band1dExperimentArchive from "@/components/training/Band1dExperimentArchive";
import Band1wExperimentArchive from "@/components/training/Band1wExperimentArchive";
import {
  ApiError,
  classifyApiError,
  describeApiError,
} from "@/lib/apiErrors";
import { BAND_TIMELINE } from "@/lib/training/bandTimeline";
import { LINE_TIMELINE } from "@/lib/training/lineTimeline";
import {
  getPptMapping,
  getStaticEvaluation,
  getStaticSignificance,
  getV1ExtraIndicators,
  type StaticGoalCard,
} from "@/lib/training/staticEvaluation";

type SelectedItem =
  | { kind: "slot"; slotId: ProductSlotId }
  | { kind: "experiment"; runId: string; category: ExperimentCategory };

// 계획서 모델 소개 (PatchTST / TiDE / CNN-LSTM). 쉬운 한마디 → 깊은 상세의 "쉬운" 진입부.
const MODEL_ARCHITECTURES = [
  {
    name: "PatchTST",
    role: "예측선에 사용",
    type: "Transformer 계열",
    desc: "가격 흐름을 여러 조각(patch)으로 나눠 학습합니다. 긴 흐름을 효율적으로 보는 데 강해, 예측선의 다중 seed 앙상블로 씁니다.",
  },
  {
    name: "TiDE",
    role: "AI 밴드에 사용",
    type: "MLP 기반 · Google 2023",
    desc: "Transformer 없이도 빠르고 가벼운 구조입니다. 예상 변동 범위를 분위수로 안정적으로 만드는 데 강해 밴드에 씁니다.",
  },
  {
    name: "CNN-LSTM",
    role: "비교 실험",
    type: "CNN + LSTM 결합",
    desc: "로컬 패턴(CNN)과 시계열 흐름(LSTM)을 합친 검증된 구조입니다. 성능 비교 기준으로 둡니다.",
  },
];

interface ComparisonRow {
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

interface GoalCardProps {
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

interface GoalCardData extends GoalCardProps {
  id: string;
}

interface ExperimentListItem {
  run: AiRunSummary;
  detail: AiRunDetail;
  category: ExperimentCategory;
  kind: ExperimentKind;
  tag: string;
}

// DetailField interface 는 @/lib/training/detailFields 로 이동했다.
// constants / formatters / runUtils 도 마찬가지.

function getMetricByKeys(metrics: Record<string, unknown> | null | undefined, keys: string[]) {
  if (!metrics) {
    return null;
  }
  for (const key of keys) {
    const value = metrics[key];
    if (value != null) {
      return value;
    }
  }
  return null;
}

function getMetricText(
  detail: AiRunDetail | null,
  definition: MetricDefinition,
  fallback: string,
  source: "test" | "val" = "test"
) {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  return formatMetric(getMetricByKeys(metrics, definition.keys), definition.format, fallback);
}

function getMetricNumber(detail: AiRunDetail | null, definition: MetricDefinition, source: "test" | "val" = "test") {
  const metrics = source === "test" ? detail?.test_metrics : detail?.val_metrics;
  const value = getMetricByKeys(metrics, definition.keys);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getMetricNumberFromStoredEvaluation(detail: AiRunDetail | null, definition: MetricDefinition) {
  return getMetricNumber(detail, definition, "test") ?? getMetricNumber(detail, definition, "val");
}

function hasStoredEvaluationMetrics(detail: AiRunDetail | null, definitions: MetricDefinition[]) {
  return definitions.some((definition) => getMetricNumberFromStoredEvaluation(detail, definition) != null);
}

function getProductMetricDefinitions(slot: ProductSlot) {
  if (slot.kind === "band") {
    return BAND_METRICS;
  }
  if (slot.kind === "line") {
    return LINE_METRICS;
  }
  return [];
}

function getProductSlotStatus(slot: ProductSlot, detail: AiRunDetail | null, isLoading: boolean): ProductSlotStatus {
  if (slot.kind === "preparing-line" || slot.kind === "preparing-band" || !slot.runId) {
    return "준비 중";
  }
  if (slot.runId) {
    return "사용 중";
  }
  if (detail && hasStoredEvaluationMetrics(detail, getProductMetricDefinitions(slot))) {
    return "사용 중";
  }
  return "연결 필요";
}

function getStatusPillClass(status: ProductSlotStatus) {
  if (status === "사용 중") {
    return "active";
  }
  if (status === "연결 필요") {
    return "warning";
  }
  return "pending";
}

function getComparisonDefinitions(kind: ExperimentKind) {
  return kind === "band" ? BAND_COMPARISON_METRICS : LINE_COMPARISON_METRICS;
}

function getComparisonResult(metric: ComparisonMetricDefinition, productValue: number, experimentValue: number) {
  const tolerance = metric.format === "rate" || metric.format === "pct_point" ? 0.002 : 0.0005;
  if (metric.better === "neutral") {
    return "neutral" as const;
  }
  if (metric.better === "target_coverage") {
    const productDistance = Math.abs(productValue - 0.7);
    const experimentDistance = Math.abs(experimentValue - 0.7);
    if (Math.abs(productDistance - experimentDistance) <= tolerance) {
      return "similar" as const;
    }
    return experimentDistance < productDistance ? "better" as const : "worse" as const;
  }
  const diff = experimentValue - productValue;
  if (Math.abs(diff) <= tolerance) {
    return "similar" as const;
  }
  if (metric.better === "higher") {
    return diff > 0 ? "better" as const : "worse" as const;
  }
  return diff < 0 ? "better" as const : "worse" as const;
}

function getMetricInterpretation(metric: ComparisonMetricDefinition, result: ComparisonRow["result"], productValue: number, experimentValue: number) {
  if (metric.id === "ic_mean") {
    return result === "better" ? "방향/순위 구분은 현재 모델보다 좋았습니다." : result === "worse" ? "방향/순위 구분력이 현재 모델보다 약합니다." : "방향/순위 구분력은 현재 모델과 비슷합니다.";
  }
  if (metric.id === "long_short_spread") {
    return result === "better" ? "좋은 종목과 나쁜 종목을 나누는 힘은 더 좋았습니다." : result === "worse" ? "좋은 종목과 나쁜 종목을 나누는 힘이 약합니다." : "상위/하위 구분력은 비슷합니다.";
  }
  if (metric.id === "fee_adjusted_sharpe") {
    return result === "better" ? "수수료 반영 안정성은 더 좋았습니다." : result === "worse" ? "수수료 반영 안정성이 현재 모델보다 약합니다." : "수수료 반영 안정성은 비슷합니다.";
  }
  if (metric.id === "false_safe_tail_rate" || metric.id === "false_safe_severe_rate") {
    return result === "better" ? "위험 구간을 안전하다고 본 비율은 더 낮았습니다." : result === "worse" ? "위험 구간을 안전하다고 본 비율이 높습니다." : "위험 오판율은 비슷합니다.";
  }
  if (metric.id === "severe_downside_recall") {
    return result === "better" ? "큰 하락을 포착하는 힘은 더 좋았습니다." : result === "worse" ? "큰 하락을 포착하는 힘이 약합니다." : "큰 하락 포착력은 비슷합니다.";
  }
  if (metric.id === "empirical_coverage") {
    return result === "better" ? "목표 포함률에 더 가까웠습니다." : result === "worse" ? "목표 포함률과 실제 포함률의 차이가 더 큽니다." : "목표 포함률과의 거리는 비슷합니다.";
  }
  if (metric.id === "coverage_abs_error") {
    return result === "better" ? "포함률 오차는 더 작았습니다." : result === "worse" ? "목표 포함률과 실제 포함률 차이가 더 큽니다." : "포함률 오차는 비슷합니다.";
  }
  if (metric.id === "lower_breach_rate") {
    return result === "better" ? "하단 이탈률은 더 낮았습니다." : result === "worse" ? "하방 위험을 충분히 덮지 못했습니다." : "하방 위험 커버는 비슷합니다.";
  }
  if (metric.id === "upper_breach_rate") {
    return result === "better" ? "상단 이탈률은 더 낮았습니다." : result === "worse" ? "상방 변동 범위가 부족했습니다." : "상방 변동 커버는 비슷합니다.";
  }
  if (metric.id === "asymmetric_interval_score") {
    return result === "better" ? "하방 페널티를 포함한 종합 점수는 더 좋았습니다." : result === "worse" ? "하방 페널티를 포함한 종합 품질이 약합니다." : "비대칭 구간 점수는 비슷합니다.";
  }
  if (metric.id === "avg_band_width") {
    if (experimentValue > productValue) {
      return "밴드가 더 넓어 보수적이지만 화면 해석은 무거워질 수 있습니다.";
    }
    if (experimentValue < productValue) {
      return "밴드가 더 좁아 보이지만 위험을 덜 덮을 수 있습니다.";
    }
    return "밴드 폭은 비슷합니다.";
  }
  if (metric.id === "band_width_ic") {
    return result === "better" ? "변동성이 커질 때 밴드가 더 잘 넓어졌습니다." : result === "worse" ? "변동성이 커질 때 밴드가 같이 넓어지는 반응이 약합니다." : "변동성 반응은 비슷합니다.";
  }
  if (metric.id === "downside_width_ic") {
    return result === "better" ? "하락 위험이 커질 때 밴드 반응은 더 좋았습니다." : result === "worse" ? "하락 위험이 커질 때 밴드가 반응하는 힘이 약합니다." : "하방 위험 반응은 비슷합니다.";
  }
  return result === "better" ? "제품 모델보다 나은 지표입니다." : result === "worse" ? "제품 모델보다 약한 지표입니다." : "제품 모델과 비슷합니다.";
}

function buildComparisonRows(detail: AiRunDetail, productDetail: AiRunDetail | null) {
  const kind = getExperimentKind(detail);
  if (!kind || !productDetail) {
    return [];
  }
  return getComparisonDefinitions(kind)
    .map((metric): ComparisonRow | null => {
      const productValue = getMetricNumber(productDetail, metric);
      const experimentValue = getMetricNumber(detail, metric);
      if (productValue == null || experimentValue == null) {
        return null;
      }
      const result = getComparisonResult(metric, productValue, experimentValue);
      return {
        id: metric.id,
        label: metric.label,
        productValue,
        experimentValue,
        productText: formatMetric(productValue, metric.format),
        experimentText: formatMetric(experimentValue, metric.format),
        diffText: formatComparisonDiff(experimentValue - productValue, metric.format),
        interpretation: getMetricInterpretation(metric, result, productValue, experimentValue),
        result,
      };
    })
    .filter((row): row is ComparisonRow => row != null);
}

function hasDisplayableComparison(detail: AiRunDetail, productDetail: AiRunDetail | null) {
  if (detail.timeframe === "1W") {
    return hasDisplayableExperimentMetrics(detail);
  }
  return buildComparisonRows(detail, productDetail).length >= 2;
}

// run 분류 / config 추출 / experiment 명명 helper 는 @/lib/training/runUtils 로 이동했다.
// 아래는 component 내부에서만 쓰는 helper 만 남긴다.

function GoalCard({ title, target, actual, diff, judgement, description, tone = "neutral", source, targetRationale }: GoalCardProps) {
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
function staticCardsToGoalCardData(cards: StaticGoalCard[]): GoalCardData[] {
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

/** CP217.2 — CP216.2 통계 검정 (학계 톤). 메인 8셀 · narrow 화면은 d/p/CI 접힘. 1W 만 GW regime sub-table. */
function SignificanceSection({ slotId }: { slotId: string | null | undefined }) {
  const block = getStaticSignificance(slotId);
  if (!block) {
    return null;
  }
  const verdictClass = (verdict: string) =>
    verdict === "통계 우위" ? "good" : verdict === "통계 열위" ? "warn" : "neutral";
  const hasPartial = block.rows.some((row) => row.partialWindow);
  return (
    <section className="significance">
      <div className="panel-heading panel-heading--compact">
        <h3>통계 검정</h3>
      </div>
      <p className="significance__headline">{block.headline}</p>

      {/* CP217.2 모델별 결과 박스 — 검정 종류(DM / Bootstrap CI / GW)별 묶음. */}
      <div className="significance__findings">
        {block.findings.map((f, fi) => (
          <article key={fi} className={`significance__finding significance__finding--${f.tone}`}>
            <div className="significance__finding-title">{f.title}</div>
            <div className="significance__finding-question">{f.question}</div>
            <div className="significance__finding-verdict">{f.verdict}</div>
            <div className="significance__finding-detail">{f.detail}</div>
          </article>
        ))}
      </div>

      <details className="significance__details-table">
        <summary>통계 검정 상세 표 (DM · Bootstrap CI{block.gwRegime ? " · GW regime" : ""})</summary>
      <div className="significance__meta">
        <div className="significance__meta-guide">
          이 표 읽는 법 — <strong>&ldquo;모델&rdquo; / &ldquo;비교군&rdquo;</strong> 컬럼은 각각의 지표 값입니다. 그 외 컬럼 (<strong>Δ · Cohen&apos;s d · Bonf. p · 95% CI</strong>) 은 모두 <strong>&ldquo;모델 vs 비교군&rdquo; 의 차이</strong> 를 분석한 결과입니다 (개별 점수 X).
        </div>
        <span className="significance__meta-key">지표</span>{" "}<strong>{block.metricLabel}</strong>{" · "}{block.metricDirection}
        <br />
        <span className="significance__meta-key">Δ</span>{" "}모델 ({block.opsLabel}) − 비교군. <strong>{block.metricDirection.includes("낮") ? "음수면 모델이 더 좋음 (loss 감소)" : "양수면 모델이 더 좋음 (지표 상승)"}.</strong>
        <br />
        <span className="significance__meta-key">Cohen&apos;s d</span>{" "}두 모델 차이를 표준화한 효과 크기. 0.2 작음 · 0.5 중간 · 0.8 큼.
        <br />
        <span className="significance__meta-key">Bonf. p</span>{" "}그 차이가 우연일 확률 (n=11 검정 다중비교 보정 후). <strong>0.05 미만이면 우연 아님.</strong>
        <br />
        <span className="significance__meta-key">95% CI</span>{" "}차이가 들어있을 95% 범위. <strong>0 포함하면 우열 못 가림 (= 동등).</strong> 티커별 / 시간 block 별 둘 다 같은 부호면 결과 신뢰 가능.
        <br />
        <span className="significance__meta-key">표본</span>{" "}베이스라인 공통 구간 · Bonferroni n_tests=11
      </div>
      <div className="significance__table" role="table">
        <div className="significance__row significance__row--head" role="row">
          <span role="columnheader">베이스라인</span>
          <span role="columnheader">모델 ({block.opsLabel})</span>
          <span role="columnheader">비교군</span>
          <span role="columnheader">Δ</span>
          <span role="columnheader" className="significance__cell--wide-only">Cohen&apos;s d</span>
          <span role="columnheader" className="significance__cell--wide-only">Bonferroni p</span>
          <span role="columnheader" className="significance__cell--wide-only">95% CI (Bootstrap)</span>
          <span role="columnheader">판정</span>
        </div>
        {block.rows.flatMap((row, i) => [
          <div key={`row-${i}`} className="significance__row" role="row">
            <span role="cell" className="significance__cell--baseline">
              <strong>{row.baseline}</strong>{row.partialWindow ? <sup className="significance__mark">†</sup> : null}
              <span className="significance__code">{row.baselineCode}</span>
            </span>
            <span role="cell" className="significance__num significance__ops">{row.opsValue}</span>
            <span role="cell" className="significance__num">{row.baselineValue}</span>
            <span role="cell" className="significance__num">{row.delta}</span>
            <span role="cell" className="significance__num significance__cell--wide-only">{row.cohensD}</span>
            <span role="cell" className="significance__num significance__cell--wide-only">{row.bonferroniP}</span>
            <span role="cell" className="significance__cell--ci significance__cell--wide-only">
              <span className="significance__ci-line">
                <em>티커</em>
                <span className="significance__ci-value">{row.ciCluster === "—" ? <span className="significance__ci-na">미적용 *</span> : row.ciCluster}</span>
              </span>
              <span className="significance__ci-line">
                <em>시간</em>
                <span className="significance__ci-value">{row.ciBlock}</span>
              </span>
            </span>
            <span role="cell">
              <span className={`significance__verdict significance__verdict--${verdictClass(row.verdict)}`}>{row.verdict}</span>
            </span>
          </div>,
          <details key={`extras-${i}`} className="significance__row-extras">
            <summary>{row.baseline} 상세 (Cohen&apos;s d · Bonf. p · 95% CI)</summary>
            <dl className="significance__extras-dl">
              <div><dt>Cohen&apos;s d</dt><dd>{row.cohensD}</dd></div>
              <div><dt>Bonf. p</dt><dd>{row.bonferroniP}</dd></div>
              <div><dt>95% CI (티커별 Bootstrap)</dt><dd>{row.ciCluster}</dd></div>
              <div><dt>95% CI (시간 block 별 Bootstrap, √T)</dt><dd>{row.ciBlock}</dd></div>
            </dl>
          </details>,
        ])}
      </div>
      {block.rows.some((row) => row.ciCluster === "—") ? (
        <p className="significance__footnote">
          * 라인 IC 는 일별 cross-section 통계라 ticker 단위 cluster bootstrap 미적용 (cluster_n=0). 시간 block 별만 산출.
        </p>
      ) : null}
      {hasPartial ? (
        <p className="significance__footnote">
          † walk-forward 한정 구간 — 해당 행의 {block.opsLabel} 값도 같은 fold 영역 기준.
        </p>
      ) : null}
      {block.gwRegime && block.gwRegime.length > 0 ? (
        <div className="significance__gw">
          <div className="significance__gw-title">Giacomini–White (GW) regime-conditional test · walk-forward GARCH 대비</div>
          <div className="significance__gw-table" role="table">
            <div className="significance__gw-row significance__gw-row--head" role="row">
              <span role="columnheader">Regime</span>
              <span role="columnheader">β</span>
              <span role="columnheader">wald p (Bonferroni)</span>
              <span role="columnheader">판정</span>
            </div>
            {block.gwRegime.map((g) => (
              <div key={g.regimeCode} className="significance__gw-row" role="row">
                <span role="cell" className="significance__cell--baseline">
                  <strong>{g.regime}</strong>
                  <span className="significance__code">{g.regimeCode}</span>
                </span>
                <span role="cell" className="significance__num">{g.betaCoef}</span>
                <span role="cell" className="significance__num">{g.bonferroniP}</span>
                <span role="cell">
                  <span className="significance__verdict significance__verdict--neutral">{g.verdict}</span>
                </span>
              </div>
            ))}
          </div>
          <p className="significance__footnote">
            나머지 3 베이스라인 (bollinger · historical_quantile · garch_p_q_1_1) 도 동일하게 3 regime 모두 wald Bonferroni p&lt;0.001 — regime effect 일관.
          </p>
          {block.gwInterpretation ? (
            <div className="significance__gw-interpretation">
              <div className="significance__gw-interpretation-title">GW 결과 해석 — 구간별 무엇이 어떻게 다른가</div>
              <p className="significance__gw-interpretation-baseline">{block.gwInterpretation.baselineMeanDiff}</p>
              <dl className="significance__gw-interpretation-dl">
                {block.gwInterpretation.paragraphs.map((p, pi) => (
                  <div key={pi}>
                    <dt>{p.heading}</dt>
                    <dd>{p.body}</dd>
                  </div>
                ))}
              </dl>
              <div className="significance__gw-interpretation-trigger">{block.gwInterpretation.triggerImplication}</div>
            </div>
          ) : null}
        </div>
      ) : null}
      </details>
    </section>
  );
}

function PptMappingSection({ slotId }: { slotId: string | null | undefined }) {
  const rows = getPptMapping(slotId);
  if (rows.length === 0) {
    return null;
  }
  return (
    <section className="ppt-mapping">
      <div className="panel-heading panel-heading--compact">
        <h3>초기 계획 평가</h3>
      </div>
      <div className="ppt-mapping__table" role="table">
        <div className="ppt-mapping__row ppt-mapping__row--head" role="row">
          <span role="columnheader">계획 지표</span>
          <span role="columnheader">계획 목표</span>
          <span role="columnheader">v1 운영 대응</span>
          <span role="columnheader">차이 / 사유</span>
        </div>
        {rows.map((row) => (
          <div key={row.pptMetric} className="ppt-mapping__row" role="row">
            <span role="cell"><strong>{row.pptMetric}</strong></span>
            <span role="cell">{row.pptTarget}</span>
            <span role="cell">{row.v1Reality}</span>
            <span role="cell" className="ppt-mapping__diff">{row.diff}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

/** CP216 — 초기 계획과 차이점 (v1 운영에서 추가/대체된 지표). 모델별. */
function V1ExtraIndicatorsSection({ slotId }: { slotId: string | null | undefined }) {
  const extras = getV1ExtraIndicators(slotId);
  if (extras.length === 0) {
    return null;
  }
  return (
    <section className="v1-extra">
      <div className="panel-heading panel-heading--compact">
        <h3>초기 계획과 차이점</h3>
      </div>
      <ul className="v1-extra__list">
        {extras.map((item) => (
          <li key={item.metricKey} className="v1-extra__row">
            <div className="v1-extra__head">
              <strong>{item.title}</strong>
              <span className="v1-extra__value">{item.value}</span>
              <span className="v1-extra__source">{item.source}</span>
            </div>
            <p>{item.note}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

function DataList({ items }: { items: string[] }) {
  return (
    <ul className="model-data-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

// detail field 관련 constants / helpers 는 @/lib/training/detailFields 로 이동했다.

function DetailFieldGrid({ fields }: { fields: DetailField[] }) {
  if (fields.length === 0) {
    return <div className="compact-note">표시할 값이 없습니다.</div>;
  }
  return (
    <div className="detail-field-grid">
      {fields.map((field) => (
        <div key={field.key} className="detail-field">
          <span>{field.label}</span>
          <strong className={field.monospace ? "detail-field__mono" : undefined}>{field.value}</strong>
        </div>
      ))}
    </div>
  );
}

function ModelRunDetails({ detail, metricDefinitions }: { detail: AiRunDetail | null; metricDefinitions: MetricDefinition[] }) {
  if (!detail) {
    return null;
  }
  const usedKeys = new Set<string>();
  const groupedFields = DETAIL_GROUPS.map((group) => ({
    ...group,
    fields: buildDetailFields(detail, group.keys, usedKeys),
  })).filter((group) => group.fields.length > 0);
  const additionalFields = buildAdditionalFields(detail, usedKeys);
  const metricRows = metricDefinitions.map((metric) => ({
    ...metric,
    target: getMetricTargetLabel(metric),
    testValue: getMetricText(detail, metric, "-", "test"),
    valValue: getMetricText(detail, metric, "-", "val"),
  })).filter((metric) => metric.testValue !== "-" || metric.valValue !== "-");
  const wandbStatusValue = detail.wandb_status ?? getConfigValue(detail, "wandb_status");
  const storageFields: DetailField[] = [
    { key: "run_id", label: "실행 ID", value: detail.run_id, monospace: true },
    { key: "status", label: "상태", value: formatStatusLabel(detail.status) },
    { key: "checkpoint_exists", label: "체크포인트", value: detail.checkpoint_path ? "저장됨" : "없음" },
    ...(wandbStatusValue != null ? [{ key: "wandb_status", label: "실험 추적 상태", value: formatDetailValue("wandb_status", wandbStatusValue) }] : []),
    ...(detail.wandb_run_id ? [{ key: "wandb_run_id", label: "실험 추적 ID", value: detail.wandb_run_id, monospace: true }] : []),
    ...(detail.created_at ? [{ key: "created_at", label: "생성 시각", value: formatKoreanDateTime(detail.created_at) }] : []),
  ].filter((field) => shouldShowDetailValue(field.key, field.value));

  return (
    <details className="model-run-details">
      <summary className="model-run-details__summary">
        <span>상세 정보</span>
        <em>모델 설정·평가 지표·저장 정보</em>
      </summary>
      <div className="model-run-details__content">
        <div className="model-run-details__header">
          <div>
            <span className="eyebrow">모델 설정</span>
            <h3>{formatModelLabel(detail.model_name)} · {formatRoleLabel(getRunRole(detail))}</h3>
            <p>{getPredictionDescription(detail)}</p>
          </div>
          <div className="detail-status-card">
            <span>버전</span>
            <strong>{detail.model_ver ?? "v1"}</strong>
            <em>{formatStatusLabel(detail.status)}</em>
          </div>
        </div>

        <div className="model-detail-section">
          <h4>모델 구조</h4>
          <p>{getStructureDescription(detail)}</p>
        </div>

        {groupedFields.map((group) => (
          <div key={group.id} className="model-detail-section">
            <h4>{group.title}</h4>
            <DetailFieldGrid fields={group.fields} />
          </div>
        ))}

        {additionalFields.length > 0 ? (
          <div className="model-detail-section">
            <h4>추가 설정</h4>
            <DetailFieldGrid fields={additionalFields} />
          </div>
        ) : null}

        {metricRows.length > 0 ? (
          <div className="model-detail-section">
            <h4>평가 지표</h4>
            <div className="detail-metric-grid">
              {metricRows.map((metric) => (
                <div key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>목표 {metric.target}</strong>
                  <em>test {metric.testValue}</em>
                  <em>val {metric.valValue}</em>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="model-detail-section model-detail-section--storage">
          <h4>저장 정보</h4>
          <DetailFieldGrid fields={storageFields} />
        </div>
      </div>
    </details>
  );
}

function ProductSlotCard({
  slot,
  status,
  active,
  onSelect,
}: {
  slot: ProductSlot;
  status: ProductSlotStatus;
  active: boolean;
  onSelect: (slotId: ProductSlotId) => void;
}) {
  return (
    <button
      type="button"
      className={`product-slot-card${active ? " product-slot-card--active" : ""}`}
      onClick={() => onSelect(slot.id)}
    >
      <div className="product-slot-card__header">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <span>{slot.timeframe}</span>
      </div>
      <strong>{slot.title}</strong>
      <p>{slot.summary}</p>
      <div className="product-slot-card__meta">
        <span>{slot.version ? `${slot.model} · ${slot.version}` : slot.model}</span>
      </div>
    </button>
  );
}

function PreparingSlotDetail({ slot }: { slot: ProductSlot }) {
  const description =
    slot.id === "line-1w"
      ? "1W 보수적 기준선은 v1에서 제공하지 않습니다. 1W AI 밴드는 자동 갱신 상태로 별도 카드에서 확인할 수 있습니다."
      : "이 슬롯은 다음 학습 단계에서 채울 예정입니다.";

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className="status-pill status-pill--pending">준비 중</span>
        <h2>{slot.title}</h2>
        <p>{description}</p>
      </div>
      <div className="goal-grid">
        <GoalCard
          title="제품 모델"
          judgement="준비 중"
          target="검증 완료"
          actual="아직 없음"
          diff="검증 필요"
          description="현재는 주간 AI 밴드만 활성 상태입니다. 1W 보수적 기준선은 v1에서 제공하지 않습니다."
          tone="neutral"
        />
      </div>
    </div>
  );
}

function StoredEvaluationSection({ cards }: { cards: GoalCardData[] }) {
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

function LineModelDetail({ detail, slot }: { detail: AiRunDetail | null; slot?: ProductSlot | null }) {
  // CP216 — 정적 평가 카드 우선. 운영 모델 (CP210) 은 v1 동안 안 바뀌므로 detail 동적 read 보다 정적 박은 값 사용.
  const staticBlock = getStaticEvaluation(slot?.id);
  const cards: GoalCardData[] = staticBlock
    ? staticCardsToGoalCardData(staticBlock.cards)
    : detail
    ? buildLineExperimentCards(detail)
    : [];
  const hasEvaluation = cards.length > 0;
  const isWeekly = (detail?.timeframe ?? slot?.timeframe) === "1W";
  const status: ProductSlotStatus = slot?.runId ? "사용 중" : "준비 중";
  const horizonLabel = isWeekly ? "4주" : "5거래일";
  const title = slot?.title ?? (isWeekly ? "1W 보수적 기준선" : "1D 보수적 기준선");
  const summary =
    slot?.summary ??
    (isWeekly
      ? "1W 보수적 기준선은 v1에서 제공하지 않습니다. 여러 walk-forward 실험을 돌렸지만 1D 라인만큼의 안정 성능(IC·severe recall)을 주간 단위에서 확보하지 못해 v1 제외했습니다."
      : "수익 방향과 종목 순위 판단에는 사용할 수 있지만, 위험 회피 품질은 개선 중입니다.");

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <h2>{title}</h2>
        <p>{summary}</p>
      </div>

      <article className="model-role-card">
        <h3>모델 역할</h3>
        <p>
          PatchTST 5-seed 앙상블 예측선입니다. 최근 가격과 시장 스트레스 지표를 보고 앞으로 {horizonLabel} 동안의
          도착가를 보수적으로 추정합니다. 차트에는 미래 각 거래일 위치에 표시되고, 매일 새 예측으로 갱신됩니다.
        </p>
      </article>

      <div className="notice">
        출력 계약: 수익률 단위 score 를 학습하고, 화면에는 기준 종가에 환산해 미래 5거래일 구간에
        표시합니다. 모델은 CP210 F4 β=4, 5-seed 앙상블이며, 낙관 오차에 더 큰 페널티를 주는 비대칭 손실로 보수적으로
        학습했습니다.
      </div>

      <section className="model-story-grid">
        <article>
          <h3>좋은 점</h3>
          <ul className="model-copy-list">
            <li><strong>큰 하락 포착률 0.7727</strong> — 통계 베이스라인·이전 모델 0.62~0.79 평균 위. 위험 회피 모델로 안정.</li>
            <li><strong>위험 오판율 0.2048</strong> — 안전 판정 구간의 실제 손실 비율. 목표 0.210 통과, CP175 0.197 와 비슷한 수준 유지.</li>
            <li><strong>상위 하위 수익 차 0.0055</strong> — 양수 = 종목 순위 구분력 존재. 비대칭 손실 α=1 β=4 로 낙관 오차를 누른 결과.</li>
          </ul>
        </article>
        <article>
          <h3>아쉬운 점</h3>
          <ul className="model-copy-list">
            <li><strong>수익 순위 예측력 0.0325</strong> — 통계 베이스라인 0.042~0.044 보다 낮음. 정밀한 순위 예측보다 위험 회피 보조에 적합.</li>
            <li><strong>시장 구간 별 안정성 0.0457</strong> — walk-forward 4 fold 간 IC 편차가 ship 기준 0.040 초과. NO_SHIP 사유. 단독 판정 신호가 아닌 참고선으로 사용.</li>
          </ul>
        </article>
      </section>

      <div className="notice notice--muted">
        커트라인 기준: 직전 수익률을 그대로 가정하는 naive 예측·과거 평균 수익률을 쓰는 historical mean 같은 통계
        베이스라인과, 이전 운영 모델 CP175 v1 을 기준선으로 두고, 그보다 의미 있게 높은 지표만 장점으로 인정했습니다.
      </div>

      <StoredEvaluationSection cards={cards} />
      {staticBlock?.note ? <div className="notice notice--muted">{staticBlock.note}</div> : null}

      <V1ExtraIndicatorsSection slotId={slot?.id} />
      <PptMappingSection slotId={slot?.id} />
      <SignificanceSection slotId={slot?.id} />
      <LineExperimentArchive />

      <div className="notice">이 모델은 투자 조언이 아니라 보조 판단선입니다.</div>
      <ModelRunDetails detail={detail} metricDefinitions={LINE_METRICS} />
    </div>
  );
}

function BandModelDetail({ detail, slot }: { detail: AiRunDetail | null; slot?: ProductSlot | null }) {
  // CP216 — 정적 평가 카드 우선.
  const staticBlock = getStaticEvaluation(slot?.id);
  const cards: GoalCardData[] = staticBlock
    ? staticCardsToGoalCardData(staticBlock.cards)
    : detail
    ? buildBandExperimentCards(detail)
    : [];
  const hasEvaluation = cards.length > 0;
  const status: ProductSlotStatus = slot?.runId ? "사용 중" : "준비 중";
  const isWeekly = (detail?.timeframe ?? slot?.timeframe) === "1W";
  const horizonLabel = isWeekly ? "4주" : "5거래일";
  const title = slot?.title ?? (isWeekly ? "1W AI 밴드 v1" : "1D AI 밴드 v1");
  const summary =
    slot?.summary ??
    (isWeekly
      ? "1W 예상 변동 범위를 보여주는 주간 리스크 참고 밴드입니다."
      : "저장된 평가 결과가 확인된 1D 위험 범위 보조지표입니다.");

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className={`status-pill status-pill--${getStatusPillClass(status)}`}>{status}</span>
        <h2>{title}</h2>
        <p>{summary}</p>
      </div>

      <article className="model-role-card">
        <h3>모델 역할</h3>
        <p>
          TiDE 기반 제품 밴드입니다. 최근 가격·변동성 흐름을 보고 앞으로 {horizonLabel}의 예상 변동 범위를 분위수로
          계산합니다. 밴드가 넓어지는 구간은 모델이 더 큰 변동 가능성을 보는 구간입니다.
        </p>
      </article>

      <div className="notice">
        출력 계약: 하단 {isWeekly ? "q10" : "q15"} · 상단 {isWeekly ? "q90" : "q85"} 분위수를 학습하고{" "}
        {isWeekly ? "walk-forward 별 lower calibration" : "validation lower_focused conformal 보정"}
        으로 목표 포함 비율
        {isWeekly ? " 0.80" : " 0.70"} 에 맞춥니다. 예측 기간은 {isWeekly ? "4주 (CP178)" : "5거래일 (CP153)"} 입니다.
      </div>

      {isWeekly ? (
        <section className="model-story-grid">
          <article>
            <h3>좋은 점</h3>
            <ul className="model-copy-list">
              <li><strong>안정 구간 포함률 0.8002</strong> — fold_2 의 3 seed 평균이 목표 0.80 거의 정확히. bootstrap 95% CI 폭 ±0.005p 로 같은 seed 내 모델 일관성 높음.</li>
              <li><strong>운영 평균 포함률 오차 0.039</strong> — 목표 0.05 안. fold_1·fold_3 미달분을 fold_2 가 정확히 만회.</li>
              <li><strong>밴드 폭 반응도 0.34</strong> — 1D 0.376 대비 약하지만 양수. 주간 변동성 확장도 어느 정도 따라감.</li>
              <li><strong>안정 구간 하단 이탈률 0.0922</strong> — q10 분위수 목표 0.10 에 근접. fold 별 lower calibration 으로 fold_1 의 0.124 대비 회복.</li>
            </ul>
          </article>
          <article>
            <h3>아쉬운 점</h3>
            <ul className="model-copy-list">
              <li><strong>분포 이동 구간 포함률 0.7462 / 0.7472</strong> — fold_1·fold_3 에서 목표 0.80 대비 약 5%p 미달. 분포 변화에는 추격 한계.</li>
              <li><strong>밴드 폭 반응도 0.34</strong> — 1D 0.376 대비 낮음. 주봉 표본 수가 적어 폭 신호가 일봉만큼 정밀하지 않음.</li>
              <li><strong>분포 이동 구간 하단 이탈률 0.124</strong> — q10 목표 0.10 대비 +0.024 초과. 분포 이동 구간 calibration 한계.</li>
            </ul>
          </article>
        </section>
      ) : (
        <section className="model-story-grid">
          <article>
            <h3>좋은 점</h3>
            <ul className="model-copy-list">
              <li><strong>포함률 오차 0.0099</strong> — 목표 0.05 대비 5배 마진. Stage 5T TiDE 참조값 0.0254 대비 0.016 개선.</li>
              <li><strong>밴드 폭 반응도 0.376</strong> — 강한 양수, Stage 5T 참조값 0.374 와 동등 이상. 위험 확장 잘 반영.</li>
              <li><strong>하방 폭 손실 반영도 0.0866</strong> — 양수 = 하방 위험 반영. tail risk calibration 효과.</li>
              <li><strong>상단 이탈률 0.1513</strong> — q85 분위수 목표 0.15 에 거의 일치, +0.0013.</li>
            </ul>
          </article>
          <article>
            <h3>아쉬운 점</h3>
            <ul className="model-copy-list">
              <li><strong>하단 이탈률 0.1586</strong> — q15 분위수 목표 0.15 보다 +0.0086 초과. Stage 5T 참조값 0.1425 대비 +0.016. 하방 calibration 의 살짝 느슨함.</li>
              <li><strong>목표 포함률 70%</strong> — 초기 계획서의 90% CI 대비 좁힘. 90% 가 5거래일 단위에서 너무 넓어 실용성 떨어진다고 판단한 결과, 정직하게 70% 로 운영.</li>
            </ul>
          </article>
        </section>
      )}

      <div className="notice notice--muted">
        커트라인 기준: GARCH·historical quantile·볼린저 같은 통계 베이스라인을 기준선으로 두고, 그보다 명확한 장점이
        있는 후보만 제품 모델로 채택했습니다.
      </div>

      <StoredEvaluationSection cards={cards} />
      {staticBlock?.note ? <div className="notice notice--muted">{staticBlock.note}</div> : null}

      <V1ExtraIndicatorsSection slotId={slot?.id} />
      <PptMappingSection slotId={slot?.id} />
      <SignificanceSection slotId={slot?.id} />
      {isWeekly ? <Band1wExperimentArchive /> : <Band1dExperimentArchive />}

      <ModelRunDetails detail={detail} metricDefinitions={BAND_METRICS} />
    </div>
  );
}

function buildLineExperimentCards(detail: AiRunDetail): GoalCardData[] {
  const cards: GoalCardData[] = [];
  const ic = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[0]);
  const spread = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[1]);
  const falseSafe = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[2]);
  const recall = getMetricNumberFromStoredEvaluation(detail, LINE_METRICS[3]);

  if (ic != null) {
    cards.push({
      id: "ic",
      title: "순위 상관",
      judgement: ic > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(ic),
      diff: formatSignedNumber(ic),
      description: ic > 0 ? "수익 방향을 어느 정도 구분했습니다." : "순위 상관이 약해 수익 방향을 안정적으로 구분하지 못했습니다.",
      tone: ic > 0 ? "good" : "warn",
    });
  }
  if (spread != null) {
    cards.push({
      id: "spread",
      title: "상위-하위 수익 차",
      judgement: spread > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(spread),
      diff: formatSignedNumber(spread),
      description: spread > 0 ? "높게 본 구간이 낮게 본 구간보다 나았습니다." : "상위 구간과 하위 구간의 성과 차이가 약했습니다.",
      tone: spread > 0 ? "good" : "warn",
    });
  }
  if (falseSafe != null) {
    cards.push({
      id: "false-safe",
      title: "위험 오판율",
      judgement: falseSafe <= 0.25 ? "통과" : "개선 필요",
      target: "25% 이하",
      actual: formatMetric(falseSafe, "rate"),
      diff: formatSignedPctPoint((falseSafe - 0.25) * 100),
      description: falseSafe <= 0.25 ? "위험 구간 오판이 목표 안에 있습니다." : "위험 구간을 안전하다고 보는 경우가 많았습니다.",
      tone: falseSafe <= 0.25 ? "good" : "warn",
    });
  }
  if (recall != null) {
    cards.push({
      id: "downside-recall",
      title: "큰 하락 포착률",
      judgement: recall >= 0.7 ? "통과" : "개선 필요",
      target: "70% 이상",
      actual: formatMetric(recall, "rate"),
      diff: formatSignedPctPoint((recall - 0.7) * 100),
      description: recall >= 0.7 ? "큰 하락 구간을 비교적 잘 포착했습니다." : "큰 하락을 모두 잡아내기에는 아직 부족합니다.",
      tone: recall >= 0.7 ? "good" : "warn",
    });
  }
  return cards;
}

function buildBandExperimentCards(detail: AiRunDetail): GoalCardData[] {
  const cards: GoalCardData[] = [];
  const empirical = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[1]);
  const coverageError = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[2]);
  const lower = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[3]);
  const upper = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[4]);
  const widthIc = getMetricNumberFromStoredEvaluation(detail, BAND_METRICS[7]);

  if (empirical != null) {
    cards.push({
      id: "coverage",
      title: "실제 포함률",
      judgement: Math.abs(empirical - 0.7) <= 0.05 ? "통과" : "개선 필요",
      target: "70%",
      actual: formatMetric(empirical, "rate"),
      diff: formatSignedPctPoint((empirical - 0.7) * 100),
      description: empirical >= 0.65 ? "목표 포함률에 비교적 가깝습니다." : "실제 포함률이 목표보다 낮아 위험 범위를 충분히 덮지 못했습니다.",
      tone: Math.abs(empirical - 0.7) <= 0.05 ? "good" : "warn",
    });
  }
  if (coverageError != null) {
    cards.push({
      id: "coverage-error",
      title: "포함률 오차",
      judgement: coverageError <= 0.05 ? "통과" : "개선 필요",
      target: "5%p 이하",
      actual: formatMetric(coverageError, "pct_point"),
      diff: coverageError <= 0.05 ? "기준 안" : formatSignedPctPoint((coverageError - 0.05) * 100),
      description: coverageError <= 0.05 ? "목표 포함률과의 차이가 허용 범위에 있습니다." : "목표 포함률과 실제 포함률의 차이가 큽니다.",
      tone: coverageError <= 0.05 ? "good" : "warn",
    });
  }
  if (lower != null) {
    cards.push({
      id: "lower",
      title: "하단 이탈률",
      judgement: lower <= 0.15 ? "통과" : "개선 필요",
      target: "15% 근처",
      actual: formatMetric(lower, "rate"),
      diff: formatSignedPctPoint((lower - 0.15) * 100),
      description: lower <= 0.15 ? "하방 위험을 어느 정도 덮었습니다." : "하단 이탈률이 높아 하방 위험을 충분히 덮지 못했습니다.",
      tone: lower <= 0.15 ? "good" : "warn",
    });
  }
  if (upper != null) {
    cards.push({
      id: "upper",
      title: "상단 이탈률",
      judgement: upper <= 0.15 ? "통과" : "보통",
      target: "15% 근처",
      actual: formatMetric(upper, "rate"),
      diff: formatSignedPctPoint((upper - 0.15) * 100),
      description: upper <= 0.15 ? "상단 방향도 비교적 안정적으로 덮었습니다." : "상단 방향 이탈이 다소 많습니다.",
      tone: upper <= 0.15 ? "good" : "neutral",
    });
  }
  if (widthIc != null) {
    cards.push({
      id: "width-ic",
      title: "밴드 폭 반응도",
      judgement: widthIc > 0 ? "통과" : "개선 필요",
      target: "0보다 큼",
      actual: formatMetric(widthIc),
      diff: formatSignedNumber(widthIc),
      description: widthIc > 0 ? "실제 변동성이 큰 구간에서 밴드가 넓어지는 경향이 있습니다." : "변동성이 커지는 구간에 밴드가 충분히 반응하지 못했습니다.",
      tone: widthIc > 0 ? "good" : "warn",
    });
  }
  return cards;
}

function buildExperimentCards(detail: AiRunDetail) {
  return getExperimentKind(detail) === "band" ? buildBandExperimentCards(detail) : buildLineExperimentCards(detail);
}

function hasDisplayableExperimentMetrics(detail: AiRunDetail) {
  return buildExperimentCards(detail).length >= 2;
}

function getExperimentFailureReason(detail: AiRunDetail) {
  const kind = getExperimentKind(detail);
  if (kind === "band") {
    const empirical = getMetricNumber(detail, BAND_METRICS[1]);
    const lower = getMetricNumber(detail, BAND_METRICS[3]);
    const widthIc = getMetricNumber(detail, BAND_METRICS[7]);
    if (empirical != null && empirical < 0.65) {
      return "실제 포함률이 목표보다 낮아 위험 범위를 충분히 덮지 못했습니다.";
    }
    if (lower != null && lower > 0.15) {
      return "하단 이탈률이 높아 하방 위험을 충분히 덮지 못했습니다.";
    }
    if (widthIc != null && widthIc <= 0) {
      return "변동성이 커지는 구간에 밴드 폭이 충분히 반응하지 못했습니다.";
    }
    return "현재 제품 모델보다 설명력이나 안정성에서 우선순위가 낮아 제품 화면에는 쓰지 않습니다.";
  }

  const ic = getMetricNumber(detail, LINE_METRICS[0]);
  const falseSafe = getMetricNumber(detail, LINE_METRICS[2]);
  const recall = getMetricNumber(detail, LINE_METRICS[3]);
  if (ic != null && ic <= 0) {
    return "순위 상관이 약해 수익 방향을 안정적으로 구분하지 못했습니다.";
  }
  if (falseSafe != null && falseSafe > 0.25) {
    return "위험 오판율이 높아 위험 구간을 안전하다고 보는 경우가 많았습니다.";
  }
  if (recall != null && recall < 0.7) {
    return "큰 하락을 모두 잡아내기에는 아직 부족했습니다.";
  }
  return "현재 제품 모델보다 품질이나 해석 우선순위가 낮아 제품 화면에는 쓰지 않습니다.";
}

function describeWeaknesses(rows: ComparisonRow[]) {
  const weakRows = rows.filter((row) => row.result === "worse");
  if (weakRows.length === 0) {
    return "제품 모델보다 뚜렷하게 약한 비교 지표는 제한적입니다. 다만 현재 제품 모델이 더 안정적인 기준으로 쓰이고 있어 이 실험은 이전 실험으로 남겼습니다.";
  }
  return weakRows
    .slice(0, 3)
    .map((row) => `${row.label}: ${row.interpretation}`)
    .join(" ");
}

function describeStrengths(rows: ComparisonRow[]) {
  const betterRows = rows.filter((row) => row.result === "better");
  if (betterRows.length === 0) {
    return "제품 모델보다 더 좋았던 핵심 비교 지표는 확인되지 않았습니다.";
  }
  return betterRows
    .slice(0, 3)
    .map((row) => `${row.label}은 더 좋았습니다.`)
    .join(" ");
}

function getComparisonVerdictTag(detail: AiRunDetail, rows: ComparisonRow[], category: ExperimentCategory) {
  if (detail.timeframe === "1W") {
    return "제품 기준 미확정";
  }
  if (category === "quality_failed") {
    return "제품 후보 탈락";
  }
  if (rows.some((row) => row.result === "worse")) {
    return "이전 실험";
  }
  return "보류";
}

function getFinalJudgement(detail: AiRunDetail, rows: ComparisonRow[], category: ExperimentCategory) {
  if (detail.timeframe === "1W") {
    return "1W 보수적 기준선은 v1에서 제공하지 않지만 1W AI 밴드는 활성 (CP178 walk-forward lower calibration)입니다. 이 실험 결과를 현재 1W 제품 모델 대비 우열로 과장하지 않습니다.";
  }
  const kind = getExperimentKind(detail);
  const weakRows = rows.filter((row) => row.result === "worse");
  const betterRows = rows.filter((row) => row.result === "better");
  const roleText = kind === "band" ? "AI 밴드" : "보수적 기준선";
  if (weakRows.length > 0 && betterRows.length > 0) {
    return `${formatExperimentName(detail)}은 ${betterRows[0].label}에서는 제품 모델보다 나은 면이 있었지만, ${weakRows[0].label}에서 약해 ${roleText} 제품 모델로 쓰기 어렵습니다.`;
  }
  if (weakRows.length > 0) {
    return `${formatExperimentName(detail)}은 ${weakRows[0].label} 지표가 현재 제품 모델보다 약해 ${roleText} 제품 모델로 선택하지 않았습니다.`;
  }
  if (category === "quality_failed") {
    return "품질 기준을 통과하지 못해 제품 화면에는 쓰지 않습니다.";
  }
  return "일부 지표는 제품 모델과 비슷했지만, 현재 제품 모델을 대체할 만큼 명확한 우위가 확인되지 않아 이전 실험으로 남겼습니다.";
}

function ComparisonTable({ rows }: { rows: ComparisonRow[] }) {
  if (rows.length === 0) {
    return <div className="compact-note">제품 기준 미확정 상태입니다.</div>;
  }
  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>항목</th>
            <th>제품 모델</th>
            <th>이 실험</th>
            <th>차이</th>
            <th>해석</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.label}</td>
              <td>{row.productText}</td>
              <td>{row.experimentText}</td>
              <td>{row.diffText}</td>
              <td>{row.interpretation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExperimentDetail({
  detail,
  category,
  productDetail,
}: {
  detail: AiRunDetail | null;
  category: ExperimentCategory;
  productDetail: AiRunDetail | null;
}) {
  if (!detail) {
    return <div className="empty-state">실험을 선택하면 상세 설명을 표시합니다.</div>;
  }
  const role = getRunRole(detail);
  const metrics = role === "band_model" ? BAND_METRICS : LINE_METRICS;
  const experimentKind = getExperimentKind(detail);
  const changedFields = getChangedExperimentFields(detail);
  const comparisonRows = buildComparisonRows(detail, productDetail);
  const verdict = getComparisonVerdictTag(detail, comparisonRows, category);

  return (
    <div className="model-detail-stack">
      <div className="model-detail-hero">
        <span className="status-pill status-pill--muted">{verdict}</span>
        <h2>{formatExperimentName(detail)}</h2>
        <p>{getExperimentDescription(detail, category)}</p>
      </div>

      <section className="model-story-grid">
        <article>
          <h3>역할</h3>
          <p>{experimentKind === "band" ? "AI 밴드 실험입니다. 예상 변동 범위가 목표 비율에 맞게 실제 수익률을 덮는지 확인합니다." : "예측선 실험입니다. 수익 방향과 위험 구간을 얼마나 안정적으로 구분하는지 확인합니다."}</p>
        </article>
        <article>
          <h3>실험에서 바꾼 것</h3>
          <DataList items={changedFields} />
        </article>
      </section>

      <section className="model-story-grid">
        <article>
          <h3>제품 모델 대비 부족했던 점</h3>
          <p>{detail.timeframe === "1W" ? "1W 제품 기준이 아직 확정되지 않아 1D 제품 모델처럼 직접 비교하지 않습니다." : describeWeaknesses(comparisonRows)}</p>
        </article>
        <article>
          <h3>제품 모델 대비 좋았던 점</h3>
          <p>{detail.timeframe === "1W" ? "주간 제품 기준 확정 뒤 다시 평가할 수 있습니다." : describeStrengths(comparisonRows)}</p>
        </article>
      </section>

      <section>
        <div className="panel-heading panel-heading--compact">
          <h3>비교 지표</h3>
        </div>
        <ComparisonTable rows={comparisonRows} />
      </section>

      <section className="model-story-grid">
        <article>
          <h3>최종 판단</h3>
          <p>{getFinalJudgement(detail, comparisonRows, category)}</p>
        </article>
        <article>
          <h3>다음 확인 방향</h3>
          <p>{detail.timeframe === "1W" ? "1W 제품 기준이 확정된 뒤 같은 지표로 다시 비교합니다." : "실험 조건 상세는 상세 정보에서 확인하고, 제품 모델보다 나았던 지표를 다음 학습 조건에 반영합니다."}</p>
        </article>
      </section>
      <ModelRunDetails detail={detail} metricDefinitions={metrics} />
    </div>
  );
}

export default function TrainingView() {
  const [runs, setRuns] = useState<AiRunSummary[]>([]);
  const [failedQualityRuns, setFailedQualityRuns] = useState<AiRunSummary[]>([]);
  const [experimentDetails, setExperimentDetails] = useState<Record<string, AiRunDetail>>({});
  const [productLineDetail, setProductLineDetail] = useState<AiRunDetail | null>(null);
  const [productBandDetail, setProductBandDetail] = useState<AiRunDetail | null>(null);
  const [productWeeklyLineDetail, setProductWeeklyLineDetail] = useState<AiRunDetail | null>(null);
  const [selected, setSelected] = useState<SelectedItem>({ kind: "slot", slotId: "line-1d" });
  const [detail, setDetail] = useState<AiRunDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [runsApiError, setRunsApiError] = useState<ApiError | null>(null);
  const [detailApiError, setDetailApiError] = useState<ApiError | null>(null);

  const experimentGroups = useMemo(() => {
    const candidates = [
        ...runs
          .filter((run) => !PRODUCT_RUN_IDS.has(run.run_id))
          .filter((run) => !isLegacyRun(run))
          .map((run) => ({ run, category: "previous" as const })),
        ...failedQualityRuns
          .filter((run) => !isLegacyRun(run))
          .map((run) => ({ run, category: "quality_failed" as const })),
      ];
    const displayable = candidates
      .map((item) => {
        const itemDetail = experimentDetails[item.run.run_id];
        const kind = itemDetail ? getExperimentKind(itemDetail) : null;
        const productDetail = kind === "band" ? productBandDetail : productLineDetail;
        if (!itemDetail || !kind || !hasDisplayableComparison(itemDetail, productDetail)) {
          return null;
        }
        return {
          ...item,
          detail: itemDetail,
          kind,
          tag: getExperimentTag(item.run, item.category),
        };
      })
      .filter((item): item is ExperimentListItem => item != null);
    const seenExperimentNames = new Set<string>();
    const uniqueDisplayable = displayable.filter((item) => {
      const key = `${item.kind}-${item.category}-${formatExperimentName(item.detail)}`;
      if (seenExperimentNames.has(key)) {
        return false;
      }
      seenExperimentNames.add(key);
      return true;
    });
    return {
      line: uniqueDisplayable.filter((item) => item.kind === "line"),
      band: uniqueDisplayable.filter((item) => item.kind === "band"),
    };
  }, [runs, failedQualityRuns, experimentDetails, productLineDetail, productBandDetail]);

  async function loadDetail(selection: SelectedItem, runId: string | null) {
    setSelected(selection);
    setErrorMessage(null);
    if (selection.kind === "slot") {
      setDetail(null);
      return;
    }
    if (!runId) {
      setDetail(null);
      return;
    }
    setIsDetailLoading(true);
    setDetailApiError(null);
    try {
      const detailResponse = await fetchAiRun(runId, { includeConfig: false });
      setDetail(detailResponse.data);
    } catch (error) {
      setDetail(null);
      const classified = classifyApiError(error, `/api/v1/ai/runs/${runId}`);
      setDetailApiError(classified);
      setErrorMessage(describeApiError(classified));
    } finally {
      setIsDetailLoading(false);
    }
  }

  async function loadRuns() {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const [completedResult, failedQualityResult] = await Promise.allSettled([
        fetchAiRuns({ status: "completed", modelName: "", includeLegacy: true, limit: 100 }),
        fetchAiRuns({ status: "failed_quality_gate", modelName: "", includeLegacy: true, limit: 100 }),
      ]);
      const completedRuns = completedResult.status === "fulfilled" ? completedResult.value.data : [];
      const qualityRuns = failedQualityResult.status === "fulfilled" ? failedQualityResult.value.data : [];
      const filterModelRuns = (items: AiRunSummary[]) =>
        items.filter((run) => Boolean(getRunRole(run)) || TRAINING_RUN_MODELS.has(String(run.model_name ?? "")));
      const filteredCompletedRuns = filterModelRuns(completedRuns);
      const filteredQualityRuns = filterModelRuns(qualityRuns);
      const [productLineResult, productBandResult] = await Promise.allSettled([
        PRODUCT_LINE_1D_RUN_ID ? fetchAiRun(PRODUCT_LINE_1D_RUN_ID, { includeConfig: false }) : Promise.resolve(null),
        PRODUCT_BAND_1D_RUN_ID ? fetchAiRun(PRODUCT_BAND_1D_RUN_ID, { includeConfig: false }) : Promise.resolve(null),
      ]);
      const nextProductLineDetail = productLineResult.status === "fulfilled" ? productLineResult.value?.data ?? null : null;
      const nextProductBandDetail = productBandResult.status === "fulfilled" ? productBandResult.value?.data ?? null : null;
      const nextProductWeeklyLineDetail = null;
      const experimentCandidates = [...filteredCompletedRuns, ...filteredQualityRuns]
        .filter((run) => !PRODUCT_RUN_IDS.has(run.run_id))
        .filter((run) => !isLegacyRun(run));
      const detailResults = await Promise.allSettled(
        experimentCandidates.map(async (run) => {
          const response = await fetchAiRun(run.run_id, { includeConfig: false });
          return [run.run_id, response.data] as const;
        })
      );
      const nextExperimentDetails: Record<string, AiRunDetail> = {};
      detailResults.forEach((result) => {
        if (result.status === "fulfilled") {
          const [runId, runDetail] = result.value;
          if (hasDisplayableExperimentMetrics(runDetail)) {
            nextExperimentDetails[runId] = runDetail;
          }
        }
      });

      setRuns(filteredCompletedRuns);
      setFailedQualityRuns(filteredQualityRuns);
      setProductLineDetail(nextProductLineDetail);
      setProductBandDetail(nextProductBandDetail);
      setProductWeeklyLineDetail(nextProductWeeklyLineDetail);
      setExperimentDetails(nextExperimentDetails);
      if (PRODUCT_LINE_1D_RUN_ID) {
        await loadDetail({ kind: "slot", slotId: "line-1d" }, PRODUCT_LINE_1D_RUN_ID);
      }
    } catch (error) {
      setRuns([]);
      setFailedQualityRuns([]);
      setExperimentDetails({});
      setProductLineDetail(null);
      setProductBandDetail(null);
      setProductWeeklyLineDetail(null);
      setDetail(null);
      const classified = classifyApiError(error, "/api/v1/ai/runs");
      setRunsApiError(classified);
      setErrorMessage(describeApiError(classified));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadRuns();
  }, []);

  const selectedSlot = selected.kind === "slot" ? PRODUCT_SLOTS.find((slot) => slot.id === selected.slotId) ?? PRODUCT_SLOTS[0] : null;
  const selectedExperimentCategory = selected.kind === "experiment" ? selected.category : "previous";
  const selectedExperimentProductDetail = detail && getExperimentKind(detail) === "band" ? productBandDetail : productLineDetail;
  const getLoadedProductDetailForSlot = (slot: ProductSlot) => {
    if (slot.id === "line-1d") {
      return productLineDetail;
    }
    if (slot.id === "line-1w") {
      return productWeeklyLineDetail;
    }
    if (slot.id === "band-1d") {
      return productBandDetail;
    }
    return null;
  };

  return (
    <div className="view-stack">
      <header className="view-header">
        <div className="view-header__title">
          <h1>AI 모델</h1>
          <p>딥러닝 모델로 예측선과 AI 밴드를 만들고, 검증된 모델만 화면에 씁니다.</p>
        </div>
      </header>

      {runsApiError ? (
        <StatusInline kind="error" label="AI 실행 목록" error={runsApiError} hint="백엔드 /ai/runs 응답 또는 로그 확인" />
      ) : detailApiError ? (
        <StatusInline kind="error" label="실행 상세" error={detailApiError} hint="해당 run_id 응답 또는 백엔드 로그 확인" />
      ) : errorMessage ? (
        <div className="notice notice--error">{errorMessage}</div>
      ) : null}

      <section className="panel">
        <div className="panel-heading">
          <h2>사용한 딥러닝 모델</h2>
        </div>
        <div className="model-arch-grid">
          {MODEL_ARCHITECTURES.map((model) => (
            <article key={model.name} className="model-arch-card">
              <span className="model-arch-card__role">{model.role}</span>
              <h3>{model.name}</h3>
              <p className="model-arch-card__type">{model.type}</p>
              <p>{model.desc}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel model-status-panel">
        <div className="panel-heading">
          <h2>현재 사용 중인 모델</h2>
        </div>
        <div className="product-slot-grid">
          {PRODUCT_SLOTS.map((slot) => (
            <ProductSlotCard
              key={slot.id}
              slot={slot}
              status={getProductSlotStatus(slot, getLoadedProductDetailForSlot(slot), isLoading)}
              active={selected.kind === "slot" && selected.slotId === slot.id}
              onSelect={(slotId) => {
                const nextSlot = PRODUCT_SLOTS.find((item) => item.id === slotId);
                void loadDetail({ kind: "slot", slotId }, nextSlot?.runId ?? null);
              }}
            />
          ))}
        </div>
      </section>

      <section className="panel model-detail-panel">
        {isDetailLoading || isLoading ? (
          <div className="empty-state">AI 모델 정보를 불러오는 중입니다.</div>
        ) : selectedSlot?.kind === "line" ? (
          <LineModelDetail detail={detail} slot={selectedSlot} />
        ) : selectedSlot?.kind === "band" ? (
          <BandModelDetail detail={detail} slot={selectedSlot} />
        ) : selectedSlot ? (
          <PreparingSlotDetail slot={selectedSlot} />
        ) : (
          <ExperimentDetail detail={detail} category={selectedExperimentCategory} productDetail={selectedExperimentProductDetail} />
        )}
      </section>

      <section className="panel model-experiment-panel">
        <div className="panel-heading">
          <h2>이전 실험</h2>
        </div>
        <div className="experiment-disclosure-grid">
          {/* CP218 — 라인 실험 정적 타임라인 (1W 사이드 트랙 + 1D 메인 흐름). */}
          <details className="experiment-disclosure">
            <summary>예측선 실험 보기</summary>
            <div className="experiment-row-list">
              <ExperimentTimeline nodes={LINE_TIMELINE} kind="line" />
            </div>
          </details>
          {/* CP219 — 밴드 실험 정적 타임라인 (CP153 1D + CP178 1W + 공통 후속). */}
          <details className="experiment-disclosure">
            <summary>밴드 실험 보기</summary>
            <div className="experiment-row-list">
              <ExperimentTimeline nodes={BAND_TIMELINE} kind="band" />
            </div>
          </details>
        </div>
      </section>
    </div>
  );
}
