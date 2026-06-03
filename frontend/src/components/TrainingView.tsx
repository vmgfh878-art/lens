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
  getStaticEvaluation,
  type StaticGoalCard,
} from "@/lib/training/staticEvaluation";
import type { GoalCardData, GoalCardProps } from "@/lib/training/cardTypes";
import {
  getMetricNumber,
  getMetricText,
  getProductMetricDefinitions,
  hasStoredEvaluationMetrics,
} from "@/lib/training/metricAccess";
import {
  buildBandExperimentCards,
  buildLineExperimentCards,
  hasDisplayableExperimentMetrics,
} from "@/lib/training/experimentBuilder";
import { PptMappingSection, V1ExtraIndicatorsSection } from "@/components/training/ProductMetricsSection";
import type { ComparisonRow } from "@/lib/training/comparisonTypes";
import { ComparisonTable } from "@/components/training/ComparisonTable";
import { SignificanceSection } from "@/components/training/SignificanceSection";
import { GoalCard, StoredEvaluationSection, staticCardsToGoalCardData } from "@/components/training/GoalCards";

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

interface ExperimentListItem {
  run: AiRunSummary;
  detail: AiRunDetail;
  category: ExperimentCategory;
  kind: ExperimentKind;
  tag: string;
}

// DetailField interface 는 @/lib/training/detailFields 로 이동했다.
// constants / formatters / runUtils 도 마찬가지.


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

/** CP217.2 — CP216.2 통계 검정 (학계 톤). 메인 8셀 · narrow 화면은 d/p/CI 접힘. 1W 만 GW regime sub-table. */
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
