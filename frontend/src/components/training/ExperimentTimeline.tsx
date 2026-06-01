/**
 * CP218 — 라인/밴드 공용 정적 실험 타임라인 (압축 카드).
 * CP219 — timeframe 배지 추가 (1D / 1W / both). 라인/밴드 둘 다 표시 (1W 라인은 v1 deferred).
 *
 * 기본 상태: 한 줄 요약만 (CP + sub + 카테고리 + timeframe + 제목 + 날짜).
 * 클릭하면 무엇/왜/결과/교훈/보고서 ref 펼침.
 */

import type { ExperimentCategory, ExperimentNode } from "@/lib/training/lineTimeline";

export type ExperimentTimelineKind = "line" | "band";

interface ExperimentTimelineProps {
  nodes: ExperimentNode[];
  kind: ExperimentTimelineKind;
  emptyMessage?: string;
}

const CATEGORY_LABEL: Record<ExperimentCategory, string> = {
  baseline: "기준선",
  exploration: "탐색",
  failure: "실패",
  frozen: "고정 후보",
  shipped: "운영 진입",
  planning: "계획",
  analysis: "평가",
};

export default function ExperimentTimeline({ nodes, kind, emptyMessage }: ExperimentTimelineProps) {
  if (!nodes || nodes.length === 0) {
    return (
      <div className="experiment-timeline experiment-timeline--empty">
        <p className="compact-note">{emptyMessage ?? "표시할 실험 분기가 없습니다."}</p>
      </div>
    );
  }

  return (
    <ol className={`experiment-timeline experiment-timeline--${kind}`}>
      {nodes.map((node, index) => {
        const last = index === nodes.length - 1;
        const showTimeframe = !!node.timeframe;
        return (
          <li
            key={`${node.cp}-${node.sub ?? "main"}-${index}`}
            className={`experiment-timeline__item experiment-timeline__item--${node.category}${
              last ? " experiment-timeline__item--last" : ""
            }`}
          >
            <div className="experiment-timeline__rail" aria-hidden="true">
              <span className={`experiment-timeline__dot experiment-timeline__dot--${node.category}`} />
            </div>
            <details className="experiment-timeline__card">
              <summary className="experiment-timeline__summary">
                <span className="experiment-timeline__chevron" aria-hidden="true" />
                <strong className="experiment-timeline__cp">{node.cp}</strong>
                {node.sub ? <span className="experiment-timeline__sub">{node.sub}</span> : null}
                <span className={`experiment-timeline__badge experiment-timeline__badge--${node.category}`}>
                  {CATEGORY_LABEL[node.category]}
                </span>
                {showTimeframe ? (
                  <span className={`experiment-timeline__timeframe experiment-timeline__timeframe--${node.timeframe}`}>
                    {node.timeframe === "both" ? "1D + 1W" : node.timeframe}
                  </span>
                ) : null}
                <span className="experiment-timeline__heading">{node.title}</span>
              </summary>
              <div className="experiment-timeline__body-wrap">
                <dl className="experiment-timeline__body">
                  <div>
                    <dt>무엇</dt>
                    <dd>{node.what}</dd>
                  </div>
                  <div>
                    <dt>왜</dt>
                    <dd>{node.why}</dd>
                  </div>
                  <div>
                    <dt>결과</dt>
                    <dd>{node.result}</dd>
                  </div>
                  {node.lesson ? (
                    <div>
                      <dt>교훈</dt>
                      <dd>{node.lesson}</dd>
                    </div>
                  ) : null}
                </dl>
              </div>
            </details>
          </li>
        );
      })}
    </ol>
  );
}
