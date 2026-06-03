// 운영 슬롯의 "초기 계획 평가" + "초기 계획과 차이점" 두 섹션.
// slotId를 받아 staticEvaluation accessor에서 데이터를 꺼내 렌더한다.
// CP216 — UsageDataSection / ReproducibilitySection과 같은 slotId-accessor 패턴.

import { getPptMapping, getV1ExtraIndicators } from "@/lib/training/staticEvaluation";

export function PptMappingSection({ slotId }: { slotId: string | null | undefined }) {
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
export function V1ExtraIndicatorsSection({ slotId }: { slotId: string | null | undefined }) {
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
