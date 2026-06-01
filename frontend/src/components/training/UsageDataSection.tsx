/**
 * CP220 — 사용 데이터 섹션 (전문가용).
 * 메타 (feature_pack/set/version/target/provider/universe/parquet/data_hash) + features 표 + 전처리 리스트.
 * 단일 진리: docs/v1_operating_models_reproducibility.md (§1~3 데이터).
 */
import { getUsageData } from "@/lib/training/usageData";

interface Props {
  slotId: string | null | undefined;
}

export default function UsageDataSection({ slotId }: Props) {
  const block = getUsageData(slotId);
  if (!block) {
    return null;
  }
  return (
    <section className="usage-data">
      <div className="panel-heading panel-heading--compact">
        <h3>사용 데이터</h3>
        <span className="usage-data__src">v1 매니페스트 · 출처 <code>docs/v1_operating_models_reproducibility.md</code></span>
      </div>
      <dl className="usage-data__meta">
        <div>
          <dt>feature_pack</dt>
          <dd><code>{block.featurePack}</code></dd>
        </div>
        <div>
          <dt>feature_set</dt>
          <dd><code>{block.featureSet}</code></dd>
        </div>
        <div>
          <dt>feature_version</dt>
          <dd><code>{block.featureVersion}</code></dd>
        </div>
        <div>
          <dt>target</dt>
          <dd>{block.target}</dd>
        </div>
        <div>
          <dt>provider</dt>
          <dd>{block.provider}</dd>
        </div>
        <div>
          <dt>universe</dt>
          <dd>{block.universe}</dd>
        </div>
        <div>
          <dt>indicator parquet</dt>
          <dd><code>{block.parquetPath}</code></dd>
        </div>
        {block.dataHash ? (
          <div>
            <dt>source_data_hash</dt>
            <dd><code>{block.dataHash}</code></dd>
          </div>
        ) : null}
      </dl>
      <div className="usage-data__features-title">feature 컬럼 ({block.features.length})</div>
      <div className="usage-data__features" role="table">
        <div className="usage-data__feature-row usage-data__feature-row--head" role="row">
          <span role="columnheader">name</span>
          <span role="columnheader">설명</span>
          <span role="columnheader">coverage</span>
        </div>
        {block.features.map((f) => (
          <div key={f.name} className="usage-data__feature-row" role="row">
            <span role="cell"><code>{f.name}</code></span>
            <span role="cell">{f.description}</span>
            <span role="cell" className="usage-data__num">{f.coverage ?? "—"}</span>
          </div>
        ))}
      </div>
      <div className="usage-data__pre-title">전처리</div>
      <ul className="usage-data__pre">
        {block.preprocessing.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
    </section>
  );
}
