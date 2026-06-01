/**
 * CP220 — 재현 매니페스트 섹션 (전문가용 · 하단 아코디언).
 * 환경 (Python/torch/GPU) + 패키지 표 + fold 윈도우 + 산출물 (깃 / 외부) + 학습 재현 명령.
 * 단일 진리: docs/v1_operating_models_reproducibility.md (§0 환경 + §1~3 모델별).
 * TODO 항목 폐기 (재현성/데이터 설명에 불필요).
 */
import { getReproducibility } from "@/lib/training/reproducibility";

interface Props {
  slotId: string | null | undefined;
}

export default function ReproducibilitySection({ slotId }: Props) {
  const block = getReproducibility(slotId);
  if (!block) {
    return null;
  }
  return (
    <details className="reproducibility">
      <summary>
        <span className="reproducibility__summary-title">재현 매니페스트</span>
        <span className="reproducibility__summary-meta">
          {block.backbone} · runId <code>{block.runId}</code> · seeds [{block.seeds.join(", ")}]
        </span>
      </summary>
      <div className="reproducibility__body">
        <div className="reproducibility__src">
          v1 매니페스트 · 출처 <code>docs/v1_operating_models_reproducibility.md</code>
        </div>

        <div className="reproducibility__block-title">모델 식별</div>
        <dl className="reproducibility__meta">
          <div><dt>run_id</dt><dd><code>{block.runId}</code></dd></div>
          <div><dt>source_cp</dt><dd><code>{block.sourceCp}</code></dd></div>
          <div><dt>backbone</dt><dd>{block.backbone}</dd></div>
          <div><dt>output_contract</dt><dd>{block.outputContract}</dd></div>
          {block.calibration ? (
            <div><dt>calibration</dt><dd>{block.calibration}</dd></div>
          ) : null}
          <div><dt>seeds</dt><dd><code>[{block.seeds.join(", ")}]</code></dd></div>
          <div><dt>serving parquet</dt><dd><code>{block.servingParquetPath}</code></dd></div>
        </dl>

        <div className="reproducibility__block-title">학습/검증/테스트 윈도우</div>
        <div className="reproducibility__fold" role="table">
          <div className="reproducibility__fold-row reproducibility__fold-row--head" role="row">
            <span role="columnheader">fold</span>
            <span role="columnheader">train_start</span>
            <span role="columnheader">train_end</span>
            <span role="columnheader">val_start</span>
            <span role="columnheader">val_end</span>
            <span role="columnheader">test_start</span>
            <span role="columnheader">test_end</span>
          </div>
          {block.folds.map((f) => (
            <div key={f.foldId} className="reproducibility__fold-row" role="row">
              <span role="cell"><strong>{f.foldId}</strong></span>
              <span role="cell">{f.trainStart ?? "—"}</span>
              <span role="cell">{f.trainEnd}</span>
              <span role="cell">{f.valStart ?? "—"}</span>
              <span role="cell">{f.valEnd ?? "—"}</span>
              <span role="cell">{f.testStart}</span>
              <span role="cell">{f.testEnd}</span>
            </div>
          ))}
        </div>

        <div className="reproducibility__block-title">환경</div>
        <dl className="reproducibility__meta">
          <div><dt>Python</dt><dd><code>{block.pythonVersion}</code></dd></div>
          <div><dt>torch</dt><dd><code>{block.torchVersion}</code></dd></div>
          <div><dt>GPU</dt><dd>{block.gpuName}</dd></div>
          <div><dt>arch</dt><dd><code>{block.gpuArch}</code></dd></div>
          <div><dt>CUDA runtime</dt><dd><code>{block.cudaRuntime}</code></dd></div>
        </dl>
        <ul className="reproducibility__env-list">
          {block.gpuEnv.map((e, i) => (
            <li key={i}><code>{e}</code></li>
          ))}
        </ul>

        <div className="reproducibility__block-title">핵심 패키지</div>
        <div className="reproducibility__pkg" role="table">
          <div className="reproducibility__pkg-row reproducibility__pkg-row--head" role="row">
            <span role="columnheader">package</span>
            <span role="columnheader">version</span>
          </div>
          {block.keyPackages.map((p) => (
            <div key={p.name} className="reproducibility__pkg-row" role="row">
              <span role="cell"><code>{p.name}</code></span>
              <span role="cell"><code>{p.version}</code></span>
            </div>
          ))}
        </div>

        <div className="reproducibility__block-title">산출물 — 깃 안에 있음</div>
        <ul className="reproducibility__list">
          {block.artifactsInGit.map((p, i) => (
            <li key={i}><code>{p}</code></li>
          ))}
        </ul>

        <div className="reproducibility__block-title">산출물 — 외부 패키지 (드롭박스)</div>
        <ul className="reproducibility__list reproducibility__list--external">
          {block.artifactsExternal.map((p, i) => (
            <li key={i}><code>{p}</code></li>
          ))}
        </ul>

        <div className="reproducibility__block-title">학습 재현 — 한 줄 wrapper · GPU sm_120 필수</div>
        <ol className="reproducibility__steps">
          {block.trainingSteps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      </div>
    </details>
  );
}
