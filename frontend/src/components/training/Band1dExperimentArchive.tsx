/**
 * CP220 — 1D Band (CP153 TiDE) 전용 실험 매니페스트 아카이브.
 * 위치: SignificanceSection 다음. 기본 접힘 (details).
 * 사용 데이터 (feature pack / 전처리) + 재현 매니페스트 (환경 / fold / 산출물 / 절차 / TODO) 한 덩어리.
 *
 * 단일 진리: docs/v1_operating_models_reproducibility.md §0 (환경) + §2 (1D 밴드).
 */
import { USAGE_DATA } from "@/lib/training/usageData";
import { REPRODUCIBILITY } from "@/lib/training/reproducibility";

export default function Band1dExperimentArchive() {
  const usage = USAGE_DATA["band-1d"];
  const repro = REPRODUCIBILITY["band-1d"];
  if (!usage || !repro) {
    return null;
  }
  return (
    <details className="experiment-archive">
      <summary>
        <span className="experiment-archive__title">실험 매니페스트 — 1D Band · CP153</span>
        <span className="experiment-archive__meta-line">
          {repro.backbone} · run_id <code>{repro.runId}</code> · seeds [{repro.seeds.join(", ")}] · {repro.folds.length}-fold walk-forward · calibration {repro.calibration}
        </span>
      </summary>
      <div className="experiment-archive__body">
        {/* ─────────────────────────── 사용 데이터 ─────────────────────────── */}
        <section className="experiment-archive__group">
          <h4 className="experiment-archive__group-title">사용 데이터 · feature pack</h4>
          <dl className="experiment-archive__dl">
            <div><dt>feature_pack</dt><dd><code>{usage.featurePack}</code></dd></div>
            <div><dt>feature_set</dt><dd><code>{usage.featureSet}</code></dd></div>
            <div><dt>feature_version</dt><dd><code>{usage.featureVersion}</code></dd></div>
            <div><dt>target</dt><dd>{usage.target}</dd></div>
            <div><dt>provider</dt><dd>{usage.provider}</dd></div>
            <div><dt>universe</dt><dd>{usage.universe}</dd></div>
            <div><dt>indicator parquet</dt><dd><code>{usage.parquetPath}</code></dd></div>
            {usage.dataHash ? (
              <div><dt>source_data_hash</dt><dd><code>{usage.dataHash}</code></dd></div>
            ) : null}
          </dl>

          <div className="experiment-archive__block-title">feature 컬럼 · {usage.features.length}개</div>
          <div className="experiment-archive__features" role="table">
            <div className="experiment-archive__feature-row experiment-archive__feature-row--head" role="row">
              <span role="columnheader">name</span>
              <span role="columnheader">설명</span>
              <span role="columnheader">coverage</span>
            </div>
            {usage.features.map((f) => (
              <div key={f.name} className="experiment-archive__feature-row" role="row">
                <span role="cell"><code>{f.name}</code></span>
                <span role="cell">{f.description}</span>
                <span role="cell" className="experiment-archive__num">{f.coverage ?? "—"}</span>
              </div>
            ))}
          </div>

          <div className="experiment-archive__block-title">전처리</div>
          <ul className="experiment-archive__list">
            {usage.preprocessing.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </section>

        {/* ─────────────────────────── 재현 매니페스트 ─────────────────────────── */}
        <section className="experiment-archive__group">
          <h4 className="experiment-archive__group-title">재현 매니페스트</h4>

          <div className="experiment-archive__block-title">모델 식별</div>
          <dl className="experiment-archive__dl">
            <div><dt>run_id</dt><dd><code>{repro.runId}</code></dd></div>
            <div><dt>source_cp</dt><dd><code>{repro.sourceCp}</code></dd></div>
            <div><dt>backbone</dt><dd>{repro.backbone}</dd></div>
            <div><dt>output_contract</dt><dd>{repro.outputContract}</dd></div>
            {repro.calibration ? (
              <div><dt>calibration</dt><dd>{repro.calibration}</dd></div>
            ) : null}
            <div><dt>seeds</dt><dd><code>[{repro.seeds.join(", ")}]</code></dd></div>
            <div><dt>serving parquet</dt><dd><code>{repro.servingParquetPath}</code></dd></div>
          </dl>

          <div className="experiment-archive__block-title">학습/검증/테스트 윈도우 · walk-forward {repro.folds.length}-fold</div>
          <div className="experiment-archive__fold" role="table">
            <div className="experiment-archive__fold-row experiment-archive__fold-row--head" role="row">
              <span role="columnheader">fold</span>
              <span role="columnheader">train_start</span>
              <span role="columnheader">train_end</span>
              <span role="columnheader">val_start</span>
              <span role="columnheader">val_end</span>
              <span role="columnheader">test_start</span>
              <span role="columnheader">test_end</span>
            </div>
            {repro.folds.map((f) => (
              <div key={f.foldId} className="experiment-archive__fold-row" role="row">
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

          <div className="experiment-archive__block-title">환경</div>
          <dl className="experiment-archive__dl">
            <div><dt>Python</dt><dd><code>{repro.pythonVersion}</code></dd></div>
            <div><dt>torch</dt><dd><code>{repro.torchVersion}</code></dd></div>
            <div><dt>GPU</dt><dd>{repro.gpuName}</dd></div>
            <div><dt>arch</dt><dd><code>{repro.gpuArch}</code></dd></div>
            <div><dt>CUDA runtime</dt><dd><code>{repro.cudaRuntime}</code></dd></div>
          </dl>
          <ul className="experiment-archive__list">
            {repro.gpuEnv.map((e, i) => (
              <li key={i}><code>{e}</code></li>
            ))}
          </ul>

          <div className="experiment-archive__block-title">핵심 패키지</div>
          <div className="experiment-archive__pkg" role="table">
            <div className="experiment-archive__pkg-row experiment-archive__pkg-row--head" role="row">
              <span role="columnheader">package</span>
              <span role="columnheader">version</span>
            </div>
            {repro.keyPackages.map((p) => (
              <div key={p.name} className="experiment-archive__pkg-row" role="row">
                <span role="cell"><code>{p.name}</code></span>
                <span role="cell"><code>{p.version}</code></span>
              </div>
            ))}
          </div>

          <div className="experiment-archive__block-title">산출물 — 깃 안에 있음</div>
          <ul className="experiment-archive__list">
            {repro.artifactsInGit.map((p, i) => (
              <li key={i}><code>{p}</code></li>
            ))}
          </ul>

          <div className="experiment-archive__block-title">산출물 — 외부 패키지 (드롭박스)</div>
          <ul className="experiment-archive__list experiment-archive__list--external">
            {repro.artifactsExternal.map((p, i) => (
              <li key={i}><code>{p}</code></li>
            ))}
          </ul>

          <div className="experiment-archive__block-title">학습 재현 — GPU sm_120 필수</div>
          <ol className="experiment-archive__steps">
            {repro.trainingSteps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </section>
      </div>
    </details>
  );
}
