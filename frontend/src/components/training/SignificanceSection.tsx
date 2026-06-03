// "통계 검정" 섹션. CP216.2 — DM · Bootstrap CI · GW regime test 결과 표.
// slotId로 staticEvaluation에서 block을 꺼내 헤드라인 / findings / 상세 표 / footnote /
// GW regime sub-table / GW interpretation 까지 한 화면에 렌더한다.

import { getStaticSignificance } from "@/lib/training/staticEvaluation";

export function SignificanceSection({ slotId }: { slotId: string | null | undefined }) {
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
