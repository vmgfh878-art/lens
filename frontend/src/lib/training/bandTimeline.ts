/**
 * CP219 — 밴드 v1 실험 정적 타임라인.
 *
 * 운영 밴드 두 갈래까지 도달한 의미 있는 분기:
 *   - 1D: CP153 stage0 → stage5T → primary save-run (TiDE q15/q85 + lower_focused calibration)
 *   - 1W: CP178 stage0 → stage5 → rescue/diag/calibration → WFLOCK
 * + 공통 후속: CP202.1 GARCH 비교, CP204 v2 plan, CP210 운영 자동화, CP216 / CP216.2 통계 검정.
 *
 * "일기장" 톤 유지. CP216.2 결론 (1W walk-forward GARCH 우위 d=-0.98 / 1D historical_quantile 못 이김)
 * 마지막 노드에 정직 박음.
 */

import type { ExperimentNode } from "./lineTimeline";

export const BAND_TIMELINE: ExperimentNode[] = [
  // ────────── 100 ticker 시기 묶음 (1D + 1W 밴드) ──────────
  {
    cp: "CP 106, 109",
    sub: "100 ticker 시기 1D 밴드",
    title: "초기 1D 밴드 — 위험 대응 + 라인과 결합 방식 후보 비교",
    category: "exploration",
    timeframe: "1D",
    what:
      "CP106 에서 밴드 위험 대응 후보들을 비교. CP109 에서 라인과 밴드를 어떻게 같이 쓸지 결합 방식을 비교. 작은 100 ticker 표본으로 가설만 거름.",
    why:
      "본격 TiDE 학습인 CP153 전에 밴드 가설과 라인-밴드 결합 방식을 추리기.",
    result:
      "후보 비교 정리. 라인과의 결합 방식이 아닌 밴드 단독 방향 채택 — CP153 의 첫 기준선으로 이어짐.",
    lesson:
      "100 ticker 시기엔 라인-밴드 결합 가설도 비교했으나 최종은 밴드 단독으로 결정. 본격 학습은 CP153 500 ticker 부터.",
    reportRef:
      "ai/cp106_band_risk_strategy_grid.py, cp109_line_band_balance_strategy_grid.py",
  },
  {
    cp: "CP 119, 120, 121, 124, 125, 138",
    sub: "100 ticker 시기 1W 밴드",
    title: "초기 1W 밴드 — feature / band_mode / 안정성 / calibration",
    category: "exploration",
    timeframe: "1W",
    what:
      "CP119 feature group, CP120 band_mode/target, CP121 안정성 검증, CP124 하방 loss guard, CP125 regime calibration, CP138 context backfill 재검증.",
    why:
      "CP178 stage 500 ticker 본격 학습 전에 1W 밴드의 feature / band_mode / loss / calibration 가설 거르기.",
    result:
      "feature 묶음 / band_mode = param / 하방 calibration 우선 정책 잠금. CP178 stage0 preflight 의 직전 baseline.",
    lesson:
      "1W 밴드의 약점은 표본 부족 + horizon 길이. CP178 에서도 동일 약점 반복 확인.",
    reportRef:
      "ai/cp119_bm_1w_band_feature_group_experiment.py, cp120_bm_1w_band_mode_target_experiment.py, cp121_bm_1w_band_verified_stability.py, cp124_bm_1w_band_loss_downside_guard.py, cp125_bm_1w_band_calibration_regime_eval.py, cp138_bm_1w_band_context_backfill_revalidation.py",
  },

  // ────────── 500 ticker 시기 1D 갈래 (CP153) ──────────
  {
    cp: "CP153",
    sub: "stage0_1 baseline",
    title: "feature_set / quantile / calibration 기준선 잠금",
    category: "baseline",
    timeframe: "1D",
    what:
      "1D 밴드 baseline 으로 feature 묶음 / 분위수 / calibration 첫 잠금. 비교 기준 확립.",
    why:
      "이후 단계의 변형 비교가 가능하려면 출발점 수치가 잠겨 있어야 함.",
    result:
      "baseline 수치 잠금. 다음 단계의 비교 anchor 가 됨.",
    reportRef: "docs/cp153_bm_1d_band_500_stage0_1_baseline_report.md",
  },
  {
    cp: "CP153",
    sub: "stage2 model zoo",
    title: "backbone 후보 비교 (TiDE / TCN / Transformer 변형)",
    category: "exploration",
    timeframe: "1D",
    what:
      "여러 후보 backbone 을 1D 밴드 동일 조건에서 비교. TiDE, TCN, Transformer 변형.",
    why:
      "분위수 밴드 모델의 best 구조 찾기 — 단일 모델 가설 의존 회피.",
    result:
      "TiDE 와 TCN 두 후보가 살아남음. 다음 단계에서 확장 비교.",
    reportRef: "docs/cp153_bm_1d_band_500_stage2_model_zoo_report.md",
  },
  {
    cp: "CP153",
    sub: "stage2.5 expansion",
    title: "TiDE / TCN 확장 — TiDE primary 방향",
    category: "exploration",
    timeframe: "1D",
    what:
      "stage2 통과한 TiDE / TCN 후보 공간 확장 비교 (s60/s120, q10q90/q15q85, head 변형).",
    why:
      "지나치게 빠른 잠금 회피 — 후보 공간을 한 번 더 펼쳐 확인.",
    result:
      "TiDE s60 q15/q85 후보 primary 방향. TCN 은 비교 보존.",
    reportRef:
      "docs/cp153_bm_1d_band_500_stage2_5_tide_tcn_expansion_report.md, stage2_5_to_5_final_report.md",
  },
  {
    cp: "CP153",
    sub: "stage3 calibration rescue",
    title: "conformal calibration 변형 — lower_focused 채택",
    category: "exploration",
    timeframe: "1D",
    what:
      "raw / symmetric / lower_focused 등 conformal calibration 변형 비교. coverage 정확도 측정.",
    why:
      "분위수 예측의 raw 출력은 coverage 가 nominal 과 어긋남 → 보정 필요. 어느 방식이 '리스크 인식 도구' 정체성에 맞나.",
    result:
      "lower_focused 채택 — 하방 위험 우선. lower_scale=1.05, upper_scale=1.0.",
    lesson:
      "calibration 정책이 정체성을 결정 — 대칭 보정 X, 하방 가중 보정 O. v1 운영 1D 밴드 calibration 의 출발.",
    reportRef: "docs/cp153_bm_1d_band_500_stage3_calibration_rescue_report.md",
  },
  {
    cp: "CP153",
    sub: "stage4 seed stability",
    title: "seed 7 / 42 / 123 안정성",
    category: "exploration",
    timeframe: "1D",
    what:
      "고정된 후보로 seed 3개 재학습 후 metric 분산 측정.",
    why:
      "단일 seed 결과가 운 일 가능성 — 분산으로 신뢰성 확보.",
    result:
      "seed 간 분산 허용 범위. 다음 단계 진행 자격 확보.",
    reportRef: "docs/cp153_bm_1d_band_500_stage4_seed_stability_report.md",
  },
  {
    cp: "CP153",
    sub: "stage4r reassessment",
    title: "seed/val/test 재평가 — val/test gap 점검",
    category: "exploration",
    timeframe: "1D",
    what:
      "stage4 결과를 val/test gap 관점에서 재평가. seed별 metric overfit 여부 점검.",
    why:
      "val 잘 보인 후보가 test 에서 무너지는 흔한 함정 회피.",
    result:
      "재평가 통과. stage5 진입 자격.",
    reportRef: "docs/cp153_bm_1d_band_500_stage4r_seed_val_test_reassessment_report.md",
  },
  {
    cp: "CP153",
    sub: "stage5 walk-forward replay",
    title: "replay 방식 walk-forward — 진짜 OOS 아님",
    category: "failure",
    timeframe: "1D",
    what:
      "기존 체크포인트를 fold 마다 재대입 replay 방식으로 walk-forward 흉내.",
    why:
      "fold 마다 fresh 학습은 비용 큼 → 우선 replay 로 baseline.",
    result:
      "같은 ckpt 재대입은 진짜 OOS 가 아님 — 학습 시점 데이터가 test 영역에 일부 누설. stage5T 로 다시 가야 함.",
    lesson:
      "walk-forward 는 fold 마다 fresh 학습이 정답. replay 는 baseline 비교용일 뿐.",
    reportRef: "docs/cp153_bm_1d_band_500_stage5_walk_forward_report.md",
  },
  {
    cp: "CP153",
    sub: "stage5T true walk-forward",
    title: "진짜 walk-forward — fold별 fresh 학습 + val 만 calibration",
    category: "shipped",
    timeframe: "1D",
    what:
      "3-fold 마다 fresh 학습 (seed 7/42/123). val 구간에서만 calibration fit, test 는 보정값 고정 적용.",
    why:
      "운영 진입 직전 진짜 OOS 평가가 필요. replay 의 누설 우려 해소.",
    result:
      "60일 lookback / q15/q85 후보 + 하방 우선 calibration 채택. coverage_abs_error 0.0099 / band_width_ic 0.376 / lower_breach_rate 0.158.",
    lesson:
      "stage5T 결과가 v1 운영 1D 밴드의 정체 — seed 42 fold_3 체크포인트가 그대로 운영에 박힘.",
    reportRef: "docs/cp153_bm_1d_band_500_stage5t_true_walk_forward_report.md",
  },
  {
    cp: "CP153",
    sub: "primary product save-run",
    title: "운영 candidate 최종 점검 + 운영 진입",
    category: "shipped",
    timeframe: "1D",
    what:
      "primary candidate save-run — DB attach 직전 점검. calibration 고정 적용 + 운영 parquet 박음.",
    why:
      "stage5T 의 fold_3 ckpt 가 운영 화면에 직접 붙기 전 sanity.",
    result:
      "gate 통과, 운영 채택. asof 2026-05-08 시점 잠금.",
    reportRef: "docs/cp153_bm_1d_band_primary_product_candidate_save_run_report.md",
  },

  // ────────── 500 ticker 시기 1W 갈래 (CP178) ──────────
  {
    cp: "CP178",
    sub: "stage0 preflight",
    title: "1W 데이터 충분성 + overfit 가드",
    category: "baseline",
    timeframe: "1W",
    what:
      "1W 표본이 일봉 대비 1/5 수준 — 학습 가능성/overfit 사전 점검.",
    why:
      "1W 모델이 표본 부족으로 학습 자체 안 되면 후속 단계 의미 없음.",
    result:
      "preflight 통과 — 후보 ticker 별 78 주봉 이상 sufficiency 만족.",
    lesson:
      "1W 는 sufficiency gate 가 빡빡함 — 운영 445 ticker, 일봉 474 대비 -29. 시작부터 표본 제약.",
    reportRef: "docs/cp178_bm_1w_band_500_stage0_preflight_report.md",
  },
  {
    cp: "CP178",
    sub: "stage1 baseline",
    title: "1W 기준선 잠금",
    category: "baseline",
    timeframe: "1W",
    what:
      "1W 밴드 baseline 수치 잠금 — 이후 단계의 비교 anchor.",
    why:
      "1D 와 비교 가능한 baseline 출발점 필요.",
    result:
      "baseline metric 잠금. 1W 는 1D 대비 coverage 오차 + band_width_ic 모두 약함을 baseline 단계에서 확인.",
    reportRef: "docs/cp178_bm_1w_band_500_stage1_baseline_report.md",
  },
  {
    cp: "CP178",
    sub: "stage2 model zoo 기본 검증",
    title: "1W backbone 기본 검증 — TiDE 후보화",
    category: "exploration",
    timeframe: "1W",
    what:
      "TiDE / TCN 1W 적용성 기본 검증. 1D 와 같은 backbone 이 1W 에서도 살아남나.",
    why:
      "1W 표본 부족 환경에서 큰 모델이 못 버틸 가능성.",
    result:
      "TiDE 후보화 — 1D 와 동일 backbone 유지. TCN 도 비교 보존.",
    reportRef: "docs/cp178_bm_1w_band_500_stage2_model_zoo_smoke_report.md",
  },
  {
    cp: "CP178",
    sub: "stage3 calibration rescue",
    title: "1W calibration 변형 — coverage 어려움 대응",
    category: "exploration",
    timeframe: "1W",
    what:
      "1W 분위수 출력의 coverage 가 nominal 과 크게 어긋남 → calibration 변형 비교.",
    why:
      "1W 는 1D 보다 표본 적음 + horizon 4주 → calibration 어려움.",
    result:
      "rescue 진행 — 하방 calibration 분리 시도. 후속 cal 단계의 출발.",
    reportRef: "docs/cp178_bm_1w_band_500_stage3_calibration_rescue_report.md",
  },
  {
    cp: "CP178",
    sub: "stage4 seed stability",
    title: "seed 7 / 42 / 123 1W 안정성",
    category: "exploration",
    timeframe: "1W",
    what:
      "1W 후보 seed 3개 재학습 후 metric 분산 측정.",
    why:
      "1W 표본 부족 환경에서 seed 분산이 더 클 우려.",
    result:
      "허용 범위 통과. stage5 진입.",
    reportRef: "docs/cp178_bm_1w_band_500_stage4_seed_stability_report.md",
  },
  {
    cp: "CP178",
    sub: "stage5 true walk-forward",
    title: "1W 진짜 walk-forward — 후보 metric 확보",
    category: "exploration",
    timeframe: "1W",
    what:
      "1W 도 1D 와 동일 calendar 의 3-fold true walk-forward.",
    why:
      "1W 도 운영 진입 직전 진짜 OOS 평가 필요.",
    result:
      "후보 metric 확보. coverage 는 여전히 1D 보다 약함 — rescue 단계 진입.",
    reportRef: "docs/cp178_bm_1w_band_500_stage5_true_walk_forward_report.md",
  },
  {
    cp: "CP178",
    sub: "rescue expansion",
    title: "1W 약점 보강 시도 — coverage 부족",
    category: "failure",
    timeframe: "1W",
    what:
      "1W coverage 부족 보강 — 후보 확장 (q 변경 / lookback 변경 / head 변형).",
    why:
      "1W 가 1D 와 같은 정체성을 가지려면 coverage 가 적정 nominal 에 가까워야.",
    result:
      "일부 회복. coverage 의 절대 수준은 1D 따라잡지 못함 — 구조적 한계 인정.",
    lesson:
      "1W 의 약점은 단순 변형으로 안 해결됨. 다른 calibration 정책 필요.",
    reportRef: "docs/cp178_bm_1w_band_500_rescue_expansion_report.md",
  },
  {
    cp: "CP178",
    sub: "diag rescue",
    title: "rescue 진단 — 효과 분석",
    category: "exploration",
    timeframe: "1W",
    what:
      "rescue expansion 결과의 ticker 별 / fold 별 효과 분해 분석.",
    why:
      "어디서 회복됐고 어디서 안 됐나 정량화 — 다음 단계 정의.",
    result:
      "ticker 별 효과 분포 파악. adaptive / lower 별도 calibration 분기 결정.",
    reportRef: "docs/cp178_diag_1w_band_rescue_report.md",
  },
  {
    cp: "CP178",
    sub: "adaptive calibration",
    title: "동적 coverage 보정 시도",
    category: "exploration",
    timeframe: "1W",
    what:
      "정적 conformal 대신 동적 adaptive calibration 적용.",
    why:
      "coverage 가 시간에 따라 변함 → 고정 보정값보다 동적이 나을 수 있다는 가설.",
    result:
      "효과 제한적. 1W 구조적 약점이 calibration 동적화로 안 풀림.",
    reportRef: "docs/cp178_alt_1w_band_adaptive_calibration_report.md",
  },
  {
    cp: "CP178",
    sub: "cal lower calibration",
    title: "하방 별도 calibration — 위험 우선",
    category: "exploration",
    timeframe: "1W",
    what:
      "상하 대칭 calibration 대신 하방만 별도 보정.",
    why:
      "1D 의 lower_focused 와 같은 철학 — '리스크 인식 도구' 정체성을 1W 에도 적용.",
    result:
      "효과 확인. WFLOCK 통합의 직전 단계.",
    reportRef: "docs/cp178_cal_1w_band_lower_calibration_report.md",
  },
  {
    cp: "CP178",
    sub: "WFLOCK",
    title: "walk-forward + 하방 calibration 잠금 — 운영 진입",
    category: "shipped",
    timeframe: "1W",
    what:
      "walk-forward 학습 + 하방 별도 calibration 을 한 묶음으로 잠금. 60일 lookback / q10/q90 후보 운영 채택.",
    why:
      "1W 의 구조적 약점을 인정하면서도 하방 위험 우선 정체성은 유지 — 그 절충점.",
    result:
      "coverage_abs_error 0.039 / band_width_ic 0.34 — 1D 대비 약하지만 운영 채택. 사용자가 1W 의 한계 인지하고 진입.",
    lesson:
      "1W 는 1D 만큼 정밀하지 않다는 사실을 정직 표기 + 그래도 표시는 함 — '주간 위험 범위' 라는 보조 narrative 로.",
    reportRef: "docs/cp178_wflock_1w_band_walk_forward_lower_report.md",
  },

  // ────────── 공통 후속 (1D + 1W 묶음) ──────────
  {
    cp: "CP202.1",
    title: "밴드 vs GARCH 점추정 비교 — narrative seed",
    category: "exploration",
    timeframe: "both",
    what:
      "운영 밴드를 GARCH 점추정과 일대일 비교. ticker / 시점 별 우열 분포.",
    why:
      "통계 베이스라인 (GARCH) 대비 우위가 어느 영역에 있나 — 정직한 평가.",
    result:
      "평상시는 우리 우위, stress 시 GARCH 우위 — selective output 의 narrative seed.",
    lesson:
      "regime 에 따라 우리 우위 / 열위 갈리는 게 처음 정량화됨. CP216 통계 검정으로 이어짐.",
    reportRef: "docs/cp202_*",
  },
  {
    cp: "CP204",
    title: "밴드 v2 계획 정리 — 미래 예측성 + 설명 가능성",
    category: "planning",
    timeframe: "both",
    what:
      "v1 평가 결과 밴드가 lagging — 과거 변동성 따라가는 수준에 머무름 — 임을 인정. v2 는 미래 예측성 강화를 1번 목표로 두고, 그 위에 보정 정확도 / 조건부 평가 / 선택적 출력 / 설명 가능 AI (XAI) 를 같이 얹는 계획.",
    why:
      "v1 밴드가 '변동성 커지기 전에 미리 넓어짐' 을 못 한다는 진단이 확인됨 (CP202.2 forward/backward 변동성 상관 0.34~0.38). v2 는 그 한계 자체를 풀어야 함.",
    result:
      "v2 트랙 정의 완료. 후속 v2 작업들의 출발점.",
    lesson:
      "v2 1번 목표는 미래 예측성을 살리는 것. 그 위에 정직성과 설명 가능성을 같이 얹어서 — '미리 보이고, 왜 그런지 설명되며, 모르면 침묵하는' 밴드.",
    reportRef: "docs/cp204_*",
  },
  {
    cp: "CP210",
    title: "밴드 refresh 자동화 + line export 트랙",
    category: "shipped",
    timeframe: "both",
    what:
      "운영 진입 후 자동 갱신 — 매일 refresh (1D) / 매주 refresh (1W) cron + 운영 parquet 자동 박음.",
    why:
      "정적 잠금된 모델도 매일 새 asof 가 들어와야 화면에 의미 있는 표시.",
    result:
      "refresh schedule 가동. CP153 1D / CP178 1W 운영 parquet 자동 갱신 + 라인 export 도 동일 트랙.",
    reportRef: "docs/cp210_band_refresh_*, cp210_progress_latest.md",
  },
  {
    cp: "CP216 / CP216.2",
    title: "통계 검정 — 1D 단순 분위수 못 이김, 1W walk-forward GARCH 우위",
    category: "analysis",
    timeframe: "both",
    what:
      "운영 1D / 1W 밴드를 4개 기준선 (bollinger / historical_quantile / GARCH 정적 fit / GARCH walk-forward) 과 DM(HAC) + Bootstrap CI + GW conditional 로 비교. CP216.2 에서 분위수 fix (1D q15/q85) + walk-forward GARCH + 1W 500 ticker 보강.",
    why:
      "운영 모델이 단순 통계 기준선보다 진짜로 통계적 우위인가 — 정직 평가.",
    result:
      "1D: historical_quantile 열위 d=0.93 large, walk-forward GARCH 열위 d=0.45 medium, bollinger 우위 d=-1.97. " +
      "1W: historical_quantile 열위 d=1.17 large, walk-forward GARCH 우위 d=-0.98 large 우리 쪽, bollinger 우위 d=-0.66. " +
      "1W 3 regime (vix_high / drawdown_low / combined) 모두 GW wald p<0.001 — regime-conditional 차이 systematic.",
    lesson:
      "라인 자체 정확도 경쟁은 v2 에서 ROI 낮음. 반면 밴드는 v2 1번 목표가 미래 예측성 강화 — v1 의 lagging 한계 (CP202.2 forward/backward 변동성 상관 0.34~0.38) 를 직접 풀고, 그 위에 selective output / 설명 가능성을 얹는 방향.",
    reportRef:
      "docs/cp216_significance/, docs/cp216_2_significance/, baselines parquet 11 개",
  },
];
