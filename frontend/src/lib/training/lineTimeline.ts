/**
 * CP218 — 라인 v1 실험 정적 타임라인.
 *
 * 운영 라인 (CP210 F4 β=4 5-seed 앙상블) 까지 도달한 의미 있는 분기.
 * "일기장" 톤 — 실패/보류/부분 성공 정직 표기. 통계적으로 압도하지 못한 사실 (CP216.2) 도 마지막 노드에 인정.
 *
 * CP219 에서 type 을 ExperimentNode 로 일반화 — 라인/밴드 공유.
 *
 * 출처: docs/cp*_*.md 보고서 헤더 + 진행 노트.
 */

export type ExperimentCategory =
  | "baseline"      // 기준선 — 비교 기준 확립
  | "exploration"   // 탐색 — 변형 / 후보 비교
  | "failure"       // 실패 / 구조 결함 / 통계 미확인
  | "frozen"        // 고정 후보 — 운영 직전 잠금
  | "shipped"       // 운영 진입
  | "planning"      // 계획 정리 (v2 plan 류)
  | "analysis";     // 평가 / 통계 검정 / 사후 분석

/** 밴드 노드에서만 사용. 라인은 항상 1D 라 생략. */
export type ExperimentTimeframe = "1D" | "1W" | "both";

export interface ExperimentNode {
  /** CP 식별자 (예: "CP203"). 같은 CP 하위 단계면 sub 로 분해. */
  cp: string;
  /** 하위 단계 라벨 (예: "w4 / track_c2"). 옵션. */
  sub?: string;
  /** 한 줄 제목. */
  title: string;
  /** 카테고리 — 색/배지 분류. */
  category: ExperimentCategory;
  /** 밴드 1D/1W 구분 배지. 라인은 생략. */
  timeframe?: ExperimentTimeframe;
  /** 무엇을 시도했는지 (한두 줄). */
  what: string;
  /** 왜 시도했는지 (한두 줄). */
  why: string;
  /** 결과 한 줄 — 정직하게. 실패면 실패라고. */
  result: string;
  /** 다음으로 이어진 교훈 (옵션). */
  lesson?: string;
  /** docs/ 안 보고서 경로 (옵션, 표시만). */
  reportRef?: string;
}

// CP218 호환 alias — 다른 코드가 LineExperimentNode 로 import 한 경우.
export type LineExperimentCategory = ExperimentCategory;
export type LineExperimentNode = ExperimentNode;

export const LINE_TIMELINE: ExperimentNode[] = [
  // ────────── 100 ticker 시기 묶음 (라인 1D + 1W) ──────────
  {
    cp: "CP 67, 75, 146, 148",
    sub: "100 ticker 시기 1D 라인",
    title: "초기 1D 라인 — smoke 부터 500 ticker 전환까지",
    category: "exploration",
    timeframe: "1D",
    what:
      "CP67 h20 100 ticker 검증, CP75 1D h5 라인 첫 후보, CP148 phase 0~1 stage0_2, CP146 에서 S&P 500 500 ticker 첫 학습으로 전환.",
    why:
      "본격 500 ticker 학습 전에 작은 표본에서 모델 동작 + smoke 검증. 비용 통제하며 가설 빠르게 거르기.",
    result:
      "100 ticker 시기에 라인 기본 동작 / horizon / loss 가설 확립. CP146 시점부터 500 ticker 본격 학습. 이후 CP148 phase 2 부터가 운영 흐름의 출발.",
    lesson:
      "100 ticker 시기는 가설 거르기 용도. 운영 정체성 / 정량 수치는 500 ticker 시기 부터.",
    reportRef:
      "ai/cp67_lm_h20_100ticker_validation.py, cp75_lm_1d_h5_full_line_product_candidate.py, cp146_lm_eodhd500_line_full_training.py, cp148_lm_1d_stage0_2.py",
  },
  {
    cp: "CP 112, 113, 114, 136, 137, 139",
    sub: "100 ticker 시기 1W 라인",
    title: "1W 라인 사이드 트랙 — v1 deferred 로 종결",
    category: "failure",
    timeframe: "1W",
    what:
      "CP112 smoke → CP113 rescue → CP114 후보 확장 → CP136 horizon/feature 확장 → CP137 보수성 손실 연구 → CP139 product save 시도.",
    why:
      "1D 의 5거래일 도착가 정체성을 1W 4주 도착가로 이식 가능한지 점검. β=5 비대칭 손실도 1W 에 적용 가능성 시험.",
    result:
      "smoke 통과, 후보 확장 진행, β 효과는 1W 에서 약함. product save 시점에서 metric 이 1D 수준 미달 — 운영 채택 안 함. v1 line-1w slot 은 deferred. 1W 는 밴드만 운영.",
    lesson:
      "1W 라인은 v1 5거래일 도착가 정체성의 4주 판으로 가기엔 표본/coverage 부족. β 효과도 표본 수에 의존 — 1W 의 78~104 주봉으로는 1D 의 250+ 일봉만큼 안 잡힘. v2 에서 horizon 정의 자체 재검토 필요.",
    reportRef:
      "ai/cp112_lm_1w_line_smoke.py, cp113_lm_1w_line_rescue.py, cp114_lm_1w_line_candidate_expansion.py, cp136_lm_1w_line_horizon_feature_expansion.py, cp137_lm_1w_line_conservatism_loss_study.py, cp139_lm_1w_line_product_save.py",
  },

  // ────────── 500 ticker 시기 1D 라인 메인 흐름 (CP148~CP216) ──────────
  {
    cp: "CP148",
    sub: "LM 1D ph2 / stage3~4.5",
    title: "false-safe 탐색 + feature 세트 탐색 + seed 안정성",
    category: "baseline",
    timeframe: "1D",
    what:
      "1D 라인 LM 단계 별로 false-safe 재정렬, feature 세트 탐색, seed 안정성 검증.",
    why:
      "v1 직전 라인 모델의 기준선 확보 — false-safe / severe recall 우선 정책 탐색.",
    result:
      "false-safe 와 feature 세트 효과 정리. CP175 고정의 직전 기준선으로 사용.",
    lesson:
      "라인 학습은 정확도 단독보다 false-safe / severe recall 트레이드 오프가 핵심 — 이후 CP175 β=5 비대칭 손실 결정의 근거.",
    reportRef: "docs/cp148_lm_1d_ph2_0_cfix_training_procedure_report.md (+ stage3~4.5 시리즈)",
  },
  {
    cp: "CP154",
    sub: "LM 1D line v2 dual-head 계획",
    title: "라인 + 위험 동시 학습 (dual-head) 시도",
    category: "exploration",
    timeframe: "1D",
    what:
      "라인과 위험 신호를 동시에 학습하는 dual-head 구조. 사전 점검 → 기본 검증 → 종합 검증까지 진행.",
    why:
      "단일 head 모델보다 위험 신호를 같이 학습하면 라인 예측 calibration 이 좋아질 것이라는 가설.",
    result:
      "종합 검증까지 끝낸 뒤 v1 운영 시점에서는 보류. dual-head 가 단일 head 대비 명확한 우위 없음.",
    lesson:
      "v1 운영 라인은 single-head + 비대칭 손실 CP175 β=5 로 정착.",
    reportRef: "docs/cp154_175_lm_1d_line_v2_synthesis_report.md",
  },
  {
    cp: "CP164",
    sub: "LM calendar 분할 라인+위험 점검",
    title: "달력 정렬 분할로 잡은 alpha 기준선",
    category: "baseline",
    timeframe: "1D",
    what:
      "walk-forward 가 깨지지 않게 달력 정렬 분할로 라인+위험 동시 학습 기본 검증. feature overlay 비교.",
    why:
      "이전 random / ticker-leave-one-out 분할이 미래 정보 누설 가능성 있음 → 달력 강제.",
    result:
      "IC 0.044 / severe recall 0.67 / false-safe 0.206 — CP208Z 기준선 잠금의 alpha 비교축으로 박힘.",
    lesson:
      "분할 모드가 metric 신뢰성을 좌우. 이후 모든 라인 학습은 달력 정렬로 통일.",
    reportRef: "docs/cp164_lm_calendar_split_line_risk_smoke_report.md",
  },
  {
    cp: "CP175",
    sub: "보수적 라인 학습 재검토",
    title: "β=5 비대칭 손실 라인 고정 — v1 직전 기준선",
    category: "frozen",
    timeframe: "1D",
    what:
      "Asymmetric Huber β=5 — 낙관 오차 5배 페널티 — 로 라인 고정 후보화. ATR-only / β-only / β×ATR pareto 정리.",
    why:
      "라인이 가격을 보수적으로 underestimate 하도록 강제 — '리스크 인식 도구' 정체성과 일치. severe recall 0.79 목표.",
    result:
      "IC 0.042 / severe recall 0.79 / false-safe 0.197 — 운영 직전 고정으로 채택. 단 일부 ticker 분포 collapse 잔재.",
    lesson:
      "β=5 비대칭 손실이 라인 정체성을 잡았지만 분포 collapse 잔재. 후속에서 해소 시도.",
    reportRef:
      "docs/cp175_lm_1d_conservative_line_learning_revisit_report.md (+ pareto / collapse_diagnostic)",
  },
  {
    cp: "CP179",
    sub: "LM 1D line v3 기준선 잠금",
    title: "라인 v3 기준선 — feature 마감 + 누설 점검",
    category: "baseline",
    timeframe: "1D",
    what:
      "라인 v3 기준선에서 feature 마감, 누설 점검, 분포 가정 정리, 사전 점검 준비도까지.",
    why:
      "v3 진입 전 feature 누출 가능성 + 분포 가정 점검. v2 진입 길 닦기.",
    result:
      "feature 출처 매핑 / 누설 감사 통과. 분포 가정 정리. 라인 v3 기준선으로 잠금.",
    lesson:
      "feature 누출 점검을 기준선 단계에서 강제 — 이후 v1 운영 학습 모두 같은 점검 기준 통과.",
    reportRef: "docs/cp179_lm_1d_line_v3_baseline_lock_report.md (+ feature_*, leakage_audit)",
  },
  {
    cp: "CP186",
    sub: "미래 이벤트 커버리지",
    title: "미래 이벤트 (실적/FOMC/OPEX) feature + 누설 점검",
    category: "exploration",
    timeframe: "1D",
    what:
      "미래 달력 이벤트 feature 추가 → 커버리지 + 누설 동시 점검.",
    why:
      "라인 예측이 미래 이벤트 일 때 calibration 더 잘 잡힌다는 가설.",
    result:
      "커버리지 확보 + 누설 없는 형태로 feature 추가. v1 운영 feature 묶음에는 직접 안 들어갔으나 calendar_sin/cos 등 보존.",
    lesson:
      "미래 이벤트 직접 feature 는 v1 배포 결정에 결정적 영향 없었음.",
    reportRef: "docs/cp186_future_event_coverage_report.md (+ feature_setup, leakage_audit)",
  },
  {
    cp: "CP194",
    sub: "Warning stacking",
    title: "라인 + 보조 위험 신호 stacking 시도",
    category: "exploration",
    timeframe: "1D",
    what:
      "라인 기반 모델 위에 warning stacking (joint matrix, threshold 탐색, ablation) 으로 묶음 효과 측정.",
    why:
      "단일 라인 모델보다 warning stack 으로 false-safe 줄어들 것이라는 가설.",
    result:
      "후보 요약 정리. 운영 배포까지 가지는 못함.",
    lesson:
      "단순 stacking 으로 false-safe 가 의미 있게 줄지 않음. 묶음은 결국 CP210 의 5-seed 평균으로 정착.",
    reportRef: "docs/cp194_lm_1d_warning_stacking_report.md (+ resume_note)",
  },
  {
    cp: "CP196",
    sub: "1~4단계 분포 + h5/h20",
    title: "h5 vs h20 horizon 비교 — h5 채택",
    category: "exploration",
    timeframe: "1D",
    what:
      "분포 기반 target 정리 + h5 vs h20 horizon 비교. joint matrix / threshold 탐색 / 판정 보고서.",
    why:
      "5거래일 vs 20거래일 둘 중 어느 horizon 이 라인 정체성에 맞는지 정리.",
    result:
      "4단계에서 h5 vs h20 비교 후 h5 채택. 판정 보고서에 명시.",
    lesson:
      "h5 가 라인 운영 horizon — 사용자에게 '5거래일 후 도착가' 라는 한 줄 narrative 확정.",
    reportRef: "docs/cp196_stage3_judgment_report.md (+ stage1/2/4)",
  },
  {
    cp: "CP203",
    sub: "w2 / w3 / w4 / track_c2",
    title: "분포 갈라짐 (multi-modal collapse) → 구조 결함 인정",
    category: "failure",
    timeframe: "1D",
    what:
      "라인 출력 분포가 여러 봉우리로 갈라지는 현상 (collapse) 을 w2 → w3 → w4 → w4_final / track_c2 까지 추적.",
    why:
      "CP175 고정의 collapse 진단에서 본 분포 갈라짐 원인 + 완화책 찾기.",
    result:
      "구조 결함 — 단일 라인 모델 구조로는 완화 어려움. 기본 시범 / calibration 그림에 증상 보존.",
    lesson:
      "다음 단계로 — 앙상블 / multi-head / 다른 backbone 검토. 이게 CP208~CP210 분기의 시작.",
    reportRef:
      "docs/cp203_baseline_pilot_report.md (+ baseline_calibration_plots.png, w2/w3/w4 산출물)",
  },
  {
    cp: "CP208",
    sub: "라인 재구축 + 갱신 가능성",
    title: "라인 재구축 촉발 — 새 단계 진입",
    category: "exploration",
    timeframe: "1D",
    what:
      "라인 재구축 촉발 정의 + 갱신 가능성 점검 + 환경 잠금. F0~F8 feature 묶음 비교 시작.",
    why:
      "CP203 분포 갈라짐 구조 결함을 받아들이고, feature 묶음 + backbone 재정의로 새 단계 진입.",
    result:
      "재구축 metric / 갱신 가능성 보고서 / 환경 잠금 확정. F4 feature 묶음 후보화.",
    lesson:
      "라인 모델은 feature 묶음 정의가 metric 결정의 1차 lever. backbone 은 PatchTST 유지.",
    reportRef:
      "docs/cp208_line_rebuild_report.md (+ rebuild_trigger, refreshability, environment_lock)",
  },
  {
    cp: "CP208Z",
    sub: "feature 묶음 / 기준선 / 환경 잠금",
    title: "feature 묶음 35조합 비교 → F4 채택 확정",
    category: "exploration",
    timeframe: "1D",
    what:
      "백본 모델과 feature 묶음 조합 35가지를 같은 조건으로 학습 비교. CP208 에서 후보로 올린 F4 가 정말 최선인지 다른 묶음과 직접 비교.",
    why:
      "v1 운영 학습 직전 마지막 정리 — feature 묶음을 더 살펴봐도 F4 보다 나은 게 없는지 확인.",
    result:
      "F4 (stress delta + yield curve 묶음) 채택 확정. 학습 시점 데이터도 같이 잠가서 나중에 재현 가능하게 함. CP209 학습의 직접 출발점.",
    lesson:
      "feature 묶음과 학습 시점 데이터 둘 다 잠가야 같은 결과 재현 가능.",
    reportRef:
      "docs/cp208z_feature_pack_lock.md, cp208z_baseline_lock_report.md, cp208z_environment_lock.md",
  },
  {
    cp: "CP209",
    sub: "F4 β=4 / F6 β=7 5-seed",
    title: "5-seed 재학습 — 검증 + seed 안정성",
    category: "exploration",
    timeframe: "1D",
    what:
      "F4 β=4 / F6 β=7 5-seed (seeds 7, 13, 23, 42, 71) 학습 + 4 fold walk-forward 검증. CP208Z 단일 seed metric 보강.",
    why:
      "CP208Z F4 β=4 seed=7 단 한 케이스만 IC metric 저장 → false-safe / severe recall 부족. 5-seed 로 분산 확보.",
    result:
      "F4 β=4 5-seed 체크포인트 + 4 fold 검증 metric 확보. seed 안정성 표 별도. contract 수정 적용.",
    lesson:
      "단일 seed 결과는 배포 결정 근거로 부족 — 5-seed 분산 기준 도입. verification_metrics.json 운영 박음.",
    reportRef:
      "docs/cp209_verification_report.md, cp209_verification_metrics.json, cp209_seed_stability.csv",
  },
  {
    cp: "CP210",
    sub: "F4 β=4 5-seed 앙상블 검증",
    title: "앙상블 평균 배포 판정 — NO_SHIP (WARN) → 사용자 채택",
    category: "shipped",
    timeframe: "1D",
    what:
      "CP209 의 5-seed 체크포인트를 평균 앙상블로 묶어 forward 검증. 새 학습 없이 배포 판정.",
    why:
      "앙상블로 단일 seed 분산 줄이고, F4 잠금 결과의 운영 배포 결정.",
    result:
      "IC 0.0325 (통과), false-safe 0.205 (통과), severe recall 0.773 (통과), WF IC range 0.0457 (실패 — 기준 0.040). 종합 NO_SHIP. 사용자가 '보류 위험 인지하고 채택' 으로 운영 진입.",
    lesson:
      "WF IC range 가 fold 별 변동성 큰 신호. 앙상블로 줄였으나 기준 미달 — v2 에서는 fold 분산 자체를 학습 목표로 잡을지 검토.",
    reportRef: "docs/cp210_ensemble_report.md, docs/cp210_progress_latest.md",
  },
  {
    cp: "CP216",
    sub: "통계 유의성 검정 (라인)",
    title: "통계 기준선과 동등 — ML 우위 통계 미확인",
    category: "failure",
    timeframe: "1D",
    what:
      "운영 CP210 라인을 naive_zero / historical_mean / CP175 β=5 고정 기준선과 DM HAC + Bootstrap CI + GW conditional 로 비교. CP216.2 에서 walk-forward GARCH + 1W 500 ticker 보강.",
    why:
      "배포 판정 NO_SHIP (WARN) 의 정직성 — '운영 앙상블이 실제로 단순 기준선보다 통계 우위인가' 확인.",
    result:
      "historical_mean 대비 IC +0.008, DM p=0.22 (Bonferroni 1.00), Cohen's d=0.12 (negligible). CP175 β=5 고정 대비도 통계 동등. → 운영 앙상블 통계 우위 미확인.",
    lesson:
      "라인 자체 정확도 경쟁은 v2 에서 ROI 낮음. 다른 가치 (위험 인지 도구) 에 집중.",
    reportRef:
      "docs/cp216_significance/, docs/cp216_2_significance/",
  },
];
