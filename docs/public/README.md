# 공개 문서 읽기 순서

이 폴더는 원본 CP 보고서 전체를 공개하는 대신, 프로젝트 판단과 제품 방향을 설명하는 데 필요한 핵심 문서만 추려 둔 공개용 묶음입니다.

원본 CP 파일은 로컬 연구 기록에 가깝고, 이 폴더의 문서는 외부 독자가 프로젝트의 의도, 실패 판단, 제품 후보, 다음 계획을 빠르게 이해하도록 정리한 진입점입니다.

## 추천 순서

| 순서 | 문서 | 읽는 이유 |
|---|---|---|
| 1 | `line_v2_synthesis.md` | 1D line v2 계열이 왜 제품 후보가 되지 못했는지, 그리고 CP175 beta5가 왜 남았는지 |
| 2 | `line_beta5_candidate.md` | 현재 1D line 후보의 근거가 된 beta5 보수 학습 결과 |
| 3 | `stop_loss_line_v2_plan.md` | stop-loss line v2의 제품 의도와 평가 지표 설계 |
| 4 | `band_1d_save_run.md` | 1D band product candidate save-run 근거 |
| 5 | `band_baseline_comparison.md` | band가 통계 baseline 대비 어디까지 의미 있고 어디서 한계가 있는지 |
| 6 | `v1_save_run_status.md` | v1 package의 실제 준비 상태와 차단 항목 |
| 7 | `band_v2_plan.md` | Phase 2 band v2의 conformal, uncertainty, concept, selective output 계획 |
| 8 | `band_mlops_plan.md` | 기존 1D/1W band에 MLOps를 적용해 자동 재평가, 재학습, 승격 추천을 만드는 계획 |

## 공개 기준

- 판단 문서, 제품 계획, 확정 모델 후보와 직접 연결된 문서만 포함한다.
- 단순 실행 로그, 중간 실험 산출물, 임시 metrics, raw prediction, provider 수집 로그는 포함하지 않는다.
- 실패 문서도 숨기지 않되, 외부 독자가 해석 가능한 종합 문서 위주로 둔다.
- 원본 CP 번호보다 문서의 역할이 먼저 보이도록 공개용 파일명을 사용한다.
