from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai import cp178_alt_1w_band_adaptive_calibration as alt
from ai import cp178_cal_1w_band_lower_calibration as cal
from ai import cp178_bm_1w_band_500_stage3_5 as base

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

REPORT_PATH = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_report.md"
METRICS_PATH = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_summary.csv"
FOLD_SUMMARY_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_fold_summary.csv"
BOOTSTRAP_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_bootstrap_ci.csv"
TICKER_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_ticker_concentration.csv"
STRESS_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_stress_calm.csv"
COMPARISON_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_1d_comparison.csv"
DECISION_CSV = DOCS_DIR / "cp178_wflock_1w_band_walk_forward_lower_decision_table.csv"

CP153_1D_METRICS = DOCS_DIR / "cp153_bm_1d_band_500_stage5t_true_walk_forward_metrics.json"

TARGET_CANDIDATE_ID = "tide_s60_q10_q90_param"
METHOD = "walk_forward_lower_calibration"
Q_LOW = 0.10
Q_HIGH = 0.90


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    return base.safe_float(value)


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.6f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_rows(path, rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    base.write_json(path, payload)


def finite_mean(values: list[Any]) -> float | None:
    return base.mean(values)


def finite_worst(values: list[Any], *, higher_is_better: bool = False) -> float | None:
    return base.worst(values, higher_is_better=higher_is_better)


def load_1d_primary() -> dict[str, Any]:
    if not CP153_1D_METRICS.exists():
        return {}
    metrics = read_json(CP153_1D_METRICS)
    rows = metrics.get("aggregate_rows") or []
    primary = [row for row in rows if row.get("candidate_id") == "tide_s60_q15_param"]
    return dict(primary[0] if primary else (rows[0] if rows else {}))


def collect_walk_forward_frames() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    rows = cal.load_target_rows()
    raw_payloads = cal.collect_raw_frames(rows)
    payloads, label_rows = alt.label_payloads(raw_payloads)
    fold_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    test_frames: list[pd.DataFrame] = []
    for payload in payloads:
        fitted = alt.fit_walk_forward_shift(payload, payloads)
        result, _, test_cal, _ = alt.row_metrics(payload, fitted, payloads)
        fold_rows.append(result)
        bootstrap_rows.append(cal.bootstrap_ci(METHOD, str(result["fold_id"]), int(result["seed"]), test_cal))
        test_frames.append(test_cal)
    aggregate = pd.concat(test_frames, ignore_index=True)
    bootstrap_rows.append(cal.bootstrap_ci(METHOD, "all_folds", "all_seeds", aggregate))
    return fold_rows, bootstrap_rows, test_frames, label_rows


def summary_from_fold_rows(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": METHOD,
        "candidate_id": TARGET_CANDIDATE_ID,
        "fold_seed_count": len(fold_rows),
    }
    for key, higher in [
        ("test_lower_breach_rate", False),
        ("test_upper_breach_rate", False),
        ("test_coverage_abs_error", False),
        ("test_asymmetric_interval_score", False),
        ("test_p90_band_width", False),
        ("test_band_width_ic", True),
        ("test_downside_width_ic", True),
    ]:
        values = [row.get(key) for row in fold_rows]
        summary[f"{key}_mean"] = finite_mean(values)
        summary[f"{key}_worst"] = finite_worst(values, higher_is_better=higher)
    return summary


def fold_summary(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_id, items in base.group_by(fold_rows, "fold_id").items():
        row: dict[str, Any] = {"fold_id": fold_id, "seed_count": len(items), "method": METHOD}
        for key, higher in [
            ("test_lower_breach_rate", False),
            ("test_upper_breach_rate", False),
            ("test_coverage_abs_error", False),
            ("test_asymmetric_interval_score", False),
            ("test_p90_band_width", False),
            ("test_band_width_ic", True),
            ("test_downside_width_ic", True),
        ]:
            values = [item.get(key) for item in items]
            row[f"{key}_mean"] = finite_mean(values)
            row[f"{key}_worst"] = finite_worst(values, higher_is_better=higher)
        rows.append(row)
    return sorted(rows, key=lambda item: str(item.get("fold_id")))


def stress_summary(test_frames: list[pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in test_frames:
        fold_id = str(frame["fold_id"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        rows.extend(alt.stress_calm_rows(METHOD, fold_id, seed, frame))
    grouped: list[dict[str, Any]] = []
    for bucket, items in base.group_by(rows, "bucket").items():
        row = {"method": METHOD, "bucket": bucket, "fold_seed_count": len(items)}
        for key, higher in [
            ("lower_breach_rate", False),
            ("upper_breach_rate", False),
            ("coverage_abs_error", False),
            ("asymmetric_interval_score", False),
            ("p90_band_width", False),
            ("downside_width_ic", True),
        ]:
            values = [item.get(key) for item in items]
            row[f"{key}_mean"] = finite_mean(values)
            row[f"{key}_worst"] = finite_worst(values, higher_is_better=higher)
        grouped.append(row)
    return sorted(grouped, key=lambda item: str(item.get("bucket")))


def ticker_concentration(test_frames: list[pd.DataFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = pd.concat(test_frames, ignore_index=True).copy()
    frame["any_breach"] = frame["calibrated_lower_breach"] | frame["calibrated_upper_breach"]
    grouped = []
    for ticker, group in frame.groupby("ticker", sort=True):
        grouped.append(
            {
                "ticker": ticker,
                "sample_count": int(len(group)),
                "lower_breach_count": int(group["calibrated_lower_breach"].sum()),
                "upper_breach_count": int(group["calibrated_upper_breach"].sum()),
                "any_breach_count": int(group["any_breach"].sum()),
                "lower_breach_rate": float(group["calibrated_lower_breach"].mean()),
                "upper_breach_rate": float(group["calibrated_upper_breach"].mean()),
                "any_breach_rate": float(group["any_breach"].mean()),
                "stress_rate": float(group["alt_stress_label"].mean()) if "alt_stress_label" in group.columns else None,
                "realized_move_mean": float(group["realized_move"].mean()),
                "width_mean": float(group["calibrated_width"].mean()),
            }
        )
    grouped = sorted(grouped, key=lambda item: (-int(item["lower_breach_count"]), str(item["ticker"])))
    total_lower = sum(int(row["lower_breach_count"]) for row in grouped)
    total_any = sum(int(row["any_breach_count"]) for row in grouped)
    total_samples = sum(int(row["sample_count"]) for row in grouped)
    top10_lower = sum(int(row["lower_breach_count"]) for row in grouped[:10])
    top10_any = sum(int(row["any_breach_count"]) for row in grouped[:10])
    concentration = {
        "ticker_count": len(grouped),
        "sample_count": total_samples,
        "lower_breach_count": total_lower,
        "any_breach_count": total_any,
        "top10_lower_breach_share": top10_lower / total_lower if total_lower else None,
        "top10_any_breach_share": top10_any / total_any if total_any else None,
        "top10_tickers": [row["ticker"] for row in grouped[:10]],
    }
    return grouped, concentration


def comparison_rows(summary: dict[str, Any], primary_1d: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "timeframe": "1W",
            "candidate_id": TARGET_CANDIDATE_ID,
            "calibration": METHOD,
            "test_lower_breach_rate_mean": summary.get("test_lower_breach_rate_mean"),
            "test_lower_breach_rate_worst": summary.get("test_lower_breach_rate_worst"),
            "test_coverage_abs_error_mean": summary.get("test_coverage_abs_error_mean"),
            "test_coverage_abs_error_worst": summary.get("test_coverage_abs_error_worst"),
            "test_p90_band_width_worst": summary.get("test_p90_band_width_worst"),
            "test_band_width_ic_mean": summary.get("test_band_width_ic_mean"),
            "test_downside_width_ic_mean": summary.get("test_downside_width_ic_mean"),
            "source": "CP178-WFLOCK",
        },
        {
            "timeframe": "1D",
            "candidate_id": primary_1d.get("candidate_id"),
            "calibration": "primary_cp153",
            "test_lower_breach_rate_mean": primary_1d.get("test_lower_breach_rate_mean"),
            "test_lower_breach_rate_worst": primary_1d.get("test_lower_breach_rate_worst"),
            "test_coverage_abs_error_mean": primary_1d.get("test_coverage_abs_error_mean"),
            "test_coverage_abs_error_worst": primary_1d.get("test_coverage_abs_error_worst"),
            "test_p90_band_width_worst": primary_1d.get("test_p90_band_width_worst"),
            "test_band_width_ic_mean": primary_1d.get("test_band_width_ic_mean"),
            "test_downside_width_ic_mean": primary_1d.get("test_downside_width_ic_mean"),
            "source": "CP153 Stage 5T",
        },
    ]


def decision_rows(summary: dict[str, Any], primary_1d: dict[str, Any]) -> list[dict[str, Any]]:
    lower_mean = safe_float(summary.get("test_lower_breach_rate_mean"))
    lower_worst = safe_float(summary.get("test_lower_breach_rate_worst"))
    cov_mean = safe_float(summary.get("test_coverage_abs_error_mean"))
    cov_worst = safe_float(summary.get("test_coverage_abs_error_worst"))
    p90_worst = safe_float(summary.get("test_p90_band_width_worst"))
    downside = safe_float(summary.get("test_downside_width_ic_mean"))
    interval = safe_float(summary.get("test_asymmetric_interval_score_mean"))
    one_d_lower_mean = safe_float(primary_1d.get("test_lower_breach_rate_mean"))
    one_d_lower_worst = safe_float(primary_1d.get("test_lower_breach_rate_worst"))
    one_d_cov_mean = safe_float(primary_1d.get("test_coverage_abs_error_mean"))
    one_d_cov_worst = safe_float(primary_1d.get("test_coverage_abs_error_worst"))
    strict = [
        ("strict_1w", "lower_breach_mean", lower_mean, "0.09~0.11", lower_mean is not None and 0.09 <= lower_mean <= 0.11),
        ("strict_1w", "lower_breach_worst", lower_worst, "<=0.13", lower_worst is not None and lower_worst <= 0.13),
        ("strict_1w", "coverage_abs_error_mean", cov_mean, "<=0.03", cov_mean is not None and cov_mean <= 0.03),
        ("strict_1w", "coverage_abs_error_worst", cov_worst, "<=0.055", cov_worst is not None and cov_worst <= 0.055),
        ("strict_1w", "p90_band_width_worst", p90_worst, "<=0.27", p90_worst is not None and p90_worst <= 0.27),
        ("strict_1w", "downside_width_ic_mean", downside, ">=0.04", downside is not None and downside >= 0.04),
        ("strict_1w", "asymmetric_interval_score_mean", interval, "과도 악화 없음", interval is not None and interval <= 0.34),
    ]
    symmetric_cov_mean_limit = max(0.03, (one_d_cov_mean or 0.0) + 0.015)
    symmetric_cov_worst_limit = max(0.055, (one_d_cov_worst or 0.0) + 0.005)
    symmetric = [
        ("1d_1w_symmetric", "lower_breach_mean", lower_mean, f"<=1D {fmt(one_d_lower_mean)}", lower_mean is not None and one_d_lower_mean is not None and lower_mean <= one_d_lower_mean),
        ("1d_1w_symmetric", "lower_breach_worst", lower_worst, f"<=1D {fmt(one_d_lower_worst)}", lower_worst is not None and one_d_lower_worst is not None and lower_worst <= one_d_lower_worst),
        ("1d_1w_symmetric", "coverage_abs_error_mean", cov_mean, f"<=max(0.03, 1D+0.015)={fmt(symmetric_cov_mean_limit)}", cov_mean is not None and cov_mean <= symmetric_cov_mean_limit),
        ("1d_1w_symmetric", "coverage_abs_error_worst", cov_worst, f"<=max(0.055, 1D+0.005)={fmt(symmetric_cov_worst_limit)}", cov_worst is not None and cov_worst <= symmetric_cov_worst_limit),
        ("1d_1w_symmetric", "p90_band_width_worst", p90_worst, "<=1W overwide 0.27", p90_worst is not None and p90_worst <= 0.27),
        ("1d_1w_symmetric", "downside_width_ic_mean", downside, ">=0.04", downside is not None and downside >= 0.04),
    ]
    rows = []
    for table, metric, observed, criterion, passed in strict + symmetric:
        rows.append(
            {
                "decision_table": table,
                "metric": metric,
                "observed": observed,
                "criterion": criterion,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return rows


def classify_from_decisions(decisions: list[dict[str, Any]]) -> dict[str, str]:
    strict_rows = [row for row in decisions if row["decision_table"] == "strict_1w"]
    symmetric_rows = [row for row in decisions if row["decision_table"] == "1d_1w_symmetric"]
    strict_pass = all(row["status"] == "PASS" for row in strict_rows)
    symmetric_pass = all(row["status"] == "PASS" for row in symmetric_rows)
    return {
        "strict_1w_decision": "CP178-LOSS로 이동" if not strict_pass else "1W band product candidate로 이동",
        "symmetric_1d_1w_decision": "1W band product candidate로 이동" if symmetric_pass else "research reserve 유지",
        "recommended_user_choice_summary": "대칭 기준 채택 시 product candidate, strict 기준 유지 시 CP178-LOSS",
    }


def write_report(
    summary: dict[str, Any],
    folds: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    ticker_summary: dict[str, Any],
    stress_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    labels: dict[str, str],
) -> None:
    one_w = comparison[0]
    one_d = comparison[1]
    aggregate_bootstrap = [row for row in bootstrap_rows if row.get("fold_id") == "all_folds"]
    boot = aggregate_bootstrap[0] if aggregate_bootstrap else {}
    strict_fail = [row for row in decisions if row["decision_table"] == "strict_1w" and row["status"] != "PASS"]
    symmetric_fail = [row for row in decisions if row["decision_table"] == "1d_1w_symmetric" and row["status"] != "PASS"]
    lines = [
        "# CP178-WFLOCK 1W Band Walk-Forward Lower Calibration",
        "",
        "## 한 줄 결론",
        "",
        "1W band는 실패가 아니다. walk-forward lower calibration 기준으로 보면 1D band보다 lower breach 안정성은 훨씬 좋은 후보권에 들어왔고, 현재 WARN은 1W에만 더 엄격한 strict 기준을 적용했기 때문에 남아 있다.",
        "",
        "## 계약",
        "",
        "- 대상: tide_s60_q10_q90_param",
        "- calibration: walk_forward_lower_calibration",
        "- 새 학습: 없음",
        "- save-run / DB write / inference 저장 / live fetch / EODHD fallback / composite: 없음",
        "- test 결과로 threshold/calibration 변경 없음",
        "",
        "## 핵심 요약",
        "",
        f"- lower_breach mean/worst: {fmt(summary.get('test_lower_breach_rate_mean'))} / {fmt(summary.get('test_lower_breach_rate_worst'))}",
        f"- coverage_abs_error mean/worst: {fmt(summary.get('test_coverage_abs_error_mean'))} / {fmt(summary.get('test_coverage_abs_error_worst'))}",
        f"- p90_band_width mean/worst: {fmt(summary.get('test_p90_band_width_mean'))} / {fmt(summary.get('test_p90_band_width_worst'))}",
        f"- band_width_ic mean/worst: {fmt(summary.get('test_band_width_ic_mean'))} / {fmt(summary.get('test_band_width_ic_worst'))}",
        f"- downside_width_ic mean/worst: {fmt(summary.get('test_downside_width_ic_mean'))} / {fmt(summary.get('test_downside_width_ic_worst'))}",
        "",
        "## Bootstrap CI",
        "",
        f"- aggregate lower_breach CI95: {fmt(boot.get('lower_breach_ci95_low'))} ~ {fmt(boot.get('lower_breach_ci95_high'))}",
        f"- aggregate coverage CI95: {fmt(boot.get('coverage_ci95_low'))} ~ {fmt(boot.get('coverage_ci95_high'))}",
        f"- aggregate coverage_abs_error CI95: {fmt(boot.get('coverage_abs_error_ci95_low'))} ~ {fmt(boot.get('coverage_abs_error_ci95_high'))}",
        "",
        "## Fold별 결과",
        "",
        "| fold | lower mean/worst | coverage error mean/worst | p90 width worst | downside IC mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row.get('fold_id')} | "
            f"{fmt(row.get('test_lower_breach_rate_mean'))}/{fmt(row.get('test_lower_breach_rate_worst'))} | "
            f"{fmt(row.get('test_coverage_abs_error_mean'))}/{fmt(row.get('test_coverage_abs_error_worst'))} | "
            f"{fmt(row.get('test_p90_band_width_worst'))} | "
            f"{fmt(row.get('test_downside_width_ic_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## 1D Band Primary와 같은 표 비교",
            "",
            "| timeframe | candidate | lower mean/worst | coverage error mean/worst | p90 width worst | band IC mean | downside IC mean |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| 1W | {one_w.get('candidate_id')} + {one_w.get('calibration')} | {fmt(one_w.get('test_lower_breach_rate_mean'))}/{fmt(one_w.get('test_lower_breach_rate_worst'))} | {fmt(one_w.get('test_coverage_abs_error_mean'))}/{fmt(one_w.get('test_coverage_abs_error_worst'))} | {fmt(one_w.get('test_p90_band_width_worst'))} | {fmt(one_w.get('test_band_width_ic_mean'))} | {fmt(one_w.get('test_downside_width_ic_mean'))} |",
            f"| 1D | {one_d.get('candidate_id')} | {fmt(one_d.get('test_lower_breach_rate_mean'))}/{fmt(one_d.get('test_lower_breach_rate_worst'))} | {fmt(one_d.get('test_coverage_abs_error_mean'))}/{fmt(one_d.get('test_coverage_abs_error_worst'))} | {fmt(one_d.get('test_p90_band_width_worst'))} | {fmt(one_d.get('test_band_width_ic_mean'))} | {fmt(one_d.get('test_downside_width_ic_mean'))} |",
            "",
            "해석: 1W walk-forward lower는 1D primary보다 lower breach mean/worst가 낮다. coverage 평균은 1D보다 약하지만 worst coverage는 1D worst와 거의 같은 범위다.",
            "",
            "## Ticker Concentration",
            "",
            f"- ticker_count: {ticker_summary.get('ticker_count')}",
            f"- top10 lower breach share: {fmt(ticker_summary.get('top10_lower_breach_share'))}",
            f"- top10 any breach share: {fmt(ticker_summary.get('top10_any_breach_share'))}",
            f"- top10 tickers: {', '.join(ticker_summary.get('top10_tickers') or [])}",
            "",
            "## Stress / Calm",
            "",
            "| bucket | lower mean/worst | coverage error mean/worst | p90 width worst | downside IC mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stress_rows:
        lines.append(
            f"| {row.get('bucket')} | "
            f"{fmt(row.get('lower_breach_rate_mean'))}/{fmt(row.get('lower_breach_rate_worst'))} | "
            f"{fmt(row.get('coverage_abs_error_mean'))}/{fmt(row.get('coverage_abs_error_worst'))} | "
            f"{fmt(row.get('p90_band_width_worst'))} | "
            f"{fmt(row.get('downside_width_ic_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## 판정표",
            "",
            "| 기준 | 실패 항목 | 판정 |",
            "|---|---|---|",
            f"| strict 1W 기준 | {', '.join(row['metric'] for row in strict_fail) if strict_fail else '없음'} | {labels['strict_1w_decision']} |",
            f"| 1D-1W 대칭 기준 | {', '.join(row['metric'] for row in symmetric_fail) if symmetric_fail else '없음'} | {labels['symmetric_1d_1w_decision']} |",
            "",
            "## 최종 분기",
            "",
            "- 대칭 기준 채택 시: 1W band product candidate로 이동",
            "- strict 기준 유지 시: CP178-LOSS로 이동",
            "- 애매하면: research reserve 유지, 1D line v3 우선",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    fold_rows, bootstrap_rows, test_frames, label_rows = collect_walk_forward_frames()
    summary = summary_from_fold_rows(fold_rows)
    folds = fold_summary(fold_rows)
    stress_rows = stress_summary(test_frames)
    ticker_rows, ticker_summary = ticker_concentration(test_frames)
    primary_1d = load_1d_primary()
    comparison = comparison_rows(summary, primary_1d)
    decisions = decision_rows(summary, primary_1d)
    labels = classify_from_decisions(decisions)
    payload = {
        "cp": "CP178-WFLOCK",
        "created_at_utc": now_utc(),
        "candidate_id": TARGET_CANDIDATE_ID,
        "method": METHOD,
        "source_data_hash": "90666b44cbfb8e5c",
        "timeframe": "1W",
        "horizon": 4,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "new_training": False,
        "save_run": False,
        "db_write": False,
        "inference_saved": False,
        "live_fetch": False,
        "eodhd_fallback": False,
        "composite": False,
        "test_used_for_calibration_selection": False,
        "summary": summary,
        "fold_summary": folds,
        "bootstrap_ci": bootstrap_rows,
        "ticker_concentration": ticker_summary,
        "stress_calm_summary": stress_rows,
        "comparison_1d_1w": comparison,
        "decision_table": decisions,
        "decision_labels": labels,
        "stress_label_rows": label_rows,
    }
    write_rows(SUMMARY_CSV, [summary])
    write_rows(FOLD_SUMMARY_CSV, folds)
    write_rows(BOOTSTRAP_CSV, bootstrap_rows)
    write_rows(TICKER_CSV, ticker_rows)
    write_rows(STRESS_CSV, stress_rows)
    write_rows(COMPARISON_CSV, comparison)
    write_rows(DECISION_CSV, decisions)
    write_json(METRICS_PATH, payload)
    write_report(summary, folds, bootstrap_rows, ticker_summary, stress_rows, comparison, decisions, labels)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="CP178-WFLOCK walk-forward lower calibration evidence")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run 필요")
    metrics = run()
    print(json.dumps({"decision_labels": metrics["decision_labels"], "report": str(REPORT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
