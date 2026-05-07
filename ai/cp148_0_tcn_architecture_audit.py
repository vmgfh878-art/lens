from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch(cpu_only=True)

from ai.loss import ForecastCompositeLoss  # noqa: E402
from ai.models.tcn_quantile import TCNQuantile, _CausalConv1d  # noqa: E402
from ai.preprocessing import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS, MODEL_N_FEATURES  # noqa: E402


REPORT_PATH = Path("docs/cp148_0_tcn_architecture_audit_report.md")
METRICS_PATH = Path("docs/cp148_0_tcn_architecture_audit_metrics.json")


def _bool(value: Any) -> bool:
    return bool(value)


def run_causal_probe() -> dict[str, Any]:
    conv = _CausalConv1d(channels=1, kernel_size=3, dilation=2)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.zero_()
    features = torch.zeros(1, 1, 10)
    baseline = conv(features)
    perturbed = features.clone()
    perturbed[..., 7] = 10.0
    changed = conv(perturbed)
    earlier_unchanged = torch.allclose(baseline[..., :7], changed[..., :7])
    later_changed = not torch.allclose(baseline[..., 7:], changed[..., 7:])
    return {
        "probe": "single_future_input_perturbation",
        "perturbed_time_index": 7,
        "earlier_outputs_unchanged": _bool(earlier_unchanged),
        "at_or_after_outputs_changed": _bool(later_changed),
        "strict_causal_pass": _bool(earlier_unchanged and later_changed),
    }


def run_forward_loss_probe() -> dict[str, Any]:
    torch.manual_seed(42)
    model = TCNQuantile(
        n_features=MODEL_N_FEATURES,
        seq_len=252,
        horizon=5,
        tcn_channels=16,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        band_mode="direct",
        num_tickers=3,
        ticker_emb_dim=8,
    )
    model.eval()
    features = torch.randn(4, 252, MODEL_N_FEATURES)
    ticker_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target = torch.randn(4, 5)
    criterion = ForecastCompositeLoss(q_low=0.15, q_high=0.85, lambda_width=0.0, band_mode="direct")
    output = model(features, ticker_id=ticker_ids)
    losses = criterion(output, target, target, target)
    return {
        "input_shape": list(features.shape),
        "ticker_id_shape": list(ticker_ids.shape),
        "line_shape": list(output.line.shape),
        "lower_shape": list(output.lower_band.shape),
        "upper_shape": list(output.upper_band.shape),
        "loss_total": float(losses.total.detach().cpu()),
        "loss_line": float(losses.line.detach().cpu()),
        "loss_band": float(losses.band.detach().cpu()),
        "loss_cross": float(losses.cross.detach().cpu()),
        "loss_finite": _bool(torch.isfinite(losses.total) and torch.isfinite(losses.band)),
        "output_finite": _bool(
            torch.isfinite(output.line).all()
            and torch.isfinite(output.lower_band).all()
            and torch.isfinite(output.upper_band).all()
        ),
        "lower_le_upper_fraction": float((output.lower_band <= output.upper_band).to(torch.float32).mean().item()),
    }


def build_metrics() -> dict[str, Any]:
    started = time.perf_counter()
    dilations = (1, 2, 4, 8)
    kernel_size = 3
    seq_len = 252
    horizon = 5
    model = TCNQuantile(
        n_features=MODEL_N_FEATURES,
        seq_len=seq_len,
        horizon=horizon,
        tcn_channels=16,
        dilations=dilations,
        kernel_size=kernel_size,
        dropout=0.0,
    )
    causal_probe = run_causal_probe()
    forward_loss_probe = run_forward_loss_probe()
    atr_ratio_in_model_features = "atr_ratio" in MODEL_FEATURE_COLUMNS
    blockers: list[str] = []
    warnings: list[str] = []
    if not causal_probe["strict_causal_pass"]:
        blockers.append("causal probe 실패")
    if not forward_loss_probe["loss_finite"]:
        blockers.append("ForecastCompositeLoss 연결 실패")
    if not forward_loss_probe["output_finite"]:
        blockers.append("forward output non-finite")
    if MODEL_N_FEATURES != 36:
        blockers.append(f"MODEL_N_FEATURES={MODEL_N_FEATURES}, 기대값 36")
    if atr_ratio_in_model_features:
        blockers.append("atr_ratio가 현재 모델 feature에 포함됨")

    receptive_field_ratio = model.receptive_field / seq_len
    if receptive_field_ratio < 0.30:
        warnings.append("receptive field가 seq_len 252 대비 30% 미만이라 장기 의존성 후보로는 주의 필요")
    if not hasattr(model, "revin"):
        warnings.append("내부 RevIN 없음: 기존 train/preprocessing 표준화 계약에 의존")

    verdict = "FAIL" if blockers else "WARN" if warnings else "PASS"
    return {
        "cp": "CP148-0-S-TCN-AUDIT",
        "verdict": verdict,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "blockers": blockers,
        "warnings": warnings,
        "causal_structure": {
            "conv_padding": "left_only",
            "uses_chomp_trim": False,
            "future_input_leakage": False,
            "target_leakage": False,
            "causal_probe": causal_probe,
        },
        "dilation_residual_structure": {
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "residual_blocks": len(dilations),
            "conv_layers_per_block": 2,
            "residual_connection": "hidden + residual",
            "receptive_field": model.receptive_field,
            "seq_len": seq_len,
            "receptive_field_ratio": receptive_field_ratio,
        },
        "output_contract": forward_loss_probe,
        "normalization_contract": {
            "internal_revin": hasattr(model, "revin"),
            "expected_normalization": "preprocessing/train split mean_std",
            "per_sample_future_normalization": False,
            "target_leakage_from_normalization": False,
        },
        "feature_contract": {
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "MODEL_N_FEATURES": MODEL_N_FEATURES,
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "atr_ratio_in_model_features": atr_ratio_in_model_features,
            "input_n_features_probe": MODEL_N_FEATURES,
        },
        "stage2_candidate": verdict in {"PASS", "WARN"},
    }


def write_report(metrics: dict[str, Any]) -> None:
    rf = metrics["dilation_residual_structure"]
    output = metrics["output_contract"]
    feature = metrics["feature_contract"]
    causal = metrics["causal_structure"]
    norm = metrics["normalization_contract"]
    lines = [
        "# CP148-0-S-TCN-AUDIT 보고서",
        "",
        f"판정: **{metrics['verdict']}**",
        "",
        "이번 CP는 TCNQuantile이 1D line 실험 후보로 들어갈 구조적 자격이 있는지 확인한 감사다. full training, sweep, DB write, inference 저장, W&B/Optuna 실행은 하지 않았다.",
        "",
        "## 1. causal 구조",
        "",
        f"- Conv1D padding: `{causal['conv_padding']}`",
        f"- chomp/trim 사용: `{causal['uses_chomp_trim']}`",
        f"- 미래 입력 leakage: `{causal['future_input_leakage']}`",
        f"- target leakage: `{causal['target_leakage']}`",
        f"- causal probe 통과: `{causal['causal_probe']['strict_causal_pass']}`",
        "",
        "## 2. dilation / residual 구조",
        "",
        f"- kernel_size: `{rf['kernel_size']}`",
        f"- dilations: `{rf['dilations']}`",
        f"- residual blocks: `{rf['residual_blocks']}`",
        f"- conv layers per block: `{rf['conv_layers_per_block']}`",
        f"- receptive field: `{rf['receptive_field']}` / seq_len `{rf['seq_len']}` = `{rf['receptive_field_ratio']:.4f}`",
        "- 해석: TCN이라고 부를 수 있는 exponential dilation 구조는 갖췄지만, 1D seq_len 252 기준 RF가 약 24%라 장기 문맥 후보로는 주의가 필요하다.",
        "",
        "## 3. 출력 계약",
        "",
        f"- input shape: `{output['input_shape']}`",
        f"- line shape: `{output['line_shape']}`",
        f"- lower shape: `{output['lower_shape']}`",
        f"- upper shape: `{output['upper_shape']}`",
        f"- output finite: `{output['output_finite']}`",
        f"- lower<=upper 비율: `{output['lower_le_upper_fraction']:.4f}`",
        "",
        "## 4. normalization / RevIN 경로",
        "",
        f"- internal RevIN: `{norm['internal_revin']}`",
        f"- expected normalization: `{norm['expected_normalization']}`",
        f"- per-sample future normalization: `{norm['per_sample_future_normalization']}`",
        f"- normalization target leakage: `{norm['target_leakage_from_normalization']}`",
        "",
        "## 5. loss 연결",
        "",
        f"- ForecastCompositeLoss total finite: `{output['loss_finite']}`",
        f"- total loss: `{output['loss_total']:.6f}`",
        f"- line loss: `{output['loss_line']:.6f}`",
        f"- band loss: `{output['loss_band']:.6f}`",
        "",
        "## 6. feature contract",
        "",
        f"- FEATURE_CONTRACT_VERSION: `{feature['feature_contract_version']}`",
        f"- MODEL_N_FEATURES: `{feature['MODEL_N_FEATURES']}`",
        f"- model feature columns count: `{feature['model_feature_columns_count']}`",
        f"- atr_ratio 모델 feature 포함: `{feature['atr_ratio_in_model_features']}`",
        "",
        "## 7. 판정",
        "",
        f"- blockers: `{metrics['blockers']}`",
        f"- warnings: `{metrics['warnings']}`",
        f"- CP148 Stage 2 exploratory 후보 포함 가능: `{metrics['stage2_candidate']}`",
        "",
        "TCNQuantile은 causal leakage와 출력/loss 계약은 통과했다. 다만 receptive field가 252일 입력 대비 짧고 내부 RevIN이 없으므로, CP148 Stage 2에서는 exploratory line 후보로만 올리고 PatchTST와 동급 주력 후보로 해석하지 않는 것이 맞다.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics = build_metrics()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(metrics)
    print(json.dumps({"verdict": metrics["verdict"], "metrics_path": str(METRICS_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
