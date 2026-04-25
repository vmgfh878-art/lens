from __future__ import annotations

import torch


def apply_band_postprocess(
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """line은 그대로 두고 band만 정렬해 lower <= upper를 보장한다."""
    band = torch.stack((lower, upper), dim=-1)
    ordered = torch.sort(band, dim=-1).values
    return line, ordered[..., 0], ordered[..., 1]
