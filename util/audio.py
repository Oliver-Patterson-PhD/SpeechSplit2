__all__ = [
    "norm_audio",
]

import torch


def norm_audio(
    x: torch.Tensor,
) -> torch.Tensor:
    return (((x - x.min()) / (x.max() - x.min())) * 2) - 1
