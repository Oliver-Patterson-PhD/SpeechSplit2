__all__ = [
    "norm_audio",
]

import torch


def norm_audio(
    x: torch.Tensor,
) -> torch.Tensor:
    return x / x.abs().max()
