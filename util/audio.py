__all__ = [
    "norm_audio",
]

from torch import Tensor


def norm_audio(x: Tensor) -> Tensor:
    return x / x.abs().max()
