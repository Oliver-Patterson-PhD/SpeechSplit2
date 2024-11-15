from typing import Self

import torch

from util import Config

from .parallel_wavegan.melgan import MelGANGenerator
from .synthesizer import Synthesizer


class MelGan(Synthesizer):
    def __init__(
        self: Self,
        device: torch.device,
        config: Config,
    ) -> None:
        super().__init__(
            device=device,
            model=MelGANGenerator,
            model_name="melgan-4M",
            config=config,
        )
        self.model.eval()

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        outwav = self.model.inference(c=spect.to(self.device)).view(-1)
        return outwav
