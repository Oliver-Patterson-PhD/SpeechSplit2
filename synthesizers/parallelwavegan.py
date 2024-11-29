from typing import Self

import torch

from models.parallel_wavegan import ParallelWaveGANGenerator
from util import Config

from .synthesizer import Synthesizer


class ParallelWaveGan(Synthesizer):
    def __init__(
        self: Self,
        device: torch.device,
        config: Config,
    ) -> None:
        super().__init__(
            device=device,
            model=ParallelWaveGANGenerator,
            model_name="parallelwavegan-3M",
            config=config,
        )
        self.model.eval()

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        outwav = self.model.inference(
            c=spect.to(self.device),
        ).view(-1)
        return outwav
