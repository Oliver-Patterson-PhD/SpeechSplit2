from typing import Optional, Self

import torch

from models.parallel_wavegan import ParallelWaveGANGenerator
from util import Config

from .synthesizer import Synthesizer


class ParallelWaveGan(Synthesizer):
    def __init__(
        self: Self,
        device: torch.device,
        config: Optional[Config] = None,
    ) -> None:
        self.config: Config = config or Config()
        super().__init__(
            device=device,
            model=ParallelWaveGANGenerator,
            model_name="parallelwavegan/" + self.config.options.parallelwavegan_name,
            config=self.config,
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
