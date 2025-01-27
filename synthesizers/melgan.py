from typing import Optional, Self

import torch

from models.melgan import MelGANGenerator
from util import Config

from .synthesizer import Synthesizer


class MelGan(Synthesizer):
    def __init__(
        self: Self,
        device: torch.device,
        config: Optional[Config] = None,
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
        outwav = self.model.inference(
            c=spect.to(self.device),
        ).view(-1)
        return outwav
