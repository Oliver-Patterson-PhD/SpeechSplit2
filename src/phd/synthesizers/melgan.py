import torch

from ..models.melgan import MelGANGenerator
from .synthesizer import Synthesizer


class MelGan(Synthesizer):
    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, model=MelGANGenerator, model_name="melgan-4M")
        self.model.eval()

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        outwav = self.model.inference(c=spect.to(self.device)).view(-1)
        return outwav
