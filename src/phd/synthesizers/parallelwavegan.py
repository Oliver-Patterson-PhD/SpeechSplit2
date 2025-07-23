import torch

from ..models.parallel_wavegan import ParallelWaveGANGenerator
from ..util import config
from .synthesizer import Synthesizer


class ParallelWaveGan(Synthesizer):
    def __init__(self, device: torch.device) -> None:
        super().__init__(
            device=device,
            model=ParallelWaveGANGenerator,
            model_name="parallelwavegan/" + config.options.parallelwavegan_name,
        )
        self.model.eval()

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        outwav = self.model.inference(c=spect.to(self.device)).view(-1)
        return outwav
