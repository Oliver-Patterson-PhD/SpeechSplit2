import torch
from parallel_wavegan.melgan import MelGANGenerator
from synthesizers.synthesizer import Synthesizer
from util.config import Config


class MelGanSynthesizer(Synthesizer):
    def __init__(self, device: torch.device, config: Config) -> None:
        super().__init__(
            device=device,
            model=MelGANGenerator,
            model_name="lj_melgan-4M",
            config=config,
        )

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outwav = self.model.inference(c=spect.to(self.device)).view(-1)
        return outwav
