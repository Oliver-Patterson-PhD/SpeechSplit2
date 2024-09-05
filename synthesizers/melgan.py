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
