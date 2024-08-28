import torch
from parallel_wavegan.parallel_wavegan import ParallelWaveGANGenerator
from synthesizers.synthesizer import Synthesizer
from util.config import Config


class ParallelWaveGanSynthesizer(Synthesizer):
    def __init__(self, device: torch.device, config: Config) -> None:
        super().__init__(
            device=device,
            model=ParallelWaveGANGenerator,
            model_name="parallelwavegan-lj-3M",
            config=config,
        )
