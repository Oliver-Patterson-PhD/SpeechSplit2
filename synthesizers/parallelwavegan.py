import torch
from parallel_wavegan.parallel_wavegan import ParallelWaveGANGenerator
from synthesizers.synthesizer import Synthesizer


class ParallelWaveGanSynthesizer(Synthesizer):
    def __init__(self, device: torch.device) -> None:
        super().__init__(
            device=device,
            model=ParallelWaveGANGenerator,
            model_name="lj_parallelwavegan-3M",
        )
