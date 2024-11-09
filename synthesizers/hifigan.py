from tomllib import load as loadtoml
from typing import Self

import torch

from util import Config

from .hifi_gan import Generator as HifiGanGenerator
from .hifi_gan import HiFiConfig
from .synthesizer import Synthesizer


class HiFiGAN(Synthesizer):
    model_name: str = "hifigan"

    @torch.no_grad()
    def __init__(
        self: Self,
        device: torch.device,
        config: Config,
    ) -> None:
        data_dir = self.config.paths.full_models
        config_file = f"{data_dir}/{self.model_name}.toml"
        self.configtoml = HiFiConfig(**loadtoml(open(config_file, "rb")))
        self.model = HifiGanGenerator(hifi_config=self.configtoml).to(device)
        self.model.eval()
        self.model.remove_weight_norm()

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(spect)
