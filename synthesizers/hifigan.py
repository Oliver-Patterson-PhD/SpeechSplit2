# from tomllib import load as loadtoml
from typing import Optional, Self

import torch

from util import Config, Logger

# from models.hifi_gan import Generator as HifiGanGenerator
# from models.hifi_gan import HiFiConfig
from .synthesizer import Synthesizer


class HiFiGAN(Synthesizer):
    model_name: str = "hifigan"

    @torch.no_grad()
    def __init__(
        self: Self,
        device: torch.device,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config()
        self.device = device
        self.config = config
        # data_dir = self.config.paths.full_models
        # config_file = f"{data_dir}/{self.model_name}.toml"
        # tomlconf = {**loadtoml(open(config_file, "rb")), "fmax_for_loss": None}
        # self.configtoml = HiFiConfig(**tomlconf)
        # self.model = HifiGanGenerator(hifi_config=self.configtoml).to(device)
        self.logger = Logger()
        self.hifigan, vocoder_train_setup, self.denoiser = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_hifigan",
        )
        CHECKPOINT_SPECIFIC_ARGS = [
            "sampling_rate",
            "hop_length",
            "win_length",
            "p_arpabet",
            "text_cleaners",
            "symbol_set",
            "max_wav_value",
            "prepend_space_to_text",
            "append_space_to_text",
        ]

        for k in CHECKPOINT_SPECIFIC_ARGS:
            self.logger.debug(f"{k}: {vocoder_train_setup.get(k, None)}")
        self.div_val = vocoder_train_setup.get("max_wav_value", 1)
        self.hifigan.to(device)
        self.denoiser.to(device)
        self.denoising_strength = 0.05
        self.hifigan.eval()
        self.denoiser.eval()

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        retval = (
            self.denoiser(
                self.hifigan(spect.T).float().squeeze(1), self.denoising_strength
            )
            .squeeze()
            .T
        )
        return retval / self.div_val
