from tomllib import load as loadtoml
from typing import Self

import torch

from util.config import Config


class Synthesizer(object):
    device: torch.device
    model: torch.nn.Module
    checkpoint_path: str
    model_name: str
    config: Config

    def __init__(
        self: Self,
        device: torch.device,
        model: type,
        model_name: str,
        config: Config,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.config = config
        config_file = f"{self.config.paths.full_models}/{self.model_name}.toml"
        pickle_file = f"{self.config.paths.full_models}/{self.model_name}.pkl"
        tomlconfig = loadtoml(open(config_file, "rb"))
        state_dict = torch.load(
            pickle_file,
            map_location="cpu",
            weights_only=True,
        )
        model_params = {
            k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
            for k, v in tomlconfig["generator_params"].items()
        }
        self.model = model(**model_params)
        self.model.load_state_dict(state_dict["model"]["generator"])
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __str__(
        self: Self,
    ) -> str:
        return self.model_name
