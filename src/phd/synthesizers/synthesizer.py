from tomllib import load as loadtoml

import torch

from ..util import config


class Synthesizer(object):
    device: torch.device
    model: torch.nn.Module
    checkpoint_path: str
    model_name: str

    def __init__(self, device: torch.device, model: type, model_name: str) -> None:
        self.model_name = model_name
        self.device = device
        config_file = f"{config.paths.full_models}/{self.model_name}.toml"
        pickle_file = f"{config.paths.full_models}/{self.model_name}.pkl"
        with open(config_file, "rb") as openconf:
            tomlconfig = loadtoml(openconf)
        state_dict = torch.load(pickle_file, map_location="cpu", weights_only=True)
        model_params = {
            k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
            for k, v in tomlconfig["generator_params"].items()
        }
        self.model = model(**model_params)
        self.model.load_state_dict(state_dict["model"]["generator"])
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.model_name.partition("/")[0]
