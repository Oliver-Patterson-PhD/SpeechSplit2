from tomllib import load as loadtoml
from typing import Optional

import torch


class Synthesizer(object):
    device: torch.device
    model: torch.nn.Module
    checkpoint_path: str = "full_models/ParallelWaveGan"
    model_name: str

    def __init__(
        self,
        device: torch.device,
        model: type,
        model_name: str,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        config_file = f"{self.checkpoint_path}/{self.model_name}.toml"
        pickle_file = f"{self.checkpoint_path}/{self.model_name}.pkl"

        config = loadtoml(open(config_file, "rb"))
        state_dict = torch.load(pickle_file, map_location="cpu")
        model_params = {
            k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
            for k, v in config["generator_params"].items()
        }

        self.model = model(**model_params)
        self.model.load_state_dict(state_dict["model"]["generator"])
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outwav = self.model.inference(c=spect).view(-1)
        return outwav
