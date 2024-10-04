import torch
from safetensors import safe_open

from transcribers.transcriber import Transcriber
from util.config import Config


class Whisper(Transcriber):
    device: torch.device
    model: torch.nn.Module
    model_name: str
    config: Config

    def __init__(
        self,
        device: torch.device,
        model_name: str,
        config: Config,
    ):
        self.device = device
        self.model_name = model_name
        self.config = config
        config_file = f"{self.config.paths.trained_models}/{self.model_name}.toml"
        tensors_file = "{}/{}.safetensors".format(
            self.config.paths.trained_models,
            self.model_name,
        )
        model_params = loadtoml(open(config_file, "rb"))
        tensors = {}
        self.model = model(**model_params)
        with safe_open(tensors_file, framework="pt", device=self.device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return
