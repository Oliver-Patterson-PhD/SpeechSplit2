from typing import Self

import torch

from util.config import Config


class Transcriber:
    device: torch.device
    model: torch.nn.Module
    model_name: str
    config: Config

    def __init__(
        self: Self,
        device: torch.device,
        model_name: str,
        config: Config,
    ):
        self.device = device
        self.model_name = model_name
        self.config = config
        return

    def transcribe(
        self: Self,
        melspec: torch.Tensor,
        name: str,
    ) -> None:
        raise NotImplementedError()
