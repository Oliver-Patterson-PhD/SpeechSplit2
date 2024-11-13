from typing import Self

import torch

from util import Config

from .layers.linear_norm import LinearNorm


class SpeechSplitDecoder(torch.nn.Module):
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        super().__init__()
        self.dim_emb = config.model.dim_spk_emb
        self.dim_freq = config.model.dim_freq
        self.dim_neck = config.model.dim_neck_1
        self.dim_neck_2 = config.model.dim_neck_2
        self.dim_neck_3 = config.model.dim_neck_3
        self.lstm = torch.nn.LSTM(
            self.dim_neck * 2
            + self.dim_neck_2 * 2
            + self.dim_neck_3 * 2
            + self.dim_emb,
            512,
            3,
            batch_first=True,
            bidirectional=True,
        )
        self.linear_projection = LinearNorm(1024, self.dim_freq)

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output
