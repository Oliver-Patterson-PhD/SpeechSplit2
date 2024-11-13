from typing import Self

import torch

from util import Config

from .layers.conv_norm import ConvNorm


class EncoderRhythm(torch.nn.Module):
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        super().__init__()
        self.chs_grp = config.model.chs_grp
        self.dim_emb = config.model.dim_spk_emb
        self.dim_enc_2 = config.model.dim_enc_2
        self.dim_neck_2 = config.model.dim_neck_2
        self.dim_rhy = config.model.dim_rhy
        self.dropout = config.model.dropout
        self.freq_2 = config.model.freq_2
        convolutions = []
        for i in range(1):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_rhy if i == 0 else self.dim_enc_2,
                    self.dim_enc_2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)
        self.lstm = torch.nn.LSTM(
            self.dim_enc_2,
            self.dim_neck_2,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convolutions:
            x = torch.nn.functional.relu(conv(x))
        x = x.transpose(-2, -1)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, : self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2 :]
        codes = torch.cat(
            (
                out_forward[:, self.freq_2 - 1 :: self.freq_2, :],
                out_backward[:, :: self.freq_2, :],
            ),
            dim=-1,
        )
        return codes
