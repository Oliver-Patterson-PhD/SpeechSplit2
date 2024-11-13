from typing import Self, Tuple

import torch

from util import Config

from .layers.conv_norm import ConvNorm


class EncoderSync(torch.nn.Module):
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        super().__init__()
        self.chs_grp = config.model.chs_grp
        self.dim_con = config.model.dim_con
        self.dim_enc = config.model.dim_enc_1
        self.dim_enc_3 = config.model.dim_enc_3
        self.dim_f0 = config.model.dim_f0
        self.dim_neck = config.model.dim_neck_1
        self.dim_neck_3 = config.model.dim_neck_3
        self.dim_pit = config.model.dim_pit
        self.freq = config.model.freq_1
        self.freq_3 = config.model.freq_3
        self.register_buffer("len_org", torch.tensor(config.model.max_len_pad))
        # convolutions for code 1
        convolutions = []
        for i in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_con if i == 0 else self.dim_enc,
                    self.dim_enc,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc),
            )
            convolutions.append(conv_layer)
        self.convolutions_1 = torch.nn.ModuleList(convolutions)
        self.lstm_1 = torch.nn.LSTM(
            self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True
        )
        # convolutions for f0
        convolutions = []
        for i in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_pit if i == 0 else self.dim_enc_3,
                    self.dim_enc_3,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3),
            )
            convolutions.append(conv_layer)
        self.convolutions_2 = torch.nn.ModuleList(convolutions)
        self.lstm_2 = torch.nn.LSTM(
            self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True
        )
        self.interp = InterpLnr(config)

    def forward(
        self: Self,
        x_f0: torch.Tensor,
        rr: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_f0[:, : self.dim_con, :]
        f0 = x_f0[:, self.dim_con :, :]
        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = torch.nn.functional.relu(conv_1(x))
            f0 = torch.nn.functional.relu(conv_2(f0))
            x_f0 = torch.cat((x, f0), dim=-2).transpose(-2, -1)
            if rr:
                x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))
            x_f0 = x_f0.transpose(-2, -1)
            x = x_f0[:, : self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc :, :]
        x_f0 = x_f0.transpose(-2, -1)
        x = x_f0[:, :, : self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc :]
        # code 1
        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()
        x = self.lstm_1(x)[0]
        f0 = self.lstm_2(f0)[0]
        x_forward = x[:, :, : self.dim_neck]
        x_backward = x[:, :, self.dim_neck :]
        f0_forward = f0[:, :, : self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3 :]
        codes_x = torch.cat(
            (
                x_forward[:, self.freq - 1 :: self.freq, :],
                x_backward[:, :: self.freq, :],
            ),
            dim=-1,
        )
        codes_f0 = torch.cat(
            (
                f0_forward[:, self.freq_3 - 1 :: self.freq_3, :],
                f0_backward[:, :: self.freq_3, :],
            ),
            dim=-1,
        )
        return codes_x, codes_f0
