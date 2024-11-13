from typing import Optional, Self

import torch


class ConvNorm(torch.nn.Module):
    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain="linear",
    ) -> None:
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(
        self: Self,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        conv_signal = self.conv(signal)
        return conv_signal
