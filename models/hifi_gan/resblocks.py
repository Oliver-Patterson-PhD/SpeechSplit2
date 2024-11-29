from typing import Self

import torch

from .util import LRELU_SLOPE, HiFiConfig, get_padding, init_weights


class ResBlock1(torch.nn.Module):
    def __init__(
        self: Self,
        hifi_config: HiFiConfig,
        channels: int,
        kernel_size=3,
        dilation=(1, 3, 5),
    ) -> None:
        super(ResBlock1, self).__init__()
        self.hifi_config = hifi_config
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self: Self) -> None:
        for i in self.convs1:
            torch.nn.utils.remove_weight_norm(i)
        for j in self.convs2:
            torch.nn.utils.remove_weight_norm(j)


class ResBlock2(torch.nn.Module):
    def __init__(
        self: Self,
        hifi_config: HiFiConfig,
        channels: int,
        kernel_size=3,
        dilation=(1, 3),
    ) -> None:
        super(ResBlock2, self).__init__()
        self.hifi_config = hifi_config
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for c in self.convs:
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self: Self) -> None:
        for i in self.convs:
            torch.nn.utils.remove_weight_norm(i)
