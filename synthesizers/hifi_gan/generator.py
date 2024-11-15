from typing import Self

import torch

from .resblocks import ResBlock1, ResBlock2
from .util import LRELU_SLOPE, HiFiConfig, init_weights


class Generator(torch.nn.Module):
    num_kernels: int
    num_upsamples: int

    def __init__(
        self: Self,
        hifi_config: HiFiConfig,
    ) -> None:
        super(Generator, self).__init__()
        self.hifi_config = hifi_config
        self.num_kernels = len(hifi_config.resblock_kernel_sizes)
        self.num_upsamples = len(hifi_config.upsample_rates)
        self.conv_pre = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(80, hifi_config.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if hifi_config.resblock == "1" else ResBlock2

        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(hifi_config.upsample_rates, hifi_config.upsample_kernel_sizes)
        ):
            self.ups.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(
                        hifi_config.upsample_initial_channel // (2**i),
                        hifi_config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hifi_config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    hifi_config.resblock_kernel_sizes,
                    hifi_config.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(resblock(hifi_config, ch, k, d))

        self.conv_post = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(ch, 1, 7, 1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self: Self) -> None:
        print("Removing weight norm...")
        for i in self.ups:
            torch.nn.utils.remove_weight_norm(i)
        for j in self.resblocks:
            j.remove_weight_norm()
        torch.nn.utils.remove_weight_norm(self.conv_pre)
        torch.nn.utils.remove_weight_norm(self.conv_post)
