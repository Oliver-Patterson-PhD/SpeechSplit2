from typing import List, Optional

import torch

from .conv2d import Conv2d
from .stretch2d import Stretch2d


## Upsampling network module
class UpsampleNetwork(torch.nn.Module):

    ## Initialize upsampling network module.
    # @param    upsample_scales             List of upsampling scales.
    # @param    nonlinear_activation        Activation function name.
    # @param    nonlinear_activation_params Arguments for specified activation function.
    # @param    interpolate_mode            Interpolation mode.
    # @param    freq_axis_kernel_size       Kernel size in the direction of frequency axis.
    # @param    use_causal_conv             Use Causal Convolution
    def __init__(
        self,
        upsample_scales: List[int],
        nonlinear_activation: Optional[str] = None,
        nonlinear_activation_params: dict = {},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        use_causal_conv: bool = False,
    ) -> None:
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    ## Calculate forward propagation.
    # @param    c   Import tensor (B, C, T).
    # @return   Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., : c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)  # (B, C, T')
