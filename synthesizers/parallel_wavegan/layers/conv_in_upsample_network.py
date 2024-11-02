from typing import Any, Dict, List, Optional

import torch

from .conv1d import Conv1d
from .upsample_network import UpsampleNetwork


## Convolution + upsampling network module
class ConvInUpsampleNetwork(torch.nn.Module):

    ## Initialize convolution + upsampling network module.
    # @param upsample_scales             List of upsampling scales.
    # @param nonlinear_activation        Activation function name.
    # @param nonlinear_activation_params Arguments for specified activation function.
    # @param mode                        Interpolation mode.
    # @param freq_axis_kernel_size       Kernel size in the direction of frequency axis.
    # @param aux_channels                Number of channels of pre-convolutional layer.
    # @param aux_context_window          Context window size of the pre-convolutional layer.
    # @param use_causal_conv             Whether to use causal structure.
    def __init__(
        self,
        upsample_scales: List[int],
        nonlinear_activation: Optional[str] = None,
        nonlinear_activation_params: Dict[str, Any] = {},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        aux_channels: int = 80,
        aux_context_window: int = 0,
        use_causal_conv: bool = False,
    ) -> None:
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = (
            aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        )
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = Conv1d(
            aux_channels, aux_channels, kernel_size=kernel_size, bias=False
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    ## Calculate forward propagation.
    # @param c Input tensor (B, C, T')
    # @return Upsampled tensor (B, C, T), where T = (T' - aux_context_window * 2) * prod(upsample_scales).
    # @note The length of inputs considers the context window size.
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c_ = self.conv_in(c)
        c = c_[:, :, : -self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)
