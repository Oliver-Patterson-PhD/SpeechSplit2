from typing import Any, Dict

import torch

from .causal_conv1d import CausalConv1d


## Residual stack module introduced in MelGAN
class ResidualStack(torch.nn.Module):

    ## Initialize ResidualStack module
    # @param    kernel_size                 Kernel size of dilation convolution layer.
    # @param    channels                    Number of channels of convolution layers.
    # @param    dilation                    Dilation factor.
    # @param    bias                        Whether to add bias parameter in convolution layers.
    # @param    nonlinear_activation        Activation function module name.
    # @param    nonlinear_activation_params Hyperparameters for activation function.
    # @param    pad                         Padding function module name before dilated convolution layer.
    # @param    pad_params                  Hyperparameters for padding function.
    # @param    use_causal_conv             Whether to use causal convolution.
    def __init__(
        self,
        kernel_size: int = 3,
        channels: int = 32,
        dilation: int = 1,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
        use_causal_conv: bool = False,
    ):
        super(ResidualStack, self).__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
                torch.nn.Conv1d(
                    channels, channels, kernel_size, dilation=dilation, bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    ## Calculate forward propagation
    # @param    c   Input tensor (B, channels, T).
    # @return   Output tensor (B, chennels, T).
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.stack(c) + self.skip_layer(c)
