import math
from typing import Optional, Tuple

import torch

from .conv1d import Conv1d
from .conv1d1x1 import Conv1d1x1


## Residual block module in WaveNet
class ResidualBlock(torch.nn.Module):
    conv1x1_aux: Optional[Conv1d1x1]

    ## Initialize WaveNetResidualBlock module.
    # @param kernel_size         Kernel size of dilation convolution layer.
    # @param residual_channels   Number of channels for residual connection.
    # @param skip_channels       Number of channels for skip connection.
    # @param aux_channels        Local conditioning channels i.e. auxiliary input dimension.
    # @param dropout             Dropout probability.
    # @param dilation            Dilation factor.
    # @param bias                Whether to add bias parameter in convolution layers.
    # @param use_causal_conv     Whether to use use_causal_conv or non-use_causal_conv convolution.
    def __init__(
        self,
        kernel_size: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        dropout: float = 0.0,
        dilation: int = 1,
        bias: bool = True,
        use_causal_conv: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        # no future time stamps available
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    ## Calculate forward propagation.
    # @param x  Input tensor (B, residual_channels, T).
    # @param c  Local conditioning auxiliary tensor (B, aux_channels, T).
    # @return   residual connection, skip connection
    #           (B, residual_channels, T), (B, skip_channels, T)
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        # remove future time steps if use_causal_conv conv
        x = x[:, :, : residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s
