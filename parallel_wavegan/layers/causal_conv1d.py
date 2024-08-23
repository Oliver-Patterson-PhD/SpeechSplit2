from typing import Any, Dict

import torch


## CausalConv1d module with customized initialization
class CausalConv1d(torch.nn.Module):

    ## Initialize CausalConv1d module
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        pad: str = "ConstantPad1d",
        pad_params: Dict[str, Any] = {"value": 0.0},
    ) -> None:
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    ## Calculate forward propagation
    # @param    x   Input tensor (B, in_channels, T).
    # @return   Tensor: Output tensor (B, out_channels, T).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))[:, :, : x.size(2)]
