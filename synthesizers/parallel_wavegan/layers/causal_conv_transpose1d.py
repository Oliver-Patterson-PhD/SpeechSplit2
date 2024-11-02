from typing import Any, Dict

import torch


## CausalConvTranspose1d module with customized initialization
class CausalConvTranspose1d(torch.nn.Module):

    ## Initialize CausalConvTranspose1d module
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        pad: str = "ReplicationPad1d",
        pad_params: Dict[str, Any] = {},
    ) -> None:
        super(CausalConvTranspose1d, self).__init__()
        # NOTE (yoneyama): This padding is to match the number of inputs
        #   used to calculate the first output sample with the others.
        self.pad = getattr(torch.nn, pad)((1, 0), **pad_params)
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )
        self.stride = stride

    ## Calculate forward propagation.
    # @param    x   Input tensor (B, in_channels, T_in).
    # @return   Output tensor (B, out_channels, T_out).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.pad(x))[:, :, self.stride : -self.stride]
