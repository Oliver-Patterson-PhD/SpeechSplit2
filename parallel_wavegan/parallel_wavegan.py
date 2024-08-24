# -*- coding: utf-8 -*-
# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

## Parallel WaveGAN Modules.

import logging
import math
from typing import Any, Dict, Optional

import torch

from .layers.conv1d1x1 import Conv1d1x1
from .layers.conv_in_upsample_network import ConvInUpsampleNetwork
from .layers.residual_block import ResidualBlock


## Parallel WaveGAN Generator module.
class ParallelWaveGANGenerator(torch.nn.Module):
    aux_channels: int
    aux_context_window: int
    in_channels: int
    kernel_size: int
    layers: int
    out_channels: int
    stacks: int
    upsample_factor: int
    upsample_net: Optional[ConvInUpsampleNetwork]

    ## Initialize Parallel WaveGAN Generator module.
    # @param in_channels                    Number of input channels.
    # @param out_channels                   Number of output channels.
    # @param kernel_size                    Kernel size of dilated convolution.
    # @param layers                         Number of residual block layers.
    # @param stacks                         Number of stacks i.e., dilation cycles.
    # @param residual_channels              Number of channels in residual conv.
    # @param gate_channels                  Number of channels in gated conv.
    # @param skip_channels                  Number of channels in skip conv.
    # @param aux_channels                   Number of channels for auxiliary feature conv.
    # @param aux_context_window             Context window size for auxiliary feature.
    # @param dropout                        Dropout rate. 0.0 means no dropout applied.
    # @param bias                           Whether to use bias parameter in conv layer.
    # @param use_weight_norm                Whether to use weight norm.
    #                                       Applied to all of the conv layers if True.
    # @param use_causal_conv                Whether to use causal structure.
    # @param upsample_conditional_features  Whether to use upsampling network.
    # @param upsample_net                   Upsampling network architecture.
    # @param upsample_params                Upsampling network parameters.
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        aux_context_window: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
        upsample_conditional_features: bool = True,
        upsample_net: str = "ConvInUpsampleNetwork",
        upsample_params: Dict[str, Any] = {"upsample_scales": [4, 4, 4, 4]},
    ) -> None:
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features:
            upsample_params.update(
                {
                    "use_causal_conv": use_causal_conv,
                }
            )
            if upsample_net == "MelGANGenerator":
                assert aux_context_window == 0
                upsample_params.update(
                    {
                        "use_weight_norm": False,
                        "use_final_nonlinear_activation": False,
                    }
                )
                self.upsample_net = ConvInUpsampleNetwork(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update(
                        {
                            "aux_channels": aux_channels,
                            "aux_context_window": aux_context_window,
                        }
                    )
                self.upsample_net = ConvInUpsampleNetwork(**upsample_params)
            self.upsample_factor = math.floor(
                torch.prod(torch.tensor(upsample_params["upsample_scales"])).item()
            )
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    ## Calculate forward propagation.
    # @param    z Input noise signal (B, 1, T).
    # @param    c Local conditioning auxiliary features (B, C ,T').
    # @return   Output tensor (B, out_channels, T)
    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == z.size(-1)

        # encode to hidden representation
        x = self.first_conv(z)
        skips = 0.0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    ## Remove weight normalization module from all of the layers
    def remove_weight_norm(self) -> None:

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    ## Apply weight normalization module from all of the layers
    def apply_weight_norm(self) -> None:

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    ## Perform Inverence
    # @param    c                   Local conditioning auxiliary features (T' ,C).
    # @param    x                   Input noise signal (T, 1).
    # @param    normalize_before    Whether to perform normalization.
    # @return   Output tensor (T, out_channels)
    def inference(
        self,
        c: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        normalize_before: bool = False,
    ) -> torch.Tensor:
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(
                next(self.parameters()).device
            )
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            if normalize_before:
                c = (c - self.mean) / self.scale
            assert c is not None
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        assert c is not None
        return self.forward(x, c).squeeze(0).transpose(1, 0)
