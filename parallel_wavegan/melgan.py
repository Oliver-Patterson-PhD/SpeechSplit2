from typing import Any, Dict, List

import torch

from .causal_conv1d import CausalConv1d
from .causal_conv_transpose1d import CausalConvTranspose1d
from .residual_stack import ResidualStack


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    ## Initialize MelGANGenerator module.
    # @param in_channels                     Number of input channels.
    # @param out_channels                    Number of output channels.
    # @param kernel_size                     Kernel size of initial and final conv layer.
    # @param channels                        Initial number of channels for conv layer.
    # @param bias                            Whether to add bias parameter in convolution layers.
    # @param upsample_scales                 List of upsampling scales.
    # @param stack_kernel_size               Kernel size of dilated conv layers in residual stack.
    # @param stacks                          Number of stacks in a single residual stack.
    # @param nonlinear_activation            Activation function module name.
    # @param nonlinear_activation_params     Hyperparameters for activation function.
    # @param pad                             Padding function module name before dilated convolution layer.
    # @param pad_params                      Hyperparameters for padding function.
    # @param use_final_nonlinear_activation  Activation function for the final layer.
    # @param use_weight_norm                 Whether to use weight norm. (applied to all of the conv layers.)
    # @param use_causal_conv                 Whether to use causal convolution.
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        kernel_size: int = 7,
        channels: int = 512,
        bias: bool = True,
        upsample_scales: List[int] = [8, 8, 2, 2],
        stack_kernel_size: int = 3,
        stacks: int = 3,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: dict = {},
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
    ) -> None:
        super(MelGANGenerator, self).__init__()

        # check hyper parameters is valid
        assert channels >= torch.prod(torch.tensor(upsample_scales))
        assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd."

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
                ),
            ]
        else:
            layers += [
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    ## Calculate forward propagation.
    # @param c Input tensor (B, channels, T).
    # @return Output tensor (B, 1, T ** prod(upsample_scales)).
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.melgan(c)

    ## Reset parameters.
    def reset_parameters(self):

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)

    ## Perform inference.
    # @param c                 Input tensor (T, in_channels).
    # @param normalize_before  Whether to perform normalization.
    # @return Output tensor (T ** prod(upsample_scales), out_channels).
    def inference(
        self,
        c: torch.Tensor,
        normalize_before: bool = False,
    ) -> torch.Tensor:
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)
