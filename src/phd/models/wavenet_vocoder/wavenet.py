# coding: utf-8
from __future__ import absolute_import, print_function, with_statement

import math
from typing import Callable, List, Optional, Self, Iterable

import torch

from .mixture import sample_from_discretized_mix_logistic
from .modules import Conv1d1x1, ConvTranspose2d, Embedding, ResidualConv1dGLU


## Expand global conditioning features to all time steps
# @param B      Batch size.
# @param T      Time length.
# @param g      Global features, (B x C) or (B x C x 1).
# @param bct    returns (B x C x T) if True, otherwise (B x T x C)
# @return       Tensor B x C x T or B x T x C or None
def _expand_global_features(
    B: int,
    T: int,
    g: Optional[torch.Tensor],
    bct: bool = True,
) -> torch.Tensor | None:
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


## Compute receptive field size
# @param total_layers   total layers
# @param num_cycles     cycles
# @param kernel_size    kernel size
# @param dilation       lambda to compute dilation factor. ``lambda x : 1`` to disable dilated convolution.
# @return               receptive field size in sample
def receptive_field_size(
    total_layers: int,
    num_cycles: int,
    kernel_size: int,
    dilation: Callable[[int], int] = lambda x: 2**x,
) -> int:
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


## The WaveNet model that supports local and global conditioning.
class WaveNet(torch.nn.Module):
    conv_layers: Iterable[torch.nn.Module]

    ## Initialiser
    # @param out_channels                   Output channels.
    #                                       If input_type is mu-law quantized one-hot vecror.
    #                                       This must equal to the quantize channels.
    #                                       Other wise num_mixtures x 3 (pi, mu, log_scale).
    # @param layers                         Number of total layers
    # @param stacks                         Number of dilation cycles
    # @param residual_channels              Residual input / output channels
    # @param gate_channels                  Gated activation channels.
    # @param skip_out_channels              Skip connection channels.
    # @param kernel_size                    Kernel size of convolution layers.
    # @param dropout                        Dropout probability.
    # @param cin_channels                   Local conditioning channels.
    #                                       If negative value is set, local conditioning is disabled.
    # @param gin_channels                   Global conditioning channels.
    #                                       If negative value is set, global conditioning is disabled.
    # @param n_speakers                     Number of speakers.
    #                                       Used only if global conditioning is enabled.
    # @param weight_normalization           If True, DeepVoice3-style weight normalization is applied.
    # @param upsample_conditional_features  Whether upsampling local conditioning features by transposed convolution layers or not.
    # @param upsample_scales                List of upsample scale. ``prod(upsample_scales)`` must equal to hop size.
    #                                       Used only if upsample_conditional_features is enabled.
    # @param freq_axis_kernel_size          Freq-axis kernel_size for transposed convolution layers for upsampling. If you only care about time-axis upsampling, set this to 1.
    #
    # @param scalar_input                   If True, scalar input ([-1, 1]) is expected, otherwise quantized one-hot vector is expected.
    # @param use_speaker_embedding          Use speaker embedding or Not.
    #                                       Set to False if you want to disable embedding layer and use external features directly.
    # @param legacy                         Use legacy code or not.
    #                                       Default is True for backward compatibility.
    def __init__(
        self: Self,
        out_channels: int = 256,
        layers: int = 20,
        stacks: int = 2,
        residual_channels: int = 512,
        gate_channels: int = 512,
        skip_out_channels: int = 512,
        kernel_size: int = 3,
        dropout: float = 1 - 0.95,
        cin_channels: int = -1,
        gin_channels: int = -1,
        n_speakers: Optional[int] = None,
        weight_normalization: bool = True,
        upsample_conditional_features: bool = False,
        upsample_scales: Optional[List[int]] = None,
        freq_axis_kernel_size: int = 3,
        scalar_input: bool = False,
        use_speaker_embedding: bool = True,
        legacy: bool = True,
    ) -> None:
        super(WaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.legacy = legacy
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)

        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels,
                gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=skip_out_channels,
                bias=True,  # magenda uses bias, but musyoku doesn't
                dilation=dilation,
                dropout=dropout,
                cin_channels=cin_channels,
                gin_channels=gin_channels,
                weight_normalization=weight_normalization,
            )
            self.conv_layers.append(conv)
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(
                    skip_out_channels,
                    skip_out_channels,
                    weight_normalization=weight_normalization,
                ),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(
                    skip_out_channels,
                    out_channels,
                    weight_normalization=weight_normalization,
                ),
            ]
        )

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1
            )
        else:
            self.embed_speakers = None

        # Upsample conv net
        if upsample_conditional_features:
            self.upsample_conv: Optional[torch.nn.ModuleList] = torch.nn.ModuleList()
            assert upsample_scales is not None
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(
                    1,
                    1,
                    (freq_axis_kernel_size, s),
                    padding=(freq_axis_padding, 0),
                    dilation=1,
                    stride=(1, s),
                    weight_normalization=weight_normalization,
                )
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(torch.nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self: Self) -> bool:
        return self.embed_speakers is not None

    def local_conditioning_enabled(self: Self) -> bool:
        return self.cin_channels > 0

    ## Forward step
    #
    # @param x          One-hot encoded audio signal, shape (B x C x T)
    # @param c          Local conditioning features, shape (B x cin_channels x T)
    # @param g          Global conditioning features, shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
    #                   Note that `self.use_speaker_embedding` must be False when you want to disable embedding layer and use external features directly (e.g., one-hot vector).
    #                   Also type of input tensor must be FloatTensor, not LongTensor in case of `self.use_speaker_embedding` equals False.
    # @param softmax    Whether applies softmax or not.
    #
    # @returns output, shape B x out_channels x T
    def forward(
        self: Self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        softmax: bool = False,
    ) -> torch.Tensor:
        batch_size, _, time_size = x.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(batch_size, -1))
                # (B x gin_channels x 1)
                assert g is not None
                g = g.transpose(1, 2)
                assert g.dim() == 3
        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(batch_size, time_size, g, bct=True)

        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            assert c is not None
            c = c.squeeze(1)
            assert c.size(-1) == x.size(-1)

        # Feed data to network
        x = self.first_conv(x)
        skips = None
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            if skips is None:
                skips = h
            else:
                skips += h
                if self.legacy:
                    skips *= math.sqrt(0.5)

        assert isinstance(skips, torch.Tensor)
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        x = torch.torch.nn.functional.softmax(x, dim=1) if softmax else x

        return x

    ## Incremental forward step
    #
    # Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
    # to (B x T x C) internally and fed to the network for each time step.
    # Input of each time step will be of shape (B x 1 x C).
    #
    # @param initial_input  Initial decoder input, (B x C x 1)
    # @param c              Local conditioning features, shape (B x C' x T)
    # @param g              Global conditioning features, shape (B x C'' or B x C''x 1)
    # @param time_steps     Number of time steps to generate.
    # @param test_inputs    Teacher forcing inputs (for debugging)
    # @param tqdm           tqdm
    # @param softmax        Whether applies softmax or not
    # @param quantize       Whether quantize softmax output before feeding the network output to input for the next time step.
    # @param log_scale_min  Log scale minimum value.
    #
    # @return   Generated one-hot encoded samples.
    #           B x C x T or scaler vector B x 1 x T
    def incremental_forward(
        self: Self,
        initial_input: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        time_steps: int = 100,
        test_inputs: Optional[torch.Tensor] = None,
        tqdm: Callable = lambda x: x,
        softmax: bool = True,
        quantize: bool = True,
        log_scale_min: float = -7.0,
    ) -> torch.Tensor:
        self.clear_buffer()
        batch_size = 1
        # Note: B = batch_size
        # Note: T = test_size
        # Note: shape should be **(B x T x C)**, not (B x C x T) opposed to
        # batch forward due to linealized convolution
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            else:
                if test_inputs.size(1) == self.out_channels:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            batch_size = test_inputs.size(0)
            if time_steps is None:
                time_steps = test_inputs.size(1)
            else:
                time_steps = max(time_steps, test_inputs.size(1))
        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(batch_size, -1))
                # (B x gin_channels, 1)
                assert g is not None
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(batch_size, time_steps, g, bct=False)
        # Local conditioning
        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            assert c is not None
            c = c.squeeze(1)
            assert c.size(-1) == time_steps
        if c is not None and c.size(-1) == time_steps:
            c = c.transpose(1, 2).contiguous()
        outputs: List[torch.Tensor] = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = torch.zeros(batch_size, 1, 1)
            else:
                initial_input = torch.zeros(batch_size, 1, self.out_channels)
                initial_input[:, :, 127] = 1  # TODO: is this ok?
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()
        else:
            if initial_input.size(1) == self.out_channels:
                initial_input = initial_input.transpose(1, 2).contiguous()
        current_input = initial_input
        for t in tqdm(range(time_steps), desc="WaveNet Increments"):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            # Conditioning features for single time step
            ct = None if c is None else c[:, t, :].unsqueeze(1)
            gt = None if g_btc is None else g_btc[:, t, :].unsqueeze(1)
            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = None
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                if self.legacy:
                    skips = h if skips is None else (skips + h) * math.sqrt(0.5)
                else:
                    skips = h if skips is None else (skips + h)
            assert isinstance(skips, torch.Tensor)
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)
            # Generate next input by sampling
            assert self.scalar_input is True
            x = sample_from_discretized_mix_logistic(
                x.view(batch_size, -1, 1), log_scale_min=log_scale_min
            )
            outputs += [x.data]
        # T x B x C
        toutputs = torch.stack(outputs)
        # B x C x T
        toutputs = toutputs.transpose(0, 1).transpose(1, 2).contiguous()
        self.clear_buffer()
        return toutputs

    def clear_buffer(self: Self) -> None:
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self: Self) -> None:
        def remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
