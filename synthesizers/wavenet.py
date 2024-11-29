from tomllib import load as loadtoml
from typing import List, Self

import torch
from tqdm import tqdm

from models.wavenet_vocoder import WaveNet as WavenetGenerator
from util import Config

from .synthesizer import Synthesizer


class WavenetConfig:
    out_channels: int
    layers: int
    stacks: int
    residual_channels: int
    gate_channels: int
    skip_out_channels: int
    cin_channels: int
    gin_channels: int
    weight_normalization: bool
    n_speakers: int
    dropout: float
    kernel_size: int
    upsample_conditional_features: bool
    upsample_scales: List[int]
    freq_axis_kernel_size: int
    scalar_input: bool
    hop_size: int
    log_scale_min: float
    legacy: bool

    def __init__(self: Self, configdict: dict) -> None:
        self.__dict__.update(configdict)


class Wavenet(Synthesizer):
    device: torch.device
    model: torch.nn.Module
    model_name: str = "wavenet_vocoder"
    checkpoint_path: str = "full_models"
    configtoml: dict
    config: Config

    def __init__(
        self: Self,
        device: torch.device,
        config: Config,
    ) -> None:
        self.config = config
        data_dir = self.config.paths.full_models
        config_file = f"{data_dir}/{self.model_name}.toml"
        self.wavconf = WavenetConfig(loadtoml(open(config_file, "rb")))
        self.model = WavenetGenerator(
            out_channels=self.wavconf.out_channels,
            layers=self.wavconf.layers,
            stacks=self.wavconf.stacks,
            residual_channels=self.wavconf.residual_channels,
            gate_channels=self.wavconf.gate_channels,
            skip_out_channels=self.wavconf.skip_out_channels,
            cin_channels=self.wavconf.cin_channels,
            gin_channels=self.wavconf.gin_channels,
            weight_normalization=self.wavconf.weight_normalization,
            n_speakers=self.wavconf.n_speakers,
            dropout=self.wavconf.dropout,
            kernel_size=self.wavconf.kernel_size,
            upsample_conditional_features=self.wavconf.upsample_conditional_features,
            upsample_scales=self.wavconf.upsample_scales,
            freq_axis_kernel_size=self.wavconf.freq_axis_kernel_size,
            scalar_input=True,
            legacy=True,
        )
        self.device = device
        ckpt = torch.load(
            f"{data_dir}/{self.model_name}.pth",
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.make_generation_fast_()

    @torch.no_grad()
    def spect2wav(
        self: Self,
        spect: torch.Tensor,
    ) -> torch.Tensor:
        model_out = self.model.incremental_forward(
            initial_input=None,
            c=spect.mT.to(dtype=torch.float).unsqueeze(0).to(self.device),
            g=None,
            T=spect.shape[0] * self.wavconf.hop_size,
            tqdm=tqdm,
            softmax=True,
            quantize=True,
            log_scale_min=self.wavconf.log_scale_min,
        )
        return model_out.view(-1)
