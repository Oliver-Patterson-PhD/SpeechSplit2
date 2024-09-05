from tomllib import load as loadtoml

import torch
from synthesizers.synthesizer import Synthesizer
from tqdm import tqdm
from util.config import Config
from util.logging import Logger
from wavenet_vocoder import builder


class WavenetSynthesizer(Synthesizer):
    device: torch.device
    model: torch.nn.Module
    model_name: str = "lj_wavenet_vocoder"
    checkpoint_path: str = "full_models"
    configtoml: dict
    config: Config

    def __init__(self, device: torch.device, config: Config) -> None:
        self.config = config
        data_dir = self.config.paths.trained_models
        config_file = f"{data_dir}/{self.model_name}.toml"
        self.configtoml = loadtoml(open(config_file, "rb"))
        self.model = getattr(builder, "wavenet")(
            out_channels=self.configtoml["out_channels"],
            layers=self.configtoml["layers"],
            stacks=self.configtoml["stacks"],
            residual_channels=self.configtoml["residual_channels"],
            gate_channels=self.configtoml["gate_channels"],
            skip_out_channels=self.configtoml["skip_out_channels"],
            cin_channels=self.configtoml["cin_channels"],
            gin_channels=self.configtoml["gin_channels"],
            weight_normalization=self.configtoml["weight_normalization"],
            n_speakers=self.configtoml["n_speakers"],
            dropout=self.configtoml["dropout"],
            kernel_size=self.configtoml["kernel_size"],
            upsample_conditional_features=self.configtoml[
                "upsample_conditional_features"
            ],
            upsample_scales=self.configtoml["upsample_scales"],
            freq_axis_kernel_size=self.configtoml["freq_axis_kernel_size"],
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

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.make_generation_fast_()
        model_out = self.model.incremental_forward(
            torch.zeros(1, 1, 1).fill_(0.0).to(self.device),
            c=spect.T.to(dtype=torch.float).unsqueeze(0).to(self.device),
            g=None,
            T=spect.shape[0] * self.configtoml["hop_size"],
            tqdm=tqdm,
            softmax=True,
            quantize=True,
            log_scale_min=self.configtoml["log_scale_min"],
        )
        return model_out.view(-1)
