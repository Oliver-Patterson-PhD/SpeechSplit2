from tomllib import load as loadtoml

import torch
from tqdm import tqdm
from wavenet_vocoder import builder
from synthesizer.synthesizer import Synthesizer


class WavenetSynthesizer(Synthesizer):
    device: torch.device
    model: torch.nn.Module
    model_name: str = "lj_wavenet_vocoder"
    checkpoint_path: str = "full_models"
    config: dict

    def __init__(self, device: torch.device) -> None:
        config_file = f"{self.checkpoint_path}/{self.model_name}.toml"
        self.config = loadtoml(open(config_file, "rb"))
        self.model = getattr(builder, "wavenet")(
            out_channels=self.config["out_channels"],
            layers=self.config["layers"],
            stacks=self.config["stacks"],
            residual_channels=self.config["residual_channels"],
            gate_channels=self.config["gate_channels"],
            skip_out_channels=self.config["skip_out_channels"],
            cin_channels=self.config["cin_channels"],
            gin_channels=self.config["gin_channels"],
            weight_normalization=self.config["weight_normalization"],
            n_speakers=self.config["n_speakers"],
            dropout=self.config["dropout"],
            kernel_size=self.config["kernel_size"],
            upsample_conditional_features=self.config["upsample_conditional_features"],
            upsample_scales=self.config["upsample_scales"],
            freq_axis_kernel_size=self.config["freq_axis_kernel_size"],
            scalar_input=True,
            legacy=self.config["legacy"],
        )
        self.device = device
        ckpt = torch.load(
            f"{self.checkpoint_path}/{self.model_name}.pth",
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def spect2wav(self, c: torch.Tensor) -> None:
        self.model.eval()
        self.model.make_generation_fast_()
        return self.model.incremental_forward(
            torch.zeros(1, 1, 1).fill_(0.0).to(self.device),
            c=torch.FloatTensor(c.T).unsqueeze(0).to(self.device),
            g=None,
            T=c.shape[0] * self.config["hop_size"],
            tqdm=tqdm,
            softmax=True,
            quantize=True,
            log_scale_min=self.config["log_scale_min"],
        ).view(-1)
