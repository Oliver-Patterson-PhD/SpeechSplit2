import torch
from parallel_wavegan.models import ParallelWaveGANGenerator


class Synthesizer(object):
    device: torch.device
    model: ParallelWaveGANGenerator

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = ParallelWaveGANGenerator(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            layers=30,
            stacks=3,
            residual_channels=64,
            gate_channels=128,
            skip_channels=64,
            aux_channels=80,
            aux_context_window=2,
            dropout=0.0,
            bias=True,
            use_weight_norm=True,
            use_causal_conv=False,
            upsample_conditional_features=True,
            upsample_net="ConvInUpsampleNetwork",
            upsample_params={"upsample_scales": [4, 4, 4, 4]},
        )
        self.model = self.model.to(self.device)

    def load_ckpt(
        self, ckpt_path: str = "full_models/ParallelWaveGan/lj_parallelwavegan-1M.pkl"
    ) -> None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"]["generator"])

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outwav = self.model.inference(c=spect).view(-1)
        return outwav
