import torch
import torchaudio

from synthesizers.synthesizer import Synthesizer
from util.config import Config


class GriffinLim(Synthesizer):
    n_fft = 1024
    hop_length = 256
    dim_freq = 80
    f_min = 90
    f_max = 7600
    power = 1
    sample_rate = 16000
    n_iter = 64

    def __init__(
        self,
        device: torch.device,
        config: Config,
    ) -> None:
        self.demel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.dim_freq,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=None,
            mel_scale="htk",
            driver="gels",
        )
        self.glim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            n_iter=self.n_iter,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            power=self.power,
        )

    @torch.no_grad()
    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        tspec = spect.T
        return self.glim(self.demel(tspec))
