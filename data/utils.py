__all__ = [
    "AudioProcs",
]

from typing import Optional, Self, Tuple

import pyworld
import torch
import torchaudio
from pysptk.sptk import rapt

from util import Config, Compute


class AudioProcs:
    dim_freq: int
    stft: torchaudio.transforms.Spectrogram
    melbasis: torchaudio.transforms.MelScale
    min_level = torch.exp(-100 / 20 * torch.log(torch.tensor(10)))

    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        self.compute = Compute()
        self.dim_freq = config.model.dim_freq
        self.n_fft = config.audio.n_fft
        self.sample_rate = config.audio.sample_rate
        self.freq_min = config.audio.freq_min
        self.freq_max = config.audio.freq_max
        self.hop_len = config.audio.hop_len
        self.vtlp_fft = config.audio.vtlp_fft
        self.hi_pass_cutoff = config.audio.hi_pass_cutoff
        self.vtlp_window = torch.hann_window(self.vtlp_fft)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_len,
            window_fn=torch.hann_window,
            power=1,
        )
        self.melbasis = torchaudio.transforms.MelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.dim_freq,
            sample_rate=self.sample_rate,
            f_min=self.freq_min,
            f_max=self.freq_max,
            mel_scale="htk",
            norm=None,
        )
        self.demel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.dim_freq,
            sample_rate=self.sample_rate,
            f_min=self.freq_min,
            f_max=self.freq_max,
            norm=None,
            mel_scale="htk",
            driver="gels",
        )
        self.simplewindow = torch.hann_window(self.n_fft)
        return

    def simplestft(
        self: Self,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stft(
            input=wav,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=self.simplewindow.to(wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def simpleistft(
        self: Self,
        spec: torch.Tensor,
    ) -> torch.Tensor:
        return torch.istft(
            input=spec,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=self.simplewindow.to(spec.device),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )

    def get_spenv(
        self: Self,
        wav: torch.Tensor,
        cutoff: int = 3,
    ) -> torch.Tensor:
        spec = torch.abs(self.simplestft(wav)).T
        ceps = torch.fft.irfft(torch.log(spec + 1e-6), axis=-1).to(dtype=torch.double)
        lifter = torch.zeros(ceps.shape[1], dtype=torch.double)
        lifter[:cutoff] = 1
        lifter[cutoff] = 0.5
        mmul = torch.matmul(ceps, torch.diag(lifter).to(ceps.device))
        expfft = torch.exp(torch.fft.rfft(mmul, axis=-1))
        maxval = torch.maximum(self.min_level, torch.abs(expfft))
        env = self.zero_one_norm((20 * torch.log10(maxval) - 16 + 100) / 100)
        retval = torch.nn.functional.pad(
            torchaudio.functional.resample(
                env,
                orig_freq=env.size(dim=-1),
                new_freq=self.dim_freq,
            ),
            (0, 0, 0, 1),
        )
        return retval.to(dtype=wav.dtype)

    def get_spmel(
        self: Self,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.melbasis = self.melbasis.to(self.compute.device())
        rawspec: torch.Tensor = self.simplestft(wav.float())
        mags: torch.Tensor = rawspec.abs()
        phases: torch.Tensor = rawspec.angle()
        mel_spec = self.melbasis(mags.to(self.compute.device())).T
        log_spec = torchaudio.functional.amplitude_to_DB(
            x=mel_spec,
            multiplier=20,
            amin=self.min_level.to(self.compute.device()),
            db_multiplier=1,
        )
        # log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        # log_spec = (log_spec + 4.0) / 4.0
        return (
            torch.nn.functional.pad(log_spec, (0, 0, 0, 1)).to(dtype=wav.dtype),
            phases,
        )

    def zero_one_norm(
        self: Self,
        s: torch.Tensor,
    ) -> torch.Tensor:
        s_norm = s - torch.min(s)
        s_norm /= torch.max(s_norm)
        return s_norm

    def vtlp(
        self: Self,
        x: torch.Tensor,
        fs: int,
        alpha: Optional[float],
    ) -> torch.Tensor:
        if alpha is None:
            alpha = 0.2 * torch.rand(1).item() + 0.9
        vtlp_stft = torch.stft(
            x,
            n_fft=self.vtlp_fft,
            window=self.vtlp_window,
            return_complex=True,
        ).T
        dtype = vtlp_stft.dtype
        shape_t, shape_k = vtlp_stft.shape
        f_warps = self.warp_freq(
            shape_k,
            fs,
            alpha=alpha,
        )
        f_warps *= (shape_k - 1) / max(f_warps)
        new_S = torch.zeros(
            [shape_t, shape_k],
            dtype=dtype,
            device=vtlp_stft.device,
        )
        for k in range(shape_k):
            # first and last freq
            if k == 0 or k == shape_k - 1:
                new_S[:, k] += vtlp_stft[:, k]
            else:
                warp_up = f_warps[k] - torch.floor(f_warps[k])
                warp_down = 1 - warp_up
                pos = int(torch.floor(f_warps[k]))
                new_S[:, pos] += warp_down * vtlp_stft[:, k]
                new_S[:, pos + 1] += warp_up * vtlp_stft[:, k]
        y = torch.istft(
            new_S.T,
            n_fft=self.vtlp_fft,
            window=self.vtlp_window,
        )
        if len(x) <= len(y):
            y = y[: len(x)]
        else:
            y = torch.nn.functional.pad(
                y,
                (0, len(x) - len(y)),
                mode="constant",
                value=0,
            )
        return y

    def warp_freq(
        self: Self,
        n_fft: int,
        fs: int,
        fhi: int = 4800,
        alpha: float = 0.9,
    ) -> torch.Tensor:
        bins = torch.linspace(0, 1, n_fft)
        f_warps = []
        scale = fhi * min(alpha, 1)
        f_boundary = scale / alpha
        fs_half = fs // 2
        for k in bins:
            f_ori = k * fs
            if f_ori <= f_boundary:
                f_warp = f_ori * alpha
            else:
                f_warp = fs_half - (
                    (fs_half - scale) / (fs_half - scale / alpha) * (fs_half - f_ori)
                )
            f_warps.append(f_warp)
        return torch.Tensor(f_warps)

    def extract_f0(
        self: Self,
        wav: torch.Tensor,
        fs: int,
        lo: int,
        hi: int,
        normalise: bool = True,
    ) -> torch.Tensor:
        f0_rapt = torch.tensor(
            rapt(
                wav.cpu().numpy() * 32768,
                fs,
                self.hop_len,
                min=lo,
                max=hi,
                otype=2,
            ),
            device=wav.device,
        )
        if not normalise:
            return f0_rapt
        index_nonzero = f0_rapt != -1e10
        nonzero_rapt = f0_rapt[index_nonzero]
        if len(index_nonzero) == 0 or len(nonzero_rapt) == 0:
            mean_f0 = std_f0 = -1e10
        else:
            mean_f0 = nonzero_rapt.mean().item()
            std_f0 = nonzero_rapt.std().item()
        f0_norm = self.speaker_normalization(
            f0_rapt,
            index_nonzero,
            mean_f0,
            std_f0,
        )
        return f0_norm

    def speaker_normalization(
        self: Self,
        f0: torch.Tensor,
        index_nonzero: torch.Tensor,
        mean_f0: float,
        std_f0: float,
    ) -> torch.Tensor:
        f0.dtype
        std_f0 += 1e-6
        f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
        f0[index_nonzero] = torch.clip(f0[index_nonzero], -1, 1)
        f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
        return f0

    def filter_wav(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torchaudio.functional.highpass_biquad(
            x,
            self.sample_rate,
            self.hi_pass_cutoff,
        )

    def get_monotonic_wav(
        self: Self,
        x: torch.Tensor,
        f0: torch.Tensor,
        sp: torch.Tensor,
        ap: torch.Tensor,
        fs: int,
    ) -> torch.Tensor:
        y = torch.tensor(
            pyworld.synthesize(
                f0.cpu().numpy(),
                sp.cpu().numpy(),
                ap.cpu().numpy(),
                fs,
            ),
            device=x.device,
        )
        if len(y) < len(x):
            y = torch.nn.functional.pad(y, (0, len(x) - len(y)))
        assert len(y) >= len(x)
        return y[: len(x)]

    def get_world_params(
        self: Self,
        x_torch: torch.Tensor,
        fs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x_torch.squeeze().double().cpu().numpy()
        _f0, t = pyworld.dio(x, fs)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        sp = pyworld.cheaptrick(x, f0, t, fs)
        ap = pyworld.d4c(x, f0, t, fs)
        return (
            torch.tensor(f0, device=x_torch.device),
            torch.tensor(sp, device=x_torch.device),
            torch.tensor(ap, device=x_torch.device),
        )
