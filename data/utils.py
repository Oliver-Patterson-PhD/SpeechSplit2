__all__ = [
    "AudioProcs",
]

import os
from math import floor
from typing import List, Optional, Self, Tuple

import pyworld
import torch
import torchaudio
from pysptk.sptk import rapt

from util import Compute, Config, norm_audio

from .dataset import DatasetParser


class AudioProcs:
    min_level = torch.exp(-100 / 20 * torch.log(torch.tensor(10)))

    def __init__(
        self: Self,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config()
        self.compute = Compute()
        self.parser = DatasetParser(config=config)
        self.__dim_freq = config.model.dim_freq
        self.__n_fft = config.audio.n_fft
        self.__sample_rate = config.audio.sample_rate
        self.__freq_min = config.audio.freq_min
        self.__freq_max = config.audio.freq_max
        self.__hop_length = config.audio.hop_len
        self.__vtlp_fft = config.audio.vtlp_fft
        self.__hi_pass_cutoff = config.audio.hi_pass_cutoff
        self.__vtlp_window = torch.hann_window(self.__vtlp_fft)
        self.__f0_m_lo = config.audio.f0_m_lo
        self.__f0_m_hi = config.audio.f0_m_hi
        self.__f0_f_lo = config.audio.f0_f_lo
        self.__f0_f_hi = config.audio.f0_f_hi
        self.__fold_div = config.audio.fold_div
        self.__sample_rate = config.audio.sample_rate
        self.__max_len_pad = config.audio.max_len_pad
        self.__vad_transform = torchaudio.transforms.Vad(
            sample_rate=self.__sample_rate,
        )
        self.__stft = torchaudio.transforms.Spectrogram(
            n_fft=self.__n_fft,
            win_length=self.__n_fft,
            hop_length=self.__hop_length,
            window_fn=torch.hann_window,
            power=1,
        )
        self.__melbasis = torchaudio.transforms.MelScale(
            n_stft=self.__n_fft // 2 + 1,
            n_mels=self.__dim_freq,
            sample_rate=self.__sample_rate,
            f_min=self.__freq_min,
            f_max=self.__freq_max,
            mel_scale="htk",
            norm=None,
        )
        self.__demel = torchaudio.transforms.InverseMelScale(
            n_stft=self.__n_fft // 2 + 1,
            n_mels=self.__dim_freq,
            sample_rate=self.__sample_rate,
            f_min=self.__freq_min,
            f_max=self.__freq_max,
            norm=None,
            mel_scale="htk",
            driver="gels",
        )
        self.__simplewindow = torch.hann_window(self.__n_fft)
        return

    def stft(
        self: Self,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stft(
            input=wav,
            n_fft=self.__n_fft,
            hop_length=self.__hop_length,
            win_length=self.__n_fft,
            window=self.__simplewindow.to(wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def istft(
        self: Self,
        spec: torch.Tensor,
    ) -> torch.Tensor:
        return torch.istft(
            input=spec,
            n_fft=self.__n_fft,
            hop_length=self.__hop_length,
            win_length=self.__n_fft,
            window=self.__simplewindow.to(spec.device),
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
        spec = torch.abs(self.stft(wav)).T
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
                new_freq=self.__dim_freq,
            ),
            (0, 0, 0, 1),
        )
        return retval.to(dtype=wav.dtype)

    def rev_spmel(
        self: Self,
        logspec: torch.Tensor,
    ) -> torch.Tensor:
        logspec = (logspec * 4) - 4
        melspec = torch.pow(10, logspec)
        mags = self.__demel(melspec)
        return mags

    def get_spmel(
        self: Self,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.__melbasis = self.__melbasis.to(self.compute.device())
        rawspec: torch.Tensor = self.stft(wav.float())
        mags: torch.Tensor = rawspec.abs()
        phases: torch.Tensor = rawspec.angle()
        mel_spec = self.__melbasis(mags.to(self.compute.device())).T
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
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
            n_fft=self.__vtlp_fft,
            window=self.__vtlp_window,
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
            n_fft=self.__vtlp_fft,
            window=self.__vtlp_window,
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
        lo: int,
        hi: int,
        fs: Optional[int] = None,
        normalise: bool = True,
    ) -> torch.Tensor:
        if fs is None:
            fs = self.__sample_rate
        f0_rapt = torch.tensor(
            rapt(
                wav.cpu().numpy() * 32768,
                fs,
                self.__hop_length,
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
        wav: torch.Tensor,
    ) -> torch.Tensor:
        return torchaudio.functional.highpass_biquad(
            wav,
            self.__sample_rate,
            self.__hi_pass_cutoff,
        )

    def get_monotonic_wav(
        self: Self,
        wav: torch.Tensor,
        f0: torch.Tensor,
        sp: torch.Tensor,
        ap: torch.Tensor,
        fs: Optional[int] = None,
    ) -> torch.Tensor:
        if fs is None:
            fs = self.__sample_rate
        y = torch.tensor(
            pyworld.synthesize(
                f0.cpu().numpy(),
                sp.cpu().numpy(),
                ap.cpu().numpy(),
                fs,
            ),
            device=wav.device,
        )
        if len(y) < len(wav):
            y = torch.nn.functional.pad(y, (0, len(wav) - len(y)))
        assert len(y) >= len(wav)
        return y[: len(wav)]

    def get_world_params(
        self: Self,
        wav: torch.Tensor,
        fs: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if fs is None:
            fs = self.__sample_rate
        x = wav.squeeze().double().cpu().numpy()
        _f0, t = pyworld.dio(x, fs)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        sp = pyworld.cheaptrick(x, f0, t, fs)
        ap = pyworld.d4c(x, f0, t, fs)
        return (
            torch.tensor(f0, device=wav.device),
            torch.tensor(sp, device=wav.device),
            torch.tensor(ap, device=wav.device),
        )

    def get_f0_lohi(
        self: Self,
        speaker: str,
    ) -> Tuple[int, int]:
        if self.parser.sex(speaker) == "M":
            return self.__f0_m_lo, self.__f0_m_hi
        elif self.parser.sex(speaker) == "F":
            return self.__f0_f_lo, self.__f0_f_hi
        else:
            raise ValueError

    def fold_pad(
        self: Self,
        item: torch.Tensor,
        fold_size: int,
    ) -> torch.Tensor:
        fold_step = fold_size // self.__fold_div
        fold_pad_len = floor((1 - (1 / self.__fold_div)) * fold_size)
        pads: Tuple[int, ...]
        opads: Tuple[int, ...]
        item.squeeze_()
        if item.ndim == 1:
            dim = -1
            pad_i = (
                ((item.size(dim) // fold_size) + 1) * fold_size
                - item.size(dim)
                + fold_pad_len
            )
            pads = (fold_pad_len, pad_i)
            opads = (0, fold_size - item.size(dim))
        elif item.ndim == 2:
            dim = -2
            pad_i = (
                ((item.size(dim) // fold_size) + 1) * fold_size
                - item.size(dim)
                + fold_pad_len
            )
            pads = (0, 0, fold_pad_len, pad_i)
            opads = (0, 0, 0, fold_size - item.size(dim))
        else:
            raise ValueError
        if item.size(dim=dim) > fold_size:
            full_pad = torch.nn.functional.pad(item, pads)
            ones_mat = torch.ones_like(full_pad)
            norm_mat = ones_mat.unfold(dimension=dim, size=fold_size, step=fold_step)
            retunf = full_pad.unfold(dimension=dim, size=fold_size, step=fold_step)
            retval = (retunf / norm_mat).mT
            if dim == -1:
                retval.transpose_(0, 1)
        else:
            retval = torch.nn.functional.pad(item, opads).unsqueeze(0)
        return retval

    def getraw(
        self: Self,
        full_fname: str | os.PathLike,
    ) -> torch.Tensor:
        x: torch.Tensor
        inaud, sr = torchaudio.load(full_fname, channels_first=True)
        assert sr == self.__sample_rate
        try:
            x = self.clean_audio(inaud)
        except Exception as e:
            raise Exception(f"failed to load: {full_fname}") from e
        if x.shape[0] % self.__hop_length == 0:
            x = torch.cat(
                (x, torch.tensor([1e-10], device=x.device)),
                dim=0,
            )
        return x

    def clean_audio(
        self: Self,
        audio: torch.Tensor,
    ):
        return torchaudio.sox_effects.apply_effects_tensor(
            self.__vad_transform(
                torchaudio.sox_effects.apply_effects_tensor(
                    self.__vad_transform(norm_audio(audio)),
                    self.__sample_rate,
                    [["reverse"]],
                )[0]
            ),
            self.__sample_rate,
            [["reverse"]],
        )[0].squeeze()

    def has_content(self: Self, audio: torch.Tensor) -> bool:
        return bool(
            (audio.size(dim=-1) > 1)
            and (audio.max().item() > 1e-03)
            and (audio != 0).any()
        )

    def combine(
        self: Self,
        listitem: List[torch.Tensor],
    ) -> torch.Tensor:
        fold_size = self.__max_len_pad
        fold_step = fold_size // self.__fold_div
        if len(listitem) == 1:
            return listitem[0]
        item_mod = torch.stack([item for item in listitem]).transpose(0, -1)
        fold_fn = torch.nn.Fold(
            output_size=(1, ((item_mod.size(-1) + 1) * fold_step)),
            kernel_size=(1, fold_size),
            stride=(1, fold_step),
        )
        norm_mod = torch.ones_like(item_mod)
        folded = fold_fn(item_mod).squeeze(1).squeeze(1).T
        out_norm = fold_fn(norm_mod).squeeze(1).squeeze(1).T
        return folded / out_norm
