import json
from typing import List, Tuple

import pyworld
import scipy
import torch
import torchaudio
from pysptk.sptk import rapt

n_fft = 1024
hop_length = 256
dim_freq = 80
f_min = 90
f_max = 7600

torch_stft = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    window_fn=torch.hann_window,
    power=1,
)
torch_melbasis = torchaudio.transforms.MelScale(
    n_mels=dim_freq,
    sample_rate=16000,
    n_stft=n_fft // 2 + 1,
    f_min=f_min,
    f_max=f_max,
)
min_level = torch.exp(-100 / 20 * torch.log(torch.tensor(10)))
vtlp_window = torch.hann_window(2048)


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def dict2json(d, file_w):
    j = json.dumps(d, indent=4)
    with open(file_w, "w") as w_f:
        w_f.write(j)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def speaker_normalization(
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


def quantize_f0_torch(
    x: torch.Tensor, num_bins: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = x <= 0
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins + 1), x.view(B, -1).long()


def filter_wav(x: torch.Tensor) -> torch.Tensor:
    bn, an = butter_highpass(30, 16000, order=5)
    a = torch.tensor(an, device=x.device, dtype=x.dtype)
    b = torch.tensor(bn, device=x.device, dtype=x.dtype)
    y = torchaudio.functional.filtfilt(x, a, b)
    wav = y * 0.96 + (torch.rand(y.shape[0]) - 0.5) * 1e-06
    return wav


def get_spmel(wav: torch.Tensor) -> torch.Tensor:
    return torch_melbasis(torch_stft(wav))


def get_spenv(
    wav: torch.Tensor,
    cutoff: int = 3,
) -> torch.Tensor:
    ceps = torch.fft.irfft(
        torch.log(torch_stft(wav).T + 1e-6),
        axis=-1,
    ).real.to(dtype=torch.double)
    lifter = torch.zeros(
        ceps.shape[1],
        dtype=torch.double,
    )
    lifter[:cutoff] = 1
    lifter[cutoff] = 0.5
    env = zero_one_norm(
        (
            20
            * torch.log10(
                torch.maximum(
                    min_level,
                    torch.abs(
                        torch.exp(
                            torch.fft.rfft(
                                torch.matmul(
                                    ceps,
                                    torch.diag(lifter),
                                ),
                                axis=-1,
                            )
                        )
                    ),
                )
            )
            - 16
            + 100
        )
        / 100
    )
    return torchaudio.functional.resample(
        env,
        orig_freq=env.size(dim=-1),
        new_freq=dim_freq,
    )


def extract_f0(
    wav: torch.Tensor,
    fs: int,
    lo: int,
    hi: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    f0_rapt = torch.tensor(
        rapt(
            wav.cpu().numpy() * 32768,
            fs,
            256,
            min=lo,
            max=hi,
            otype=2,
        ),
        device=wav.device,
    )
    index_nonzero = f0_rapt != -1e10
    nonzero_rapt = f0_rapt[index_nonzero]
    if len(index_nonzero) == 0 or len(nonzero_rapt) == 0:
        mean_f0 = std_f0 = -1e10
    else:
        mean_f0 = nonzero_rapt.mean().item()
        std_f0 = nonzero_rapt.std().item()
    f0_norm = speaker_normalization(
        f0_rapt,
        index_nonzero,
        mean_f0,
        std_f0,
    )
    return f0_rapt, f0_norm


def zero_one_norm(
    s: torch.Tensor,
) -> torch.Tensor:
    s_norm = s - torch.min(s)
    s_norm /= torch.max(s_norm)
    return s_norm


def get_world_params(
    x: torch.Tensor,
    fs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xn = x.squeeze().double().cpu().numpy()
    estimated_f0, temporal_pos = pyworld.dio(xn, fs)
    refined_f0 = pyworld.stonemask(
        xn,
        estimated_f0,
        temporal_pos,
        fs,
    )
    spectral_envelope = pyworld.cheaptrick(
        xn,
        refined_f0,
        temporal_pos,
        fs,
    )
    aperiodicity = pyworld.d4c(
        xn,
        refined_f0,
        temporal_pos,
        fs,
    )
    return (
        torch.tensor(refined_f0, device=x.device),
        torch.tensor(spectral_envelope, device=x.device),
        torch.tensor(aperiodicity, device=x.device),
    )


def average_f0s(
    f0s: List[torch.Tensor],
) -> List[torch.Tensor]:
    f0_voiced: torch.Tensor = torch.tensor([])
    for i, f0 in enumerate(f0s):
        v = f0 > 0
        f0_voiced = torch.cat((f0_voiced.to(f0.device), f0[v]))
    f0_avg = torch.mean(f0_voiced)

    def mapfn(
        f0: torch.Tensor,
    ) -> torch.Tensor:
        v = f0 > 0
        uv = f0 <= 0
        if any(v):
            f0 = torch.ones_like(f0) * f0_avg
            f0[uv] = 0
        else:
            f0 = torch.zeros_like(f0)
        return f0

    f0s = [mapfn(f0) for f0 in f0s]
    return f0s


def get_monotonic_wav(
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


def tensor2onehot(x: torch.Tensor) -> torch.Tensor:
    indices = torch.argmax(x, dim=-1)
    return torch.nn.functional.one_hot(indices, x.size(-1))


def warp_freq(
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


def vtlp(
    x: torch.Tensor,
    fs: int,
    alpha,
) -> torch.Tensor:
    if alpha is None:
        alpha = 0.2 * torch.rand(1).item() + 0.9
    vtlp_stft = torch.stft(
        x,
        n_fft=2048,
        window=vtlp_window,
        return_complex=True,
    ).T
    dtype = vtlp_stft.dtype
    shape_t, shape_k = vtlp_stft.shape
    f_warps = warp_freq(
        shape_k,
        fs,
        alpha=alpha,
    )
    f_warps *= (shape_k - 1) / max(f_warps)
    new_S = torch.zeros([shape_t, shape_k], dtype=dtype)
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
        n_fft=2048,
        window=vtlp_window,
    )
    if len(x) <= len(y):
        y = y[: len(x)]
    else:
        y = torch.nn.functional.pad(y, (0, len(x) - len(y)), mode="constant", value=0)
    return y


def clip(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    idx_over = x > max
    idx_under = x < min
    x[idx_over] = max
    x[idx_under] = min
    return x
