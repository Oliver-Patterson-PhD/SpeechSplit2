from typing import Iterable, List, Optional, Tuple

import pyworld
import torch
import torchaudio
import torchvision
from pysptk.sptk import rapt

from transcribers.whisper.audio import log_mel_spectrogram
from util.logging import Logger

N_FFT: int = 1024
HOP_LENGTH: int = 256
DIM_FREQ: int = 80
FREQ_MIN: int = 90
FREQ_MAX: int = 7600
SAMPLE_RATE: int = 16000
HI_PASS_CUTOFF: int = 30


vtlp_fft: int = N_FFT * 2
torch_stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    window_fn=torch.hann_window,
    power=1,
)
torch_melbasis = torchaudio.transforms.MelScale(
    n_stft=N_FFT // 2 + 1,
    n_mels=DIM_FREQ,
    sample_rate=SAMPLE_RATE,
    f_min=FREQ_MIN,
    f_max=FREQ_MAX,
    mel_scale="htk",
    norm=None,
)
min_level = torch.exp(-100 / 20 * torch.log(torch.tensor(10)))
vtlp_window = torch.hann_window(vtlp_fft)


class MelSpec:
    def __init__(
        self,
        device: torch.device,
    ) -> None:
        self.to_melscale = torchaudio.transforms.MelScale()
        self.de_melscale = torchaudio.transforms.InverseMelScale()
        self.to_spec = torchaudio.transforms.Spectrogram()
        self.de_spec = torchaudio.transforms.GriffinLim()


def has_nans(
    x: torch.Tensor,
) -> str:
    return "Has NaNs" if x.isnan().any().item() else "No NaNs"


def is_nan(
    x: torch.Tensor,
) -> bool:
    return True if x.isnan().any().item() else False


def any_nans(
    xi: Iterable[torch.Tensor],
) -> bool:
    for x in xi:
        if is_nan(x):
            return True
    return False


def filter_wav(
    x: torch.Tensor,
) -> torch.Tensor:
    return torchaudio.functional.highpass_biquad(x, SAMPLE_RATE, HI_PASS_CUTOFF)


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
    x: torch.Tensor,
    num_bins: int = 256,
) -> torch.Tensor:
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = x <= 0
    x[uv] = 0
    x[x >= 1] = 1
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins - 1)) + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins + 1)


def get_spmel(wav: torch.Tensor, whispercheck: bool = True) -> torch.Tensor:
    if whispercheck:
        return torch.nn.functional.pad(
            log_mel_spectrogram(
                wav,
                DIM_FREQ,
                0,
                wav.device,
            ).T,
            (0, 0, 0, 1),
        )
    else:
        return torch_melbasis(torch_stft(wav)).T


def get_spenv(
    wav: torch.Tensor,
    cutoff: int = 3,
) -> torch.Tensor:
    spec = torch_stft(wav).T
    ceps = torch.fft.irfft(torch.log(spec + 1e-6), axis=-1).to(dtype=torch.double)
    lifter = torch.zeros(ceps.shape[1], dtype=torch.double)
    lifter[:cutoff] = 1
    lifter[cutoff] = 0.5
    mmul = torch.matmul(ceps, torch.diag(lifter).to(ceps.device))
    expfft = torch.exp(torch.fft.rfft(mmul, axis=-1))
    maxval = torch.maximum(min_level, torch.abs(expfft))
    env = zero_one_norm((20 * torch.log10(maxval) - 16 + 100) / 100)
    retval = torchaudio.functional.resample(
        env,
        orig_freq=env.size(dim=-1),
        new_freq=DIM_FREQ,
    )
    if __debug__ and is_nan(retval):
        Logger().error("Tensors with Nans: ")
        Logger().trace_nans(mmul)
        Logger().trace_nans(expfft)
        Logger().trace_nans(maxval)
        Logger().trace_nans(env)
        Logger().trace_nans(retval)
    return retval


def extract_f0(
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
            HOP_LENGTH,
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
    f0_norm = speaker_normalization(
        f0_rapt,
        index_nonzero,
        mean_f0,
        std_f0,
    )
    return f0_norm


def zero_one_norm(
    s: torch.Tensor,
) -> torch.Tensor:
    s_norm = s - torch.min(s)
    s_norm /= torch.max(s_norm)
    return s_norm


def get_world_params(
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


def average_f0s(
    f0s: List[torch.Tensor],
) -> List[torch.Tensor]:
    f0_voiced: torch.Tensor = torch.tensor([])
    for i, f0 in enumerate(f0s):
        v = f0 > 0
        f0_voiced = torch.cat(
            (f0_voiced.to(f0.device), f0[v]),
        )
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


def tensor2onehot(
    x: torch.Tensor,
) -> torch.Tensor:
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
    alpha: Optional[float],
) -> torch.Tensor:
    if alpha is None:
        alpha = 0.2 * torch.rand(1).item() + 0.9
    vtlp_stft = torch.stft(
        x,
        n_fft=vtlp_fft,
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
        n_fft=vtlp_fft,
        window=vtlp_window,
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


def clip(
    x: torch.Tensor,
    min: float,
    max: float,
) -> torch.Tensor:
    x[x > max] = max
    x[x < min] = min
    return x


def save_tensor(tensor: torch.Tensor, save_path: str) -> None:
    image = try_image(tensor)
    if image is not None:
        im_min = image.min()
        im_max = image.max()
        norm_image = 1.0 / (im_max - im_min) * image + 1.0 * im_min / (im_min - im_max)
        torchvision.utils.save_image(norm_image, save_path)
    else:
        torch.save(tensor, save_path.rsplit(",", 1)[0] + ".pth")
    return


def try_image(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    elif tensor.dim() == 2:
        return tensor
    elif tensor.dim() < 2:
        return None
    elif tensor.dim() > 2 and tensor.size(0) == 1:
        for in_tensor in tensor:
            return try_image(in_tensor)
    return None


def norm_audio(x: torch.Tensor) -> torch.Tensor:
    return (((x - x.min()) / (x.max() - x.min())) * 2) - 1


def masked_mse(
    prediction: torch.Tensor,
    ground_t: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.flatten()
    ground_t = ground_t.flatten()
    mask: torch.Tensor = ground_t != 0.0
    sum: torch.Tensor = torch.nn.functional.mse_loss(
        prediction,
        ground_t,
        reduction="sum",
    )
    return sum / mask.sum()
