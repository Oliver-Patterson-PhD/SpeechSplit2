import json

import librosa
import numpy
import pysptk
import pyworld
import scipy
import torch

mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T

min_level = numpy.exp(-100 / 20 * numpy.log(10))


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


def stride_wav(x, fft_length=1024, hop_length=256):
    x = numpy.pad(x, int(fft_length // 2), mode="reflect")
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = numpy.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def pySTFT(x, fft_length=1024, hop_length=256):
    result = stride_wav(x, fft_length=fft_length, hop_length=hop_length)
    fft_window = scipy.signal.get_window("hann", fft_length, fftbins=True)
    result = numpy.fft.rfft(fft_window * result, n=fft_length).T
    return numpy.abs(result)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    # index_nonzero = f0 != 0
    f0 = f0.astype(float).copy()
    std_f0 += 1e-6
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = numpy.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def inverse_quantize_f0_numpy(x, num_bins=257):
    assert x.ndim == 2
    assert x.shape[1] == num_bins
    y = numpy.argmax(x, axis=1).astype(float)
    y /= num_bins - 1
    return y


def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim == 1
    x = x.astype(float).copy()
    uv = x <= 0
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = numpy.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = numpy.zeros((len(x), num_bins + 1), dtype=numpy.float32)
    enc[numpy.arange(len(x)), x.astype(numpy.int32)] = 1.0
    return enc, x.astype(numpy.int64)


def quantize_f0_torch(x, num_bins=256):
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


def filter_wav(x, prng):
    b, a = butter_highpass(30, 16000, order=5)
    y = scipy.signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
    return wav


def get_spmel(wav):
    D = pySTFT(wav).T
    D_mel = numpy.dot(D, mel_basis)
    D_db = 20 * numpy.log10(numpy.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100
    return S


def get_spenv(wav, cutoff=3):
    D = pySTFT(wav).T
    ceps = numpy.fft.irfft(numpy.log(D + 1e-6), axis=-1).real  # [T, F]
    F = ceps.shape[1]
    lifter = numpy.zeros(F)
    lifter[:cutoff] = 1
    lifter[cutoff] = 0.5
    lifter = numpy.diag(lifter)
    env = numpy.matmul(ceps, lifter)
    env = numpy.abs(numpy.exp(numpy.fft.rfft(env, axis=-1)))
    env = 20 * numpy.log10(numpy.maximum(min_level, env)) - 16
    env = (env + 100) / 100
    env = zero_one_norm(env)
    env = scipy.signal.resample(env, 80, axis=-1)
    return env


def extract_f0(wav, fs, lo, hi):
    f0_rapt = pysptk.sptk.rapt(
        wav.astype(numpy.float32) * 32768, fs, 256, min=lo, max=hi, otype=2
    )
    index_nonzero = f0_rapt != -1e10
    if len(index_nonzero) == 0:
        mean_f0 = std_f0 = -1e10
    else:
        mean_f0, std_f0 = (
            numpy.mean(f0_rapt[index_nonzero]),
            numpy.std(f0_rapt[index_nonzero]),
        )
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm


def zero_one_norm(S):
    S_norm = S - numpy.min(S)
    S_norm /= numpy.max(S_norm)
    return S_norm


def get_world_params(x, fs=16000):
    _f0, t = pyworld.dio(x, fs)  # raw pitch extractor
    f0 = pyworld.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pyworld.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pyworld.d4c(x, f0, t, fs)  # extract aperiodicity
    return f0, sp, ap


def average_f0s(f0s, mode="global"):
    # average f0s using global mean
    if mode == "global":
        f0_voiced = []  # f0 in voiced frames
        for f0 in f0s:
            v = f0 > 0
            f0_voiced = numpy.concatenate((f0_voiced, f0[v]))
        f0_avg = numpy.mean(f0_voiced)
        for i in range(len(f0s)):
            f0 = f0s[i]
            v = f0 > 0
            uv = f0 <= 0
            if any(v):
                f0 = numpy.ones_like(f0) * f0_avg
                f0[uv] = 0
            else:
                f0 = numpy.zeros_like(f0)
            f0s[i] = f0
    # average f0s using local mean
    elif mode == "local":
        for i in range(len(f0s)):
            f0 = f0s[i]
            v = f0 > 0
            uv = f0 <= 0
            if any(v):
                f0_avg = numpy.mean(f0[v])
                f0 = numpy.ones_like(f0) * f0_avg
                f0[uv] = 0
            else:
                f0 = numpy.zeros_like(f0)
            f0s[i] = f0
    else:
        raise ValueError
    return f0s


def get_monotonic_wav(x, f0, sp, ap, fs=16000):
    # synthesize an utterance using the parameters
    y = pyworld.synthesize(f0, sp, ap, fs)
    if len(y) < len(x):
        y = numpy.pad(y, (0, len(x) - len(y)))
    assert len(y) >= len(x)
    return y[: len(x)]


def tensor2onehot(x):
    indices = torch.argmax(x, dim=-1)
    return torch.nn.functional.one_hot(indices, x.size(-1))


def warp_freq(n_fft, fs, fhi=4800, alpha=0.9):
    bins = numpy.linspace(0, 1, n_fft)
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
    return numpy.array(f_warps)


def vtlp(x, fs, alpha):
    S = librosa.stft(x).T
    T, K = S.shape
    dtype = S.dtype
    f_warps = warp_freq(K, fs, alpha=alpha)
    f_warps *= (K - 1) / max(f_warps)
    new_S = numpy.zeros([T, K], dtype=dtype)
    for k in range(K):
        # first and last freq
        if k == 0 or k == K - 1:
            new_S[:, k] += S[:, k]
        else:
            warp_up = f_warps[k] - numpy.floor(f_warps[k])
            warp_down = 1 - warp_up
            pos = int(numpy.floor(f_warps[k]))
            new_S[:, pos] += warp_down * S[:, k]
            new_S[:, pos + 1] += warp_up * S[:, k]
    y = librosa.istft(new_S.T)
    y = librosa.util.fix_length(data=y, size=len(x))
    return y
