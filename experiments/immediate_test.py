import os
from typing import Any, Self, Tuple

import torch

from data.dataset import DatasetParser
from data.utils import AudioProcs
from util import Config, Logger

from .cack import TorchGate as TG


class Immediate:
    config: Config
    exit_after: bool = True

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        return

    def test(self: Self) -> None:
        self.logger = Logger()
        self.in_path = "/mnt/datasets/raw/UASpeech/audio/original"
        self.out_path = self.config.paths.features
        self.max_len_pad = self.config.audio.max_len_pad
        self.hop_length = self.config.audio.hop_len
        self.proc = AudioProcs(config=self.config)
        self.parser = DatasetParser(config=self.config)
        self.logger.debug(f"In  Path: {self.in_path}")
        self.logger.debug(f"Out Path: {self.out_path}")
        spk_dir_list = next(os.walk(self.in_path))[1]
        speakers = [spk for spk in spk_dir_list if spk in self.parser.speakers()]
        self.logger.info(f"Found {len(speakers)} speakers")
        [
            self.process_file(spk_dir=spk_dir, fname=fname)
            for spk_dir in sorted(speakers)
            for fname in sorted(next(os.walk(os.path.join(self.in_path, spk_dir)))[-1])
        ]
        self.logger.info("Preprocessing Complete")
        return

    def process_file(
        self: Self,
        spk_dir: str,
        fname: str,
    ) -> None:
        rawwav = torch.tensor([])
        clnwav = torch.tensor([])
        wav = torch.tensor([])
        energy = torch.tensor([float("nan")])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        period = float("nan")
        auto_corr = None
        try:
            sample_rate = 16000
            frame_length = 400
            hop_length = 100
            tg = TG(sr=sample_rate, nonstationary=True)
            rawwav = tg(self.proc.getraw(os.path.join(self.in_path, spk_dir, fname)))
            clnwav = self.proc.clean_audio(rawwav)
            wav = self.proc.filter_wav(clnwav)
            energy = short_time_energy(wav.squeeze(), frame_length, hop_length)
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav)
            wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav)
            f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
            threshold = adaptive_threshold(energy)
            impulse_indices = torch.nonzero(energy > threshold).flatten()
            refined_peaks = refine_peaks(
                impulse_indices,
                wav,
                frame_length,
                hop_length,
                sample_rate,
            )
            p, auto_corr = analyze_periodicity(
                refined_peaks.to("cpu"),
                sample_rate,
            )
            period = p or 0.0
        except Exception:
            pass
        if auto_corr is None:
            auto_corr = torch.tensor([float("nan")])
        self.logger.info(
            format_log_message(
                fname,
                [
                    auto_corr.std().item(),
                    auto_corr.max().item(),
                    auto_corr.min().item(),
                    period,
                ],
                [
                    self.proc.has_content(rawwav),
                    self.proc.has_content(wav),
                    self.proc.has_content(wav_mono),
                    self.proc.has_content(spmel),
                    self.proc.has_content(f0_norm),
                    len(auto_corr),
                    len(energy),
                    len(wav),
                ],
            )
        )


def short_time_energy(
    audio_data: torch.Tensor,
    frame_length: int,
    hop_length: int,
):
    if len(audio_data.shape) == 1:
        audio_data = audio_data.unsqueeze(0)
    window = torch.ones(1, 1, frame_length, device=audio_data.device)
    energy = torch.nn.functional.conv1d(
        audio_data.unsqueeze(1) ** 2, window, stride=hop_length, padding=0
    ).squeeze()
    return energy


def adaptive_threshold(
    energy: torch.Tensor,
    k: int = 2,
) -> torch.Tensor:
    mean_energy = torch.mean(energy)
    std_energy = torch.std(energy)
    threshold = mean_energy + k * std_energy
    return threshold


def refine_peaks(
    impulse_indices: torch.Tensor,
    audio_data: torch.Tensor,
    frame_length: int,
    hop_length: int,
    sample_rate: int,
    min_distance_ms: int = 10,
) -> torch.Tensor:
    min_distance_samples = int(min_distance_ms * sample_rate / 1000)
    refined_peaks = []
    for index in impulse_indices:
        start = index * hop_length
        end = start + frame_length
        frame = audio_data[start:end]
        peaks = (
            torch.nonzero(
                (torch.diff(torch.sign(torch.diff(torch.abs(frame)))) < 0)
            ).flatten()
            + 1
        )
        if len(peaks) > 0:
            peak_index = peaks[torch.argmax(torch.abs(frame[peaks]))]
            refined_peaks.append(start + peak_index)
    final_peaks = []
    if refined_peaks:
        final_peaks.append(refined_peaks[0])
        for peak in refined_peaks[1:]:
            if peak - final_peaks[-1] >= min_distance_samples:
                final_peaks.append(peak)
    return torch.tensor(final_peaks, dtype=torch.long)


def autocorrelation(signal: torch.Tensor) -> torch.Tensor:
    n = len(signal)
    signal_padded = torch.nn.functional.pad(signal, (0, n - 1))
    signal_start = torch.nn.functional.pad(signal, (n - 1, 0))
    correlation = torch.fft.irfft(
        torch.fft.rfft(signal_start) * torch.fft.rfft(signal_padded).conj()
    )
    return correlation[:n]


def analyze_periodicity(
    peak_locations: torch.Tensor,
    sample_rate: int,
) -> Tuple[float | None, torch.Tensor | None]:
    if len(peak_locations) < 2:
        return None, None
    impulse_train_len = int(peak_locations[-1] + sample_rate * 0.1)
    impulse_train = torch.zeros(impulse_train_len)
    impulse_train[peak_locations] = 1
    auto_corr = autocorrelation(impulse_train)
    peaks = (
        torch.nonzero((torch.diff(torch.sign(torch.diff(auto_corr[1:]))) < 0)).flatten()
        + 1
    )
    if len(peaks) > 0:
        period_samples = peaks[0] + 1
        period_seconds = period_samples / sample_rate
        return period_seconds, auto_corr
    else:
        return None, auto_corr


def format_log_message(
    fname: str,
    data_list: list[float],
    other: list[Any] = [],
) -> str:
    precision = 5
    width = precision + 3
    format_string = "{:<25} " + " ".join(
        ["{{:>{},.{}f}}".format(width, precision)] * len(data_list)
        + ["{{!r:>{}}}".format(width)] * len(other)
    )
    log_message = format_string.format(fname, *data_list, *other)
    return log_message
