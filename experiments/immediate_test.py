import os
from glob import glob
from typing import Any, Self

import matplotlib.backends.backend_pdf
import matplotlib.pyplot
import torch

from data.dataset import DatasetParser
from data.utils import AudioProcs
from util import Config, Logger

from .cack import TorchGate as TG


class Immediate:
    config: Config
    exit_after: bool = True
    batch_test: bool = True
    batch_graph: bool = False
    single_test: bool = False

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.experiment_dir = os.path.join(self.config.paths.artefacts, "immediate")
        for root, _, files in os.walk(self.experiment_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.makedirs(self.experiment_dir, exist_ok=True)
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
        speakers = set(
            spk
            for spk in next(os.walk(self.in_path))[1]
            if spk in self.parser.speakers()
        )
        self.logger.info(f"Found {len(speakers)} speakers")
        sample_rate = 16000
        tg = TG(sr=sample_rate, nonstationary=True)
        batches = set(
            (
                spk_dir,
                os.path.join(self.in_path, spk_dir, fname[: len(fname) - 7]),
            )
            for spk_dir in speakers
            for fname in set(next(os.walk(os.path.join(self.in_path, spk_dir)))[-1])
        )
        if self.batch_graph:
            with matplotlib.backends.backend_pdf.PdfPages(
                os.path.join(self.experiment_dir, "waveforms.pdf")
            ) as pdf:
                [
                    pdf.savefig(
                        self.graph_batch(
                            spk,
                            set(fname for fname in glob(filebase + "_M*.wav")),
                            filebase.rpartition("/")[-1],
                        )
                    )
                    for spk, filebase in self.logger.progress_bar(batches)
                ]
        if self.batch_test:
            [
                self.process_batch(
                    spk,
                    set(fname for fname in glob(filebase + "_M*.wav")),
                    filebase.rpartition("/")[-1],
                    tg,
                )
                for spk, filebase in sorted(batches, key=lambda c: c[1])
            ]
        if self.single_test:
            [
                self.process_file(filebase, tg)
                for filebase in sorted(
                    os.path.join(self.in_path, spk_dir, fname)
                    for spk_dir in speakers
                    for fname in next(os.walk(os.path.join(self.in_path, spk_dir)))[-1]
                )
            ]
            return
        self.logger.info("Preprocessing Complete")

    def process_batch(
        self: Self,
        spk_dir: str,
        batch: set[str],
        origname: str,
        tg: TG,
    ) -> None:
        rawwavs = torch.stack(
            [
                self.proc.getraw(
                    os.path.join(self.in_path, spk_dir, fname.rpartition("/")[-1])
                ).squeeze()
                for fname in batch
            ]
        )
        wav = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            wav = self.proc.clean_audio(tg(rawwavs))
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav)
            wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav)
            f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        except Exception:
            pass
        self.logger.info(
            format_log_message(
                origname,
                [
                    rawwavs.std().item(),
                    rawwavs.max().item(),
                    rawwavs.min().item(),
                ],
                [
                    rawwavs.size(dim=-1),
                    wav.size(dim=-1),
                    self.proc.has_content(wav),
                    self.proc.has_content(wav_mono),
                    self.proc.has_content(spmel),
                    self.proc.has_content(f0_norm),
                ],
            )
        )

    def graph_batch(
        self: Self,
        spk_dir: str,
        batch: set[str],
        origname: str,
    ):
        fig = matplotlib.pyplot.figure()
        fig.set_size_inches(15.44, 27.45)
        fig.suptitle(f"Sample: {origname}")
        nrows: int = len(batch)
        ncols: int = 1
        fig.subplots(nrows, ncols)
        for fname in batch:
            idx = int(fname[-5:-4])
            sample = (
                self.proc.getraw(
                    os.path.join(self.in_path, spk_dir, fname.rpartition("/")[-1])
                )
                .squeeze()
                .numpy()
            )
            ax = matplotlib.pyplot.subplot(nrows, ncols, idx - 1)
            ax.plot(sample)
            ax.set_title(f"{origname}_M{idx}")
            ax.set_xlim(0, len(sample))
        return fig

    def process_file(
        self: Self,
        fname: str,
        tg: TG,
    ) -> None:
        spk_dir: str = fname.split("/")[-2]
        rawwav = torch.tensor([])
        wav = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            rawwav = self.proc.getraw(os.path.join(self.in_path, spk_dir, fname))
            wav = self.proc.clean_audio(tg(rawwav))
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav)
            wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav)
            f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        except Exception:
            pass
        self.logger.info(
            format_log_message(
                fname.rpartition("/")[-1],
                [
                    rawwav.std().item(),
                    rawwav.max().item(),
                    rawwav.min().item(),
                ],
                [
                    rawwav.size(dim=-1),
                    wav.size(dim=-1),
                    self.proc.has_content(wav),
                    self.proc.has_content(wav_mono),
                    self.proc.has_content(spmel),
                    self.proc.has_content(f0_norm),
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


def autocorrelation(signal: torch.Tensor) -> torch.Tensor:
    n = len(signal)
    signal_padded = torch.nn.functional.pad(signal, (0, n - 1))
    signal_start = torch.nn.functional.pad(signal, (n - 1, 0))
    correlation = torch.fft.irfft(
        torch.fft.rfft(signal_start) * torch.fft.rfft(signal_padded).conj()
    )
    return correlation[:n]


def iter_autocorrelation(signal: torch.Tensor) -> torch.Tensor:
    n = len(signal)
    correlation = torch.zeros(n, device=signal.device)
    for lag in range(n):
        correlation[lag] = torch.sum(signal[: n - lag] * signal[lag:])
    return correlation


def format_log_message(
    fname: str,
    data_list: list[float],
    other: list[Any] = [],
) -> str:
    precision = 3
    width = precision + 3
    format_string = "{:<25} " + " ".join(
        ["{{:>{}.{}f}}".format(width, precision)] * len(data_list)
        + ["{{!r:>{}}}".format(width)] * len(other)
    )
    log_message = format_string.format(fname, *data_list, *other)
    return log_message
