import os
from glob import glob
from typing import Any, Self

import matplotlib.backends.backend_pdf
import matplotlib.pyplot
import torch
import torchaudio

from data.dataset import DatasetParser
from data.utils import AudioProcs
from util import Config, Logger
from util.file import basename, strip_path, walkdirs, walkfiles


class Immediate:
    config: Config
    exit_after: bool = False
    batch_test: bool = False
    batch_graph: bool = False
    single_test: bool = False
    make_clean: bool = False

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
        self.in_path = self.config.paths.raw_wavs
        self.out_path = self.config.paths.features
        self.max_len_pad = self.config.audio.max_len_pad
        self.hop_length = self.config.audio.hop_len
        self.proc = AudioProcs(config=self.config)
        self.parser = DatasetParser(config=self.config)
        self.logger.debug(f"In  Path: {self.in_path}")
        self.logger.debug(f"Out Path: {self.out_path}")
        speakers = set(
            spk for spk in walkdirs(self.in_path) if spk in self.parser.speakers()
        )
        self.logger.info(f"Found {len(speakers)} speakers")
        for spk_idx, spk_dir in enumerate(speakers):
            self.logger.info(
                f"Processing {spk_idx + 1:>2}/{len(speakers):>2} {spk_dir}"
            )
            batches = set(
                (
                    spk_dir,
                    os.path.join(self.in_path, spk_dir, fname[: len(fname) - 7]),
                )
                for fname in set(walkfiles(os.path.join(self.in_path, spk_dir)))
            )
            if self.batch_graph:
                try:
                    with matplotlib.backends.backend_pdf.PdfPages(
                        os.path.join(self.experiment_dir, f"waveforms-{spk_dir}.pdf")
                    ) as pdf:
                        [
                            pdf.savefig(
                                self.graph_batch(
                                    spk,
                                    set(fname for fname in glob(filebase + "_M*.wav")),
                                    strip_path(filebase),
                                )
                            )
                            for spk, filebase in self.logger.progress_bar(batches)
                        ]
                except Exception as e:
                    self.logger.error(f"Failure in batch_graph: {e.__str__()}")
            if self.batch_test:
                try:
                    [
                        self.process_batch(
                            spk,
                            set(fname for fname in glob(filebase + "_M*.wav")),
                            strip_path(filebase),
                        )
                        for spk, filebase in sorted(batches, key=lambda c: c[1])
                    ]
                except Exception as e:
                    self.logger.error(f"Failure in batch_test: {e.__str__()}")
            if self.single_test:
                try:
                    [
                        self.process_file(filebase)
                        for filebase in sorted(
                            os.path.join(self.in_path, spk_dir, fname)
                            for fname in walkfiles(os.path.join(self.in_path, spk_dir))
                        )
                    ]
                except Exception as e:
                    self.logger.error(f"Failure in single_test: {e.__str__()}")
            if self.make_clean:
                try:
                    dset_path = os.path.join(self.experiment_dir, "clean_dataset")
                    os.makedirs(dset_path, exist_ok=True)
                    [
                        self.save_cleaned_audio(filebase)
                        for filebase in self.logger.progress_bar(
                            sorted(
                                os.path.join(self.in_path, spk_dir, fname)
                                for fname in walkfiles(
                                    os.path.join(self.in_path, spk_dir)
                                )
                            )
                        )
                    ]
                except Exception as e:
                    self.logger.error(f"Failure in make_clean: {e.__str__()}")
                    raise e
        self.logger.info("Immediate Test Complete")

    def save_cleaned_audio(self: Self, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        fullpath = os.path.join(self.in_path, spk_dir, fname)
        sfname = basename(fname)
        outpath = os.path.join(self.experiment_dir, "clean_dataset", f"{sfname}.wav")
        clean_wav = self.proc.load_audio(fullpath).unsqueeze(0).cpu()
        torchaudio.save(
            uri=outpath,
            src=clean_wav,
            sample_rate=16000,
        )

    def process_batch(
        self: Self,
        spk_dir: str,
        batch: set[str],
        origname: str,
    ) -> None:
        wav = torch.stack(
            [
                self.proc.load_audio(
                    os.path.join(self.in_path, spk_dir, strip_path(fname))
                ).squeeze()
                for fname in batch
            ]
        )
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav)
            wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav)
            f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        except Exception as e:
            raise e
        self.logger.info(
            format_log_message(
                origname,
                [
                    wav.std().item(),
                    wav.max().item(),
                    wav.min().item(),
                ],
                [
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
        nrows: int = 7
        ncols: int = 1
        fig.subplots(nrows, ncols)
        for fname in batch:
            idx = int(fname[-5:-4])
            sample = (
                self.proc.load_audio(
                    os.path.join(self.in_path, spk_dir, strip_path(fname))
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
    ) -> None:
        spk_dir: str = fname.split("/")[-2]
        wav_prc = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            wav_prc = self.proc.load_audio(os.path.join(self.in_path, spk_dir, fname))
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav_prc)
            wav_mono = self.proc.get_monotonic_wav(wav=wav_prc, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav_prc)
            f0_norm = self.proc.extract_f0(wav=wav_prc, lo=lo, hi=hi)
        except Exception as e:
            self.logger.error(f"Failed to generate: {fname}, {e.__str__()}")
            return
        energy_prc = torch.tensor([])
        if self.proc.has_content(wav_prc):
            energy_prc = self.proc.short_time_energy(wav_prc)
        self.logger.info(
            format_log_message(
                strip_path(fname),
                [
                    wav_prc.std().item(),
                    wav_prc.max().item(),
                    wav_prc.min().item(),
                ],
                [
                    tuple(wav_prc.size()),
                    tuple(wav_mono.size()),
                    tuple(spmel.size()),
                    tuple(f0_norm.size()),
                    self.proc.has_content(wav_prc),
                    self.proc.has_content(wav_mono),
                    self.proc.has_content(spmel),
                    self.proc.has_content(f0_norm),
                ],
            )
        )
        try:
            sfname = basename(fname)
            fig = matplotlib.pyplot.figure()
            fig.set_size_inches(15.44, 27.45)
            fig.suptitle(f"Sample: {sfname}")
            nrows = 5
            ncols = 1
            fig.subplots(nrows, ncols)
            plot_thing((nrows, ncols, 1), wav_prc, "waveform")
            plot_thing((nrows, ncols, 2), energy_prc, "energy")
            plot_thing((nrows, ncols, 3), wav_mono, "wav_mono")
            if spmel.dim() == 2:
                plot_thing((nrows, ncols, 4), spmel.mT, "spmel")
            plot_thing((nrows, ncols, 5), f0_norm, "f0_norm")
            outpath = os.path.join(self.experiment_dir, f"energy-{sfname}.pdf")
            fig.savefig(outpath)
            matplotlib.pyplot.close()
        except Exception as e:
            self.logger.error(f"Failed to plot: {fname}, {e.__str__()}")


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
    precision = 5
    width = precision + 5
    format_string = "{:<25} " + " ".join(
        ["{{:>{}.{}f}}".format(width, precision)] * len(data_list)
        + ["{{!r:>{}}}".format(width)] * len(other)
    )
    log_message = format_string.format(fname, *data_list, *other)
    return log_message


def plot_thing(subp: tuple[int, int, int], thing: torch.Tensor, title: str) -> None:
    try:
        ax = matplotlib.pyplot.subplot(*subp)
        if thing.dim() == 1:
            ax.plot(thing.squeeze().cpu().numpy())
            ax.set_xlim(0, thing.size(dim=-1))
        elif thing.dim() == 2:
            ax.imshow(
                thing.squeeze().cpu().numpy(),
                interpolation="none",
                aspect="auto",
                origin="lower",
            )
        else:
            raise RuntimeError(f"Invalid Tensor has shape: {thing.size()}")
        ax.set_title(title)
    except Exception as e:
        Logger().error(f"Failed at: {title}, {e.__str__()}")
        return
