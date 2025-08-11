from typing import Any

import matplotlib
import matplotlib.ticker as ticker
import torch
import torchaudio

from ..data import parser
from ..util import config, logger
from ..util.file import (basename, exists, newpath, path, rm_rf, strip_path,
                         walkdirs, walkfiles)
from ..util.tensor import Tensor


class Immediate:
    run: bool = False
    clean_data_before_run: bool = False
    exit_after: bool = False
    batch_test: bool = False
    batch_graph: bool = False
    single_test: bool = False
    make_clean: bool = False
    graph_clean: bool = True
    clean_path: str
    fs: int

    def __init__(self) -> None:
        self.experiment_dir = newpath(config.paths.artefacts, "immediate")
        return

    def check_single(self) -> None:
        return

    def test(self) -> None:
        self.check_single()
        if not self.run:
            return
        self.in_path = config.paths.raw_wavs
        self.out_path = config.paths.features
        self.max_len_pad = config.audio.max_len_pad
        self.hop_length = config.audio.hop_len
        self.fs = config.audio.sample_rate
        logger.debug(f"In  Path: {self.in_path}")
        logger.debug(f"Out Path: {self.out_path}")
        speakers = set(
            spk for spk in walkdirs(self.in_path) if spk in parser.speakers()
        )
        logger.info(f"Found {len(speakers)} speakers")
        for spk_idx, spk_dir in enumerate(speakers):
            logger.info(f"Processing {spk_idx + 1:>2}/{len(speakers):>2} {spk_dir}")
            if self.single_test:
                self.run_single_test(spk_dir)
            if self.make_clean:
                self.run_make_clean(spk_dir)
            if self.graph_clean:
                self.run_graph_clean(spk_dir)
        logger.info("Immediate Test Complete")

    def subdir(self, subdir: str) -> str:
        subpath = path(self.experiment_dir, subdir)
        rm_rf(subpath)
        return newpath(subpath)

    def run_single_test(self, spk_dir: str):
        try:
            [
                self.process_file(filebase)  # type: ignore [func-returns-value]
                for filebase in sorted(
                    path(self.in_path, spk_dir, fname)
                    for fname in walkfiles(path(self.in_path, spk_dir))
                )
            ]
        except Exception as e:
            logger.error(f"Failure in single_test: {e.__str__()}")

    def run_make_clean(self, spk_dir: str):
        try:
            self.clean_path = path(self.experiment_dir, "clean_dataset")
            procdata_exists = all(
                [
                    exists(path(self.clean_path, speaker))
                    for speaker in parser.speakers()
                ]
            )
            if procdata_exists and not self.clean_data_before_run:
                logger.info("Clean Data Generation Skipped")
                return
            self.subdir("clean_dataset")
            [
                self.save_cleaned_audio(filebase)  # type: ignore [func-returns-value]
                for filebase in logger.progress_bar(
                    sorted(
                        path(self.in_path, spk_dir, fname)
                        for fname in walkfiles(path(self.in_path, spk_dir))
                    )
                )
            ]
        except Exception as e:
            logger.error(f"Failure in make_clean: {e.__str__()}")
            raise e

    def run_graph_clean(self, spk_dir: str):
        try:
            from data.preprocess import preprocess_data

            preprocess_data(config)
            self.clean_path = config.paths.cleanwavs

            out_dir = self.subdir("cleanup")
            bad_dir = self.subdir("bad")
            [
                self.graph_cleaned_audio(
                    out_dir=out_dir,
                    bad_dir=bad_dir,
                    fname=filebase,
                )  # type: ignore [func-returns-value]
                for filebase in logger.progress_bar(
                    sorted(
                        path(self.in_path, spk_dir, fname)
                        for fname in walkfiles(path(self.in_path, spk_dir))
                    )
                )
            ]
        except Exception as e:
            logger.error(f"Failure in graph_clean: {e.__str__()}")
            raise e

    def save_cleaned_audio(self, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        fullpath = path(self.in_path, spk_dir, fname)
        sfname = basename(fname)
        outpath = path(self.clean_path, f"{sfname}.wav")
        proc = self.proc.full_load_parts(fullpath)
        test = self.proc.full_load_check(*proc)
        if test is not None:
            logger.warn(f"Failure in {test}: {sfname}")
            return
        _, _, _, clean_wav = proc
        torchaudio.save(
            uri=outpath,
            src=clean_wav.unsqueeze(0).cpu(),
            sample_rate=16000,
        )

    def debug_audio(self, audio: Tensor, name: str) -> tuple[tuple[Tensor, str], ...]:
        audio = audio.squeeze()
        spect = self.proc.get_spmel(audio)[0].squeeze().mT
        return (
            (audio, f"{name} Audio"),
            (spect, f"{name} Spectrum"),
        )

    def plot_waveform(
        self, ax: matplotlib.axes.Axes, item: Tensor, name: str, n_samples: int
    ) -> matplotlib.axes.Axes:
        item[item == 0.0] = float("nan")
        ax.plot(
            [i / self.fs for i in range(item.size(-1))],
            item.cpu().numpy(),
        )
        ax.set_xlim(0, item.size(dim=-1) / self.fs)
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Amplitude (A.U.)")
        ax.set_title(name, loc="left")
        return ax

    def plot_melspec(
        self, ax: matplotlib.axes.Axes, item: Tensor, name: str, n_samples: int
    ) -> matplotlib.axes.Axes:
        endtime = n_samples / self.fs
        xvals = torch.linspace(0, endtime, item.size(dim=-1) + 1)
        yvals = self.proc.melmap().tolist()[1:]
        ax.pcolormesh(
            xvals,
            yvals,
            item.cpu().numpy(),
            shading="flat",
        )
        ax.set_xlim(0, endtime)
        ax.set_ylim(config.audio.freq_min, self.fs / 2)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(name, loc="left")
        return ax

    def make_plot(
        self, ax: matplotlib.axes.Axes, item: Tensor, name: str, n_samples: int
    ) -> matplotlib.axes.Axes:
        if item.dim() == 1:
            ax = self.plot_waveform(ax=ax, item=item, name=name, n_samples=n_samples)
        elif item.dim() == 2:
            ax = self.plot_melspec(ax=ax, item=item, name=name, n_samples=n_samples)
        else:
            raise RuntimeError(f"Invalid Tensor with shape: {item.size()}")
        return ax

    def graph_cleaned_audio(self, out_dir: str, bad_dir: str, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        sfname = basename(fname)
        rawpath = path(self.in_path, spk_dir, fname)
        raw_wav, nonoise, nopop, cln_wav = self.proc.full_load_parts(rawpath, keep=True)
        failure = self.proc.full_load_check(raw_wav, nonoise, nopop, cln_wav, keep=True)
        if failure is not None:
            logger.warn(f"Failure in {failure}: {fname}")
            if exists(path(self.clean_path, fname)):
                logger.warn("Failure is in immediate data")
            if exists(path(config.paths.cleanwavs, parser.speaker(fname), fname)):
                logger.warn("Failure is in clean data")
        plot_items: list[tuple[tuple[Tensor, str], ...]] = [
            self.debug_audio(raw_wav, "Raw"),
            self.debug_audio(nonoise, "Noisereduced"),
            self.debug_audio(nopop, "Pop-Free"),
            self.debug_audio(cln_wav, "Clean"),
        ]
        fig = matplotlib.pyplot.figure()
        word = parser.get_real_text(sfname)
        fig.set_size_inches(24, 12)
        fig.set_dpi(300)
        fig.suptitle(f"Sample: {sfname} ({word})")
        n_rows = 4
        n_cols = 2
        [
            self.make_plot(
                ax=matplotlib.pyplot.subplot(
                    n_rows, n_cols, (n_cols * row_idx) + col_idx + 1
                ),
                item=item,
                name=name,
                n_samples=len(row_item[0][0]),
            )
            for row_idx, row_item in enumerate(plot_items)
            for col_idx, (item, name) in enumerate(row_item)
        ]
        fig.tight_layout()
        fig.savefig(path(out_dir if failure is None else bad_dir, f"{sfname}.png"))
        matplotlib.pyplot.close(fig=fig)

    def process_file(self, fname: str) -> None:
        spk_dir: str = fname.split("/")[-2]
        wav_prc = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            full_path = path(self.in_path, spk_dir, fname)
            wav_prc = self.proc.full_load_parts(full_path)[-1]
            lo, hi = self.proc.get_f0_lohi(parser.sex(spk_dir))
            f0, sp, ap = self.proc.get_world_params(wav=wav_prc)
            wav_mono = self.proc.get_monotonic_wav(wav=wav_prc, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav_prc)
            f0_norm = self.proc.extract_f0(wav=wav_prc, lo=lo, hi=hi)
        except Exception as e:
            logger.error(f"Failed to generate: {fname}, {e.__str__()}")
            return
        logger.info(
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
                    self.proc.only_content(wav_prc),
                    self.proc.only_content(wav_mono),
                    self.proc.only_content(spmel),
                    self.proc.only_content(f0_norm),
                ],
            )
        )


def autocorrelation(signal: Tensor) -> Tensor:
    n = len(signal)
    signal_padded = torch.nn.functional.pad(signal, (0, n - 1))
    signal_start = torch.nn.functional.pad(signal, (n - 1, 0))
    correlation = torch.fft.irfft(
        torch.fft.rfft(signal_start) * torch.fft.rfft(signal_padded).conj()
    )
    return correlation[:n]


def iter_autocorrelation(signal: Tensor) -> Tensor:
    n = len(signal)
    correlation = torch.zeros(n, device=signal.device)
    for lag in range(n):
        correlation[lag] = torch.sum(signal[: n - lag] * signal[lag:])
    return correlation


def format_log_message(
    fname: str, data_list: list[float], other: list[Any] = []
) -> str:
    precision = 5
    width = precision + 5
    format_string = "{:<25} " + " ".join(
        ["{{:>{}.{}f}}".format(width, precision)] * len(data_list)
        + ["{{!r:>{}}}".format(width)] * len(other)
    )
    log_message = format_string.format(fname, *data_list, *other)
    return log_message
