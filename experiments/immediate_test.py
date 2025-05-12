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
from util.file import basename, newpath, strip_path, walkdirs, walkfiles


class Immediate:
    clean_data_before_run: bool = False
    config: Config
    exit_after: bool = True
    batch_test: bool = False
    batch_graph: bool = False
    single_test: bool = False
    make_clean: bool = False
    graph_clean: bool = True
    clean_path: str

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.experiment_dir = newpath(self.config.paths.artefacts, "immediate")
        return

    def subdir(self: Self, subdir: str) -> str:
        subpath = os.path.join(self.experiment_dir, subdir)
        for root, _, files in os.walk(subpath, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.makedirs(subpath, exist_ok=True)
        return subpath

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
            if self.single_test:
                self.run_single_test(spk_dir)
            if self.make_clean:
                self.run_make_clean(spk_dir)
            if self.graph_clean:
                self.run_graph_clean(spk_dir)
        self.logger.info("Immediate Test Complete")

    def run_single_test(self, spk_dir: str):
        try:
            [
                self.process_file(filebase)  # type: ignore [func-returns-value]
                for filebase in sorted(
                    os.path.join(self.in_path, spk_dir, fname)
                    for fname in walkfiles(os.path.join(self.in_path, spk_dir))
                )
            ]
        except Exception as e:
            self.logger.error(f"Failure in single_test: {e.__str__()}")

    def run_make_clean(self, spk_dir: str):
        try:
            self.clean_path = os.path.join(self.experiment_dir, "clean_dataset")
            procdata_exists = all(
                [
                    os.path.exists(f"{self.clean_path}/{speaker}")
                    for speaker in self.parser.speakers()
                ]
            )
            if procdata_exists and not self.clean_data_before_run:
                self.logger.info("Clean Data Generation Skipped")
                return
            self.subdir("clean_dataset")
            [
                self.save_cleaned_audio(filebase)  # type: ignore [func-returns-value]
                for filebase in self.logger.progress_bar(
                    sorted(
                        os.path.join(self.in_path, spk_dir, fname)
                        for fname in walkfiles(os.path.join(self.in_path, spk_dir))
                    )
                )
            ]
        except Exception as e:
            self.logger.error(f"Failure in make_clean: {e.__str__()}")
            raise e

    def run_graph_clean(self, spk_dir: str):
        try:
            from data.preprocess import preprocess_data

            preprocess_data(self.config)

            out_dir = self.subdir("cleanup")
            bad_dir = self.subdir("bad")
            [
                self.graph_cleaned_audio(
                    out_dir=out_dir,
                    bad_dir=bad_dir,
                    fname=filebase,
                )  # type: ignore [func-returns-value]
                for filebase in self.logger.progress_bar(
                    sorted(
                        os.path.join(self.in_path, spk_dir, fname)
                        for fname in walkfiles(os.path.join(self.in_path, spk_dir))
                    )
                )
            ]
        except Exception as e:
            self.logger.error(f"Failure in graph_clean: {e.__str__()}")
            raise e

    def save_cleaned_audio(self: Self, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        fullpath = os.path.join(self.in_path, spk_dir, fname)
        sfname = basename(fname)
        outpath = os.path.join(self.clean_path, f"{sfname}.wav")
        proc = self.proc.full_load_parts(fullpath)
        test = self.proc.full_load_check(*proc)
        if test is not None:
            self.logger.warn(f"Failure in {test}: {sfname}")
            return
        _, _, _, clean_wav = proc
        torchaudio.save(
            uri=outpath,
            src=clean_wav.unsqueeze(0).cpu(),
            sample_rate=16000,
        )

    def debug_audio(
        self, audio: torch.Tensor, name: str
    ) -> list[tuple[torch.Tensor, str]]:
        audio = audio.squeeze()
        spect = self.proc.get_spmel(audio)[0].squeeze().mT
        return [
            (audio, f"{name} Audio"),
            (spect, f"{name} Spectrum"),
        ]

    def graph_cleaned_audio(self: Self, out_dir: str, bad_dir: str, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        sfname = basename(fname)
        rawpath = os.path.join(self.in_path, spk_dir, fname)
        raw_wav, nonoise, nopop, cln_wav = self.proc.full_load_parts(rawpath, keep=True)
        failure = self.proc.full_load_check(raw_wav, nonoise, nopop, cln_wav, keep=True)
        if failure is not None:
            self.logger.warn(f"Failure in {failure}: {fname}")
            if os.path.exists(os.path.join(self.clean_path, fname)):
                self.logger.warn("Failure is in immediate data")
            if os.path.exists(
                os.path.join(
                    self.config.paths.cleanwavs, self.parser.speaker(fname), fname
                )
            ):
                self.logger.warn("Failure is in clean data")
        plot_items = [
            *self.debug_audio(raw_wav, "Raw"),
            *self.debug_audio(nonoise, "Noisereduced"),
            *self.debug_audio(nopop, "Pop-Free"),
            *self.debug_audio(cln_wav, "Clean"),
        ]
        plot_things(out_dir if failure is None else bad_dir, sfname, plot_items)

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
            full_path = os.path.join(self.in_path, spk_dir, fname)
            wav_prc = self.proc.full_load_parts(full_path)[-1]
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav_prc)
            wav_mono = self.proc.get_monotonic_wav(wav=wav_prc, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav_prc)
            f0_norm = self.proc.extract_f0(wav=wav_prc, lo=lo, hi=hi)
        except Exception as e:
            self.logger.error(f"Failed to generate: {fname}, {e.__str__()}")
            return
        energy_prc = torch.tensor([])
        if self.proc.only_content(wav_prc):
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
                    self.proc.only_content(wav_prc),
                    self.proc.only_content(wav_mono),
                    self.proc.only_content(spmel),
                    self.proc.only_content(f0_norm),
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


def plot_things(plot_out: str, sample: str, things: list[tuple[torch.Tensor, str]]):
    nrows = len(things)
    ncols = 1
    fig = matplotlib.pyplot.figure()
    fig.set_size_inches(15.44, 27.45)
    fig.suptitle(f"Sample: {sample}")
    fig.subplots(nrows, ncols)
    for i, (item, name) in enumerate(things):
        plot_thing((nrows, ncols, i + 1), item, name)
    fig.savefig(os.path.join(plot_out, f"{sample}.pdf"))
    matplotlib.pyplot.close()
