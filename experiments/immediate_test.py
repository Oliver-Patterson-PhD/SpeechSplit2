from typing import Any, Self

import torch
import torchaudio

from data.dataset import DatasetParser
from data.utils import AudioProcs
from util import Config, Logger
from util.file import (basename, exists, newpath, path, rm_rf, strip_path,
                       walkdirs, walkfiles)
from util.plot import plot_things


class Immediate:
    run: bool = False
    config: Config
    clean_data_before_run: bool = False
    exit_after: bool = False
    batch_test: bool = False
    batch_graph: bool = False
    single_test: bool = False
    make_clean: bool = False
    graph_clean: bool = True
    clean_path: str

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.experiment_dir = newpath(config.paths.artefacts, "immediate")
        self.logger = Logger()
        return

    def check_single(self: Self) -> None:
        return

    def test(self: Self) -> None:
        self.check_single()
        if not self.run:
            return
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

    def subdir(self: Self, subdir: str) -> str:
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
            self.logger.error(f"Failure in single_test: {e.__str__()}")

    def run_make_clean(self, spk_dir: str):
        try:
            self.clean_path = path(self.experiment_dir, "clean_dataset")
            procdata_exists = all(
                [
                    exists(path(self.clean_path, speaker))
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
                        path(self.in_path, spk_dir, fname)
                        for fname in walkfiles(path(self.in_path, spk_dir))
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
            self.clean_path = self.config.paths.cleanwavs

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
                        path(self.in_path, spk_dir, fname)
                        for fname in walkfiles(path(self.in_path, spk_dir))
                    )
                )
            ]
        except Exception as e:
            self.logger.error(f"Failure in graph_clean: {e.__str__()}")
            raise e

    def save_cleaned_audio(self: Self, fname: str) -> None:
        spk_dir = fname.split("/")[-2]
        fullpath = path(self.in_path, spk_dir, fname)
        sfname = basename(fname)
        outpath = path(self.clean_path, f"{sfname}.wav")
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
        rawpath = path(self.in_path, spk_dir, fname)
        raw_wav, nonoise, nopop, cln_wav = self.proc.full_load_parts(rawpath, keep=True)
        failure = self.proc.full_load_check(raw_wav, nonoise, nopop, cln_wav, keep=True)
        if failure is not None:
            self.logger.warn(f"Failure in {failure}: {fname}")
            if exists(path(self.clean_path, fname)):
                self.logger.warn("Failure is in immediate data")
            if exists(
                path(self.config.paths.cleanwavs, self.parser.speaker(fname), fname)
            ):
                self.logger.warn("Failure is in clean data")
        plot_items = [
            *self.debug_audio(raw_wav, "Raw"),
            *self.debug_audio(nonoise, "Noisereduced"),
            *self.debug_audio(nopop, "Pop-Free"),
            *self.debug_audio(cln_wav, "Clean"),
        ]
        plot_things(out_dir if failure is None else bad_dir, sfname, plot_items)

    def process_file(self: Self, fname: str) -> None:
        spk_dir: str = fname.split("/")[-2]
        wav_prc = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            full_path = path(self.in_path, spk_dir, fname)
            wav_prc = self.proc.full_load_parts(full_path)[-1]
            lo, hi = self.proc.get_f0_lohi(self.parser.sex(spk_dir))
            f0, sp, ap = self.proc.get_world_params(wav=wav_prc)
            wav_mono = self.proc.get_monotonic_wav(wav=wav_prc, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav_prc)
            f0_norm = self.proc.extract_f0(wav=wav_prc, lo=lo, hi=hi)
        except Exception as e:
            self.logger.error(f"Failed to generate: {fname}, {e.__str__()}")
            return
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
