__all__ = [
    "PreProcess",
]

import os
from typing import Optional, Self

import torch
import torchaudio

from util import Config, Logger
from util.file import newpath
from util.patterns import Singleton

from .dataset import DatasetParser
from .utils import AudioProcs


class PreProcess(metaclass=Singleton):
    logger: Logger
    config: Config
    in_path: str
    out_path: str
    proc: AudioProcs
    parser: DatasetParser

    def __init__(
        self: Self,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or Config()
        self.logger = Logger()
        self.in_path = self.config.paths.raw_wavs
        self.out_path = self.config.paths.features
        self.max_len_pad = self.config.audio.max_len_pad
        self.hop_length = self.config.audio.hop_len
        self.proc = AudioProcs(config=self.config)
        self.parser = DatasetParser(config=config)
        self.logger.debug(f"Out Path: {self.out_path}")
        procdata_exists = all(
            [
                os.path.exists(f"{self.out_path}/freqs/{speaker}")
                for speaker in self.parser.speakers()
            ]
        )
        if procdata_exists and not self.config.options.regenerate_data:
            self.logger.info("Preprocessing Skipped")
            return
        spk_dir_list = next(os.walk(self.config.paths.raw_wavs))[1]
        speakers = [spk for spk in spk_dir_list if spk in self.parser.speakers()]
        self.logger.info(f"Found {len(speakers)} speakers")
        [
            self.process_file(spk_dir=spk_dir, fname=fname)
            for spk_dir in sorted(speakers)
            for fname in self.logger.progress_bar(
                sorted(next(os.walk(os.path.join(self.in_path, spk_dir)))[-1]),
                desc=f"items in {spk_dir}",
            )
        ]
        self.logger.info("Preprocessing Complete")

    def process_file(self: Self, spk_dir: str, fname: str) -> None:
        self.logger.trace(f"Processing: {fname}")

        raw_path = os.path.join(self.in_path, spk_dir, fname)
        raw_wav, nonoise, nopop, wav = self.proc.full_load_parts(raw_path, keep=False)
        if not self.proc.is_valid(raw_wav):
            self.logger.warn(f"No Content in raw file: {fname}")
            return
        if not self.proc.is_valid(nonoise):
            self.logger.warn(f"No Content after noisereduction: {fname}")
            return
        if not self.proc.is_valid(nopop):
            self.logger.warn(f"No Content after removing pop: {fname}")
            return
        if not self.proc.only_content(wav):
            self.logger.warn(f"No Content after filtering: {fname}")
            return

        lo, hi = self.proc.get_f0_lohi(spk_dir)
        f0, sp, ap = self.proc.get_world_params(wav=wav)

        wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
        if not self.proc.only_content(wav_mono):
            self.logger.warn(f"Failed to get monotonic wav for: {spk_dir}/{fname}")
            return

        spmel, phase = self.proc.get_spmel(wav)
        if not self.proc.only_content(spmel):
            self.logger.warn(f"Failed to get mel spectrogram for: {spk_dir}/{fname}")
            return

        f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        if not self.proc.only_content(f0_norm):
            self.logger.warn(f"Failed to get f0 for: {spk_dir}/{fname}")
            return

        if len(spmel) != len(f0_norm):
            if (len(spmel) - 1) == len(f0_norm):
                spmel = spmel[:-1]
            else:
                msg = (
                    f"melspec and f0 lengths do not match for {fname}\n"
                    f"spmel: {len(spmel)}\n"
                    f"f0_norm: {len(f0_norm)}\n"
                )
                self.logger.fatal(msg)
                raise Exception(msg)

        wav_full_split = self.proc.fold_pad(
            item=wav,
            fold_size=self.max_len_pad * (self.hop_length - 1),
        )
        wav_mono_split = self.proc.fold_pad(
            item=wav_mono,
            fold_size=self.max_len_pad * (self.hop_length - 1),
        )
        spmel_split = self.proc.fold_pad(
            item=spmel,
            fold_size=self.max_len_pad,
        )
        f0_split = self.proc.fold_pad(
            item=f0_norm,
            fold_size=self.max_len_pad,
        )

        fullwavs = newpath(self.config.paths.fullwavs, spk_dir)
        monowavs = newpath(self.config.paths.monowavs, spk_dir)
        spmels = newpath(self.config.paths.spmels, spk_dir)
        freqs = newpath(self.config.paths.freqs, spk_dir)
        phases = newpath(self.config.paths.phases, spk_dir)
        cleanwavs = newpath(self.config.paths.cleanwavs, spk_dir)
        namebase = os.path.splitext(fname)[0]
        torchaudio.save(
            uri=os.path.join(cleanwavs, f"{namebase}.wav"),
            src=wav.unsqueeze(0),
            sample_rate=16000,
        )
        for idx, (wav_fu_i, wav_mo_i, spmel_i, f0_i) in enumerate(
            zip(wav_full_split, wav_mono_split, spmel_split, f0_split)
        ):
            filename = f"{namebase}_{idx}.pt"
            if (
                self.proc.only_content(wav_fu_i)
                and self.proc.only_content(wav_mo_i)
                and self.proc.only_content(spmel_i)
                and self.proc.only_content(f0_i)
            ):
                torch.save(wav_fu_i.to("cpu"), os.path.join(fullwavs, filename))
                torch.save(wav_mo_i.to("cpu"), os.path.join(monowavs, filename))
                torch.save(spmel_i.to("cpu"), os.path.join(spmels, filename))
                torch.save(f0_i.to("cpu"), os.path.join(freqs, filename))
        filename = f"{os.path.splitext(fname)[0]}.pt"
        torch.save(phase.to("cpu"), os.path.join(phases, filename))


def preprocess_data(
    config: Optional[Config] = None,
) -> None:
    PreProcess(config)
