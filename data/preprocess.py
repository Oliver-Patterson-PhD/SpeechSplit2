__all__ = [
    "PreProcess",
]

import os
from typing import Optional, Self

import torch

from util import Config, Logger
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
        if config is None:
            config = Config()
        self.config = config
        self.logger = Logger()
        self.in_path = config.paths.raw_wavs
        self.out_path = config.paths.features
        self.max_len_pad = config.audio.max_len_pad
        self.hop_length = config.audio.hop_len
        self.proc = AudioProcs(config=config)
        self.parser = DatasetParser(config=config)
        procdata_exists = all(
            [
                os.path.exists(f"{self.out_path}/freqs/{speaker}")
                for speaker in self.parser.speakers()
            ]
        )
        if procdata_exists and not config.options.regenerate_data:
            self.logger.info("Preprocessing Skipped")
            return
        spk_dir_list = next(os.walk(config.paths.raw_wavs))[1]
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
        wav = self.proc.filter_wav(
            self.proc.getraw(os.path.join(self.in_path, spk_dir, fname))
        )
        if not self.proc.has_content(wav):
            self.logger.warn(f"No Content: {fname}")
            return
        lo, hi = self.proc.get_f0_lohi(spk_dir)
        f0, sp, ap = self.proc.get_world_params(wav=wav)

        wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
        if not self.proc.has_content(wav_mono):
            raise ValueError

        spmel, phase = self.proc.get_spmel(wav)
        if not self.proc.has_content(spmel):
            raise ValueError

        f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        if not self.proc.has_content(f0_norm):
            raise ValueError

        if len(spmel) != len(f0_norm):
            if (len(spmel) - 1) == len(f0_norm):
                spmel = spmel[:-1]
            else:
                self.logger.fatal(
                    f"melspec and f0 lengths do not match for {fname}\n"
                    f"spmel: {len(spmel)}\n"
                    f"f0_norm: {len(f0_norm)}\n"
                )
                raise Exception(
                    f"melspec and f0 lengths do not match for {fname}\n"
                    f"spmel: {len(spmel)}\n"
                    f"f0_norm: {len(f0_norm)}\n"
                )
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
        fullwavs = os.path.join(self.out_path, "fullwavs", spk_dir)
        monowavs = os.path.join(self.out_path, "monowavs", spk_dir)
        spmels = os.path.join(self.out_path, "spmels", spk_dir)
        freqs = os.path.join(self.out_path, "freqs", spk_dir)
        phases = os.path.join(self.out_path, "phases", spk_dir)
        os.makedirs(fullwavs, exist_ok=True)
        os.makedirs(monowavs, exist_ok=True)
        os.makedirs(spmels, exist_ok=True)
        os.makedirs(freqs, exist_ok=True)
        os.makedirs(phases, exist_ok=True)
        for idx, (wav_fu_i, wav_mo_i, spmel_i, f0_i) in enumerate(
            zip(wav_full_split, wav_mono_split, spmel_split, f0_split)
        ):
            filename = f"{os.path.splitext(fname)[0]}_{idx}.pt"
            if (
                self.proc.has_content(wav_fu_i)
                and self.proc.has_content(wav_mo_i)
                and self.proc.has_content(spmel_i)
                and self.proc.has_content(f0_i)
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
