import os
from math import floor
from typing import Self, Tuple

import torch

from data_preprocessing import getraw, has_content
from util import Config, Logger
from utils import (extract_f0, filter_wav, get_monotonic_wav, get_spmel,
                   get_world_params)


class PreProcess:
    logger: Logger
    config: Config
    in_path: str
    out_path: str
    f0_m_lo: int
    f0_m_hi: int
    f0_f_lo: int
    f0_f_hi: int
    sample_rate: int
    max_len_pad: int
    hop_length: int

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.logger = Logger()
        self.in_path = config.paths.raw_wavs
        self.out_path = config.paths.features
        self.f0_m_lo = config.audio.f0_m_lo
        self.f0_m_hi = config.audio.f0_m_hi
        self.f0_f_lo = config.audio.f0_f_lo
        self.f0_f_hi = config.audio.f0_f_hi
        self.sample_rate = config.audio.sample_rate
        self.max_len_pad = config.audio.max_len_pad
        self.hop_length = config.audio.hop_len
        self.fold_div = 2
        self.spk_meta = getattr(
            __import__("meta_dicts"),
            self.config.options.dataset_name,
        )
        procdata_exists = all(
            [
                os.path.exists(f"{self.out_path}/freqs/{speaker}")
                for speaker in self.spk_meta.keys()
            ]
        )
        if procdata_exists and not config.options.regenerate_data:
            self.logger.info("Preprocessing Skipped")
            return
        spk_dir_list = next(os.walk(config.paths.raw_wavs))[1]
        speakers = [spk for spk in spk_dir_list if spk in self.spk_meta]
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
        wav = filter_wav(getraw(os.path.join(self.in_path, spk_dir, fname)))
        if not has_content(wav):
            self.logger.warn(f"No Content: {fname}")
            return
        lo, hi = self.get_f0_lohi(spk_dir)
        f0, sp, ap = get_world_params(wav, self.sample_rate)

        wav_mono = get_monotonic_wav(wav, f0, sp, ap, self.sample_rate)
        if not has_content(wav_mono):
            raise ValueError

        spmel = get_spmel(wav)
        if not has_content(spmel):
            raise ValueError

        f0_norm = extract_f0(wav, self.sample_rate, lo, hi)
        if not has_content(f0_norm):
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
        wav_mono_split = self.fold_pad(
            item=wav_mono,
            fold_size=self.max_len_pad * (self.hop_length - 1),
        )
        spmel_split = self.fold_pad(
            item=spmel,
            fold_size=self.max_len_pad,
        )
        f0_split = self.fold_pad(
            item=f0_norm,
            fold_size=self.max_len_pad,
        )
        monowavs = os.path.join(self.out_path, "monowavs", spk_dir)
        spmels = os.path.join(self.out_path, "spmels", spk_dir)
        freqs = os.path.join(self.out_path, "freqs", spk_dir)
        os.makedirs(monowavs, exist_ok=True)
        os.makedirs(spmels, exist_ok=True)
        os.makedirs(freqs, exist_ok=True)
        for idx, (wav_mo_i, spmel_i, f0_i) in enumerate(
            zip(wav_mono_split, spmel_split, f0_split)
        ):
            filename = f"{os.path.splitext(fname)[0]}_{idx}.pt"
            if has_content(wav_mo_i) and has_content(spmel_i) and has_content(f0_i):
                torch.save(wav_mo_i, os.path.join(monowavs, filename))
                torch.save(spmel_i, os.path.join(spmels, filename))
                torch.save(f0_i, os.path.join(freqs, filename))

    def get_f0_lohi(self: Self, spk_dir: str) -> Tuple[int, int]:
        if self.spk_meta[spk_dir].sex == "M":
            return self.f0_m_lo, self.f0_m_hi
        elif self.spk_meta[spk_dir].sex == "F":
            return self.f0_f_lo, self.f0_f_hi
        else:
            raise ValueError

    def fold_pad(
        self: Self,
        item: torch.Tensor,
        fold_size: int,
    ) -> torch.Tensor:
        fold_step = fold_size // self.fold_div
        fold_pad_len = floor((1 - (1 / self.fold_div)) * fold_size)
        pads: Tuple[int, ...]
        opads: Tuple[int, ...]
        item.squeeze_()
        if item.ndim == 1:
            dim = -1
            pad_i = (
                ((item.size(dim) // fold_size) + 1) * fold_size
                - item.size(dim)
                + fold_pad_len
            )
            pads = (fold_pad_len, pad_i)
            opads = (0, fold_size - item.size(dim))
        elif item.ndim == 2:
            dim = -2
            pad_i = (
                ((item.size(dim) // fold_size) + 1) * fold_size
                - item.size(dim)
                + fold_pad_len
            )
            pads = (0, 0, fold_pad_len, pad_i)
            opads = (0, 0, 0, fold_size - item.size(dim))
        else:
            raise ValueError
        if item.size(dim=dim) > fold_size:
            full_pad = torch.nn.functional.pad(item, pads)
            ones_mat = torch.ones_like(full_pad)
            norm_mat = ones_mat.unfold(dimension=dim, size=fold_size, step=fold_step)
            retunf = full_pad.unfold(dimension=dim, size=fold_size, step=fold_step)
            retval = (retunf / norm_mat).mT
            if dim == -1:
                retval.transpose_(0, 1)
        else:
            retval = torch.nn.functional.pad(item, opads).unsqueeze(0)
        return retval
