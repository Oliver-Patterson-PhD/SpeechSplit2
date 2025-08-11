__all__ = [
    "preprocess_data",
]

import torch
import torchaudio

from ..util import config, logger
from ..util.file import exists, newpath, path, strip_ext
from ..util.patterns import Singleton
from .audio_procs import processor
from .dataset import parser


class PreProcess(metaclass=Singleton):
    __in_path: str
    __out_path: str

    def __init__(self) -> None:
        self.__in_path = config.paths.raw_wavs
        self.__out_path = config.paths.features
        logger.debug(f"Out Path: {self.__out_path}")
        self.max_len_pad = config.audio.max_len_pad
        self.hop_length = config.audio.hop_len
        self.path_fullwavs = config.paths.fullwavs
        self.path_monowavs = config.paths.monowavs
        self.path_spmels = config.paths.spmels
        self.path_freqs = config.paths.freqs
        self.path_phases = config.paths.phases
        self.path_cleanwavs = config.paths.cleanwavs
        procdata_exists = all(
            [
                exists(path(self.__out_path, "freqs", parser.get_spkdir(speaker)))
                for speaker in parser.speakers()
            ]
        )
        if procdata_exists and not config.options.regenerate_data:
            logger.info("Preprocessing Skipped")
            return
        speakers = parser.speakers()
        logger.info(f"Found {len(speakers)} speakers")
        [
            self.process_file(spk=spk, fname=fname)  # type: ignore[func-returns-value]
            for spk in sorted(speakers)
            for fname in logger.progress_bar(
                sorted(parser.raw_samples(spk)),
                desc=f"items in {spk}",
            )
        ]
        logger.info("Preprocessing Complete")

    def process_file(self, spk: str, fname: str) -> None:
        raw_path = path(self.__in_path, parser.get_spkdir(spk), fname)
        logger.trace(f"Processing: {raw_path}")
        raw_wav, nonoise, nopop, wav = processor.full_load_parts(raw_path, keep=False)
        if not processor.is_valid(raw_wav):
            logger.warn(f"No Content in raw file: {raw_path}")
            return
        if not processor.is_valid(nonoise):
            logger.warn(f"No Content after noisereduction: {raw_path}")
            return
        if not processor.is_valid(nopop):
            logger.warn(f"No Content after removing pop: {raw_path}")
            return
        if not processor.only_content(wav):
            logger.warn(f"No Content after filtering: {raw_path}")
            return

        lo, hi = processor.get_f0_lohi(parser.sex(spk))
        f0, sp, ap = processor.get_world_params(wav=wav)

        wav_mono = processor.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
        if not processor.only_content(wav_mono):
            logger.warn(f"Failed to get monotonic wav for: {raw_path}")
            return

        spmel, phase = processor.get_spmel(wav)
        if not processor.only_content(spmel):
            logger.warn(f"Failed to get mel spectrogram for: {raw_path}")
            return

        f0_norm = processor.extract_f0(wav=wav, lo=lo, hi=hi)
        if not processor.only_content(f0_norm):
            logger.warn(f"Failed to get f0 for: {raw_path}")
            return

        if len(spmel) != len(f0_norm):
            if (len(spmel) - 1) == len(f0_norm):
                spmel = spmel[:-1]
            else:
                msg = (
                    f"melspec and f0 lengths do not match for {raw_path}\n"
                    f"spmel: {len(spmel)}\n"
                    f"f0_norm: {len(f0_norm)}\n"
                )
                logger.fatal(msg)
                raise Exception(msg)

        wav_full_split = processor.fold_pad(
            wav, self.max_len_pad * (self.hop_length - 1)
        )
        wav_mono_split = processor.fold_pad(
            wav_mono, self.max_len_pad * (self.hop_length - 1)
        )
        spmel_split = processor.fold_pad(spmel, self.max_len_pad)
        f0_split = processor.fold_pad(f0_norm, self.max_len_pad)

        spk_dir = parser.get_spkdir(spk)
        fullwavs = newpath(self.path_fullwavs, spk_dir)
        monowavs = newpath(self.path_monowavs, spk_dir)
        spmels = newpath(self.path_spmels, spk_dir)
        freqs = newpath(self.path_freqs, spk_dir)
        phases = newpath(self.path_phases, spk_dir)
        cleanwavs = newpath(self.path_cleanwavs, spk_dir)
        namebase = strip_ext(fname)
        torchaudio.save(
            uri=path(cleanwavs, f"{namebase}.wav"),
            src=wav.unsqueeze(0),
            sample_rate=16000,
        )
        for idx, (wav_full_i, wav_mono_i, spmel_i, f0_i) in enumerate(
            zip(wav_full_split, wav_mono_split, spmel_split, f0_split)
        ):
            filename = f"{namebase}_{idx}.pt"
            if (
                processor.only_content(wav_full_i)
                and processor.only_content(wav_mono_i)
                and processor.only_content(spmel_i)
                and processor.only_content(f0_i)
            ):
                torch.save(wav_full_i.to("cpu"), path(fullwavs, filename))
                torch.save(wav_mono_i.to("cpu"), path(monowavs, filename))
                torch.save(spmel_i.to("cpu"), path(spmels, filename))
                torch.save(f0_i.to("cpu"), path(freqs, filename))
        filename = f"{strip_ext(fname)[0]}.pt"
        torch.save(phase.to("cpu"), path(phases, filename))


def preprocess_data() -> None:
    PreProcess()
