import os
from typing import Self, Any
from traceback import format_exception
import torch

from util import Config, Logger

from data.dataset import DatasetParser
from data.utils import AudioProcs


class Immediate:
    config: Config
    exit_after: bool = True

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        return

    def test(self: Self) -> None:
        self.logger = Logger()
        self.in_path = "/mnt/datasets/raw/SmolSpeech/original"
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

    def process_file(self: Self, spk_dir: str, fname: str) -> None:
        rawwav = torch.tensor([])
        clnwav = torch.tensor([])
        wav = torch.tensor([])
        energy = torch.tensor([])
        wav_mono = torch.tensor([])
        spmel = torch.tensor([])
        f0_norm = torch.tensor([])
        try:
            rawwav = self.proc.getraw(os.path.join(self.in_path, spk_dir, fname))
            clnwav = self.proc.clean_audio(rawwav)
            wav = self.proc.filter_wav(clnwav)
            frame_length = 400  # 20ms frame
            hop_length = 100  # 10ms hop
            energy = short_time_energy(wav.squeeze(), frame_length, hop_length)
            lo, hi = self.proc.get_f0_lohi(spk_dir)
            f0, sp, ap = self.proc.get_world_params(wav=wav)
            wav_mono = self.proc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
            spmel, phase = self.proc.get_spmel(wav)
            self.logger.trace_tensor(wav)
            f0_norm = self.proc.extract_f0(wav=wav, lo=lo, hi=hi)
        except Exception as e:
            for tb in format_exception(e):
                self.logger.error(tb.rstrip())
        self.logger.info(
            format_log_message(
                fname,
                len(energy),
                [
                    energy.std().item(),
                    energy.max().item(),
                    energy.min().item(),
                ],
                [
                    self.proc.has_content(rawwav),
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
) -> torch.Tensor:
    energy: list[float] = []
    for i in range(0, len(audio_data) - frame_length, hop_length):
        frame = audio_data[i : i + frame_length]
        energy.append(torch.sum(frame**2).item())
    return torch.tensor(energy)


def format_log_message(
    fname: str,
    siz: int,
    data_list: list[float],
    other: list[Any] = [],
) -> str:
    precision = 5
    width = precision + 4
    format_string = "{:<25} energy len: {:>4} : " + " ".join(
        ["{{:>{},.{}f}}".format(width, precision)] * len(data_list)
        + ["{{!r:>{}}}".format(width)] * len(other)
    )
    log_message = format_string.format(fname, siz, *data_list, *other)
    return log_message
