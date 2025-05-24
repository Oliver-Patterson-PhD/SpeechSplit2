from typing import Self

import torch

from util.file import basename, newpath, path
from util.math import find_peaks
from util.plot import plot_things

from .experiment import Experiment


class SyllableEstimation(Experiment):
    fold_params: dict
    kernel_overlap: float = 0.2
    kernel_time: float = 0.02
    dims_log: str = "TRACE"

    def run(self: Self) -> None:
        self.logger.debug("Running Syllable Estimation")
        self.kernel_size = int(self.config.audio.sample_rate * self.kernel_time)
        self.kernel_hop = int(self.kernel_size * self.kernel_overlap)
        self.fold_params = dict(
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            stride=self.kernel_hop,
        )
        speakers = self.dataset.speakers()
        speakers = self.dataset.phonetic_labelled_speakers()
        self.logger.info(f"Found {len(speakers)} speakers")
        [
            self.run_estimation(speaker, fname)
            for speaker in self.logger.progress_bar(speakers, desc="speakers")
            for fname in sorted(self.dataset.raw_samples(speaker))
        ]
        return

    def run_estimation(self: Self, speaker: str, fname: str) -> None:
        fullpath = self.dataset.get_fullpath(speaker, basename(fname))
        wav = self.load_audio(speaker, fname)
        spmel = self.spectrum(wav)
        intensity = self.intensity(wav)
        intensity_peaks = self.peaks(intensity)
        pitch_contour = self.pitch_contour(wav, speaker)
        plot_items = [
            (wav.squeeze(), "Waveform"),
            (spmel.squeeze().mT, "Mel Spectrogram"),
            (intensity.squeeze(), "Intensity"),
            (intensity_peaks.squeeze(), "Intensity Peaks"),
            (pitch_contour.squeeze(), "Pitch Contour"),
        ]
        try:
            plot_things(
                plot_out=newpath(
                    self.experiment_dir,
                    str(self.dataset.dataset_type()),
                    self.dataset.get_spkdir(speaker),
                ),
                sample=basename(fullpath),
                things=plot_items,
                word=self.dataset.get_utterance(fullpath),
                sample_time=(wav.size(dim=-1) / self.config.audio.sample_rate),
            )
        except RuntimeError:
            return

    def intensity(self: Self, audio: torch.Tensor) -> torch.Tensor:
        intensity = torch.nn.functional.avg_pool1d(audio.abs(), **self.fold_params)
        norm_intensity = intensity / intensity.max()
        self.logger.trace_tensor(norm_intensity, self.dims_log)
        return norm_intensity

    def load_audio(self: Self, speaker: str, fname: str) -> torch.Tensor:
        subpath = self.dataset.get_wavfile(speaker, basename(fname))
        fullpath = path(self.config.paths.raw_wavs, subpath)
        wav = self.audproc.full_load_parts(fullpath, True)[-1]
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        self.logger.trace_tensor(wav, self.dims_log)
        return wav

    def spectrum(self: Self, wav: torch.Tensor) -> torch.Tensor:
        spmel, _ = self.audproc.get_spmel(wav)
        self.logger.trace_tensor(spmel, self.dims_log)
        return spmel

    def peaks(self: Self, intensity: torch.Tensor) -> torch.Tensor:
        intensity_peaks = find_peaks(intensity, 8) * (intensity > intensity.median())
        self.logger.trace_tensor(intensity_peaks, self.dims_log)
        return intensity_peaks

    def pitch_contour(self: Self, wav: torch.Tensor, speaker: str) -> torch.Tensor:
        lo, hi = self.audproc.get_f0_lohi(self.dataset.sex(speaker))
        self.logger.trace_var(lo, self.dims_log)
        self.logger.trace_var(hi, self.dims_log)
        pitch_contour = self.audproc.extract_f0(
            wav.squeeze(), lo, hi, hop_len=self.kernel_hop, otype=1
        )
        self.logger.trace_tensor(pitch_contour, self.dims_log)
        return pitch_contour
