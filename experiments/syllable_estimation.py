import torch

from util.file import basename, newpath, path
from util.math import find_peaks
from util.plot import plot_things
from util.tensor import Tensor, pad_to

from .experiment import Experiment


class SyllableEstimation(Experiment):
    fold_params: dict
    kernel_overlap: float = 0.2
    kernel_time: float = 0.02
    dims_log: str = "TRACE"

    def run(self) -> None:
        self.logger.debug("Running Syllable Estimation")
        self.kernel_size = int(self.config.audio.sample_rate * self.kernel_time)
        self.kernel_hop = int(self.kernel_size * self.kernel_overlap)
        self.fold_params = dict(
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            stride=self.kernel_hop,
        )
        speakers = self.parser.phonetic_labelled_speakers()
        self.logger.info(f"Found {len(speakers)} speakers")
        runs = [
            (speaker, fname)
            for speaker in sorted(speakers)
            for fname in sorted(self.parser.raw_samples(speaker))
        ]
        [
            self.run_estimation(speaker, fname)  # type: ignore[func-returns-value]
            for speaker, fname in self.logger.progress_bar(runs, unit=" files")
        ]

    def run_estimation(self, speaker: str, fname: str) -> None:
        fullpath = self.parser.get_fullpath(speaker, basename(fname))
        wav = self.load_audio(speaker, fname)
        spmel = self.spectrum(wav)
        intensity = self.intensity(wav)
        pitch_contour = self.pitch_contour(wav, speaker)
        minval = pitch_contour.min()
        intensity, pitch_contour = pad_to(intensity, pitch_contour)
        is_voiced = self.is_voiced(pitch_contour)

        intensity_peaks = self.peaks(intensity)
        voiced_peaks = is_voiced * intensity_peaks
        unvoiced_peaks = (is_voiced != 1) * intensity_peaks
        peaks: Tensor = torch.stack((voiced_peaks.squeeze(), unvoiced_peaks.squeeze()))

        pitch_contour[pitch_contour == minval] = float("nan")
        contours: Tensor = torch.stack((intensity.squeeze(), pitch_contour.squeeze()))
        plot_items: list[tuple[Tensor, str | tuple[str, ...]]] = [
            (spmel.squeeze().mT, "Mel Spectrogram"),
            (contours, ("Intensity", "Pitch Contour")),
            (peaks, ("Voiced Peaks", "Unvoiced Peaks")),
        ]
        try:
            plot_things(
                plot_out=newpath(
                    self.experiment_dir,
                    str(self.parser.dataset_type()),
                    self.parser.get_spkdir(speaker),
                ),
                sample=basename(fullpath),
                things=plot_items,
                word=self.parser.get_utterance(fullpath),
                sample_time=(wav.size(dim=-1) / self.config.audio.sample_rate),
                ftype="png",
            )
        except RuntimeError:
            self.logger.warn(f"Failed to plot: {speaker}, {fname}")
            return

    def load_audio(self, speaker: str, fname: str) -> Tensor:
        subpath = self.parser.get_wavfile(speaker, basename(fname))
        fullpath = path(self.config.paths.raw_wavs, subpath)
        raw, nonoise, nopop, wav = self.audproc.full_load_parts(fullpath, True)
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        self.logger.trace_tensor(wav, self.dims_log)
        return raw if self.parser.is_timit() else wav

    def intensity(self, audio: Tensor) -> Tensor:
        intensity = torch.nn.functional.avg_pool1d(audio.abs(), **self.fold_params)
        norm_intensity = intensity / intensity.max()
        self.logger.trace_tensor(norm_intensity, self.dims_log)
        return norm_intensity

    def spectrum(self, wav: Tensor) -> Tensor:
        spmel, _ = self.audproc.get_spmel(wav)
        self.logger.trace_tensor(spmel, self.dims_log)
        return spmel

    def peaks(self, intensity: Tensor) -> Tensor:
        intensity_peaks = find_peaks(intensity, 8) * (intensity > intensity.median())
        self.logger.trace_tensor(intensity_peaks, self.dims_log)
        return intensity_peaks

    def pitch_contour(self, wav: Tensor, speaker: str) -> Tensor:
        lo, hi = self.audproc.get_f0_lohi(self.parser.sex(speaker))
        self.logger.trace_var(lo, self.dims_log)
        self.logger.trace_var(hi, self.dims_log)
        pitch_contour = self.audproc.extract_f0(
            wav.squeeze(), lo, hi, hop_len=self.kernel_hop, otype=1
        )
        self.logger.trace_tensor(pitch_contour, self.dims_log)
        return pitch_contour

    def is_voiced(self, pitch_contour: torch.Tensor) -> Tensor:
        norm_contour = pitch_contour - pitch_contour.min()
        return norm_contour > norm_contour.median()
