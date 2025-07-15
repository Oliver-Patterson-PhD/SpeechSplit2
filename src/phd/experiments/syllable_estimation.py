from itertools import product

import matplotlib
import torch

from ..util.file import basename, newpath, path
from ..util.math import find_peaks
from ..util.plot import plot_things
from ..util.tensor import Tensor, pad_to
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

        speaker_dict: dict[str, set[str]] = {
            speaker: set(uttr for uttr in self.parser.get_utterances(speaker))
            for speaker in speakers
        }
        self.lossdir = newpath(self.experiment_dir, str(self.parser.dataset_type()))
        [
            self.run_loss(spk1, spk2, uttr)  # type: ignore[func-returns-value]
            for spk1, spk2 in product(sorted(speakers), sorted(speakers))
            for uttr in sorted(speaker_dict[spk1] & speaker_dict[spk2])
            if spk1 != spk2 and uttr not in {"SA1", "SA2"}
        ]
        self.logger.info(f"Found {len(speakers)} speakers")

        # runs = [
        #     (speaker, fname)
        #     for speaker in sorted(speakers)
        #     for fname in sorted(self.parser.raw_samples(speaker))
        # ]
        # [
        #     self.run_estimation(speaker, fname)  # type: ignore[func-returns-value]
        #     for speaker, fname in self.logger.progress_bar(runs, unit=" files")
        # ]

    def get_feats(self, wav: Tensor, speaker: str) -> tuple[Tensor, tuple[str, ...]]:
        melspec, _ = self.audproc.get_spmel(wav)
        intensity = self.intensity(wav)
        pitch_contour = self.pitch_contour(wav, speaker)
        minpitch = pitch_contour.min()
        intensity, pitch_contour = pad_to(intensity, pitch_contour)
        voicedness = self.is_voiced(pitch_contour)
        intensity_peaks = self.peaks(intensity)
        voiced_peaks = intensity_peaks * (voicedness == 1)
        unvoiced_peaks = intensity_peaks * (voicedness != 1)
        pitch_contour[pitch_contour == minpitch] = float("nan")
        plot = torch.stack(
            (
                intensity.squeeze(),
                pitch_contour.squeeze(),
                voiced_peaks.squeeze(),
                unvoiced_peaks.squeeze(),
            )
        )
        label = ("Intensity", "Pitch Contour", "Voiced Peaks", "Unvoiced Peaks")
        return plot, label

    def plot_features(
        self,
        ax: matplotlib.axes.Axes,
        item: Tensor,
        name: tuple[str, ...],
        sample_time: float,
        sample_div: float,
    ) -> matplotlib.axes.Axes:
        for i_item, i_name in zip(item, name):
            ax.plot(
                [i / sample_div for i in range(i_item.size(-1))],
                i_item.cpu().numpy(),
                label=i_name,
            )
        ax.set_xlim(0, sample_time)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Magnitude (A.U.)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.95, 0.1))
        return ax

    def plot_wavform(
        self,
        ax: matplotlib.axes.Axes,
        item: Tensor,
        name: str,
        sample_time: float,
        sample_div: float,
    ) -> matplotlib.axes.Axes:
        item[item == 0.0] = float("nan")
        ax.plot(
            [i / sample_div for i in range(item.size(-1))],
            item.cpu().numpy(),
        )
        ax.set_xlim(0, sample_time)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Magnitude (A.U.)")
        return ax

    def run_loss(self, spk1: str, spk2: str, uttr: str) -> None:
        assert self.parser.is_timit()
        try:
            self.logger.debug(f"Running loss with {uttr} between {spk1} and {spk2}")
            wav1, wav2 = pad_to(self.load_audio(spk1, uttr), self.load_audio(spk2, uttr))
            plot_items: list[tuple[Tensor, str | tuple[str, ...]]] = [
                (wav1.squeeze(), "waveform"),
                self.get_feats(wav1, spk1),
                (wav2.squeeze(), "waveform"),
                self.get_feats(wav2, spk2),
            ]
            realtext1 = self.parser.get_utterance(self.parser.get_fullpath(spk1, uttr))
            realtext2 = self.parser.get_utterance(self.parser.get_fullpath(spk2, uttr))
            uttrs = [realtext1, realtext1, realtext2, realtext2]
            assert len(uttrs) == len(plot_items)
            fig = matplotlib.pyplot.figure()
            fig.set_size_inches(19, 15)
            fig.set_dpi(300)
            label_colour = "k"
            sample = f"{uttr}-{spk1}-{spk2}"
            sample_time = wav1.size(dim=-1) / self.config.audio.sample_rate
            fig.suptitle(f"Sample: {sample} ({uttrs[0].word})")
            subplots = fig.subplots(len(plot_items), 1)
            for i, ((item, name), word, ax) in enumerate(zip(plot_items, uttrs, subplots)):
                sample_div = 1 if sample_time is None else item.size(-1) / sample_time
                if isinstance(name, tuple):
                    ax = self.plot_features(
                        ax=ax,
                        item=item,
                        name=name,
                        sample_time=sample_time,
                        sample_div=sample_div,
                    )
                elif isinstance(name, str):
                    if item.dim() == 1:
                        ax = self.plot_wavform(
                            ax=ax,
                            item=item,
                            name=name,
                            sample_time=sample_time,
                            sample_div=sample_div,
                        )
                for phon in word.phones:
                    start_time = phon.start / self.config.audio.sample_rate
                    half_time = (
                        phon.start + ((phon.end - phon.start) / 2)
                    ) / self.config.audio.sample_rate
                    ax.annotate(
                        text=phon.phon,
                        xy=(half_time, ax.get_ylim()[1]),
                        xytext=(0, 3),
                        textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="baseline",
                        color=label_colour,
                    )
                    ax.axvline(start_time, color=label_colour, alpha=0.1)
            subplots[-1].set_xlabel("Time (Seconds)")
            subplots[0].set_title(spk1, loc="right")
            subplots[2].set_title(spk2, loc="right")
            fig.savefig(
                path(
                    newpath(
                        self.experiment_dir,
                        str(self.parser.dataset_type()),
                        f"{spk1}-{spk2}",
                    ),
                    f"{sample}.png",
                )
            )
            fig.tight_layout()
            matplotlib.pyplot.close(fig=fig)
        except RuntimeError as e:
            self.logger.warn(f"Failed to plot: {uttr}, {spk1}-{spk2}")
            raise e

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
                utterances=self.parser.get_utterance(fullpath),
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
