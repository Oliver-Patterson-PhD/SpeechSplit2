from itertools import product

from ..transcribers import CompareItem, Transcriber
from ..util import compute, config, logger
from ..util.file import newpath, path
from ..util.plot import plot_things
from ..util.tensor import Tensor, pad_to
from .experiment import Experiment


class TranscriptionLoss(Experiment):
    transcriber: Transcriber

    def load_audio(self, speaker: str, uttr: str) -> Tensor:
        subpath = self.parser.get_wavfile(speaker, uttr)
        fullpath = path(config.paths.raw_wavs, subpath)
        raw, nonoise, nopop, wav = self.audproc.full_load_parts(fullpath, True)
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        return raw

    def run(self) -> None:
        self.transcriber = Transcriber(device=compute.device())
        logger.debug("Running Transcription Loss")
        compute.set_gpu()
        speakers = self.parser.speakers()
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

    def run_loss(self, spk1: str, spk2: str, uttr: str) -> None:
        wav1 = self.load_audio(spk1, uttr)
        wav2 = self.load_audio(spk2, uttr)
        wav1, wav2 = pad_to(wav1, wav2)
        mel1, _ = self.audproc.get_spmel(wav1)
        mel2, _ = self.audproc.get_spmel(wav2)

        plot_items: list[tuple[Tensor, str | tuple[str, ...]]] = [
            (wav1.squeeze(), f"waveform {spk1}"),
            (wav2.squeeze(), f"waveform {spk2}"),
        ]
        try:
            realtext1 = self.parser.get_utterance(self.parser.get_fullpath(spk1, uttr))
            realtext2 = self.parser.get_utterance(self.parser.get_fullpath(spk2, uttr))
            uttrs = [realtext1, realtext2] if self.parser.is_timit() else None
            plot_things(
                plot_out=newpath(
                    self.experiment_dir,
                    str(self.parser.dataset_type()),
                    f"{spk1}-{spk2}",
                ),
                sample=f"{uttr}-{spk1}-{spk2}",
                things=plot_items,
                utterances=uttrs,
                sample_time=(wav1.size(dim=-1) // config.audio.sample_rate),
                ftype="png",
            )
        except RuntimeError as e:
            logger.warn(f"Failed to plot: {uttr}, {spk1}-{spk2}")
            raise e
        message = CompareItem(
            f"{uttr}-{spk1}-{spk2}",
            mel1=self.audproc.get_spmel(wav1)[0].mT,
            mel2=self.audproc.get_spmel(wav2)[0].mT,
            model=self.transcriber,
            text=realtext1.word,
        )
        logger.info("\n" + message.__str__())
