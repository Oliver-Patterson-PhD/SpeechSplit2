from util.tensor import Tensor
from util.file import path, basename
from .experiment import Experiment
from disvoice.glottal import Glottal


class DisVoiceTest(Experiment):
    dims_log: str = "TRACE"

    def run(self) -> None:
        self.logger.debug("Running Syllable Estimation")
        self.in_path = self.config.paths.raw_wavs
        speakers = self.parser.phonetic_labelled_speakers()

        speaker_dict: dict[str, set[str]] = {
            speaker: set(uttr for uttr in self.parser.get_utterances(speaker))
            for speaker in speakers
        }
        [
            self.run_disvoice(speaker, uttr)  # type: ignore[func-returns-value]
            for speaker in sorted(speakers)
            for uttr in sorted(speaker_dict[speaker])
        ]
        self.logger.info(f"Found {len(speakers)} speakers")

    def run_disvoice(self, speaker: str, utterance_id: str) -> None:
        self.logger.debug(f"Running disvoice for {speaker}-{utterance_id}")
        full_path = path(self.in_path, self.parser.get_wavfile(speaker, utterance_id))
        rawwav, _, _, procwav = self.audproc.full_load_parts(full_path)[-1]
        if self.parser.is_timit():
            wav_prc = rawwav.unsqueeze(0) if rawwav.dim() == 1 else rawwav
        else:
            wav_prc = procwav.unsqueeze(0) if procwav.dim() == 1 else procwav
        lo, hi = self.audproc.get_f0_lohi(self.parser.sex(speaker))
        f0, sp, ap = self.audproc.get_world_params(wav=wav_prc)
        _ = self.audproc.get_monotonic_wav(wav=wav_prc, f0=f0, sp=sp, ap=ap)
        _, _ = self.audproc.get_spmel(wav_prc)
        _ = self.audproc.extract_f0(wav=wav_prc, lo=lo, hi=hi)
        wav = self.load_audio(speaker, utterance_id)

        try:
            uttr = self.parser.get_utterance(
                self.parser.get_fullpath(speaker, utterance_id)
            )
            self.logger.trace_tensor(wav, self.dims_log)
            self.logger.trace_var(uttr, self.dims_log)
            glottal = Glottal()
        except RuntimeError as e:
            self.logger.warn(f"Failed on: {speaker}-{utterance_id}")
            raise e

    def load_audio(self, speaker: str, fname: str) -> Tensor:
        subpath = self.parser.get_wavfile(speaker, basename(fname))
        fullpath = path(self.config.paths.raw_wavs, subpath)
        raw, _, _, wav = self.audproc.full_load_parts(fullpath, True)
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        return raw if self.parser.is_timit() else wav
