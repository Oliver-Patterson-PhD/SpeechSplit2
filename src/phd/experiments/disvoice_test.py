from ..util.file import basename, path
from ..util.tensor import Tensor
from .disvoice_requirements import Articulation, Glottal, Phonation
from .experiment import Experiment


class DisVoiceTest(Experiment):
    dims_log: str = "TRACE"

    def run(self) -> None:
        logger.debug("Running Syllable Estimation")
        self.in_path = config.paths.raw_wavs
        self.fs = config.audio.sample_rate
        speakers = self.parser.speakers()

        speaker_dict: dict[str, set[str]] = {
            speaker: set(uttr for uttr in self.parser.get_utterances(speaker))
            for speaker in speakers
        }
        [
            self.run_disvoice(speaker, uttr)  # type: ignore[func-returns-value]
            for speaker in sorted(speakers)
            for uttr in sorted(speaker_dict[speaker])
        ]
        logger.info(f"Found {len(speakers)} speakers")

    def run_disvoice(self, speaker: str, utterance_id: str) -> None:
        logger.debug(f"Running disvoice for {speaker}-{utterance_id}")
        wav = self.load_audio(speaker, utterance_id)
        lo, hi = self.audproc.get_f0_lohi(self.parser.sex(speaker))
        f0, sp, ap = self.audproc.get_world_params(wav=wav)
        _ = self.audproc.get_monotonic_wav(wav=wav, f0=f0, sp=sp, ap=ap)
        melspec, _ = self.audproc.get_spmel(wav)
        f0_stuff = self.audproc.extract_f0(wav=wav.squeeze(), lo=lo, hi=hi)

        try:
            do_static = False
            uttr = self.parser.get_utterance(self.parser.get_fullpath(speaker, utterance_id))
            full_path = path(self.in_path, self.parser.get_wavfile(speaker, utterance_id))
            logger.trace_tensor(wav, self.dims_log)
            logger.trace_var(uttr, self.dims_log)
            articulation = Articulation(full_path, do_static)
            glottal = Glottal(full_path, do_static)
            phonation = Phonation(full_path, do_static)

            articulation_lens = [len(item) for item in articulation]
            assert all(x == articulation_lens[0] for x in articulation_lens)
            len_articulation = articulation_lens[0]
            glottal_lens = [len(item) for item in glottal]
            assert all(x == glottal_lens[0] for x in glottal_lens)
            len_glottal = glottal_lens[0]
            phonation_lens = [len(item) for item in phonation]
            assert all(x == phonation_lens[0] for x in phonation_lens)
            len_phonation = phonation_lens[0]

            wavlen = wav.size(dim=-1)
            mellen = melspec.size(dim=-1)
            f0len = f0_stuff.size(dim=-1)
            time = wavlen / self.fs

            logger.info(f"wav:             {wavlen}")
            logger.info(f"mel:             {mellen}")
            logger.info(f"f0:              {f0len}")
            logger.info(f"articulation:    {len_articulation}")
            logger.info(f"glottal:         {len_glottal}")
            logger.info(f"phonation:       {len_phonation}")
            logger.info(f"wav_t:           {time: 8.4f}")
            logger.info(f"mel_t:           {time / mellen: 8.4f}")
            logger.info(f"f0_t:            {time / f0len: 8.4f}")
            logger.info(f"articulation_t:  {time / len_articulation: 8.4f}")
            logger.info(f"glottal_t:       {time / len_glottal: 8.4f}")
            logger.info(f"phonation_t:     {time / len_phonation: 8.4f}")
        except RuntimeError as e:
            logger.warn(f"Failed on: {speaker}-{utterance_id}")
            raise e
        exit(0)

    def load_audio(self, speaker: str, fname: str) -> Tensor:
        subpath = self.parser.get_wavfile(speaker, basename(fname))
        fullpath = path(config.paths.raw_wavs, subpath)
        raw, _, _, wav = self.audproc.full_load_parts(fullpath, True)
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        return raw if self.parser.is_timit() else wav
