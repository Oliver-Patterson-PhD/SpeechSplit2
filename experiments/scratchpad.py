import os

import torch
import torchaudio

from synthesizers import Synthesizer
from transcribers import CompareItem, Transcriber
from util.tensor import Tensor, TensorPair, save_tensor

from .experiment import Experiment

DataType = tuple[
    str,  # Filename
    str,  # Speaker ID string
    Tensor,  # Mel Spectrogram
    Tensor,  # Rhythm Input
    Tensor,  # Content Input
    Tensor,  # Pitch Input
    Tensor,  # Timbre Input
    Tensor,  # len_crop (required in padding)
]


def zero_one_norm(s: Tensor) -> Tensor:
    s_norm = s - torch.min(s)
    s_norm /= torch.max(s_norm)
    return s_norm


class Scratchpad(Experiment):
    retries: int = 5
    transcriber: Transcriber
    synth_list: list[Synthesizer]

    @torch.no_grad()
    def run(self) -> None:
        self.transcriber = Transcriber(
            device=self.compute.device(),
            config=self.config,
        )
        model_name = os.path.join(
            "speechsplit2-large",
            "trainmask-large-SpeechSplit2-2024-11-07.ckpt",
        )
        self.load_trained(model_name)
        self.logger.info("Full Process On")
        self.compute.set_gpu()
        self.synth_list = [
            getattr(
                __import__("synthesizers"),
                synthname,
            )(self.compute.device())
            for synthname in [
                # "MelGan",
                "ParallelWaveGan",
                # "HiFiGAN",
                # "Wavenet",
                # "GriffinLim",
            ]
        ]
        ddir = os.path.join(self.experiment_dir, self.config.options.dataset_name)
        self.lossdir = os.path.join(ddir, "losses")
        self.normdir = os.path.join(ddir, "norm")
        self.spmldir = os.path.join(ddir, "spmels")
        self.wavsdir = os.path.join(ddir, "wavs")
        os.makedirs(self.lossdir, exist_ok=True)
        os.makedirs(self.normdir, exist_ok=True)
        os.makedirs(self.spmldir, exist_ok=True)
        os.makedirs(self.wavsdir, exist_ok=True)
        self.process()

    @torch.no_grad()
    def process(self) -> None:
        self.load_data(singleitem=True, sequential=True)
        proc_data = [item for item in self.data_loader]
        proc_data = sorted(proc_data, key=lambda i: i[0])[:5]
        proc_name: set[str]
        proc_name = set(self.parser.sample_name(item[0][0]) for item in proc_data)
        self.logger.trace_var(proc_name, "DEBUG")
        for name in proc_name:
            items: list[DataType] = sorted(
                [item for item in proc_data if str(item[0][0]).find(name) != -1],
                key=lambda i: i[0],
            )
            list_gt: list[Tensor] = []
            list_out: list[Tensor] = []
            for (
                fname,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) in items:
                spmel_outgt, spmel_output = self.process_item(
                    spmel_gt=spmel_gt,
                    rhythm_input=rhythm_input,
                    content_input=content_input,
                    pitch_input=pitch_input,
                    timbre_input=timbre_input,
                    len_crop=len_crop,
                )
                list_gt.append(spmel_outgt)
                list_out.append(spmel_output)
            self.logger.trace_var(list_gt)
            orig_spmel = self.audproc.combine(list_gt)
            self.logger.trace_var(list_out)
            proc_spmel = self.audproc.combine(list_out)
            self.save_item(name, orig_spmel, proc_spmel)

    @torch.no_grad()
    def save_item(self, name: str, orig: Tensor, proc: Tensor) -> None:
        self.logger.debug(f"Processing: {name}")
        open(os.path.join(self.lossdir, name + ".txt"), "w").write(
            CompareItem(
                name,
                mel1=orig,
                mel2=proc,
                model=self.transcriber,
                text=self.parser.get_real_text(name),
            ).__str__()
        )
        raw_audio_file = os.path.join(
            self.config.paths.raw_wavs, self.parser.speaker(name), f"{name}.wav"
        )
        raw_aud_raw, _ = torchaudio.load(raw_audio_file)
        raws, _ = self.audproc.get_spmel(
            self.audproc.filter_wav(self.audproc.run_clean(raw_aud_raw))
        )
        save_tensor(raws, os.path.join(self.spmldir, f"{name}-mel-raws.png"))
        save_tensor(orig, os.path.join(self.spmldir, f"{name}-mel-orig.png"))
        save_tensor(proc, os.path.join(self.spmldir, f"{name}-mel-proc.png"))
        [self.process_synth(synth, name, orig, proc) for synth in self.synth_list]
        # self.full_convert(name, orig, proc)

    def get_phases(self, name: str) -> Tensor:
        raw_phase_file = os.path.join(
            self.config.paths.features,
            "phases",
            self.parser.speaker(name),
            f"{name}.pt",
        )
        raw_phases = torch.load(
            raw_phase_file,
            weights_only=True,
        ).to(self.compute.device())
        return raw_phases

    def full_convert(self, name: str, orig: Tensor, proc: Tensor) -> None:
        self.logger.trace_tensor(orig, "DEBUG")
        self.logger.trace_tensor(proc, "DEBUG")
        self.logger.debug(f"orig: {orig.device.__str__()}")
        self.logger.debug(f"proc: {proc.device.__str__()}")
        origmt = self.audproc.rev_spmel(orig.mT)
        procmt = self.audproc.rev_spmel(proc.mT)
        self.logger.debug(f"origmt: {origmt.device.__str__()}")
        self.logger.debug(f"procmt: {procmt.device.__str__()}")
        raw_mags_orig = torch.sqrt(origmt)
        raw_mags_proc = torch.sqrt(procmt)

        raw_audio_file = os.path.join(
            self.config.paths.raw_wavs, self.parser.speaker(name), f"{name}.wav"
        )
        raw_aud_raw, _ = torchaudio.load(raw_audio_file)
        raw_raw_stft = self.audproc.stft(raw_aud_raw.squeeze())
        pad_raw_stft = torch.nn.functional.pad(
            raw_raw_stft, (0, raw_mags_proc.size(-1) - raw_raw_stft.size(-1))
        )

        raw_phases = self.get_phases(name)
        raw_phases = torch.nn.functional.pad(
            raw_phases, (0, raw_mags_proc.size(-1) - raw_phases.size(-1))
        )
        self.logger.trace_tensor(raw_phases, "DEBUG")

        self.logger.trace_tensor(raw_mags_orig, "DEBUG")
        self.logger.trace_tensor(raw_mags_proc, "DEBUG")

        # fmt: off
        save_tensor(zero_one_norm(pad_raw_stft.abs()), os.path.join(self.spmldir, f"{name}-raw-stft.png"))
        save_tensor(zero_one_norm(raw_mags_orig), os.path.join(self.spmldir, f"{name}-raw-orig.png"))
        save_tensor(zero_one_norm(raw_mags_proc), os.path.join(self.spmldir, f"{name}-raw-proc.png"))
        # fmt: on

        raw_spec_orig = torch.polar(raw_mags_orig, raw_phases)
        raw_spec_proc = torch.polar(raw_mags_proc, raw_phases)

        self.logger.trace_tensor(pad_raw_stft, "DEBUG")
        self.logger.trace_tensor(raw_spec_orig, "DEBUG")
        self.logger.trace_tensor(raw_spec_proc, "DEBUG")

        raw_audio_stft = self.audproc.istft(pad_raw_stft).unsqueeze(0)
        raw_audio_orig = self.audproc.istft(raw_spec_orig).unsqueeze(0)
        raw_audio_proc = self.audproc.istft(raw_spec_proc).unsqueeze(0)

        self.logger.trace_tensor(raw_audio_stft, "DEBUG")
        self.logger.trace_tensor(raw_audio_orig, "DEBUG")
        self.logger.trace_tensor(raw_audio_proc, "DEBUG")

        self.save_audio(
            raw_audio_stft, os.path.join(self.wavsdir, f"{name}-raw-stft.wav")
        )
        self.save_audio(
            raw_audio_orig, os.path.join(self.wavsdir, f"{name}-raw-orig.wav")
        )
        self.save_audio(
            raw_audio_proc, os.path.join(self.wavsdir, f"{name}-raw-proc.wav")
        )
        return

    def save_audio(self, wav: Tensor, file: str) -> None:
        torchaudio.save(
            file,
            wav.cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )
        torchaudio.save(
            file.replace(self.wavsdir, self.normdir),
            self.audproc.norm_audio(wav).cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )
        return

    @torch.no_grad()
    def process_synth(
        self, synt: Synthesizer, name: str, orig: Tensor, proc: Tensor
    ) -> None:
        self.logger.debug(f"Synthesizing: {synt}")
        synth_gt = os.path.join(self.wavsdir, f"{name}-{synt}-orig.wav")
        synth_out = os.path.join(self.wavsdir, f"{name}-{synt}-proc.wav")
        os.makedirs(os.path.dirname(synth_gt), exist_ok=True)
        os.makedirs(os.path.dirname(synth_out), exist_ok=True)
        self.logger.trace_tensor(orig)
        self.logger.trace_tensor(proc)
        self.single_spmel_to_audio(synth_gt, orig, synt)
        self.single_spmel_to_audio(synth_out, proc, synt)

    @torch.no_grad()
    def single_spmel_to_audio(self, file: str, spec: Tensor, synt: Synthesizer) -> None:
        wav = synt.spect2wav(spec).unsqueeze(dim=0)
        self.logger.trace_tensor(wav)
        torchaudio.save(
            file,
            wav.cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )
        torchaudio.save(
            file.replace(self.wavsdir, self.normdir),
            self.audproc.norm_audio(wav).cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )

    @torch.no_grad()
    def process_item(
        self,
        spmel_gt: Tensor,
        rhythm_input: Tensor,
        content_input: Tensor,
        pitch_input: Tensor,
        timbre_input: Tensor,
        len_crop: Tensor,
    ) -> TensorPair:
        spmel_gt = spmel_gt.to(self.compute.device())
        rhythm_input = rhythm_input.to(self.compute.device())
        content_input = content_input.to(self.compute.device())
        pitch_input = pitch_input.to(self.compute.device()).unsqueeze(-1)
        timbre_input = timbre_input.to(self.compute.device())
        len_crop = len_crop.to(self.compute.device())
        content_pitch_input = self.prepare_input(
            content_input,
            pitch_input,
            len_crop,
        )
        if self.config.options.return_latents:
            (
                spmel_output,
                code_exp_1,
                code_exp_2,
                code_exp_3,
                code_exp_4,
            ) = self.model(
                content_pitch_input,
                rhythm_input,
                timbre_input,
            )
        else:
            spmel_output = self.model(
                content_pitch_input,
                rhythm_input,
                timbre_input,
            )
        return (spmel_gt.squeeze(), spmel_output.squeeze())
