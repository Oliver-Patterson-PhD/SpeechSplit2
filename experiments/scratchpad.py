import os
from typing import List, Self, Set, Tuple

import torch
import torchaudio

from synthesizers import Synthesizer
from transcribers import Transcriber
from util import CompareItem
from util.audio import norm_audio
from util.tensor import save_tensor
from utils import zero_one_norm

from .experiment import Experiment

DataType = Tuple[
    str,  # Filename
    str,  # Speaker ID string
    torch.Tensor,  # Mel Spectrogram
    torch.Tensor,  # Rhythm Input
    torch.Tensor,  # Content Input
    torch.Tensor,  # Pitch Input
    torch.Tensor,  # Timbre Input
    torch.Tensor,  # len_crop (required in padding)
]


class Scratchpad(Experiment):
    retries: int = 5
    transcriber: Transcriber
    synth_list: List[Synthesizer]

    @torch.no_grad()
    def run(self: Self) -> None:
        self.transcriber = Transcriber(
            device=self.compute.device(),
            model_name=self.config.options.whisper_type,
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
    def process(self: Self) -> None:
        self.load_data(singleitem=True, sequential=True)
        proc_data = [item for item in self.data_loader]
        proc_data = sorted(proc_data, key=lambda i: i[0])[:5]
        proc_name: Set[str]
        proc_name = set(self.dataset.sample_name(item[0][0]) for item in proc_data)
        self.logger.trace_var(proc_name, "DEBUG")
        for name in proc_name:
            items: List[DataType] = sorted(
                [item for item in proc_data if str(item[0][0]).find(name) != -1],
                key=lambda i: i[0],
            )
            list_gt: List[torch.Tensor] = []
            list_out: List[torch.Tensor] = []
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
    def save_item(
        self: Self, name: str, orig: torch.Tensor, proc: torch.Tensor
    ) -> None:
        self.logger.debug(f"Processing: {name}")
        open(os.path.join(self.lossdir, name + ".txt"), "w").write(
            CompareItem(
                name,
                source=orig,
                destin=proc,
                model=self.transcriber,
                text=self.dataset.get_real_text(name),
            ).__str__()
        )
        raw_audio_file = os.path.join(
            self.config.paths.raw_wavs, self.dataset.speaker(name), f"{name}.wav"
        )
        raw_aud_raw, _ = torchaudio.load(raw_audio_file)
        raws, _ = self.audproc.get_spmel(
            self.audproc.filter_wav(self.audproc.clean_audio(raw_aud_raw))
        )
        save_tensor(raws, os.path.join(self.spmldir, f"{name}-mel-raws.png"))
        save_tensor(orig, os.path.join(self.spmldir, f"{name}-mel-orig.png"))
        save_tensor(proc, os.path.join(self.spmldir, f"{name}-mel-proc.png"))
        [self.process_synth(synth, name, orig, proc) for synth in self.synth_list]
        # self.full_convert(name, orig, proc)

    def get_phases(self: Self, name: str) -> torch.Tensor:
        raw_phase_file = os.path.join(
            self.config.paths.features,
            "phases",
            self.dataset.speaker(name),
            f"{name}.pt",
        )
        raw_phases = torch.load(
            raw_phase_file,
            weights_only=True,
        ).to(self.compute.device())
        return raw_phases

    def full_convert(
        self: Self, name: str, orig: torch.Tensor, proc: torch.Tensor
    ) -> None:
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
            self.config.paths.raw_wavs, self.dataset.speaker(name), f"{name}.wav"
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

    def save_audio(self: Self, wav: torch.Tensor, file: str) -> None:
        torchaudio.save(
            file,
            wav.cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )
        torchaudio.save(
            file.replace(self.wavsdir, self.normdir),
            norm_audio(wav).cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )
        return

    @torch.no_grad()
    def process_synth(
        self: Self, synt: Synthesizer, name: str, orig: torch.Tensor, proc: torch.Tensor
    ) -> None:
        self.logger.debug(f"Synthesizing: {synt.model_name}")
        synth_gt = os.path.join(self.wavsdir, f"{name}-{synt}-orig.wav")
        synth_out = os.path.join(self.wavsdir, f"{name}-{synt}-proc.wav")
        self.logger.trace_tensor(orig)
        self.logger.trace_tensor(proc)
        self.single_spmel_to_audio(synth_gt, orig, synt)
        self.single_spmel_to_audio(synth_out, proc, synt)

    @torch.no_grad()
    def single_spmel_to_audio(
        self: Self, file: str, spec: torch.Tensor, synt: Synthesizer
    ) -> None:
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
            norm_audio(wav).cpu(),
            sample_rate=self.config.audio.sample_rate,
            backend="sox",
        )

    @torch.no_grad()
    def process_item(
        self: Self,
        spmel_gt: torch.Tensor,
        rhythm_input: torch.Tensor,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        timbre_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
