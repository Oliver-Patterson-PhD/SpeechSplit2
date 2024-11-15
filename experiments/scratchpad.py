import os
from typing import List, Self, Set, Tuple

import torch
import torchaudio

from synthesizers import MelGan, ParallelWaveGan, Synthesizer
from transcribers import Transcriber
from util import CompareItem, norm_audio
from utils import save_tensor

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
    def run(
        self: Self,
    ) -> None:
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
            MelGan(self.compute.device(), config=self.config),
            ParallelWaveGan(self.compute.device(), config=self.config),
            # Wavenet(self.compute.device(), config=self.config),
            # GriffinLim(self.compute.device(), config=self.config),
        ]
        ddir = os.path.join(self.experiment_dir, self.config.options.dataset_name)
        self.lossdir = os.path.join(ddir, "losses")
        self.normdir = os.path.join(ddir, "norm")
        self.spmelsdir = os.path.join(ddir, "spmels")
        self.wavsdir = os.path.join(ddir, "wavs")
        os.makedirs(self.lossdir, exist_ok=True)
        os.makedirs(self.normdir, exist_ok=True)
        os.makedirs(self.spmelsdir, exist_ok=True)
        os.makedirs(self.wavsdir, exist_ok=True)
        self.process()

    @torch.no_grad()
    def process(
        self: Self,
    ) -> None:
        self.load_data(singleitem=True, sequential=True)
        proc_data = [item for item in self.data_loader]
        proc_data = sorted(proc_data, key=lambda i: i[0])[:5]
        proc_name: Set[str]
        proc_name = set(self.dataset.sample_name(item[0][0]) for item in proc_data)
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
            self.logger.trace_var(list_out)
            orig_spmel = self.combine(list_gt)
            proc_spmel = self.combine(list_out)
            self.save_item(name, orig_spmel, proc_spmel)

    @torch.no_grad()
    def save_item(
        self: Self,
        name: str,
        orig: torch.Tensor,
        proc: torch.Tensor,
    ) -> None:
        self.logger.trace(f"Processing: {name}")
        open(os.path.join(self.lossdir, name + ".txt"), "w").write(
            CompareItem(
                name,
                source=orig,
                destin=proc,
                model=self.transcriber,
                text=self.dataset.get_real_text(name),
            ).__str__()
        )
        [self.process_synth(synth, name, orig, proc) for synth in self.synth_list]

    @torch.no_grad()
    def process_synth(
        self: Self,
        synt: Synthesizer,
        name: str,
        orig: torch.Tensor,
        proc: torch.Tensor,
    ) -> None:
        file_gt = f"{self.spmelsdir}/{name}-gt.png"
        file_out = f"{self.spmelsdir}/{name}-out.png"
        synth_gt = os.path.join(self.wavsdir, f"{name}-{synt}-gt.wav")
        synth_out = os.path.join(self.wavsdir, f"{name}-{synt}-out.wav")
        save_tensor(orig, file_gt)
        save_tensor(proc, file_out)
        self.single_spmel_to_audio(synth_gt, orig, synt)
        self.single_spmel_to_audio(synth_out, proc, synt)

    @torch.no_grad()
    def single_spmel_to_audio(
        self: Self,
        file: str,
        spec: torch.Tensor,
        synt: Synthesizer,
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
