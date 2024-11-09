import os
from math import floor
from typing import List, Optional, Self, Tuple

import torch
import torchaudio

from data_loader import CollaterItemType
from synthesizers import Synthesizer
from transcribers import Transcriber, WhisperTranscriber
from util import CompareItem
from utils import norm_audio, save_tensor

from .experiment import Experiment


class Scratchpad(Experiment):
    retries: int = 5
    transcriber: Transcriber
    synth_list: List[Synthesizer]

    @torch.no_grad()
    def run(
        self: Self,
    ) -> None:
        self.transcriber = WhisperTranscriber(
            device=self.compute.device(),
            model_name=self.config.options.whisper_type,
            config=self.config,
        )
        ckpt_name = "{2}-{1}/{0}/{0}-{1}-{2}-{3}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            self.config.options.resume_iters,
        )
        model_name = os.path.join(self.config.paths.models, ckpt_name)
        self.restore_model(model_name=model_name)
        self.logger.info("Full Process On")
        self.compute.set_gpu()
        from synthesizers import MelGan, ParallelWaveGan

        self.synth_list = [
            MelGan(self.compute.device(), config=self.config),
            ParallelWaveGan(self.compute.device(), config=self.config),
            # Wavenet(self.compute.device(), config=self.config),
            # GriffinLim(self.compute.device(), config=self.config),
        ]
        dname = self.config.options.dataset_name
        self.lossdir = os.path.join(self.experiment_dir, dname, "losses")
        self.wavsdir = os.path.join(self.experiment_dir, dname, "wavs")
        self.spmelsdir = os.path.join(self.experiment_dir, dname, "spmels")
        os.makedirs(self.lossdir, exist_ok=True)
        os.makedirs(self.wavsdir, exist_ok=True)
        os.makedirs(self.spmelsdir, exist_ok=True)
        self.fold_div = 2
        self.fold_size = self.config.model.max_len_pad
        self.fold_step = self.config.model.max_len_pad // self.fold_div
        self.fold_pad_len = floor((1 - (1 / self.fold_div)) * self.fold_size)
        self.process()

    @torch.no_grad()
    def process(
        self: Self,
        max_items: Optional[int] = 5,
    ) -> None:
        self.load_data(singleitem=True, full_process=True)
        if max_items is not None:
            proc_data = [
                data_item
                for number, data_item in enumerate(self.data_loader)
                if number <= max_items
            ]
        else:
            max_items = self.data_loader
        [
            (
                save_tensor(
                    item[2],
                    f"{self.spmelsdir}/{item[0][0].split("/")[-1].split(".")[0]}-orig.png",
                ),  # type: ignore [func-returns-value]
                self.save_item(
                    item[0][0].split("/")[-1].split(".")[0],
                    *self.unstack(
                        [
                            self.process_item(
                                spmel_gt,
                                rhythm_input,
                                content_input,
                                pitch_input,
                                timbre_input,
                                len_crop,
                            )
                            for (
                                _,
                                _,
                                spmel_gt,
                                rhythm_input,
                                content_input,
                                pitch_input,
                                timbre_input,
                                len_crop,
                            ) in zip(*self.make_stack(item))
                        ]
                    ),
                ),  # type: ignore [func-returns-value]
            )
            for item in self.logger.progress_bar(proc_data)
        ]

    def save_item(
        self: Self,
        fname: str,
        out_gt: torch.Tensor,
        out_out: torch.Tensor,
    ) -> None:
        self.logger.trace(f"Processing: {fname}")
        open(os.path.join(self.lossdir, fname + ".txt"), "w").write(
            str(
                CompareItem(
                    fname,
                    source=out_gt,
                    destin=out_out,
                    model=self.transcriber,
                )
            )
        )
        [
            (
                save_tensor(out_gt, f"{self.spmelsdir}/{fname}-gt.png"),  # type: ignore [func-returns-value]
                save_tensor(out_out, f"{self.spmelsdir}/{fname}-out.png"),  # type: ignore [func-returns-value]
                self.single_spmel_to_audio(
                    os.path.join(self.wavsdir, f"{fname}-{synth}-gt.wav"),
                    out_gt,
                    synth,
                ),
                self.single_spmel_to_audio(
                    os.path.join(self.wavsdir, f"{fname}-{synth}-out.wav"),
                    out_out,
                    synth,
                ),
            )
            for synth in self.synth_list
        ]
        return None

    @torch.no_grad()
    def single_spmel_to_audio(
        self: Self,
        file: str,
        spmel: torch.Tensor,
        synthesizer: Synthesizer,
    ) -> None:
        wav = synthesizer.spect2wav(spmel).unsqueeze(dim=0)
        self.logger.trace_tensor(wav)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        torchaudio.save(file, wav.cpu(), sample_rate=16000, backend="sox")
        torchaudio.save(
            file.replace(".wav", "-norm.wav"),
            norm_audio(wav).cpu(),
            sample_rate=16000,
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
        spmel_gt = spmel_gt.to(self.compute.device()).unsqueeze(0)
        rhythm_input = rhythm_input.to(self.compute.device()).unsqueeze(0)
        content_input = content_input.to(self.compute.device()).unsqueeze(0)
        pitch_input = pitch_input.to(self.compute.device()).unsqueeze(0).unsqueeze(-1)
        timbre_input = timbre_input.to(self.compute.device()).unsqueeze(0)
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

    @torch.no_grad()
    def make_stack(
        self: Self,
        item: CollaterItemType,
    ) -> CollaterItemType:
        fnamelist: List[str]
        spk_id_org: List[str]
        spmel_gt: torch.Tensor  # batch, max_len_pad, n_mels
        rhythm_input: torch.Tensor  # batch, max_len_pad, n_mels
        content_input: torch.Tensor  # batch, max_len_pad, n_mels
        pitch_input: torch.Tensor  # batch, max_len_pad, 1
        timbre_input: torch.Tensor  # batch, n_mels + 2
        len_crop: torch.Tensor  # batch
        (
            fnamelist,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = item
        assert len(fnamelist) == 1
        self.logger.trace_var(fnamelist)
        self.logger.trace_var(spk_id_org)
        self.logger.trace_tensor(spmel_gt)
        self.logger.trace_tensor(rhythm_input)
        self.logger.trace_tensor(content_input)
        self.logger.trace_tensor(pitch_input)
        self.logger.trace_tensor(timbre_input)
        self.logger.trace_tensor(len_crop)
        i_spmel_gt = self.fold_pad(spmel_gt)
        repeater = i_spmel_gt.size(dim=0)
        return (
            [fnamelist[0] for i in range(repeater)],
            [spk_id_org[0] for i in range(repeater)],
            i_spmel_gt,
            self.fold_pad(rhythm_input),
            self.fold_pad(content_input),
            self.repeat_and_pad(pitch_input, repeater),
            torch.stack([timbre_input.squeeze() for i in range(repeater)]),
            torch.stack([len_crop.squeeze() for i in range(repeater)]),
        )

    @torch.no_grad()
    def repeat_and_pad(
        self: Self,
        item: torch.Tensor,
        dim0: int,
    ) -> torch.Tensor:
        pads = (0, self.fold_size - item.size(-1))
        if item.size(dim=-1) > self.config.model.max_len_pad:
            paditem = torch.nn.functional.pad(item.squeeze(), pads)
            return torch.stack([paditem for i in range(dim0)])
        else:
            return torch.nn.functional.pad(item, pads)

    @torch.no_grad()
    def fold_pad(
        self: Self,
        item: torch.Tensor,
    ) -> torch.Tensor:
        pads: Tuple[int, ...]
        if item.size(dim=-2) > self.config.model.max_len_pad:
            pad_i = (
                ((item.size(-2) // self.fold_size) + 1) * self.fold_size
                - item.size(-2)
                + self.fold_pad_len
            )
            pads = (0, 0, self.fold_pad_len, pad_i)
            full_pad = torch.nn.functional.pad(item.squeeze(), pads)
            ones_mat = torch.ones_like(full_pad)
            self.logger.trace_tensor(full_pad)
            norm_mat = ones_mat.unfold(
                dimension=-2, size=self.fold_size, step=self.fold_step
            )
            retunf = full_pad.unfold(
                dimension=-2, size=self.fold_size, step=self.fold_step
            )
            return (retunf / norm_mat).mT
        else:
            pads = (0, 0, 0, self.fold_size - item.size(-2))
            return torch.nn.functional.pad(item, pads)

    @torch.no_grad()
    def unstack(
        self: Self,
        listitem: List[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(listitem) == 1:
            return listitem[0]
        gt_mod = torch.stack([item[0] for item in listitem]).transpose(0, -1)
        out_mod = torch.stack([item[1] for item in listitem]).transpose(0, -1)
        fold_fn = torch.nn.Fold(
            output_size=(1, ((gt_mod.size(-1) + 1) * self.fold_step)),
            kernel_size=(1, self.fold_size),
            stride=(1, self.fold_step),
        )
        out_gt = fold_fn(gt_mod).squeeze(1).squeeze(1).T
        out_out = fold_fn(out_mod).squeeze(1).squeeze(1).T
        self.logger.trace_tensor(out_gt)
        self.logger.trace_tensor(out_out)
        return out_gt, out_out
