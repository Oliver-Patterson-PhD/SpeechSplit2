import os
import time
from typing import List, Self

import torch
import torchaudio

from data_preprocessing import clean_audio, new_epsilon
from experiments.experiment import Experiment
from transcribers.whisper import WhisperTranscriber
from transcribers.whisper.audio import N_SAMPLES, log_mel_spectrogram
from utils import filter_wav, get_spmel, quantize_f0_torch


class Scratchpad(Experiment):
    @torch.no_grad()
    def run(self: Self) -> None:
        run_loop = True
        if run_loop:
            self.load_data()
            self.model.eval()
            self.start_time = time.time()
            self.batch_size = self.config.dataloader.batch_size
        self.transcriber = WhisperTranscriber(
            device=self.compute.device(),
            model_name=self.config.options.whisper_name,
            config=self.config,
            output_dir=self.experiment_dir,
        )
        if not run_loop:
            spk_meta = getattr(
                __import__("meta_dicts"),
                self.config.options.dataset_name,
            )
            dir_name, spk_dir_list, _ = next(os.walk(self.config.paths.raw_wavs))
            for spk_dir in sorted(spk_dir_list):
                if spk_dir in spk_meta:
                    for root, _, files in os.walk(os.path.join(dir_name, spk_dir)):
                        self.proc(root, files)
        else:
            self.process()

    @torch.no_grad()
    def get_real_text(self: Self, fname: str):
        if self.config.options.dataset_name == "vctk":
            fname = os.path.splitext(fname)[0]
            spcode = fname[0:2]
            fullpath = "{}/VCTK-Corpus/txt/{}/{}.txt".format(
                self.paths.raw_data, spcode, fname
            )
            with open(fullpath, "rb") as txtfile:
                return txtfile.read().replace("\n", "")
        else:
            raise NotImplementedError

    @torch.no_grad()
    def proc(self: Self, root: str, files: List[str]) -> None:
        for fname in sorted(files):
            x: torch.Tensor
            loaded, fs = torchaudio.load(f"{root}/{fname}", channels_first=True)
            assert fs == 16000
            x = clean_audio(loaded)
            if x.shape[0] % 256 == 0:
                x = torch.cat(
                    (x, torch.tensor([new_epsilon], device=x.device)),
                    dim=0,
                )
            wav = filter_wav(x)
            spmel = log_mel_spectrogram(
                wav,
                self.transcriber.model.dims.n_mels,
                N_SAMPLES,
                self.compute.device(),
            )
            my_spmel = get_spmel(wav)
            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]
            self.logger.trace_tensor(spmel)
            self.logger.trace_tensor(my_spmel)
            self.logger.info(f"Transcribing LMS {fname}")
            self.transcriber.transcribe(spmel, f"{fname}_lms")
            self.logger.info(f"Transcribing mys {fname}")
            self.transcriber.transcribe(my_spmel, f"{fname}_mys")
            exit(0)

    @torch.no_grad()
    def process(self: Self) -> None:
        for item in self.data_loader:
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
            fname: str = fnamelist[0].split("/")[-1].split(".")[0]

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #

            # Move data to GPU if available
            pitch_input.unsqueeze_(-1)
            spmel_gt = spmel_gt.to(self.compute.device())
            rhythm_input = rhythm_input.to(self.compute.device())
            content_input = content_input.to(self.compute.device())
            pitch_input = pitch_input.to(self.compute.device())
            timbre_input = timbre_input.to(self.compute.device())
            len_crop = len_crop.to(self.compute.device())

            # Prepare input data and apply random resampling
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

            self.logger.info(
                "Saving {fname}".format(
                    fname=fname,
                )
            )
            self.transcriber.transcribe(spmel_gt, f"{fname}_gt")
            self.transcriber.transcribe(spmel_output, f"{fname}_out")

    @torch.no_grad()
    def prepare_input(
        self: Self,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> torch.Tensor:
        content_pitch_input = torch.cat(
            (content_input, pitch_input), dim=-1
        )  # [B, T, F+1]
        content_pitch_input_intrp = self.intrp(
            content_pitch_input, len_crop
        )  # [B, T, F+1]
        pitch_input_intrp = quantize_f0_torch(
            content_pitch_input_intrp[:, :, -1],
        )  # [B, T, 257]
        content_pitch_input_intrp_2 = torch.cat(
            # [B, T, F+257]
            (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp),
            dim=-1,
        )
        return content_pitch_input_intrp_2
