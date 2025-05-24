import os
from itertools import product

import torch

from util.tensor import Tensor, save_tensor

from .experiment import Experiment


class Swapper(Experiment):
    latents = [
        "code_exp_1",
        "code_exp_2",
        "code_exp_3",
        "code_exp_4",
    ]

    DataItem = tuple[
        list[str], list[str], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]

    @torch.no_grad()
    def save_latents(self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/{self.latents[0]}"):
            return
        self.load_data(singleitem=True, sequential=True)
        [self.save_single_latent(batch) for batch in self.data_loader]  # type: ignore [func-returns-value]

    @torch.no_grad()
    def save_single_latent(self, batch: DataItem) -> None:
        (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = batch
        main_name = fname[0]
        self.logger.debug(f"Saving Latents for: {main_name}")
        # Move data to GPU if available
        spmel_gt = spmel_gt.to(self.compute.device())
        rhythm_input = rhythm_input.to(self.compute.device())
        content_input = content_input.to(self.compute.device())
        pitch_input = pitch_input.to(self.compute.device()).unsqueeze(-1)
        timbre_input = timbre_input.to(self.compute.device())
        len_crop = len_crop.to(self.compute.device())

        self.logger.trace_tensor(spmel_gt, "DEBUG")
        self.logger.trace_tensor(rhythm_input, "DEBUG")
        self.logger.trace_tensor(content_input, "DEBUG")
        self.logger.trace_tensor(pitch_input, "DEBUG")
        self.logger.trace_tensor(timbre_input, "DEBUG")
        self.logger.trace_tensor(len_crop, "DEBUG")

        # Prepare input data and apply random resampling
        content_pitch_input = self.prepare_input(content_input, pitch_input, len_crop)

        # Run model
        (spmel_output, code_exp_1, code_exp_2, code_exp_3, code_exp_4) = self.model(
            content_pitch_input, rhythm_input, timbre_input
        )

        for latent in self.latents:
            latentfile = f"{self.config.paths.latents}/{latent}/{main_name}"
            os.makedirs(os.path.dirname(latentfile), exist_ok=True)
            save_tensor(eval(latent), f"{latentfile}.png")
            torch.save(eval(latent), latentfile)

    @torch.no_grad()
    def swap_latents(self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/out_spec"):
            return

        speakers = self.dataset.speakers()

        [
            self.swap_single_latent(uttr, spk, spk, "None")  # type: ignore [func-returns-value]
            for spk in speakers
            for uttr in self.get_valid(spk, spk)
        ]

        con_speakers = set(spk for spk in speakers if self.dataset.dysarthric(spk))
        dys_speakers = set(spk for spk in speakers if not self.dataset.dysarthric(spk))
        [
            (
                self.swap_single_latent(uttr, dys, con, latent),  # type: ignore [func-returns-value]
                self.swap_single_latent(uttr, con, dys, latent),  # type: ignore [func-returns-value]
            )
            for dys, con in product(dys_speakers, con_speakers)
            for latent in self.latents
            for uttr in self.get_valid(dys, con)
        ]

    @torch.no_grad()
    def swap_single_latent(self, uttr: str, dys: str, con: str, latent: str) -> None:
        fstring = self.config.paths.latents + "/{0}/{1}/{1}_" + uttr + ".pt"
        c1, code_1 = get_code(fstring, "code_exp_1", latent, dys, con)
        c2, code_2 = get_code(fstring, "code_exp_2", latent, dys, con)
        c3, code_3 = get_code(fstring, "code_exp_3", latent, dys, con)
        c4, code_4 = get_code(fstring, "code_exp_4", latent, dys, con)
        if c1:
            swapped = "Sync-Code-1"
        elif c2:
            swapped = "Rhythm-Code"
        elif c3:
            swapped = "Sync-Code-2"
        elif c4:
            swapped = "Speaker-Embedding"
        else:
            swapped = "None"
        code_spec = self.model.decode(code_1, code_2, code_3, code_4, 192)
        code_file = "{}/out_spec/{}-to-{}-{}/{}.pt".format(
            self.config.paths.latents, con, dys, swapped, uttr
        )
        os.makedirs(os.path.dirname(code_file), exist_ok=True)
        torch.save(code_spec, code_file)
        spec_file = code_file.replace("out_spec", "out_imag").replace(".pt", ".png")
        os.makedirs(os.path.dirname(spec_file), exist_ok=True)
        save_tensor(code_spec.flip(-1).mT, spec_file)

    def get_valid(self, dys: str, con: str) -> set:
        dys_uttrs = self.dataset.get_utterances(dys)
        con_uttrs = self.dataset.get_utterances(con)
        return dys_uttrs and con_uttrs


@torch.no_grad()
def get_code(
    fstring: str, name: str, latent: str, swap: str, orig: str
) -> tuple[bool, Tensor]:
    speaker_code, swapped = (swap, True) if latent == name else (orig, False)
    filename = fstring.format(name, speaker_code)
    code = torch.load(filename, weights_only=True)
    return swapped, code
