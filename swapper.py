import os
from collections import OrderedDict
from glob import iglob
from itertools import product
from typing import Tuple

import torch
import torchaudio
from data_loader import get_loader
from data_preprocessing import MetaDictType
from meta_dicts import NamedMetaDictType
from model import Generator_3 as Generator
from model import InterpLnr
from util.compute import Compute
from util.logging import Logger
from utils import quantize_f0_torch
from parallelwavegan import Synthesizer


class Swapper(object):
    def __init__(self, config):
        self.logger = Logger()
        self.compute = Compute()
        self.return_latents = True
        self.model_type = "G"
        self.config = config
        self.compute.print_compute()
        self.compute.set_gpu()
        self.device = self.compute.device()

        self.build_model()
        self.restore_model()
        self.latents = [
            "code_exp_1",
            "code_exp_2",
            "code_exp_3",
            "code_exp_4",
        ]

    def build_model(self):
        self.model = Generator(self.config)
        # Print out the network information.
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.logger.info(self.model)
        self.logger.info(self.model_type)
        self.logger.info("The number of parameters: {}".format(num_params))
        self.model.to(self.device)

        self.Interp = InterpLnr(self.config)
        self.Interp.to(self.device)

    def restore_model(self):
        ckpt_name = "{}-{}-{}-{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.model_type,
            self.config.options.resume_iters,
        )
        self.logger.info(f"Loading model {ckpt_name}")
        ckpt = torch.load(
            os.path.join(self.config.paths.models, ckpt_name),
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)

    @torch.no_grad()
    def prepare_input(
        self,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> torch.Tensor:
        content_pitch_input = torch.cat(
            (content_input, pitch_input), dim=-1
        )  # [B, T, F+1]
        content_pitch_input_intrp = self.Interp(
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

    @torch.no_grad()
    def save_latents(self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/{self.latents[0]}"):
            return
        self.data_loader = get_loader(config=self.config, singleitem=True)
        [self.save_single_latent(batch) for batch in self.data_loader]  # type: ignore [func-returns-value]

    @torch.no_grad()
    def save_single_latent(self, batch) -> None:
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
        spmel_gt = spmel_gt.to(self.device)
        rhythm_input = rhythm_input.to(self.device)
        content_input = content_input.to(self.device)
        pitch_input = pitch_input.to(self.device)
        timbre_input = timbre_input.to(self.device)
        len_crop = len_crop.to(self.device)

        # Prepare input data and apply random resampling
        content_pitch_input = self.prepare_input(
            content_input,
            pitch_input,
            len_crop,
        )

        # Run model
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

        for latent in self.latents:
            latentfile = f"{self.config.paths.latents}/{latent}/{main_name}"
            os.makedirs(os.path.dirname(latentfile), exist_ok=True)
            torch.save(eval(latent), latentfile)

    @torch.no_grad()
    def swap_latents(self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/out_spec"):
            return

        speaker_data: NamedMetaDictType = getattr(
            __import__("meta_dicts"),
            f"named{self.config.options.dataset_name}",
        )
        meta_file = os.path.join(self.config.paths.features, "metadata.pkl")
        metadata: MetaDictType = torch.load(meta_file, weights_only=True)
        for dys, con in product(
            [speaker for speaker, data in speaker_data.items() if data.dysarthric],
            [speaker for speaker, data in speaker_data.items() if not data.dysarthric],
        ):
            [
                (
                    self.swap_single_latent(uttr, dys, con, latent),  # type: ignore [func-returns-value]
                    self.swap_single_latent(uttr, con, dys, latent),  # type: ignore [func-returns-value]
                )
                for latent in self.latents
                for uttr in get_valid(metadata, dys, con)
            ]

    @torch.no_grad()
    def swap_single_latent(
        self,
        uttr: str,
        dys: str,
        con: str,
        latent: str,
    ) -> None:
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
        code_spec = self.model.decode(
            code_1,
            code_2,
            code_3,
            code_4,
            192,
        )
        code_file = "{}/out_spec/{}-to-{}-{}/{}.pt".format(
            self.config.paths.latents, con, dys, swapped, uttr
        )
        os.makedirs(os.path.dirname(code_file), exist_ok=True)
        torch.save(code_spec, code_file)

    @torch.no_grad()
    def save_audios(self) -> None:
        self.compute.set_gpu()
        self.device = self.compute.device()
        self.synthesizer = Synthesizer(self.device)
        self.synthesizer.load_ckpt("full_models/lj_wavenet_vocoder.pth")
        [
            self.single_spmel_to_audio(file)  # type: ignore [func-returns-value]
            for file in iglob(
                f"{self.config.paths.latents}/out_spec/**/*.pt",
                recursive=True,
            )
        ]
        return

    @torch.no_grad()
    def single_spmel_to_audio(self, file: str) -> None:
        self.logger.debug(f"file: {file}")
        outfile = file.replace("out_spec", "out_wav").replace(".pt", ".wav")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        wav = self.synthesizer.spect2wav(torch.load(file, weights_only=True).squeeze())
        torchaudio.save(outfile, wav, sample_rate=16000, backend="soundfile")


def get_valid(
    meta: MetaDictType,
    dys: str,
    con: str,
) -> set:
    dys_uttrs = set(item[-1].split("/")[1][4:-3] for item in meta if item[0] == dys)
    con_uttrs = set(item[-1].split("/")[1][5:-3] for item in meta if item[0] == con)
    return dys_uttrs and con_uttrs


def get_code(
    fstring: str,
    name: str,
    latent: str,
    swap: str,
    orig: str,
) -> Tuple[bool, torch.Tensor]:
    speaker_code, swapped = (swap, True) if latent == name else (orig, False)
    filename = fstring.format(name, speaker_code)
    code = torch.load(filename, weights_only=True)
    return swapped, code
