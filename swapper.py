import os
from collections import OrderedDict
from glob import glob
from itertools import product
from typing import Self, Tuple

import torch
import torchaudio
from data_loader import CollaterItemType, get_loader
from data_preprocessing import MetaDictType
from meta_dicts import NamedMetaDictType
from model import Generator_3 as Generator
from model import InterpLnr
from synthesizers.melgan import MelGanSynthesizer as MelGan
from synthesizers.parallelwavegan import ParallelWaveGanSynthesizer as ParWavGan
from synthesizers.synthesizer import Synthesizer
from synthesizers.wavenet import WavenetSynthesizer as Wavenet
from util.compute import Compute
from util.config import Config
from util.logging import Logger
from utils import norm_audio, quantize_f0_torch, save_tensor


class GriffinLim(Synthesizer):
    n_fft = 1024
    hop_length = 256
    dim_freq = 80
    f_min = 90
    f_max = 7600
    power = 1
    sample_rate = 16000
    logger = Logger()

    def __init__(
        self,
        device: torch.device,
        config: Config,
    ) -> None:
        self.demel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.dim_freq,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=None,
            mel_scale="htk",
            driver="gels",
        )
        self.glim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            power=self.power,
        )

    def spect2wav(self, spect: torch.Tensor) -> torch.Tensor:
        tspec = spect.T
        self.logger.trace_var(tspec.shape)
        return self.glim(self.demel(tspec))


class Swapper(object):
    synthesizer: Synthesizer
    use_synth_melgan: bool = True
    use_synth_parallelwavegan: bool = True
    use_synth_wavenet: bool = True
    use_synth_griffinlim: bool = True

    def __init__(self: Self, config: Config) -> None:
        self.logger = Logger()
        self.compute = Compute()
        self.return_latents = True
        self.model_type = "SpeechSplit2"
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

    def build_model(self: Self) -> None:
        self.model = Generator(self.config)
        self.intrp = InterpLnr(self.config)
        # Print out the network information.
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.model.to(self.device)
        self.intrp.to(self.device)
        self.model.eval()
        self.intrp.eval()
        self.logger.info(self.model_type)
        # self.logger.info(str(self.model))
        # self.logger.info("The number of parameters: {}".format(num_params))

    def restore_model(self: Self) -> None:
        ckpt_name = "{}-{}-{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.model_type,
        )
        self.logger.info(f"Loading model {ckpt_name}")
        ckpt = torch.load(
            os.path.join(self.config.paths.trained_models, ckpt_name),
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k] = v
            self.logger.debug("model keys:")
            self.logger.debug(ckpt["model"].keys())
            self.logger.debug("")
            self.logger.debug("new_state_dict keys:")
            for key in new_state_dict.keys():
                self.logger.debug(key)
            self.model.load_state_dict(new_state_dict)
        finally:
            exit

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

    @torch.no_grad()
    def save_latents(self: Self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/{self.latents[0]}"):
            return
        self.data_loader = get_loader(config=self.config, singleitem=True)
        [self.save_single_latent(batch) for batch in self.data_loader]  # type: ignore [func-returns-value]

    @torch.no_grad()
    def save_single_latent(self: Self, batch: CollaterItemType) -> None:
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
            save_tensor(eval(latent), f"{latentfile}.png")
            torch.save(eval(latent), latentfile)

    @torch.no_grad()
    def swap_latents(self: Self) -> None:
        if os.path.exists(f"{self.config.paths.latents}/out_spec"):
            return

        speaker_data: NamedMetaDictType = getattr(
            __import__("meta_dicts"),
            f"named{self.config.options.dataset_name}",
        )
        meta_file = os.path.join(self.config.paths.features, "metadata.pkl")
        metadata: MetaDictType = torch.load(meta_file, weights_only=True)

        [
            self.swap_single_latent(uttr, spk, spk, "None")  # type: ignore [func-returns-value]
            for spk in speaker_data.keys()
            for uttr in get_valid(metadata, spk, spk)
        ]
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
        self: Self,
        uttr: str,
        dys: str,
        con: str,
        latent: str,
    ) -> None:
        fstring = self.config.paths.latents + "/{0}/{1}/{1}_" + uttr + ".pt"
        self.logger.debug(fstring)
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
        spec_file = code_file.replace("out_spec", "out_imag").replace(".pt", ".png")
        os.makedirs(os.path.dirname(spec_file), exist_ok=True)
        save_tensor(code_spec, spec_file)

    @torch.no_grad()
    def save_audios(self: Self) -> None:
        self.compute.set_gpu()
        self.device = self.compute.device()

        [
            self.spec_image(file, "orig")
            for file in glob(
                f"{self.config.paths.spmels}/**/*_0.pt",
                recursive=True,
            )
        ]

        filelist = glob(
            f"{self.config.paths.latents}/out_spec/**/*_0.pt",
            recursive=True,
        )
        [self.spec_image(file, "full") for file in filelist]

        if self.use_synth_griffinlim:
            self.synthesizer = GriffinLim(self.device, config=self.config)
            [self.single_spmel_to_audio(file, "griffinlim") for file in filelist]

        if self.use_synth_melgan:
            self.synthesizer = MelGan(self.device, config=self.config)
            [self.single_spmel_to_audio(file, "melgan") for file in filelist]

        if self.use_synth_parallelwavegan:
            self.synthesizer = ParWavGan(self.device, config=self.config)
            [self.single_spmel_to_audio(file, "parallelwavegan") for file in filelist]

        if self.use_synth_wavenet:
            self.synthesizer = Wavenet(self.device, config=self.config)
            [self.single_spmel_to_audio(file, "wavenet") for file in filelist]

        return

    def spec_image(self: Self, file: str, name: str) -> None:
        if name == "orig":
            inpath = f"{self.config.paths.spmels}"
        else:
            inpath = f"{self.config.paths.latents}/out_spec"
        outpath = f"{self.config.paths.latents}/out_spec_{name}"
        outfile = file.replace(inpath, outpath).replace("_0.pt", ".png")
        filelist = sorted(glob(file.replace("_0.pt", "_*.pt")))
        if len(filelist) > 1:
            spec = torch.cat(
                tuple(
                    [torch.load(file, weights_only=True).squeeze() for file in filelist]
                ),
                dim=0,
            )
        else:
            spec = torch.load(file, weights_only=True).squeeze()
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        save_tensor(spec, outfile)

    @torch.no_grad()
    def single_spmel_to_audio(self: Self, file: str, name: str) -> None:
        outfile = file.replace("out_spec", f"out_wav_{name}").replace("_0.pt", ".wav")
        self.logger.debug(f"Creating Audio: {outfile}")
        filelist = sorted(glob(file.replace("_0.pt", "_*.pt")))
        if len(filelist) > 1:
            self.logger.debug(file)
            spec = torch.cat(
                tuple(
                    [torch.load(file, weights_only=True).squeeze() for file in filelist]
                ),
                dim=0,
            )
        else:
            spec = torch.load(file, weights_only=True).squeeze()
        wav = self.synthesizer.spect2wav(spec).unsqueeze(dim=0)
        norm_wav = norm_audio(wav)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        torchaudio.save(outfile, wav.cpu(), sample_rate=16000, backend="sox")
        torchaudio.save(
            outfile.replace(".wav", "-norm.wav"),
            norm_wav.cpu(),
            sample_rate=16000,
            backend="sox",
        )


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
