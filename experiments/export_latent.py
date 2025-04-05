import os
from typing import Self

import torch

from .experiment import Experiment

DataType = tuple[
    str,  # Filename
    str,  # Speaker ID string
    torch.Tensor,  # Mel Spectrogram
    torch.Tensor,  # Rhythm Input
    torch.Tensor,  # Content Input
    torch.Tensor,  # Pitch Input
    torch.Tensor,  # Timbre Input
    torch.Tensor,  # len_crop (required in padding)
]


class ExportLatents(Experiment):
    latents = [
        "code_exp_1",
        "code_exp_2",
        "code_exp_3",
        "code_exp_4",
    ]

    @torch.no_grad()
    def export(self: Self) -> None:
        self.logger.info("Running export")
        model_name = os.path.join(
            "speechsplit2-large",
            "trainmask-large-SpeechSplit2-2024-11-07.ckpt",
        )
        self.load_trained(model_name)
        self.logger.info("Full Process On")
        self.compute.set_gpu()
        ddir = os.path.join(self.experiment_dir, self.config.options.dataset_name)
        os.makedirs(ddir, exist_ok=True)
        self.experiment_dir = ddir
        self.process()

    def process(self: Self) -> None:
        self.load_data(singleitem=True, sequential=True)
        proc_data = [item for item in self.data_loader]
        proc_data = sorted(proc_data, key=lambda i: i[0])
        proc_name: set[str]
        proc_name = set(self.dataset.sample_name(item[0][0]) for item in proc_data)
        self.logger.trace_var(proc_name, "DEBUG")
        for name in proc_name:
            items: list[DataType] = sorted(
                [item for item in proc_data if str(item[0][0]).find(name) != -1],
                key=lambda i: i[0],
            )
            [
                self.process_item(
                    fname=fname,
                    spmel_gt=spmel_gt,
                    rhythm_input=rhythm_input,
                    content_input=content_input,
                    pitch_input=pitch_input,
                    timbre_input=timbre_input,
                    len_crop=len_crop,
                )
                for (
                    fname,
                    spk_id_org,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop,
                ) in items
            ]

    @torch.no_grad()
    def process_item(
        self: Self,
        fname: str,
        spmel_gt: torch.Tensor,
        rhythm_input: torch.Tensor,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        timbre_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> None:
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
        assert self.config.options.return_latents
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
        fbase = (fname[0].rpartition("/")[-1]).rpartition(".")[0]
        self.logger.debug(f"Saving: {fbase}")
        self.save_tensor(code_exp_1, f"{fbase}-L1.pt", save_raw=True)
        self.save_tensor(code_exp_2, f"{fbase}-L2.pt", save_raw=True)
        self.save_tensor(code_exp_3, f"{fbase}-L3.pt", save_raw=True)
        self.save_tensor(code_exp_4, f"{fbase}-L4.pt", save_raw=True)
