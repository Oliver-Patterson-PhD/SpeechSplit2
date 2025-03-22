__all__ = [
    "get_loader",
    "MyDataset",
]

import os
from typing import List, Self, Tuple

import torch

from util import Config, Logger, Compute

from .utils import AudioProcs


class MyDataset(torch.utils.data.Dataset):
    dataset_name: str
    dataset: List[
        Tuple[
            str,  # spk_dir
            torch.Tensor,  # spk_emb
            Tuple[
                torch.Tensor,  # wav_mono
                torch.Tensor,  # spmel
                torch.Tensor,  # f0
            ],
            str,  # filepath
        ]
    ]
    num_tokens: int
    sample_rate: int
    max_len_seq: int
    path_freqs: str
    path_monowavs: str
    path_spmels: str
    full_info: bool
    map_device: torch.device

    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        self.dataset_name = config.options.dataset_name
        self.sample_rate = config.audio.sample_rate
        self.max_len_seq = config.model.max_len_seq
        self.path_freqs = config.paths.freqs
        self.path_monowavs = config.paths.monowavs
        self.path_fullwavs = config.paths.fullwavs
        self.path_spmels = config.paths.spmels
        self.full_info = not config.options.train
        self.myproc = AudioProcs(config)
        spk_meta = getattr(__import__("meta_dicts"), config.options.dataset_name)
        _, spk_dir_list, _ = next(os.walk(config.paths.monowavs))
        self.map_device = torch.device("cpu")
        self.dataset = [
            (
                str(spk_dir),
                torch.zeros(
                    (config.model.dim_spk_emb,), dtype=torch.float32
                ).index_fill(0, torch.tensor([int(spk_meta[spk_dir][0])]), 1),
                self.load_from_meta(str(os.path.join(spk_dir, filepath))),
                str(os.path.join(spk_dir, filepath)),
            )
            for spk_dir in Logger().progress_bar(
                sorted(spk_dir_list), desc="speakers loaded"
            )
            if spk_dir in spk_meta
            for filepath in next(
                os.walk(
                    os.path.join(
                        (
                            config.paths.fullwavs
                            if self.full_info
                            else config.paths.monowavs
                        ),
                        spk_dir,
                    )
                )
            )[2]
        ]
        self.num_tokens = len(self.dataset)

    def pinnable(
        self: Self,
    ) -> bool:
        return Compute().could_be_gpu() and self.map_device == torch.device("cpu")

    def __len__(
        self: Self,
    ) -> int:
        return self.num_tokens

    def __getitem__(
        self: Self,
        index: int,
    ) -> Tuple[
        str,
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        spk_dir, spk_emb, (wav_mono, spmel, f0), fname = self.dataset[index]
        p_mono: torch.Tensor
        if self.full_info:
            p_mono = wav_mono
        else:
            alpha: float = 0.2 * torch.rand(1).item() + 0.9
            p_mono = self.myproc.vtlp(wav_mono, self.sample_rate, alpha)
        len_crop = torch.tensor([self.max_len_seq], dtype=torch.double, device="cpu")
        spenv = self.check(self.myproc.get_spenv(p_mono), f"spenv invalid: {fname}")
        p_mel, _ = self.myproc.get_spmel(p_mono)
        spmel = self.check(p_mel, f"spmel invalid: {fname}")
        return (
            fname,  # Filename
            spk_dir,  # Speaker ID string
            spmel,  # Mel Spectrogram
            spenv,  # Rhythm Input
            spmel,  # Content Input
            f0,  # Pitch Input
            spk_emb,  # Timbre Input
            len_crop,  # len_crop (required in padding)
        )

    def load_from_meta(
        self: Self,
        filepath: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav_mono: torch.Tensor
        spmel: torch.Tensor
        f0: torch.Tensor
        if self.full_info:
            wav_mono = torch.load(
                os.path.join(self.path_fullwavs, filepath),
                weights_only=True,
                map_location=self.map_device,
            )
        else:
            wav_mono = torch.load(
                os.path.join(self.path_monowavs, filepath),
                weights_only=True,
                map_location=self.map_device,
            )
        spmel = torch.load(
            os.path.join(self.path_spmels, filepath),
            weights_only=True,
            map_location=self.map_device,
        )
        f0 = torch.load(
            os.path.join(self.path_freqs, filepath),
            weights_only=True,
            map_location=self.map_device,
        )
        o_wav_mono = self.check(wav_mono.float(), f"wav invalid: {filepath}")
        o_spmel = self.check(spmel.float(), f"spmel invalid: {filepath}")
        o_f0 = self.check(f0.float(), f"f0 invalid: {filepath}")
        return (o_wav_mono, o_spmel, o_f0)

    def check(
        self: Self,
        x: torch.Tensor,
        msg: str,
    ) -> torch.Tensor:
        assert (
            (x.size(dim=-1) > 1)
            and (x.max().item() > 1e-03)
            and (x != 0).any()
            and not x.isnan().any().item()
            and (x != 0.0).any()
        ), msg
        return x


def worker_init_fn(x):
    return torch.random.manual_seed(
        (torch.initial_seed()) % (2**32),
    )


def get_loader(
    config: Config,
    sequential: bool = False,
    singleitem: bool = False,
) -> torch.utils.data.DataLoader:
    logger = Logger()
    logger.debug(f"Initialising DataLoader for {config.options.dataset_name}")
    dataset: torch.utils.data.Dataset
    sampler: torch.utils.data.sampler.Sampler
    dataset = MyDataset(config)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        gen = torch.Generator(device=Compute().device())
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset,
            replacement=True,
            generator=gen,
            num_samples=(
                (len(dataset) * config.dataloader.samplier)
                if singleitem
                else (
                    config.dataloader.batch_size
                    * len(dataset)
                    * config.dataloader.samplier
                )
            ),
        )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1 if singleitem else config.dataloader.batch_size,
        sampler=sampler,
        num_workers=0 if singleitem else config.dataloader.num_workers,
        prefetch_factor=None if singleitem else config.dataloader.num_workers,
        drop_last=False,
        pin_memory=dataset.pinnable(),
        worker_init_fn=worker_init_fn,
    )
    logger.debug("Created DataLoader")
    return data_loader
