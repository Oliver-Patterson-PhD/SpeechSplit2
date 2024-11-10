import os
from typing import List, Self, Tuple

import torch

from util import Config, Logger
from utils import get_spenv, get_spmel, is_nan, vtlp


def __check(
    x: torch.Tensor,
    msg: str,
) -> torch.Tensor:
    assert (
        (x.size(dim=-1) > 1)
        and (x.max().item() > 1e-03)
        and (x != 0).any()
        and not is_nan(x)
        and (x != 0.0).any()
    ), msg
    return x


def __worker_init_fn(x):
    return torch.random.manual_seed(
        (torch.initial_seed()) % (2**32),
    )


class MyDataset(torch.utils.data.Dataset):
    config: Config
    logger: Logger
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

    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        self.config = config
        self.logger = Logger()
        self.dataset_name = config.options.dataset_name
        spk_meta = getattr(
            __import__("meta_dicts"),
            config.options.dataset_name,
        )
        _, spk_dir_list, _ = next(os.walk(config.paths.monowavs))
        self.dataset = [
            (
                spk_dir,
                torch.zeros(
                    (config.model.dim_spk_emb,),
                    dtype=torch.float32,
                ).index_fill(0, torch.tensor([int(spk_meta[spk_dir][0])]), 1),
                self.load_from_meta(str(os.path.join(spk_dir, filepath))),
                str(os.path.join(spk_dir, filepath)),
            )
            for spk_dir in self.logger.progress_bar(
                sorted(spk_dir_list), desc="speakers loaded"
            )
            if spk_dir in spk_meta
            for filepath in next(
                os.walk(
                    os.path.join(
                        config.paths.monowavs,
                        spk_dir,
                    )
                )
            )[-1]
        ]
        self.num_tokens = len(self.dataset)
        self.logger.trace_var(self.dataset)
        return

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
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.DoubleTensor,
    ]:
        spk_dir, spk_emb, (wav_mono, spmel, f0), fname = self.dataset[index]
        alpha: float = 0.2 * torch.rand(1).item() + 0.9
        pert_mono: torch.FloatTensor = vtlp(
            wav_mono, self.config.audio.sample_rate, alpha
        )
        len_crop = torch.tensor([self.config.model.max_len_seq], dtype=torch.double)
        return (
            fname,  # Filename
            spk_dir,  # Speaker ID string
            spmel,  # Mel Spectrogram
            __check(get_spenv(pert_mono), f"spenv invalid: {fname}"),  # Rhythm Input
            __check(get_spmel(pert_mono), f"spmel invalid: {fname}"),  # Content Input
            f0,  # Pitch Input
            spk_emb,  # Timbre Input
            len_crop,  # len_crop (required in padding)
        )

    def load_from_meta(
        self: Self,
        filepath: str,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        wav_mono: torch.Tensor = torch.load(
            os.path.join(self.config.paths.monowavs, filepath),
            weights_only=True,
        )
        spmel: torch.Tensor = torch.load(
            os.path.join(self.config.paths.spmels, filepath),
            weights_only=True,
        )
        f0: torch.Tensor = torch.load(
            os.path.join(self.config.paths.freqs, filepath),
            weights_only=True,
        )
        return (
            __check(wav_mono.float(), f"wav invalid: {filepath}"),
            __check(spmel.float(), f"spmel invalid: {filepath}"),
            __check(f0.float(), f"f0 invalid: {filepath}"),
        )


def get_loader(
    config: Config,
    sequential: bool = False,
) -> torch.utils.data.Dataloader:
    dataset: torch.utils.data.Dataset
    sampler: torch.utils.data.sampler.Sampler
    dataset = MyDataset(config)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset,
            replacement=config.dataloader.shuffle,
            num_samples=config.dataloader.samplier,
        )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1 if sequential else config.dataloader.batch_size,
        sampler=sampler,
        num_workers=0 if sequential else config.dataloader.num_workers,
        prefetch_factor=None if sequential else config.dataloader.num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=__worker_init_fn,
    )
    return data_loader
