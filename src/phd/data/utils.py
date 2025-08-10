from __future__ import annotations

__all__ = [
    "make_loader",
    "split_loaders",
    "DataLoader",
    "Dataset",
]

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..util import compute, config, logger


class Dataset[T](torch.utils.data.Dataset[T]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index) -> T:
        raise NotImplementedError(
            self.__class__.__name__
            + "does not define function: __getitem__(self, index) -> T"
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            self.__class__.__name__ + "does not define function: __len__(self) -> int"
        )


def _init_worker(x: int) -> None:
    return ((torch.initial_seed()) % (2**32),)  # type: ignore


def make_loader(
    dset: Dataset,
    batch_size: int | None = None,
    parallel: bool = True,
    shuffle: bool = config.dataloader.shuffle,
) -> DataLoader:
    sampler = (
        RandomSampler(
            data_source=dset,
            replacement=False,
            generator=torch.Generator(device=compute.device()),
            num_samples=len(dset),
        )
        if shuffle
        else SequentialSampler(
            data_source=dset,
        )
    )
    p = parallel and (config.dataloader.num_workers > 0)
    loader = DataLoader(
        dataset=dset,
        batch_size=batch_size or config.dataloader.batch_size,
        sampler=sampler,
        num_workers=config.dataloader.num_workers if p else 0,
        prefetch_factor=config.dataloader.num_workers if p else None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_init_worker if p else None,
    )
    return loader


def split_loaders[T](
    dset: Dataset[T],
    test_split: float = 0.2,
    device: torch.device | None = None,
    batch_size: int | None = None,
    parallel: bool = True,
) -> tuple[DataLoader[T], DataLoader[T]]:
    dset_t = dset.__class__
    train_split = 1.0 - test_split
    gen = torch.Generator(device=device or compute.device()).manual_seed(42)
    train, test = torch.utils.data.random_split(
        dataset=dset,
        lengths=(train_split, test_split),
        generator=gen,
    )
    logger.debug(f"Splitting training data for {dset.__class__.__name__}")
    train_data = dset_t([dset.dump(i) for i in train.indices])
    logger.debug(f"Splitting testing data for {dset.__class__.__name__}")
    test_data = dset_t([dset.dump(i) for i in test.indices])
    logger.debug(f"Creating Samplers for {dset.__class__.__name__}")
    train_sampler = RandomSampler(
        data_source=train_data,
        replacement=False,
        generator=torch.Generator(device=compute.device()),
        num_samples=len(train_data),
    )
    test_sampler = SequentialSampler(data_source=test_data)
    p = parallel and (config.dataloader.num_workers > 0)
    logger.debug(
        "Creating {} DataLoaders for {}".format(
            "Parallel" if p else "Sequential",
            dset.__class__.__name__,
        )
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size or config.dataloader.batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader.num_workers if p else 0,
        prefetch_factor=config.dataloader.num_workers if p else None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_init_worker if p else None,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size or config.dataloader.batch_size,
        sampler=test_sampler,
        num_workers=config.dataloader.num_workers if p else 0,
        prefetch_factor=config.dataloader.num_workers if p else None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_init_worker if p else None,
    )
    logger.debug(f"Data split for {dset.__class__.__name__}")
    return (train_loader, test_loader)
