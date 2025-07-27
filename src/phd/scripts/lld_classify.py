# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

import pickle
from typing import overload

import torch

from ..data import AudioProcs, DatasetParser
from ..data.dataset import SampleInfo
from ..util import compute, config, logger
from ..util.arff import ArffRowType
from ..util.arff import load as arff_load
from ..util.file import basename, exists, newpath, path, walkfiles
from ..util.tensor import Tensor

parser = DatasetParser()
processor = AudioProcs()

compute.set_gpu()
compute.set_default()

experiment_dir = newpath(config.paths.artefacts, basename(__name__))
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate
lld_len = 26
testing = False

LLD_Data = tuple[str, Tensor]


class LLD:
    name: str
    frametime: float
    loudness: float
    alpharatio: float
    hammarbergindex: float
    slope0to500: float
    slope500to1500: float
    spectralflux: float
    mfcc1: float
    mfcc2: float
    mfcc3: float
    mfcc4: float
    f0semitone: float
    jitter: float
    shimmer: float
    hnr: float
    logrelf0h1h2: float
    logrelf0h1a3: float
    f1frequency: float
    f1bandwidth: float
    f1amplitude: float
    f2frequency: float
    f2bandwidth: float
    f2amplitude: float
    f3frequency: float
    f3bandwidth: float
    f3amplitude: float

    def __init__(self, row: ArffRowType | None = None) -> None:
        if row is None:
            return

        def add_attr(name: str, item: str, t: type) -> None:
            thing = row.get(item)
            if isinstance(thing, t):
                setattr(self, name, thing)
            else:
                raise ValueError(
                    f"Incorrect type for {item}: {type(thing)}, expected {t}"
                )

        add_attr("name", "name", str)
        add_attr("frametime", "frameTime", float)
        add_attr("loudness", "Loudness_sma3", float)
        add_attr("alpharatio", "alphaRatio_sma3", float)
        add_attr("hammarbergindex", "hammarbergIndex_sma3", float)
        add_attr("slope0to500", "slope0-500_sma3", float)
        add_attr("slope500to1500", "slope500-1500_sma3", float)
        add_attr("spectralflux", "spectralFlux_sma3", float)
        add_attr("mfcc1", "mfcc1_sma3", float)
        add_attr("mfcc2", "mfcc2_sma3", float)
        add_attr("mfcc3", "mfcc3_sma3", float)
        add_attr("mfcc4", "mfcc4_sma3", float)
        add_attr("f0semitone", "F0semitoneFrom27.5Hz_sma3nz", float)
        add_attr("jitter", "jitterLocal_sma3nz", float)
        add_attr("shimmer", "shimmerLocaldB_sma3nz", float)
        add_attr("hnr", "HNRdBACF_sma3nz", float)
        add_attr("logrelf0h1h2", "logRelF0-H1-H2_sma3nz", float)
        add_attr("logrelf0h1a3", "logRelF0-H1-A3_sma3nz", float)
        add_attr("f1frequency", "F1frequency_sma3nz", float)
        add_attr("f1bandwidth", "F1bandwidth_sma3nz", float)
        add_attr("f1amplitude", "F1amplitudeLogRelF0_sma3nz", float)
        add_attr("f2frequency", "F2frequency_sma3nz", float)
        add_attr("f2bandwidth", "F2bandwidth_sma3nz", float)
        add_attr("f2amplitude", "F2amplitudeLogRelF0_sma3nz", float)
        add_attr("f3frequency", "F3frequency_sma3nz", float)
        add_attr("f3bandwidth", "F3bandwidth_sma3nz", float)
        add_attr("f3amplitude", "F3amplitudeLogRelF0_sma3nz", float)

    def to_data(self) -> LLD_Data:
        return (
            self.name,
            torch.tensor(
                [
                    self.frametime,
                    self.loudness,
                    self.alpharatio,
                    self.hammarbergindex,
                    self.slope0to500,
                    self.slope500to1500,
                    self.spectralflux,
                    self.mfcc1,
                    self.mfcc2,
                    self.mfcc3,
                    self.mfcc4,
                    self.f0semitone,
                    self.jitter,
                    self.shimmer,
                    self.hnr,
                    self.logrelf0h1h2,
                    self.logrelf0h1a3,
                    self.f1frequency,
                    self.f1bandwidth,
                    self.f1amplitude,
                    self.f2frequency,
                    self.f2bandwidth,
                    self.f2amplitude,
                    self.f3frequency,
                    self.f3bandwidth,
                    self.f3amplitude,
                ]
            ),
        )


@overload
def make_lld(name: str, tensor: Tensor) -> LLD:
    pass


@overload
def make_lld(name: list[str], tensor: Tensor) -> list[LLD]:
    pass


def make_lld(name: str | list[str], tensor: Tensor) -> LLD | list[LLD]:
    if not tensor.size(dim=-1) == lld_len:
        raise TypeError(
            f"Invalid LLD tensor shape: {tensor.shape}\n"
            f"Last dimension must be {lld_len}"
        )
    if tensor.size(dim=0) == 1:
        tensor.squeeze_()
    if tensor.ndimension() == 1:
        if not isinstance(name, str):
            raise TypeError(f"invalid LLD set with tensor shape: {tensor.shape}")
        lld = LLD()
        lld.name = name
        lld.frametime = tensor[0].item()
        lld.loudness = tensor[1].item()
        lld.alpharatio = tensor[2].item()
        lld.hammarbergindex = tensor[3].item()
        lld.slope0to500 = tensor[4].item()
        lld.slope500to1500 = tensor[5].item()
        lld.spectralflux = tensor[6].item()
        lld.mfcc1 = tensor[7].item()
        lld.mfcc2 = tensor[8].item()
        lld.mfcc3 = tensor[9].item()
        lld.mfcc4 = tensor[10].item()
        lld.f0semitone = tensor[11].item()
        lld.jitter = tensor[12].item()
        lld.shimmer = tensor[13].item()
        lld.hnr = tensor[14].item()
        lld.logrelf0h1h2 = tensor[15].item()
        lld.logrelf0h1a3 = tensor[16].item()
        lld.f1frequency = tensor[17].item()
        lld.f1bandwidth = tensor[18].item()
        lld.f1amplitude = tensor[19].item()
        lld.f2frequency = tensor[20].item()
        lld.f2bandwidth = tensor[21].item()
        lld.f2amplitude = tensor[22].item()
        lld.f3frequency = tensor[23].item()
        lld.f3bandwidth = tensor[24].item()
        lld.f3amplitude = tensor[25].item()
        return lld
    elif tensor.ndimension() == 2:
        if not isinstance(name, list):
            raise TypeError(f"invalid LLD set with tensor shape: {tensor.shape}")
        return [make_lld(iname, batch) for iname, batch in zip(name, tensor)]
    else:
        raise RuntimeError(f"Invalid tensor dimensions for: {tensor.shape}")
    return lld


class LLDDataset(torch.utils.data.Dataset[LLD_Data]):
    dataset: list[LLD]
    small_cache_file: str = path(
        config.paths.proc_data, "OpenSMILE", "LLDDataset-small.pkl"
    )
    large_cache_file: str = path(config.paths.proc_data, "OpenSMILE", "LLDDataset.pkl")
    cache_file: str

    def __init__(self) -> None:
        self.cache_file = self.small_cache_file if testing else self.large_cache_file
        if not exists(self.cache_file):
            logger.info("Generating LLD Dataset")
            lld_path = path(config.paths.proc_data, "OpenSMILE", "custom-arff")
            arff_files = walkfiles(lld_path, fullpaths=True)
            if len(arff_files) == 0:
                raise Exception("ERROR: No LLD files in dataset")
            self.dataset = []
            [
                self.dataset.extend(
                    [
                        lld
                        for lld in [LLD(line) for line in arff_load(file)]
                        if lld.name is not None
                    ]
                )
                for file in logger.progress_bar(arff_files, unit="files")
            ]
            logger.info("Saving LLD Dataset")
            with open(self.large_cache_file, "wb") as pklfile:
                pickle.dump(self.dataset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.small_cache_file, "wb") as pklfile:
                pickle.dump(
                    self.dataset[0:100], pklfile, protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            logger.info("Loading LLD Dataset")
            with open(self.cache_file, "rb") as pklfile:
                self.dataset = pickle.load(pklfile)
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LLD_Data:
        return self.dataset[index].to_data()


class LLDClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=13,
            out_channels=2,
            kernel_size=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_vdim = (x.size(dim=0), 13, 2) if x.ndim == 2 else (13, 2)
        x_view = x.view(*x_vdim)
        logger.trace_tensor(x_view)
        logger.trace_nans(x_view)
        x_conv = self.conv(x_view)
        logger.trace_tensor(x_conv)
        logger.trace_nans(x_conv)
        return x_conv


def arff_init_worker(x: int) -> None:
    return ((torch.initial_seed()) % (2**32),)  # type: ignore


def arff_result(name: str) -> SampleInfo:
    return SampleInfo(name)


def arff_results(names: list[str]) -> Tensor:
    results = torch.tensor(
        [[0.0, 1.0] if arff_result(name).dysarthric else [1.0, 0.0] for name in names]
    ).unsqueeze(-1)
    logger.trace_tensor(results)
    return results


def lld_classify() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()

    arff_dataset = LLDDataset()
    arff_batch_size = 64
    arff_samplier = 1
    arff_n_workers = 8
    arff_sampler = torch.utils.data.RandomSampler(
        data_source=arff_dataset,
        replacement=True,
        generator=torch.Generator(device=compute.device()),
        num_samples=(arff_batch_size * len(arff_dataset) * arff_samplier),
    )
    arff_data = torch.utils.data.DataLoader(
        dataset=arff_dataset,
        batch_size=arff_batch_size,
        sampler=arff_sampler,
        num_workers=arff_n_workers,
        prefetch_factor=None if arff_n_workers == 0 else arff_n_workers,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=arff_init_worker,
    )

    arff_model = LLDClassifier()
    arff_model.train()
    arff_optim = torch.optim.SGD(
        arff_model.parameters(),
        lr=0.001,
        momentum=0.9,
    )
    arff_loss = torch.nn.CrossEntropyLoss()

    log_div = 100
    for epoch in range(100):
        logger.debug(f"Epoch: {epoch}")
        running_loss = 0.0
        for i, (names, items) in enumerate(arff_data):
            logger.trace_var(items)
            logger.trace_tensor(items)
            logger.trace_nans(items)

            out_data = arff_model(items)
            logger.trace_tensor(out_data)
            logger.trace_nans(out_data)

            truth = arff_results(names)
            logger.trace_tensor(truth)
            logger.trace_nans(truth)

            loss = arff_loss(out_data, truth)
            loss.backward()
            arff_optim.step()
            running_loss += loss.item()

            if i % log_div == 0:
                logger.info(f"[{epoch}, {i:7d}] loss: {running_loss / log_div}")
                running_loss = 0.0
        torch.save((arff_model, arff_optim), path(config.paths.artefacts, f"arff_model-{epoch}-{i}.pt"))
