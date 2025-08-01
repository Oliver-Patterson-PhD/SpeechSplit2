# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

import pickle
from typing import overload

import torch

from ..data import AudioProcs, DatasetParser
from ..data.dataset import SampleInfo
from ..data.utils import DataLoader, Dataset, split_loaders
from ..util import compute, config, logger
from ..util.arff import ArffRowType
from ..util.arff import load as arff_load
from ..util.file import basename, exists, newpath, path, walkfiles
from ..util.tensor import Tensor
from ..util.tensorboard import TensorBoard

TESTING = False
log_div = 100
eval_div = 1000

parser = DatasetParser()
processor = AudioProcs()
tb: TensorBoard

experiment_dir = newpath(config.paths.artefacts, basename(__name__))
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate

LLD_Data = tuple[str, Tensor]


def add_attr(self: object, row: ArffRowType, name: str, item: str, t: type) -> None:
    thing = row.get(item)
    if isinstance(thing, t):
        setattr(self, name, thing)
    else:
        raise ValueError(f"Incorrect type for {item}: {type(thing)}, expected {t}")


def dys_tensor(dysarthric: bool) -> list[float]:
    return [0.0, 1.0] if dysarthric else [1.0, 0.0]


def dys_bool(label: Tensor | list[float]) -> bool:
    if isinstance(label, Tensor):
        return label[0].item() < label[1].item()
    else:
        return label[0] < label[1]


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

    @classmethod
    def n_llds(cls) -> int:
        return 26

    def __init__(self, row: ArffRowType | None = None) -> None:
        if row is None:
            return

        add_attr(self, row, "name", "name", str)
        add_attr(self, row, "frametime", "frameTime", float)
        add_attr(self, row, "loudness", "Loudness_sma3", float)
        add_attr(self, row, "alpharatio", "alphaRatio_sma3", float)
        add_attr(self, row, "hammarbergindex", "hammarbergIndex_sma3", float)
        add_attr(self, row, "slope0to500", "slope0-500_sma3", float)
        add_attr(self, row, "slope500to1500", "slope500-1500_sma3", float)
        add_attr(self, row, "spectralflux", "spectralFlux_sma3", float)
        add_attr(self, row, "mfcc1", "mfcc1_sma3", float)
        add_attr(self, row, "mfcc2", "mfcc2_sma3", float)
        add_attr(self, row, "mfcc3", "mfcc3_sma3", float)
        add_attr(self, row, "mfcc4", "mfcc4_sma3", float)
        add_attr(self, row, "f0semitone", "F0semitoneFrom27.5Hz_sma3nz", float)
        add_attr(self, row, "jitter", "jitterLocal_sma3nz", float)
        add_attr(self, row, "shimmer", "shimmerLocaldB_sma3nz", float)
        add_attr(self, row, "hnr", "HNRdBACF_sma3nz", float)
        add_attr(self, row, "logrelf0h1h2", "logRelF0-H1-H2_sma3nz", float)
        add_attr(self, row, "logrelf0h1a3", "logRelF0-H1-A3_sma3nz", float)
        add_attr(self, row, "f1frequency", "F1frequency_sma3nz", float)
        add_attr(self, row, "f1bandwidth", "F1bandwidth_sma3nz", float)
        add_attr(self, row, "f1amplitude", "F1amplitudeLogRelF0_sma3nz", float)
        add_attr(self, row, "f2frequency", "F2frequency_sma3nz", float)
        add_attr(self, row, "f2bandwidth", "F2bandwidth_sma3nz", float)
        add_attr(self, row, "f2amplitude", "F2amplitudeLogRelF0_sma3nz", float)
        add_attr(self, row, "f3frequency", "F3frequency_sma3nz", float)
        add_attr(self, row, "f3bandwidth", "F3bandwidth_sma3nz", float)
        add_attr(self, row, "f3amplitude", "F3amplitudeLogRelF0_sma3nz", float)

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
    if not tensor.size(dim=-1) == LLD.n_llds():
        raise TypeError(
            f"Invalid LLD tensor shape: {tensor.shape}\n"
            f"Last dimension must be {LLD.n_llds()}"
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


class LLDDataset(Dataset[LLD_Data]):
    dataset: list[LLD]
    small_cache_file: str = path(
        config.paths.proc_data, "OpenSMILE", "LLDDataset-small.pkl"
    )
    large_cache_file: str = path(config.paths.proc_data, "OpenSMILE", "LLDDataset.pkl")
    cache_file: str
    small_size: int = 100

    def __init__(self, rawdata: list[LLD] | None = None) -> None:
        if rawdata is None:
            self.cache_file = (
                self.small_cache_file if TESTING else self.large_cache_file
            )
            if not exists(self.cache_file):
                logger.debug("LLDDataset Generating")
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
                logger.debug("LLDDataset Saving")
                with open(self.large_cache_file, "wb") as pklfile:
                    pickle.dump(
                        self.dataset,
                        pklfile,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                with open(self.small_cache_file, "wb") as pklfile:
                    pickle.dump(
                        self.dataset[0 : self.small_size - 1],
                        pklfile,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                logger.debug("LLDDataset Saved")
            else:
                logger.debug("LLDDataset Loading")
                with open(self.cache_file, "rb") as pklfile:
                    self.dataset = pickle.load(pklfile)
                logger.debug("LLDDataset Loaded")
        else:
            logger.debug("LLDDataset creating from raw list")
            self.dataset = rawdata
            logger.debug("LLDDataset created from raw list")
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LLD_Data:
        item = self.dataset[index]
        if isinstance(item, LLD):
            return item.to_data()
        else:
            return item


class LLDClassifier(torch.nn.Module):
    n_classes: int = 2
    __view_div: int = 2

    def __init__(self) -> None:
        super().__init__()
        assert LLD.n_llds() % self.__view_div == 0
        self.__i_len = LLD.n_llds() // self.__view_div
        i_size = self.__i_len * 2
        self.feat_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.__i_len, i_size, 2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(2, 1, 1, 1),
        )
        self.out_layers = torch.nn.Sequential(
            torch.nn.Linear(i_size * 2, self.n_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(dim=-1) == LLD.n_llds()
        x_vdim = (
            (x.size(dim=0), self.__i_len, self.__view_div)
            if x.ndim == 2
            else (self.__i_len, self.__view_div)
        )
        return self.out_layers(
            torch.flatten(self.feat_layers(x.view(*x_vdim)), start_dim=-2)
        )


def arff_init_worker(x: int) -> None:
    return ((torch.initial_seed()) % (2**32),)  # type: ignore


def arff_result(name: str) -> SampleInfo:
    return SampleInfo(name)


def arff_results(names: list[str]) -> Tensor:
    reslist = [dys_tensor(arff_result(name).dysarthric) for name in names]
    results = torch.tensor(reslist)
    return results


@torch.no_grad()
def lld_evaluate(model: LLDClassifier, val: DataLoader[LLD_Data], step: int) -> None:
    model.eval()
    valitems = [(n, i) for nn, ii in val for n, i in zip(nn, ii)]
    predictions = torch.tensor([model(items).tolist() for _, items in valitems])
    labels = torch.tensor([dys_tensor(parser.dysarthric(spk)) for spk, _ in valitems])
    tb.add_pr_curve("pr_curve", labels, predictions, step)
    good = [(dys_bool(pred) == dys_bool(lbl)) for pred, lbl in zip(predictions, labels)]
    tb.add_scalar("eval_true", sum(good) / len(good), step)
    tb.flush()
    model.train()


def lld_classify() -> None:
    global tb
    tb = TensorBoard()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()

    arff_train, arff_test = split_loaders(LLDDataset(), parallel=False)
    arff_model = LLDClassifier()
    arff_model.train()
    arff_optim = torch.optim.SGD(arff_model.parameters(), lr=0.001, momentum=0.9)
    arff_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        running_loss = 0.0
        i = 0
        for names, items in arff_train:
            i += 1
            step = (epoch * len(arff_train)) + i
            logger.trace_nans(items)

            out_data = arff_model(items)
            logger.trace_nans(out_data)

            truth = arff_results(names)
            logger.trace_nans(truth)

            loss = arff_loss(out_data, truth)
            loss.backward()
            arff_optim.step()
            running_loss += loss.item()

            tb.add_scalar(name="loss", item=loss.item(), step=step)
            if i % log_div == 0:
                logger.info(
                    "[{:10d}: {:5d}, {:7d}] loss: {}".format(
                        step, epoch, i, running_loss / log_div
                    )
                )
                running_loss = 0.0
            if i % eval_div == 0:
                lld_evaluate(arff_model, arff_test, step)
        torch.save(
            (arff_model, arff_optim), path(out_path, f"arff_model-{epoch}-{i}.pt")
        )
