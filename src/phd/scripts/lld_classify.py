# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

import inspect
from datetime import datetime
from typing import Generator

import torch
from torchvision.transforms import v2

from ..data import Dataset, SampleInfo, parser, split_loaders
from ..util import TensorBoard, compute, config, logger
from ..util.arff import ArffRowType, loadarff
from ..util.file import basename, newpath, path, walkfiles
from ..util.tensor import Tensor, TensorPair

TESTING = False
log_div = 1000

tb: TensorBoard

in_path = config.paths.raw_wavs
sample_rate = config.audio.sample_rate
smile_path = path(config.paths.proc_data, "OpenSMILE")


def ret_attr[T](row: ArffRowType, item: str, t: type[T]) -> T:
    thing = row.get(item)
    if not isinstance(thing, t):
        raise ValueError(f"Incorrect type for {item}: {type(thing)}, expected {t}")
    return thing


def add_attr(self: object, row: ArffRowType, name: str, item: str, t: type) -> None:
    thing = row.get(item)
    if not isinstance(thing, t):
        raise ValueError(f"Incorrect type for {item}: {type(thing)}, expected {t}")
    setattr(self, name, thing)


def dys_tensor(dysarthric: bool) -> list[float]:
    return [0.0, 1.0] if dysarthric else [1.0, 0.0]


def dys_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 1]


def cln_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 0]


def is_dys(dystensor: Tensor) -> Tensor:
    return cln_probs(dystensor) < dys_probs(dystensor)


N_LLDS = 26


def row_to_data(row: ArffRowType) -> TensorPair:
    return (
        torch.tensor(
            dys_tensor(SampleInfo(ret_attr(row, "name", str)).dysarthric),
            device=compute.device(),
        ),
        torch.tensor(
            [
                ret_attr(row, "frameTime", float),
                ret_attr(row, "Loudness_sma3", float),
                ret_attr(row, "alphaRatio_sma3", float),
                ret_attr(row, "hammarbergIndex_sma3", float),
                ret_attr(row, "slope0-500_sma3", float),
                ret_attr(row, "slope500-1500_sma3", float),
                ret_attr(row, "spectralFlux_sma3", float),
                ret_attr(row, "mfcc1_sma3", float),
                ret_attr(row, "mfcc2_sma3", float),
                ret_attr(row, "mfcc3_sma3", float),
                ret_attr(row, "mfcc4_sma3", float),
                ret_attr(row, "F0semitoneFrom27.5Hz_sma3nz", float),
                ret_attr(row, "jitterLocal_sma3nz", float),
                ret_attr(row, "shimmerLocaldB_sma3nz", float),
                ret_attr(row, "HNRdBACF_sma3nz", float),
                ret_attr(row, "logRelF0-H1-H2_sma3nz", float),
                ret_attr(row, "logRelF0-H1-A3_sma3nz", float),
                ret_attr(row, "F1frequency_sma3nz", float),
                ret_attr(row, "F1bandwidth_sma3nz", float),
                ret_attr(row, "F1amplitudeLogRelF0_sma3nz", float),
                ret_attr(row, "F2frequency_sma3nz", float),
                ret_attr(row, "F2bandwidth_sma3nz", float),
                ret_attr(row, "F2amplitudeLogRelF0_sma3nz", float),
                ret_attr(row, "F3frequency_sma3nz", float),
                ret_attr(row, "F3bandwidth_sma3nz", float),
                ret_attr(row, "F3amplitudeLogRelF0_sma3nz", float),
            ],
            device=compute.device(),
        ),
    )


class LLDDataset(Dataset[TensorPair]):
    def __init__(
        self, testing: bool = TESTING, rawdata: list[TensorPair] | None = None
    ) -> None:
        filenames = (
            walkfiles(path(smile_path, "custom-arff"), fullpaths=True)
            if rawdata is None
            else None
        )
        super().__init__(testing=testing, filenames=filenames, rawdata=rawdata)
        if rawdata is None:
            # dyslist = [(meta, item) for meta, item in self.dataset if is_dys(meta)]
            # clnlist = [(meta, item) for meta, item in self.dataset if not is_dys(meta)]
            # logger.debug(f"Dysarthric Samples: {len(dyslist)}")
            # logger.debug(f"Clean      Samples: {len(clnlist)}")
            # minsamples = min(len(dyslist), len(clnlist))
            # self.dataset = dyslist[:minsamples] + clnlist[:minsamples]
            # self.length = len(self.dataset)
            logger.debug(f"Total      Samples: {len(self.dataset)}")

    def __getitem__(self, index: int) -> TensorPair:
        return self.dataset[index]

    def generator_func(
        self, fnames: list[str], limit: int | None
    ) -> Generator[TensorPair]:
        for fname in logger.progress_bar(fnames, unit=" files"):
            for item in self.generate_items(fname):
                if limit is not None:
                    if limit == 0:
                        return
                    if item is not None:
                        limit -= 1
                yield item

    def generate_items(self, fname: str) -> Generator[TensorPair]:
        data = loadarff(fname)
        if data is None:
            return
        for item in data:
            yield row_to_data(item)


class LLDClassifier(torch.nn.Module):
    n_classes: int = 2
    __view_div: int = 2

    def __init__(self) -> None:
        super().__init__()
        assert N_LLDS % self.__view_div == 0
        self.__i_len = N_LLDS // self.__view_div
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
        assert x.size(dim=-1) == N_LLDS
        x_vdim = (
            (x.size(dim=0), self.__i_len, self.__view_div)
            if x.ndim == 2
            else (self.__i_len, self.__view_div)
        )
        return self.out_layers(
            torch.flatten(self.feat_layers(x.view(*x_vdim)), start_dim=-2)
        )


def evaluate(model: LLDClassifier, valitems: list[TensorPair], step: int) -> None:
    model.eval()
    logger.info(f"Evaluating step: {step}")

    predictions = torch.stack([model(items) for _, items in valitems])
    label_bools = torch.tensor([is_dys(spk) for spk, _ in valitems])
    labels = torch.tensor([dys_tensor(val) for val in label_bools])
    tb.add_pr_curve("pr_curve", labels, predictions, step)

    logger.debug("Calculating hit-rate")
    pred_bools = is_dys(predictions)
    good = pred_bools == label_bools
    good_ave = good.sum() / good.size(dim=-1)
    logger.trace_var(good_ave)
    tb.add_scalar("eval_true", good_ave.item(), step)

    logger.trace_var(label_bools)
    dysprobs = dys_probs(predictions)
    clnprobs = cln_probs(predictions)
    logger.trace_var(dysprobs.min())
    logger.trace_var(clnprobs.min())
    logger.trace_var(dysprobs.max())
    logger.trace_var(clnprobs.max())
    logger.trace_var(dysprobs.mean())
    logger.trace_var(clnprobs.mean())

    tb.flush()
    model.train()


def lld_classify() -> None:
    global tb
    tb = TensorBoard()
    logger.set_file()
    experiment = basename(inspect.stack()[0].filename)
    experiment_dir = newpath(config.paths.artefacts, experiment)
    out_path = newpath(experiment_dir, str(parser.dataset_type()))
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("forkserver", force=True)
    compute.set_gpu()
    compute.set_default()

    logger.info("Building Model")
    model = LLDClassifier()
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    arff_fulldata = LLDDataset()
    logger.info("Augmenting dysarthric speech")
    augment = v2.RandomSolarize(threshold=0.1, p=1.0)
    arff_fulldata.dataset = [
        (meta, augment(item)) if is_dys(meta) else (meta, item)
        for meta, item in arff_fulldata.dataset
    ]
    arff_train, arff_test = split_loaders(arff_fulldata, parallel=False)

    logger.info("Calculating data distribution")
    dys_rate = [parser.dysarthric(spk) for spks, _ in arff_train for spk in spks]
    logger.trace_var(sum(dys_rate))
    logger.trace_var(len(dys_rate))

    logger.info("Loading evaluation data")
    valitems = [(n, i) for nn, ii in arff_test for n, i in zip(nn, ii)]
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logger.info("Starting Training")
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        i = 0
        for meta, items in arff_train:
            i += 1
            step = (epoch * len(arff_train)) + i
            out_data = model(items)
            loss = loss_fn(out_data, meta)
            loss.backward()
            optim.step()
            logger.trace_var(loss)
            optim.step()
            running_loss += loss.item()
            tb.add_scalar(name="loss", item=loss.item(), step=step)
            if i % log_div == 0:
                ave_loss = running_loss / log_div
                logger.info(f"[{step:8d}: {epoch:3d}, {i:7d}] loss: {ave_loss}")
                running_loss = 0.0
        evaluate(model, valitems, step)
        torch.save(
            (model, optim),
            path(out_path, f"arff_model{start_time}-{epoch}.pt"),
        )
