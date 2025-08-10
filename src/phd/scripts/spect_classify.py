# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

from datetime import datetime

import torch
import torchvision
from torchvision.transforms import v2

from ..data import (AudioProcs, Dataset, DatasetParser, SampleInfo,
                    split_loaders)
from ..util import TensorBoard, compute, config, logger
from ..util.file import basename, exists, newpath, path, walkfiles
from ..util.tensor import Tensor

parser = DatasetParser()
processor = AudioProcs()

TESTING = False

tb: TensorBoard


def dys_tensor(dysarthric: bool) -> list[float]:
    return [0.0, 1.0] if dysarthric else [1.0, 0.0]


def dys_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 1]


def cln_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 0]


def is_dys(dystensor: Tensor) -> Tensor:
    return cln_probs(dystensor) < dys_probs(dystensor)


def dys_item(fname: str) -> Tensor:
    return torch.tensor(dys_tensor(SampleInfo(fname.rpartition("_")[0]).dysarthric))


class SpectDataset(Dataset[tuple[Tensor, Tensor]]):
    dataset_name = config.options.dataset_name
    dataset: list[tuple[Tensor, Tensor]]
    small_cache_file: str = path(config.paths.features, "spects-small.pkl")
    large_cache_file: str = path(config.paths.features, "spects-large.pkl")
    small_size: int = 1000
    cache_file: str

    def __init__(self, rawdata: list[tuple[Tensor, Tensor]] | None = None) -> None:
        self.cache_file = self.small_cache_file if TESTING else self.large_cache_file
        if rawdata is not None:
            logger.debug("SpectDataset creating from raw list")
            self.dataset = rawdata
            logger.debug("SpectDataset created from raw list")
        elif exists(self.cache_file):
            logger.debug("SpectDataset Loading")
            self.dataset = torch.load(self.cache_file)
            logger.debug("SpectDataset Loaded")
        else:
            logger.info("Generating Spectrogram Dataset")
            filenames = walkfiles(config.paths.spmels, fullpaths=True)
            if len(filenames) == 0:
                raise Exception("ERROR: No Spectrogram files in dataset")
            self.dataset = [
                (dys_item(file), torch.load(file))
                for file in logger.progress_bar(filenames, unit="loaded")
            ]
            logger.info("Saving Spectrogram Dataset")
            torch.save(self.dataset, self.large_cache_file)
            logger.info("Saving Small Spectrogram Dataset")
            torch.save(self.dataset[0 : self.small_size - 1], self.small_cache_file)
        self.length = len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.dataset[index]

    def __len__(self) -> int:
        return self.length

    def dump(self, index: int) -> tuple[Tensor, Tensor]:
        return self.__getitem__(index)


class SpectClassifier(torchvision.models.AlexNet):
    def __init__(self) -> None:
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        self.classifier[-1] = torch.nn.Linear(4096, 2)
        self.features.train(False)
        self.avgpool.train(False)

    def train(self, mode: bool = True) -> None:
        self.classifier.train(mode)


@torch.no_grad()
def evaluate(
    model: SpectClassifier,
    testdata: list[tuple[Tensor, Tensor]],
    step: int,
) -> None:
    model.eval()
    logger.info(f"Evaluating step: {step}")

    predictions = torch.stack([model(data) for _, data in testdata])
    labels = torch.tensor([val for val, _ in testdata])
    tb.add_pr_curve("pr_curve", labels, predictions, step)

    logger.debug("Calculating hit-rate")
    label_bools = torch.tensor([is_dys(val) for val, _ in testdata])
    preds_bools = torch.tensor([is_dys(val) for val in predictions])
    good_vals = preds_bools == label_bools
    good_ave = (good_vals.sum() / good_vals.size(dim=-1)).item()
    tb.add_scalar("eval_rate", good_ave, step)
    logger.trace_var(good_ave)
    logger.trace_var(dys_probs(predictions))
    logger.trace_var(cln_probs(predictions))

    tb.flush()
    model.train()
    return


def spect_classify() -> None:
    global tb
    tb = TensorBoard()
    logger.set_file()
    log_div = 100

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()

    model = SpectClassifier()
    logger.trace_var(model)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    spect_train, spect_test = split_loaders(SpectDataset(), parallel=False)
    testdata = [
        (dyst, spec) for dysts, specs in spect_test for dyst, spec in zip(dysts, specs)
    ]
    loss_fn = torch.nn.BCELoss()
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    experiment_path = newpath(config.paths.artefacts, basename(__name__))
    out_path = newpath(experiment_path, str(parser.dataset_type()))
    model.train()
    spect_process = v2.Compose([v2.Resize((224, 224)), v2.RGB()]).to(compute.device())
    for epoch in range(10):
        running_loss = 0.0
        for i, (truths, raw_specs) in enumerate(spect_train):
            truths = truths.to(compute.device())
            raw_specs = raw_specs.to(compute.device())
            logger.trace_tensor(raw_specs)
            specs = spect_process(raw_specs.unsqueeze(1))
            logger.trace_tensor(truths)
            logger.trace_tensor(specs)
            step = (epoch * (len(spect_train))) + i
            out_data = model(specs)
            loss = loss_fn(out_data, truths)
            loss.backward()
            logger.trace_var(loss)
            running_loss += loss.item()
            tb.add_scalar(name="loss", item=loss.item(), step=step)
            if i % log_div == 0:
                ave_loss = running_loss / log_div
                logger.info(f"[{step:8d}: {epoch:3d}, {i:7d}] loss: {ave_loss}")
                running_loss = 0.0
            evaluate(model, testdata, step)
            torch.save(
                (model, optim),
                path(out_path, f"SpectClassifier-{start_time}-{epoch}.pt"),
            )
