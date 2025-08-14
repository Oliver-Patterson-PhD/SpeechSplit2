# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

from datetime import datetime

import torch
import torchaudio
import torchvision
from matplotlib import colormaps
from torchvision.transforms import v2

from ..data import Dataset, SampleInfo, parser, processor, split_loaders
from ..util import TensorBoard, compute, config, logger
from ..util.file import basename, newpath, path, walkfiles
from ..util.tensor import Tensor, TensorPair

TESTING = False

out_path: str
tb: TensorBoard
eps = 1e-10
minval = eps
maxval = 1.0 - eps


def dys_tensor(dysarthric: bool) -> list[float]:
    return [0.0, 1.0] if dysarthric else [1.0, 0.0]


def dys_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 1]


def cln_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 0]


def is_dys(dystensor: Tensor) -> Tensor:
    return cln_probs(dystensor) < dys_probs(dystensor)


class Colourise(torch.nn.Module):
    cmap = colormaps.get_cmap("jet")

    def forward(self, x: Tensor) -> Tensor:
        mapped = torch.tensor(self.cmap(x.cpu().numpy()))
        final = mapped.transpose(-1, -3).transpose(-1, -2)
        return final.to(x.device)


def norm(x: Tensor) -> Tensor:
    x -= x.min()
    x /= x.max()
    return x


class SpectDataset(Dataset[TensorPair]):
    makespec = torchaudio.transforms.MelSpectrogram(
        n_fft=400,
        n_mels=128,
        f_min=40,  # Below this there are artefacts in UASpeech
        normalized=True,
    )
    process = v2.Compose(
        [
            Colourise(),
            v2.ToDtype(torch.float32),
            v2.Lambda(lambda x: x[:3, ...]),
        ]
    )

    def __init__(
        self, testing: bool = TESTING, rawdata: list[TensorPair] | None = None
    ) -> None:
        filenames = (
            walkfiles(config.paths.raw_wavs, fullpaths=False)
            if rawdata is None
            else None
        )
        super().__init__(testing=testing, filenames=filenames, rawdata=rawdata)
        if rawdata is None:
            dyslist = [(meta, item) for meta, item in self.dataset if is_dys(meta)]
            clnlist = [(meta, item) for meta, item in self.dataset if not is_dys(meta)]
            logger.debug(f"Dysarthric Samples: {len(dyslist)}")
            logger.debug(f"Clean      Samples: {len(clnlist)}")
            minsamples = min(len(dyslist), len(clnlist))
            self.dataset = dyslist[:minsamples] + clnlist[:minsamples]
            self.length = len(self.dataset)
            logger.debug(f"Total      Samples: {len(self.dataset)}")

    def generate_item(self, fname: str) -> TensorPair | None:
        a, b, c, wav = processor.full_load_parts(
            path(
                config.paths.raw_wavs,
                parser.get_spkdir(parser.speaker(fname)),
                basename(fname) + ".wav",
            )
        )
        if processor.full_load_check(a, b, c, wav) is not None:
            return None
        info = torch.tensor(dys_tensor(SampleInfo(fname).dysarthric))
        colourised = self.process(
            norm(self.makespec(norm(wav)).clamp(min=minval).log10())
        )
        processed = (
            v2.functional.pad(colourised, (0, 0, 128 - colourised.size(-1), 0))
            if colourised.size(-1) > 128
            else v2.functional.crop(colourised, 0, 0, 128, 128)
        )
        return (info, processed.mT)


class SpectClassifier(torchvision.models.AlexNet):
    def __init__(self) -> None:
        super().__init__(num_classes=2)
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        state = weights.get_state_dict(progress=True, check_hash=True)
        state.popitem("classifier.6.weight")
        state.popitem("classifier.6.bias")
        self.load_state_dict(state, strict=False)
        self.classifier[-1].zero_grad()


@torch.no_grad()
def evaluate(model: SpectClassifier, testdata: list[TensorPair], step: int) -> None:
    model.eval()
    logger.info(f"Evaluating step: {step}")

    predictions = torch.stack([model(data) for _, data in testdata]).squeeze(-2)
    labels = torch.stack([val for val, _ in testdata])
    tb.add_pr_curve("pr_curve", labels, predictions, step)

    logger.debug("Calculating hit-rate")
    label_bools = torch.tensor([is_dys(val) for val, _ in testdata])
    preds_bools = torch.tensor([is_dys(val) for val in predictions])
    good_vals = preds_bools == label_bools
    good_ave = (good_vals.sum() / good_vals.size(dim=-1)).item()
    tb.add_scalar("eval_rate", good_ave, step)

    logger.trace("")
    logger.trace(f"Correct prediction rate: {good_ave}")
    logger.trace(f"Dysarthric mean actual vals: {dys_probs(labels).mean()}")
    logger.trace(f"Dysarthric mean predictions: {dys_probs(predictions).mean()}")
    logger.trace(f"Clean mean actual vals: {cln_probs(labels).mean()}")
    logger.trace(f"Clean mean predictions: {cln_probs(predictions).mean()}")
    logger.trace("")

    tb.flush()
    model.train()
    return


def spect_classify() -> None:
    import inspect

    global tb, out_path
    experiment_path = newpath(config.paths.artefacts, inspect.stack()[0][3])
    out_path = newpath(experiment_path, str(parser.dataset_type()))

    logger.set_file()
    log_div = 100

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()

    model = SpectClassifier()
    logger.trace_var(model)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    spect_train, spect_test = split_loaders(SpectDataset(), parallel=False)
    testdata = [
        (dyst, spec.unsqueeze(0).to(compute.device()))
        for dysts, specs in spect_test
        for dyst, spec in zip(dysts, specs)
    ]

    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tb = TensorBoard()

    model.train()
    for epoch in range(10000):
        running_loss = 0.0
        for i, (truths, raw_specs) in enumerate(spect_train):
            truths = truths.to(compute.device())
            specs = raw_specs.to(compute.device())
            step = (epoch * (len(spect_train))) + i
            out_data = model(specs)
            loss = loss_fn(out_data, truths)
            loss.backward()
            running_loss += loss.item()
            tb.add_scalar(name="loss", item=loss.item(), step=step)
            if i % log_div == 0 and i != 0:
                ave_loss = running_loss / log_div
                logger.info(f"[{step:8d}: {epoch:3d}, {i:7d}] loss: {ave_loss}")
                running_loss = 0.0
        if epoch % 10 == 0:
            evaluate(model, testdata, step)
            torch.save(
                (model, optim),
                path(out_path, f"SpectClassifier-{start_time}-{epoch}.pt"),
            )
