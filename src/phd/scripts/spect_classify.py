# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

import random
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torchaudio
import torchvision
from matplotlib import colormaps
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..data import SampleInfo, parser, processor
from ..util import TensorBoard, compute, config, logger
from ..util.file import newpath, path, walkfiles, rm_rf, exists
from ..util.tensor import Tensor, TensorPair, pad_to

TESTING = False

out_path: str
tb: TensorBoard


def _init_worker(x: int) -> None:
    return ((torch.initial_seed()) % (2**32),)  # type: ignore


def dys_tensor(dysarthric: bool) -> list[float]:
    return [0.0, 1.0] if dysarthric else [1.0, 0.0]


def dys_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 1]


def cln_probs(dystensor: Tensor) -> Tensor:
    return dystensor[..., 0]


def is_dys(dystensor: Tensor) -> Tensor:
    return cln_probs(dystensor) < dys_probs(dystensor)


def norm(x: Tensor) -> Tensor:
    x -= x.min()
    x /= x.max()
    return x


class Colourise(torch.nn.Module):
    cmap = colormaps.get_cmap("jet")

    def forward(self, x: Tensor) -> Tensor:
        mapped = torch.tensor(self.cmap(x.cpu().numpy()))
        final = mapped.transpose(-1, -3).transpose(-1, -2)
        final = final[..., :3, :, :]
        return final.to(x.device, dtype=torch.float32)

    def reverse(self, x: Tensor) -> Tensor:
        return x.to(x.device, dtype=torch.float32)


colourise = Colourise()


class SpectDataset(torch.utils.data.Dataset[TensorPair]):
    dataset: list[TensorPair] = []
    length: int = 0
    n_fft: int = 512
    n_mels: int = 64
    trans_spec: torchaudio.transforms.Spectrogram
    trans_mel: torchaudio.transforms.MelScale
    trans_invspec: torchaudio.transforms.InverseSpectrogram
    trans_invmel: torchaudio.transforms.InverseMelScale
    sr: int = config.audio.sample_rate
    win_length: int | None = None
    hop_length: int | None = None
    pad_mode: str = "reflect"
    mel_scale: str = "htk"
    pad: int = 0
    normalized: bool = False
    center: bool = True
    onesided: bool = True
    norm: str | None = None
    f_min: float = 0.0
    f_max: float | None = None
    dys_procs_file: str = path(config.paths.artefacts, "dys_procs.pt")
    cln_procs_file: str = path(config.paths.artefacts, "cln_procs.pt")
    testonly: bool = False

    def __getitem__(self, index) -> TensorPair:
        return self.dataset[index]

    def __len__(self) -> int:
        return self.length

    def __init__(self, *args, data: list[TensorPair] | None = None, **kwargs) -> None:
        super().__init__()
        if data is not None:
            self.dataset = data
            self.length = len(self.dataset)
            return
        self.trans_spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            normalized=self.normalized,
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=self.onesided,
            power=None,
        ).to("cpu")
        self.trans_invspec = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            normalized=self.normalized,
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=self.onesided,
        ).to("cpu")
        self.trans_mel = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_fft // 2 + 1,
            norm=self.norm,
            mel_scale=self.mel_scale,
        ).to("cpu")
        self.trans_invmel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sr,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm,
            mel_scale=self.mel_scale,
        ).to("cpu")

    def make_spec(self, audio: Tensor) -> TensorPair:
        windows = audio.unfold(
            dimension=0,
            size=config.audio.sample_rate - 1,
            step=int(config.audio.sample_rate / 0.5),
        ).to("cpu")
        spmel = self.trans_spec(windows)
        mags = spmel.abs()
        phases = spmel.angle()
        mel = self.trans_mel(mags).to(audio.device)
        return mel, phases

    def inv_spec(self, mags: Tensor, phases: Tensor) -> Tensor:
        decol = colourise.reverse(mags)
        magspec = self.trans_invmel(decol)
        spec = torch.polar(magspec, phases)
        return self.trans_invspec(spec)

    def export_imgs(self, specs: Tensor, angles: Tensor, name: str) -> None:
        for i, (spec, phase) in enumerate(zip(specs, angles)):
            torchvision.utils.save_image(spec, path(out_path, f"{name}-{i}.png"))
            wav = self.inv_spec(spec, phase)
            torchaudio.save(path(out_path, f"{name}-{i}.wav"), wav, self.sr)
        torch.save(specs, path(out_path, f"{name}.pt"))
        return

    def neg_one_norm(self, x: Tensor) -> Tensor:
        x /= x.abs().max()
        return x

    def zero_one_norm(self, x: Tensor) -> Tensor:
        x -= x.min()
        x /= x.max()
        return x

    def crop_ool(self, aud: Tensor) -> Tensor:
        raw = self.neg_one_norm(aud)
        _, _, nopop = processor.kill_pop(audio=raw) if parser.is_uaspeech() else raw
        clean = processor.run_clean(audio=nopop, keep=True)
        rawpad, cleanpad = pad_to(raw.squeeze(), clean.squeeze())
        outaud = rawpad[cleanpad != 0.0]
        return outaud

    def loadfn(self, fname: str) -> Tensor:
        fullpath = path(config.paths.raw_wavs, parser.fullwav(fname))
        wav = processor.getraw(full_fname=fullpath)
        ool = processor.crop_ool(wav)
        logger.trace_tensor(ool)
        if ool.numel() == 0:
            return ool
        ool /= ool.abs().max()
        return ool

    def preprocess(self) -> None:
        if exists(self.dys_procs_file) and exists(self.cln_procs_file):
            return
        rm_rf(out_path)
        fnames = walkfiles(config.paths.raw_wavs, fullpaths=False)
        finfos = [(f, SampleInfo(f)) for f in fnames]
        dys_wav = [f for f, i in finfos if i.dysarthric]
        cln_wav = [f for f, i in finfos if (not i.dysarthric)]
        sublen = min(len(dys_wav), len(cln_wav))
        random.seed(42069)
        dys_wav = random.sample(dys_wav, sublen)
        cln_wav = random.sample(cln_wav, sublen)
        logger.debug(f"Dysarthric Files: {len(dys_wav)}")
        logger.debug(f"Clean      Files: {len(cln_wav)}")
        dys_lst = [self.loadfn(f) for f in logger.progress_bar(dys_wav)]
        cln_lst = [self.loadfn(f) for f in logger.progress_bar(cln_wav)]
        dys_ten = torch.cat([t for t in dys_lst if processor.is_valid(t)])
        cln_ten = torch.cat([t for t in cln_lst if processor.is_valid(t)])
        logger.trace_tensor(dys_ten, "DEBUG")
        logger.trace_tensor(cln_ten, "DEBUG")
        dys_mels, dys_phases = self.make_spec(dys_ten)
        cln_mels, cln_phases = self.make_spec(cln_ten)
        dset_mins = [t.abs().min() for t in dys_mels] + [
            t.abs().min() for t in cln_mels
        ]
        dset_min = sum(dset_mins) / len(dset_mins)
        dys_mels -= dset_min
        cln_mels -= dset_min
        dset_maxs = [t.abs().max() for t in dys_mels] + [
            t.abs().max() for t in cln_mels
        ]
        dset_max = sum(dset_maxs) / len(dset_maxs)
        dys_mels /= dset_max
        cln_mels /= dset_max
        logger.debug("All Spectrograms")
        logger.trace_tensor(dys_mels, "DEBUG")
        logger.trace_tensor(cln_mels, "DEBUG")
        logger.trace_tensor(dys_phases, "DEBUG")
        logger.trace_tensor(cln_phases, "DEBUG")
        n_items = min(dys_mels.size(dim=0), cln_mels.size(dim=0))
        dys_mels = dys_mels[:n_items, ...]
        cln_mels = cln_mels[:n_items, ...]
        dys_phases = dys_phases[:n_items, ...]
        cln_phases = cln_phases[:n_items, ...]
        logger.debug("Restricted Spectrograms")
        logger.trace_tensor(dys_phases, "DEBUG")
        logger.trace_tensor(cln_phases, "DEBUG")
        logger.trace_tensor(dys_mels, "DEBUG")
        logger.trace_tensor(cln_mels, "DEBUG")
        dys_procs = colourise(dys_mels)
        cln_procs = colourise(cln_mels)
        logger.trace_tensor(dys_procs, "DEBUG")
        logger.trace_tensor(cln_procs, "DEBUG")
        self.export_imgs(dys_procs, dys_phases, "dys_mels")
        self.export_imgs(cln_procs, cln_phases, "cln_mels")
        torch.save(dys_procs, self.dys_procs_file)
        torch.save(cln_procs, self.cln_procs_file)
        torch.save(dys_phases, path(config.paths.artefacts, "dys_phases.pt"))
        torch.save(cln_phases, path(config.paths.artefacts, "cln_phases.pt"))

    def load(self) -> None:
        dys_procs: list[Tensor] = torch.load(self.dys_procs_file)
        cln_procs: list[Tensor] = torch.load(self.cln_procs_file)
        self.dataset = []
        self.dataset += [(torch.tensor(dys_tensor(True)), i) for i in dys_procs]
        self.dataset += [(torch.tensor(dys_tensor(False)), i) for i in cln_procs]
        self.length = len(self.dataset)


class SpectClassifier(torchvision.models.AlexNet):
    def __init__(self) -> None:
        super().__init__(num_classes=2)
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        state = weights.get_state_dict(progress=True, check_hash=True)
        state.popitem("classifier.6.weight")
        state.popitem("classifier.6.bias")
        self.load_state_dict(state, strict=False)
        self.classifier[-1].zero_grad()


def evaluate(model: SpectClassifier, testdata: list[TensorPair], step: int) -> None:
    model.eval()
    logger.info(f"Evaluating step: {step}")
    predictions = torch.stack(
        [model(data) for _, data in testdata],
    )
    labels = torch.stack([val for val, _ in testdata])
    tb.add_pr_curve("pr_curve", labels, predictions, step)
    logger.debug("Calculating hit-rate")
    testvals = torch.tensor([val for val, _ in testdata])
    logger.trace_var(testvals)
    logger.trace_var(predictions)
    label_bools = torch.tensor([is_dys(val) for val, _ in testdata])
    preds_bools = torch.tensor([is_dys(val) for val in predictions])
    logger.trace_var(label_bools)
    logger.trace_var(preds_bools)
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


def eval_only() -> None:
    dataset = SpectDataset()
    dataset.preprocess()
    dataset.load()
    model = SpectClassifier().to(compute.device())
    model_name = "SpectClassifier-2025-09-21-15-38-16"
    checkpoint, optim_state = torch.load(
        f"artefacts/spect_classify_runs/{model_name}/{model_name}-200.pt",
        weights_only=False,
    )
    model.load_state_dict(checkpoint.state_dict())
    model.eval()
    sampler = SequentialSampler(data_source=dataset)
    loader: DataLoader[TensorPair] = DataLoader(
        dataset=dataset,
        batch_size=config.dataloader.batch_size,
        sampler=sampler,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=None,
    )

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    preds: list[bool] = []
    trues: list[bool] = []
    for i, (truths, specs) in enumerate(logger.progress_bar(loader)):
        truths = truths.to(compute.device())
        specs = specs.to(compute.device())
        out_data = model(specs)
        all_preds = is_dys(out_data)
        all_truths = is_dys(truths)
        for truth, pred in zip(all_truths, all_preds):
            preds.append(pred.item())
            trues.append(truth.item())
            if pred and truth:
                true_pos += 1
            if pred and (not truth):
                false_pos += 1
            if (not pred) and (not truth):
                true_neg += 1
            if (not pred) and truth:
                false_neg += 1
    logger.trace_var(true_pos, "DEBUG")
    logger.trace_var(true_neg, "DEBUG")
    logger.trace_var(false_pos, "DEBUG")
    logger.trace_var(false_neg, "DEBUG")
    len_trues = true_pos + false_neg
    len_false = true_neg + false_pos
    logger.debug(f"True  pos rate: {100*(true_pos/len_trues)}%")
    logger.debug(f"False neg rate: {100*(false_neg/len_trues)}%")
    logger.debug(f"True  neg rate: {100*(true_neg/len_false)}%")
    logger.debug(f"False pos rate: {100*(false_pos/len_false)}%")
    conf_matrix = confusion_matrix(y_true=trues, y_pred=preds)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)
    plt.savefig(path(out_path, f"model-evaluation-{model_name}.pdf"))
    logger.debug(f"Correct   rate: {100*((true_neg+true_pos)/len(dataset))}%")
    logger.debug(f"Incorrect rate: {100*((false_neg+false_pos)/len(dataset))}%")
    exit(1)
    return


def spect_classify() -> None:
    import inspect

    global tb, out_path
    experiment_path = newpath(config.paths.artefacts, inspect.stack()[0][3])
    out_path = newpath(experiment_path, str(parser.dataset_type()))
    logger.set_file()
    log_div = 1000
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()
    eval_only()
    dataset = SpectDataset()
    dataset.preprocess()
    dataset.load()
    model = SpectClassifier()
    logger.trace_var(model)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    n_test: int = dataset.length // 10
    list_test: list[TensorPair] = []
    for i in range(n_test // 2):
        list_test.append(dataset.dataset.pop(0))
        list_test.append(dataset.dataset.pop(-1))
    dset_train = SpectDataset(data=dataset.dataset)
    dset_test = SpectDataset(data=list_test)
    train_sampler = RandomSampler(
        data_source=dset_train,
        replacement=False,
        generator=torch.Generator(device=compute.device()),
        num_samples=len(dset_train),
    )
    test_sampler = SequentialSampler(data_source=dset_test)
    spect_train: DataLoader[TensorPair] = DataLoader(
        dataset=dset_train,
        batch_size=config.dataloader.batch_size,
        sampler=train_sampler,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=None,
    )
    spect_test: DataLoader[TensorPair] = DataLoader(
        dataset=dset_test,
        batch_size=config.dataloader.batch_size,
        sampler=test_sampler,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        pin_memory=False,
        worker_init_fn=None,
    )
    testdata = [
        (dyst, spec.unsqueeze(0).to(compute.device()))
        for dysts, specs in spect_test
        for dyst, spec in zip(dysts, specs)
    ]
    logger.debug(f"Test data length: {len(testdata)}")
    logger.debug(f"Train Sampler length: {len(train_sampler)}")
    logger.debug(f"Train Dataset length: {len(dset_train)}")
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tb = TensorBoard()
    model.train()
    for epoch in range(10000):
        running_loss = 0.0
        for i, (truths, raw_specs) in enumerate(spect_train):
            truths = truths.to(compute.device())
            specs = raw_specs.to(compute.device())
            out_data = model(specs)
            loss = loss_fn(out_data, truths)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            step = (epoch * (len(spect_train))) + i
            tb.add_scalar(name="loss", item=loss.item(), step=step)
            if i % log_div == 0 and i != 0:
                ave_loss = running_loss / log_div
                logger.info(f"[{step:8d}: {epoch:3d}, {i:7d}] loss: {ave_loss}")
                running_loss = 0.0
        if epoch % 100 == 0:
            evaluate(model, testdata, step)
            torch.save(
                (model, optim),
                path(out_path, f"SpectClassifier-{start_time}-{epoch}.pt"),
            )
