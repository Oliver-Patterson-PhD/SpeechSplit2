__all__ = [
    "get_loader",
    "MyDataset",
]

import torch

from util import Compute, Config, Logger
from util.file import exists, path
from util.tensor import Tensor, TensorTriple

from .dataset import DatasetParser
from .utils import AudioProcs


class MyDataset(torch.utils.data.Dataset):
    dataset_name: str
    DataItem = tuple[
        str,  # speaker
        Tensor,  # spk_emb
        TensorTriple,  # wav_mono, spmel, f0
        str,  # filepath
    ]
    DataLoadType = tuple[str, str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    dataset: list[DataItem]
    num_tokens: int
    sample_rate: int
    max_len_seq: int
    path_freqs: str
    path_monowavs: str
    path_spmels: str
    full_info: bool
    map_device: torch.device

    def __init__(self, config: Config) -> None:
        self.dataset_name = config.options.dataset_name
        self.sample_rate = config.audio.sample_rate
        self.max_len_seq = config.model.max_len_seq
        self.dim_spk_emb = config.model.dim_spk_emb
        self.path_freqs = config.paths.freqs
        self.path_monowavs = config.paths.monowavs
        self.path_fullwavs = config.paths.fullwavs
        self.path_spmels = config.paths.spmels
        self.full_info = not config.options.train
        self.myproc = AudioProcs(config)
        self.parser = DatasetParser(config)
        self.map_device = torch.device("cpu")
        self.dataset = [
            (
                speaker,
                self.speaker_id_mask(speaker),
                self.load_from_meta(self.parser.get_wavfile(speaker, uttr)),
                self.parser.get_wavfile(speaker, uttr),
            )
            for speaker in Logger().progress_bar(
                self.parser.speakers(), desc="speakers loaded"
            )
            for uttr in self.parser.get_utterances(speaker)
            if self.uttr_exists(speaker, uttr)
        ]
        self.num_tokens = len(self.dataset)

    def speaker_id_mask(self, speaker: str) -> Tensor:
        return torch.zeros((self.dim_spk_emb,), dtype=torch.float32).index_fill(
            0, torch.tensor([self.parser.get_speaker_id(speaker)]), 1
        )

    def pinnable(self) -> bool:
        return Compute().could_be_gpu() and self.map_device == torch.device("cpu")

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, index: int) -> DataLoadType:
        spk_dir, spk_emb, (wav_mono, spmel, f0), fname = self.dataset[index]
        p_mono: Tensor
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

    def uttr_exists(self, speaker: str, uttr: str) -> bool:
        filepath = self.parser.get_wavfile(speaker, uttr)
        return (
            exists(path(self.path_fullwavs, filepath))
            and exists(path(self.path_monowavs, filepath))
            and exists(path(self.path_spmels, filepath))
            and exists(path(self.path_freqs, filepath))
        )

    def load_from_meta(self, filepath: str) -> TensorTriple:
        wav_mono: Tensor
        spmel: Tensor
        f0: Tensor
        if self.full_info:
            wav_mono = torch.load(
                path(self.path_fullwavs, filepath),
                weights_only=True,
                map_location=self.map_device,
            )
        else:
            wav_mono = torch.load(
                path(self.path_monowavs, filepath),
                weights_only=True,
                map_location=self.map_device,
            )
        spmel = torch.load(
            path(self.path_spmels, filepath),
            weights_only=True,
            map_location=self.map_device,
        )
        f0 = torch.load(
            path(self.path_freqs, filepath),
            weights_only=True,
            map_location=self.map_device,
        )
        o_wav_mono = self.check(wav_mono.float(), f"wav invalid: {filepath}")
        o_spmel = self.check(spmel.float(), f"spmel invalid: {filepath}")
        o_f0 = self.check(f0.float(), f"f0 invalid: {filepath}")
        return (o_wav_mono, o_spmel, o_f0)

    def check(self, x: Tensor, msg: str) -> Tensor:
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
    config: Config, sequential: bool = False, singleitem: bool = False
) -> torch.utils.data.DataLoader:
    dataset: torch.utils.data.Dataset
    sampler: torch.utils.data.sampler.Sampler
    dataset = MyDataset(config)
    device = Compute().device()
    logger = Logger()
    logger.debug(f"Initialising DataLoader for {config.options.dataset_name}")
    batch_size = config.dataloader.batch_size
    samplier = config.dataloader.samplier
    num_workers = config.dataloader.num_workers
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset,
            replacement=True,
            generator=torch.Generator(device=device),
            num_samples=(
                (len(dataset) * samplier)
                if singleitem
                else (batch_size * len(dataset) * samplier)
            ),
        )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1 if singleitem else batch_size,
        sampler=sampler,
        num_workers=0 if singleitem else num_workers,
        prefetch_factor=None if singleitem else num_workers,
        drop_last=False,
        pin_memory=dataset.pinnable() and False,
        worker_init_fn=worker_init_fn,
    )
    logger.debug("Created DataLoader")
    return data_loader
