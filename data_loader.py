import os
from typing import List, Tuple

import torch
from data_preprocessing import DataPreProcessType
from util.config import Config
from util.logging import Logger
from utils import clip, get_spenv, get_spmel, vtlp

logger = Logger()
DataLoadItemType = Tuple[str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]
DataLoadType = List[DataLoadItemType]


class Utterances(torch.utils.data.Dataset):
    """Dataset class for the Utterances dataset."""

    dataset: list
    dataset_name: str
    experiment: str
    f0_dir: str
    feat_dir: str
    model_type: str
    num_tokens: int
    spmel_dir: str
    wav_dir: str

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize and preprocess the Utterances dataset."""
        self.feat_dir = config.paths.features
        self.wav_dir = config.paths.monowavs
        self.spmel_dir = config.paths.spmels
        self.f0_dir = config.paths.freqs
        self.experiment = config.options.experiment
        self.dataset_name = config.options.dataset_name
        self.model_type = "G"
        logger.info("Loading data...")

        metadata: DataPreProcessType = torch.load(
            os.path.join(self.feat_dir, "dataset.pkl"),
            weights_only=True,
        )

        self.dataset = self.load_data(metadata)
        self.num_tokens = len(self.dataset)

    def load_data(
        self,
        metadata,
    ) -> DataLoadType:
        dataset: DataLoadType = []
        for sbmt in metadata:
            # load speaker id and embedding
            # load features
            wav_mono = torch.load(
                os.path.join(self.wav_dir, sbmt[2]),
                weights_only=True,
            )
            spmel = torch.load(
                os.path.join(self.spmel_dir, sbmt[2]),
                weights_only=True,
            )
            f0 = torch.load(
                os.path.join(self.f0_dir, sbmt[2]),
                weights_only=True,
            )
            dataset.append(
                (
                    sbmt[0],
                    sbmt[1],
                    (wav_mono, spmel, f0),
                    sbmt[2],
                )
            )
        return dataset

    def __getitem__(
        self,
        index: int,
    ):
        list_uttrs = self.dataset[index]
        spk_id_org = list_uttrs[0]
        emb_org = list_uttrs[1]
        if self.dataset_name == "uaspeech":
            dysarthric = list_uttrs[3]
        else:
            dysarthric = None
        wav_mono, spmel, f0 = list_uttrs[2]
        alpha = 0.2 * torch.rand(1).item() + 0.9
        wav_mono = vtlp(wav_mono, 16000, alpha)
        spenv = get_spenv(wav_mono)
        spmel_mono = get_spmel(wav_mono)
        rhythm_input = spenv
        content_input = spmel_mono
        pitch_input = f0
        timbre_input = emb_org
        # print(f"Collator: dysarthric    {dysarthric}")
        # print(f"Collator: wav_mono      {wav_mono.shape}")
        # print(f"Collator: spk_id_org    {spk_id_org}")
        # print(f"Collator: spmel         {spmel.shape}")
        # print(f"Collator: rhythm_input  {rhythm_input.shape}")
        # print(f"Collator: content_input {content_input.shape}")
        # print(f"Collator: pitch_input   {pitch_input.shape}")
        # print(f"Collator: timbre_input  {timbre_input.shape}")

        return (
            dysarthric,
            wav_mono,
            spk_id_org,
            spmel,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
        )

    def __len__(
        self,
    ):
        """Return the number of spkrs."""
        return self.num_tokens


class Collator(object):
    def __init__(
        self,
        config: Config,
    ):
        self.min_len_seq = config.model.min_len_seq
        self.max_len_seq = config.model.max_len_seq
        self.max_len_pad = config.model.max_len_pad

    def __call__(self, batch):
        new_batch = []
        for token in batch:

            (
                _,
                _,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
            ) = token
            len_crop = torch.randint(
                low=self.min_len_seq, high=self.max_len_seq + 1, size=(1,)
            )
            left = torch.randint(low=0, high=len(spmel_gt) - len_crop, size=(1,))
            spmel_gt = spmel_gt[left : left + len_crop, :]  # [Lc, F]
            rhythm_input = rhythm_input[left : left + len_crop, :]  # [Lc, F]
            content_input = content_input[left : left + len_crop, :]  # [Lc, F]
            pitch_input = pitch_input[left : left + len_crop]  # [Lc, ]

            spmel_gt = clip(spmel_gt, 0, 1)
            rhythm_input = clip(rhythm_input, 0, 1)
            content_input = clip(content_input, 0, 1)

            spmel_gt = torch.nn.functional.pad(
                spmel_gt,
                ((0, 0, 0, self.max_len_pad - spmel_gt.shape[0])),
                "constant",
            )
            rhythm_input = torch.nn.functional.pad(
                rhythm_input,
                ((0, 0, 0, self.max_len_pad - rhythm_input.shape[0])),
                "constant",
            )
            content_input = torch.nn.functional.pad(
                content_input,
                ((0, 0, 0, self.max_len_pad - content_input.shape[0])),
                "constant",
            )
            pitch_input = torch.nn.functional.pad(
                pitch_input[:, None],
                ((0, 0, 0, self.max_len_pad - pitch_input.shape[0])),
                "constant",
                value=-1e10,
            )

            new_batch.append(
                (
                    spk_id_org,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop,
                )
            )

        batch = new_batch
        (
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = zip(*batch)
        spk_id_org = list(spk_id_org)
        spmel_gt = torch.stack(spmel_gt, axis=0).float()
        rhythm_input = torch.stack(rhythm_input, axis=0).float()
        content_input = torch.stack(content_input, axis=0).float()
        pitch_input = torch.stack(pitch_input, axis=0).float()
        timbre_input = torch.stack(timbre_input, axis=0).float()
        len_crop = torch.stack(len_crop, axis=0).double()

        return (
            spk_id_org,
            spmel_gt.to("cpu"),
            rhythm_input.to("cpu"),
            content_input.to("cpu"),
            pitch_input.to("cpu"),
            timbre_input.to("cpu"),
            len_crop.to("cpu"),
        )


class MultiSampler(torch.utils.data.sampler.Sampler):
    """Samples elements more than once in a single pass through the data."""

    def __init__(self, num_samples, n_repeats, shuffle=False) -> None:
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(
        self,
    ) -> torch.Tensor:
        self.sample_idx_array = torch.arange(
            self.num_samples,
            dtype=torch.int64,
        ).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[
                torch.randperm(
                    len(self.sample_idx_array),
                )
            ]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)


def worker_init_fn(x):
    return torch.random.manual_seed(
        (torch.initial_seed()) % (2**32),
    )


def get_loader(config) -> torch.utils.data.DataLoader:
    """Build and return a data loader list."""

    dataset = Utterances(config)
    collator = Collator(config)
    sampler = MultiSampler(
        len(dataset),
        config.samplier,
        shuffle=config.shuffle,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.num_workers if config.num_workers != 0 else None,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collator,
    )
    return data_loader
