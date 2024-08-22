import os
from typing import List, Tuple

import torch
from data_preprocessing import make_metadata
from util.config import Config
from util.logging import Logger
from utils import any_nans, clip, get_spenv, get_spmel, vtlp

DataLoadItemType = Tuple[
    str,  # spk_dir
    torch.Tensor,  # spk_emb
    Tuple[
        torch.Tensor,  # wav_mono
        torch.Tensor,  # spmel
        torch.Tensor,  # f0
    ],
    str,  # filepath
]


## Dataset class for the Utterances dataset.
class Utterances(torch.utils.data.Dataset):
    dataset: List[DataLoadItemType]
    dataset_name: str
    experiment: str
    f0_dir: str
    feat_dir: str
    model_type: str
    num_tokens: int
    spmel_dir: str
    wav_dir: str

    ## Initialize and preprocess the Utterances dataset.
    def __init__(self, config: Config) -> None:
        self.feat_dir = config.paths.features
        self.wav_dir = config.paths.monowavs
        self.spmel_dir = config.paths.spmels
        self.f0_dir = config.paths.freqs
        self.experiment = config.options.experiment
        self.dataset_name = config.options.dataset_name
        self.model_type = "G"
        meta_file = os.path.join(self.feat_dir, "metadata.pkl")
        if os.path.exists(meta_file):
            os.remove(meta_file)
        make_metadata(config, meta_file)
        metadata = torch.load(meta_file, weights_only=True)
        Logger().info("Loading data")
        self.dataset = [self.load_item(sbmt=sbmt) for sbmt in metadata]
        self.num_tokens = len(self.dataset)

    def load_item(
        self,
        sbmt: Tuple[str, torch.Tensor, str],
    ) -> DataLoadItemType:
        wav_mono: torch.Tensor = torch.load(
            os.path.join(self.wav_dir, sbmt[2]),
            weights_only=True,
        )
        spmel: torch.Tensor = torch.load(
            os.path.join(self.spmel_dir, sbmt[2]),
            weights_only=True,
        )
        f0: torch.Tensor = torch.load(
            os.path.join(self.f0_dir, sbmt[2]),
            weights_only=True,
        )
        assert not wav_mono.isnan().any().item(), f"wav has NaNs: {sbmt[2]}"
        assert not spmel.isnan().any().item(), f"spmel has NaNs: {sbmt[2]}"
        assert not f0.isnan().any().item(), f"f0 has NaNs: {sbmt[2]}"
        return (
            sbmt[0],
            sbmt[1],
            (wav_mono, spmel, f0),
            sbmt[2],
        )

    def __getitem__(self, index: int):
        list_uttrs = self.dataset[index]
        spk_id_org: str = list_uttrs[0]
        emb_org: torch.Tensor = list_uttrs[1]
        wav_mono: torch.Tensor
        spmel: torch.Tensor
        f0: torch.Tensor
        wav_mono, spmel, f0 = list_uttrs[2]
        dysarthric: str = list_uttrs[3]
        alpha: float = 0.2 * torch.rand(1).item() + 0.9
        perturbed_wav_mono: torch.Tensor = vtlp(wav_mono, 16000, alpha)
        spenv: torch.Tensor = get_spenv(perturbed_wav_mono)
        spmel_mono: torch.Tensor = get_spmel(perturbed_wav_mono)
        if __debug__ and any_nans(
            [
                perturbed_wav_mono,
                spmel,
                spenv,
                spmel_mono,
                f0,
                emb_org,
            ]
        ):
            Logger().log_if_nan(perturbed_wav_mono)
            Logger().log_if_nan(spmel)
            Logger().log_if_nan(spenv)
            Logger().log_if_nan(spmel_mono)
            Logger().log_if_nan(f0)
            Logger().log_if_nan(emb_org)
            Logger().error(f"Dataset: {list_uttrs[0]}, {list_uttrs[3]}")
        return (
            list_uttrs[3],  # Filename
            dysarthric,  # Single char string D=Dysarthric, C=Control
            perturbed_wav_mono,  # Monotonic wavform with VTLP
            spk_id_org,  # speaker ID string
            spmel,  # MelSpectrogram
            spenv,  # rhythm_input
            spmel_mono,  # content_input
            f0,  # pitch_input
            emb_org,  # timbre_input
        )

    def __len__(self):
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
                fname,  # Filename
                dysarthric,  # Single char string Dysarthric
                perturbed_wav_mono,  # Monotonic waveform with VTLP
                spk_id_org,  # speaker ID string
                melspec,  # MelSpectrogram
                rhythm_input,  # spenv
                content_input,  # spmel_mono
                pitch_input,  # f0
                timbre_input,  # emb_org
            ) = token
            len_crop = torch.randint(
                low=self.min_len_seq, high=self.max_len_seq + 1, size=(1,)
            )
            left = torch.randint(low=0, high=len(melspec) - len_crop, size=(1,))
            spmel_gt = melspec[left : left + len_crop, :]  # [Lc, F]
            rhythm_input = rhythm_input[left : left + len_crop, :]  # [Lc, F]
            content_input = content_input[left : left + len_crop, :]  # [Lc, F]
            pitch_input = pitch_input[left : left + len_crop]  # [Lc, ]

            spmel_gt = torch.nn.functional.pad(
                clip(spmel_gt, 0, 1),
                ((0, 0, 0, self.max_len_pad - spmel_gt.shape[0])),
                "constant",
            )
            rhythm_input = torch.nn.functional.pad(
                clip(rhythm_input, 0, 1),
                ((0, 0, 0, self.max_len_pad - rhythm_input.shape[0])),
                "constant",
            )
            content_input = torch.nn.functional.pad(
                clip(content_input, 0, 1),
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
                    fname,
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
            fname,
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
            fname,
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


def get_loader(
    config: Config,
    singleitem: bool = False,
) -> torch.utils.data.DataLoader:
    """Build and return a data loader list."""
    dataset = Utterances(config)
    collator = Collator(config)
    sampler: torch.utils.data.sampler.Sampler
    if singleitem:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = MultiSampler(
            len(dataset),
            config.dataloader.samplier,
            shuffle=config.dataloader.shuffle,
        )
    batch = 1 if singleitem else config.batch_size
    workers = 0 if singleitem else config.num_workers
    prefetch = None if config.num_workers == 0 or singleitem else config.num_workers
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collator,
    )
    return data_loader
