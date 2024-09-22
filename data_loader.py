import os
from typing import Any, Callable, Iterable, List, Tuple

import torch
from data_preprocessing import has_content, make_metadata
from util.config import Config
from util.logging import Logger
from utils import clip, get_spenv, get_spmel, is_nan, vtlp

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

DataGetItemType = Tuple[
    str,
    str,
    torch.Tensor,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

CollaterInternalItemType = Tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

CollaterItemType = Tuple[
    List[str],
    List[str],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
        self.model_type = "SpeechSplit2"
        meta_file = os.path.join(self.feat_dir, "metadata.pkl")
        if os.path.exists(meta_file):
            os.remove(meta_file)
        make_metadata(config, meta_file)
        metadata = torch.load(meta_file, weights_only=True)
        Logger().info(f"Loading data: {config.options.dataset_name}")
        tmp_dataset = [self.load_item(sbmt=sbmt) for sbmt in metadata]
        Logger().debug(f"Refining data: {config.options.dataset_name}")
        self.dataset = [item for item in tmp_dataset if item_works(item)]
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
        assert not is_nan(wav_mono), f"wav has NaNs: {sbmt[2]}"
        assert not is_nan(spmel), f"spmel has NaNs: {sbmt[2]}"
        assert not is_nan(f0), f"f0 has NaNs: {sbmt[2]}"
        return (
            sbmt[0],
            sbmt[1],
            (wav_mono, spmel, f0),
            sbmt[2],
        )

    def __getitem__(self, index: int) -> DataGetItemType:
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
        assert not is_nan(perturbed_wav_mono), f"{list_uttrs[3]} has NaNs"
        assert not is_nan(spmel), f"{list_uttrs[3]} has NaNs"
        assert not is_nan(spenv), f"{list_uttrs[3]} has NaNs"
        assert not is_nan(spmel_mono), f"{list_uttrs[3]} has NaNs"
        assert not is_nan(f0), f"{list_uttrs[3]} has NaNs"
        assert not is_nan(emb_org), f"{list_uttrs[3]} has NaNs"
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
    ) -> None:
        self.min_len_seq = config.model.min_len_seq
        self.max_len_seq = config.model.max_len_seq
        self.max_len_pad = config.model.max_len_pad
        self.drop_and_pad = config.dataloader.drop_and_pad

    def __internal_collate(self, token: DataGetItemType) -> CollaterInternalItemType:
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
        if self.drop_and_pad:
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
        else:
            len_crop = torch.tensor([self.max_len_seq])
            spmel_gt = melspec
        return (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        )

    def __call__(self, batch: Iterable[DataGetItemType]) -> CollaterItemType:
        new_batch: List[CollaterInternalItemType] = [
            self.__internal_collate(token) for token in batch
        ]

        secbatch = new_batch
        (
            it_fname,
            it_spk_id_org,
            it_spmel_gt,
            it_rhythm_input,
            it_content_input,
            it_pitch_input,
            it_timbre_input,
            it_len_crop,
        ) = zip(*secbatch)
        out_fname: List[str] = list(it_fname)
        out_spk_id_org: List[str] = list(it_spk_id_org)
        out_spmel_gt: torch.Tensor = torch.stack(it_spmel_gt, dim=0).float()
        out_rhythm_input: torch.Tensor = torch.stack(it_rhythm_input, dim=0).float()
        out_content_input: torch.Tensor = torch.stack(it_content_input, dim=0).float()
        out_pitch_input: torch.Tensor = torch.stack(it_pitch_input, dim=0).float()
        out_timbre_input: torch.Tensor = torch.stack(it_timbre_input, dim=0).float()
        out_len_crop: torch.Tensor = torch.stack(it_len_crop, dim=0).double()

        return (
            out_fname,
            out_spk_id_org,
            out_spmel_gt.to("cpu"),
            out_rhythm_input.to("cpu"),
            out_content_input.to("cpu"),
            out_pitch_input.to("cpu"),
            out_timbre_input.to("cpu"),
            out_len_crop.to("cpu"),
        )


## Samples elements more than once in a single pass through the data
class MultiSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        num_samples: int,
        n_repeats: int,
        shuffle: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self) -> torch.Tensor:
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

    def __len__(self) -> int:
        return len(self.sample_idx_array)


def worker_init_fn(x):
    return torch.random.manual_seed(
        (torch.initial_seed()) % (2**32),
    )


## Build and return a data loader list
def get_loader(
    config: Config,
    singleitem: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = Utterances(config)
    sampler: torch.utils.data.sampler.Sampler
    collator: Callable[[list[Any]], Any] = Collator(config)
    if singleitem:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = MultiSampler(
            len(dataset),
            config.dataloader.samplier,
            shuffle=config.dataloader.shuffle,
        )
    batch = 1
    workers = 0 if singleitem else config.num_workers
    prefetch = None if workers == 0 else workers
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


def item_works(item: DataLoadItemType) -> bool:
    (_, _, (wav_mono, spmel, f0), _) = item
    wa_nonzero = wav_mono != 0.0
    sp_nonzero = spmel != 0.0
    f0_nonzero = f0 != 0.0
    is_nonzero: bool = (
        wa_nonzero.any() and sp_nonzero.any() and f0_nonzero.any()
    ).item() != 0
    if has_content(wav_mono) and has_content(spmel) and has_content(f0) and is_nonzero:
        return True
    return False
