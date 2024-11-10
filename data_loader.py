import os
from typing import Any, Callable, Iterable, List, Self, Tuple

import torch

from data_preprocessing import filter_wav, getraw, has_content, make_metadata
from meta_dicts import MetaDictType
from util import Config, Logger
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


class AudioDataset(torch.utils.data.Dataset):
    dataset: List[DataLoadItemType]
    dataset_name: str
    experiment: str
    num_tokens: int
    whispercheck: bool = False

    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        self.config = config
        self.experiment = config.options.experiment
        self.dataset_name = config.options.dataset_name

    def __getitem__(
        self: Self,
        index: int,
    ) -> DataGetItemType:
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

    def __len__(
        self: Self,
    ) -> int:
        return self.num_tokens


## Dataset class for the Utterances dataset.
class Utterances(AudioDataset):
    ## Initialize and preprocess the Utterances dataset.
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        super(type(self), self).__init__(config)
        meta_file = os.path.join(config.paths.features, "metadata.pkl")
        if os.path.exists(meta_file):
            os.remove(meta_file)
        make_metadata(config, meta_file)
        logger = Logger()
        metadata = torch.load(meta_file, weights_only=True)
        tmp_dataset = [
            self.load_item(sbmt=sbmt, config=config)
            for sbmt in logger.progress_bar(
                metadata,
                desc=f"Loading {config.options.dataset_name}",
            )
        ]
        self.dataset = [
            item
            for item in logger.progress_bar(
                tmp_dataset,
                desc=f"Refining {config.options.dataset_name}",
            )
            if item_works(item)
        ]
        self.num_tokens = len(self.dataset)

    def load_item(
        self: Self,
        sbmt: Tuple[str, torch.Tensor, str],
        config: Config,
    ) -> DataLoadItemType:
        wav_mono: torch.Tensor = torch.load(
            os.path.join(config.paths.monowavs, sbmt[2]),
            weights_only=True,
        )
        spmel: torch.Tensor = torch.load(
            os.path.join(config.paths.spmels, sbmt[2]),
            weights_only=True,
        )
        f0: torch.Tensor = torch.load(
            os.path.join(config.paths.freqs, sbmt[2]),
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


class FullAudios(AudioDataset):
    ## Initialize and preprocess the Utterances dataset.
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        super(type(self), self).__init__(config)
        self.min_len_seq = config.model.min_len_seq
        self.max_len_seq = config.model.max_len_seq
        self.max_len_pad = config.model.max_len_pad
        self.drop_and_pad = config.dataloader.drop_and_pad

        self.whispercheck = True
        self.load_dataset(config)
        self.num_tokens = len(self.dataset)

    def load_dataset(
        self: Self,
        config: Config,
    ):
        base_wav_dir, spk_dir_list, _ = next(os.walk(config.paths.raw_wavs))
        spk_meta: MetaDictType = getattr(
            __import__("meta_dicts"),
            config.options.dataset_name,
        )
        spk_dirs = [
            spk_dir for spk_dir in sorted(spk_dir_list) if spk_dir in spk_meta.keys()
        ]
        tmp_dataset = []
        Logger().info(
            "Loading data: {} ({} speakers)".format(
                config.options.dataset_name,
                len(spk_dir_list),
            )
        )
        for spk_dir in spk_dirs:
            tmp_dataset.extend(
                self.load_speaker_items(
                    dim_spk_emb=config.model.dim_spk_emb,
                    spk_meta=spk_meta,
                    dir_name=base_wav_dir,
                    spk_dir=spk_dir,
                )
            )
        logger = Logger()

        self.dataset = [
            item
            for item in logger.progress_bar(
                tmp_dataset,
                desc=f"Refining {config.options.dataset_name}",
            )
            if item_works(item)
        ]

    def load_speaker_items(
        self: Self,
        dim_spk_emb: int,
        spk_meta: MetaDictType,
        dir_name: str,
        spk_dir: str,
    ) -> List[DataLoadItemType]:
        from data_preprocessing import F_HI, F_LO, M_HI, M_LO

        _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))
        if spk_meta[spk_dir][1] == "M":
            lo, hi = M_LO, M_HI
        elif spk_meta[spk_dir][1] == "F":
            lo, hi = F_LO, F_HI
        else:
            raise ValueError
        spk_id, _, _ = spk_meta[spk_dir]
        spk_emb = torch.zeros(
            (dim_spk_emb,),
            dtype=torch.float32,
        )
        spk_emb[int(spk_id)] = 1.0
        logger = Logger()
        return [
            self.load_item(
                dir_name=dir_name,
                spk_dir=spk_dir,
                fname=fname,
                spk_meta=spk_meta,
                spk_emb=spk_emb,
                lo=lo,
                hi=hi,
            )
            for fname in logger.progress_bar(
                sorted(file_list), desc=f"Processing: {spk_dir:>4}"
            )
        ]

    def load_item(
        self: Self,
        dir_name: str,
        spk_dir: str,
        fname: str,
        spk_meta: MetaDictType,
        spk_emb: torch.Tensor,
        lo: int,
        hi: int,
    ) -> DataLoadItemType:
        from data_preprocessing import SAMPLE_RATE
        from utils import extract_f0, get_monotonic_wav, get_world_params

        full_fname = os.path.join(dir_name, spk_dir, fname)
        wav = filter_wav(getraw(full_fname))
        f0, sp, ap = get_world_params(wav, SAMPLE_RATE)
        wav_mono = get_monotonic_wav(wav, f0, sp, ap, SAMPLE_RATE)
        spmel = get_spmel(wav, whispercheck=True)
        f0_norm = extract_f0(wav, SAMPLE_RATE, lo, hi)
        assert not is_nan(wav_mono), f"wav has NaNs: {fname}"
        assert not is_nan(spmel), f"spmel has NaNs: {fname}"
        assert not is_nan(f0_norm), f"f0 has NaNs: {fname}"
        return (
            spk_dir,
            spk_emb,
            (wav_mono, spmel, f0_norm),
            fname,
        )


class Collator(object):
    def __init__(
        self: Self,
        config: Config,
    ) -> None:
        self.min_len_seq = config.model.min_len_seq
        self.max_len_seq = config.model.max_len_seq
        self.max_len_pad = config.model.max_len_pad
        self.drop_and_pad = config.dataloader.drop_and_pad

    def __internal_collate(
        self: Self,
        token: DataGetItemType,
    ) -> CollaterInternalItemType:
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

    def __call__(
        self: Self,
        batch: Iterable[DataGetItemType],
    ) -> CollaterItemType:
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
        self: Self,
        num_samples: int,
        n_repeats: int,
        shuffle: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(
        self: Self,
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

    def __iter__(
        self: Self,
    ):
        return iter(self.gen_sample_array())

    def __len__(
        self: Self,
    ) -> int:
        return len(self.sample_idx_array)


def worker_init_fn(x):
    return torch.random.manual_seed(
        (torch.initial_seed()) % (2**32),
    )


## Build and return a data loader list
def get_loader(
    config: Config,
    singleitem: bool = False,
    full_process: bool = False,
) -> torch.utils.data.DataLoader:
    data_loader: torch.utils.data.DataLoader
    dataset: AudioDataset
    sampler: torch.utils.data.sampler.Sampler
    collator: Callable[[list[Any]], Any] = Collator(config)
    dataset = FullAudios(config) if full_process else Utterances(config)
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


def item_works(
    item: DataLoadItemType,
) -> bool:
    (_, _, (wav_mono, spmel, f0), _) = item
    wa_nonzero = wav_mono != 0.0
    sp_nonzero = spmel != 0.0
    f0_nonzero = f0 != 0.0
    is_nonzero = wa_nonzero.any() and sp_nonzero.any() and f0_nonzero.any()
    if has_content(wav_mono) and has_content(spmel) and has_content(f0) and is_nonzero:
        return True
    return False
