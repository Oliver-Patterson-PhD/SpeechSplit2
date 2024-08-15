import os
from typing import List, Tuple

import torch
import torchaudio
from meta_dicts import MetaDictType
from util.config import Config
from util.logging import Logger
from utils import (average_f0s, extract_f0, filter_wav, get_monotonic_wav,
                   get_spmel, get_world_params)

DataPreProcessType = List[Tuple[str, torch.Tensor, str]]


def split_feats(
    fea: torch.Tensor,
    fea_dir: str,
    trunk_len: int,
    spk_dir: str,
    filename: str,
) -> None:
    start_idx = 0
    while start_idx * trunk_len < len(fea):
        this_trunk = start_idx * trunk_len
        next_trunk = (start_idx + 1) * trunk_len
        fea_trunk = fea[this_trunk:next_trunk]
        if len(fea_trunk) < trunk_len:
            if fea_trunk.ndim == 2:
                fea_trunk = torch.nn.functional.pad(
                    fea_trunk,
                    (0, 0, 0, trunk_len - len(fea_trunk)),
                )
            elif fea_trunk.ndim == 1:
                fea_trunk = torch.nn.functional.pad(
                    fea_trunk,
                    (0, trunk_len - len(fea_trunk)),
                )
            else:
                raise ValueError
        torch.save(
            fea_trunk.to(torch.float32),
            "{}/{}/{}_{}.pt".format(
                fea_dir,
                spk_dir,
                os.path.splitext(filename)[0],
                start_idx,
            ),
        )
        start_idx += 1
    return


def process_file(
    filename: str,
    wav: torch.Tensor,
    f0: torch.Tensor,
    sp: torch.Tensor,
    ap: torch.Tensor,
    fs: int,
    lo: int,
    hi: int,
    wav_dir: str,
    f0_dir: str,
    spk_dir: str,
    spmel_dir: str,
) -> None:
    wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)
    spmel = get_spmel(wav)
    f0_norm = extract_f0(wav, fs, lo, hi)
    assert len(spmel) == len(f0_norm), (
        f"melspec and f0 lengths do not match for {filename}",
        f"spmel: {len(spmel)}\n",
        f"f0_rapt: {len(f0_norm)}\n",
    )
    split_feats(
        fea=wav_mono,
        fea_dir=wav_dir,
        trunk_len=49151,
        spk_dir=spk_dir,
        filename=filename,
    )
    split_feats(
        fea=spmel,
        fea_dir=spmel_dir,
        trunk_len=192,
        spk_dir=spk_dir,
        filename=filename,
    )
    split_feats(
        fea=f0_norm,
        fea_dir=f0_dir,
        trunk_len=192,
        spk_dir=spk_dir,
        filename=filename,
    )


def world_list(
    fname: str,
    dir_name: str,
    spk_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_wav: torch.Tensor
    raw_wav, _ = torchaudio.load(
        f"{dir_name}/{spk_dir}/{fname}",
        channels_first=True,
    )[0].squeeze()
    if raw_wav.shape[0] % 256 == 0:
        raw_wav = torch.cat(
            (raw_wav, torch.tensor([[1e-06]], device=raw_wav.device)), dim=0
        )
    wav = filter_wav(raw_wav)
    f0, sp, ap = get_world_params(wav, 16000)
    return (wav, f0, sp, ap)


def make_spect_f0(config: Config) -> None:
    fs = 16000
    spk_meta: MetaDictType = getattr(
        __import__("meta_dicts"),
        config.options.dataset_name,
    )
    print(f"config.paths.raw_wavs: {config.paths.raw_wavs}")
    dir_name, spk_dir_list, _ = next(os.walk(config.paths.raw_wavs))
    print(f"dir_name: {dir_name}")
    [
        make_sf_item(spk_dir, config, spk_meta, dir_name, fs)  # type: ignore [func-returns-value]
        for spk_dir in sorted(spk_dir_list)
        if spk_dir in spk_meta
    ]


def make_sf_item(
    spk_dir: str,
    config: Config,
    spk_meta,
    dir_name: str,
    fs: int,
) -> None:
    Logger().info(f"Generating features for speaker {spk_dir}")
    for fea_dir in [config.paths.monowavs, config.paths.spmels, config.paths.freqs]:
        if not os.path.exists(os.path.join(fea_dir, spk_dir)):
            os.makedirs(os.path.join(fea_dir, spk_dir))

    _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))
    sorted_file_list = sorted(file_list)

    if spk_meta[spk_dir][1] == "M":
        lo, hi = 50, 250
    elif spk_meta[spk_dir][1] == "F":
        lo, hi = 100, 600
    else:
        raise ValueError

    wavs: List[torch.Tensor] = []
    f0s: List[torch.Tensor] = []
    sps: List[torch.Tensor] = []
    aps: List[torch.Tensor] = []

    def getraw(fname: str) -> torch.Tensor:
        x: torch.Tensor = torchaudio.load(
            f"{dir_name}/{spk_dir}/{fname}",
            channels_first=True,
        )[0].squeeze()
        if x.shape[0] % 256 == 0:
            x = torch.cat(
                (
                    x,
                    torch.tensor([1e-06], device=x.device),
                ),
                dim=0,
            )
        return x

    wavs = [filter_wav(getraw(filename)) for filename in sorted_file_list]
    for wav in wavs:
        # get WORLD analyzer parameters
        f0, sp, ap = get_world_params(wav, fs)
        f0s.append(f0)
        sps.append(sp)
        aps.append(ap)

    [
        process_file(
            filename=filename,
            wav=wav,
            f0=f0,
            sp=sp,
            ap=ap,
            fs=fs,
            lo=lo,
            hi=hi,
            wav_dir=config.paths.monowavs,
            f0_dir=config.paths.freqs,
            spk_dir=spk_dir,
            spmel_dir=config.paths.spmels,
        )  # type: ignore [func-returns-value]
        for filename, wav, f0, sp, ap in zip(
            sorted_file_list,  # filename
            wavs,  # wav
            average_f0s(f0s),  # f0
            sps,  # sp
            aps,  # ap
        )
    ]


def process_item(
    spk_dir: str,
    spk_meta: MetaDictType,
    config: Config,
    dir_name: str,
) -> DataPreProcessType:
    spk_id: str
    spk_id, _, _ = spk_meta[spk_dir]
    # may use generalized speaker embedding for zero-shot conversion
    spk_emb: torch.Tensor = torch.zeros(
        (config.model.dim_spk_emb,),
        dtype=torch.float32,
    )
    spk_emb[int(spk_id)] = 1.0
    _, _, file_list = next(
        os.walk(
            os.path.join(
                dir_name,
                spk_dir,
            )
        )
    )
    filepaths: List[str] = [
        str(os.path.join(spk_dir, filename)) for filename in sorted(file_list)
    ]
    return [
        (
            spk_dir,
            spk_emb,
            filepath,
        )
        for filepath in filepaths
    ]


def make_metadata(
    config: Config,
) -> None:
    # use wav directory simply because all inputs have the same filename
    dir_name, spk_dir_list, _ = next(os.walk(config.paths.monowavs))
    spk_meta: MetaDictType = getattr(
        __import__("meta_dicts"),
        config.options.dataset_name,
    )
    dataset: DataPreProcessType = []

    [
        dataset.extend(
            process_item(
                spk_dir,
                spk_meta,
                config,
                dir_name,
            )
        )  # type: ignore [func-returns-value]
        for spk_dir in sorted(spk_dir_list)
        if spk_dir in spk_meta
    ]

    torch.save(
        dataset,
        os.path.join(config.paths.features, "dataset.pkl"),
    )


def preprocess_data(
    config: Config,
):
    dataset_file = os.path.join(config.paths.features, "dataset.pkl")
    dataset_exists = os.path.exists(dataset_file)
    if not dataset_exists or config.options.regenerate_metadata:
        feat_dir = config.paths.features
        procdata_exists = all(
            [
                os.path.exists(f"{feat_dir}/freqs/{speaker}")
                for speaker in getattr(
                    __import__("meta_dicts"),
                    config.options.dataset_name,
                ).keys()
            ]
        )
        if config.options.regenerate_data or not procdata_exists:
            Logger().info("Generating Spectrograms and Frequency Contours")
            make_spect_f0(config)
        if dataset_exists:
            os.remove(dataset_file)
        Logger().info("Generating Metadata")
        make_metadata(config)
        Logger().info("Preprocessing Complete")
    else:
        Logger().info(
            f"Dataset '{config.paths.features}/dataset.pkl' exists, "
            "skipping preprocessing"
        )
