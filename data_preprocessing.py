import os
import pickle
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from util.config import Config
from util.logging import Logger
from utils import (average_f0s, extract_f0, filter_wav, get_monotonic_wav,
                   get_spmel, get_world_params)

logger = Logger()


def process_file(
    idx: int,
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
    f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)
    assert len(spmel) == len(f0_rapt), (
        f"\nspmel: {len(spmel)}\nf0_rapt: {len(f0_rapt)}",
    )

    start_idx = 0
    trunk_len = 49151
    while start_idx * trunk_len < len(wav_mono):
        this_trunk = start_idx * trunk_len
        next_trunk = (start_idx + 1) * trunk_len
        wav_mono_trunk = wav_mono[this_trunk:next_trunk]
        if len(wav_mono_trunk) < trunk_len:
            wav_mono_trunk = torch.nn.functional.pad(
                wav_mono_trunk, (0, trunk_len - len(wav_mono_trunk))
            )
        torch.save(
            wav_mono_trunk.to(torch.float32),
            "{}/{}/{}_{}.pt".format(
                wav_dir,
                spk_dir,
                os.path.splitext(filename)[0],
                start_idx,
            ),
        )
        start_idx += 1
    feas = [spmel, f0_norm]
    fea_dirs = [spmel_dir, f0_dir]
    for fea, fea_dir in zip(feas, fea_dirs):
        start_idx = 0
        trunk_len = 192
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
                        (
                            0,
                            trunk_len - len(fea_trunk),
                        ),
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


def make_spect_f0(config: Config) -> None:
    fs = 16000
    spk_meta = getattr(__import__("meta_dicts"), config.options.dataset_name)
    dir_name, spk_dir_list, _ = next(os.walk(config.paths.raw_wavs))
    dirlist = [thedir for thedir in sorted(spk_dir_list) if thedir in spk_meta]
    [
        make_sf_item(spk_dir, config, spk_meta, dir_name, fs)  # type: ignore [func-returns-value]
        for spk_dir in dirlist
    ]


def make_sf_item(
    spk_dir: str,
    config: Config,
    spk_meta,
    dir_name: str,
    fs: int,
) -> None:
    logger.debug(f"Generating features for speaker {spk_dir}")
    for fea_dir in [config.paths.monowavs, config.paths.spmels, config.paths.freqs]:
        if not os.path.exists(os.path.join(fea_dir, spk_dir)):
            os.makedirs(os.path.join(fea_dir, spk_dir))

    _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))

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

    def getraw(fname):
        x = torchaudio.load(
            f"{dir_name}/{spk_dir}/{fname}",
            channels_first=True,
        )[0].squeeze()
        if x.shape[0] % 256 == 0:
            x = torch.cat((x, torch.tensor([1e-06], device=x.device)), axis=0)
        return x

    wavs = [filter_wav(getraw(filename)) for filename in sorted(file_list)]
    for wav in wavs:
        # get WORLD analyzer parameters
        f0, sp, ap = get_world_params(wav, fs)
        f0s.append(f0)
        sps.append(sp)
        aps.append(ap)

    # smooth pitch to synthesize monotonic speech
    f0s = average_f0s(f0s)

    [
        process_file(
            idx,
            filename,
            wav,
            f0,
            sp,
            ap,
            fs,
            lo,
            hi,
            config.paths.monowavs,
            config.paths.freqs,
            spk_dir,
            config.paths.spmels,
        )  # type: ignore [func-returns-value]
        for idx, (filename, wav, f0, sp, ap) in enumerate(
            zip(file_list, wavs, f0s, sps, aps)
        )
    ]


def process_item(
    spk_dir: str,
    spk_meta: Dict[str, Any],
    config: Config,
    dir_name: str,
) -> List[Tuple[str, torch.Tensor, str]]:
    spk_id, _ = spk_meta[spk_dir] if spk_dir in spk_meta else None
    # may use generalized speaker embedding for zero-shot conversion
    spk_emb = torch.zeros(
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
    file_list = sorted(file_list)
    utterances = [os.path.join(spk_dir, filename) for filename in file_list]
    return [
        (
            spk_dir,
            spk_emb,
            utterance,
        )
        for utterance in utterances
    ]


def make_metadata(config: Config):
    # use wav directory simply because all inputs have the same filename
    dir_name, spk_dir_list, _ = next(os.walk(config.paths.monowavs))
    spk_meta = getattr(
        __import__("meta_dicts"),
        config.options.dataset_name,
    )
    dataset = []

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

    with open(
        os.path.join(
            config.paths.features,
            "dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(dataset, handle)


def preprocess_data(config: Config):
    if (
        not os.path.exists(
            os.path.join(
                config.paths.features,
                "dataset.pkl",
            )
        )
    ) or config.options.regenerate_data:
        logger.debug("Start preprocessing...")
        make_spect_f0(config)
        make_metadata(config)
        logger.debug("Done")
    else:
        logger.debug(
            f"Dataset '{config.paths.features}/dataset.pkl' exists, skipping preprocessing"
        )
