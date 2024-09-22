import os
from typing import List, Tuple

import torch
import torchaudio
from meta_dicts import MetaDictType
from util.config import Config
from util.logging import Logger
from utils import (average_f0s, extract_f0, filter_wav, get_monotonic_wav,
                   get_spmel, get_world_params, norm_audio)

sample_rate = 16000
new_epsilon = 1e-10


def split_feats(
    fea: torch.Tensor,
    trunk_len: int,
) -> List[torch.Tensor]:
    start_idx = 0
    split_list: List[torch.Tensor] = []
    while start_idx * trunk_len < len(fea):
        this_trunk = start_idx * trunk_len
        next_trunk = (start_idx + 1) * trunk_len
        start_idx += 1
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
        if has_content(fea_trunk):
            split_list.append(fea_trunk.to(torch.float32))
    return split_list


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
    if has_content(wav):
        wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)
        spmel = get_spmel(wav)
        f0_norm = extract_f0(wav, fs, lo, hi)
        if len(spmel) != len(f0_norm):
            Logger().fatal(
                f"melspec and f0 lengths do not match for {filename}"
                f"spmel: {len(spmel)}\n"
                f"f0_rapt: {len(f0_norm)}\n"
            )
        if (has_content(wav_mono)) and (has_content(spmel)) and (has_content(f0_norm)):
            wav_mono_split = split_feats(
                fea=wav_mono,
                trunk_len=49151,
            )
            spmel_split = split_feats(
                fea=spmel,
                trunk_len=192,
            )
            f0_split = split_feats(
                fea=f0_norm,
                trunk_len=192,
            )
            tmpdir = f"{wav_dir}/orig_vad/{spk_dir}/"
            os.makedirs(tmpdir, exist_ok=True)
            torchaudio.save(
                uri=tmpdir + f"{os.path.splitext(filename)[0]}.wav",
                src=wav.unsqueeze(dim=0),
                sample_rate=sample_rate,
            )
            for idx, (wav_mono_i, spmel_i, f0_i) in enumerate(
                zip(wav_mono_split, spmel_split, f0_split)
            ):
                fname = f"{os.path.splitext(filename)[0]}_{idx}.pt"
                good: bool = True
                good &= has_content(wav_mono_i)
                good &= has_content(spmel_i)
                good &= has_content(f0_i)
                if good:
                    torch.save(wav_mono_i, f"{wav_dir}/{spk_dir}/" + fname)
                    torch.save(spmel_i, f"{spmel_dir}/{spk_dir}/" + fname)
                    torch.save(f0_i, f"{f0_dir}/{spk_dir}/" + fname)


def make_spect_f0(config: Config) -> None:
    fs = 16000
    spk_meta: MetaDictType = getattr(
        __import__("meta_dicts"),
        config.options.dataset_name,
    )
    dir_name, spk_dir_list, _ = next(os.walk(config.paths.raw_wavs))
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
    f0s: List[torch.Tensor] = []
    sps: List[torch.Tensor] = []
    aps: List[torch.Tensor] = []
    Logger().info(f"Generating features for speaker {spk_dir}")
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

    def getraw(fname: str) -> torch.Tensor:
        x: torch.Tensor
        x = clean_audio(
            torchaudio.load(
                f"{dir_name}/{spk_dir}/{fname}",
                channels_first=True,
            )[0],
            fname,
        )
        if x.shape[0] % 256 == 0:
            x = torch.cat(
                (
                    x,
                    torch.tensor([new_epsilon], device=x.device),
                ),
                dim=0,
            )
        return x

    wavs: List[torch.Tensor] = []
    fnames: List[str] = []
    for fname in sorted(file_list):
        wav = filter_wav(getraw(fname))
        if has_content(wav):
            fnames.append(fname)
            wavs.append(wav)
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
            fnames,  # filename
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
) -> List[Tuple[str, torch.Tensor, str]]:
    spk_id: str
    spk_emb: torch.Tensor
    filepaths: List[str]
    spk_id, _, _ = spk_meta[spk_dir]
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
    filepaths = [str(os.path.join(spk_dir, filename)) for filename in sorted(file_list)]
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
    meta_file: str,
) -> None:
    spk_meta: MetaDictType
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

    torch.save(dataset, meta_file)


def preprocess_data(
    config: Config,
):
    speaker_list = getattr(
        __import__("meta_dicts"),
        config.options.dataset_name,
    ).keys()
    feat_dir = config.paths.features
    procdata_exists = all(
        [os.path.exists(f"{feat_dir}/freqs/{speaker}") for speaker in speaker_list]
    )
    if config.options.regenerate_data or not procdata_exists:
        Logger().info("Generating Spectrograms and Frequency Contours")
        make_spect_f0(config)
    Logger().info("Preprocessing Complete")


vad_transform = torchaudio.transforms.Vad(
    sample_rate=sample_rate,
)


def clean_audio(audio: torch.Tensor, fname: str):
    retval = torchaudio.sox_effects.apply_effects_tensor(
        vad_transform(
            torchaudio.sox_effects.apply_effects_tensor(
                vad_transform(norm_audio(audio)), sample_rate, [["reverse"]]
            )[0]
        ),
        sample_rate,
        [["reverse"]],
    )[0]
    return retval.squeeze()


def has_content(audio: torch.Tensor) -> bool:
    return (
        (audio.size(dim=-1) > 1) and (audio.max().item() > 1e-03) and (audio != 0).any()
    )
