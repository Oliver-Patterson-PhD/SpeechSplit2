import os
import pickle
from typing import List

import torch
import torchaudio
from utils import (average_f0s, extract_f0, filter_wav, get_monotonic_wav,
                   get_spmel, get_world_params)


def process_file(
    idx, filename, wav, f0, sp, ap, fs, lo, hi, wav_dir, f0_dir, spk_dir, spmel_dir
):
    wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)
    spmel = get_spmel(wav)
    f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)
    assert len(spmel) == len(f0_rapt)

    # segment feature into trunks with the same length during training
    start_idx = 0
    trunk_len = 49151
    while start_idx * trunk_len < len(wav_mono):
        wav_mono_trunk = wav_mono[start_idx * trunk_len : (start_idx + 1) * trunk_len]
        if len(wav_mono_trunk) < trunk_len:
            wav_mono_trunk = torch.nn.functional.pad(
                wav_mono_trunk, (0, trunk_len - len(wav_mono_trunk))
            )
        torch.save(
            wav_mono_trunk.float(),
            os.path.join(
                wav_dir,
                spk_dir,
                os.path.splitext(filename)[0] + "_" + str(start_idx),
            ),
        )
        start_idx += 1
    feas = [spmel, f0_norm]
    fea_dirs = [spmel_dir, f0_dir]
    for fea, fea_dir in zip(feas, fea_dirs):
        start_idx = 0
        trunk_len = 192
        while start_idx * trunk_len < len(fea):
            fea_trunk = fea[start_idx * trunk_len : (start_idx + 1) * trunk_len]
            if len(fea_trunk) < trunk_len:
                if fea_trunk.ndim == 2:
                    fea_trunk = torch.nn.functional.pad(
                        fea_trunk, (0, trunk_len - len(fea_trunk), 0, 0)
                    )
                else:
                    fea_trunk = torch.nn.functional.pad(
                        fea_trunk,
                        (
                            0,
                            trunk_len - len(fea_trunk),
                        ),
                    )
            torch.save(
                fea_trunk.float(),
                os.path.join(
                    fea_dir,
                    spk_dir,
                    os.path.splitext(filename)[0] + "_" + str(start_idx),
                ),
            )
            start_idx += 1


def make_spect_f0(config) -> None:
    fs = 16000
    if config.dataset_name == "vctk":
        data_dir = config.vctk_dir
    elif config.dataset_name == "uaspeech":
        data_dir = config.uaspeech_dir
    else:
        raise ValueError
    feat_dir = os.path.join(config.base_feats, config.dataset_name)
    wav_dir = os.path.join(feat_dir, config.wav_dir)
    spmel_dir = os.path.join(feat_dir, config.spmel_dir)
    f0_dir = os.path.join(feat_dir, config.f0_dir)
    spk_meta = getattr(__import__("meta_dicts"), config.dataset_name)

    dir_name, spk_dir_list, _ = next(os.walk(data_dir))

    for spk_dir in sorted(spk_dir_list):
        if spk_dir not in spk_meta:
            print(f"skip generating features for {spk_dir}")
            continue
        print(f"Generating features for speaker {spk_dir}")

        for fea_dir in [wav_dir, spmel_dir, f0_dir]:
            if not os.path.exists(os.path.join(fea_dir, spk_dir)):
                os.makedirs(os.path.join(fea_dir, spk_dir))

        _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))

        if spk_meta[spk_dir][1] == "M":
            lo, hi = 50, 250
        elif spk_meta[spk_dir][1] == "F":
            lo, hi = 100, 600
        else:
            continue

        wavs: List[torch.Tensor] = []
        f0s: List[torch.Tensor] = []
        sps: List[torch.Tensor] = []
        aps: List[torch.Tensor] = []

        def getraw(fname):
            x = torchaudio.load(
                os.path.join(dir_name, spk_dir, fname), channels_first=True
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
                wav_dir,
                f0_dir,
                spk_dir,
                spmel_dir,
            )
            for idx, (filename, wav, f0, sp, ap) in enumerate(
                zip(file_list, wavs, f0s, sps, aps)
            )
        ]


def make_metadata(config):
    feat_dir = os.path.join(config.base_feats, config.dataset_name)
    # use wav directory simply because all inputs have the same filename
    wav_dir = os.path.join(feat_dir, config.wav_dir)
    dir_name, spk_dir_list, _ = next(os.walk(wav_dir))
    spk_meta = getattr(__import__("meta_dicts"), config.dataset_name)
    dataset = []

    for spk_dir in sorted(spk_dir_list):
        spk_id, _ = spk_meta[spk_dir]

        # may use generalized speaker embedding for zero-shot conversion
        spk_emb = torch.zeros((config.dim_spk_emb,), dtype=torch.float32)
        spk_emb[int(spk_id)] = 1.0

        _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))
        file_list = sorted(file_list)
        utterances = [os.path.join(spk_dir, filename) for filename in file_list]
        for utterance in utterances:
            dataset.append((spk_dir, spk_emb, utterance))

    with open(os.path.join(feat_dir, "dataset.pkl"), "wb") as handle:
        pickle.dump(dataset, handle)


def preprocess_data(config):
    feat_dir = os.path.join(config.base_feats, config.dataset_name)
    if not os.path.exists(os.path.join(feat_dir, "dataset.pkl")):
        print("Start preprocessing...")
        make_spect_f0(config)
        make_metadata(config)
        print("Done")
    else:
        print(f"Dataset '{feat_dir}/dataset.pkl' exists, skipping preprocessing")
