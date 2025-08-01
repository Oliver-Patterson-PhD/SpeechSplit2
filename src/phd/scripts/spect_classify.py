# mypy: disable-error-code="func-returns-value"
from __future__ import annotations

import torch

from ..data import AudioProcs, DatasetParser
from ..util import compute, config, logger
from ..util.file import basename, exists, newpath, path, walkfiles

parser = DatasetParser()
processor = AudioProcs()

experiment_dir = newpath(config.paths.artefacts, basename(__name__))
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate
testing = False


class SpectDataset(torch.utils.data.Dataset):
    dataset_name = config.options.dataset_name
    dataset: list[torch.Tensor]
    small_cache_file: str = path(
        config.paths.proc_data, config.options.dataset_name, "spects", "small.pkl"
    )
    large_cache_file: str = path(
        config.paths.proc_data, config.options.dataset_name, "spects", "large.pkl"
    )
    cache_file: str

    def __init__(self) -> None:
        self.cache_file = self.small_cache_file if testing else self.large_cache_file
        if not exists(self.cache_file):
            logger.info("Generating Spectrogram Dataset")
            spec_path = path(config.paths.proc_data, "OpenSMILE", "custom-arff")
            wav_files = walkfiles(config.paths.raw_wavs, fullpaths=True)
            if len(wav_files) == 0:
                raise Exception("ERROR: No Spectrogram files in dataset")
            []


def spect_classify() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    compute.set_gpu()
    compute.set_default()

    data = SpectDataset()
