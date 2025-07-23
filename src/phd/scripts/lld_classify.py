# mypy: disable-error-code="func-returns-value"
import pickle

import torch

from ..data import AudioProcs, DatasetParser
from ..data.dataset import SampleInfo
from ..util import compute, config, logger
from ..util.arff import ArffRowType
from ..util.arff import load as arff_load
from ..util.file import basename, exists, newpath, path, walkfiles

parser = DatasetParser()
processor = AudioProcs()

compute.set_gpu()
compute.set_default()

experiment_dir = newpath(config.paths.artefacts, basename(__name__))
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate


class LLD:
    name: str
    frametime: float
    loudness: float
    alpharatio: float
    hammarbergindex: float
    slope0to500: float
    slope500to1500: float
    spectralflux: float
    mfcc1: float
    mfcc2: float
    mfcc3: float
    mfcc4: float
    f0semitone: float
    jitter: float
    shimmer: float
    hnr: float
    logrelf0h1h2: float
    logrelf0h1a3: float
    f1frequency: float
    f1bandwidth: float
    f1amplitude: float
    f2frequency: float
    f2bandwidth: float
    f2amplitude: float
    f3frequency: float
    f3bandwidth: float
    f3amplitude: float
    theclass: float

    def __init__(self, row: ArffRowType) -> None:
        def add_attr(name: str, item: str, t: type) -> None:
            thing = row.get(item)
            if isinstance(thing, t):
                setattr(self, name, thing)
            else:
                raise ValueError(
                    f"Incorrect type for {item}: {type(thing)}, expected {t}"
                )

        add_attr("name", "name", str)
        add_attr("frametime", "frameTime", float)
        add_attr("loudness", "Loudness_sma3", float)
        add_attr("alpharatio", "alphaRatio_sma3", float)
        add_attr("hammarbergindex", "hammarbergIndex_sma3", float)
        add_attr("slope0to500", "slope0-500_sma3", float)
        add_attr("slope500to1500", "slope500-1500_sma3", float)
        add_attr("spectralflux", "spectralFlux_sma3", float)
        add_attr("mfcc1", "mfcc1_sma3", float)
        add_attr("mfcc2", "mfcc2_sma3", float)
        add_attr("mfcc3", "mfcc3_sma3", float)
        add_attr("mfcc4", "mfcc4_sma3", float)
        add_attr("f0semitone", "F0semitoneFrom27.5Hz_sma3nz", float)
        add_attr("jitter", "jitterLocal_sma3nz", float)
        add_attr("shimmer", "shimmerLocaldB_sma3nz", float)
        add_attr("hnr", "HNRdBACF_sma3nz", float)
        add_attr("logrelf0h1h2", "logRelF0-H1-H2_sma3nz", float)
        add_attr("logrelf0h1a3", "logRelF0-H1-A3_sma3nz", float)
        add_attr("f1frequency", "F1frequency_sma3nz", float)
        add_attr("f1bandwidth", "F1bandwidth_sma3nz", float)
        add_attr("f1amplitude", "F1amplitudeLogRelF0_sma3nz", float)
        add_attr("f2frequency", "F2frequency_sma3nz", float)
        add_attr("f2bandwidth", "F2bandwidth_sma3nz", float)
        add_attr("f2amplitude", "F2amplitudeLogRelF0_sma3nz", float)
        add_attr("f3frequency", "F3frequency_sma3nz", float)
        add_attr("f3bandwidth", "F3bandwidth_sma3nz", float)
        add_attr("f3amplitude", "F3amplitudeLogRelF0_sma3nz", float)
        add_attr("theclass", "class", float)


class LLDDataset(torch.utils.data.Dataset[LLD]):
    dataset: list[LLD]
    cache_file: str = path(config.paths.proc_data, "OpenSMILE", "LLDDataset.pkl")

    def __init__(self) -> None:
        if not exists(self.cache_file):
            logger.info("Generating LLD Dataset")
            lld_path = path(config.paths.proc_data, "OpenSMILE", "custom-arff")
            arff_files = walkfiles(lld_path, fullpaths=True)
            if len(arff_files) == 0:
                raise Exception("ERROR: No LLD files in dataset")
            self.dataset = []
            [
                self.dataset.extend(
                    [
                        lld
                        for lld in [LLD(line) for line in arff_load(file)]
                        if lld.name is not None
                    ]
                )
                for file in logger.progress_bar(arff_files, unit="files")
            ]
            logger.info("Saving LLD Dataset")
            with open(self.cache_file, "wb") as pklfile:
                pickle.dump(self.dataset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Loading LLD Dataset")
            with open(self.cache_file, "rb") as pklfile:
                self.dataset = pickle.load(pklfile)
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LLD:
        return self.dataset[index]


def lld_classify() -> None:
    arff_dataset = LLDDataset()
    arff_data = torch.utils.data.DataLoader(dataset=arff_dataset)
    for row in arff_data:
        if row.get("name") is None:
            logger.debug(f"Row name is None, items: {row}")
        else:
            lld = LLD(row)
            info = SampleInfo(lld.name)
            logger.trace_var(info.speaker, "DEBUG")
            logger.trace_var(info.utterance, "DEBUG")
            logger.trace_var(info.sex, "DEBUG")
            logger.trace_var(info.dysarthric, "DEBUG")
    return
