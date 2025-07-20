# mypy: disable-error-code="func-returns-value"
import arff
import torch

from ..data import AudioProcs, DatasetParser, get_loader
from ..util import Compute, Config, Logger
from ..util.file import newpath, path, basename, walkfiles

logger = Logger()
config = Config("base")
compute = Compute()
parser = DatasetParser()
processor = AudioProcs()

compute.set_gpu()
compute.set_default()

experiment_dir = newpath(config.paths.artefacts, basename(__name__))
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate
data_loader = get_loader(config)


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

    def __init__(self, row) -> None:
        self.name = getattr(row, "name")
        self.frametime = getattr(row, "frameTime")
        self.loudness = getattr(row, "Loudness_sma3")
        self.alpharatio = getattr(row, "alphaRatio_sma3")
        self.hammarbergindex = getattr(row, "hammarbergIndex_sma3")
        self.slope0to500 = getattr(row, "slope0-500_sma3")
        self.slope500to1500 = getattr(row, "slope500-1500_sma3")
        self.spectralflux = getattr(row, "spectralFlux_sma3")
        self.mfcc1 = getattr(row, "mfcc1_sma3")
        self.mfcc2 = getattr(row, "mfcc2_sma3")
        self.mfcc3 = getattr(row, "mfcc3_sma3")
        self.mfcc4 = getattr(row, "mfcc4_sma3")
        self.f0semitone = getattr(row, "F0semitoneFrom27.5Hz_sma3nz")
        self.jitter = getattr(row, "jitterLocal_sma3nz")
        self.shimmer = getattr(row, "shimmerLocaldB_sma3nz")
        self.hnr = getattr(row, "HNRdBACF_sma3nz")
        self.logrelf0h1h2 = getattr(row, "logRelF0-H1-H2_sma3nz")
        self.logrelf0h1a3 = getattr(row, "logRelF0-H1-A3_sma3nz")
        self.f1frequency = getattr(row, "F1frequency_sma3nz")
        self.f1bandwidth = getattr(row, "F1bandwidth_sma3nz")
        self.f1amplitude = getattr(row, "F1amplitudeLogRelF0_sma3nz")
        self.f2frequency = getattr(row, "F2frequency_sma3nz")
        self.f2bandwidth = getattr(row, "F2bandwidth_sma3nz")
        self.f2amplitude = getattr(row, "F2amplitudeLogRelF0_sma3nz")
        self.f3frequency = getattr(row, "F3frequency_sma3nz")
        self.f3bandwidth = getattr(row, "F3bandwidth_sma3nz")
        self.f3amplitude = getattr(row, "F3amplitudeLogRelF0_sma3nz")
        self.theclass = getattr(row, "class")
        return


class LLDDataset(torch.utils.data.Dataset[LLD]):
    dataset: list[LLD]

    def __init__(self) -> None:
        lld_path = path(config.paths.proc_data, "OpenSMILE", "custom-arffs")
        self.dataset = []
        [
            self.dataset.extend([LLD(line) for line in arff.load(file)])
            for file in logger.progress_bar(walkfiles(lld_path), unit="files")
        ]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LLD:
        return self.dataset[index]


def lld_classify() -> None:
    return
