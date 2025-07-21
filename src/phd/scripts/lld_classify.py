# mypy: disable-error-code="func-returns-value"
import torch

from ..data import AudioProcs, DatasetParser, get_loader
from ..util import Compute, Config, Logger
from ..util.file import newpath, path, basename, walkfiles
from ..util.arff import load as arff_load

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
        self.name = row.get("name")
        self.frametime = row.get("frameTime")
        self.loudness = row.get("Loudness_sma3")
        self.alpharatio = row.get("alphaRatio_sma3")
        self.hammarbergindex = row.get("hammarbergIndex_sma3")
        self.slope0to500 = row.get("slope0-500_sma3")
        self.slope500to1500 = row.get("slope500-1500_sma3")
        self.spectralflux = row.get("spectralFlux_sma3")
        self.mfcc1 = row.get("mfcc1_sma3")
        self.mfcc2 = row.get("mfcc2_sma3")
        self.mfcc3 = row.get("mfcc3_sma3")
        self.mfcc4 = row.get("mfcc4_sma3")
        self.f0semitone = row.get("F0semitoneFrom27.5Hz_sma3nz")
        self.jitter = row.get("jitterLocal_sma3nz")
        self.shimmer = row.get("shimmerLocaldB_sma3nz")
        self.hnr = row.get("HNRdBACF_sma3nz")
        self.logrelf0h1h2 = row.get("logRelF0-H1-H2_sma3nz")
        self.logrelf0h1a3 = row.get("logRelF0-H1-A3_sma3nz")
        self.f1frequency = row.get("F1frequency_sma3nz")
        self.f1bandwidth = row.get("F1bandwidth_sma3nz")
        self.f1amplitude = row.get("F1amplitudeLogRelF0_sma3nz")
        self.f2frequency = row.get("F2frequency_sma3nz")
        self.f2bandwidth = row.get("F2bandwidth_sma3nz")
        self.f2amplitude = row.get("F2amplitudeLogRelF0_sma3nz")
        self.f3frequency = row.get("F3frequency_sma3nz")
        self.f3bandwidth = row.get("F3bandwidth_sma3nz")
        self.f3amplitude = row.get("F3amplitudeLogRelF0_sma3nz")
        self.theclass = row.get("class")
        return


class LLDDataset(torch.utils.data.Dataset[LLD]):
    dataset: list[LLD]

    def __init__(self) -> None:
        lld_path = path(config.paths.proc_data, "OpenSMILE", "custom-arffs")
        self.dataset = []
        [
            self.dataset.extend([LLD(line) for line in arff_load(file)])
            for file in logger.progress_bar(walkfiles(lld_path), unit="files")
        ]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LLD:
        return self.dataset[index]


def lld_classify() -> None:
    return
