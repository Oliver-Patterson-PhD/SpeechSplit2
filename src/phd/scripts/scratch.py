# mypy: disable-error-code="func-returns-value"
import matplotlib
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

from ..data import AudioProcs, DatasetParser
from ..util import Compute, Config, Logger
from ..util.file import newpath, path, walkdirs, walkfiles

logger = Logger()
config = Config("base")
compute = Compute()
parser = DatasetParser()
processor = AudioProcs()

experiment_dir = newpath(config.paths.artefacts, "scratch")
in_path = config.paths.raw_wavs
out_path = newpath(experiment_dir, str(parser.dataset_type()))
sample_rate = config.audio.sample_rate
metric_t = tuple[float, float, float, float]
ftype = "pdf"
metric = DeepNoiseSuppressionMeanOpinionScore(
    fs=sample_rate,
    personalized=False,
)


def run_item(spk: str, uttr: str) -> tuple[float, float, float, float] | None:
    file = path(in_path, parser.get_wavfile(spk, parser.utterance(uttr)))
    raw, nonoise, nopop, clean = processor.full_load_parts(file)
    if processor.full_load_check(raw, nonoise, nopop, clean) is None:
        return metric(clean)
    else:
        return None


def run_speaker(spk: str) -> None:
    logger.debug(f"Loading: {spk}")
    p808: list[float] = []
    sig: list[float] = []
    bak: list[float] = []
    ovr: list[float] = []
    results = [
        run_item(spk, uttr)
        for uttr in logger.progress_bar(
            walkfiles(path(in_path, parser.get_spkdir(spk))), unit="loaded"
        )
    ]
    logger.debug(f"Splitting: {spk}")
    [
        (
            p808.append(float(p)),
            sig.append(float(s)),
            bak.append(float(b)),
            ovr.append(float(o)),
        )
        for p, s, b, o in [result for result in results if result is not None]
    ]
    logger.debug(f"Plotting: {spk}")
    fig = matplotlib.pyplot.figure()
    ax = fig.gca()
    ax.plot(p808)
    ax.set_label("P.808")
    ax.plot(sig)
    ax.set_label("SIG")
    ax.plot(bak)
    ax.set_label("BAK")
    ax.plot(ovr)
    ax.set_label("OVR")
    ax.legend(loc="upper left")
    fig.savefig(path(out_path, str(parser.dataset_type()), f"{spk}.{ftype}"))
    matplotlib.pyplot.close(fig=fig)
    logger.debug(f"Plots finished: {spk}")


def scratch() -> None:
    for spk in set(walkdirs(in_path)):
        if spk in parser.speakers():
            run_speaker(spk)
