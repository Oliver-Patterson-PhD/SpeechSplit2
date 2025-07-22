# mypy: disable-error-code="func-returns-value"
import pickle

import matplotlib
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

from ..data import AudioProcs, DatasetParser
from ..util import Compute, Config, Logger
from ..util.file import newpath, path, walkdirs, walkfiles, exists

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


def run_item(spk: str, uttr: str) -> tuple[float, float, float, float] | str:
    file = path(in_path, parser.get_wavfile(spk, parser.utterance(uttr)))
    if not exists(file):
        logger.warn(f"File doesn't exist: {file}")
        return file
    raw, nonoise, nopop, clean = processor.full_load_parts(file)
    check = processor.full_load_check(raw, nonoise, nopop, clean)
    if check is None:
        return metric(clean)
    else:
        return f"{file}: {check}"


def print_stats(outfile, thing: list[int | float], name: str) -> None:
    thing_len = len(thing)
    thing_mean = sum(thing) / thing_len
    thing_max = max(thing)
    thing_min = min(thing)
    logger.debug(f"{name:>5} Max:  {thing_max}")
    logger.debug(f"{name:>5} Min:  {thing_min}")
    logger.debug(f"{name:>5} Mean: {thing_mean}")


def run_speaker(spk: str) -> None:
    logger.debug(f"Running: {spk}")
    p808: list[float] = []
    sig: list[float] = []
    bak: list[float] = []
    ovr: list[float] = []
    try:
        with open(path(out_path, f"{spk}.txt"), "w") as outfile:
            logger.debug(f"Loading: {spk}")
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
                for p, s, b, o in [
                    result for result in results if not isinstance(result, str)
                ]
            ]
            print_stats(outfile, p808, "P808")
            print_stats(outfile, sig, "sig")
            print_stats(outfile, bak, "bak")
            print_stats(outfile, ovr, "ovr")
            print("data", file=outfile)
            print(f"p808: {p808}", file=outfile)
            print(f"sig: {sig}", file=outfile)
            print(f"bak: {bak}", file=outfile)
            print(f"ovr: {ovr}", file=outfile)
            print("errors", file=outfile)
            [
                print(f"{result}", file=outfile)
                for result in results
                if isinstance(result, str)
            ]
        dumppath = newpath(out_path, f"dumps-{spk}")
        with open(path(dumppath, "p808.pkl"), "wb") as pklfile:
            pickle.dump(p808, pklfile, pickle.HIGHEST_PROTOCOL)
        with open(path(dumppath, "sig.pkl"), "wb") as pklfile:
            pickle.dump(sig, pklfile, pickle.HIGHEST_PROTOCOL)
        with open(path(dumppath, "bak.pkl"), "wb") as pklfile:
            pickle.dump(bak, pklfile, pickle.HIGHEST_PROTOCOL)
        with open(path(dumppath, "ovr.pkl"), "wb") as pklfile:
            pickle.dump(ovr, pklfile, pickle.HIGHEST_PROTOCOL)
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
        fig_out = newpath(out_path, str(parser.dataset_type()))
        fig.savefig(path(fig_out, f"{spk}.{ftype}"))
        matplotlib.pyplot.close(fig=fig)
        logger.debug(f"Plots finished: {spk}")
    except Exception as e:
        logger.warn(str(e))


def scratch() -> None:
    for spk in set(walkdirs(in_path)):
        if spk in parser.speakers():
            run_speaker(spk)
