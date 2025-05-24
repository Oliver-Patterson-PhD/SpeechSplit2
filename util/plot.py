import matplotlib
import torch

from data import Utterance
from util.file import path


def plot_things(
    plot_out: str,
    sample: str,
    things: list[tuple[torch.Tensor, str]],
    word: Utterance | None = None,
    label_colour: str = "r",
    lines_colour: str = "k",
    spect_cmap: str = "binary",
    sample_time: int | None = None,
    sample_start: int | None = None,
    sample_end: int | None = None,
):
    nrows = len(things)
    ncols = 1
    fig = matplotlib.pyplot.figure()
    fig.set_size_inches(15.44, 27.45)
    if word is None:
        fig.suptitle(f"Sample: {sample}")
    else:
        fig.suptitle(f"Sample: {sample} ({word.word})")
    fig.subplots(nrows, ncols)
    for i, (item, name) in enumerate(things):
        ax = matplotlib.pyplot.subplot(nrows, ncols, i + 1)
        try:
            if sample_time is not None:
                sample_div = item.size(-1) / sample_time
            else:
                sample_div = 1
            if item.dim() == 1:
                if sample_time is not None:
                    ax.plot(
                        [i / sample_div for i in range(item.size(-1))],
                        item.cpu().numpy(),
                    )
                    ax.set_xlabel("Time (Seconds)")
                    ax.set_xlim(0, sample_time)
                else:
                    ax.plot(item.cpu().numpy())
                    ax.set_xlabel("Samples")
                    ax.set_xlim(sample_start or 0, sample_end or item.size(dim=-1))
            elif item.dim() == 2:
                ax.imshow(
                    item.cpu().numpy(),
                    interpolation="none",
                    aspect="auto",
                    origin="lower",
                )
            else:
                raise RuntimeError(f"Invalid Tensor has shape: {item.size()}")
            ax.set_title(name, loc="left", pad=16)
            if word is not None:
                for phon in word.phones:
                    div = word.end / item.size(dim=-1)
                    if item.dim() == 1:
                        half_point = (
                            (phon.start + ((phon.end - phon.start) / 2)) / div
                        ) / sample_div
                        start_point = (phon.start / sample_div) / div
                    if item.dim() == 2:
                        half_point = (phon.start + ((phon.end - phon.start) / 2)) / div
                        start_point = phon.start / div
                    ax.annotate(
                        text=phon.phon,
                        xy=(half_point, ax.get_ylim()[1]),
                        xytext=(0, 3),
                        textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="baseline",
                        color=label_colour,
                    )
                    ax.axvline(
                        start_point,
                        color=label_colour,
                        alpha=0.3,
                    )
        except Exception as e:
            raise RuntimeError(f"Failed at: {name}") from e
    fig.savefig(path(plot_out, f"{sample}.pdf"))
    matplotlib.pyplot.close()
