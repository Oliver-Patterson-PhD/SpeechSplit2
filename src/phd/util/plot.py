import matplotlib

from data import Utterance
from util.file import path
from util.tensor import Tensor


def _plot_single(
    ax: matplotlib.axes.Axes,
    item: Tensor,
    name: str,
    sample_time: float | None,
    sample_start: float | None,
    sample_end: float | None,
    sample_div: float,
    line: bool,
    annotate: bool,
) -> matplotlib.axes.Axes:
    ax.plot(
        (
            [i / sample_div for i in range(item.size(-1))]
            if sample_time is not None
            else list(range(item.size(-1)))
        ),
        item.cpu().numpy(),
    )
    if line:
        ax.set_label(name)
    ax.set_xlim(sample_start or 0, sample_time or sample_end or item.size(dim=-1))
    if annotate:
        ax.set_xlabel("Time (Seconds)" if sample_time is not None else "Samples")
    if not line:
        ax.set_title(name, loc="left", pad=16)
    return ax


def _plot_multi(
    ax: matplotlib.axes.Axes,
    item: Tensor,
    name: tuple[str, ...],
    sample_time: float | None,
    sample_start: float | None,
    sample_end: float | None,
    sample_div: float,
    line: bool,
    annotate: bool,
) -> matplotlib.axes.Axes:
    for i_item, i_name in zip(item, name):
        ax = _plot_single(
            ax=ax,
            item=i_item,
            name=i_name,
            sample_time=sample_time,
            sample_start=sample_start,
            sample_end=sample_end,
            sample_div=sample_div,
            line=True,
            annotate=annotate,
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.8, 1.3))
    return ax


def _plot_image(
    ax: matplotlib.axes.Axes,
    item: Tensor,
    name: str,
    sample_time: float | None,
) -> matplotlib.axes.Axes:
    ysize = item.size(dim=-2)
    xsize = sample_time or item.size(dim=-1)
    ax.imshow(
        item.cpu().numpy(),
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=(0, xsize, -ysize / 2, ysize / 2),
    )
    # ax.set_axis_off()
    ax.set_title(name, loc="left", pad=16)
    return ax


def _make_plot(
    ax: matplotlib.axes.Axes,
    item: Tensor,
    name: str | tuple[str, ...],
    sample_time: float | None,
    sample_start: float | None,
    sample_end: float | None,
    label_colour: str,
    word: Utterance | None,
    annotate: bool,
) -> matplotlib.axes.Axes:
    sample_div = 1 if sample_time is None else item.size(-1) // sample_time
    is_image = False
    if isinstance(name, tuple):
        ax = _plot_multi(
            ax=ax,
            item=item,
            name=name,
            sample_time=sample_time,
            sample_start=sample_start,
            sample_end=sample_end,
            sample_div=sample_div,
            line=True,
            annotate=annotate,
        )
    elif isinstance(name, str):
        if item.dim() == 1:
            ax = _plot_single(
                ax=ax,
                item=item,
                name=name,
                sample_time=sample_time,
                sample_start=sample_start,
                sample_end=sample_end,
                sample_div=sample_div,
                line=False,
                annotate=annotate,
            )
        elif item.dim() == 2:
            is_image = True
            _plot_image(
                ax=ax,
                item=item,
                name=name,
                sample_time=sample_time,
            )
    else:
        raise RuntimeError(f"Invalid Tensor with shape: {item.size()}")
    if word is not None:
        for phon in word.phones:
            if item.dim() == 1 or isinstance(name, tuple):
                size_item = item.masked_select(item != 0.0)
                div = word.end / size_item.size(dim=-1)
                half_point = (
                    (phon.start + ((phon.end - phon.start) / 2)) / div
                ) / sample_div
                start_point = (phon.start / sample_div) / div
            elif item.dim() == 2:
                div = word.end / (sample_time or 1)
                half_point = (phon.start + ((phon.end - phon.start) / 2)) / div
                start_point = phon.start / div
            else:
                raise RuntimeError(f"Invalid Tensor ({item.size()}) with str ({name})")
            if not isinstance(name, tuple):
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
                    start_point, color=label_colour, alpha=0.4 if is_image else 0.1
                )
    return ax


def plot_things(
    plot_out: str,
    sample: str,
    things: list[tuple[Tensor, str | tuple[str, ...]]],
    utterances: list[Utterance] | Utterance | None = None,
    label_colour: str = "k",
    spect_cmap: str = "binary",
    sample_time: float | None = None,
    sample_start: float | None = None,
    sample_end: float | None = None,
    ftype: str = "pdf",
) -> None:
    fig = matplotlib.pyplot.figure()
    match ftype:
        case "pdf":
            fig.set_size_inches(15.44, 27.45)
        case "png":
            fig.set_size_inches(12, 12)
            fig.set_dpi(300)
    uttr_list: list[Utterance | None]
    if utterances is None:
        fig.suptitle(f"Sample: {sample}")
        uttr_list = [None for _ in things]
    elif isinstance(utterances, list):
        fig.suptitle(f"Sample: {sample} ({utterances[0].word})")
        uttr_list = [uttr for uttr in utterances]
        assert len(uttr_list) == len(things)
    elif isinstance(utterances, Utterance):
        fig.suptitle(f"Sample: {sample} ({utterances.word})")
        uttr_list = [utterances for _ in things]
    else:
        raise RuntimeError(f"invalid type for utterances: {type(utterances)}")
    fig.subplots(len(things), 1)
    [
        _make_plot(
            ax=matplotlib.pyplot.subplot(len(things), 1, i + 1),
            item=item,
            name=name,
            sample_time=sample_time,
            sample_start=sample_start,
            sample_end=sample_end,
            label_colour=label_colour,
            word=uttr,
            annotate=(i == len(things)),
        )
        for i, ((item, name), uttr) in enumerate(zip(things, uttr_list))
    ]
    fig.savefig(path(plot_out, f"{sample}.{ftype}"))
    matplotlib.pyplot.close(fig=fig)


def make_idx(n_rows: int, n_cols: int, row_idx: int, col_idx: int) -> int:
    return (n_cols * row_idx) + col_idx + 1


def plot_multicol_things(
    plot_out: str,
    sample: str,
    things: list[tuple[tuple[Tensor, str | tuple[str, ...]], ...]],
    utterances: list[Utterance] | Utterance | None = None,
    label_colour: str = "k",
    spect_cmap: str = "binary",
    sample_time: float | None = None,
    sample_start: float | None = None,
    sample_end: float | None = None,
    ftype: str = "pdf",
) -> None:
    fig = matplotlib.pyplot.figure()
    match ftype:
        case "pdf":
            fig.set_size_inches(15.44, 27.45)
        case "png":
            fig.set_size_inches(12, 12)
            fig.set_dpi(300)
    uttr_list: list[Utterance | None]
    if utterances is None:
        fig.suptitle(f"Sample: {sample}")
        uttr_list = [None for _ in things]
    elif isinstance(utterances, list):
        fig.suptitle(f"Sample: {sample} ({utterances[0].word})")
        uttr_list = [uttr for uttr in utterances]
        assert len(uttr_list) == len(things)
    elif isinstance(utterances, Utterance):
        fig.suptitle(f"Sample: {sample} ({utterances.word})")
        uttr_list = [utterances for _ in things]
    else:
        raise RuntimeError(f"invalid type for utterances: {type(utterances)}")
    n_rows = len(things)
    n_cols = len(things[0])
    [
        _make_plot(
            ax=matplotlib.pyplot.subplot(
                n_rows, n_cols, make_idx(n_rows, n_cols, row_idx, col_idx)
            ),
            item=item,
            name=name,
            sample_time=sample_time,
            sample_start=sample_start,
            sample_end=sample_end,
            label_colour=label_colour,
            word=None,
            annotate=(row_idx == len(things)),
        )
        for row_idx, row_item in enumerate(things)
        for col_idx, (item, name) in enumerate(row_item)
    ]
    fig.savefig(path(plot_out, f"{sample}.{ftype}"))
    matplotlib.pyplot.close(fig=fig)
