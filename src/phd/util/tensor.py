from math import ceil
from pathlib import Path

import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import figure
from PIL import Image
from torch import Tensor

from .file import strip_ext

__all__ = [
    "Tensor",
    "TensorPair",
    "TensorQuad",
    "TensorTriple",
    "pad_like",
    "pad_list",
    "pad_to",
    "stack_list",
    "save_tensor",
]

TensorPair = tuple[Tensor, Tensor]
TensorTriple = tuple[Tensor, Tensor, Tensor]
TensorQuad = tuple[Tensor, Tensor, Tensor, Tensor]


## Save a tensor as an image if possible or as a `.pth` file if not
# @param tensor     The tensor to save
# @param save_path  Image path
# @param save_raw   Don't save as an image, only as a `.pth` file
@torch.no_grad()
def save_tensor(tensor: Tensor, save_path: str, save_raw: bool = False) -> None:
    image = None
    if not save_raw:
        image = _try_resize(tensor.abs())
    if image is not None:
        im_min = image.min()
        im_max = image.max()
        norm_image = 1.0 / (im_max - im_min) * image + 1.0 * im_min / (im_min - im_max)
        save_image(norm_image, save_path)
    else:
        torch.save(tensor, strip_ext(save_path) + ".pth")
    return


@torch.no_grad()
def _try_resize(tensor: Tensor) -> Tensor | None:
    if tensor is None:
        return None
    elif tensor.dim() == 2:
        return tensor
    elif tensor.dim() < 2:
        return None
    elif tensor.dim() > 2 and tensor.size(0) == 1:
        for in_tensor in tensor:
            return _try_resize(in_tensor)
    return None


## Save a given Tensor into an image file.
# @param tensor     Image to be saved. If given a mini-batch tensor, saves the tensor as a grid of images by calling ``make_grid``.
# @param filename   The filename
@torch.no_grad()
def save_image(tensor: Tensor, filename: str | Path) -> None:
    grid = make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(filename)


@torch.no_grad()
def make_grid(tensor: Tensor) -> Tensor:
    nrow: int = 8
    padding: int = 2
    pad_value: float = 0.0
    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)
    if tensor.size(0) == 1:
        return tensor.squeeze(0)
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def plot_batch(batch: Tensor, base: str) -> Figure:
    fig = figure()
    fig.set_size_inches(15.44, 27.45)
    nrows: int = batch.shape[0]
    ncols: int = 1
    for i, sample in enumerate(batch):
        idx = i + 1
        ax = fig.add_subplot(nrows, ncols, idx)
        ax.plot(sample.numpy())
        ax.set_title(f"{base}_M{idx}")
        ax.set_xlim(0, sample.size[-1])
    return fig


## Pad a tensor's final dimension to match a reference
# @param x      Tensor to pad
# @param ref    Reference tensor
def pad_like(x: Tensor, ref: Tensor) -> Tensor:
    assert x.shape <= ref.shape
    return torch.nn.functional.pad(x, (0, ref.size(dim=-1) - x.size(dim=-1)))


## Pad the final dimension of tensors in a list to the largest contained final dimension
def pad_list(xlist: list[Tensor]) -> list[Tensor]:
    biggest = max(xlist, key=lambda x: x.size(dim=-1)).size(dim=-1)
    return [torch.nn.functional.pad(x, (0, biggest - x.size(dim=-1))) for x in xlist]


## Stack a list of tensors, padding them to match the largest final dimension
def stack_list(xlist: list[Tensor]) -> Tensor:
    return torch.stack(pad_list(xlist))


## Pad the final dimension of x and y to the larger of the two.
def pad_to(x: Tensor, y: Tensor) -> TensorPair:
    if x.size(dim=-1) < y.size(dim=-1):
        x = pad_like(x, y)
    if x.size(dim=-1) > y.size(dim=-1):
        y = pad_like(y, x)
    return x, y


## Check if a tensor has any NaN values in it.
def is_nan(x: Tensor) -> bool:
    return True if x.isnan().any().item() else False
