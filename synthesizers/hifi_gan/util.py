from dataclasses import dataclass
from typing import List

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(model, mean: float = 0.0, std: float = 0.01) -> None:
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        model.weight.data.normal_(mean, std)


@dataclass
class HiFiConfig:
    resblock: int
    num_gpus: int
    batch_size: int
    learning_rate: float
    adam_b1: float
    adam_b2: float
    lr_decay: float
    seed: int
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    segment_size: int
    num_mels: int
    num_freq: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fmax_for_loss: None
    num_workers: int
