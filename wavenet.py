from typing import List, Optional

import torch
from tqdm import tqdm
from wavenet_vocoder import builder


# Default hyperparameters:
class wavenet_hparams:
    name: str = "wavenet_vocoder"
    builder: str = "wavenet"
    input_type: str = "raw"
    quantize_channels: int = 65536  # 65536 or 256

    # Audio:
    sample_rate: int = 16000  # This is different to the json
    silence_threshold: int = 2  # this is only valid for mulaw is True
    num_mels: int = 80
    fmin: int = 125
    fmax: int = 7600
    fft_size: int = 1024
    hop_size: int = 256  # shift can be specified by either hop_size or frame_shift_ms
    frame_shift_ms: Optional[int] = None
    min_level_db: int = -100
    ref_level_db: int = 20
    rescaling: bool = True  # whether to rescale waveform or not.
    rescaling_max: float = 0.999
    allow_clipping_in_normalization: bool = True
    log_scale_min: float = float(-32.23619130191664)  # Mixture of log distributions:

    # Model:
    out_channels: int = 30  # Equal to `quantize_channels` if mu-law quantize enabled
    layers: int = 24
    stacks: int = 4
    residual_channels: int = 512
    gate_channels: int = 512  # split in 2 groups internally for gated activation
    skip_out_channels: int = 256
    dropout: float = 1 - 0.95
    kernel_size: int = 3
    weight_normalization: bool = True
    legacy: bool = True
    cin_channels: int = 80  # Local conditioning (set negative value to disable)
    upsample_conditional_features: bool = True
    upsample_scales: List[int] = [4, 4, 4, 4]
    freq_axis_kernel_size: int = 3  # Freq axis kernel size for upsampling network
    # Global conditioning (set negative value to disable)
    # enabled for multi-speaker dataset
    gin_channels: int = -1
    n_speakers: int = -1

    # Data loader
    pin_memory: bool = True
    num_workers: int = 2
    # train/test
    test_size: float = 0.0441
    test_num_samples: Optional[int] = None
    random_state: int = 1234

    # Training:
    batch_size: int = 2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    amsgrad: bool = False
    initial_learning_rate: float = 1e-3
    lr_schedule: str = "noam_learning_rate_decay"  # see lrschedule.py
    lr_schedule_kwargs: dict = {}
    nepochs: int = 2000
    weight_decay: float = 0.0
    clip_thresh: int = -1
    max_time_sec: Optional[int] = None
    max_time_steps: int = 8000
    exponential_moving_average: bool = True  # use moving average params for evaluation
    ema_decay: float = 0.9999  # averaged = decay * averaged + (1 - decay) * x

    # Save
    checkpoint_interval: int = 10000  # per-step
    train_eval_interval: int = 10000  # per-step
    test_eval_epoch_interval: int = 5  # per-epoch
    save_optimizer_state: bool = True


class Synthesizer(object):

    def __init__(self, device):

        self.model = getattr(builder, wavenet_hparams.builder)(
            out_channels=wavenet_hparams.out_channels,
            layers=wavenet_hparams.layers,
            stacks=wavenet_hparams.stacks,
            residual_channels=wavenet_hparams.residual_channels,
            gate_channels=wavenet_hparams.gate_channels,
            skip_out_channels=wavenet_hparams.skip_out_channels,
            cin_channels=wavenet_hparams.cin_channels,
            gin_channels=wavenet_hparams.gin_channels,
            weight_normalization=wavenet_hparams.weight_normalization,
            n_speakers=wavenet_hparams.n_speakers,
            dropout=wavenet_hparams.dropout,
            kernel_size=wavenet_hparams.kernel_size,
            upsample_conditional_features=wavenet_hparams.upsample_conditional_features,
            upsample_scales=wavenet_hparams.upsample_scales,
            freq_axis_kernel_size=wavenet_hparams.freq_axis_kernel_size,
            scalar_input=True,
            legacy=wavenet_hparams.legacy,
        )
        self.device = device

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(
            ckpt_path,
            weights_only=False,
        )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])

    def spect2wav(self, c=None, tqdm=tqdm):
        self.model.eval()
        self.model.make_generation_fast_()

        Tc = c.shape[0]
        upsample_factor = wavenet_hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)
        initial_input = torch.zeros(1, 1, 1).fill_(0.0)
        # Transform data to GPU
        initial_input = initial_input.to(self.device)
        c = None if c is None else c.to(self.device)

        with torch.no_grad():
            y_hat = self.model.incremental_forward(
                initial_input,
                c=c,
                g=None,
                T=length,
                tqdm=tqdm,
                softmax=True,
                quantize=True,
                log_scale_min=wavenet_hparams.log_scale_min,
            )
        y_hat = y_hat.view(-1)
        return y_hat
