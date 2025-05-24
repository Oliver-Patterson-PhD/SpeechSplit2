__all__ = [
    "Experiment",
]

import datetime
import os
import time
from collections import OrderedDict
from typing import Optional, Self, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from data import AudioProcs, DatasetParser, get_loader
from models.speechsplit import InterpLnr, SpeechSplit
from util import Compute, Config, Logger, LogLevel, NanError
from util.file import path
from util.tensor import save_tensor


class Experiment(object):
    logger: Logger
    compute: Compute
    intrp: InterpLnr
    model: SpeechSplit
    optimizer: torch.optim.Optimizer
    start_time: float
    writer: SummaryWriter
    tb_prefix: str
    experiment_dir: str
    dataset: DatasetParser
    audproc: AudioProcs
    ret_item_t = Tuple[
        str,
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]

    def __init__(
        self: Self, config: Optional[Config] = None, currtime: int = int(time.time())
    ) -> None:
        config = config or Config()
        self.logger = Logger()
        self.compute = Compute()
        self.compute.print_compute()

        self.experiment_name = config.options.experiment
        self.train_models_path = config.paths.models
        self.full_models_path = config.paths.full_models
        self.model_bottleneck = config.options.bottleneck
        self.model_type = config.options.model_type
        self.resume_iters = config.options.resume_iters
        self.num_iters = config.options.num_iters
        self.dataset_name = config.options.dataset_name

        self.model = SpeechSplit(config)
        self.intrp = InterpLnr(config)
        self.model.to(self.compute.device())
        self.intrp.to(self.compute.device())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            config.training.lr,
            (config.training.beta1, config.training.beta2),
            weight_decay=1e-6,
        )
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                config.paths.tensorboard,
                config.options.model_type,
                config.options.experiment,
                str(currtime),
            )
        )
        self.tb_prefix = os.path.join(
            config.options.experiment,
            config.options.model_type,
        )
        self.experiment_dir = os.path.join(
            config.paths.artefacts,
            config.options.experiment,
        )
        self.dataset = DatasetParser(config)
        self.audproc = AudioProcs(config)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.config = config

    def tb_add_scalar(self: Self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(
            tag=f"{self.tb_prefix}/{name}",
            scalar_value=value,
            global_step=step,
        )

    def tb_add_melspec(self: Self, name: str, tensor: torch.Tensor, step: int) -> None:
        self.writer.add_image(
            tag=f"{self.tb_prefix}/melspec/{name}",
            img_tensor=tensor,
            global_step=step,
        )

    def print_model_info(self: Self) -> None:
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.logger.info(str(self.model), depth=2)
        self.logger.info(self.model_type, depth=2)
        self.logger.info("The number of parameters: {}".format(num_params), depth=2)

    def load_trained(self: Self, model_name: str) -> None:
        model_path = os.path.join(self.full_models_path, model_name)
        self.logger.info(
            f"Loading the trained model {model_path}",
            depth=2,
        )
        ckpt = torch.load(
            model_path,
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)

    def restore_model(
        self: Self,
        resume_iters: int = 0,
        model_name: Optional[str] = None,
        load_optim: bool = False,
    ) -> None:
        if resume_iters == 0:
            resume_iters = self.resume_iters
        self.logger.info(
            f"Loading the trained models from step {resume_iters}...", depth=2
        )
        name_dir = "{}-{}".format(self.model_type, self.model_bottleneck)
        name_file = "{}-{}-{}-{}.ckpt".format(
            self.experiment_name, self.model_bottleneck, self.model_type, resume_iters
        )
        save_dir = (
            self.train_models_path if resume_iters != 0 else self.full_models_path
        )
        ckpt_file = os.path.join(save_dir, name_dir, self.experiment_name, name_file)
        ckpt = torch.load(
            ckpt_file if model_name is None else model_name,
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )
        try:
            self.model.load_state_dict(ckpt["model"])
            if load_optim:
                if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                else:
                    self.logger.error("Failed to load optimizer", depth=2)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)

    def save_checkpoint(self: Self, current_iter: int, save_optim: bool = True) -> None:
        os.makedirs(self.train_models_path, exist_ok=True)
        self.logger.info(
            f"Saving model checkpoint into {self.train_models_path}...", depth=2
        )
        ckpt_name = path(
            f"{self.model_bottleneck}-{self.model_type}",
            self.experiment_name,
            f"{self.experiment_name}-{self.model_bottleneck}-{self.model_type}-{current_iter}.ckpt",
        )
        ckpt_file = os.path.join(self.train_models_path, ckpt_name)
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict() if save_optim else None,
            },
            ckpt_file,
        )

    def log_training_step(
        self: Self,
        step: int,
        loss: float,
        orig: Optional[torch.Tensor] = None,
        proc: Optional[torch.Tensor] = None,
    ) -> None:
        self.logger.info(
            "Elapsed [{}], Iteration [{}/{}], loss: {:.8f}".format(
                str(datetime.timedelta(seconds=time.time() - self.start_time))[:-7],
                f"{step : >{len(str(self.num_iters))}}",
                self.num_iters,
                loss,
            ),
            depth=2,
        )
        if orig is not None and proc is not None:
            self.tb_add_melspec(name="orig", tensor=orig, step=step)
            self.tb_add_melspec(name="proc", tensor=proc, step=step)
            self.writer.flush()

    def load_data(self: Self, **kwargs) -> None:
        self.data_loader = get_loader(self.config, **kwargs)

    def save_tensor(self: Self, tensor: torch.Tensor, fname: str) -> None:
        save_tensor(tensor, path(self.experiment_dir, fname))
        return

    def prepare_input(
        self: Self,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> torch.Tensor:
        content_pitch_input = torch.cat(
            (content_input, pitch_input), dim=-1
        )  # [B, T, F+1]
        content_pitch_input_intrp = self.intrp(
            content_pitch_input, len_crop
        )  # [B, T, F+1]
        pitch_input_intrp = self.audproc.quantize_f0(
            content_pitch_input_intrp[:, :, -1],
        )  # [B, T, 257]
        content_pitch_input_intrp_2 = torch.cat(
            (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp), dim=-1
        )  # [B, T, F+257]
        return content_pitch_input_intrp_2

    def filter_item(self: Self, item: ret_item_t) -> ret_item_t:
        (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = item
        # Move data to GPU if available
        spmel_gt = spmel_gt.to(self.compute.device())
        rhythm_input = rhythm_input.to(self.compute.device())
        content_input = content_input.to(self.compute.device())
        pitch_input = pitch_input.to(self.compute.device()).unsqueeze(-1)
        timbre_input = timbre_input.to(self.compute.device())
        len_crop = len_crop.to(self.compute.device())
        return (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        )

    def get_next_data(self: Self) -> ret_item_t:
        fname: str
        spk_id_org: str
        spmel_gt: torch.Tensor
        rhythm_input: torch.Tensor
        content_input: torch.Tensor
        pitch_input: torch.Tensor
        timbre_input: torch.Tensor
        len_crop: torch.Tensor
        if not hasattr(self, "data_iter"):
            self.data_iter = iter(self.data_loader)
        try:
            (
                fname,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = self.filter_item(next(self.data_iter))
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            (
                fname,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = self.filter_item(next(self.data_iter))
        except AssertionError as e:
            raise NanError("Found NaNs in data") from e
        return (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        )

    @torch.no_grad()
    def check_data(self: Self) -> None:
        for item in self.logger.progress_bar(
            self.data_loader, desc=f"Verifying {self.dataset_name}"
        ):
            (
                fname,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = self.filter_item(item)
            # Prepare input data and apply random resampling
            try:
                content_pitch_input = self.prepare_input(
                    content_input,
                    pitch_input,
                    len_crop,
                )
            except Exception as e:
                self.logger.trace_tensor(spmel_gt, LogLevel.ERROR)
                self.logger.trace_tensor(rhythm_input, LogLevel.ERROR)
                self.logger.trace_tensor(content_input, LogLevel.ERROR)
                self.logger.trace_tensor(pitch_input, LogLevel.ERROR)
                self.logger.trace_tensor(timbre_input, LogLevel.ERROR)
                self.logger.trace_tensor(len_crop, LogLevel.ERROR)
                self.logger.fatal(str(e.__cause__))
                raise Exception(f"Failure during check_data for {fname}") from e
            found_nan = False
            found_nan |= self.logger.log_if_nan_ret(spmel_gt)
            found_nan |= self.logger.log_if_nan_ret(spmel_gt)
            found_nan |= self.logger.log_if_nan_ret(rhythm_input)
            found_nan |= self.logger.log_if_nan_ret(content_input)
            found_nan |= self.logger.log_if_nan_ret(pitch_input)
            found_nan |= self.logger.log_if_nan_ret(timbre_input)
            found_nan |= self.logger.log_if_nan_ret(len_crop)
            found_nan |= self.logger.log_if_nan_ret(content_pitch_input)
            if found_nan:
                self.logger.error("Step has NaN loss")
                self.logger.error(f"filename: {fname}")
                raise NanError(f"{fname}")
