import datetime
import os
import time
from collections import OrderedDict
from typing import Optional, Self

import torch
from torch.utils.tensorboard import SummaryWriter

from data.loader import get_loader
from model import InterpLnr, SpeechSplit
from util import Compute, Config, Logger, LogLevel, NanError
from utils import quantize_f0_torch, save_tensor


class Experiment(object):
    logger: Logger
    compute: Compute
    config: Config
    intrp: InterpLnr
    model: SpeechSplit
    optimizer: torch.optim.Optimizer
    start_time: float
    writer: SummaryWriter
    tb_prefix: str
    experiment_dir: str

    def __init__(self: Self, config: Config, currtime: int = int(time.time())) -> None:
        self.config = config
        self.logger = Logger()
        self.compute = Compute()
        self.compute.print_compute()
        self.model = SpeechSplit(self.config)
        self.intrp = InterpLnr(self.config)
        self.model.to(self.compute.device())
        self.intrp.to(self.compute.device())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.training.lr,
            (self.config.training.beta1, self.config.training.beta2),
            weight_decay=1e-6,
        )
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.config.paths.tensorboard,
                self.config.options.model_type,
                self.config.options.experiment,
                str(currtime),
            )
        )
        self.tb_prefix = os.path.join(
            self.config.options.experiment,
            self.config.options.model_type,
        )
        self.experiment_dir = os.path.join(
            self.config.paths.artefacts,
            self.config.options.experiment,
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

    def tb_add_scalar(
        self: Self,
        name: str,
        value: float,
        step: int,
    ) -> None:
        self.writer.add_scalar(
            tag=f"{self.tb_prefix}/{name}",
            scalar_value=value,
            global_step=step,
        )

    def tb_add_melspec(
        self: Self,
        name: str,
        tensor: torch.Tensor,
        step: int,
    ) -> None:
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
        self.logger.info(self.config.options.model_type, depth=2)
        self.logger.info("The number of parameters: {}".format(num_params), depth=2)

    def load_trained(
        self: Self,
        model_name: str,
    ) -> None:
        model_path = os.path.join(self.config.paths.full_models, model_name)
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
        self.config.training.lr = self.optimizer.param_groups[0]["lr"]

    def restore_model(
        self: Self,
        resume_iters: int = 0,
        model_name: Optional[str] = None,
        load_optim: bool = False,
    ) -> None:
        if resume_iters == 0:
            resume_iters = self.config.options.resume_iters
        self.logger.info(
            f"Loading the trained models from step {resume_iters}...",
            depth=2,
        )
        name_dir = "{}-{}".format(
            self.config.options.model_type,
            self.config.options.bottleneck,
        )
        name_file = "{}-{}-{}-{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            resume_iters,
        )
        save_dir = (
            self.config.paths.models
            if resume_iters != 0
            else self.config.paths.full_models
        )
        ckpt_file = os.path.join(
            save_dir,
            name_dir,
            self.config.options.experiment,
            name_file,
        )
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
        self.config.training.lr = self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(
        self: Self,
        current_iter: int,
        save_optim: bool = True,
    ) -> None:
        os.makedirs(self.config.paths.models, exist_ok=True)
        self.logger.info(
            f"Saving model checkpoint into {self.config.paths.models}...",
            depth=2,
        )
        ckpt_name = "{2}-{1}/{0}/{0}-{1}-{2}-{3}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            current_iter,
        )
        ckpt_file = os.path.join(self.config.paths.models, ckpt_name)
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
    ):
        self.logger.info(
            "Elapsed [{}], Iteration [{}/{}], loss: {:.8f}".format(
                str(datetime.timedelta(seconds=time.time() - self.start_time))[:-7],
                f"{step : >{len(str(self.config.options.num_iters))}}",
                self.config.options.num_iters,
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
        self.data_iter = iter(self.data_loader)

    def save_tensor(self: Self, tensor: torch.Tensor, fname: str) -> None:
        save_tensor(
            tensor,
            "{}/{}".format(
                self.experiment_dir,
                fname,
            ),
        )
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
        pitch_input_intrp = quantize_f0_torch(
            content_pitch_input_intrp[:, :, -1],
        )  # [B, T, 257]
        content_pitch_input_intrp_2 = torch.cat(
            # [B, T, F+257]
            (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp),
            dim=-1,
        )
        return content_pitch_input_intrp_2

    def get_next_data(self: Self):
        fname: str
        spk_id_org: str
        spmel_gt: torch.Tensor
        rhythm_input: torch.Tensor
        content_input: torch.Tensor
        pitch_input: torch.Tensor
        timbre_input: torch.Tensor
        len_crop: torch.Tensor
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
            ) = next(self.data_iter)
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
            ) = next(self.data_iter)
        except AssertionError as e:
            raise NanError(e)
        finally:
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

    @torch.no_grad()
    def check_data(self: Self) -> None:
        for item in self.logger.progress_bar(
            self.data_loader,
            desc=f"Verifying {self.config.options.dataset_name}",
        ):
            (
                i_fname,
                i_spk_id_org,
                i_spmel_gt,
                i_rhythm_input,
                i_content_input,
                i_pitch_input,
                i_timbre_input,
                i_len_crop,
            ) = item
            i_spmel_gt = i_spmel_gt.to(self.compute.device())
            i_rhythm_input = i_rhythm_input.to(self.compute.device())
            i_content_input = i_content_input.to(self.compute.device())
            i_pitch_input = i_pitch_input.to(self.compute.device())
            i_timbre_input = i_timbre_input.to(self.compute.device())
            i_len_crop = i_len_crop.to(self.compute.device())
            # Prepare input data and apply random resampling
            try:
                i_content_pitch_input = self.prepare_input(
                    i_content_input,
                    i_pitch_input,
                    i_len_crop,
                )
            except Exception as e:
                self.logger.trace_tensor(i_spmel_gt, LogLevel.ERROR)
                self.logger.trace_tensor(i_rhythm_input, LogLevel.ERROR)
                self.logger.trace_tensor(i_content_input, LogLevel.ERROR)
                self.logger.trace_tensor(i_pitch_input, LogLevel.ERROR)
                self.logger.trace_tensor(i_timbre_input, LogLevel.ERROR)
                self.logger.trace_tensor(i_len_crop, LogLevel.ERROR)
                self.logger.fatal(str(e))
            found_nan = False
            found_nan |= self.logger.log_if_nan_ret(i_spmel_gt)
            found_nan |= self.logger.log_if_nan_ret(i_spmel_gt)
            found_nan |= self.logger.log_if_nan_ret(i_rhythm_input)
            found_nan |= self.logger.log_if_nan_ret(i_content_input)
            found_nan |= self.logger.log_if_nan_ret(i_pitch_input)
            found_nan |= self.logger.log_if_nan_ret(i_timbre_input)
            found_nan |= self.logger.log_if_nan_ret(i_len_crop)
            found_nan |= self.logger.log_if_nan_ret(i_content_pitch_input)
            if found_nan:
                self.logger.error("Step has NaN loss")
                self.logger.error(f"filename: {i_fname}")
                raise NanError(f"{i_fname}")
        return
