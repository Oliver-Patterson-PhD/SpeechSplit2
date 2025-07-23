__all__ = [
    "Experiment",
]

import datetime
import os
import time
from collections import OrderedDict
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from ..data import AudioProcs, DatasetParser, get_loader
from ..models.speechsplit import InterpLnr, SpeechSplit
from ..util import NanError, compute, config, logger
from ..util.file import path
from ..util.tensor import save_tensor


class Experiment(object):
    intrp: InterpLnr
    model: SpeechSplit
    optimizer: torch.optim.Optimizer
    start_time: float
    writer: SummaryWriter
    tb_prefix: str
    experiment_dir: str
    parser: DatasetParser
    audproc: AudioProcs
    ret_item_t = tuple[
        str,
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]

    def __init__(self, currtime: int = int(time.time())) -> None:
        compute.print_compute()
        self.experiment_name = config.options.experiment
        self.train_models_path = config.paths.models
        self.full_models_path = config.paths.full_models
        self.model_bottleneck = config.options.bottleneck
        self.model_type = config.options.model_type
        self.resume_iters = config.options.resume_iters
        self.num_iters = config.options.num_iters
        self.dataset_name = config.options.dataset_name

        self.model = SpeechSplit()
        self.intrp = InterpLnr()
        self.model.to(compute.device())
        self.intrp.to(compute.device())
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
        self.parser = DatasetParser()
        self.audproc = AudioProcs()
        os.makedirs(self.experiment_dir, exist_ok=True)

    def tb_add_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(  # type: ignore
            tag=f"{self.tb_prefix}/{name}",
            scalar_value=value,
            global_step=step,
        )

    def tb_add_melspec(self, name: str, tensor: torch.Tensor, step: int) -> None:
        self.writer.add_image(  # type: ignore
            tag=f"{self.tb_prefix}/melspec/{name}",
            img_tensor=tensor,
            global_step=step,
        )

    def print_model_info(self) -> None:
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        logger.info(str(self.model), depth=2)
        logger.info(self.model_type, depth=2)
        logger.info("The number of parameters: {}".format(num_params), depth=2)

    def load_trained(self, model_name: str) -> None:
        model_path = os.path.join(self.full_models_path, model_name)
        logger.info(
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
            new_state_dict: OrderedDict[str, Any] = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)

    def restore_model(
        self,
        resume_iters: int = 0,
        model_name: str | None = None,
        load_optim: bool = False,
    ) -> None:
        if resume_iters == 0:
            resume_iters = self.resume_iters
        logger.info(f"Loading the trained models from step {resume_iters}...", depth=2)
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
                    logger.error("Failed to load optimizer", depth=2)
        except RuntimeError:
            new_state_dict: OrderedDict[str, Any] = OrderedDict()
            for k, v in ckpt["model"].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)

    def save_checkpoint(self, current_iter: int, save_optim: bool = True) -> None:
        os.makedirs(self.train_models_path, exist_ok=True)
        logger.info(
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
        self,
        step: int,
        loss: float,
        orig: torch.Tensor | None = None,
        proc: torch.Tensor | None = None,
    ) -> None:
        logger.info(
            "Elapsed [{}], Iteration [{}/{}], loss: {:.8f}".format(
                str(datetime.timedelta(seconds=time.time() - self.start_time))[:-7],
                f"{step: >{len(str(self.num_iters))}}",
                self.num_iters,
                loss,
            ),
            depth=2,
        )
        if orig is not None and proc is not None:
            self.tb_add_melspec(name="orig", tensor=orig, step=step)
            self.tb_add_melspec(name="proc", tensor=proc, step=step)
            self.writer.flush()

    def load_data(self, **kwargs: bool) -> None:
        self.data_loader = get_loader(**kwargs)

    def save_tensor(
        self, tensor: torch.Tensor, fname: str, save_raw: bool = False
    ) -> None:
        save_tensor(tensor, path(self.experiment_dir, fname), save_raw)
        return

    def prepare_input(
        self,
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

    def filter_item(self, item: ret_item_t) -> ret_item_t:
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
        spmel_gt = spmel_gt.to(compute.device())
        rhythm_input = rhythm_input.to(compute.device())
        content_input = content_input.to(compute.device())
        pitch_input = pitch_input.to(compute.device()).unsqueeze(-1)
        timbre_input = timbre_input.to(compute.device())
        len_crop = len_crop.to(compute.device())
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

    def get_next_data(self) -> ret_item_t:
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

    @torch.no_grad()  # type: ignore
    def check_data(self) -> None:
        for item in logger.progress_bar(
            self.data_loader, desc=f"Verifying {self.dataset_name}"
        ):
            (
                fname,
                _,  # spk_id_org
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
                logger.trace_tensor(spmel_gt, "ERROR")
                logger.trace_tensor(rhythm_input, "ERROR")
                logger.trace_tensor(content_input, "ERROR")
                logger.trace_tensor(pitch_input, "ERROR")
                logger.trace_tensor(timbre_input, "ERROR")
                logger.trace_tensor(len_crop, "ERROR")
                logger.fatal(str(e.__cause__))
                raise Exception(f"Failure during check_data for {fname}") from e
            found_nan = False
            found_nan |= logger.log_if_nan_ret(spmel_gt)
            found_nan |= logger.log_if_nan_ret(spmel_gt)
            found_nan |= logger.log_if_nan_ret(rhythm_input)
            found_nan |= logger.log_if_nan_ret(content_input)
            found_nan |= logger.log_if_nan_ret(pitch_input)
            found_nan |= logger.log_if_nan_ret(timbre_input)
            found_nan |= logger.log_if_nan_ret(len_crop)
            found_nan |= logger.log_if_nan_ret(content_pitch_input)
            if found_nan:
                logger.error("Step has NaN loss")
                logger.error(f"filename: {fname}")
                raise NanError(f"{fname}")
