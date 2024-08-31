import datetime
import os
import time
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_loader
from model import Generator_3 as Generator
from model import InterpLnr
from util.compute import Compute
from util.config import Config
from util.exception import NanError
from util.logging import Logger
from utils import quantize_f0_torch


## Solver for training
class Solver(object):
    logger = Logger()
    compute = Compute()

    def __init__(self, config: Config) -> None:
        self.config = config

        # Data loader.
        self.data_loader = get_loader(config, singleitem=True)
        self.data_iter = iter(self.data_loader)

        self.compute.print_compute()
        self.compute.set_gpu()
        self.build_model()

        # Logging
        self.min_loss_step = 0
        self.min_loss = float("inf")
        self.writer_pref = "{}/{}".format(
            self.config.options.experiment, self.config.options.model_type
        )
        self.writer = SummaryWriter(log_dir=f"tensorboard/{self.writer_pref}")

    def build_model(self) -> None:
        self.model = Generator(self.config)
        self.intrp = InterpLnr(self.config)
        # Print out the network information.
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.model.to(self.compute.device())
        self.intrp.to(self.compute.device())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.training.lr,
            (self.config.training.beta1, self.config.training.beta2),
            weight_decay=1e-6,
        )
        self.logger.info(str(self.model))
        self.logger.info(self.config.options.model_type)
        self.logger.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters: int) -> None:
        self.logger.info(f"Loading the trained models from step {resume_iters}...")
        ckpt_name = "{}-{}-{}{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            f"-{resume_iters}" if resume_iters > 0 else "",
        )
        ckpt = torch.load(
            os.path.join(self.config.paths.models, ckpt_name),
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

    def log_training_step(self, i: int, train_loss_id):
        elapsed_time = str(
            datetime.timedelta(
                seconds=time.time() - self.start_time,
            )
        )[:-7]
        self.logger.info(
            "Elapsed [{}], Iteration [{}/{}], {}/train_loss_id: {:.8f}".format(
                elapsed_time,
                i,
                self.config.options.num_iters,
                self.config.options.model_type,
                train_loss_id,
            )
        )

    def save_checkpoint(self, i: int) -> None:
        os.makedirs(self.config.paths.models, exist_ok=True)
        ckpt_name = "{}-{}-{}-{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            i,
        )
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.config.paths.models, ckpt_name),
        )
        self.logger.info(f"Saving model checkpoint into {self.config.paths.models}...")

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
        pitch_input_intrp = quantize_f0_torch(
            content_pitch_input_intrp[:, :, -1],
        )  # [B, T, 257]
        content_pitch_input_intrp_2 = torch.cat(
            # [B, T, F+257]
            (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp),
            dim=-1,
        )
        return content_pitch_input_intrp_2

    def train(self) -> None:
        # Start training from scratch or resume training.
        start_iters = 0
        if self.config.options.resume_iters:
            self.logger.info("Resuming ...")
            start_iters = self.config.options.resume_iters
            self.config.options.num_iters += self.config.options.resume_iters
            self.restore_model(self.config.options.resume_iters)
            self.logger.info(str(self.optimizer))
            self.logger.info("optimizer")

        # Learning rate cache for decaying.
        lr = self.config.training.lr
        self.logger.info("Current learning rates, lr: {}.".format(lr))

        # Start training.
        self.logger.info("Start training...")
        self.start_time = time.time()
        self.model = self.model.train()
        self.intrp = self.intrp.train()

        for i in range(start_iters, self.config.options.num_iters):
            fname: str
            spk_id_org: str
            spmel_gt: torch.Tensor
            rhythm_input: torch.Tensor
            content_input: torch.Tensor
            pitch_input: torch.Tensor
            timbre_input: torch.Tensor
            len_crop: torch.Tensor
            spmel_output: torch.Tensor
            code_exp_1: torch.Tensor
            code_exp_2: torch.Tensor
            code_exp_3: torch.Tensor
            code_exp_4: torch.Tensor

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #
            # Load data
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

            # =============================================================== #
            #                   2. Train the model                            #
            # =============================================================== #

            # Move data to GPU if available
            spmel_gt = spmel_gt.to(self.compute.device())
            rhythm_input = rhythm_input.to(self.compute.device())
            content_input = content_input.to(self.compute.device())
            pitch_input = pitch_input.to(self.compute.device())
            timbre_input = timbre_input.to(self.compute.device())
            len_crop = len_crop.to(self.compute.device())

            # Prepare input data and apply random resampling
            content_pitch_input = self.prepare_input(
                content_input,
                pitch_input,
                len_crop,
            )

            # Identity mapping loss
            if self.config.options.return_latents:
                (
                    spmel_output,
                    code_exp_1,
                    code_exp_2,
                    code_exp_3,
                    code_exp_4,
                ) = self.model(
                    content_pitch_input,
                    rhythm_input,
                    timbre_input,
                )
            else:
                spmel_output = self.model(
                    content_pitch_input,
                    rhythm_input,
                    timbre_input,
                )

            loss_id: torch.Tensor = torch.torch.nn.functional.mse_loss(
                spmel_output, spmel_gt
            )

            # Backward and optimize.
            loss: torch.Tensor = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            train_loss_id: float = loss_id.item()

            if __debug__:
                found_nan = False
                found_nan |= self.logger.log_if_nan_ret(loss)
                found_nan |= self.logger.log_if_nan_ret(loss_id)
                found_nan |= self.logger.log_if_nan_ret(spmel_gt)
                found_nan |= self.logger.log_if_nan_ret(spmel_output)
                found_nan |= self.logger.log_if_nan_ret(spmel_gt)
                found_nan |= self.logger.log_if_nan_ret(rhythm_input)
                found_nan |= self.logger.log_if_nan_ret(content_input)
                found_nan |= self.logger.log_if_nan_ret(pitch_input)
                found_nan |= self.logger.log_if_nan_ret(timbre_input)
                found_nan |= self.logger.log_if_nan_ret(len_crop)
                found_nan |= self.logger.log_if_nan_ret(content_pitch_input)
                found_nan |= self.logger.log_if_nan_ret(spmel_output)
                if self.config.options.return_latents:
                    found_nan |= self.logger.log_if_nan_ret(code_exp_1)
                    found_nan |= self.logger.log_if_nan_ret(code_exp_2)
                    found_nan |= self.logger.log_if_nan_ret(code_exp_3)
                    found_nan |= self.logger.log_if_nan_ret(code_exp_4)

                if found_nan:
                    self.log_training_step(i + 1, train_loss_id)
                    self.logger.error("Step has NaN loss")
                    self.logger.error(f"filename: {fname}")
                    self.writer.flush()
                    raise NanError

            self.writer.add_scalar(
                f"{self.config.options.experiment}/{self.config.options.model_type}/train_loss_id",
                train_loss_id,
                i,
            )

            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #

            # Print out training information.
            if (i + 1) % self.config.options.log_step == 0:
                self.log_training_step(i + 1, train_loss_id)

            # Save model checkpoints
            if (i + 1) % self.config.options.ckpt_save_step == 0:
                self.save_checkpoint(i + 1)
                self.writer.flush()
