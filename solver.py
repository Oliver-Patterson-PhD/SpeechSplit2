import datetime
import math
import os
import time
from collections import OrderedDict

import torch
from data_loader import get_loader
from model import Generator_3 as Generator
from model import InterpLnr
from torch.utils.tensorboard import SummaryWriter
from util.compute import Compute
from util.logging import Logger
from utils import is_nan, quantize_f0_torch


class Solver(object):
    """Solver for training"""

    def __init__(self, config):
        self.logger = Logger()

        # Step configuration
        self.num_iters = config.options.num_iters
        self.resume_iters = config.options.resume_iters
        self.log_step = config.options.log_step
        self.ckpt_save_step = config.options.ckpt_save_step
        self.return_latents = config.options.return_latents

        # Hyperparameters
        self.config = config

        # Data loader.
        self.data_loader = get_loader(config)
        self.data_iter = iter(self.data_loader)

        # Training configurations.
        self.lr = self.config.training.lr
        self.beta1 = self.config.training.beta1
        self.beta2 = self.config.training.beta2
        self.experiment = self.config.options.experiment
        self.bottleneck = self.config.options.bottleneck
        self.model_type = "G"

        compute = Compute()
        compute.print_compute()
        compute.set_gpu()
        self.device = compute.device()

        # Directories.
        self.model_save_dir = self.config.paths.models
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Build the model.
        self.build_model()

        # Logging
        self.min_loss_step = 0
        self.min_loss = float("inf")
        self.writer_pref = f"{self.experiment}/{self.model_type}"
        self.writer = SummaryWriter(log_dir=f"tensorboard/{self.writer_pref}")

    def build_model(self):
        self.model = Generator(self.config)
        # Print out the network information.
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.logger.info(self.model)
        self.logger.info(self.model_type)
        self.logger.info("The number of parameters: {}".format(num_params))
        self.model.to(self.device)

        self.Interp = InterpLnr(self.config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            [self.beta1, self.beta2],
            weight_decay=1e-6,
        )
        self.Interp.to(self.device)

    def restore_model(self, resume_iters):
        self.logger.info(f"Loading the trained models from step {resume_iters}...")
        ckpt_name = "{}-{}-{}-{}.ckpt".format(
            self.experiment,
            self.bottleneck,
            self.model_type,
            resume_iters,
        )
        ckpt = torch.load(
            os.path.join(self.model_save_dir, ckpt_name),
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
        self.lr = self.optimizer.param_groups[0]["lr"]

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
                self.num_iters,
                self.model_type,
                train_loss_id,
            )
        )

    def save_checkpoint(self, i: int) -> None:
        ckpt_name = "{}-{}-{}-{}.ckpt".format(
            self.experiment,
            self.bottleneck,
            self.model_type,
            i,
        )
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.model_save_dir, ckpt_name),
        )
        self.logger.info(f"Saving model checkpoint into {self.model_save_dir}...")

    def prepare_input(
        self,
        content_input: torch.Tensor,
        pitch_input: torch.Tensor,
        len_crop: torch.Tensor,
    ) -> torch.Tensor:
        content_pitch_input = torch.cat(
            (content_input, pitch_input), dim=-1
        )  # [B, T, F+1]
        content_pitch_input_intrp = self.Interp(
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

    def train(self):
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            self.logger.info("Resuming ...")
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.logger.info(self.optimizer)
            self.logger.info("optimizer")

        # Learning rate cache for decaying.
        lr = self.lr
        self.logger.info("Current learning rates, lr: {}.".format(lr))

        # Start training.
        self.logger.info("Start training...")
        self.start_time = time.time()
        self.model = self.model.train()
        for i in range(start_iters, self.num_iters):

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

            # =============================================================== #
            #                   2. Train the model                            #
            # =============================================================== #

            # Move data to GPU if available
            spmel_gt = spmel_gt.to(self.device)
            rhythm_input = rhythm_input.to(self.device)
            content_input = content_input.to(self.device)
            pitch_input = pitch_input.to(self.device)
            timbre_input = timbre_input.to(self.device)
            len_crop = len_crop.to(self.device)

            # Prepare input data and apply random resampling
            content_pitch_input = self.prepare_input(
                content_input,
                pitch_input,
                len_crop,
            )

            # Identity mapping loss
            if self.return_latents:
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

            loss_id = torch.torch.nn.functional.mse_loss(spmel_output, spmel_gt)

            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            train_loss_id = loss_id.item()

            if is_nan(loss_id) or math.isnan(train_loss_id):
                self.log_training_step(i + 1, train_loss_id)
                self.logger.trace_nans(spmel_gt)
                self.logger.trace_var(spmel_gt)
                self.logger.trace_nans(spmel_output)
                self.logger.trace_var(spmel_output)

                self.logger.trace_nans(spmel_gt)
                self.logger.trace_nans(rhythm_input)
                self.logger.trace_nans(content_input)
                self.logger.trace_nans(pitch_input)
                self.logger.trace_nans(timbre_input)
                self.logger.trace_nans(len_crop)
                self.logger.trace_nans(content_pitch_input)
                self.logger.trace_nans(spmel_output)
                if self.return_latents:
                    self.logger.trace_nans(code_exp_1)
                    self.logger.trace_nans(code_exp_2)
                    self.logger.trace_nans(code_exp_3)
                    self.logger.trace_nans(code_exp_4)
                self.logger.fatal("Step has NaN loss")

            self.writer.add_scalar(
                f"{self.experiment}/{self.model_type}/train_loss_id",
                train_loss_id,
                i,
            )

            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                self.log_training_step(i + 1, train_loss_id)

            # Save model checkpoints
            if (i + 1) % self.ckpt_save_step == 0:
                self.save_checkpoint(i + 1)
                self.writer.flush()
