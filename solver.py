import datetime
import os
import time
from collections import OrderedDict

import torch
from model import Generator_3 as Generator
from model import InterpLnr
from torch.utils.tensorboard import SummaryWriter
from util.compute import Compute
from util.logging import Logger
from utils import quantize_f0_torch


class Solver(object):
    """Solver for training"""

    def __init__(self, data_loader, args, config):
        self.logger = Logger()

        # Step configuration
        self.args = args
        self.num_iters = self.args.num_iters
        self.resume_iters = self.args.resume_iters
        self.log_step = self.args.log_step
        self.ckpt_save_step = self.args.ckpt_save_step
        self.return_latents = config.options.return_latents

        # Hyperparameters
        self.config = config

        # Data loader.
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

        # Training configurations.
        self.lr = self.config.training.lr
        self.beta1 = self.config.training.beta1
        self.beta2 = self.config.training.beta2
        self.experiment = self.config.options.experiment
        self.bottleneck = self.config.options.bottleneck
        self.model_type = "G"
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            "cuda:{}".format(self.config.options.device_id) if self.use_cuda else "cpu"
        )

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
        self.logger.debug(self.model)
        self.logger.debug(self.model_type)
        self.logger.debug("The number of parameters: {}".format(num_params))
        # Set GPU
        compute = Compute()
        compute.print_compute()
        compute.set_gpu()
        self.device = compute.device()
        # gpu_count = torch.cuda.device_count()
        # if gpu_count > 1:
        #     self.model = torch.nn.DataParallel(self.model)
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
        self.logger.debug(f"Loading the trained models from step {resume_iters}...")
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

    def train(self):
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            self.logger.debug("Resuming ...")
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.logger.debug(self.optimizer)
            self.logger.debug("optimizer")

        # Learning rate cache for decaying.
        lr = self.lr
        self.logger.debug("Current learning rates, lr: {}.".format(lr))

        # Start training.
        self.logger.debug("Start training...")
        start_time = time.time()
        self.model = self.model.train()
        for i in range(start_iters, self.num_iters):

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #

            # Load data
            try:
                (
                    _,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop,
                ) = next(self.data_iter)
            except Exception:
                self.data_iter = iter(self.data_loader)
                (
                    _,
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
            content_pitch_input = torch.cat(
                (content_input, pitch_input), dim=-1
            )  # [B, T, F+1]
            content_pitch_input_intrp = self.Interp(
                content_pitch_input, len_crop
            )  # [B, T, F+1]
            pitch_input_intrp = quantize_f0_torch(content_pitch_input_intrp[:, :, -1])[
                0
            ]  # [B, T, 257]
            content_pitch_input_intrp = torch.cat(
                # [B, T, F+257]
                (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp),
                dim=-1,
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
                    content_pitch_input_intrp,
                    rhythm_input,
                    timbre_input,
                )
            else:
                spmel_output = self.model(
                    content_pitch_input_intrp,
                    rhythm_input,
                    timbre_input,
                )
            loss_id = torch.torch.nn.functional.mse_loss(spmel_output, spmel_gt)

            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # Logging.
            train_loss_id = loss_id.item()
            self.writer.add_scalar(
                f"{self.experiment}/{self.model_type}/train_loss_id", loss, i
            )

            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                self.logger.debug(
                    "Elapsed [{}], Iteration [{}/{}], {}/train_loss_id: {:.8f}".format(
                        et,
                        i + 1,
                        self.num_iters,
                        self.model_type,
                        train_loss_id,
                    )
                )

            # Save model checkpoints
            if (i + 1) % self.ckpt_save_step == 0:
                ckpt_name = "{}-{}-{}-{}.ckpt".format(
                    self.experiment,
                    self.bottleneck,
                    self.model_type,
                    i + 1,
                )
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_save_dir, ckpt_name),
                )
                self.logger.debug(
                    f"Saving model checkpoint into {self.model_save_dir}..."
                )
                self.writer.flush()
