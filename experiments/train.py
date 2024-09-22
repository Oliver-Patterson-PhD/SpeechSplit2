import time
from typing import Self

import torch
from experiments.experiment import Experiment
from util.exception import NanError
from utils import quantize_f0_torch


## Solver for training
class Train(Experiment):
    def train(self: Self) -> None:
        # Start training from scratch or resume training.
        self.load_data()
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
        self.data_iter = iter(self.data_loader)

        # Start training.
        self.model.train()
        self.intrp.train()
        self.logger.info("Start training...")
        self.start_time = time.time()
        self.model = self.model.train()
        self.intrp = self.intrp.train()

        i = start_iters
        while i <= self.config.options.num_iters:
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
            content_pitch_input_intrp: torch.Tensor = self.intrp(
                torch.cat((content_input, pitch_input), dim=-1), len_crop
            )  # [B, T, F+1]
            content_pitch_input = torch.cat(
                (
                    content_pitch_input_intrp[:, :, :-1],
                    quantize_f0_torch(
                        content_pitch_input_intrp[:, :, -1],
                    ),
                ),
                dim=-1,
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

            loss_mask = spmel_gt != 0.0
            loss_id: torch.Tensor = torch.torch.nn.functional.mse_loss(
                spmel_output.masked_select(loss_mask), spmel_gt.masked_select(loss_mask)
            )

            # Backward and optimize.
            loss: torch.Tensor = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            train_loss_id: float = loss_id.item()

            i += 1
            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #

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
                    self.log_training_step(i, train_loss_id)
                    self.logger.error("Step has NaN loss")
                    self.logger.error(f"filename: {fname}")
                    self.writer.flush()
                    raise NanError(f"{fname}")

            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #

            # Add loss to tensorboard
            self.tb_add_scalar(name="train_loss_id", value=train_loss_id, step=i)

            # Print out training information.
            if i % self.config.options.log_step == 0:
                self.log_training_step(
                    step=i,
                    loss=train_loss_id,
                    orig=spmel_gt,
                    proc=spmel_output,
                )

            # Save model checkpoints
            if i % self.config.options.ckpt_save_step == 0:
                self.save_checkpoint(i)
                self.writer.add_graph(
                    model=self.model,
                    input_to_model=(
                        content_pitch_input,
                        rhythm_input,
                        timbre_input,
                    ),
                )
                self.writer.flush()
