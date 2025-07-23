import time

import torch

from ..util import NanError, compute, config, logger
from ..util.tensor import Tensor
from .experiment import Experiment


def masked_mse(prediction: Tensor, ground_t: Tensor) -> Tensor:
    prediction = prediction.flatten()
    ground_t = ground_t.flatten()
    mask: Tensor = ground_t != 0.0
    sum: Tensor = torch.nn.functional.mse_loss(
        prediction,
        ground_t,
        reduction="sum",
    )
    return sum / mask.sum()


## Solver for training
class Train(Experiment):
    def train(self) -> None:
        # Start training from scratch or resume training.
        compute.set_gpu()
        self.load_data()
        start_iters = 0
        if config.options.resume_iters:
            logger.info("Resuming ...")
            start_iters = config.options.resume_iters
            config.options.num_iters += config.options.resume_iters
            self.restore_model(config.options.resume_iters)
            logger.info(str(self.optimizer))
            logger.info("optimizer")

        # Learning rate cache for decaying.
        lr = config.training.lr
        logger.info("Current learning rates, lr: {}.".format(lr))

        # Start training.
        self.model.train()
        self.intrp.train()
        self.start_time = time.time()
        if config.training.mask_loss:
            self.loss_fn = masked_mse
        else:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")

        if __debug__:
            self.check_data()

        i = start_iters
        logger.manual_pbar_start(
            total=config.options.num_iters,
            initial=i,
        )
        logger.info("Start training...")
        while i <= config.options.num_iters:
            fname: str
            spk_id_org: str
            spmel_gt: Tensor
            rhythm_input: Tensor
            content_input: Tensor
            pitch_input: Tensor
            timbre_input: Tensor
            len_crop: Tensor
            spmel_output: Tensor
            code_exp_1: Tensor
            code_exp_2: Tensor
            code_exp_3: Tensor
            code_exp_4: Tensor

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #
            # Load data
            (
                fname,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = self.get_next_data()

            # =============================================================== #
            #                   2. Train the model                            #
            # =============================================================== #
            # Prepare input data and apply random resampling
            logger.trace_tensor(spmel_gt)
            logger.trace_tensor(rhythm_input)
            logger.trace_tensor(content_input)
            logger.trace_tensor(pitch_input)
            logger.trace_tensor(timbre_input)
            logger.trace_tensor(len_crop)
            content_pitch_input = self.prepare_input(
                content_input,
                pitch_input,
                len_crop,
            )

            # Identity mapping loss
            if config.options.return_latents:
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

            loss_id: Tensor
            loss_id = self.loss_fn(spmel_output, spmel_gt)

            # Backward and optimize.
            loss: Tensor = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            train_loss_id: float = loss_id.item()

            i += 1
            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #
            logger.manual_pbar_update()
            # Save model checkpoints
            if i % config.options.ckpt_save_step == 0:
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

            self.tb_add_scalar(name="train_loss_id", value=train_loss_id, step=i)

            # Print out training information.
            if i % config.options.log_step == 0:
                self.log_training_step(
                    step=i,
                    loss=train_loss_id,
                    # orig=spmel_gt,
                    # proc=spmel_output,
                )

            if __debug__:
                found_nan = False
                found_nan |= logger.log_if_nan_ret(loss)
                found_nan |= logger.log_if_nan_ret(loss_id)
                found_nan |= logger.log_if_nan_ret(spmel_output)
                if config.options.return_latents:
                    found_nan |= logger.log_if_nan_ret(code_exp_1)
                    found_nan |= logger.log_if_nan_ret(code_exp_2)
                    found_nan |= logger.log_if_nan_ret(code_exp_3)
                    found_nan |= logger.log_if_nan_ret(code_exp_4)
                if found_nan:
                    self.log_training_step(i, train_loss_id)
                    logger.error("Step has NaN loss")
                    logger.error(f"filename: {fname}")
                    logger.error(f"tensor: {spmel_gt.any()}")
                    self.writer.flush()
                    logger.manual_pbar_end()
                    raise NanError(f"{fname}")
        logger.manual_pbar_end()
