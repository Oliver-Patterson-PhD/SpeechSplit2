import time
import traceback
from typing import Self

import torch

from experiments.experiment import Experiment
from utils import masked_mse, quantize_f0_torch


class Scratchpad(Experiment):
    def run(self: Self) -> None:
        self.load_data()
        self.model.train()
        self.intrp.train()
        if self.config.training.mask_loss:
            self.loss_fn = masked_mse
        else:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.start_time = time.time()
        self.batch_size = self.config.dataloader.batch_size

        for item in self.data_loader:
            (
                fnamelist,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = item
            assert len(fnamelist) == 1
            fname: str = fnamelist[0].split("/")[-1].split(".")[0]

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #

            # Move data to GPU if available
            pitch_input.unsqueeze_(-1)
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

            loss_id: torch.Tensor
            loss_id = self.loss_fn(spmel_output, spmel_gt)

            # Backward and optimize.
            loss = loss_id

            try:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except Exception:
                self.logger.error(traceback.format_exc())

            # =============================================================== #
            #                   3. Logging and saving checkpoints             #
            # =============================================================== #
            found_nan: bool = False
            found_nan |= test_ret(loss_id)
            found_nan |= test_ret(spmel_gt)
            found_nan |= test_ret(rhythm_input)
            found_nan |= test_ret(content_input)
            found_nan |= test_ret(pitch_input)
            found_nan |= test_ret(timbre_input)
            found_nan |= test_ret(len_crop)
            found_nan |= test_ret(content_pitch_input)
            found_nan |= test_ret(spmel_output)
            la = 3
            lb = 3
            mask = spmel_gt != 0.0
            self.logger.info(
                "{fname}: {gb}: {loss:{lb}.{la}f}: ({mlos:3})".format(
                    fname=fname,
                    la=la,
                    lb=la + lb + 1,
                    gb="bad " if found_nan else "good",
                    loss=loss_id.item(),
                    mlos=int(mask.sum() / 80),
                )
            )
            if found_nan:
                self.save_tensor(spmel_gt, f"{fname}_spmel_gt")
                self.save_tensor(spmel_output, f"{fname}_spmel_output")
                if self.config.training.mask_loss:
                    self.save_tensor(loss_id, f"{fname}_loss_id")


def test_ret(
    x: torch.Tensor,
) -> bool:
    is_nan: bool = True if x.isnan().any().item() else False
    return is_nan
