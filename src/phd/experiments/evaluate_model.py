import time
from typing import Self

import torch

from utils import masked_mse

from .experiment import Experiment


## Solver for training
class Evaluate(Experiment):
    def evaluate(self: Self) -> None:
        self.model.train()
        self.intrp.train()
        self.start_time = time.time()
        if self.config.training.mask_loss:
            self.loss_fn = masked_mse
        else:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.load_data(singleitem=True, sequential=True)
        [self.evaluate_batch(batch) for batch in self.data_loader]  # type: ignore [func-returns-value]

    def evaluate_batch(
        self: Self,
        batch: tuple[
            list[str],
            list[str],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        (
            fname,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = batch
        spmel_gt = spmel_gt.to(self.compute.device())
        rhythm_input = rhythm_input.to(self.compute.device())
        content_input = content_input.to(self.compute.device())
        pitch_input = pitch_input.to(self.compute.device()).unsqueeze(-1)
        timbre_input = timbre_input.to(self.compute.device())
        len_crop = len_crop.to(self.compute.device())
        return
