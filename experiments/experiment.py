import datetime
import os
import time
from collections import OrderedDict
from typing import Optional, Self

import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_loader
from model import Generator_3 as Generator
from model import InterpLnr
from util.compute import Compute
from util.config import Config
from util.logging import Logger


class Experiment(object):
    logger: Logger = Logger()
    compute: Compute = Compute()
    config: Config
    intrp: InterpLnr
    model: Generator
    optimizer: torch.optim.Optimizer
    start_time: float
    writer: SummaryWriter

    def __init__(self: Self, config: Config, currtime: int = int(time.time())) -> None:
        self.config = config
        self.compute.print_compute()
        self.model = Generator(self.config)
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
            log_dir="tensorboard/{}/{}/{}".format(
                self.config.options.model_type,
                self.config.options.experiment,
                currtime,
            )
        )

    def print_model_info(self: Self) -> None:
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        self.logger.info(str(self.model))
        self.logger.info(self.config.options.model_type)
        self.logger.info("The number of parameters: {}".format(num_params))

    def restore_model(self: Self, resume_iters: Optional[int]) -> None:
        self.logger.info(f"Loading the trained models from step {resume_iters}...")
        ckpt_name = "{}-{}-{}{}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            f"-{resume_iters}" if resume_iters is not None else "",
        )
        save_dir = (
            self.config.paths.models
            if resume_iters is not None
            else self.config.paths.trained_models
        )
        ckpt = torch.load(
            os.path.join(save_dir, ckpt_name),
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

    def log_training_step(self: Self, i: int, train_loss_id: float):
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

    def save_checkpoint(self: Self, i: int) -> None:
        os.makedirs(self.config.paths.models, exist_ok=True)
        self.logger.info(f"Saving model checkpoint into {self.config.paths.models}...")
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

    def load_data(self: Self, singleitem: bool = False) -> None:
        self.data_loader = get_loader(self.config, singleitem=singleitem)
