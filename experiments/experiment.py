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
from utils import save_tensor


class Experiment(object):
    logger: Logger = Logger()
    compute: Compute = Compute()
    config: Config
    intrp: InterpLnr
    model: Generator
    optimizer: torch.optim.Optimizer
    start_time: float
    writer: SummaryWriter
    tb_prefix: str

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
            log_dir="{}/{}/{}/{}".format(
                self.config.paths.tensorboard,
                self.config.options.model_type,
                self.config.options.experiment,
                currtime,
            )
        )
        self.tb_prefix = (
            self.config.options.experiment + "/" + self.config.options.model_type
        )

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

    def restore_model(self: Self, resume_iters: Optional[int]) -> None:
        self.logger.info(
            f"Loading the trained models from step {resume_iters}...",
            depth=2,
        )
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
                step,
                self.config.options.num_iters,
                loss,
            ),
            depth=2,
        )
        if orig is not None and proc is not None:
            self.tb_add_melspec(name="orig", tensor=orig, step=step)
            self.tb_add_melspec(name="proc", tensor=proc, step=step)
            self.writer.flush()

    def save_checkpoint(self: Self, i: int) -> None:
        os.makedirs(self.config.paths.models, exist_ok=True)
        self.logger.info(
            f"Saving model checkpoint into {self.config.paths.models}...",
            depth=2,
        )
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

    def save_tensor(self: Self, tensor: torch.Tensor, fname: str) -> None:
        save_tensor(
            tensor,
            "{}/{}/{}".format(
                self.config.paths.artefacts,
                self.config.options.experiment,
                fname,
            ),
        )
        return
