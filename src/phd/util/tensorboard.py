import inspect

from torch.utils.tensorboard import SummaryWriter

from .config import config
from .file import basename, delete, filesize, newpath
from .logging import logger
from .tensor import Tensor


class TensorBoard:
    def __init__(self) -> None:
        self.__experiment = basename(inspect.stack()[1].filename)
        self.__log_dir = newpath(config.paths.tensorboard, self.__experiment)
        self.__sum_writer = SummaryWriter(log_dir=self.__log_dir)
        self.__prefix = f"{self.__experiment}/"

    def __del__(self) -> None:
        fname = self.__sum_writer._get_file_writer().event_writer._file_name
        self.__sum_writer.flush()
        del self.__sum_writer
        fsize = filesize(fname)
        if fsize < 200:
            logger.info(f"{fname} is not large enough to keep, removing")
            delete(fname)
        else:
            logger.info(f"{fname} is large enough to keep, size: {fsize}")

    def add_scalar(self, name: str, item: float, step: int) -> None:
        self.__sum_writer.add_scalar(
            tag=self.__prefix + name,
            scalar_value=item,
            global_step=step,
        )

    def add_audio(self, name: str, item: Tensor, step: int) -> None:
        self.__sum_writer.add_audio(
            tag=self.__prefix + name,
            snd_tensor=item,
            global_step=step,
            sample_rate=config.audio.sample_rate,
        )

    def add_item(self, name: str, item: object, step: int) -> None:
        if isinstance(item, float):
            self.add_scalar(name=name, item=item, step=step)
        if isinstance(item, Tensor):
            titem: Tensor = item
            if titem.ndim == 1:
                self.add_audio(name=name, item=titem, step=step)
            else:
                logger.warn("invalid tensor")
                logger.trace_tensor(titem, "WARN")
        pass

    def add_pr_curve(
        self,
        name: str,
        ground_truth: Tensor,
        generated: Tensor,
        step: int,
    ) -> None:
        self.__sum_writer.add_pr_curve(
            tag=self.__prefix + name,
            labels=ground_truth,
            predictions=generated,
            global_step=step,
        )

    def flush(self) -> None:
        self.__sum_writer.flush()
