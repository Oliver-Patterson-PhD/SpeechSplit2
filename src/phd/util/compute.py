__all__ = [
    "compute",
]

import torch

from .logging import logger
from .patterns import Singleton


## Compute device handler
class Compute(metaclass=Singleton):
    __device: torch.device
    __current_device: torch.device
    __device_id: int
    __gpu_name: tuple[str]
    __gpu_version: tuple[int, int]
    __gpu_compute: str
    __gpu_memory: tuple[float]

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
            if self.__device is not None:
                self.__current_device = self.__device
                self.__device_id = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(self.__device_id)
                self.__gpu_name = (gpu_properties.name,)
                self.__gpu_memory = (gpu_properties.total_memory / 1e9,)
                self.__gpu_version = (gpu_properties.major, gpu_properties.minor)
                torch.randn(1).cuda()
        else:
            self.__device = torch.device("cpu")
        return None

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (self.__class__, ())

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        if self.__device.type == "cuda":
            return "Using GPU {:d} {:s} with {:.1f}Gb total memory.".format(
                self.__device_id, self.__gpu_name[0], self.__gpu_memory[0]
            )
        else:
            return "Using CPU for inference."

    def device(self) -> torch.device:
        return self.__current_device

    def id(self) -> int:
        return self.__device_id

    def set_cpu(self) -> None:
        logger.info("Explicitly setting CPU for inference.", depth=2)
        self.__current_device = torch.device("cpu")
        torch.set_default_device("cpu")
        return None

    def set_gpu(self) -> None:
        logger.info("Explicitly setting GPU for inference.", depth=2)
        self.__current_device = self.__device
        torch.set_default_device(self.__device)
        return None

    def set_default(self) -> None:
        torch.set_default_device(self.__device)
        return None

    def print_compute(self) -> None:
        logger.info(self.__str__())

    def is_cpu(self) -> bool:
        return self.__current_device == torch.device("cpu")

    def is_gpu(self) -> bool:
        return self.__current_device != torch.device("cpu")

    def could_be_gpu(self) -> bool:
        return self.__device != torch.device("cpu")


compute = Compute()
