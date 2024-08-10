from typing import Tuple

import torch
from util.logging import Logger
from util.patterns import Singleton


## Compute device handler
class Compute(metaclass=Singleton):
    __logger: Logger
    __device: torch.device
    __current_device: torch.device
    __device_id: int
    __gpu_name: Tuple[str]
    __gpu_version: Tuple[int, int]
    __gpu_compute: str
    __gpu_memory: Tuple[float]

    def __init__(self) -> None:
        self.__logger = Logger()
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

    def device(self) -> torch.device:
        return self.__current_device

    def id(self) -> int:
        return self.__device_id

    def set_cpu(self) -> None:
        self.__logger.info("Explicitly setting CPU for inference.")
        self.__current_device = torch.device("cpu")
        torch.set_default_device("cpu")
        return None

    def set_gpu(self) -> None:
        self.__logger.info("Explicitly setting GPU for inference.")
        self.__current_device = self.__device
        torch.set_default_device(self.__device)
        return None

    def set_default(self) -> None:
        torch.set_default_device(self.__device)
        return None

    def print_compute(self):
        if self.__device.type == "cuda":
            self.__logger.info(
                "Using GPU %d (%s) with %.1fGb total memory."
                % (
                    self.__device_id,
                    self.__gpu_name[0],
                    self.__gpu_memory[0],
                )
            )
        else:
            self.__logger.info("Using CPU for inference.")
        return None
