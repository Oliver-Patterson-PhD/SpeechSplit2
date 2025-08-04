__all__ = [
    "Singleton",
]

import torch.multiprocessing as mp


class Singleton(type):
    _instances: dict = {}
    _lock = mp.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                print(f"Creating singleton {cls.__name__}")
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
