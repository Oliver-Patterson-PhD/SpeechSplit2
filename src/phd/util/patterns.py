__all__ = [
    "Singleton",
]

from torch.multiprocessing import Lock


class Singleton(type):
    _instances: dict = {}
    _lock = Lock()  # Ensures thread-safety for singleton instantiation

    def __call__(cls, *args, **kwargs):
        # Critical section where instance creation happens
        with cls._lock:
            if cls not in cls._instances:
                # If the class does not have an instance, create one
                print(f"Creating singleton {cls.__name__}")
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
