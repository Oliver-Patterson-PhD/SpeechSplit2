__all__ = [
    "Singleton",
]

from torch import multiprocessing

lock = multiprocessing.Lock()


class Singleton(type):
    def __new__(mcs, name, bases, attrs):
        cls = super(Singleton, mcs).__new__(mcs, name, bases, attrs)
        cls._lock = multiprocessing.Lock()
        return cls

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with cls._lock:
                if not hasattr(cls, "_instance"):
                    print(f"Creating singleton {cls.__name__}")
                    cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance
