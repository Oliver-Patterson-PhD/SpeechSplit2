import inspect
from functools import wraps
from typing import Callable, TypeVar, ParamSpec


P = ParamSpec("P")
R = TypeVar("R")


def dump_args(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper
