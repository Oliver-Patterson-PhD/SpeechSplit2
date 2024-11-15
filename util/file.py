__all__ = [
    "fread",
    "freadline",
]

from os import PathLike


def fread(file: str | PathLike) -> str:
    with open(file, "r") as f:
        return f.read()


def freadline(file: str | PathLike) -> str:
    with open(file, "r") as f:
        return next(f)
