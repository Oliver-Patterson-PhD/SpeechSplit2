__all__ = [
    "fread",
    "freadline",
]

from os import PathLike


def fread(file: str | PathLike, binary: bool = False) -> str:
    with open(file, "rb" if binary else "r") as f:
        return f.read()


def freadline(file: str | PathLike, binary: bool = False) -> str:
    with open(file, "rb" if binary else "r") as f:
        return next(f)
