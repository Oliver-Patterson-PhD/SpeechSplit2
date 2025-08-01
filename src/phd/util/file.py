__all__ = [
    "basename",
    "fread",
    "freadline",
    "freadlist",
    "strip_ext",
    "strip_path",
    "lsdir",
    "walkdirs",
    "walkfiles",
]

import glob
import os
from pathlib import Path
from typing import Literal, overload

FilePath = str | os.PathLike[str] | Path


@overload
def fread(file: FilePath, binary: Literal[True]) -> bytes:
    pass


@overload
def fread(file: FilePath, binary: Literal[False]) -> str:
    pass


@overload
def fread(file: FilePath) -> str:
    pass


def fread(file: FilePath, binary: bool = False) -> str | bytes:
    with open(file, "rb" if binary else "r") as f:
        if binary:
            return bytes(f.read())
        else:
            return str(f.read())


def fwrite(file: FilePath, buffer: str, binary: bool = False) -> int:
    with open(file, "wb" if binary else "w") as f:
        return f.write(buffer)


@overload
def freadline(file: FilePath, binary: Literal[True]) -> bytes:
    pass


@overload
def freadline(file: FilePath, binary: Literal[False]) -> str:
    pass


@overload
def freadline(file: FilePath) -> str:
    pass


def freadline(file: FilePath, binary: bool = False) -> str | bytes:
    with open(file, "rb" if binary else "r") as f:
        if binary:
            return bytes(next(f))
        else:
            return str(next(f))


def freadlist(file: FilePath) -> list[str]:
    with open(file, "r") as f:
        return [line for line in f]


def lsdir(dir: FilePath, showfiles: bool = True, showdirs: bool = False) -> list[str]:
    assert showfiles or showdirs
    root, dirs, files = next(os.walk(dir))[2]
    retval: list[str] = []
    if showfiles:
        retval.extend(files)
    if showdirs:
        retval.extend(dirs)
    return retval


def mywalk(top: FilePath, retdirs: bool, fullpaths: bool) -> list[str]:
    return [
        os.path.join(root, path) if fullpaths else path
        for root, dirs, files in os.walk(top)
        for path in (dirs if retdirs else files)
    ]


def walkfiles(top: FilePath, fullpaths: bool = False) -> list[str]:
    return mywalk(top, retdirs=False, fullpaths=fullpaths)


def walkdirs(top: FilePath, fullpaths: bool = False) -> list[str]:
    return mywalk(top, retdirs=True, fullpaths=fullpaths)


def strip_path(fullpath: FilePath) -> str:
    return os.path.split(str(fullpath))[-1]


def strip_ext(fullpath: FilePath) -> str:
    return os.path.splitext(str(fullpath))[0]


def basename(fullpath: FilePath) -> str:
    return strip_path(strip_ext(fullpath))


def dirname(fullpath: FilePath) -> str:
    return os.path.dirname(str(fullpath))


def path(*args: str) -> str:
    return os.path.join(*args)


def newpath(*args: str) -> str:
    dir = path(*args)
    os.makedirs(dir, exist_ok=True)
    return dir


def exists(dir: str) -> bool:
    return os.path.exists(dir)


def myglob(root: str, globstr: str) -> list[str]:
    return glob.glob(globstr, root_dir=root)


def rm_rf(dir: str) -> None:
    for root, _, files in os.walk(dir, topdown=False):
        for name in files:
            os.remove(path(root, name))


def filesize(fullpath: str) -> int:
    return os.path.getsize(fullpath)


def delete(fullpath: str) -> None:
    os.remove(fullpath)
