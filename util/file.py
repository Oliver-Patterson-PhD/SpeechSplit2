__all__ = [
    "basename",
    "fread",
    "freadline",
    "strip_ext",
    "strip_path",
    "lsdir",
    "walkdirs",
    "walkfiles",
]

import os

PathVar = str | os.PathLike[str]


def fread(file: PathVar, binary: bool = False) -> str:
    with open(file, "rb" if binary else "r") as f:
        return f.read()


def freadline(file: PathVar, binary: bool = False) -> str:
    with open(file, "rb" if binary else "r") as f:
        return next(f)


def lsdir(dir: PathVar, showfiles: bool = True, showdirs: bool = False) -> list[str]:
    assert showfiles or showdirs
    root, dirs, files = next(os.walk(dir))[2]
    retval: list[str] = []
    if showfiles:
        retval.extend(files)
    if showdirs:
        retval.extend(dirs)
    return retval


def mywalk(top: PathVar, retdirs: bool, fullpaths: bool) -> list[str]:
    return [
        os.path.join(root, path) if fullpaths else path
        for root, dirs, files in os.walk(top)
        for path in (dirs if retdirs else files)
    ]


def walkfiles(top: PathVar, fullpaths: bool = False) -> list[str]:
    return mywalk(top, retdirs=False, fullpaths=fullpaths)


def walkdirs(top: PathVar, fullpaths: bool = False) -> list[str]:
    return mywalk(top, retdirs=True, fullpaths=fullpaths)


def strip_path(fullpath: PathVar) -> str:
    return str(fullpath).rpartition(os.path.sep)[-1]


def strip_ext(fullpath: PathVar) -> str:
    return str(fullpath).rpartition(".")[0]


def basename(fullpath: PathVar) -> str:
    return strip_path(strip_ext(fullpath))


def newpath(*args: str) -> str:
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def path(*args: str) -> str:
    return os.path.join(*args)
