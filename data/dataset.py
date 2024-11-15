__all__ = [
    "Dataset",
    "DType",
]

import os
from enum import Enum, auto
from typing import Self

from util import Config

ua_uttrs = getattr(
    __import__("meta_dicts"),
    "uaspeech_uttrs",
)


class DType(Enum):
    VCTK = auto()
    UASPEECH = auto()


class Dataset:
    dtype: DType
    raw_data: str

    def __init__(self: Self, config: Config) -> None:
        self.raw_data = config.paths.raw_data
        self.dtype = self.__get_type(config.options.dataset_name)

    def __fname(self: Self, fname: str) -> str:
        return os.path.splitext(os.path.basename(fname))[0]

    def __get_type(self: Self, dname: str) -> DType:
        if dname in ("uaspeech", "smolspeech"):
            return DType.UASPEECH
        elif dname in ("vctk", "smolvctk"):
            return DType.VCTK
        else:
            raise ValueError

    def utterance(self: Self, fname: str) -> str:
        file_name = self.__fname(fname)
        match self.dtype:
            case DType.UASPEECH:
                return "_".join(file_name.split("_")[1:3])
            case DType.VCTK:
                return file_name
            case _:
                raise ValueError

    def speaker(self: Self, fname: str) -> str:
        file_name = self.__fname(fname)
        match self.dtype:
            case DType.UASPEECH:
                return file_name.split("_")[0]
            case DType.VCTK:
                return file_name.split("_")[0]
            case _:
                raise ValueError

    def sample_name(self: Self, fname: str) -> str:
        file_name = self.__fname(fname)
        match self.dtype:
            case DType.UASPEECH:
                return "_".join(file_name.split("_")[0:3])
            case DType.VCTK:
                return "_".join(file_name.split("_")[0:1])
            case _:
                raise ValueError

    def get_real_text(
        self: Self,
        fname: str,
    ) -> str:
        sample_name = self.sample_name(fname)
        match self.dtype:
            case DType.UASPEECH:
                return ua_uttrs[self.utterance(sample_name)]
            case DType.VCTK:
                with open(
                    "{}/VCTK-Corpus/txt/{}/{}.txt".format(
                        self.raw_data,
                        sample_name.split("_")[0],
                        sample_name,
                    )
                ) as uttr_file:
                    return next(uttr_file)
            case _:
                raise ValueError
