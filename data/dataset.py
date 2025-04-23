__all__ = [
    "DatasetParser",
    "DType",
]

import os
from enum import Enum, auto
from typing import Optional, Self, Set, Dict

from util import Config

from meta_dicts import MetaDictType

ua_uttrs: Dict[str, str]
ua_uttrs = getattr(
    __import__("meta_dicts"),
    "uaspeech_uttrs",
)


class DType(Enum):
    VCTK = auto()
    UASPEECH = auto()


class DatasetParser:
    __raw_data_path: str
    __dsettype: DType
    __spk_meta: MetaDictType

    def __init__(
        self: Self,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config()
        self.__raw_data_path = config.paths.raw_data
        self.__dsettype = self.__get_type(config.options.dataset_name)
        self.__spk_meta = getattr(
            __import__("meta_dicts"),
            config.options.dataset_name,
        )

    def __fname(
        self: Self,
        fname: str,
    ) -> str:
        return os.path.splitext(os.path.basename(fname))[0]

    def __get_type(self: Self, dname: str) -> DType:
        if dname in ("uaspeech", "smolspeech"):
            return DType.UASPEECH
        elif dname in ("vctk", "smolvctk"):
            return DType.VCTK
        else:
            raise ValueError

    def dataset_type(
        self: Self,
    ) -> DType:
        return self.__dsettype

    def is_uaspeech(self: Self) -> bool:
        return self.dataset_type() == DType.UASPEECH

    def is_vctk(self: Self) -> bool:
        return self.dataset_type() == DType.VCTK

    def sex(
        self: Self,
        speaker: str,
    ) -> str:
        return self.__spk_meta[speaker].sex

    def speakers(
        self: Self,
    ) -> Set[str]:
        return set(self.__spk_meta.keys())

    def utterance(
        self: Self,
        fname: str,
    ) -> str:
        file_name = self.__fname(fname)
        match self.dataset_type():
            case DType.UASPEECH:
                return "_".join(file_name.split("_")[1:3])
            case DType.VCTK:
                return file_name
            case _:
                raise ValueError

    def speaker(
        self: Self,
        fname: str,
    ) -> str:
        file_name = self.__fname(fname)
        match self.__dsettype:
            case DType.UASPEECH:
                return file_name.split("_")[0]
            case DType.VCTK:
                return file_name.split("_")[0]
            case _:
                raise ValueError

    def samples(
        self: Self,
        spk: str,
    ):
        return

    def sample_name(
        self: Self,
        fname: str,
    ) -> str:
        file_name = self.__fname(fname)
        match self.__dsettype:
            case DType.UASPEECH:
                return "_".join(file_name.split("_")[0:3])
            case DType.VCTK:
                return "_".join(file_name.split("_")[0:2])
            case _:
                raise ValueError

    def get_real_text(
        self: Self,
        fname: str,
    ) -> str:
        sample_name = self.sample_name(fname)
        match self.__dsettype:
            case DType.UASPEECH:
                return ua_uttrs[self.utterance(sample_name)]
            case DType.VCTK:
                with open(
                    "{}/VCTK-Corpus/txt/{}/{}.txt".format(
                        self.__raw_data_path,
                        sample_name.split("_")[0],
                        sample_name,
                    )
                ) as uttr_file:
                    return next(uttr_file)
            case _:
                raise ValueError
