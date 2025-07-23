from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from inspect import currentframe, getframeinfo, getouterframes, stack
from pprint import pformat
from shutil import get_terminal_size
from time import gmtime, strftime
from tomllib import load as loadtoml
from types import FrameType
from typing import Any, Iterable, TextIO

from tqdm import tqdm

from ..util.file import dirname, exists, newpath
from ..util.tensor import Tensor, is_nan

__all__ = [
    "config",
    "logger",
]


class LogLevel(IntEnum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 80
    ERROR = 90
    FATAL = 255

    def __str__(self) -> str:
        match self:
            case self.TRACE:
                return "TRACE"
            case self.DEBUG:
                return "DEBUG"
            case self.INFO:
                return "INFO "
            case self.WARN:
                return "WARN "
            case self.ERROR:
                return "ERROR"
            case self.FATAL:
                return "FATAL"
            case _:
                raise ValueError

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def _missing_(cls, value: object) -> LogLevel | None:
        if isinstance(value, str):
            for member in cls:
                if str(member.value) == value.upper().strip():
                    return member
        return None


TRACE = LogLevel.TRACE
DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARN = LogLevel.WARN
ERROR = LogLevel.ERROR
FATAL = LogLevel.FATAL


LogStr = LogLevel | str


class Logger:
    __level: LogLevel = DEBUG
    __file: TextIO | None = None
    __flush: bool = False
    __pbar_running: bool = False
    __date_format: str = "%Y/%m/%d %H:%M:%S"
    __print_callgraph: bool = False

    def enable_callgraph(self) -> None:
        self.__print_callgraph = True

    def disable_callgraph(self) -> None:
        self.__print_callgraph = False

    def set_level(self, level: LogStr) -> None:
        self.__level = self.__get_level(level)

    def set_file(self, file: str) -> None:
        newpath(dirname(file))
        self.__file = open(file, "wt", encoding="utf-8")

    def input(self, message: str, prompt: str = "", level: LogStr = INFO) -> str:
        self.__log(
            level=self.__get_level(level),
            caller=self.__get_caller(1),
            message=message,
        )
        return input(prompt)

    def input_with_default(
        self, message: str, default: str, prompt: str = "", level: LogStr = INFO
    ) -> str:
        if self.__level > self.__get_level(level):
            return default
        self.__log(
            level=self.__get_level(level),
            caller=self.__get_caller(1),
            message=message,
        )
        return input(prompt)

    def trace[T](self, var: T, depth: int = 1, level=TRACE) -> None:
        if isinstance(var, Tensor):
            message = f"{self.__get_passed_varnames()[0]}: ({var.shape})"
            message += " with NaNs" if is_nan(var) else ""
        else:
            message = f"{self.__get_passed_varnames()[0]}: {pformat(var, indent=4)}"
        self.__log(level=level, caller=self.__get_caller(depth), message=message)

    def debug(self, message: str, depth: int = 1) -> None:
        self.__log(level=DEBUG, caller=self.__get_caller(depth), message=message)

    def info(self, message: str, depth: int = 1) -> None:
        self.__log(level=INFO, caller=self.__get_caller(depth), message=message)

    def warn(self, message: str, depth: int = 1) -> None:
        self.__log(level=WARN, caller=self.__get_caller(depth), message=message)

    def error(self, message: str, depth: int = 1) -> None:
        self.__log(level=ERROR, caller=self.__get_caller(depth), message=message)

    def fatal(self, message: str | Exception, depth: int = 1) -> None:
        self.__flush = True
        self.__log(level=FATAL, caller=self.__get_caller(depth), message=str(message))

    def __format_caller(self, frame: FrameType) -> str:
        base = "/src/phd/"
        name = frame.f_code.co_qualname
        fname = frame.f_code.co_filename.partition(base)[2]
        return f"[{fname}:{frame.f_lineno}] {name}"

    def __format_callgraph(self, callgraph: str) -> str:
        # add spaces equal to additional characters in the prefix of the format string
        prefix_str = "\n" + (" " * (len(self.__date_format) + 13))
        calls = [call.strip("'") for call in callgraph.strip("[]").split(", ")]
        return prefix_str.join(calls) + prefix_str

    def __get_caller(self, depth: int) -> str:
        if self.__print_callgraph:
            frame: FrameType | None = currentframe()
            assert frame is not None
            names: list[str] = []
            while True:
                frame = frame.f_back
                if frame is None:
                    break
                name = frame.f_code.co_qualname
                if name == "<module>":
                    break
                if name.startswith("Logger"):
                    continue
                names.append(self.__format_caller(frame))
            return f"{list(reversed(names))}"
        else:
            tmp_frame = stack()[depth].frame.f_back
            assert tmp_frame is not None
            return tmp_frame.f_code.co_qualname

    def __format_msg(self, level: LogLevel, caller: str, message: str) -> str:
        if self.__print_callgraph:
            return "{} - {} - {} - {}".format(
                strftime(self.__date_format, gmtime()),
                str(level),
                self.__format_callgraph(caller),
                message,
            )
        else:
            return "{} - {:>26} - {} - {}".format(
                strftime(self.__date_format, gmtime()),
                caller,
                str(level),
                message,
            )

    def __get_passed_varnames(self) -> list[str]:
        frame = currentframe()
        assert frame is not None
        finfo = getouterframes(frame)[2]
        context = getframeinfo(finfo[0]).code_context
        assert context is not None
        string = context[0].strip().replace("Logger()", "logger")
        args = string[string.find("(") + 1 : -1].split(",")
        return [i.split("=")[1].strip() if i.find("=") != -1 else i for i in args]

    def __get_level(self, level: LogStr) -> LogLevel:
        return level if isinstance(level, LogLevel) else LogLevel[level]

    def __unformatted(self, level: LogStr, fullmsg: str) -> None:
        level = self.__get_level(level)
        if self.__file is not None or level >= self.__level:
            if level >= self.__level:
                if self.__pbar_running:
                    tqdm.write("\r" + (" " * get_terminal_size().columns), end="\r")
                tqdm.write(fullmsg)
            if self.__file is not None:
                print(fullmsg, file=self.__file, flush=self.__flush)
        return None

    def __log(self, level: LogLevel, caller: str, message: str) -> None:
        self.__unformatted(
            level=level,
            fullmsg=self.__format_msg(level=level, caller=caller, message=message),
        )

    def progress_bar[T](
        self, iter: Iterable[T], *args: Any, **kwargs: Any
    ) -> Iterable[T]:
        self.__pbar_running = True
        for item in tqdm(iter, *args, **kwargs):
            yield item
        self.__pbar_running = False

    def manual_pbar_start(self, *args: Any, **kwargs: Any) -> None:
        self.__pbar_running = True
        self.__manual_pbar = tqdm(*args, **kwargs)

    def manual_pbar_update(self, *args: Any, **kwargs: Any) -> None:
        self.__manual_pbar.update

    def manual_pbar_end(self) -> None:
        self.__manual_pbar.close
        self.__pbar_running = False
        return
        self.__pbar_running = False
        return


logger: Logger


class Config:
    start_time: str

    def __init__(self) -> None:
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_file = "config.toml"
        if not exists(config_file):
            err_str = f"Could not find file: {config_file}"
            logger.fatal(err_str)
            raise FileNotFoundError(err_str)
        tomldict = loadtoml(open(config_file, "rb"))
        self.__print_config(tomldict)
        for key, subdict in tomldict.items():
            match key.lower():
                case "log":
                    if "level" in subdict:
                        logger.set_level(subdict["level"])
                case "paths":
                    if not subdict.keys() >= {"raw_data", "proc_data"}:
                        err_str = "Could not find paths.proc_data and paths.raw_data in config"
                        logger.fatal(err_str)
                        raise RuntimeError(err_str)
        return

    def __print_config(self, config_dict: dict) -> None:
        logger.info(
            "\n".join(
                [
                    "\n".join(
                        [f"[{c}]"]
                        + [f"{k} = {v}" for k, v in config_dict[c].items()]
                        + [""]
                    )
                    for c in config_dict.keys()
                ]
            )
        )


config: Config

config = Config()
logger = Logger()
