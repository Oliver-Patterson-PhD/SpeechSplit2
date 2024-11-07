from __future__ import annotations

import inspect
from enum import IntEnum
from os import makedirs
from os.path import dirname
from shutil import get_terminal_size
from sys import _getframe
from time import gmtime, strftime
from typing import Any, List, Optional, Self, TextIO, overload

from torch import Tensor
from tqdm.auto import tqdm

from .patterns import Singleton


class LogLevel(IntEnum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 80
    ERROR = 90
    FATAL = 255

    def __str__(self: Self) -> str:
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

    def __repr__(self: Self) -> str:
        return str(self)

    @classmethod
    def _missing_(cls, value: object) -> Optional[LogLevel]:
        if isinstance(value, str):
            for member in cls:
                if str(member.value) == value.upper().strip():
                    return member
        return None


class LoggedException(Exception):
    pass


class Logger(metaclass=Singleton):
    __level: LogLevel = LogLevel.DEBUG
    __file: Optional[TextIO] = None
    __flush: bool = False
    __process_bar_running: bool = False

    def __init__(
        self: Self,
        level: Optional[LogLevel] = None,
        flush: Optional[bool] = None,
    ) -> None:
        if level is not None:
            self.__level = level
        if flush is not None:
            self.__flush = flush

    def __get_caller(
        self: Self,
        depth: int = 1,
    ) -> str:
        tmp_frame = _getframe(depth).f_back
        assert tmp_frame is not None
        return tmp_frame.f_code.co_qualname

    def __format_msg(
        self: Self,
        level: LogLevel,
        caller: str,
        message: str,
    ) -> str:
        return "{} - {:>26} - {} - {}".format(
            strftime(
                "%Y/%m/%d %H:%M:%S",
                gmtime(),
            ),
            caller,
            str(level),
            message,
        )

    def __log(
        self: Self,
        level: LogLevel,
        caller: str,
        message: str,
    ) -> None:
        if self.__file is not None or level >= self.__level:
            fullmsg = self.__format_msg(
                level=level,
                caller=caller,
                message=message,
            )
            if level >= self.__level:
                if self.__process_bar_running:
                    tqdm.write("\r" + (" " * get_terminal_size().columns), end="\r")
                tqdm.write(fullmsg)
            if self.__file is not None:
                print(
                    fullmsg,
                    file=self.__file,
                    flush=self.__flush,
                )
        return None

    def __is_nan(
        self: Self,
        x: Tensor,
    ) -> bool:
        return True if x.isnan().any().item() else False

    def __get_passed_varnames(
        self: Self,
    ) -> List[str]:
        frame = inspect.currentframe()
        assert frame is not None
        finfo = inspect.getouterframes(frame)[2]
        context = inspect.getframeinfo(finfo[0]).code_context
        assert context is not None
        string = context[0].strip().replace("Logger()", "logger")
        args = string[string.find("(") + 1 : -1].split(",")
        return [i.split("=")[1].strip() if i.find("=") != -1 else i for i in args]

    def get_level(
        self: Self,
    ) -> LogLevel:
        return self.__level

    @overload
    def set_level(
        self: Self,
        level: str,
    ) -> None:
        pass

    @overload
    def set_level(
        self: Self,
        level: LogLevel,
    ) -> None:
        pass

    def set_level(
        self: Self,
        level: LogLevel | str,
    ) -> None:
        if isinstance(level, str):
            self.__level = LogLevel[level]
        elif isinstance(level, int):
            self.__level = level
        else:
            raise ValueError

    def get_file(
        self: Self,
    ) -> Optional[TextIO]:
        return self.__file

    def set_file(
        self: Self,
        file: str,
    ) -> None:
        makedirs(
            dirname(file),
            exist_ok=True,
        )
        self.__file = open(
            file,
            "wt",
            encoding="utf-8",
        )

    def input(
        self: Self,
        message: str,
        prompt: str = "",
        level: LogLevel = LogLevel.INFO,
    ) -> str:
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=message,
        )
        return input(prompt)

    def input_with_default(
        self: Self,
        message: str,
        default: str,
        prompt: str = "",
        level: LogLevel = LogLevel.INFO,
    ) -> str:
        if self.__level <= level:
            self.__log(
                level=level,
                caller=self.__get_caller(),
                message=message,
            )
            return input(prompt)
        else:
            return default

    def trace(
        self: Self,
        message: str,
        depth: int = 1,
    ) -> None:
        self.__log(
            level=LogLevel.TRACE,
            caller=self.__get_caller(depth),
            message=message,
        )

    def debug(
        self: Self,
        message: str,
        depth: int = 1,
    ) -> None:
        self.__log(
            level=LogLevel.DEBUG,
            caller=self.__get_caller(depth),
            message=message,
        )

    def info(
        self: Self,
        message: str,
        depth: int = 1,
    ) -> None:
        self.__log(
            level=LogLevel.INFO,
            caller=self.__get_caller(depth),
            message=message,
        )

    def warn(
        self: Self,
        message: str,
        depth: int = 1,
    ) -> None:
        self.__log(
            level=LogLevel.WARN,
            caller=self.__get_caller(depth),
            message=message,
        )

    def error(
        self: Self,
        message: str,
        depth: int = 1,
    ) -> None:
        self.__log(
            level=LogLevel.ERROR,
            caller=self.__get_caller(depth),
            message=message,
        )

    def fatal(
        self: Self,
        message: str | Exception,
        depth: int = 1,
    ) -> None:
        self.__flush = True
        self.__log(
            level=LogLevel.FATAL,
            caller=self.__get_caller(depth),
            message=str(message),
        )
        raise LoggedException(message)

    def trace_var(
        self: Self,
        var: Any,
        level: LogLevel = LogLevel.TRACE,
    ) -> None:
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=f"{self.__get_passed_varnames()[0]}: ({var})",
        )

    def trace_tensor(
        self: Self,
        var: Tensor,
        level: LogLevel = LogLevel.TRACE,
    ) -> None:
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=f"{self.__get_passed_varnames()[0]}: ({var.shape})",
        )

    def trace_nans(
        self: Self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> None:
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=f"{self.__get_passed_varnames()[0]}: {
                "Has NaNs" if self.__is_nan(x) else "No NaNs"
            }",
        )

    def log_if_nan(
        self: Self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> None:
        if self.__is_nan(x):
            self.__log(
                level=level,
                caller=self.__get_caller(),
                message=f"{self.__get_passed_varnames()[0]}: Has NaNs",
            )

    def log_if_nan_ret(
        self: Self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> bool:
        if self.__is_nan(x):
            self.__log(
                level=level,
                caller=self.__get_caller(),
                message=f"{self.__get_passed_varnames()[0]}: Has NaNs",
            )
            return True
        else:
            return False

    def progress_bar(
        self: Self,
        *args,
        **kwargs,
    ):
        self.__process_bar_running = True
        for item in tqdm(*args, **kwargs):
            yield item
        self.__process_bar_running = False

    def manual_process_bar_start(
        self: Self,
        *args,
        **kwargs,
    ):
        self.__process_bar_running = True
        return tqdm(*args, **kwargs)

    def manual_process_bar_end(self: Self, pbar):
        pbar.close()
        self.__process_bar_running = False
        return
