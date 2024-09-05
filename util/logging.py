from __future__ import annotations

from enum import IntEnum
from os import makedirs
from os.path import dirname
from sys import _getframe, stderr, stdout
from time import gmtime, strftime
from traceback import extract_stack
from typing import Any, Optional, TextIO, overload

from torch import Tensor
from util.patterns import Singleton


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
    __stream: TextIO = stdout
    __file: Optional[TextIO] = None

    def __init__(
        self,
        level: Optional[LogLevel] = None,
        use_stderr: Optional[bool] = None,
    ) -> None:
        if level is not None:
            self.__level = level
        if use_stderr is not None:
            self.use_stderr(use_stderr)

    def __get_caller(self) -> str:
        tmp_frame = _getframe(1).f_back
        if tmp_frame is not None:
            return tmp_frame.f_code.co_qualname
        else:
            raise RuntimeError

    def __format_msg(
        self,
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
        self,
        level: LogLevel,
        caller: str,
        message: str,
    ) -> None:
        if level >= self.__level:
            fullmsg = self.__format_msg(
                level=level,
                caller=caller,
                message=message,
            )
            print(
                fullmsg,
                end="\n",
                file=self.__stream,
                flush=True,
            )
            if self.__file is not None:
                print(
                    fullmsg,
                    end="\n",
                    file=self.__file,
                    flush=True,
                )
        return None

    def __is_nan(self, x: Tensor) -> bool:
        return True if x.isnan().any().item() else False

    def get_level(self) -> LogLevel:
        return self.__level

    @overload
    def set_level(self, level: str) -> None:
        pass

    @overload
    def set_level(self, level: LogLevel) -> None:
        pass

    def set_level(self, level: LogLevel | str) -> None:
        if isinstance(level, str):
            self.__level = LogLevel[level]
        elif isinstance(level, int):
            self.__level = level
        else:
            raise ValueError

    def get_stream(self) -> TextIO:
        return self.__stream

    def get_file(self) -> Optional[TextIO]:
        return self.__file

    def set_stream(self, stream: TextIO) -> None:
        self.__stream = stream

    def set_file(self, file: str) -> None:
        makedirs(
            dirname(file),
            exist_ok=True,
        )
        self.__file = open(
            file,
            "wt",
            encoding="utf-8",
        )

    def use_stderr(
        self,
        use_stderr: bool,
    ) -> None:
        if use_stderr:
            self.__steam = stderr
        else:
            self.__steam = stdout

    def input(
        self,
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
        self,
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

    def trace(self, message: str) -> None:
        self.__log(
            level=LogLevel.TRACE,
            caller=self.__get_caller(),
            message=message,
        )

    def debug(self, message: str) -> None:
        self.__log(
            level=LogLevel.DEBUG,
            caller=self.__get_caller(),
            message=message,
        )

    def info(self, message: str) -> None:
        self.__log(
            level=LogLevel.INFO,
            caller=self.__get_caller(),
            message=message,
        )

    def warn(self, message: str) -> None:
        self.__log(
            level=LogLevel.WARN,
            caller=self.__get_caller(),
            message=message,
        )

    def error(self, message: str) -> None:
        self.__log(
            level=LogLevel.ERROR,
            caller=self.__get_caller(),
            message=message,
        )

    def fatal(self, message: str) -> None:
        self.__log(
            level=LogLevel.FATAL,
            caller=self.__get_caller(),
            message=message,
        )
        raise LoggedException(message)

    def trace_var(
        self,
        var: Any,
        level: LogLevel = LogLevel.TRACE,
    ) -> None:
        code = extract_stack()[-2][-1]
        varname = code[code.find("trace_var(") + 10 : code.rfind(")")]
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=f"{varname}: ({var})",
        )

    def trace_nans(
        self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> None:
        code = extract_stack()[-2][-1]
        varname = code[code.find("trace_nans(") + 11 : code.rfind(")")]
        self.__log(
            level=level,
            caller=self.__get_caller(),
            message=f"{varname}: {
                "Has NaNs" if self.__is_nan(x) else "No NaNs"
            }",
        )

    def log_if_nan(
        self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> None:
        if self.__is_nan(x):
            code = extract_stack()[-2][-1]
            varname = code[code.find("log_if_nan(") + 11 : code.rfind(")")]
            self.__log(
                level=level,
                caller=self.__get_caller(),
                message=f"{varname}: Has NaNs",
            )

    def log_if_nan_ret(
        self,
        x: Tensor,
        level: LogLevel = LogLevel.ERROR,
    ) -> bool:
        if self.__is_nan(x):
            code = extract_stack()[-2][-1]
            varname = code[code.find("log_if_nan(") + 11 : code.rfind(")")]
            self.__log(
                level=level,
                caller=self.__get_caller(),
                message=f"{varname}: Has NaNs",
            )
            return True
        else:
            return False
