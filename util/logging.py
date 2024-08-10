from enum import IntEnum
from os import makedirs
from os.path import dirname
from sys import _getframe, stderr, stdout
from time import gmtime, strftime
from typing import Collection, Iterable, Iterator, Optional, TextIO
from util.patterns import Singleton


class LogLevel(IntEnum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 80
    ERROR = 90
    FATAL = 255


class LoggedException(Exception):
    pass


class Logger(metaclass=Singleton):
    __level: LogLevel = LogLevel.DEBUG
    __stream: TextIO = stdout
    __file: Optional[TextIO] = None

    def __init__(
        self, level: Optional[LogLevel] = None, use_stderr: Optional[bool] = None
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

    def __format_msg(self, level: LogLevel, caller: str, message: str) -> str:
        return "{} - {:>26} - {} - {}".format(
            strftime("%Y/%m/%d %H:%M:%S", gmtime()),
            caller,
            self.__level_to_string(level),
            message,
        )

    def __log(self, level: LogLevel, caller: str, message: str) -> None:
        if level >= self.__level:
            fullmsg = self.__format_msg(level=level, caller=caller, message=message)
            print(fullmsg, end="\n", file=self.__stream, flush=True)
            if self.__file is not None:
                print(fullmsg, end="\n", file=self.__file)
        return None

    def __level_to_string(self, level: Optional[LogLevel] = None) -> str:
        if level is None:
            level = self.__level
        match level:
            case LogLevel.TRACE:
                return "TRACE"
            case LogLevel.DEBUG:
                return "DEBUG"
            case LogLevel.INFO:
                return "INFO "
            case LogLevel.WARN:
                return "WARN "
            case LogLevel.ERROR:
                return "ERROR"
            case LogLevel.FATAL:
                return "FATAL"

    def get_level(self) -> LogLevel:
        return self.__level

    def set_level(self, level: LogLevel) -> None:
        self.__level = level

    def set_level_str(self, string: str) -> None:
        match string.upper():
            case "TRACE":
                self.__level = LogLevel.TRACE
            case "DEBUG":
                self.__level = LogLevel.DEBUG
            case "INFO":
                self.__level = LogLevel.INFO
            case "WARN":
                self.__level = LogLevel.WARN
            case "ERROR":
                self.__level = LogLevel.ERROR
            case "FATAL":
                self.__level = LogLevel.FATAL
            case _:
                raise RuntimeError(f"Invalid config level string '{string.upper()}'")

    def get_stream(self) -> TextIO:
        return self.__stream

    def get_file(self) -> Optional[TextIO]:
        return self.__file

    def set_stream(self, stream: TextIO) -> None:
        self.__stream = stream

    def set_file(self, file: str) -> None:
        makedirs(dirname(file), exist_ok=True)
        self.__file = open(file, "wt", encoding="utf-8")

    def use_stderr(self, use_stderr: bool) -> None:
        if use_stderr:
            self.__steam = stderr
        else:
            self.__steam = stdout

    def counter(
        self,
        iterable: Collection,
        unit: str = "",
        level: LogLevel = LogLevel.DEBUG,
    ) -> Iterator:
        countlen = len(iterable)
        count_num_size = len(str(countlen))
        for i, item in enumerate(iterable):
            print(
                f"\r{i + 1: {count_num_size}d}/{countlen} {unit}",
                end="",
                file=self.__stream,
                flush=True,
            )
            yield item
        print(
            end="\n",
            file=self.__stream,
            flush=True,
        )
        if self.__file is not None:
            print(
                self.__format_msg(
                    level=level,
                    caller=self.__get_caller(),
                    message=f"{unit} all finished",
                ),
                end="\n",
                file=self.__file,
            )
        return None

    def print_counter(
        self,
        iterable: Collection,
        prefix: str = "",
        unit: str = "",
        level: LogLevel = LogLevel.DEBUG,
    ) -> Iterator:
        countlen = len(iterable)
        count_num_size = len(str(countlen))
        for i, item in enumerate(iterable):
            print(
                f"{prefix}{i + 1: {count_num_size}d}/{countlen} {unit}",
                file=self.__stream,
            )
            yield item
        if self.__file is not None:
            print(
                self.__format_msg(
                    level=level,
                    caller=self.__get_caller(),
                    message=f"{prefix}{unit} all finished",
                ),
                end="\n",
                file=self.__file,
            )
        return None

    def print_iterator(
        self,
        iterable: Iterable,
        size: int,
        prefix: str = "",
        unit: str = "",
        level: LogLevel = LogLevel.DEBUG,
    ) -> Iterator:
        count_num_size = len(str(size))
        for i, item in enumerate(iterable):
            print(
                f"{prefix}{i + 1: {count_num_size}d}/{size} {unit}",
                file=self.__stream,
            )
            yield item
        if self.__file is not None:
            print(
                self.__format_msg(
                    level=level,
                    caller=self.__get_caller(),
                    message=f"{prefix}{unit} all finished",
                ),
                end="\n",
                file=self.__file,
            )
        return None

    def trace(self, message: str) -> None:
        self.__log(level=LogLevel.TRACE, caller=self.__get_caller(), message=message)

    def debug(self, message: str) -> None:
        self.__log(level=LogLevel.DEBUG, caller=self.__get_caller(), message=message)

    def info(self, message: str) -> None:
        self.__log(level=LogLevel.INFO, caller=self.__get_caller(), message=message)

    def warn(self, message: str) -> None:
        self.__log(level=LogLevel.WARN, caller=self.__get_caller(), message=message)

    def error(self, message: str) -> None:
        self.__log(level=LogLevel.ERROR, caller=self.__get_caller(), message=message)

    def fatal(self, message: str) -> None:
        self.__log(level=LogLevel.FATAL, caller=self.__get_caller(), message=message)
        raise LoggedException(message)
