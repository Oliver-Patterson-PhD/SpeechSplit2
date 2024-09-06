from typing import Self

import requests

from util.config import Config


class Notifier(object):
    service: str = "ntfy"
    url: str

    def __init__(self: Self) -> None:
        config = Config()
        self.url = config.options.ntfy_url
        return

    def send(self: Self, message: str) -> None:
        requests.post(
            self.url,
            data=message.encode(encoding="utf-8"),
        )
