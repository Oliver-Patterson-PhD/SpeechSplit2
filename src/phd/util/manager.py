__all__ = [
    "Application",
]

# import torch.multiprocessing as mp
import multiprocessing as mp


class PhDManager(mp.managers.BaseManager):
    address: str = "127.0.0.1"
    serializer: str = "pickle"
    ctx: mp.context.BaseContext | None = None
    shutdown_timeout: float = 1.0

    def __init__(self) -> None:
        super().__init__(
            address=self.address,
            serializer=self.serializer,
            ctx=self.ctx,
            shutdown_timeout=self.shutdown_timeout,
        )


class Application:
    def __init__(self) -> None:
        self.manager = PhDManager()
