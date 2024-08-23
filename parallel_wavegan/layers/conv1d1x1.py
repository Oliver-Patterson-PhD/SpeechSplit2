from .conv1d import Conv1d


## 1x1 Conv1d with customized initialization.
class Conv1d1x1(Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
    ) -> None:
        super(Conv1d1x1, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=bias,
        )
