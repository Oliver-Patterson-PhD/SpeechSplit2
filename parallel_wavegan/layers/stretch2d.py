import torch


## Stretch2d module
class Stretch2d(torch.nn.Module):

    ## Initialize Stretch2d module.
    # @param x_scale    X scaling factor (Time axis in spectrogram).
    # @param y_scale    Y scaling factor (Frequency axis in spectrogram).
    # @param mode       Interpolation mode.
    def __init__(
        self,
        x_scale: int,
        y_scale: int,
        mode: str = "nearest",
    ) -> None:
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    ## Calculate forward propagation.
    # @param  x Input tensor (B, C, F, T).
    # @return Interpolated tensor (B, C, F * y_scale, T * x_scale),
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )
