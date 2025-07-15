from typing import Self

import torch


class LinearNorm(torch.nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain="linear",
    ) -> None:
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear_layer(x)
