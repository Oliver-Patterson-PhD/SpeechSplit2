from enum import Enum

import torch


class TurningPointType(Enum):
    MaxOnly = 1
    MinOnly = 2
    Both = 3


def turning_points(
    x: torch.Tensor,
    tp_type: TurningPointType = TurningPointType.Both,
    ave_step: int = 6,
) -> torch.Tensor:
    x_grad = gradient(x, ave_step)
    match tp_type:
        case TurningPointType.MaxOnly:
            check = x_grad.diff(n=1, append=torch.zeros((1, 1))) < 0
        case TurningPointType.MinOnly:
            check = x_grad.diff(n=1, append=torch.zeros((1, 1))) > 0
        case TurningPointType.Both:
            check = torch.Tensor(1)
    x_grad_dual = torch.cat(
        (torch.Tensor([[0, 0]]), x_grad.squeeze().unfold(0, 2, 1)), dim=0
    ).T
    x_pos_tps = (
        ((x_grad_dual[0] > 0) * (x_grad_dual[1] < 0))
        + ((x_grad_dual[0] < 0) * (x_grad_dual[1] > 0))
    ) * check
    x_pos_tps = x_pos_tps.squeeze()[0:-1].unsqueeze(dim=0)
    assert (
        x.size()[-1] == x_pos_tps.size()[-1]
    ), f"Size mismatch: in: {x.size()}  out: {x_pos_tps.size()}"
    return x_pos_tps


def find_peaks(x, ave_step: int = 6) -> torch.Tensor:
    return turning_points(x, TurningPointType.MaxOnly, ave_step)


def find_troughs(x, ave_step: int = 6) -> torch.Tensor:
    return turning_points(x, TurningPointType.MinOnly, ave_step)


def gradient(x: torch.Tensor, ave_step: int) -> torch.Tensor:
    return torch.nn.functional.avg_pool1d(
        x, kernel_size=ave_step, stride=1, padding=int(ave_step / 2)
    ).diff(n=1, append=torch.zeros((1, 1)))


def float_equal(a: float, b: float, eps: float = 0.001) -> bool:
    return (a == b) or (a > (b - eps) and a < (b + eps))
