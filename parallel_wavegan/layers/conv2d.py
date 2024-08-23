import torch


## Conv2d module with customized initialization
class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super(Conv2d, self).__init__(*args, **kwargs)

    ## Reset parameters
    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0 / torch.prod(torch.tensor(self.kernel_size)).item())
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)
