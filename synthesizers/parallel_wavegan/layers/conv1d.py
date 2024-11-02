import torch


## Conv1d module with customized initialization
class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super(Conv1d, self).__init__(*args, **kwargs)

    ## Reset parameters
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)
