import torch

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_start_method("spawn")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    if dev is not None:
        torch.set_default_device(dev)

from .scripts.check import check
from .scripts.legacy import main
from .scripts.scratch import scratch
from .scripts.lld_classify import lld_classify
from .tests.unit import unit_tests
