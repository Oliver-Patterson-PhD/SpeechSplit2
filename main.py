import argparse

import torch
from data_loader import get_loader
from data_preprocessing import preprocess_data
from solver import Solver
from torch.backends import cudnn
from util.config import Config
from util.logging import Logger


def main(config, args):
    # For fast training.
    cudnn.benchmark = True
    preprocess_data(config)
    data_loader = get_loader(config)
    solver = Solver(data_loader, args, config)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    Logger().info(
        ("Using GPU %d (%s) of compute capability " + "%d.%d with %.1fGb total memory.")
        % (
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9,
        )
    )
    solver.train()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--num_iters", type=int, default=800000)
    parser.add_argument("--resume_iters", type=int, default=0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--ckpt_save_step", type=int, default=1000)
    parser.add_argument("--config_name", type=str, default="spsp2-large")
    # fmt: on
    args = parser.parse_args()

    logger = Logger()
    config = Config(f"configs/{args.config_name}.toml")

    if args.trace or config.options.trace:
        hr = __import__("heartrate")
        hr.trace(files=hr.files.all)

    main(config, args)
