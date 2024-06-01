import os
import yaml
import argparse
import torch
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from data_preprocessing import preprocess_data
from utils import Dict2Class


def main(config, args):
    # For fast training.
    cudnn.benchmark = True
    if args.stage == 0:
        preprocess_data(config)
    elif args.stage == 1:
        data_loader = get_loader(config)
        solver = Solver(data_loader, args, config)
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(
            (
                "Using GPU %d (%s) of compute capability "
                + "%d.%d with %.1fGb total memory."
            )
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
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument("--num_iters", type=int, default=800000)
    parser.add_argument("--resume_iters", type=int, default=0)
    parser.add_argument("--resume_last", action="store_true", default=False)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--ckpt_save_step", type=int, default=20000)
    parser.add_argument("--stage", type=int, default=1, help="0: preprocessing; 1: training")
    parser.add_argument("--config_name", type=str, default="spsp2-large")
    parser.add_argument("--model_type", type=str, default="G", help="G: generator; F: f0 converter")
    # fmt:on
    args = parser.parse_args()

    config = yaml.safe_load(
        open(os.path.join("configs", f"{args.config_name}.yaml"), "r")
    )
    config = Dict2Class(config)

    if args.resume_last:
        print("resuming")
        args.resume_iters = int(
            sorted(
                [
                    int(key.split("-")[-1].split(".")[0])
                    for key in os.listdir(path=config.model_save_dir)
                ],
                reverse=True,
            )[0]
        )
        args.num_iters = int(args.num_iters - args.resume_iters)

    if args.model_type == "F":
        config.model_type = "F"
        # concatenate spectrogram and quantized pitch contour as the
        # f0 converter input
        config.dim_pit = config.dim_con + config.dim_pit
    __import__("pprint").pprint(config.__dict__)

    main(config, args)
