import argparse

import requests
import torch
from torch.backends import cudnn

from data_preprocessing import preprocess_data
from solver import Solver
from swapper import Swapper
from util.config import Config, RunTests
from util.exception import NotifyError


def swapping(config: Config):
    swapper = Swapper(config)
    if RunTests.SAVE_LATENTS in config.options.run_tests:
        try:
            swapper.save_latents()
        except NotifyError as e:
            msg = f"Saving Latents failed with error: {e}"
            requests.post(
                config.options.ntfy_url,
                data=msg.encode(encoding="utf-8"),
            )
            raise e

    if RunTests.SWAP_LATENTS in config.options.run_tests:
        try:
            swapper.swap_latents()
        except NotifyError as e:
            msg = f"Swapping Latents failed with error: {e}"
            requests.post(
                config.options.ntfy_url,
                data=msg.encode(encoding="utf-8"),
            )
            raise e

    if RunTests.SAVE_AUDIOS in config.options.run_tests:
        try:
            swapper.save_audios()
        except NotifyError as e:
            msg = f"Saving Audios failed with error: {e}"
            requests.post(
                config.options.ntfy_url,
                data=msg.encode(encoding="utf-8"),
            )
            raise e


def main(config: Config):
    if config.options.run_tests == RunTests.NOTHING:
        return
    cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    preprocess_data(config)
    if RunTests.TRAIN in config.options.run_tests:
        try:
            solver = Solver(config)
            solver.train()
        except NotifyError as e:
            msg = f"Training failed with error: {e}"
            requests.post(
                config.options.ntfy_url,
                data=msg.encode(encoding="utf-8"),
            )
            raise e
    try:
        swapping(config)
    except Exception as e:
        raise e
    finally:
        msg = "Main Finished"
        requests.post(
            config.options.ntfy_url,
            data=msg.encode(encoding="utf-8"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config_name", type=str, default="base")
    args = parser.parse_args()
    config = Config(args.config_name)
    if args.trace or config.options.trace:
        hr = __import__("heartrate")
        hr.trace(files=hr.files.all)
    main(config)
    main(config)
    main(config)
