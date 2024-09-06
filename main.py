import argparse

import torch
from torch.backends import cudnn

from data_preprocessing import preprocess_data
from util.config import Config, RunTests
from util.exception import NotifyError
from util.logging import Logger
from util.notify import Notifier


def main(config: Config, notifier: Notifier):
    if config.options.run_tests == RunTests.NOTHING:
        return
    cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    preprocess_data(config)

    if RunTests.TRAIN in config.options.run_tests:
        from experiments.train import Train

        try:
            trainer = Train(config)
            trainer.train()
        except NotifyError as e:
            msg = f"Training failed with error: {e}"
            Logger().error(msg)
            notifier.send(msg)
            raise e

    from experiments.swapper import Swapper

    swapper = Swapper(config)
    if RunTests.SAVE_LATENTS in config.options.run_tests:
        try:
            swapper.save_latents()
        except NotifyError as e:
            msg = f"Saving Latents failed with error: {e}"
            Logger().error(msg)
            notifier.send(msg)
            raise e

    if RunTests.SWAP_LATENTS in config.options.run_tests:
        try:
            swapper.swap_latents()
        except NotifyError as e:
            msg = f"Swapping Latents failed with error: {e}"
            Logger().error(msg)
            notifier.send(msg)
            raise e

    if RunTests.SAVE_AUDIOS in config.options.run_tests:
        try:
            swapper.save_audios()
        except NotifyError as e:
            msg = f"Saving Audios failed with error: {e}"
            Logger().error(msg)
            notifier.send(msg)
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config_name", type=str, default="base")
    args = parser.parse_args()
    config = Config(args.config_name)
    notifier = Notifier()
    if args.trace or config.options.trace:
        hr = __import__("heartrate")
        hr.trace(files=hr.files.all)
    try:
        main(config=config, notifier=notifier)
    except Exception as e:
        msg = f"Could not successfully finish main due to: {e}"
        Logger().error(msg)
        notifier.send(msg)
    finally:
        msg = "Main Finished"
        Logger().error(msg)
        notifier.send(msg)
