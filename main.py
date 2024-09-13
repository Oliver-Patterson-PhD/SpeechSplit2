import argparse

import torch

from data_preprocessing import preprocess_data
from experiments.swapper import Swapper
from experiments.test_samples import TestSamples
from experiments.train import Train
from util.config import Config, RunTests
from util.exception import NotifyError
from util.logging import Logger
from util.notify import Notifier

global notifier


def notify(msg: str) -> None:
    global notifier
    Logger().error(msg, depth=2)
    notifier.send(msg)


def main(config: Config):
    Logger().debug("Starting Main")
    if config.options.run_tests == RunTests.NOTHING:
        return

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    preprocess_data(config)

    if RunTests.TEST in config.options.run_tests:
        tester = TestSamples(config)
        tester.test()

    if RunTests.TRAIN in config.options.run_tests:
        try:
            trainer = Train(config)
            trainer.train()
        except NotifyError as e:
            notify(f"Training failed with error: {e}")

    if (
        (RunTests.SAVE_LATENTS | RunTests.SWAP_LATENTS | RunTests.SAVE_AUDIOS)
        & config.options.run_tests
    ).__bool__():
        swapper = Swapper(config)
        try:
            if RunTests.SAVE_LATENTS in config.options.run_tests:
                swapper.save_latents()
            if RunTests.SWAP_LATENTS in config.options.run_tests:
                swapper.swap_latents()
            if RunTests.SAVE_AUDIOS in config.options.run_tests:
                swapper.save_audios()
        except NotifyError as e:
            notify(f"Swapper failed with error: {e}")
    Logger().debug("Finished main")


if __name__ == "__main__":
    global notifier
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config_name", type=str, default="base")
    args = parser.parse_args()
    config = Config(args.config_name)
    if args.trace or config.options.trace:
        hr = __import__("heartrate")
        hr.trace(files=hr.files.all)
    notifier = Notifier()
    main(config=config)
