import argparse

import torch

from data_preprocessing import preprocess_data
from experiments.scratchpad import Scratchpad
from experiments.swapper import Swapper
from experiments.test_samples import TestSamples
from experiments.train import Train
from util.config import Config, RunTests
from util.exception import NotifyError
from util.logging import Logger
from util.notify import Notifier

global notifier
global doscratch


def notify(msg: str) -> None:
    global notifier
    Logger().error(msg, depth=2)
    notifier.send(msg)


def main(config: Config):
    global doscratch
    Logger().debug("Starting Main")
    if config.options.run_tests == RunTests.NOTHING and not doscratch:
        return

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    preprocess_data(config)

    if doscratch:
        scratch = Scratchpad(config)
        scratch.run()
        exit(0)

    if RunTests.TEST in config.options.run_tests:
        tester = TestSamples(config)
        tester.test()

    if RunTests.TRAIN in config.options.run_tests:
        try:
            trainer = Train(config)
            trainer.train()
        except NotifyError as e:
            notify(f"Training failed with error: {e}")

    run_saves: bool = (
        (RunTests.SAVE_LATENTS | RunTests.SWAP_LATENTS | RunTests.SAVE_AUDIOS)
        & config.options.run_tests
    ).__bool__()
    if run_saves:
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
    global doscratch
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    config = Config("scratch" if args.scratch else args.config_name)
    doscratch = args.scratch
    if args.trace or config.options.trace:
        hr = __import__("heartrate")
        hr.trace(files=hr.files.all)
    notifier = Notifier()
    main(config=config)
