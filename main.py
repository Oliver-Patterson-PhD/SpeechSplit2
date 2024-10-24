import argparse

import torch

from data_preprocessing import preprocess_data
from experiments.scratchpad import Scratchpad
from experiments.swapper import Swapper
from experiments.test_samples import TestSamples
from experiments.train import Train
from util.config import Config, RunTests
from util.logging import Logger

global doscratch


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
        trainer = Train(config)
        trainer.train()

    run_saves: bool = (
        (RunTests.SAVE_LATENTS | RunTests.SWAP_LATENTS | RunTests.SAVE_AUDIOS)
        & config.options.run_tests
    ).__bool__()
    if run_saves:
        swapper = Swapper(config)
        if RunTests.SAVE_LATENTS in config.options.run_tests:
            swapper.save_latents()
        if RunTests.SWAP_LATENTS in config.options.run_tests:
            swapper.swap_latents()
        if RunTests.SAVE_AUDIOS in config.options.run_tests:
            swapper.save_audios()
    Logger().debug("Finished main")


if __name__ == "__main__":
    global doscratch
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    config = Config("scratch" if args.scratch else args.config_name)
    doscratch = args.scratch
    main(config=config)
