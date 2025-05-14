def main() -> None:
    from argparse import ArgumentParser

    import torch

    from data import PreProcess
    from experiments import (Immediate, Scratchpad, Swapper,
                             SyllableEstimation, TestSamples, Train)
    from util import Compute, Config, Logger, RunTests

    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    config = Config("scratch" if args.scratch else args.config_name)
    doscratch = args.scratch
    logger = Logger()
    Compute().set_gpu()
    logger.debug("Starting Main")
    if config.options.run_tests == RunTests.NOTHING and not doscratch:
        return
    import os

    modelfiles = os.listdir(os.path.join(config.paths.full_models, "whisper"))
    logger.trace_var(modelfiles, level="DEBUG")

    try:
        imm = Immediate(config)
        imm.test()
        if imm.exit_after:
            logger.info("Returning from Immediate test")
            return

        PreProcess(config)

        if doscratch:
            scratch = Scratchpad(config)
            scratch.run()
            exit(0)

        if RunTests.SYLLABLE_ESTIMATION in config.options.run_tests:
            syllable = SyllableEstimation(config)
            syllable.run()

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
                pass
                # swapper.save_audios()
        logger.debug("Finished main")
    except Exception as e:
        logger.fatal(str(e.__cause__))
        raise Exception from e
