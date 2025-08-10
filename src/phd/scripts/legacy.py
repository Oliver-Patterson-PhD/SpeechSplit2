import torch

from ..data import preprocess_data
from ..experiments import (DisVoiceTest, Immediate, Swapper,
                           SyllableEstimation, TestSamples, Train,
                           TranscriptionLoss)
from ..util import compute, config, logger
from ..util.config import RunTests


def main() -> None:
    torch.backends.cudnn.benchmark = True
    compute.set_gpu()
    logger.debug("Starting Main")
    if config.options.run_tests == RunTests.NOTHING:
        return
    import os

    modelfiles = os.listdir(os.path.join(config.paths.full_models, "whisper"))
    logger.trace_var(modelfiles, level="DEBUG")

    try:
        imm = Immediate()
        imm.test()
        if imm.exit_after:
            logger.info("Returning from Immediate test")
            return

        if RunTests.DISVOICE in config.options.run_tests:
            disvoice = DisVoiceTest()
            disvoice.run()

        preprocess_data()

        if RunTests.SYLLABLE_ESTIMATION in config.options.run_tests:
            syllable = SyllableEstimation()
            syllable.run()

        if RunTests.TRANSCRIPTION_LOSS in config.options.run_tests:
            transcription = TranscriptionLoss()
            transcription.run()

        if RunTests.TEST in config.options.run_tests:
            tester = TestSamples()
            tester.test()

        if RunTests.TRAIN in config.options.run_tests:
            trainer = Train()
            trainer.train()

        run_saves: bool = (
            (RunTests.SAVE_LATENTS | RunTests.SWAP_LATENTS | RunTests.SAVE_AUDIOS)
            & config.options.run_tests
        ).__bool__()
        if run_saves:
            swapper = Swapper()
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
