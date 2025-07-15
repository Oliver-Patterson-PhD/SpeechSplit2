import torch

torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_start_method("spawn")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    if dev is not None:
        torch.set_default_device(dev)


def check() -> None:
    from os import name
    from platform import machine, python_implementation, release
    from sys import implementation, platform

    print(f"OS NAME: {name}")
    print(f"SYS PLATFORM: {platform}")
    print(f"PLATFORM RELEASE: {release()}")
    print(f"IMPLEMENTATION NAME: {implementation.name}")
    print(f"PLATFORM MACHINE: {machine()}")
    print(f"PLATFORM PYTHON IMPLEMENTATION: {python_implementation()}")
    print()
    print("PYTORCH:")
    print(f"\tCUDA: {torch.cuda.is_available()}")
    print(f"\tTORCH CAPABILITIES: {torch.cuda.get_arch_list()}")
    print(f"\tDEFAULT DEVICE ID: {torch.cuda.current_device()}")
    for device_id in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_name = gpu_properties.name
        gpu_memory = (gpu_properties.total_memory / 1e9,)
        print(f"\tDEVICE ID: {device_id}")
        print(f"\t\tGPU NAME: {gpu_name}")
        print(f"\t\tGPU MEMORY: {gpu_memory}")
        minor, major = gpu_properties.major, gpu_properties.minor
        print(f"\t\tGPU VERSION: {major}.{minor}")
    return


def main() -> None:
    from argparse import ArgumentParser

    import torch
    from data import PreProcess
    from experiments import (DisVoiceTest, Immediate, Scratchpad, Swapper,
                             SyllableEstimation, TestSamples, Train,
                             TranscriptionLoss)
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

        if RunTests.DISVOICE in config.options.run_tests:
            disvoice = DisVoiceTest(config)
            disvoice.run()

        PreProcess(config)

        if doscratch:
            scratch = Scratchpad(config)
            scratch.run()
            exit(0)

        if RunTests.SYLLABLE_ESTIMATION in config.options.run_tests:
            syllable = SyllableEstimation(config)
            syllable.run()

        if RunTests.TRANSCRIPTION_LOSS in config.options.run_tests:
            transcription = TranscriptionLoss(config)
            transcription.run()

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
