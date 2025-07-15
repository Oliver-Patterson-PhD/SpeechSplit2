import torch

torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_start_method("spawn")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    if dev is not None:
        torch.set_default_device(dev)


def test() -> None:
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


if __name__ == "__main__":
    test()
