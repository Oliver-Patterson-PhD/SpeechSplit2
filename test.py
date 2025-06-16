def main() -> None:
    return


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore
    torch.multiprocessing.set_start_method("spawn")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.set_default_device(dev)

    main()
