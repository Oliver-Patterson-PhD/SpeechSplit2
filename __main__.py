if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if dev is not None:
            torch.set_default_device(dev)

    from main import main

    main()
