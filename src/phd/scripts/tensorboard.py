from tensorboard import program

from ..util import config


def tensorboard() -> None:
    tensorboard = program.TensorBoard()
    tensorboard.configure(argv=[None, "--logdir", config.paths.tensorboard])
    print("Tensorflow Start")
    code = tensorboard.main()
    print(f"Tensorflow Exited with {code}")
    exit(code)
