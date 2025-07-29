import torch
import tensorflow as tf


def check_tensorflow() -> None:
    print("TENSORFLOW:")
    print(f"\tVERSION: {tf.__version__}")
    devices = tf.config.list_physical_devices('GPU')
    print(f"\tDEVICES: {devices}")


def check_pytorch() -> None:
    print("PYTORCH:")
    print(f"\tCUDA: {torch.cuda.is_available()}")
    print(f"\tTORCH CAPABILITIES: {torch.cuda.get_arch_list()}")
    print(f"\tDEFAULT DEVICE ID: {torch.cuda.current_device()}")
    for device_id in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_name = gpu_properties.name
        gpu_memory = gpu_properties.total_memory / (1024**3)
        gpu_arch = gpu_properties.gcnArchName
        print(f"\tDEVICE ID: {device_id}")
        print(f"\t\tGPU NAME: {gpu_name}")
        print(f"\t\tGPU MEMORY: {gpu_memory:.2f}Gb")
        minor, major = gpu_properties.major, gpu_properties.minor
        print(f"\t\tGPU VERSION: {major}.{minor}")
        print(f"\t\tGPU ARCH: {gpu_arch}")


def check_general() -> None:
    from os import name
    from platform import machine, python_implementation, release
    from sys import implementation, platform

    print(f"GENERAL: {name}")
    print(f"\tOS NAME: {name}")
    print(f"\tSYS PLATFORM: {platform}")
    print(f"\tPLATFORM RELEASE: {release()}")
    print(f"\tIMPLEMENTATION NAME: {implementation.name}")
    print(f"\tPLATFORM MACHINE: {machine()}")
    print(f"\tPLATFORM PYTHON IMPLEMENTATION: {python_implementation()}")


def test_tensorflow() -> None:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])
    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)

def check() -> None:
    print()
    check_general()
    print()
    check_tensorflow()
    print()
    check_pytorch()
    print()
    test_tensorflow()
