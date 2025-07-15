__all__ = [
    "Whisper",
    "load_whisper",
]

from .model import Whisper
from .loader import load_model as load_whisper
