import os
from typing import Self

import torch

from models.whisper.audio import N_FRAMES
from models.whisper.loader import load_model
from models.whisper.model import Whisper
from models.whisper.transcribe import transcribe
from models.whisper.utils import ResultWriter


class Transcriber:
    device: torch.device
    model_name: str
    model_args: dict = {
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "beam_size": 5,
        "best_of": 5,
        "clip_timestamps": "0",
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "fp16": True,
        "hallucination_silence_threshold": None,
        "initial_prompt": None,
        "language": "en",
        "length_penalty": None,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "patience": None,
        "prepend_punctuations": "\"'“¿([{-",
        "suppress_tokens": "-1",
        "task": "transcribe",
        "temperature": 0.0,
        # "word_timestamps": True,
    }
    writer_args: dict = {
        "highlight_words": True,
        "max_line_count": None,
        "max_line_width": None,
        "max_words_per_line": 1,
    }
    writer: ResultWriter
    model: Whisper

    def __init__(
        self: Self,
        device: torch.device,
        model_name: str,
        config,
    ):
        self.device = device
        self.model_name = model_name
        self.model = load_model(
            name=self.model_name,
            device=self.device,
            download_root=os.path.join(config.paths.full_models, "whisper"),
        )

    def transcribe(
        self: Self,
        melspec: torch.Tensor,
        name: str,
    ):
        if melspec.shape[-1] == self.model.dims.n_mels:
            melspec = melspec.mT
        bigsize: int = max(melspec.size())
        padding: int = max(((2 * N_FRAMES) - bigsize), bigsize)
        padded_melspec = torch.nn.functional.pad(
            melspec.squeeze(),
            (0, padding),
        )
        from util import Logger

        logger = Logger()
        result = transcribe(
            self.model,
            mel=padded_melspec,
            **self.model_args,
        )
        if len(result["text"]) == 0:
            logger.trace(f"Unable to transcribe: {tuple(padded_melspec.shape)}, {name}")

        out_str = ""
        for segment in result["segments"]:
            out_str = out_str + segment["text"].strip() + " "
        out_tok = []
        for segment in result["segments"]:
            out_tok.extend(segment["tokens"])
        return out_str, out_tok
