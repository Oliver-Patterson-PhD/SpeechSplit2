from typing import Self

import torch

from transcribers.transcriber import Transcriber
from transcribers.whisper.audio import N_FRAMES
from util.config import Config
from util.logging import Logger

from .loader import load_model
from .transcribe import transcribe


class WhisperTranscriber(Transcriber):
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

    def __init__(
        self: Self,
        device: torch.device,
        model_name: str,
        config: Config,
        output_dir: str,
    ):
        super(WhisperTranscriber, self).__init__(
            device=device,
            model_name=model_name,
            config=config,
        )
        self.model = load_model(
            name=self.model_name,
            device=self.device,
            download_root=f"{self.config.paths.full_models}/whisper",
        )
        from .utils import get_writer

        output_format = "txt"  # txt, vtt, srt, tsv, json, all
        self.output_dir = output_dir
        self.writer = get_writer(output_format, self.output_dir)

    def transcribe(
        self: Self,
        melspec: torch.Tensor,
        name: str,
    ) -> None:
        padding: int = max(
            int((N_FRAMES - melspec.size(-1))),
            int(melspec.size(-1)),
        )
        padded_melspec = torch.nn.functional.pad(
            melspec.squeeze(),
            (0, 0, 0, padding),
        )
        logger = Logger()
        logger.trace_var(padding)
        logger.trace_tensor(melspec)
        logger.trace_tensor(padded_melspec)
        result = transcribe(
            self.model,
            mel=padded_melspec,
            **self.model_args,
        )
        if len(result["text"]) > 0:
            Logger().trace(f"Text: {result["text"]}")
            self.writer(
                result=result,
                name=name,
                **self.writer_args,
            )
        else:
            Logger().trace("Unable to transcribe")
