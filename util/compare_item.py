__all__ = [
    "CompareItem",
]

import os
import string
from difflib import SequenceMatcher
from math import sqrt
from typing import TYPE_CHECKING, List, Optional, Self, Tuple

import torch

from .compute import Compute
from .config import Config

if TYPE_CHECKING:
    from transcribers import Transcriber


class CompareItem:
    fname: str
    truth_ground: str
    trans_source: Optional[str] = None
    trans_destin: Optional[str] = None
    token_source: List[int] = []
    token_destin: List[int] = []
    loss_mse: Optional[float] = None
    loss_gse: Optional[float] = None
    loss_gte: Optional[float] = None
    loss_tok: Optional[float] = None
    __config: Config

    @torch.no_grad()
    def __init__(
        self: Self,
        fname: str,
        source: torch.Tensor,
        destin: torch.Tensor,
        text: str,
        model: Optional["Transcriber"] = None,
    ) -> None:
        self.__config = Config()
        retries = 5
        compute = Compute()
        tmpdir = os.path.join(self.__config.paths.artefacts, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        if model is None and not TYPE_CHECKING:
            from transcribers import Transcriber
        if model is None:
            transcriber = Transcriber(
                device=compute.device(),
                model_name=self.__config.options.whisper_type,
                config=self.__config,
            )
        else:
            transcriber = model
        self.fname = fname
        self.loss_mse = torch.nn.functional.mse_loss(destin, source).item()
        self.truth_ground = text
        source_transcription, self.token_source = transcriber.transcribe(
            source, f"{self.fname}_gt"
        )
        self.trans_source = clean_string(source_transcription)
        transcription_attempts = 1
        while len(self.token_destin) == 0 and transcription_attempts < retries:
            transcription_attempts += 1
            self.trans_destin, self.token_destin = transcriber.transcribe(
                destin, f"{self.fname}_out"
            )
            assert self.trans_destin is not None
            self.trans_destin = clean_string(self.trans_destin)
        if transcription_attempts >= retries:
            self.trans_destin = None
        if self.trans_source is not None and self.trans_destin is not None:
            seq_mat_str = SequenceMatcher()
            seq_mat_str.set_seqs(self.trans_source, self.trans_destin)
            self.loss_gse = seq_mat_str.ratio()
            seq_mat_tok = SequenceMatcher()
            seq_mat_tok.set_seqs(
                "".join([chr(gt_tok) for gt_tok in self.token_source]),
                "".join([chr(wp_tok) for wp_tok in self.token_destin]),
            )
            self.loss_gte = seq_mat_tok.ratio()
            self.loss_tok, corr_mat = corr_calc(
                self.token_source,
                self.token_destin,
            )
        return

    def file_text(
        self: Self,
    ) -> Tuple[str, str]:
        return self.fname, self.__str__()

    def __str__(
        self: Self,
    ) -> str:
        return self.__repr__()

    def __repr__(
        self: Self,
    ) -> str:
        return (
            "True Text:                {}\n"
            "Source Transcription:     {}\n"
            "Processed Transcription:  {}\n"
            "Source Tokens:            {}\n"
            "Processed Tokens:         {}\n"
            "Loss Mean Square Error:   {}\n"
            "Loss Gestalt String:      {}\n"
            "Loss Gestalt Token:       {}\n"
            "Loss Custom Token:        {}\n"
        ).format(
            self.truth_ground,
            self.trans_source,
            self.trans_destin,
            self.token_source,
            self.token_destin,
            f"{self.loss_mse:.5f}" if self.loss_mse is not None else "None",
            f"{self.loss_gse:.5f}" if self.loss_gse is not None else "None",
            f"{self.loss_gte:.5f}" if self.loss_gte is not None else "None",
            f"{self.loss_tok:.5f}" if self.loss_tok is not None else "None",
        )


@torch.no_grad()
def corr_calc(
    gt_list: List[int],
    wp_list: List[int],
) -> tuple[int, torch.Tensor]:
    x = torch.tensor(gt_list, dtype=torch.float)
    y = torch.tensor(wp_list, dtype=torch.float)
    gt_m, wp_m = torch.meshgrid(x, y, indexing="ij")
    m_corr = torch.abs(torch.sub(gt_m, wp_m))
    for yval in range(len(y) - 1):
        for xval in range(len(x) - 1):
            m_corr[xval, yval] = m_corr[xval, yval] * (
                1 + distance_to_diagonal(len(y), len(x), yval, xval)
            )
    return -1, m_corr


def distance_to_diagonal(
    width: int,
    height: int,
    xval: int,
    yval: int,
) -> float:
    return abs(((height / width) * (xval + 0.5)) - (yval + 0.5)) / sqrt(
        1 + (height / width) ** 2
    )


@torch.no_grad()
def clean_string(
    instring: str,
) -> str:
    stripthese = string.punctuation
    outstr: str = "".join([char for char in instring if char not in stripthese])
    return outstr.lower().strip()
