__all__ = [
    "CompareItem",
]

import string
from difflib import SequenceMatcher
from math import sqrt

import torch

from util.compute import Compute
from util.config import Config
from util.tensor import Tensor

from .transcriber import Transcriber


class CompareItem:
    fname: str
    truth_ground: str
    trans_item1: str | None = None
    trans_item2: str | None = None
    token_item1: list[int] = []
    token_item2: list[int] = []
    loss_mse: float | None = None
    loss_gse: float | None = None
    loss_gte: float | None = None
    loss_tok: float | None = None
    __config: Config
    max_retries: int = 5
    transcriber: Transcriber

    @torch.no_grad()
    def __init__(
        self,
        fname: str,
        mel1: Tensor,
        mel2: Tensor,
        text: str,
        model: Transcriber | None = None,
        name1: str = "Sample 1",
        name2: str = "Sample 2",
    ) -> None:
        self.__config = Config()
        compute = Compute()
        if model is None:
            transcriber = Transcriber(compute.device(), self.__config)
        else:
            transcriber = model
        self.name1 = name1
        self.name2 = name2
        self.fname = fname
        self.transcriber = transcriber
        self.loss_mse = torch.nn.functional.mse_loss(mel2, mel1).item()
        self.trans_truth = clean_string(text)
        self.trans_item1, self.token_item1 = self.get_transcription(mel1, "spk1")
        self.trans_item2, self.token_item2 = self.get_transcription(mel2, "spk2")
        seq_mat_str = SequenceMatcher()
        seq_mat_str.set_seqs(self.trans_item1, self.trans_item2)
        self.loss_gestalt_string = seq_mat_str.ratio()
        seq_mat_tok = SequenceMatcher()
        seq_mat_tok.set_seqs(
            "".join([chr(item1_tok) for item1_tok in self.token_item1]),
            "".join([chr(item2_tok) for item2_tok in self.token_item2]),
        )
        self.loss_gestalt_token = seq_mat_tok.ratio()
        self.loss_custom_token, corr_mat = corr_calc(
            self.token_item1,
            self.token_item2,
        )

    def get_transcription(self, melspec: Tensor, item: str) -> tuple[str, list[int]]:
        transcription_attempts: int = 0
        while True:
            transcription_attempts += 1
            transcription, tokens = self.transcriber.transcribe(
                melspec, f"{self.fname}_{item}"
            )
            if transcription is not None:
                clean_transcription = clean_string(transcription)
                if clean_transcription is not None:
                    return clean_transcription, tokens
            if transcription_attempts <= self.max_retries:
                continue
        raise RuntimeError(
            f"Could not transcribe {item} after {transcription_attempts} attempts"
        )

    def file_text(self) -> tuple[str, str]:
        return self.fname, self.__str__()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        spaces1 = (20 - len(self.name1)) * " "
        spaces2 = (20 - len(self.name2)) * " "
        return (
            f"True Text:                  {self.trans_truth}\n"
            f"Source Transcription:       {self.trans_item1}\n"
            f"Processed Transcription:    {self.trans_item2}\n"
            f"{self.name1} Tokens:{spaces1}{self.token_item1}\n"
            f"{self.name2} Tokens:{spaces2}{self.token_item2}\n"
            f"Loss Mean Square Error:     {self.loss_mse or 0:.5f}\n"
            f"Similarity Gestalt String:  {self.loss_gestalt_string or 0:.5f}\n"
            f"Similarity Gestalt Token:   {self.loss_gestalt_token or 0:.5f}\n"
            f"Loss Custom Token:          {self.loss_custom_token or 0:.5f}\n"
        )


@torch.no_grad()
def corr_calc(gt_list: list[int], wp_list: list[int]) -> tuple[float, Tensor]:
    x = torch.tensor(gt_list, dtype=torch.float)
    y = torch.tensor(wp_list, dtype=torch.float)
    gt_m, wp_m = torch.meshgrid(x, y, indexing="ij")
    m_corr = torch.abs(torch.sub(gt_m, wp_m))
    for yval in range(len(y) - 1):
        for xval in range(len(x) - 1):
            m_corr[xval, yval] = m_corr[xval, yval] * (
                1 + distance_to_diagonal(len(y), len(x), yval, xval)
            )
    return 1.0, m_corr


def distance_to_diagonal(width: int, height: int, xval: int, yval: int) -> float:
    return abs(((height / width) * (xval + 0.5)) - (yval + 0.5)) / sqrt(
        1 + (height / width) ** 2
    )


@torch.no_grad()
def clean_string(instring: str) -> str:
    stripthese = string.punctuation
    outstr: str = "".join([char for char in instring if char not in stripthese])
    return outstr.lower().strip()
