import os
import string
from difflib import SequenceMatcher
from math import sqrt
from typing import List, Optional, Self, Tuple

import torch

from transcribers import Transcriber, WhisperTranscriber

from .compute import Compute
from .config import Config
from .logging import Logger

ua_uttrs = getattr(
    __import__("meta_dicts"),
    "uaspeech_uttrs",
)


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
        model: Optional[Transcriber] = None,
    ) -> None:
        self.__config = Config()
        retries = 5
        compute = Compute()
        tmpdir = os.path.join(self.__config.paths.artefacts, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        transcriber = model or WhisperTranscriber(
            device=compute.device(),
            model_name=self.__config.options.whisper_type,
            config=self.__config,
            output_dir=tmpdir,
        )
        self.fname = fname
        self.loss_mse = torch.nn.functional.mse_loss(destin, source).item()
        self.truth_ground = self.get_real_text(fname)
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

    def get_real_text(
        self: Self,
        fname: str,
    ) -> str:
        if self.__config.options.dataset_name in ("uaspeech", "smolspeech"):
            uttr_code = self.fname.split("_")[1] + "_" + self.fname.split("_")[2]
            return clean_string(ua_uttrs[uttr_code])
        elif self.__config.options.dataset_name in ("vctk", "smolvctk"):
            uttr_code = self.fname
            full_text_path = "{}/VCTK-Corpus/txt/{}/{}.txt".format(
                self.__config.paths.raw_data,
                self.fname.split("_")[0],
                self.fname,
            )
            with open(full_text_path) as uttr_file:
                return clean_string(next(uttr_file))
        else:
            Logger().fatal(
                f"Dataset type not implemented: {self.__config.options.dataset_name}"
            )
            return ""

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
