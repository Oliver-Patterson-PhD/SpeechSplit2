import torch

from ...util import config
from .decoder_speechsplit import SpeechSplitDecoder
from .encoder_rhythm import EncoderRhythm
from .encoder_sync import EncoderSync


class SpeechSplit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = SpeechSplitDecoder()
        self.encoder_1 = EncoderSync()
        self.encoder_2 = EncoderRhythm()
        self.freq = config.model.freq_1
        self.freq_2 = config.model.freq_2
        self.freq_3 = config.model.freq_3
        self.return_latents = config.options.return_latents

    def forward(
        self,
        x_f0: torch.Tensor,
        x_org: torch.Tensor,
        c_trg: torch.Tensor,
        rr: bool = True,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        x_1 = x_f0.transpose(-1, -2)
        codes_x, codes_f0 = self.encoder_1(x_1, rr)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        x_2 = x_org.transpose(-1, -2)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        code_exp_4 = c_trg.unsqueeze(-2).expand(-1, x_1.size(-1), -1)
        encoder_outputs = torch.cat(
            (
                code_exp_1,
                code_exp_2,
                code_exp_3,
                code_exp_4,
            ),
            dim=-1,
        )
        mel_outputs = self.decoder(encoder_outputs)
        if self.return_latents is False:
            return mel_outputs
        else:
            return mel_outputs, code_exp_1, code_exp_2, code_exp_3, code_exp_4

    def rhythm(self, x_org: torch.Tensor) -> torch.Tensor:
        x_2 = x_org.transpose(-1, -2)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=-2)
        return code_exp_2

    def content_pitch(
        self, x_f0: torch.Tensor, rr: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_1 = x_f0.transpose(-1, -2)
        codes_x, codes_f0 = self.encoder_1(x_1, rr)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=-2)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=-2)
        return code_exp_1, code_exp_3

    def decode(
        self,
        code_exp_1: torch.Tensor,
        code_exp_2: torch.Tensor,
        code_exp_3: torch.Tensor,
        c_trg: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        encoder_outputs = torch.cat(
            (code_exp_1, code_exp_2, code_exp_3, c_trg.expand(-1, T, -1)),
            dim=-1,
        )
        mel_outputs = self.decoder(encoder_outputs)
        return mel_outputs
