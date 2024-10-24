import os
import string
from difflib import SequenceMatcher
from typing import List, Self

import torch

from data_loader import CollaterItemType
from experiments.experiment import Experiment
from transcribers.transcriber import Transcriber
from transcribers.whisper import WhisperTranscriber


class Scratchpad(Experiment):
    batch_size: int
    transcriber: Transcriber
    unfold_spmel_gt = torch.nn.Unfold(
        kernel_size=(192, 80),
        dilation=1,
        padding=0,
        stride=1,
    )
    fold_spmel_gt = torch.nn.Fold(
        output_size=0,
        kernel_size=(192, 80),
        dilation=1,
        padding=0,
        stride=1,
    )

    @torch.no_grad()
    def run(self: Self) -> None:
        self.transcriber = WhisperTranscriber(
            device=self.compute.device(),
            model_name=self.config.options.whisper_type,
            config=self.config,
            output_dir=self.experiment_dir,
        )
        ckpt_name = "{2}-{1}/{0}/{0}-{1}-{2}-{3}.ckpt".format(
            self.config.options.experiment,
            self.config.options.bottleneck,
            self.config.options.model_type,
            self.config.options.resume_iters,
        )
        model_name = os.path.join(self.config.paths.models, ckpt_name)
        self.restore_model(model_name=model_name)
        # self.logger.info("Full Process Off")
        # self.process(False)
        self.logger.info("Full Process On")
        self.process(True)

    @torch.no_grad()
    def process(self: Self, full_process: bool) -> None:
        loss_fn = torch.nn.MSELoss(reduction="mean")
        ua_uttrs = getattr(
            __import__("meta_dicts"),
            "uaspeech_uttrs",
        )
        self.load_data(singleitem=True, full_process=full_process)
        for item in self.data_loader:
            if full_process:
                (
                    i_fnamelist,
                    i_spk_id_org,
                    i_spmel_gt,
                    i_rhythm_input,
                    i_content_input,
                    i_pitch_input,
                    i_timbre_input,
                    i_len_crop,
                ) = self.make_stack(item)
            else:
                (
                    i_fnamelist,
                    i_spk_id_org,
                    i_spmel_gt,
                    i_rhythm_input,
                    i_content_input,
                    i_pitch_input,
                    i_timbre_input,
                    i_len_crop,
                ) = item
            outitem = []
            for stackitem in zip(
                i_fnamelist,
                i_spk_id_org,
                i_spmel_gt,
                i_rhythm_input,
                i_content_input,
                i_pitch_input,
                i_timbre_input,
                i_len_crop,
            ):
                (
                    fnamelist,
                    spk_id_org,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop,
                ) = stackitem

                fname: str = fnamelist.split("/")[-1].split(".")[0]
                # =============================================================== #
                #                   1. Load input data                            #
                # =============================================================== #
                spmel_gt = spmel_gt.to(self.compute.device())
                rhythm_input = rhythm_input.to(self.compute.device())
                content_input = content_input.to(self.compute.device())
                pitch_input = pitch_input.to(self.compute.device())
                timbre_input = timbre_input.to(self.compute.device())
                len_crop = len_crop.to(self.compute.device())
                pitch_input.unsqueeze_(-1)
                spmel_gt.unsqueeze_(0)
                rhythm_input.unsqueeze_(0)
                content_input.unsqueeze_(0)
                pitch_input.unsqueeze_(0)
                timbre_input.unsqueeze_(0)

                # self.logger.trace_var(fname)
                # self.logger.trace_var(spk_id_org)

                # self.logger.trace_tensor(spmel_gt)
                # self.logger.trace_tensor(rhythm_input)
                # self.logger.trace_tensor(content_input)
                # self.logger.trace_tensor(pitch_input)
                # self.logger.trace_tensor(timbre_input)
                # self.logger.trace_tensor(len_crop)

                # Prepare input data and apply random resampling
                content_pitch_input = self.prepare_input(
                    content_input,
                    pitch_input,
                    len_crop,
                )

                if self.config.options.return_latents:
                    (
                        spmel_output,
                        code_exp_1,
                        code_exp_2,
                        code_exp_3,
                        code_exp_4,
                    ) = self.model(
                        content_pitch_input,
                        rhythm_input,
                        timbre_input,
                    )
                else:
                    spmel_output = self.model(
                        content_pitch_input,
                        rhythm_input,
                        timbre_input,
                    )
                outitem.append((spmel_gt.squeeze(), spmel_output.squeeze()))
            out_gt, out_out = self.unstack(outitem)
            fname = fnamelist.split("/")[-1].split(".")[0]
            uttr_code = fname.split("_")[1] + "_" + fname.split("_")[2]
            loss: float = loss_fn(out_gt, out_out).item()
            truth_str = self.clean_string(ua_uttrs[uttr_code])
            trans_gt_str, trans_gt_tok = self.transcriber.transcribe(
                out_gt, f"{fname}_gt"
            )
            trans_wp_str, trans_wp_tok = self.transcriber.transcribe(
                out_out, f"{fname}_out"
            )
            str_gt_tok = "".join([chr(gt_tok) for gt_tok in trans_gt_tok])
            str_wp_tok = "".join([chr(wp_tok) for wp_tok in trans_wp_tok])
            self.logger.debug(f"Running: {fname}, loss: {loss:.8f}")
            self.logger.debug(f"Utterance cd: {uttr_code}")
            self.logger.debug(f"Desired Text: {truth_str}")

            trans_gt = self.clean_string(trans_gt_str)
            trans_wp = self.clean_string(trans_wp_str)
            self.logger.debug(f"Original : {trans_gt}")
            self.logger.debug(f"Processed: {trans_wp}")

            self.logger.debug(f"Original  Tokens: {trans_gt_tok}")
            self.logger.debug(f"Processed Tokens: {trans_wp_tok}")
            self.logger.info(f"Len GT: {len(trans_gt_tok)}")
            self.logger.info(f"Len WP: {len(trans_wp_tok)}")

            diff_str = SequenceMatcher()
            diff_str.set_seqs(trans_gt_str, trans_wp_str)
            str_loss = diff_str.ratio()
            diff_tok = SequenceMatcher()
            diff_tok.set_seqs(str_gt_tok, str_wp_tok)
            tok_loss = diff_tok.ratio()

            self.logger.info(f"Loss Str: {str_loss}")
            self.logger.info(f"Loss Tok: {tok_loss}")
            tok_corr, corr_mat = self.corr_calc(
                trans_gt_tok,
                trans_wp_tok,
            )
            self.logger.info(f"Tokens Correlation: {tok_corr}")
            self.logger.trace_tensor(corr_mat)
            self.logger.debug("\n" + str(corr_mat))

    @torch.no_grad()
    def clean_string(self: Self, instring: str) -> str:
        stripthese = string.punctuation
        outstr: str = "".join([char for char in instring if char not in stripthese])
        return outstr.lower()

    @torch.no_grad()
    def make_stack(self: Self, item: CollaterItemType) -> CollaterItemType:
        fnamelist: List[str]
        spk_id_org: List[str]
        spmel_gt: torch.Tensor  # batch, max_len_pad, n_mels
        rhythm_input: torch.Tensor  # batch, max_len_pad, n_mels
        content_input: torch.Tensor  # batch, max_len_pad, n_mels
        pitch_input: torch.Tensor  # batch, max_len_pad, 1
        timbre_input: torch.Tensor  # batch, n_mels + 2
        len_crop: torch.Tensor  # batch
        (
            fnamelist,
            spk_id_org,
            spmel_gt,
            rhythm_input,
            content_input,
            pitch_input,
            timbre_input,
            len_crop,
        ) = item
        assert len(fnamelist) == 1
        # self.logger.trace_tensor(spmel_gt)
        # self.logger.trace_tensor(rhythm_input)
        # self.logger.trace_tensor(content_input)
        # self.logger.trace_tensor(pitch_input)
        i_fnamelist = fnamelist
        i_spk_id_org = spk_id_org
        i_spmel_gt = self.fold_pad_len(spmel_gt, self.config.model.max_len_pad)
        i_rhythm_input = self.fold_pad_len(rhythm_input, self.config.model.max_len_pad)
        i_content_input = self.fold_pad_len(
            content_input, self.config.model.max_len_pad
        )
        i_pitch_input = self.fold_pad_len(pitch_input, self.config.model.max_len_pad)
        i_timbre_input = timbre_input
        i_len_crop = len_crop
        # self.logger.trace_tensor(i_spmel_gt)
        # self.logger.trace_tensor(i_rhythm_input)
        # self.logger.trace_tensor(i_content_input)
        # self.logger.trace_tensor(i_pitch_input)
        return (
            i_fnamelist,
            i_spk_id_org,
            i_spmel_gt,
            i_rhythm_input,
            i_content_input,
            i_pitch_input,
            i_timbre_input,
            i_len_crop,
        )

    @torch.no_grad()
    def unstack(
        self: Self,
        listitem: List[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_gt = torch.zeros(
            size=(0, listitem[0][0].size(-1)),
            dtype=listitem[0][0].dtype,
            device=listitem[0][0].device,
        )
        out_out = torch.zeros(
            size=(0, listitem[0][1].size(-1)),
            dtype=listitem[0][1].dtype,
            device=listitem[0][1].device,
        )
        # self.logger.trace("Initial")
        # self.logger.trace_tensor(out_gt)
        # self.logger.trace_tensor(out_out)
        for gt_item, out_item in listitem:
            out_gt = torch.cat((out_gt, gt_item), -2)
            out_out = torch.cat((out_out, out_item), -2)
        # self.logger.trace("final")
        # self.logger.trace_tensor(out_gt)
        # self.logger.trace_tensor(out_out)
        return out_gt, out_out

    @torch.no_grad()
    def fold_pad_len(self: Self, item: torch.Tensor, len: int) -> torch.Tensor:
        if item.size(dim=0) >= len:
            self.logger.trace(f"Unfolding tensor: {item.size()}")
            retval = self.unfold_spmel_gt(item)
        else:
            self.logger.trace(f"Padding tensor: {item.size()}")
            if item.ndim == 3:
                retval = torch.nn.functional.pad(item, (0, 0, 0, len - item.size(-2)))
            elif item.ndim == 2:
                retval = torch.nn.functional.pad(item, (0, len - item.size(-1)))
            else:
                raise ValueError
        self.logger.trace(f"Final tensor {retval.size()}")
        return retval

    @torch.no_grad()
    def get_real_text(self: Self, fname: str):
        if self.config.options.dataset_name == "vctk":
            fname = os.path.splitext(fname)[0]
            spcode = fname[0:2]
            fullpath = "{}/VCTK-Corpus/txt/{}/{}.txt".format(
                self.config.paths.raw_data, spcode, fname
            )
            with open(fullpath, "rb") as txtfile:
                return str(txtfile.read()).replace("\n", "")
        else:
            raise NotImplementedError

    @torch.no_grad()
    def test_whisper(self: Self) -> None:
        for item in self.data_loader:
            (
                fnamelist,
                spk_id_org,
                spmel_gt,
                rhythm_input,
                content_input,
                pitch_input,
                timbre_input,
                len_crop,
            ) = item
            assert len(fnamelist) == 1
            fname: str = fnamelist[0].split("/")[-1].split(".")[0]
            self.transcriber.transcribe(spmel_gt, f"{fname}")
        return

    def corr_calc(self: Self, gt_list: List[int], wp_list: List[int]):
        x = torch.tensor(gt_list)
        y = torch.tensor(wp_list)
        self.logger.trace(f"x: {x.dtype}")
        self.logger.trace(f"y: {y.dtype}")
        gt_m, wp_m = torch.meshgrid(x, y, indexing="ij")
        m_corr = torch.abs(torch.sub(gt_m, wp_m))
        return -1, m_corr
