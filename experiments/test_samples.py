import time
from typing import List, Self

import torch

from experiments.experiment import Experiment
from util.config import Config


class TestSamples(Experiment):
    def __init__(self: Self, config: Config, currtime: int = int(time.time())) -> None:
        self.config = config

    @torch.no_grad()
    def test(self: Self) -> None:
        self.load_data(singleitem=True)
        self.logger.info("Start eval...")
        self.logfile = open("normlog.csv", "wt")
        items: List[str] = [
            "spmel_gt",
            "rhythm_input",
            "content_input",
            "pitch_input",
            "timbre_input",
        ]
        itemlist = [
            "fname"
            + "".join(
                [
                    ", {}, {}, {}, {}".format(
                        f"{item} max",
                        f"{item} min",
                        f"{item} mean",
                        f"{item} median",
                    )
                    for item in items
                ]
            )
        ]
        itemlist.extend(
            [
                fname
                + "".join(
                    [
                        ", {}, {}, {}, {}".format(
                            eval(f"{item}.max()"),
                            eval(f"{item}.min()"),
                            eval(f"{item}.mean()"),
                            eval(f"{item}.median()"),
                        )
                        for item in items
                    ]
                )
                for (
                    l_fname,
                    l_spk_id_org,
                    l_spmel_gt,
                    l_rhythm_input,
                    l_content_input,
                    l_pitch_input,
                    l_timbre_input,
                    l_len_crop,
                ) in self.data_loader
                for (
                    fname,
                    spk_id_org,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop,
                ) in zip(
                    l_fname,
                    l_spk_id_org,
                    l_spmel_gt,
                    l_rhythm_input,
                    l_content_input,
                    l_pitch_input,
                    l_timbre_input,
                    l_len_crop,
                )
            ]
        )
        [print(line, file=self.logfile) for line in itemlist]
