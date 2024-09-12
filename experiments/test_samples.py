from typing import Self

import torch

from experiments.experiment import Experiment


class TestSamples(Experiment):
    @torch.no_grad()
    def test(self: Self) -> None:
        self.load_data(singleitem=True)
        self.logger.info("Start eval...")
        self.logfile = open("normlog.csv", "wt")
        print(
            "filename, max, min, mean, median, has_nans",
            file=self.logfile,
        )
        [
            self.test_item(fname=fname[0], item=spmel_gt)
            for fname, _, spmel_gt, _, _, _, _, _ in self.data_loader
        ]

    def test_item(self: Self, fname: str, item: torch.Tensor) -> None:
        print(
            "{}, {}, {}, {}, {}, {}".format(
                fname,
                item.max(),
                item.min(),
                item.mean(),
                item.median(),
                item.isnan().any(),
            ),
            file=self.logfile,
        )
