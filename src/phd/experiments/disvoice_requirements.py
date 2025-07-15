from util.tensor import Tensor
from disvoice.articulation import Articulation as ArticulationRunner
from disvoice.glottal import Glottal as GlottalRunner
from disvoice.phonation import Phonation as PhonationRunner
from disvoice.prosody import Prosody as ProsodyRunner


class DisVoiceItem:
    lbl: str
    feat: Tensor

    def __init__(self, lbl: str, feat: Tensor):
        self.lbl = lbl
        self.feat = feat

    def __len__(self) -> int:
        return 1 if (self.feat.ndimension() == 0) else self.feat.size(dim=-1)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        if self.feat.ndimension() == 0:
            return f"{self.lbl}: {self.feat.item(): .5g}"
        else:
            if self.feat.size(dim=-1) > 40:
                return f"{self.lbl}: {tuple(self.feat.size())}"
            else:
                items = ", ".join([f"{i:.3g}" for i in self.feat.tolist()])
                return f"{self.lbl}:\n\t{items}\n"


class DisVoiceWrapper:
    lbls_static: list[str]
    lbls_dynamic: list[str]
    runner: ArticulationRunner | GlottalRunner | PhonationRunner | ProsodyRunner
    lbls: list[str]
    feats: Tensor

    def __init__(
        self,
        file: str,
        do_static: bool,
    ) -> None:
        if do_static:
            self.lbls = self.lbls_static
        else:
            self.lbls = self.lbls_dynamic
        self.feats = self.runner.extract_features_file(
            file, static=do_static, plots=False, fmt="torch"
        )
        self.feats.squeeze_(0)
        if self.feats.ndimension() == 2 and len(self.lbls) == self.feats.size(dim=-1):
            self.feats = self.feats.mT
        # fmt: off
        assert self.feats.size(dim=0) == len(self.lbls), f"{len(self.lbls)}, {self.feats.size()}"
        # fmt: on

    def __len__(self) -> int:
        return len(self.lbls)

    def __iter__(self):
        for lbl, feat in zip(self.lbls, self.feats):
            yield DisVoiceItem(lbl, feat)

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "\n".join([str(item) for item in self.__iter__()])

    def __getitem__(self, key: int) -> DisVoiceItem:
        return DisVoiceItem(
            lbl=self.lbls[key],
            feat=self.feats[key],
        )


def make_stats(base: list[str]) -> list[str]:
    return [
        pref.format(line)
        for pref in ["Mean: {}", "Std:  {}", "Skew: {}", "Kurt: {}"]
        for line in base
    ]


_articulation_dynamic = [
    *[f"BBE {i:>2}" for i in range(1, 23)],
    *[f"MFCC {i:>2}" for i in range(1, 13)],
    *[f"DMFCC {i:>2}" for i in range(1, 13)],
    *[f"DDMFCC {i:>2}" for i in range(1, 13)],
]
assert len(_articulation_dynamic) == 58, f" got {len(_articulation_dynamic)}"

_articulation_static = [
    # fmt: off
    *[f"Bark band energies in onset transitions {i:>2}" for i in range(1, 23)],
    *[f"Mel frequency cepstral coefficients in onset transitions {i:>2}" for i in range(1, 13)],
    *[f"First derivative of the MFCCs in onset transitions {i:>2}" for i in range(1, 13)],
    *[f"Second derivative of the MFCCs in onset transitions {i:>2}" for i in range(1, 13)],
    *[f"Bark band energies in offset transitions {i:>2}" for i in range(1, 23)],
    *[f"MFCC in offset transitions {i:>2}" for i in range(1, 13)],
    *[f"First derivative of the MFCCs in offset transitions {i:>2}" for i in range(1, 13)],
    *[f"Second derivative of the MFCCs in offset transitions {i:>2}" for i in range(1, 13)],
    # fmt: on
    "First formant Frequency",
    "First Derivative of the first formant frequency",
    "Second Derivative of the first formant frequency",
    "Second formant Frequency",
    "First derivative of the Second formant Frequency",
    "Second derivative of the Second formant Frequency",
]
assert len(_articulation_static) == 122, f" got {len(_articulation_static)}"


_glottal_feats = [
    "Variability of time between consecutive glottal closure instants (GCI)",
    "Average opening quotient (OQ) for consecutive glottal cycles-> rate of opening phase duration / duration of glottal cycle",
    "Variability of opening quotient (OQ) for consecutive glottal cycles-> rate of opening phase duration /duration of glottal cycle",
    "Average normalized amplitude quotient (NAQ) for consecutive glottal cycles-> ratio of the amplitude quotient and the duration of the glottal cycle",
    "Variability of normalized amplitude quotient (NAQ) for consecutive glottal cycles-> ratio of the amplitude quotient and the duration of the glottal cycle",
    "Average H1H2: Difference between the first two harmonics of the glottal flow signal",
    "Variability H1H2: Difference between the first two harmonics of the glottal flow signal",
    "Average of Harmonic richness factor (HRF): ratio of the sum of the harmonics amplitude and the amplitude of the fundamental frequency",
    "Variability of HRF",
]


_phonation_feats = [
    "First derivative of the fundamental Frequency",
    "Second derivative of the fundamental Frequency",
    "Jitter",
    "Shimmer",
    "Amplitude perturbation quotient",
    "Pitch perturbation quotient",
    "Logaritmic Energy",
]


class Articulation(DisVoiceWrapper):
    runner = ArticulationRunner()
    lbls_dynamic = _articulation_dynamic
    lbls_static = make_stats(_articulation_static)


class Glottal(DisVoiceWrapper):
    runner = GlottalRunner()
    lbls_dynamic = _glottal_feats
    lbls_static = make_stats(_glottal_feats)


class Phonation(DisVoiceWrapper):
    runner = PhonationRunner()
    lbls_dynamic = _phonation_feats
    lbls_static = make_stats(_phonation_feats)
