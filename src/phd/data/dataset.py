__all__ = [
    "DatasetParser",
    "DType",
    "Phoneme",
    "Utterance",
]

from enum import Enum, auto

from util import Config
from util.file import (basename, dirname, freadline, freadlist, myglob, path,
                       strip_ext)

from .dataset_detail import (smolspeech_speakers, smolvctk_speakers,
                             timit_speakers, timit_spk_path, uaspeech_speakers,
                             uaspeech_uttrs, vctk_sex, vctk_speakers)


class Phoneme:
    start: int
    end: int
    phon: str

    def __init__(self, start: int, end: int, phon: str):
        self.start = start
        self.end = end
        self.phon = self.__fold_phon(phon)

    def __str__(self) -> str:
        return f"{self.start}-{self.end}: {self.phon}"

    def __repr__(self) -> str:
        return self.__str__()

    def __fold_phon(self, phon: str) -> str:
        match phon:
            case "sil" | "h#":
                return "sil"
            case _:
                return phon


class Utterance:
    word: str
    start: int
    end: int
    phones: list[Phoneme]

    def __init__(self, word: str, phones: list[Phoneme]):
        self.word = word
        self.phones = phones
        self.start = min(phone.start for phone in self.phones)
        self.end = max(phone.end for phone in self.phones)

    def __str__(self) -> str:
        return f"{self.word}: {self.phones}"

    def __repr__(self) -> str:
        return self.__str__()


def parse_phonemes(lines: list[str], setword: str | None = None) -> Utterance:
    word: str = setword or ""
    phones: list[Phoneme] = []
    for line in lines:
        items: list[str] = line.split()
        start: int = int(items[0])
        end: int = int(items[1])
        phon: str = items[2]
        if setword is None and len(items) == 4 and items[2] != "sil":
            word = items[3]
        phones.append(Phoneme(start, end, phon))
    return Utterance(word, phones)


class DType(Enum):
    VCTK = auto()
    UASPEECH = auto()
    TIMIT = auto()

    def __str__(self) -> str:
        match self:
            case self.VCTK:
                return "VCTK"
            case self.UASPEECH:
                return "UASpeech"
            case self.TIMIT:
                return "TIMIT"
            case _:
                raise ValueError


class DatasetParser:
    __raw_timit: str
    __raw_uaspeech: str
    __raw_vctk: str
    __dsettype: DType
    __ua_phone_labels: dict[str, Utterance]

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self.__dsettype = self.__get_type(config.options.dataset_name)
        self.__is_smol = config.options.dataset_name.lower().startswith("smol")
        self.__raw_timit = config.paths.raw_timit
        self.__raw_uaspeech = config.paths.raw_uaspeech
        self.__raw_vctk = config.paths.raw_vctk
        self.__dim_spk_emb = config.model.dim_spk_emb
        if self.is_uaspeech():
            self.__ua_phone_labels = self.__load_uaspeech_phones()

    def __load_uaspeech_phones(self) -> dict[str, Utterance]:
        dataset_base = self.__raw_uaspeech.rpartition("/")[0].rpartition("/")[0]
        filepath = path(dataset_base, "mlf", "M16", "M16_aligned_phones.mlf")
        uttr: str = ""
        lines: list[str] = []
        phones: dict[str, Utterance] = {}
        with open(filepath) as mlf_file:
            for i, line in enumerate(mlf_file):
                try:
                    if line.startswith("."):
                        phones[uttr] = parse_phonemes(lines)
                        uttr = ""
                        lines.clear()
                    elif line.startswith("#"):
                        pass
                    elif line.startswith('"'):
                        uttr = self.utterance(basename(line.strip().strip('"')))
                    else:
                        lines.append(line.strip())
                except Exception as e:
                    raise Exception(f"Failed on line: {i}") from e
        return phones

    def __get_type(self, dname: str) -> DType:
        match dname.lower().removeprefix("smol"):
            case "uaspeech":
                return DType.UASPEECH
            case "vctk":
                return DType.VCTK
            case "timit":
                return DType.TIMIT
            case _:
                raise ValueError

    def dataset_type(self) -> DType:
        return self.__dsettype

    def is_uaspeech(self) -> bool:
        return self.dataset_type() == DType.UASPEECH

    def is_vctk(self) -> bool:
        return self.dataset_type() == DType.VCTK

    def is_timit(self) -> bool:
        return self.dataset_type() == DType.TIMIT

    def get_spkdir(self, spk: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                return spk
            case DType.VCTK:
                return spk
            case DType.TIMIT:
                return path(*timit_spk_path[spk])
            case _:
                raise ValueError

    def get_fullpath(self, spk: str, uttr: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                return path(self.get_spkdir(spk), f"{spk}_{strip_ext(uttr)}_M2")
            case DType.VCTK:
                return path(self.get_spkdir(spk), f"{spk}_{strip_ext(uttr)}")
            case DType.TIMIT:
                return path(self.get_spkdir(spk), strip_ext(uttr))
            case _:
                raise ValueError

    def sex(self, speaker: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                return speaker.removeprefix("C")[0]
            case DType.VCTK:
                return vctk_sex[speaker]
            case DType.TIMIT:
                return speaker[0]
            case _:
                raise ValueError

    def dysarthric(self, speaker: str) -> bool:
        match self.dataset_type():
            case DType.UASPEECH:
                return speaker[0] != "C"
            case DType.VCTK:
                return False
            case DType.TIMIT:
                return False
            case _:
                raise ValueError

    def speakers(self) -> set[str]:
        match self.dataset_type():
            case DType.UASPEECH:
                return smolspeech_speakers if self.__is_smol else uaspeech_speakers
            case DType.VCTK:
                return smolvctk_speakers if self.__is_smol else vctk_speakers
            case DType.TIMIT:
                return timit_speakers
            case _:
                raise ValueError

    def utterance(self, fpath: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                spk, blk, uttr, mic = basename(fpath).split("_")
                return f"{blk}_{uttr}"
            case DType.VCTK:
                spk, uttr = basename(fpath).split("_")
                return uttr
            case DType.TIMIT:
                return basename(fpath)
            case _:
                raise ValueError

    def speaker(self, fpath: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                spk, blk, uttr, mic = basename(fpath).split("_")
                return spk
            case DType.VCTK:
                spk, uttr = basename(fpath).split("_")
                return spk
            case DType.TIMIT:
                return basename(dirname(fpath))
            case _:
                raise ValueError

    def raw_samples(self, spk: str) -> list[str]:
        match self.dataset_type():
            case DType.UASPEECH:
                return myglob(path(self.__raw_uaspeech, self.get_spkdir(spk)), "*.wav")
            case DType.VCTK:
                return myglob(path(self.__raw_vctk, self.get_spkdir(spk)), "*.wav")
            case DType.TIMIT:
                return myglob(path(self.__raw_timit, self.get_spkdir(spk)), "*.WAV")
            case _:
                raise ValueError

    def phonetic_labelled_speakers(self) -> set[str]:
        match self.dataset_type():
            case DType.UASPEECH:
                return {"M16"}
            case DType.VCTK:
                return set()
            case DType.TIMIT:
                return self.speakers()

    def sample_name(self, fpath: str) -> str:
        fname = basename(fpath)
        match self.dataset_type():
            case DType.UASPEECH:
                return f"{self.speaker(fpath)}_{self.utterance(fpath)}"
            case DType.VCTK:
                return f"{self.speaker(fpath)}_{self.utterance(fpath)}"
            case DType.TIMIT:
                return self.utterance(fname)
            case _:
                raise ValueError

    def get_real_text(self, fpath: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                return uaspeech_uttrs[self.utterance(fpath)]
            case DType.VCTK:
                spk = self.speaker(fpath)
                txtfile = f"{self.sample_name(fpath)}.txt"
                return freadline(path(self.__raw_vctk, "txt", spk, txtfile)).strip()
            case DType.TIMIT:
                spk = self.speaker(fpath)
                uttr = self.utterance(fpath)
                txtfile = f"{self.get_fullpath(spk, uttr)}.TXT"
                text = freadline(path(self.__raw_timit, txtfile))
                return text.partition(" ")[-1].partition(" ")[-1].strip()
            case _:
                raise ValueError

    def get_wavfile(self, spk: str, uttr: str) -> str:
        match self.dataset_type():
            case DType.UASPEECH:
                return self.get_fullpath(spk, uttr) + ".wav"
            case DType.VCTK:
                return self.get_fullpath(spk, uttr) + ".wav"
            case DType.TIMIT:
                return strip_ext(self.get_fullpath(spk, uttr)) + ".WAV"
            case _:
                raise ValueError

    def get_utterance(self, fpath: str) -> Utterance:
        match self.dataset_type():
            case DType.UASPEECH:
                return self.__ua_phone_labels[self.utterance(fpath)]
            case DType.VCTK:
                raise RuntimeError("VCTK does not have phoneme labels")
            case DType.TIMIT:
                phonfile = f"{self.get_fullpath(self.speaker(fpath), self.utterance(fpath))}.PHN"
                phones = freadlist(path(self.__raw_timit, phonfile))
                return parse_phonemes(phones, self.get_real_text(fpath))
            case _:
                raise ValueError

    def get_speaker_id(self, speaker: str) -> int:
        match self.dataset_type():
            case DType.UASPEECH:
                assert speaker in uaspeech_speakers
                return list(uaspeech_speakers).index(speaker)
            case DType.VCTK:
                assert speaker in vctk_speakers
                return list(vctk_speakers).index(speaker)
            case DType.TIMIT:
                assert speaker in timit_speakers
                return list(timit_speakers).index(speaker)
            case _:
                raise ValueError

    def get_utterances(self, speaker: str) -> set[str]:
        return set(self.utterance(file) for file in self.raw_samples(speaker))
