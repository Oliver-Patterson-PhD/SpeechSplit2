from typing import Self

from util.file import basename

from .dataset import DatasetParser


class Phoneme:
    start: int
    end: int
    phon: str

    def __init__(self: Self, start: int, end: int, phon: str):
        self.start = start
        self.end = end
        self.phon = phon

    def __str__(self: Self) -> str:
        return f"{self.start}-{self.end}: {self.phon}"

    def __repr__(self: Self) -> str:
        return self.__str__()


class Word:
    word: str
    start: int
    end: int
    phones: list[Phoneme]

    def __init__(self: Self, word: str, phones: list[Phoneme]):
        self.word = word
        self.phones = phones
        self.start = min(phone.start for phone in self.phones)
        self.end = max(phone.end for phone in self.phones)

    def __str__(self: Self) -> str:
        return f"{self.word}: {self.phones}"

    def __repr__(self: Self) -> str:
        return self.__str__()


def parse_phonemes(lines: list[str]) -> Word:
    word: str = ""
    phones: list[Phoneme] = []
    for line in lines:
        items: list[str] = line.split()
        start: int = int(items[0])
        end: int = int(items[1])
        phon: str = items[2]
        if len(items) == 4 and items[2] != "sil":
            word = items[3]
        phones.append(Phoneme(start, end, phon))
    return Word(word, phones)


def load_mlf(filepath) -> dict[str, Word]:
    filename: str = ""
    lines: list[str] = []
    phones: dict[str, Word] = {}
    parser = DatasetParser()
    with open(filepath) as mlf_file:
        for line in mlf_file:
            if line.startswith("."):
                phones[filename] = parse_phonemes(lines)
                filename = ""
                lines.clear()
            elif line.startswith("#"):
                pass
            elif line.startswith('"'):
                filename = parser.utterance(basename(line.strip().strip('"')))
            else:
                lines.append(line.strip())
    return phones


def load_words(filepath: str) -> dict[str, Word]:
    ftype = filepath.rpartition(".")[-1]
    match ftype:
        case "mlf":
            return load_mlf(filepath)
        case _:
            raise RuntimeError(f"Invalid filetype: {ftype}")
