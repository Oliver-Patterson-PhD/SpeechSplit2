from numpy import recarray
from scipy.io.arff import MetaData, loadarff

ArffItemType = float | str
ArffRowType = dict[str, float | str]


class ArffData:
    meta: MetaData
    data: recarray

    def __init__(self, data: recarray, meta: MetaData) -> None:
        self.data = data
        self.meta = meta

    def get(self, name: str) -> ArffItemType:
        assert name in self.meta.names()
        return self.data[name]

    def __iter__(self) -> ArffRowType:
        return


def load(fname: str) -> ArffData:
    data, meta = loadarff(fname)
    return ArffData(data, meta)
