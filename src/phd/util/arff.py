from __future__ import annotations

import csv
import ctypes
import re
from collections.abc import Generator
from io import TextIOWrapper

import numpy as np
from scipy.io.arff import MetaData, ParseArffError

ArffItemType = float | str
ArffRowType = dict[str, ArffItemType]


class ArffData:
    meta: MetaData
    data: list[ArffRowType]

    def __init__(self, data: list[ArffRowType], meta: MetaData) -> None:
        self.data = data
        self.meta = meta

    def __iter__(self) -> Generator[ArffRowType]:
        for item in self.data:
            yield item

    def __len__(self) -> int:
        return len(self.data)


r_meta = re.compile(r"^\s*@")
r_comment = re.compile(r"^%")
r_empty = re.compile(r"^\s+$")
r_headerline = re.compile(r"^\s*@\S*")
r_datameta = re.compile(r"^@[Dd][Aa][Tt][Aa]")
r_relation = re.compile(r"^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)")
r_attribute = re.compile(r"^\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)")
r_nominal = re.compile(r"{(.+)}")
r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")
r_comattrval = re.compile(r"'(..+)'\s+(..+$)")
r_wcomattrval = re.compile(r"(\S+)\s+(..+$)")


def split_data_line(line: str, dialect=None):
    delimiters = ",\t"
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
    if line[-1] == "\n":
        line = line[:-1]
    line = line.strip()
    sniff_line = line
    if not any(d in line for d in delimiters):
        sniff_line += ","
    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
    row = next(csv.reader([line], dialect))
    return row, dialect


class Attribute:
    name: str
    type_name: str | None = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.dtype = np.object_
        self.range = None

    @classmethod
    def parse_attribute(cls, name: str, attr_string: str) -> Attribute | None:
        return None

    def parse_data(self, data_str: str) -> ArffItemType:
        raise NotImplementedError(
            f"Error, no attribute implemented for {self.__class__}"
        )

    def __str__(self) -> str:
        assert self.type_name is not None
        return self.name + "," + self.type_name


class NumericAttribute(Attribute):
    def __init__(self, name):
        super().__init__(name)
        self.type_name = "numeric"
        self.dtype = np.float64

    @classmethod
    def parse_attribute(cls, name, attr_string):
        attr_string = attr_string.lower().strip()
        if (
            attr_string[: len("numeric")] == "numeric"
            or attr_string[: len("int")] == "int"
            or attr_string[: len("real")] == "real"
        ):
            return cls(name)
        else:
            return None

    def parse_data(self, data_str: str) -> float:
        if "?" in data_str:
            return np.nan
        else:
            return float(data_str)

    def _basic_stats(self, data):
        nbfac = data.size * 1.0 / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac)


class StringAttribute(Attribute):
    def __init__(self, name):
        super().__init__(name)
        self.type_name = "string"

    @classmethod
    def parse_attribute(cls, name, attr_string):
        attr_string = attr_string.lower().strip()

        if attr_string[: len("string")] == "string":
            return cls(name)
        else:
            return None

    def parse_data(self, data_str: str) -> str:
        return data_str


def tokenize_single_comma(val: str) -> tuple[str, str]:
    m = r_comattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ParseArffError("Error while tokenizing attribute") from e
    else:
        raise ParseArffError(f"Error while tokenizing single {val}")
    return name, type


def tokenize_single_wcomma(val: str) -> tuple[str, str]:
    m = r_wcomattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ParseArffError("Error while tokenizing attribute") from e
    else:
        raise ParseArffError(f"Error while tokenizing single {val}")
    return name, type


def to_attribute(name: str, attr_string: str) -> Attribute:
    attr_classes = (NumericAttribute, StringAttribute)
    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)
        if attr is not None:
            return attr
    raise ParseArffError(f"unknown attribute {attr_string}")


def read_relational_attribute(
    ofile: TextIOWrapper, relational_attribute, i: str
) -> str:
    r_end_relational = re.compile(
        r"^@[Ee][Nn][Dd]\s*" + relational_attribute.name + r"\s*$"
    )
    while not r_end_relational.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                relational_attribute.attributes.append(attr)
            else:
                raise ParseArffError(f"Error parsing line {i}")
        else:
            i = next(ofile)
    i = next(ofile)
    return i


def tokenize_attribute(
    iterable: TextIOWrapper, attribute: str
) -> tuple[Attribute, str]:
    sattr = attribute.strip()
    mattr = r_attribute.match(sattr)
    if mattr:
        # atrv is everything after @attribute
        atrv = mattr.group(1)
        if r_comattrval.match(atrv):
            name, type = tokenize_single_comma(atrv)
            next_item = next(iterable)
        elif r_wcomattrval.match(atrv):
            name, type = tokenize_single_wcomma(atrv)
            next_item = next(iterable)
        else:
            # Not sure we should support this, as it does not seem supported by weka.
            raise ParseArffError("multi line not supported yet")
    else:
        raise ParseArffError(f"First line unparsable: {sattr}")
    attr = to_attribute(name, type)
    if type.lower() == "relational":
        next_item = read_relational_attribute(iterable, attr, next_item)
    return attr, next_item


def read_header(ofile: TextIOWrapper) -> tuple[str, list[Attribute]]:
    i = next(ofile)
    while r_comment.match(i):
        i = next(ofile)
    relation = None
    attributes = []
    while not r_datameta.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                attributes.append(attr)
            else:
                isrel = r_relation.match(i)
                if isrel:
                    relation = isrel.group(1)
                else:
                    raise ParseArffError(f"Error while parsing header, line {i}")
                i = next(ofile)
        else:
            i = next(ofile)
    assert isinstance(relation, str)
    return relation, attributes


def generator(
    row_iter: TextIOWrapper, attrs: list[Attribute], delim: str = ","
) -> Generator[ArffRowType]:
    elems = list(range(len(attrs)))
    dialect = None
    for raw in row_iter:
        if raw.startswith("%") or len(raw.strip()) == 0:
            continue
        row, dialect = split_data_line(raw, dialect)
        data: dict[str, ArffItemType] = {
            key: val
            for key, val in {
                attrs[i].name: attrs[i].parse_data(row[i]) for i in elems
            }.items()
            if val is not None
        }
        yield data


def loadarff(ofile: TextIOWrapper) -> tuple[list[ArffRowType], MetaData]:
    rel, attr = read_header(ofile)
    meta = MetaData(rel, attr)
    data = list(generator(ofile, attr))
    return data, meta


def load(fname: str) -> ArffData | None:
    with open(fname) as ofile:
        data, meta = loadarff(ofile)
    if len(data) == 0:
        return None
    return ArffData(data, meta)
